"""
Module Licence and docstring

Algorithm logic does not live here.
This file defines the data architecture of IBEIS
This file should only define:
    iders
    setters
    getters
    deleter
"""
# TODO: rename annotation annotations
# TODO: make all names consistent
from __future__ import absolute_import, division, print_function
# Python
import atexit
import requests
import uuid
from itertools import izip, imap
from functools import partial
from os.path import join, split
# VTool
from vtool import image as gtool
from vtool import geometry
# UTool
import utool
# IBEIS EXPORT
import ibeis.export.export_wb as wb
# IBEIS DEV
from ibeis import constants
from ibeis.dev import ibsfuncs
# IBEIS MODEL
from ibeis.model import Config
from ibeis.model.preproc import preproc_chip
from ibeis.model.preproc import preproc_image
from ibeis.model.preproc import preproc_feat
from ibeis.model.preproc import preproc_detectimg
from ibeis.model.preproc import preproc_encounter
from ibeis.model.detect import randomforest
from ibeis.model.hots import match_chips3 as mc3
from ibeis.model.hots import QueryRequest
# IBEIS
from ibeis.control import DB_SCHEMA
from ibeis.control import SQLDatabaseControl as sqldbc
from ibeis.control.accessor_decors import (adder, setter, getter_1toM,
                                           getter_1to1, ider, deleter,
                                           default_decorator)
# CONSTANTS
from ibeis.constants import (IMAGE_TABLE,
                             ANNOTATION_TABLE,
                             LABEL_TABLE,
                             ENCOUNTER_TABLE,
                             EG_RELATION_TABLE,
                             AL_RELATION_TABLE,
                             CHIP_TABLE,
                             FEATURE_TABLE,
                             CONFIG_TABLE,
                             KEY_TABLE,)

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibs]', DEBUG=False)


__STR__ = str  # change to unicode if needed
__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers


@atexit.register
def __cleanup():
    """ prevents flann errors (not for cleaning up individual objects) """
    global __ALL_CONTROLLERS__
    del __ALL_CONTROLLERS__


#
#
#-----------------
# IBEIS CONTROLLER
#-----------------

class IBEISController(object):
    """
    IBEISController docstring
        chip  - cropped region of interest in an image, maps to one animal
        cid   - chip unique id
        gid   - image unique id (could just be the relative file path)
        name  - name unique id
        eid   - encounter unique id
        aid   - region of interest unique id
        annotation   - region of interest for a chip
        theta - angle of rotation for a chip
    """

    #
    #
    #-------------------------------
    # --- CONSTRUCTOR / PRIVATES ---
    #-------------------------------

    def __init__(ibs, dbdir=None, ensure=True, wbaddr=None):
        """ Creates a new IBEIS Controller associated with one database """
        global __ALL_CONTROLLERS__
        if utool.VERBOSE:
            print('[ibs.__init__] new IBEISController')
        ibs.qreq = None  # query requestor object
        ibs.ibschanged_callback = None
        ibs._init_dirs(dbdir=dbdir, ensure=ensure)
        ibs._init_wb(wbaddr)  # this will do nothing if no wildbook address is specified
        ibs._init_sql()
        ibs._init_config()
        ibsfuncs.inject_ibeis(ibs)
        __ALL_CONTROLLERS__.append(ibs)

    @default_decorator
    def _init_wb(ibs, wbaddr):
        if wbaddr is None:
            return
        #TODO: Clean this up to use like utool and such
        try:
            requests.get(wbaddr)
        except requests.MissingSchema as msa:
            print('[ibs._init_wb] Invalid URL: %r' % wbaddr)
            raise msa
        except requests.ConnectionError as coe:
            print('[ibs._init_wb] Could not connect to Wildbook server at %r' % wbaddr)
            raise coe
        ibs.wbaddr = wbaddr

    @default_decorator
    def _init_dirs(ibs, dbdir=None, dbname='testdb_1', workdir='~/ibeis_workdir', ensure=True):
        """ Define ibs directories """
        if ensure:
            print('[ibs._init_dirs] ibs.dbdir = %r' % dbdir)
        if dbdir is not None:
            workdir, dbname = split(dbdir)
        ibs.workdir  = utool.truepath(workdir)
        ibs.dbname = dbname
        PATH_NAMES = constants.PATH_NAMES
        ibs.sqldb_fname = PATH_NAMES.sqldb

        # Make sure you are not nesting databases
        assert PATH_NAMES._ibsdb != utool.dirsplit(ibs.workdir), \
            'cannot work in _ibsdb internals'
        assert PATH_NAMES._ibsdb != dbname,\
            'cannot create db in _ibsdb internals'
        ibs.dbdir    = join(ibs.workdir, ibs.dbname)
        # All internal paths live in <dbdir>/_ibsdb
        ibs._ibsdb      = join(ibs.dbdir, PATH_NAMES._ibsdb)
        ibs.cachedir    = join(ibs._ibsdb, PATH_NAMES.cache)
        ibs.chipdir     = join(ibs._ibsdb, PATH_NAMES.chips)
        ibs.imgdir      = join(ibs._ibsdb, PATH_NAMES.images)
        # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
        ibs.thumb_dpath = join(ibs.cachedir, PATH_NAMES.thumbs)
        ibs.flanndir    = join(ibs.cachedir, PATH_NAMES.flann)
        ibs.qresdir     = join(ibs.cachedir, PATH_NAMES.qres)
        ibs.bigcachedir = join(ibs.cachedir, PATH_NAMES.bigcache)
        if ensure:
            _verbose = utool.VERBOSE
            utool.ensuredir(ibs._ibsdb)
            utool.ensuredir(ibs.cachedir,    verbose=_verbose)
            utool.ensuredir(ibs.workdir,     verbose=_verbose)
            utool.ensuredir(ibs.imgdir,      verbose=_verbose)
            utool.ensuredir(ibs.chipdir,     verbose=_verbose)
            utool.ensuredir(ibs.flanndir,    verbose=_verbose)
            utool.ensuredir(ibs.qresdir,     verbose=_verbose)
            utool.ensuredir(ibs.bigcachedir, verbose=_verbose)
            utool.ensuredir(ibs.thumb_dpath, verbose=_verbose)
        assert dbdir is not None, 'must specify database directory'

    @default_decorator
    def _init_sql(ibs):
        """ Load or create sql database """
        ibs.db = sqldbc.SQLDatabaseController(ibs.get_ibsdir(), ibs.sqldb_fname, text_factory=__STR__)
        DB_SCHEMA.define_IBEIS_schema(ibs)
        ibs.MANUAL_CONFIG_SUFFIX = '_MANUAL_' + utool.get_computer_name()
        ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)

        ibs.key_defaults = dict(constants.KEY_DEFAULTS)
        ibs.key_names = sorted(ibs.key_defaults.keys())
        ibs.key_ids = {}
        for key_name in ibs.key_names:
            ibs.key_ids[key_name] = ibs.add_key([key_name], [ibs.key_defaults[key_name]])[0]
            ibs.key_defaults[key_name] = ibs.get_key_default(ibs.key_ids[key_name])

    @default_decorator
    def clone_handle(ibs, **kwargs):
        ibs2 = IBEISController(dbdir=ibs.get_dbdir(), ensure=False)
        if len(kwargs) > 0:
            ibs2.update_query_cfg(**kwargs)
        if ibs.qreq is not None:
            ibs2._prep_qreq(ibs.qreq.qaids, ibs.qreq.daids)
        return ibs2

    #
    #
    #--------------
    # --- DIRS ----
    #--------------

    def get_dbname(ibs):
        """ Returns database name """
        return ibs.dbname

    def get_dbdir(ibs):
        """ Returns database dir with ibs internal directory """
        #return join(ibs.workdir, ibs.dbname)
        return ibs.dbdir

    def get_ibsdir(ibs):
        """ Returns ibs internal directory """
        return ibs._ibsdb

    def get_thumbdir(ibs):
        return ibs.thumb_dpath

    def get_workdir(ibs):
        return ibs.workdir

    def get_cachedir(ibs):
        return ibs.cachedir

    def get_detectimg_cachedir(ibs):
        return join(ibs.cachedir, constants.PATH_NAMES.detectimg)

    def get_flann_cachedir(ibs):
        return ibs.flanndir

    #
    #
    #----------------
    # --- Configs ---
    #----------------

    def _init_config(ibs):
        """ Loads the database's algorithm configuration """
        ibs.cfg = Config.ConfigBase('cfg', fpath=join(ibs.dbdir, 'cfg'))
        try:
            if utool.get_flag(('--noprefload', '--noprefload')):
                raise Exception('')
            ibs.cfg.load()
            print('[ibs] successfully loaded config')
        except Exception:
            ibs._default_config()

    def _default_config(ibs):
        """ Resets the databases's algorithm configuration """
        print('[ibs] building default config')
        query_cfg    = Config.default_query_cfg()
        ibs.set_query_cfg(query_cfg)
        ibs.cfg.enc_cfg     = Config.EncounterConfig()
        ibs.cfg.preproc_cfg = Config.PreprocConfig()
        ibs.cfg.detect_cfg  = Config.DetectionConfig()
        ibs.cfg.other_cfg   = Config.OtherConfig()

    @default_decorator
    def set_query_cfg(ibs, query_cfg):
        if ibs.qreq is not None:
            ibs.qreq.set_cfg(query_cfg)
        ibs.cfg.query_cfg = query_cfg
        ibs.cfg.feat_cfg  = query_cfg._feat_cfg
        ibs.cfg.chip_cfg  = query_cfg._feat_cfg._chip_cfg

    @default_decorator
    def update_query_cfg(ibs, **kwargs):
        """ Updates query config only. Configs needs a restructure very badly """
        ibs.cfg.query_cfg.update_query_cfg(**kwargs)

    @default_decorator
    def get_chip_config_rowid(ibs):
        # FIXME: Configs are still handled poorly
        chip_cfg_suffix = ibs.cfg.chip_cfg.get_cfgstr()
        chip_cfg_rowid = ibs.add_config(chip_cfg_suffix)
        return chip_cfg_rowid

    @default_decorator
    def get_feat_config_rowid(ibs):
        # FIXME: Configs are still handled poorly
        feat_cfg_suffix = ibs.cfg.feat_cfg.get_cfgstr()
        feat_cfg_rowid = ibs.add_config(feat_cfg_suffix)
        return feat_cfg_rowid

    @default_decorator
    def get_query_config_rowid(ibs):
        # FIXME: Configs are still handled poorly
        query_cfg_suffix = ibs.cfg.query_cfg.get_cfgstr()
        query_cfg_rowid = ibs.add_config(query_cfg_suffix)
        return query_cfg_rowid

    @default_decorator
    def get_qreq_rowid(ibs):
        # FIXME: Configs are still handled poorly
        assert ibs.qres is not None
        qreq_rowid = ibs.qreq.get_cfgstr()
        return qreq_rowid

    #
    #
    #---------------
    # --- IDERS ---
    #---------------

    def get_num_images(ibs, **kwargs):
        gid_list = ibs.get_valid_gids(**kwargs)
        return len(gid_list)

    def get_num_annotations(ibs, **kwargs):
        aid_list = ibs.get_valid_aids(**kwargs)
        return len(aid_list)

    def get_num_names(ibs, **kwargs):
        nid_list = ibs.get_valid_nids(**kwargs)
        return len(nid_list)

    @ider
    def _get_all_gids(ibs):
        """ returns all unfiltered gids (image rowids) """
        all_gids = ibs.db.get_all_rowids(IMAGE_TABLE)
        return all_gids

    @ider
    def _get_all_aids(ibs):
        """ returns all unfiltered aids (annotation rowids) """
        all_aids = ibs.db.get_all_rowids(ANNOTATION_TABLE)
        return all_aids

    @ider
    def _get_all_eids(ibs):
        """ returns all unfiltered eids (encounter rowids) """
        all_eids = ibs.db.get_all_rowids(ENCOUNTER_TABLE)
        return all_eids

    @ider
    def _get_all_cids(ibs):
        """ Returns unfiltered cids (computed chip rowids) for every
        configuration (YOU PROBABLY SHOULD NOT USE THIS) """
        all_cids = ibs.db.get_all_rowids(CHIP_TABLE)
        return all_cids

    @ider
    def _get_all_fids(ibs):
        """ Returns unfiltered fids (computed feature rowids) for every
        configuration (YOU PROBABLY SHOULD NOT USE THIS)"""
        all_fids = ibs.db.get_all_rowids(FEATURE_TABLE)
        return all_fids

    @ider
    def get_valid_gids(ibs, eid=None, require_unixtime=False):
        if eid is None:
            gid_list = ibs._get_all_gids()
        else:
            gid_list = ibs.get_encounter_gids(eid)
        if require_unixtime:
            # Remove images without timestamps
            unixtime_list = ibs.get_image_unixtime(gid_list)
            isvalid_list = [unixtime != -1 for unixtime in unixtime_list]
            gid_list = utool.filter_items(gid_list, isvalid_list)
        return sorted(gid_list)

    @ider
    def _get_all_known_nids(ibs, _key='INDIVIDUAL_KEY'):
        """ Returns all nids of known animals
            (does not include unknown names) """
        all_known_nids = ibs.db.get_all_rowids_where(LABEL_TABLE, 'key_rowid=?', (ibs.key_ids[_key],))
        return all_known_nids

    @ider
    def get_valid_nids(ibs, eid=None):
        """ Returns all valid names with at least one animal
            (does not include unknown names) """
        if eid is None:
            _nid_list = ibs._get_all_known_nids()
        else:
            _nid_list = ibs.get_encounter_nids(eid)
        nRois_list = ibs.get_name_num_annotations(_nid_list)
        nid_list = [nid for nid, nRois in izip(_nid_list, nRois_list)
                    if nRois > 0]
        return sorted(nid_list)

    @ider
    def get_invalid_nids(ibs):
        """ Returns all names without any animals (does not include unknown names) """
        _nid_list = ibs._get_all_known_nids()
        nRois_list = ibs.get_name_num_annotations(_nid_list)
        nid_list = [nid for nid, nRois in izip(_nid_list, nRois_list)
                    if nRois <= 0]
        return sorted(nid_list)

    @ider
    def get_valid_eids(ibs, min_num_gids=0):
        """ returns list of all encounter ids """
        eid_list = ibs._get_all_eids()
        if min_num_gids > 0:
            num_gids_list = ibs.get_encounter_num_gids(eid_list)
            flag_list = [num_gids >= min_num_gids for num_gids in num_gids_list]
            eid_list  = utool.filter_items(eid_list, flag_list)
        return sorted(eid_list)

    @ider
    def get_valid_aids(ibs, eid=None, is_exemplar=False):
        """ returns a list of valid ANNOTATION unique ids """
        if eid is None:
            aid_list = ibs._get_all_aids()
        else:
            enctext = ibs.get_encounter_enctext(eid)
            if enctext == constants.EXEMPLAR_ENCTEXT:
                is_exemplar = True
            aid_list = ibs.get_encounter_aids(eid)
        if is_exemplar:
            flag_list = ibs.get_annotation_exemplar_flag(aid_list)
            aid_list = utool.filter_items(aid_list, flag_list)
        return sorted(aid_list)

    @ider
    def get_valid_cids(ibs):
        """ Valid chip rowids of the current configuration """
        # FIXME: configids need reworking
        chip_config_rowid = ibs.get_chip_config_rowid()
        cid_list = ibs.db.get_all_rowids_where(FEATURE_TABLE, 'config_rowid=?', (chip_config_rowid,))
        return cid_list

    @ider
    def get_valid_fids(ibs):
        """ Valid feature rowids of the current configuration """
        # FIXME: configids need reworking
        feat_config_rowid = ibs.get_feat_config_rowid()
        fid_list = ibs.db.get_all_rowids_where(FEATURE_TABLE, 'config_rowid=?', (feat_config_rowid,))
        return fid_list

    #
    #
    #---------------
    # --- ADDERS ---
    #---------------

    @adder
    def add_key(ibs, text_list, default_list):
        params_iter = izip(text_list, default_list)
        key = ibs.db.add_cleanly(KEY_TABLE, ('key_text', 'key_default'), params_iter, ibs.get_key_rowid_from_text)
        return key

    @adder
    def add_config(ibs, cfgsuffix_list):
        """ Adds an algorithm / actor configuration as a string """
        # FIXME: Configs are still handled poorly
        params_iter = ((suffix,) for suffix in cfgsuffix_list)
        get_rowid_from_uuid = partial(ibs.get_config_rowid_from_suffix, ensure=False)
        configid_list = ibs.db.add_cleanly(CONFIG_TABLE, ('config_suffix',), params_iter, get_rowid_from_uuid)
        return configid_list

    @adder
    def add_images(ibs, gpath_list):
        """
        Adds a list of image paths to the database.  Returns gids

        TEST CODE:
            from ibeis.dev.all_imports import *
            gpath_list = grabdata.get_test_gpaths(ndata=7) + ['doesnotexist.jpg']

        Initially we set the image_uri to exactely the given gpath.
        Later we change the uri, but keeping it the same here lets
        us process images asychronously."""
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))
        #print('[ibs] gpath_list = %r' % (gpath_list,))
        # Processing an image might fail, yeilding a None instead of a tup
        gpath_list = ibsfuncs.assert_and_fix_gpath_slashes(gpath_list)
        # Create param_iter
        params_list  = list(preproc_image.add_images_params_gen(gpath_list))
        # Error reporting
        print('\n'.join(
            [' ! Failed reading gpath=%r' % (gpath,) for (gpath, params)
             in izip(gpath_list, params_list) if not params]))
        # Add any unadded images
        colnames = ('image_uuid', 'image_uri', 'image_original_name',
                    'image_ext', 'image_width', 'image_height',
                    'image_time_posix', 'image_gps_lat',
                    'image_gps_lon', 'image_note',)
        # <DEBUG>
        if utool.VERBOSE:
            uuid_list = [None if params is None else params[0] for params in params_list]
            gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
            valid_gids = ibs.get_valid_gids()
            valid_uuids = ibs.get_image_uuids(valid_gids)
            print('[preadd] uuid / gid_ = ' + utool.indentjoin(zip(uuid_list, gid_list_)))
            print('[preadd] valid uuid / gid = ' + utool.indentjoin(zip(valid_uuids, valid_gids)))
        # </DEBUG>
        # Execute SQL Add
        gid_list = ibs.db.add_cleanly(IMAGE_TABLE, colnames, params_list, ibs.get_image_gids_from_uuid)

        if utool.VERBOSE:
            uuid_list = [None if params is None else params[0] for params in params_list]
            gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
            valid_gids = ibs.get_valid_gids()
            valid_uuids = ibs.get_image_uuids(valid_gids)
            print('[postadd] uuid / gid_ = ' + utool.indentjoin(zip(uuid_list, gid_list_)))
            print('[postadd] uuid / gid = ' + utool.indentjoin(zip(uuid_list, gid_list)))
            print('[postadd] valid uuid / gid = ' + utool.indentjoin(zip(valid_uuids, valid_gids)))
        return gid_list

    @adder
    def add_annotations(ibs, gid_list, bbox_list=None, theta_list=None, nid_list=None,
                      name_list=None, detect_confidence_list=None, notes_list=None,
                      vert_list=None):
        """ Adds oriented ANNOTATION bounding boxes to images """
        if utool.VERBOSE:
            print('[ibs] adding annotations')
        # Prepare the SQL input
        assert name_list is None or nid_list is None,\
            'cannot specify both names and nids'
        assert (bbox_list is     None and vert_list is not None) or \
               (bbox_list is not None and vert_list is     None) ,\
            'must specify exactly one of bbox_list or vert_list'
        if theta_list is None:
            theta_list = [0.0 for _ in xrange(len(gid_list))]
        if name_list is not None:
            nid_list = ibs.add_names(name_list)
        if detect_confidence_list is None:
            detect_confidence_list = [0.0 for _ in xrange(len(gid_list))]
        if notes_list is None:
            notes_list = ['' for _ in xrange(len(gid_list))]
        if vert_list is None:
            vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
        elif bbox_list is None:
            bbox_list = geometry.bboxes_from_vert_list(vert_list)

        # Build ~~deterministic?~~ random and unique ANNOTATION ids
        image_uuid_list = ibs.get_image_uuids(gid_list)
        annotation_uuid_list = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list,
                                                          theta_list, deterministic=False)
        nVert_list = [len(verts) for verts in vert_list]
        vertstr_list = [__STR__(verts) for verts in vert_list]
        xtl_list, ytl_list, width_list, height_list = list(izip(*bbox_list))
        assert len(nVert_list) == len(vertstr_list)
        # Define arguments to insert
        colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                    'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                    'annot_verts', 'annot_detect_confidence',
                    'annot_note',)

        params_iter = list(izip(annotation_uuid_list, gid_list, xtl_list, ytl_list,
                                width_list, height_list, theta_list, nVert_list,
                                vertstr_list, detect_confidence_list,
                                notes_list))
        #utool.embed()

        # Execute add ANNOTATIONs SQL
        get_rowid_from_uuid = ibs.get_annotation_aids_from_uuid
        aid_list = ibs.db.add_cleanly(ANNOTATION_TABLE, colnames, params_iter, get_rowid_from_uuid)

        # Also need to populate annotation_label_relationship table
        if nid_list is not None:  
            alrid_list = ibs.add_annotation_relationship(aid_list, nid_list)
            del alrid_list
        #print('alrid_list = %r' % (alrid_list,))

        # Invalidate image thumbnails
        ibs.delete_image_thumbtups(gid_list)
        return aid_list

    @adder
    def add_annotation_relationship(ibs, aid_list, labelid_list, configid_list=None,
                                  alr_confidence_list=None):
        if configid_list is None:
            configid_list = [ibs.MANUAL_CONFIGID] * len(aid_list)
        if alr_confidence_list is None:
            alr_confidence_list = [0.0] * len(aid_list)
        colnames = ('annot_rowid', 'label_rowid', 'config_rowid', 'alr_confidence')
        params_iter = list(izip(aid_list, labelid_list, configid_list,
                                alr_confidence_list))
        alrid_list = ibs.db.add_cleanly(AL_RELATION_TABLE, colnames, params_iter,
                                        ibs.get_alr_rowid_from_valtup, unique_paramx=range(0, 3))
        return alrid_list

    @adder
    def add_annotation_names(ibs, aid_list, name_list=None, nid_list=None):
        """ Sets names/nids of a list of annotations.
        Convenience function for set_annotation_nids"""
        assert name_list is None or nid_list is None, (
            'can only specify one type of name values (nid or name) not both')
        if nid_list is None:
            assert name_list is not None
            # Convert names into nids
            nid_list = ibs.add_names(name_list)
        ibs.add_annotation_relationship(aid_list, nid_list)

    @adder
    def add_image_relationship(ibs, gid_list, eid_list):
        colnames = ('image_rowid', 'encounter_rowid')
        params_iter = list(izip(gid_list, eid_list))
        egrid_list = ibs.db.add_cleanly(EG_RELATION_TABLE, colnames, params_iter,
                                        ibs.get_egr_rowid_from_valtup, unique_paramx=range(0, 2))
        return egrid_list

    @adder
    def add_chips(ibs, aid_list):
        """ Adds chip data to the ANNOTATION. (does not create ANNOTATIONs. first use add_annotations
        and then pass them here to ensure chips are computed)
        return cid_list
        """
        # Ensure must be false, otherwise an infinite loop occurs
        cid_list = ibs.get_annotation_cids(aid_list, ensure=False)
        dirty_aids = utool.get_dirty_items(aid_list, cid_list)
        if len(dirty_aids) > 0:
            print('[ibs] adding chips')
            try:
                # FIXME: Cant be lazy until chip config / delete issue is fixed
                preproc_chip.compute_and_write_chips(ibs, aid_list)
                #preproc_chip.compute_and_write_chips_lazy(ibs, aid_list)
                params_iter = preproc_chip.add_chips_params_gen(ibs, dirty_aids)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.add_chips]')
                print('[!ibs.add_chips] ' + utool.list_dbgstr('aid_list'))
                raise
            colnames = ('annot_rowid', 'chip_uri', 'chip_width', 'chip_height',
                        'config_rowid',)
            get_rowid_from_uuid = partial(ibs.get_annotation_cids, ensure=False)
            cid_list = ibs.db.add_cleanly(CHIP_TABLE, colnames, params_iter, get_rowid_from_uuid)

        return cid_list

    @adder
    def add_feats(ibs, cid_list, force=False):
        """ Computes the features for every chip without them """
        fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        dirty_cids = utool.get_dirty_items(cid_list, fid_list)
        if len(dirty_cids) > 0:
            if utool.VERBOSE:
                print('[ibs] adding %d / %d features' % (len(dirty_cids), len(cid_list)))
            params_iter = preproc_feat.add_feat_params_gen(ibs, dirty_cids)
            colnames = ('chip_rowid', 'feature_num_feats', 'feature_keypoints',
                        'feature_sifts', 'config_rowid',)
            get_rowid_from_uuid = partial(ibs.get_chip_fids, ensure=False)
            fid_list = ibs.db.add_cleanly(FEATURE_TABLE, colnames, params_iter, get_rowid_from_uuid)

        return fid_list

    @adder
    def add_names(ibs, name_list):
        """ Adds a list of names. Returns their nids """
        # nid_list_ = [namenid_dict[name] for name in name_list_]
        # ibsfuncs.assert_valid_names(name_list)
        notes_list = [''] * len(name_list)
        # All names are individuals and so may safely receive the INDIVIDUAL_KEY label
        key_rowid_list = [ibs.key_ids['INDIVIDUAL_KEY']] * len(name_list)
        nid_list = ibs.add_labels(key_rowid_list, name_list, notes_list)
        return nid_list

    @adder
    def add_species(ibs, species_list):
        """ Adds a list of species. Returns their nids """
        # nid_list_ = [namenid_dict[name] for name in species_list_]
        # ibsfuncs.assert_valid_names(species_list)
        notes_list = [''] * len(species_list)
        # All names are individuals and so may safely receive the INDIVIDUAL_KEY label
        key_rowid_list = [ibs.key_ids['SPECIES_KEY']] * len(species_list)
        nid_list = ibs.add_labels(key_rowid_list, species_list, notes_list)
        return nid_list

    @adder
    def add_labels(ibs, key_list, value_list, note_list):
        """ Adds new labels and creates a new uuid for them if it doesn't
        already exist """
        # Get random uuids
        label_uuid_list = [uuid.uuid4() for _ in xrange(len(value_list))]
        colnames = ['label_uuid', 'key_rowid', 'label_value', 'label_note']
        params_iter = list(izip(label_uuid_list, key_list, value_list, note_list))
        labelid_list = ibs.db.add_cleanly(LABEL_TABLE, colnames, params_iter,
                                          ibs.get_label_rowid_from_keyval,
                                          unique_paramx=[1, 2])
        return labelid_list

    @adder
    def add_encounters(ibs, enctext_list):
        """ Adds a list of names. Returns their nids """
        if utool.VERBOSE:
            print('[ibs] adding %d encounters' % len(enctext_list))
        # Add encounter text names to database
        notes_list = ['' for _ in xrange(len(enctext_list))]
        encounter_uuid_list = [uuid.uuid4() for _ in xrange(len(enctext_list))]
        colnames = ['encounter_text', 'encounter_uuid', 'encounter_note']
        params_iter = izip(enctext_list, encounter_uuid_list, notes_list)
        get_rowid_from_uuid = partial(ibs.get_encounter_eids_from_text, ensure=False)

        eid_list = ibs.db.add_cleanly(ENCOUNTER_TABLE, colnames, params_iter, get_rowid_from_uuid)
        return eid_list

    #
    #
    #----------------
    # --- SETTERS ---
    #----------------

    # SETTERS::IMAGE

    @setter
    def set_image_uris(ibs, gid_list, new_gpath_list):
        """ Sets the image URIs to a new local path.
        This is used when localizing or unlocalizing images.
        An absolute path can either be on this machine or on the cloud
        A relative path is relative to the ibeis image cache on this machine.
        """
        id_iter = ((gid,) for gid in gid_list)
        val_list = ((new_gpath,) for new_gpath in new_gpath_list)
        ibs.db.set(IMAGE_TABLE, ('image_uri',), val_list, id_iter)

    @setter
    def set_image_aifs(ibs, gid_list, aif_list):
        """ Sets the image all instances found bit """
        id_iter = ((gid,) for gid in gid_list)
        val_list = ((aif,) for aif in aif_list)
        ibs.db.set(IMAGE_TABLE, ('image_toggle_aif',), val_list, id_iter)

    @setter
    def set_image_notes(ibs, gid_list, notes_list):
        """ Sets the image all instances found bit """
        id_iter = ((gid,) for gid in gid_list)
        val_list = ((notes,) for notes in notes_list)
        ibs.db.set(IMAGE_TABLE, ('image_note',), val_list, id_iter)

    @setter
    def set_image_unixtime(ibs, gid_list, unixtime_list):
        """ Sets the image unixtime (does not modify exif yet) """
        id_iter = ((gid,) for gid in gid_list)
        val_list = ((unixtime,) for unixtime in unixtime_list)
        ibs.db.set(IMAGE_TABLE, ('image_time_posix',), val_list, id_iter)

    @setter
    def set_image_enctext(ibs, gid_list, enctext_list):
        """ Sets the encoutertext of each image """
        if utool.VERBOSE:
            print('[ibs] setting %r image encounter ids' % len(gid_list))
        eid_list = ibs.add_encounters(enctext_list)
        egrid_list = ibs.add_image_relationship(gid_list, eid_list)
        del egrid_list

    @setter
    def set_image_gps(ibs, gid_list, gps_list=None, lat_list=None, lon_list=None):
        # see get_image_gps for how the gps_list should look
        if gps_list is not None:
            assert lat_list is None
            assert lon_list is None
            lat_list = [tup[0] for tup in gps_list]
            lon_list = [tup[1] for tup in gps_list]
        colnames = ('image_gps_lat', 'image_gps_lon',)
        val_list = izip(lat_list, lon_list)
        id_iter = ((gid,) for gid in gid_list)
        ibs.db.set(IMAGE_TABLE, colnames, val_list, id_iter)

    # SETTERS::ANNOTATION

    @setter
    def set_annotation_exemplar_flag(ibs, aid_list, flag_list):
        id_iter = ((aid,) for aid in aid_list)
        val_iter = ((flag,) for flag in flag_list)
        ibs.db.set(ANNOTATION_TABLE, ('annot_exemplar_flag',), val_iter, id_iter)

    @setter
    def set_annotation_bboxes(ibs, aid_list, bbox_list):
        """ Sets ANNOTATIONs of a list of annotations by aid, where annotation_list is a list of
            (x, y, w, h) tuples

        NOTICE: set_annotation_bboxes is a proxy for set_annotation_verts
        """
        # changing the bboxes also changes the bounding polygon
        vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
        # naively overwrite the bounding polygon with a rectangle - for now trust the user!
        ibs.set_annotation_verts(aid_list, vert_list)
        colnames = ['annot_xtl', 'annot_ytl', 'annot_width', 'annot_height']
        ibs.db.set(ANNOTATION_TABLE, colnames, bbox_list, aid_list)

    @setter
    def set_annotation_thetas(ibs, aid_list, theta_list):
        """ Sets thetas of a list of chips by aid """
        ibs.delete_annotation_chips(aid_list)  # Changing theta redefines the chips
        id_iter = ((aid,) for aid in aid_list)
        val_list = ((theta,) for theta in theta_list)
        ibs.db.set(ANNOTATION_TABLE, ('annot_theta',), val_list, id_iter)

    @setter
    def set_annotation_verts(ibs, aid_list, verts_list):
        """ Sets the vertices [(x, y), ...] of a list of chips by aid """
        num_params = len(aid_list)
        # Compute data to set
        num_verts_list   = imap(len, verts_list)
        verts_as_strings = imap(__STR__, verts_list)
        id_iter1 = ((aid,) for aid in aid_list)
        # also need to set the internal number of vertices
        val_iter1 = ((num_verts, verts) for (num_verts, verts)
                     in izip(num_verts_list, verts_as_strings))
        colnames = ('annot_num_verts', 'annot_verts',)
        # SET VERTS in ANNOTATION_TABLE
        ibs.db.set(ANNOTATION_TABLE, colnames, val_iter1, id_iter1, num_params=num_params)
        # changing the vertices also changes the bounding boxes
        bbox_list = geometry.bboxes_from_vert_list(verts_list)  	# new bboxes
        xtl_list, ytl_list, width_list, height_list = list(izip(*bbox_list))
        val_iter2 = izip(xtl_list, ytl_list, width_list, height_list)
        id_iter2 = ((aid,) for aid in aid_list)
        colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
        # SET BBOX in ANNOTATION_TABLE
        ibs.db.set(ANNOTATION_TABLE, colnames, val_iter2, id_iter2, num_params=num_params)
        ibs.delete_annotation_chips(aid_list)  # INVALIDATE THUMBNAILS

    @setter
    def set_annotation_notes(ibs, aid_list, notes_list):
        """ Sets annotation notes """
        id_iter = ((aid,) for aid in aid_list)
        val_list = ((notes,) for notes in notes_list)
        ibs.db.set(ANNOTATION_TABLE, ('annot_note',), val_list, id_iter)

    @setter
    def set_annotation_names(ibs, aid_list, name_list=None, nid_list=None):
        """ Sets names/nids of a list of annotations.
        Convenience function for set_annotation_nids"""
        assert name_list is None or nid_list is None, (
            'can only specify one type of name values (nid or name) not both')
        if nid_list is None:
            assert name_list is not None
            # Convert names into nids
            nid_list = ibs.add_names(name_list)
        
        alrids_list = ibs.get_annotation_filtered_alrids(aid_list, ibs.key_ids['INDIVIDUAL_KEY'])
        for aid, nid, alrid_list in izip(aid_list, nid_list, alrids_list):
            if len(alrid_list) == 0:
                ibs.add_annotation_relationship([aid], [nid])
            else:
                ibs.set_annotation_nids([aid], [nid], 'INDIVIDUAL_KEY')

    @setter
    def set_annotation_species(ibs, aid_list, species_list=None, nid_list=None):
        """ Sets names/nids of a list of annotations.
        Convenience function for set_annotation_nids"""
        assert species_list is None or nid_list is None, (
            'can only specify one type of name values (nid or name) not both')
        if nid_list is None:
            assert species_list is not None
            # Convert names into nids
            nid_list = ibs.add_species(species_list)

        alrids_list = ibs.get_annotation_filtered_alrids(aid_list, ibs.key_ids['SPECIES_KEY'])
        for aid, nid, alrid_list in izip(aid_list, nid_list, alrids_list):
            if len(alrid_list) == 0:
                ibs.add_annotation_relationship([aid], [nid])
            else:
                ibs.set_annotation_nids([aid], [nid], 'SPECIES_KEY')

    @setter
    def set_annotation_nids(ibs, aid_list, nid_list, _key):
        """ Sets nids of a list of annotations """
        # Ensure we are setting true nids (not temporary distinguished nids)
        # nids are really special labelids
        alrids_list = ibs.get_annotation_filtered_alrids(aid_list, ibs.key_ids[_key])
        # SQL Setter arguments
        # Cannot use set_table_props for cross-table setters.
        [ ibs.db.set(AL_RELATION_TABLE, ('label_rowid',), [nid] * len(alrid_list), alrid_list) 
            for nid, alrid_list in izip(nid_list, alrids_list) ]

    # SETTERS::NAME

    @setter
    def set_name_notes(ibs, nid_list, notes_list):
        """ Sets notes of names (groups of animals) """
        id_iter = ((nid,) for nid in nid_list)
        val_list = ((notes,) for notes in notes_list)
        ibs.db.set(LABEL_TABLE, ('label_note',), val_list, id_iter)

    @setter
    def set_name_names(ibs, nid_list, name_list):
        """ Changes the name text. Does not affect the animals of this name """
        ibsfuncs.assert_valid_names(name_list)
        id_iter = ((nid,) for nid in nid_list)
        val_list = ((name,) for name in name_list)
        ibs.db.set(LABEL_TABLE, ('label_value',), val_list, id_iter)

    @setter
    def set_encounter_props(ibs, eid_list, key, value_list):
        print('[ibs] set_encounter_props')
        id_iter = ((eid,) for eid in eid_list)
        val_list = ((value,) for value in value_list)
        ibs.db.set(ENCOUNTER_TABLE, key, val_list, id_iter)

    @setter
    def set_encounter_enctext(ibs, eid_list, names_list):
        """ Sets names of encounters (groups of animals) """
        id_iter = ((eid,) for eid in eid_list)
        val_list = ((names,) for names in names_list)
        ibs.db.set(ENCOUNTER_TABLE, ('encounter_text',), val_list, id_iter)

    # SETTERS::ALR

    @setter
    def set_alr_confidence(ibs, alrid_list, confidence_list):
        """ sets annotation-label-relationship confidence """
        id_iter = ((alrid,) for alrid in alrid_list)
        val_iter = ((confidence,) for confidence in confidence_list)
        colnames = ('alr_confidence')
        ibs.db.set(AL_RELATION_TABLE, colnames, val_iter, id_iter)

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    #
    # GETTERS::IMAGE_TABLE

    @getter_1to1
    def get_images(ibs, gid_list):
        """ Returns a list of images in numpy matrix form by gid """
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    @getter_1to1
    def get_image_thumbtup(ibs, gid_list):
        """ Returns tuple of image paths, where the thumb path should go,
        and any bboxes """
        aids_list = ibs.get_image_aids(gid_list)
        bboxes_list = ibsfuncs.unflat_map(ibs.get_annotation_bboxes, aids_list)
        thetas_list = ibsfuncs.unflat_map(ibs.get_annotation_thetas, aids_list)
        thumb_gpaths = ibs.get_image_thumbpath(gid_list)
        image_paths = ibs.get_image_paths(gid_list)
        thumbtup_list = list(izip(thumb_gpaths, image_paths, bboxes_list, thetas_list))
        return thumbtup_list

    @getter_1to1
    def get_image_thumbpath(ibs, gid_list):
        thumb_dpath = ibs.thumb_dpath
        img_uuid_list = ibs.get_image_uuids(gid_list)
        thumbpath_list = [join(thumb_dpath, __STR__(uuid) + constants.IMAGE_THUMB_SUFFIX)
                          for uuid in img_uuid_list]
        return thumbpath_list

    @getter_1to1
    def get_image_uuids(ibs, gid_list):
        """ Returns a list of image uuids by gid """
        image_uuid_list = ibs.db.get(IMAGE_TABLE, ('image_uuid',), gid_list)
        return image_uuid_list

    @getter_1to1
    def get_image_exts(ibs, gid_list):
        """ Returns a list of image uuids by gid """
        image_uuid_list = ibs.db.get(IMAGE_TABLE, ('image_ext',), gid_list)
        return image_uuid_list

    @getter_1to1
    def get_image_uris(ibs, gid_list):
        """ Returns a list of image uris by gid """
        uri_list = ibs.db.get(IMAGE_TABLE, ('image_uri',), gid_list)
        return uri_list

    @getter_1to1
    def get_image_gids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        gid_list = ibs.db.get(IMAGE_TABLE, ('image_rowid',), uuid_list, id_colname='image_uuid')
        return gid_list

    @getter_1to1
    def get_image_paths(ibs, gid_list):
        """ Returns a list of image paths relative to img_dir? by gid """
        uri_list = ibs.get_image_uris(gid_list)
        # Images should never have null uris
        utool.assert_all_not_None(uri_list, 'uri_list', key_list=['uri_list', 'gid_list'])
        gpath_list = [join(ibs.imgdir, uri) for uri in uri_list]
        return gpath_list

    @getter_1to1
    def get_image_detectpaths(ibs, gid_list):
        """ Returns a list of image paths resized to a constant area for detection """
        new_gfpath_list = preproc_detectimg.compute_and_write_detectimg_lazy(ibs, gid_list)
        return new_gfpath_list

    @getter_1to1
    def get_image_gnames(ibs, gid_list):
        """ Returns a list of original image names """
        gname_list = ibs.db.get(IMAGE_TABLE, ('image_original_name',), gid_list)
        return gname_list

    @getter_1to1
    def get_image_sizes(ibs, gid_list):
        """ Returns a list of (width, height) tuples """
        gsize_list = ibs.db.get(IMAGE_TABLE, ('image_width', 'image_height'), gid_list)
        return gsize_list

    @utool.accepts_numpy
    @getter_1to1
    def get_image_unixtime(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        return ibs.db.get(IMAGE_TABLE, ('image_time_posix',), gid_list)

    @getter_1to1
    def get_image_gps(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        gps_list = ibs.db.get(IMAGE_TABLE, ('image_gps_lat', 'image_gps_lon'), gid_list)
        return gps_list

    @getter_1to1
    def get_image_lat(ibs, gid_list):
        lat_list = ibs.db.get(IMAGE_TABLE, ('image_gps_lat',), gid_list)
        return lat_list

    @getter_1to1
    def get_image_lon(ibs, gid_list):
        lon_list = ibs.db.get(IMAGE_TABLE, ('image_gps_lon',), gid_list)
        return lon_list

    @getter_1to1
    def get_image_aifs(ibs, gid_list):
        """ Returns "All Instances Found" flag, true if all objects of interest
        (animals) have an ANNOTATION in the image """
        aif_list = ibs.db.get(IMAGE_TABLE, ('image_toggle_aif',), gid_list)
        return aif_list

    @getter_1to1
    def get_image_detect_confidence(ibs, gid_list):
        """ Returns image detection confidence as the max of ANNOTATION confidences """
        aids_list = ibs.get_image_aids(gid_list)
        confs_list = ibsfuncs.unflat_map(ibs.get_annotation_detect_confidence, aids_list)
        maxconf_list = [max(confs) if len(confs) > 0 else -1 for confs in confs_list]
        return maxconf_list

    @getter_1to1
    def get_image_notes(ibs, gid_list):
        """ Returns image notes """
        notes_list = ibs.db.get(IMAGE_TABLE, ('image_note',), gid_list)
        return notes_list

    @getter_1to1
    def get_image_nids(ibs, gid_list):
        """ Returns the name ids associated with an image id """
        aids_list = ibs.get_image_aids(gid_list)
        kwargs = {
            '_key': 'INDIVIDUAL_KEY'
        }
        nids_list = ibsfuncs.unflat_map(ibs.get_annotation_nids, aids_list **kwargs)
        return nids_list

    @getter_1toM
    def get_image_eids(ibs, gid_list):
        """ Returns a list of encounter ids for each image by gid """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        colnames = ('encounter_rowid',)
        eids_list = ibs.db.get(EG_RELATION_TABLE, colnames, gid_list, id_colname='image_rowid', unpack_scalars=False)
        return eids_list

    @getter_1toM
    def get_image_enctext(ibs, gid_list):
        """ Returns a list of enctexts for each image by gid """
        eids_list = ibs.get_image_eids(gid_list)
        enctext_list = ibsfuncs.unflat_map(ibs.get_encounter_enctext, eids_list)
        return enctext_list

    @getter_1toM
    def get_image_aids(ibs, gid_list):
        """ Returns a list of aids for each image by gid """
        #print('gid_list = %r' % (gid_list,))
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        colnames = ('annot_rowid',)
        aids_list = ibs.db.get(ANNOTATION_TABLE, colnames, gid_list, id_colname='image_rowid', unpack_scalars=False)
        #print('aids_list = %r' % (aids_list,))
        return aids_list

    @getter_1to1
    def get_image_num_annotations(ibs, gid_list):
        """ Returns the number of chips in each image """
        return list(imap(len, ibs.get_image_aids(gid_list)))

    @getter_1to1
    def get_image_egrids(ibs, gid_list):
        """ Gets a list of encounter-image-relationship rowids for each imageid """
        # TODO: Group type
        params_iter = ((gid,) for gid in gid_list)
        where_clause = 'image_rowid=?'
        # list of relationships for each image
        egrids_list = ibs.db.get_where(EG_RELATION_TABLE, ('egr_rowid',), params_iter, where_clause, unpack_scalars=False)
        return egrids_list

    #
    # GETTERS::ANNOTATION_TABLE

    @getter_1to1
    def get_annotation_exemplar_flag(ibs, aid_list):
        annotation_uuid_list = ibs.db.get(ANNOTATION_TABLE, ('annot_exemplar_flag',), aid_list)
        return annotation_uuid_list

    @getter_1to1
    def get_annotation_uuids(ibs, aid_list):
        """ Returns a list of image uuids by gid """
        annotation_uuid_list = ibs.db.get(ANNOTATION_TABLE, ('annot_uuid',), aid_list)
        return annotation_uuid_list

    @getter_1to1
    def get_annotation_aids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        aids_list = ibs.db.get(ANNOTATION_TABLE, ('annot_rowid',), uuid_list, id_colname='annot_uuid')
        return aids_list

    @getter_1to1
    def get_annotation_detect_confidence(ibs, aid_list):
        """ Returns a list confidences that the annotations is a valid detection """
        annotation_detect_confidence_list = ibs.db.get(ANNOTATION_TABLE, ('annot_detect_confidence',), aid_list)
        return annotation_detect_confidence_list

    @getter_1to1
    def get_annotation_notes(ibs, aid_list):
        """ Returns a list of annotation notes """
        annotation_notes_list = ibs.db.get(ANNOTATION_TABLE, ('annot_note',), aid_list)
        return annotation_notes_list

    @utool.accepts_numpy
    @getter_1toM
    def get_annotation_bboxes(ibs, aid_list):
        """ returns annotation bounding boxes in image space """
        colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height')
        bbox_list = ibs.db.get(ANNOTATION_TABLE, colnames, aid_list)
        return bbox_list

    @getter_1to1
    def get_annotation_thetas(ibs, aid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.db.get(ANNOTATION_TABLE, ('annot_theta',), aid_list)
        return theta_list

    @getter_1to1
    def get_annotation_num_verts(ibs, aid_list):
        """ Returns the number of vertices that form the polygon of each chip """
        num_verts_list = ibs.db.get(ANNOTATION_TABLE, ('annot_num_verts',), aid_list)
        return num_verts_list

    @getter_1to1
    def get_annotation_verts(ibs, aid_list):
        """ Returns the vertices that form the polygon of each chip """
        vertstr_list = ibs.db.get(ANNOTATION_TABLE, ('annot_verts',), aid_list)
        # TODO: Sanatize input for eval
        #print('vertstr_list = %r' % (vertstr_list,))
        vert_list = [eval(vertstr) for vertstr in vertstr_list]
        return vert_list

    @utool.accepts_numpy
    @getter_1to1
    def get_annotation_gids(ibs, aid_list):
        """ returns annotation bounding boxes in image space """
        gid_list = ibs.db.get(ANNOTATION_TABLE, ('image_rowid',), aid_list)
        return gid_list

    @getter_1to1
    def get_annotation_cids(ibs, aid_list, ensure=True, all_configs=False):
        # FIXME:
        if ensure:
            try:
                ibs.add_chips(aid_list)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.get_annotation_cids]')
                print('[!ibs.get_annotation_cids] aid_list = %r' % (aid_list,))
                raise
        if all_configs:
            # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
            cid_list = ibs.db.get(CHIP_TABLE, ('chip_rowid',), aid_list, id_colname='annot_rowid')
        else:
            chip_config_rowid = ibs.get_chip_config_rowid()
            #print(chip_config_rowid)
            where_clause = 'annot_rowid=? AND config_rowid=?'
            params_iter = ((aid, chip_config_rowid) for aid in aid_list)
            cid_list = ibs.db.get_where(CHIP_TABLE,  ('chip_rowid',), params_iter, where_clause)
        if ensure:
            try:
                utool.assert_all_not_None(cid_list, 'cid_list')
            except AssertionError as ex:
                valid_cids = ibs.get_valid_cids()  # NOQA
                utool.printex(ex, 'Ensured cids returned None!',
                              key_list=['aid_list', 'cid_list', 'valid_cids'])
                raise
        return cid_list

    @getter_1to1
    def get_annotation_fids(ibs, aid_list, ensure=False):
        cid_list = ibs.get_annotation_cids(aid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        return fid_list

    @getter_1to1
    def get_key_rowid_from_text(ibs, text_list):
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        key_rowid = ibs.db.get(KEY_TABLE, ('key_rowid',), text_list, id_colname='key_text')
        return key_rowid

    @getter_1to1
    def get_key_default(ibs, kid_list):
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        key_defaults = ibs.db.get(KEY_TABLE, ('key_default',), kid_list)
        return key_defaults

    @getter_1toM
    def get_annotation_alrids(ibs, aid_list, configid=None):
        """ FIXME: func_name
        Get all the relationship ids belonging to the input annotations
        if label key is specified the relationship ids are filtered to
        be only of a specific key/category/type
        """
        if configid is None:
            configid = ibs.MANUAL_CONFIGID
        params_iter = ((aid, configid) for aid in aid_list)
        where_clause = 'annot_rowid=? AND config_rowid=?'
        alrids_list = ibs.db.get_where(AL_RELATION_TABLE, ('alr_rowid',), params_iter,
                                       where_clause=where_clause, unpack_scalars=False)
        # assert all([x > 0 for x in map(len, alrids_list)]), 'annotations must have at least one relationship'
        return alrids_list

    @getter_1toM
    def get_annotation_filtered_alrids(ibs, aid_list, key_rowid, configid=None):
        """ FIXME: func_name
        Get all the relationship ids belonging to the input annotations where the
        relationship ids are filtered to be only of a specific key/category/type
        """
        alrids_list = ibs.get_annotation_alrids(aid_list, configid=configid)
        # Get labelid of each relationship
        labelids_list = [ ibs.get_alr_labelids(alr_list) for alr_list in alrids_list ]
        # Get the type of each label
        labelkeys_list = [ ibs.get_label_keys(labelid_list) for labelid_list in labelids_list ]
        # only want the nids of individuals, not species, for example
        alrid_list = [ 
                        [ 
                            alrids_list[index_label][index_key] 
                            for index_key, key in enumerate(labelkeys) if key == key_rowid 
                        ] 
                        for index_label, labelkeys in enumerate(labelkeys_list) 
                    ]
        
        # for index_label, labelkeys in enumerate(labelkeys_list):
        #     temp = []
        #     for index_key, key in enumerate(labelkeys):
        #         if key == key_rowid:
        #             temp.append(alrids_list[index_label][index_key])
        #     alrid_list.append(temp)
        return alrid_list

    @utool.accepts_numpy
    @getter_1toM
    def get_annotation_nids(ibs, aid_list, _key):
        """ Returns the name id of each annotation. """
        # Get all the annotation label relationships
        # filter out only the ones which specify names
        alrid_list = ibs.get_annotation_filtered_alrids(aid_list, ibs.key_ids[_key])
        return [ ibs.get_alr_labelids(alrid) for alrid in alrid_list ] 
        
    @getter_1to1
    def get_annotation_gnames(ibs, aid_list):
        """ Returns the image names of each annotation """
        gid_list = ibs.get_annotation_gids(aid_list)
        gname_list = ibs.get_image_gnames(gid_list)
        return gname_list

    @getter_1to1
    def get_annotation_images(ibs, aid_list):
        """ Returns the images of each annotation """
        gid_list = ibs.get_annotation_gids(aid_list)
        image_list = ibs.get_images(gid_list)
        return image_list

    @getter_1to1
    def get_annotation_image_uuids(ibs, aid_list):
        gid_list = ibs.get_annotation_gids(aid_list)
        image_uuid_list = ibs.get_image_uuids(gid_list)
        return image_uuid_list

    @getter_1to1
    def get_annotation_gpaths(ibs, aid_list):
        """ Returns the image names of each annotation """
        gid_list = ibs.get_annotation_gids(aid_list)
        try:
            utool.assert_all_not_None(gid_list, 'gid_list')
        except AssertionError:
            print('[!get_annotation_gpaths] ' + utool.list_dbgstr('aid_list'))
            print('[!get_annotation_gpaths] ' + utool.list_dbgstr('gid_list'))
            raise
        gpath_list = ibs.get_image_paths(gid_list)
        utool.assert_all_not_None(gpath_list, 'gpath_list')
        return gpath_list

    @getter_1to1
    def get_annotation_chips(ibs, aid_list, ensure=True):
        utool.assert_all_not_None(aid_list, 'aid_list')
        cid_list = ibs.get_annotation_cids(aid_list, ensure=ensure)
        if ensure:
            try:
                utool.assert_all_not_None(cid_list, 'cid_list')
            except AssertionError as ex:
                utool.printex(ex, 'Invalid cid_list', key_list=[
                    'ensure', 'cid_list'])
                raise
        chip_list = ibs.get_chips(cid_list, ensure=ensure)
        return chip_list

    @getter_1to1
    def get_annotation_chip_thumbtup(ibs, aid_list):
        thumb_gpaths = ibs.get_annotation_chip_thumbpath(aid_list)
        image_paths = ibs.get_annotation_cpaths(aid_list)
        thumbtup_list = [(thumb_path, img_path, [], [])
                         for (thumb_path, img_path) in
                         izip(thumb_gpaths, image_paths,)]
        return thumbtup_list
    
    @getter_1to1
    def get_annotation_chip_thumbpath(ibs, aid_list):
        annotation_uuid_list = ibs.get_annotation_uuids(aid_list)
        thumbpath_list = [join(ibs.thumb_dpath, __STR__(uuid) + constants.CHIP_THUMB_SUFFIX)
                            for uuid in annotation_uuid_list]
        return thumbpath_list

    @utool.accepts_numpy
    @getter_1toM
    def get_annotation_kpts(ibs, aid_list, ensure=True):
        """ Returns chip keypoints """
        fid_list  = ibs.get_annotation_fids(aid_list, ensure=ensure)
        kpts_list = ibs.get_feat_kpts(fid_list)
        return kpts_list

    @getter_1to1
    def get_annotation_chipsizes(ibs, aid_list, ensure=True):
        """ Returns the imagesizes of computed annotation chips """
        cid_list  = ibs.get_annotation_cids(aid_list, ensure=ensure)
        chipsz_list = ibs.get_chip_sizes(cid_list)
        return chipsz_list

    @getter_1toM
    def get_annotation_desc(ibs, aid_list, ensure=True):
        """ Returns chip descriptors """
        fid_list  = ibs.get_annotation_fids(aid_list, ensure=ensure)
        desc_list = ibs.get_feat_desc(fid_list)
        return desc_list

    @getter_1to1
    def get_annotation_cpaths(ibs, aid_list):
        """ Returns cpaths defined by ANNOTATIONs """
        utool.assert_all_not_None(aid_list, 'aid_list')
        cfpath_list = preproc_chip.get_annotation_cfpath_list(ibs, aid_list)
        return cfpath_list

    @getter_1to1
    def get_annotation_labels(ibs, aid_list):
        """ 
        """
        def _key_dict(aid):
            _dict = {}
            for _key in ibs.key_names:
                _dict[_key] = ibs.get_annotation_nids(aid, _key)
            return _dict

        key_dict_list = [ _key_dict(aid) for aid in aid_list ]
        return key_dict_list

    @getter_1to1
    def get_annotation_from_key(ibs, aid_list, _key, getter):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        key_dict_list = ibs.get_annotation_labels(aid_list)
        key_list = [ (
                        getter(key_dict[_key])[0]
                            if len(key_dict[_key]) > 0 else
                        ibs.key_defaults[_key]
                      )
                      for key_dict in key_dict_list
                    ]
        return key_list

    @getter_1to1
    def get_annotation_names(ibs, aid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        return ibs.get_annotation_from_key(aid_list, 'INDIVIDUAL_KEY', ibs.get_names)
    
    @getter_1to1
    def get_annotation_species(ibs, aid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        return ibs.get_annotation_from_key(aid_list, 'SPECIES_KEY', ibs.get_species)

    @getter_1toM
    def get_annotation_groundtruth(ibs, aid_list):
        """ Returns a list of aids with the same name foreach aid in aid_list.
        a set of aids belonging to the same name is called a groundtruth. A list
        of these is called a groundtruth_list. """
        def _individual_ground_truth(nids_list):
            where_clause = 'label_rowid=? AND annot_rowid!=?'
            params_iter = [(nid, aid) for nid, aid in izip(nids_list, aid_list)]
            groundtruth_list = ibs.db.get_where(AL_RELATION_TABLE, ('annot_rowid',), params_iter,
                                                where_clause,
                                                unpack_scalars=False)
            return utool.flatten(groundtruth_list)

        nids_list  = ibs.get_annotation_nids(aid_list, 'INDIVIDUAL_KEY')
        groundtruth_list = [ _individual_ground_truth(nids) for nids in nids_list ]
        return groundtruth_list

    @getter_1to1
    def get_annotation_num_groundtruth(ibs, aid_list):
        """ Returns number of other chips with the same name """
        return list(imap(len, ibs.get_annotation_groundtruth(aid_list)))

    @getter_1to1
    def get_annotation_num_feats(ibs, aid_list, ensure=False):
        cid_list = ibs.get_annotation_cids(aid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        nFeats_list = ibs.get_num_feats(fid_list)
        return nFeats_list

    @getter_1to1
    def get_annotation_has_groundtruth(ibs, aid_list):
        numgts_list = ibs.get_annotation_num_groundtruth(aid_list)
        has_gt_list = [num_gts > 0 for num_gts in numgts_list]
        return has_gt_list

    #
    # GETTERS::AL_RELATION_TABLE

    @getter_1to1
    def get_alr_rowid_from_valtup(ibs, aid_list, labelid_list, configid_list):
        colnames = ('annot_rowid',)
        params_iter = izip(aid_list, labelid_list, configid_list)
        where_clause = 'annot_rowid=? AND label_rowid=? AND config_rowid=?'
        alrid_list = ibs.db.get_where(AL_RELATION_TABLE, colnames, params_iter, where_clause)
        return alrid_list

    @getter_1to1
    def get_alr_labelids(ibs, alrid_list):
        """ get the labelid belonging to each relationship """
        labelids_list = ibs.db.get(AL_RELATION_TABLE, ('label_rowid',), alrid_list)
        return labelids_list

    #
    # GETTERS::EG_RELATION_TABLE

    @getter_1to1
    def get_egr_rowid_from_valtup(ibs, gid_list, eid_list):
        """ Gets eg-relate-ids from info constrained to be unique (eid, gid) """
        colnames = ('image_rowid',)
        params_iter = izip(gid_list, eid_list)
        where_clause = 'image_rowid=? AND encounter_rowid=?'
        egrid_list = ibs.db.get_where(EG_RELATION_TABLE, colnames, params_iter, where_clause)
        return egrid_list

    #
    # GETTERS::LABEL_TABLE

    @getter_1to1
    def get_label_rowid_from_keyval(ibs, key_list, value_list):
        colnames = ('label_rowid',)
        params_iter = izip(key_list, value_list)
        where_clause = 'key_rowid=? AND label_value=?'
        labelid_list = ibs.db.get_where(LABEL_TABLE, colnames, params_iter, where_clause)
        return labelid_list

    @getter_1to1
    def get_labelid_from_uuid(ibs, label_uuid_list):
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        labelid_list = ibs.db.get(LABEL_TABLE, ('label_rowid',), label_uuid_list, id_colname='label_uuid')
        return labelid_list

    @getter_1to1
    def get_labelids_from_values(ibs, value_list, key_rowid):
        params_iter = [(value, key_rowid) for value in value_list]
        where_clause = 'label_value=? AND key_rowid=?'
        labelid_list = ibs.db.get_where(LABEL_TABLE, ('label_rowid',), params_iter, where_clause)
        return labelid_list

    @getter_1to1
    def get_label_uuids(ibs, labelid_list):
        labelkey_list = ibs.db.get(LABEL_TABLE, ('label_uuid',), labelid_list)
        return labelkey_list

    @getter_1to1
    def get_label_keys(ibs, labelid_list):
        labelkey_list = ibs.db.get(LABEL_TABLE, ('key_rowid',), labelid_list)
        return labelkey_list

    @getter_1to1
    def get_label_values(ibs, labelid_list):
        labelkey_list = ibs.db.get(LABEL_TABLE, ('label_value',), labelid_list)
        return labelkey_list

    @getter_1to1
    def get_label_notes(ibs, labelid_list):
        labelkey_list = ibs.db.get(LABEL_TABLE, ('label_note',), labelid_list)
        return labelkey_list

    #
    # GETTERS::NAME (subset of LABELS_TABLE)
    @getter_1to1
    def get_name_nids(ibs, name_list, ensure=True):
        """ Returns nid_list. Creates one if it doesnt exist """
        if ensure:
            ibs.add_names(name_list)
        nid_list = ibs.get_labelids_from_values(name_list, ibs.key_ids['INDIVIDUAL_KEY'])

        return nid_list

    @getter_1to1
    def get_names(ibs, nid_list):
        """ Returns text names """
        #print('get_names: %r' % nid_list)
        # Change the temporary negative indexes back to the unknown NID for the
        # SQL query. Then augment the name list to distinguish unknown names
        key_rowid_list = ibs.get_label_keys(nid_list)
        assert all([key == ibs.key_ids['INDIVIDUAL_KEY']
                    for key in key_rowid_list]), 'label_rowids are not individual_ids'
        name_list = ibs.db.get(LABEL_TABLE, ('label_value',), nid_list)
        return name_list

    @getter_1to1
    def get_species(ibs, nid_list):
        """ Returns text names """
        #print('get_species: %r' % nid_list)
        # Change the temporary negative indexes back to the unknown NID for the
        # SQL query. Then augment the name list to distinguish unknown names
        key_rowid_list = ibs.get_label_keys(nid_list)
        assert all([key == ibs.key_ids['SPECIES_KEY']
                    for key in key_rowid_list]), 'label_rowids are not species_ids'
        species_list = ibs.db.get(LABEL_TABLE, ('label_value',), nid_list)
        return species_list

    @getter_1toM
    def get_name_aids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        where_clause = 'label_rowid=?'
        params_iter = [(nid,) for nid in nid_list]
        aids_list = ibs.db.get_where(AL_RELATION_TABLE, ('annot_rowid',), params_iter,
                                     where_clause, unpack_scalars=False)
        return aids_list

    @getter_1toM
    def get_name_annotation_bboxes(ibs, nid_list):
        aids_list = ibs.get_name_aids(nid_list)
        bboxes_list = ibsfuncs.unflat_map(ibs.get_annotation_bboxes, aids_list)
        return bboxes_list

    @getter_1to1
    def get_name_thumbtups(ibs, nid_list):
        aids_list = ibs.get_name_aids(nid_list)
        thumbtups_list_ = ibsfuncs.unflat_map(ibs.get_annotation_chip_thumbtup, aids_list)
        thumbtups_list = utool.flatten(thumbtups_list_)
        return thumbtups_list

    @getter_1to1
    def get_name_num_annotations(ibs, nid_list):
        """ returns the number of detections for each name """
        return list(imap(len, ibs.get_name_aids(nid_list)))

    @getter_1to1
    def get_name_notes(ibs, nid_list):
        """ Returns name notes """
        notes_list = ibs.get_label_notes(nid_list)
        return notes_list

    @getter_1toM
    def get_name_gids(ibs, nid_list):
        """ Returns the image ids associated with name ids"""
        aids_list = ibs.get_name_aids(nid_list)
        gids_list = ibsfuncs.unflat_map(ibs.get_annotation_gids, aids_list)
        return gids_list

    #
    # GETTERS::CHIP_TABLE

    @getter_1to1
    def get_chips(ibs, cid_list, ensure=True):
        """ Returns a list cropped images in numpy array form by their cid """
        aid_list = ibs.get_chip_aids(cid_list)
        chip_list = preproc_chip.compute_or_read_annotation_chips(ibs, aid_list, ensure=ensure)
        return chip_list

    @getter_1to1
    def get_chip_aids(ibs, cid_list):
        aid_list = ibs.db.get(CHIP_TABLE, ('annot_rowid',), cid_list)
        return aid_list

    @getter_1to1
    def get_chip_paths(ibs, cid_list):
        """ Returns a list of chip paths by their aid """
        chip_fpath_list = ibs.db.get(CHIP_TABLE, ('chip_uri',), cid_list)
        return chip_fpath_list

    @getter_1to1
    def get_chip_sizes(ibs, cid_list):
        chipsz_list  = ibs.db.get(CHIP_TABLE, ('chip_width', 'chip_height',), cid_list)
        return chipsz_list

    @getter_1to1
    def get_chip_fids(ibs, cid_list, ensure=True):
        if ensure:
            ibs.add_feats(cid_list)
        feat_config_rowid = ibs.get_feat_config_rowid()
        colnames = ('feature_rowid',)
        where_clause = 'chip_rowid=? AND config_rowid=?'
        params_iter = ((cid, feat_config_rowid) for cid in cid_list)
        fid_list = ibs.db.get_where(FEATURE_TABLE, colnames, params_iter,
                                    where_clause)
        return fid_list

    @getter_1to1
    def get_chip_configids(ibs, cid_list):
        configid_list = ibs.db.get(CHIP_TABLE, ('config_rowid',), cid_list)
        return configid_list

    #
    # GETTERS::FEATURE_TABLE

    @getter_1toM
    def get_feat_kpts(ibs, fid_list):
        """ Returns chip keypoints in [x, y, iv11, iv21, iv22, ori] format """
        kpts_list = ibs.db.get(FEATURE_TABLE, ('feature_keypoints',), fid_list)
        return kpts_list

    @getter_1toM
    def get_feat_desc(ibs, fid_list):
        """ Returns chip SIFT descriptors """
        desc_list = ibs.db.get(FEATURE_TABLE, ('feature_sifts',), fid_list)
        return desc_list

    @getter_1to1
    def get_num_feats(ibs, fid_list):
        """ Returns the number of keypoint / descriptor pairs """
        nFeats_list = ibs.db.get(FEATURE_TABLE, ('feature_num_feats',), fid_list)
        nFeats_list = [(-1 if nFeats is None else nFeats) for nFeats in nFeats_list]
        return nFeats_list

    #
    # GETTERS::CONFIG_TABLE
    @getter_1to1
    def get_config_rowid_from_suffix(ibs, cfgsuffix_list, ensure=True):
        """
        Adds an algorithm configuration as a string
        """
        # FIXME: cfgsuffix should be renamed cfgstr? cfgtext?
        if ensure:
            return ibs.add_config(cfgsuffix_list)
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        configid_list = ibs.db.get(CONFIG_TABLE, ('config_rowid',), cfgsuffix_list, id_colname='config_suffix')

        # executeone always returns a list
        #if configid_list is not None and len(configid_list) == 1:
        #    configid_list = configid_list[0]
        return configid_list

    @getter_1to1
    def get_config_suffixes(ibs, configid_list):
        """ Gets suffixes for algorithm configs """
        cfgsuffix_list = ibs.db.get(CONFIG_TABLE, ('config_suffix',), configid_list)
        return cfgsuffix_list

    #
    # GETTERS::ENCOUNTER
    @getter_1to1
    def get_encounter_num_gids(ibs, eid_list):
        """ Returns number of images in each encounter """
        return list(imap(len, ibs.get_encounter_gids(eid_list)))

    @getter_1toM
    def get_encounter_aids(ibs, eid_list):
        """ returns a list of list of aids in each encounter """
        gids_list = ibs.get_encounter_gids(eid_list)
        aids_list_ = ibsfuncs.unflat_map(ibs.get_image_aids, gids_list)
        aids_list = list(imap(utool.flatten, aids_list_))
        #print('get_encounter_aids')
        #print('eid_list = %r' % (eid_list,))
        #print('gids_list = %r' % (gids_list,))
        #print('aids_list_ = %r' % (aids_list_,))
        #print('aids_list = %r' % (aids_list,))
        return aids_list

    @getter_1toM
    def get_encounter_gids(ibs, eid_list):
        """ returns a list of list of gids in each encounter """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        gids_list = ibs.db.get(EG_RELATION_TABLE, ('image_rowid',), eid_list, id_colname='encounter_rowid', unpack_scalars=False)
        #print('get_encounter_gids')
        #print('eid_list = %r' % (eid_list,))
        #print('gids_list = %r' % (gids_list,))
        return gids_list

    @getter_1to1
    def get_encounter_egrids(ibs, eid_list):
        """ Gets a list of encounter-image-relationship rowids for each encouterid """
        # TODO: Group type
        params_iter = ((eid,) for eid in eid_list)
        where_clause = 'encounter_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
        return egrids_list

    @getter_1toM
    def get_encounter_nids(ibs, eid_list):
        """ returns a list of list of nids in each encounter """
        aids_list = ibs.get_encounter_aids(eid_list)
        nids_list = [ ibs.get_annotation_nids(aid_list, 'INDIVIDUAL_KEY') for aid_list in aids_list ] 
        nids_list_ = []
        for nids in nids_list:
            temp = []
            for nid in nids:
                if len(nid) > 0:
                    temp.append(nid[0])
            nids_list_.append(temp)

        nids_list = list(imap(utool.unique_ordered, nids_list_))
        #print('get_encounter_nids')
        #print('eid_list = %r' % (eid_list,))
        #print('aids_list = %r' % (aids_list,))
        #print('nids_list_ = %r' % (nids_list_,))
        #print('nids_list = %r' % (nids_list,))
        return nids_list

    @getter_1to1
    def get_encounter_enctext(ibs, eid_list):
        """ Returns encounter_text of each eid in eid_list """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        enctext_list = ibs.db.get(ENCOUNTER_TABLE, ('encounter_text',), eid_list, id_colname='encounter_rowid')
        #enctext_list = list(imap(__STR__, enctext_list))
        return enctext_list

    @getter_1to1
    def get_encounter_eids_from_text(ibs, enctext_list, ensure=True):
        """ Returns a list of eids corresponding to each encounter enctext
        #FIXME: make new naming scheme for non-primary-key-getters
        get_encounter_eids_from_text_from_text
        """
        if ensure:
            ibs.add_encounters(enctext_list)
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        eid_list = ibs.db.get(ENCOUNTER_TABLE, ('encounter_rowid',), enctext_list, id_colname='encounter_text')
        return eid_list

    #
    #
    #-----------------
    # --- DELETERS ---
    #-----------------

    @deleter
    def delete_names(ibs, nid_list):
        """ deletes names from the database (CAREFUL. YOU PROBABLY DO NOT WANT
        TO USE THIS ENSURE THAT NONE OF THE NIDS HAVE ANNOTATION_TABLE) """
        ibs.delete_labels(nid_list)

    @deleter
    def delete_labels(ibs, labelid_list):
        """ deletes labels from the database """
        if utool.VERBOSE:
            print('[ibs] deleting %d labels' % len(labelid_list))
        ibs.db.delete_rowids(LABEL_TABLE, labelid_list)

    @deleter
    def delete_annotations(ibs, aid_list):
        """ deletes annotations from the database """
        if utool.VERBOSE:
            print('[ibs] deleting %d annotations' % len(aid_list))
        # Delete chips and features first
        ibs.delete_annotation_chips(aid_list)
        ibs.db.delete_rowids(ANNOTATION_TABLE, aid_list)

    @deleter
    def delete_annotation_nids(ibs, aid_list, _key):
        """ Deletes nids of a list of annotations """
        # Ensure we are setting true nids (not temporary distinguished nids)
        # nids are really special labelids
        alrid_list = ibs.get_annotation_filtered_alrids(aid_list, ibs.key_ids[_key])
        # SQL Setter arguments
        # Cannot use set_table_props for cross-table setters.
        [ ibs.db.delete_rowids(AL_RELATION_TABLE, alrid) for alrid in alrid_list ]

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes images from the database that belong to gids"""
        if utool.VERBOSE:
            print('[ibs] deleting %d images' % len(gid_list))
        # TODO: Move localized images to a trash folder
        # Delete annotations first
        aid_list = utool.flatten(ibs.get_image_aids(gid_list))
        ibs.delete_annotations(aid_list)
        ibs.db.delete_rowids(IMAGE_TABLE, gid_list)
        #egrid_list = utool.flatten(ibs.get_image_egrids(gid_list))
        #ibs.db.delete_rowids(EG_RELATION_TABLE, egrid_list)
        ibs.db.delete(EG_RELATION_TABLE, gid_list, id_colname='image_rowid')

    @deleter
    def delete_features(ibs, fid_list):
        """ deletes images from the database that belong to fids"""
        if utool.VERBOSE:
            print('[ibs] deleting %d features' % len(fid_list))
        ibs.db.delete_rowids(FEATURE_TABLE, fid_list)

    @deleter
    def delete_annotation_chips(ibs, aid_list):
        """ Clears annotation data but does not remove the annotation """
        _cid_list = ibs.get_annotation_cids(aid_list, ensure=False)
        cid_list = utool.filter_Nones(_cid_list)
        ibs.delete_chips(cid_list)
        gid_list = ibs.get_annotation_gids(aid_list)
        ibs.delete_image_thumbtups(gid_list)
        ibs.delete_annotation_chip_thumbs(aid_list)

    @deleter
    def delete_image_thumbtups(ibs, gid_list):
        """ Removes image thumbnails from disk """
        thumbtup_list = ibs.get_image_thumbtup(gid_list)
        thumbpath_list = [tup[0] for tup in thumbtup_list]
        utool.remove_file_list(thumbpath_list)

    @deleter
    def delete_annotation_chip_thumbs(ibs, aid_list):
        """ Removes chip thumbnails from disk """
        thumbtup_list = ibs.get_annotation_chip_thumbtup(aid_list)
        thumbpath_list = [tup[0] for tup in thumbtup_list]
        utool.remove_file_list(thumbpath_list)

    @deleter
    def delete_chips(ibs, cid_list):
        """ deletes images from the database that belong to gids"""
        if utool.VERBOSE:
            print('[ibs] deleting %d annotation-chips' % len(cid_list))
        # Delete chip-images from disk
        preproc_chip.delete_chips(ibs, cid_list)
        # Delete chip features from sql
        _fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        fid_list = utool.filter_Nones(_fid_list)
        ibs.delete_features(fid_list)
        # Delete chips from sql
        ibs.db.delete_rowids(CHIP_TABLE, cid_list)

    @deleter
    def delete_encounters(ibs, eid_list):
        """ Removes encounters (images are not effected) """
        if utool.VERBOSE:
            print('[ibs] deleting %d encounters' % len(eid_list))
        egrid_list = utool.flatten(ibs.get_encounter_egrids(eid_list))
        ibs.db.delete_rowids(EG_RELATION_TABLE, egrid_list)
        ibs.db.delete_rowids(ENCOUNTER_TABLE, eid_list)

    #
    #
    #----------------
    # --- WRITERS ---
    #----------------

    @default_decorator
    def export_to_wildbook(ibs):
        """ Exports identified chips to wildbook """
        print('[ibs] exporting to wildbook')
        eid_list = ibs.get_valid_eids()
        addr = "http://127.0.0.1:8080/wildbook-4.1.0-RELEASE"
        #addr = "http://tomcat:tomcat123@127.0.0.1:8080/wildbook-5.0.0-EXPERIMENTAL"
        ibs._init_wb(addr)
        wb.export_ibeis_to_wildbook(ibs, eid_list)
        #raise NotImplementedError()
        # compute encounters
        # get encounters by id
        # get ANNOTATIONs by encounter id
        # submit requests to wildbook
        return None

    #
    #
    #--------------
    # --- MODEL ---
    #--------------

    #@default_decorator
    @utool.indent_func('[ibs.compute_encounters]')
    def compute_encounters(ibs):
        """ Clusters images into encounters """
        print('[ibs] Computing and adding encounters.')
        gid_list = ibs.get_valid_gids(require_unixtime=True)
        enctext_list, flat_gids = preproc_encounter.ibeis_compute_encounters(ibs, gid_list)
        print('[ibs] Finished computing, about to add encounter.')
        ibs.set_image_enctext(flat_gids, enctext_list)
        print('[ibs] Finished computing and adding encounters.')

    @default_decorator
    def detect_existence(ibs, gid_list, **kwargs):
        """ Detects the probability of animal existence in each image """
        probexist_list = randomforest.detect_existence(ibs, gid_list, **kwargs)
        # Return for user inspection
        return probexist_list

    @default_decorator
    def detect_random_forest(ibs, gid_list, species, **kwargs):
        """ Runs animal detection in each image """
        # TODO: Return confidence here as well
        print('[ibs] detecting using random forests')
        detect_gen = randomforest.generate_detections(ibs, gid_list, species, **kwargs)
        detected_gid_list, detected_bbox_list, detected_confidence_list, detected_img_confs = [], [], [], []
        ADD_AFTER_THRESHOLD = 1

        def commit_detections(detected_gids, detected_bboxes, detected_confidences, img_confs):
            """ helper to commit detections on the fly """
            if len(detected_gids) == 0:
                return
            notes_list = ['rfdetect' for _ in xrange(len(detected_gid_list))]
            ibs.add_annotations(detected_gids, detected_bboxes,
                              notes_list=notes_list,
                              detect_confidence_list=detected_confidences)

        # Adding new detections on the fly as they are generated
        for count, (gid, bbox, confidence, img_conf) in enumerate(detect_gen):
            detected_gid_list.append(gid)
            detected_bbox_list.append(bbox)
            detected_confidence_list.append(confidence)
            detected_img_confs.append(img_conf)
            # Save detections as we go, then reset lists
            if len(detected_gid_list) >= ADD_AFTER_THRESHOLD:
                commit_detections(detected_gid_list,
                                  detected_bbox_list,
                                  detected_confidence_list,
                                  detected_img_confs)
                detected_gid_list  = []
                detected_bbox_list = []
                detected_confidence_list = []
                detected_img_confs = []
        # Save any leftover detections
        commit_detections(  detected_gid_list,
                            detected_bbox_list,
                            detected_confidence_list,
                            detected_img_confs)
        print('[ibs] finshed detecting')

    @default_decorator
    def get_recognition_database_aids(ibs):
        """ DEPRECATE: returns persistent recognition database annotations """
        daid_list = ibs.get_valid_aids()
        return daid_list

    @default_decorator
    def query_intra_encounter(ibs, qaid_list, **kwargs):
        """ DEPRECATE: _query_chips wrapper """
        daid_list = qaid_list
        qaid2_qres = ibs._query_chips(qaid_list, daid_list, **kwargs)
        return qaid2_qres

    @default_decorator
    def prep_qreq_encounter(ibs, qaid_list):
        """ Puts IBEIS into intra-encounter mode """
        daid_list = qaid_list
        ibs._prep_qreq(qaid_list, daid_list)

    @default_decorator('[querydb]')
    def query_all(ibs, qaid_list, **kwargs):
        """ _query_chips wrapper """
        daid_list = ibs.get_valid_aids()
        qaid2_qres = ibs._query_chips(qaid_list, daid_list, **kwargs)
        return qaid2_qres

    @default_decorator
    def query_encounter(ibs, qaid_list, eid, **kwargs):
        """ _query_chips wrapper """
        daid_list = ibs.get_encounter_aids(eid)  # encounter database chips
        qaid2_qres = ibs._query_chips(qaid_list, daid_list, **kwargs)
        for qres in qaid2_qres.itervalues():
            qres.eid = eid
        return qaid2_qres

    @default_decorator
    def query_exemplars(ibs, qaid_list, **kwargs):
        daid_list = ibs.get_valid_aids(is_exemplar=True)
        assert len(daid_list) > 0, 'there are no exemplars'
        qaid2_qres = ibs._query_chips(qaid_list, daid_list, **kwargs)
        return qaid2_qres

    @default_decorator
    def prep_qreq_db(ibs, qaid_list):
        """ Puts IBEIS into query database mode """
        daid_list = ibs.get_recognition_database_aids()
        ibs._prep_qreq(qaid_list, daid_list)

    @default_decorator
    def _init_query_requestor(ibs):
        # Create query request object
        ibs.qreq = QueryRequest.QueryRequest(ibs.qresdir, ibs.bigcachedir)
        ibs.qreq.set_cfg(ibs.cfg.query_cfg)

    @default_decorator
    def _prep_qreq(ibs, qaid_list, daid_list, **kwargs):
        if ibs.qreq is None:
            ibs._init_query_requestor()
        qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                      qaids=qaid_list,
                                      daids=daid_list,
                                      query_cfg=ibs.cfg.query_cfg,
                                      **kwargs)
        return qreq

    @default_decorator
    def _query_chips(ibs, qaid_list, daid_list, **kwargs):
        """
        qaid_list - query chip ids
        daid_list - database chip ids
        """
        qreq = ibs._prep_qreq(qaid_list, daid_list, **kwargs)
        # TODO: Except query error
        qaid2_qres = mc3.process_query_request(ibs, qreq)
        return qaid2_qres

    #
    #
    #--------------
    # --- MISC ---
    #--------------
    # See ibeis/dev/ibsfuncs.py
    # there is some sneaky stuff happening there
