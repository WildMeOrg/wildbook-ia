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
import six
import atexit
import requests
import uuid
from six.moves import zip, map, range
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
from ibeis import ibsfuncs
# IBEIS MODEL
from ibeis.model import Config
from ibeis.model.preproc import preproc_chip
from ibeis.model.preproc import preproc_image
from ibeis.model.preproc import preproc_feat
from ibeis.model.preproc import preproc_detectimg
from ibeis.model.preproc import preproc_encounter
from ibeis.model.detect import randomforest
from ibeis.model.hots import match_chips4 as mc4
# IBEIS
# from ibeis.control import DB_SCHEMA
from ibeis.control import _sql_helpers
from ibeis.control import SQLDatabaseControl as sqldbc
from ibeis.control.accessor_decors import (adder, setter, getter_1toM,
                                           getter_1to1, ider, deleter,
                                           default_decorator, cache_getter,
                                           cache_invalidator, init_tablecache)
# CONSTANTS
from ibeis.constants import (IMAGE_TABLE, ANNOTATION_TABLE, LBLANNOT_TABLE,
                             ENCOUNTER_TABLE, EG_RELATION_TABLE,
                             AL_RELATION_TABLE, CHIP_TABLE, FEATURE_TABLE,
                             CONFIG_TABLE, LBLTYPE_TABLE, METADATA_TABLE,
                             __STR__)

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibs]', DEBUG=False)

__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers


@atexit.register
def __cleanup():
    """ prevents flann errors (not for cleaning up individual objects) """
    global __ALL_CONTROLLERS__
    try:
        del __ALL_CONTROLLERS__
    except NameError:
        pass


#
#
#-----------------
# IBEIS CONTROLLER
#-----------------

class IBEISController(object):
    """IBEISController docstring
    chip  - cropped region of interest in an image, maps to one animal
    cid   - chip unique id
    gid   - image unique id (could just be the relative file path)
    name  - name unique id
    eid   - encounter unique id
    aid   - region of interest unique id
    annotation   - region of interest for a chip
    theta - angle of rotation for a chip
    """
    # USE THIS IN CONTROLLER CYTHON
    # Available in Python-space:
    #property period:
    #    def __get__(self):
    #        return 1.0 / self.freq
    #    def __set__(self, value):
    #        self.freq = 1.0 / value

    #
    #
    #-------------------------------
    # --- CONSTRUCTOR / PRIVATES ---
    #-------------------------------

    def __init__(ibs, dbdir=None, ensure=True, wbaddr=None, verbose=True):
        """ Creates a new IBEIS Controller associated with one database """
        global __ALL_CONTROLLERS__
        if verbose and utool.VERBOSE:
            print('[ibs.__init__] new IBEISController')
        ibs.table_cache = init_tablecache()
        #ibs.qreq = None  # query requestor object
        ibs.ibschanged_callback = None
        ibs._init_dirs(dbdir=dbdir, ensure=ensure)
        ibs._init_wb(wbaddr)  # this will do nothing if no wildbook address is specified
        ibs._init_sql()
        ibs._init_config()
        ibsfuncs.inject_ibeis(ibs)
        __ALL_CONTROLLERS__.append(ibs)

    def rrr(ibs):
        global __ALL_CONTROLLERS__
        try:
            __ALL_CONTROLLERS__.remove(ibs)
        except ValueError:
            pass
        from ibeis.control import IBEISControl
        IBEISControl.rrr()
        ibsfuncs.rrr()
        print('reloading IBEISControl')
        ibsfuncs.inject_ibeis(ibs)
        utool.reload_class_methods(ibs, IBEISControl.IBEISController)
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
        ibs.db_version_expected = '1.0.1'
        ibs.db = sqldbc.SQLDatabaseController(ibs.get_ibsdir(), ibs.sqldb_fname, text_factory=__STR__)
        _sql_helpers.ensure_correct_version(ibs)
        # ibs.db.dump_schema()
        ibs.UNKNOWN_LBLANNOT_ROWID = 0  # ADD TO CONSTANTS
        ibs.MANUAL_CONFIG_SUFFIX = 'MANUAL_CONFIG'
        ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)
        from ibeis.dev import duct_tape
        # duct_tape.fix_compname_configs(ibs)
        # duct_tape.remove_database_slag(ibs)
        lbltype_names    = constants.KEY_DEFAULTS.keys()
        lbltype_defaults = constants.KEY_DEFAULTS.values()
        lbltype_ids = ibs.add_lbltype(lbltype_names, lbltype_defaults)
        ibs.lbltype_ids = dict(zip(lbltype_names, lbltype_ids))

    @default_decorator
    def clone_handle(ibs, **kwargs):
        ibs2 = IBEISController(dbdir=ibs.get_dbdir(), ensure=False)
        if len(kwargs) > 0:
            ibs2.update_query_cfg(**kwargs)
        #if ibs.qreq is not None:
        #    ibs2._prep_qreq(ibs.qreq.qaids, ibs.qreq.daids)
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

    def get_figanalysis_dir(ibs):
        """ Returns ibs internal directory """
        return join(ibs._ibsdb, 'figure_analysis')

    def get_imgdir(ibs):
        """ Returns ibs internal directory """
        return ibs.imgdir

    def get_thumbdir(ibs):
        """ Returns database directory where thumbnails are cached """
        return ibs.thumb_dpath

    def get_workdir(ibs):
        """ Returns directory where databases are saved to """
        return ibs.workdir

    def get_cachedir(ibs):
        """ Returns database directory of all cached files """
        return ibs.cachedir

    def get_detectimg_cachedir(ibs):
        """ Returns database directory of image resized for detections """
        return join(ibs.cachedir, constants.PATH_NAMES.detectimg)

    def get_flann_cachedir(ibs):
        """ Returns database directory where the FLANN KD-Tree is stored """
        return ibs.flanndir

    def get_qres_cachedir(ibs):
        """ Returns database directory where query results are stored """
        return ibs.qresdir

    def get_big_cachedir(ibs):
        """ Returns database directory where aggregate results are stored """
        return ibs.bigcachedir

    #
    #
    #----------------
    # --- Configs ---
    #----------------

    def _init_config(ibs):
        """ Loads the database's algorithm configuration """
        ibs.cfg = Config.ConfigBase('cfg', fpath=join(ibs.dbdir, 'cfg'))
        try:
            if True or utool.get_flag(('--noprefload', '--noprefload')):
                raise Exception('')
            ibs.cfg.load()
            if utool.NOT_QUIET:
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
        #if ibs.qreq is not None:
        #    ibs.qreq.set_cfg(query_cfg)
        ibs.cfg.query_cfg = query_cfg
        ibs.cfg.feat_cfg  = ibs.cfg.query_cfg._feat_cfg
        ibs.cfg.chip_cfg  = ibs.cfg.query_cfg._feat_cfg._chip_cfg

    @default_decorator
    def update_query_cfg(ibs, **kwargs):
        """ Updates query config only. Configs needs a restructure very badly """
        ibs.cfg.query_cfg.update_query_cfg(**kwargs)
        ibs.cfg.feat_cfg  = ibs.cfg.query_cfg._feat_cfg
        ibs.cfg.chip_cfg  = ibs.cfg.query_cfg._feat_cfg._chip_cfg

    @default_decorator
    def get_chip_config_rowid(ibs):
        """ # FIXME: Configs are still handled poorly """
        chip_cfg_suffix = ibs.cfg.chip_cfg.get_cfgstr()
        chip_cfg_rowid = ibs.add_config(chip_cfg_suffix)
        return chip_cfg_rowid

    @default_decorator
    def get_feat_config_rowid(ibs):
        """ # FIXME: Configs are still handled poorly """
        feat_cfg_suffix = ibs.cfg.feat_cfg.get_cfgstr()
        feat_cfg_rowid = ibs.add_config(feat_cfg_suffix)
        return feat_cfg_rowid

    @default_decorator
    def get_query_config_rowid(ibs):
        """ # FIXME: Configs are still handled poorly """
        query_cfg_suffix = ibs.cfg.query_cfg.get_cfgstr()
        query_cfg_rowid = ibs.add_config(query_cfg_suffix)
        return query_cfg_rowid

    #@default_decorator
    #def get_qreq_rowid(ibs):
    #    """ # FIXME: Configs are still handled poorly """
    #    assert ibs.qres is not None
    #    qreq_rowid = ibs.qreq.get_cfgstr()
    #    return qreq_rowid

    #
    #
    #-------------
    # --- INFO ---
    #-------------

    def get_num_images(ibs, **kwargs):
        """ Number of valid images """
        gid_list = ibs.get_valid_gids(**kwargs)
        return len(gid_list)

    def get_num_annotations(ibs, **kwargs):
        """ Number of valid annotations """
        aid_list = ibs.get_valid_aids(**kwargs)
        return len(aid_list)

    def get_num_names(ibs, **kwargs):
        """ Number of valid name (subset of lblannot) """
        nid_list = ibs.get_valid_nids(**kwargs)
        return len(nid_list)

    #
    #
    #---------------
    # --- IDERS ---
    #---------------

    # Standard

    # Internal

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
    def _get_all_known_lblannot_rowids(ibs, _lbltype):
        """ Returns all nids of known animals
            (does not include unknown names) """
        all_known_lblannot_rowids = ibs.db.get_all_rowids_where(LBLANNOT_TABLE, 'lbltype_rowid=?', (ibs.lbltype_ids[_lbltype],))
        return all_known_lblannot_rowids

    @ider
    def _get_all_lblannot_rowids(ibs):
        all_lblannot_rowids = ibs.db.get_all_rowids(LBLANNOT_TABLE)
        return all_lblannot_rowids

    #@ider
    #def _get_all_alr_rowids(ibs):
    #    all_alr_rowids = ibs.db.get_all_rowids(AL_RELATION_TABLE)
    #    return all_alr_rowids

    @ider
    def _get_all_known_nids(ibs):
        """ Returns all nids of known animals
            (does not include unknown names) """
        all_known_nids = ibs._get_all_known_lblannot_rowids(constants.INDIVIDUAL_KEY)
        return all_known_nids

    @ider
    def _get_all_known_species_rowids(ibs):
        """ Returns all nids of known animals
            (does not include unknown names) """
        all_known_species_rowids = ibs._get_all_known_lblannot_rowids(constants.SPECIES_KEY)
        return all_known_species_rowids

    @ider
    def get_valid_gids(ibs, eid=None, require_unixtime=False, reviewed=None):
        if eid is None:
            gid_list = ibs._get_all_gids()
        else:
            gid_list = ibs.get_encounter_gids(eid)
        if require_unixtime:
            # Remove images without timestamps
            unixtime_list = ibs.get_image_unixtime(gid_list)
            isvalid_list = [unixtime != -1 for unixtime in unixtime_list]
            gid_list = utool.filter_items(gid_list, isvalid_list)
        if reviewed is not None:
            reviewed_list = ibs.get_image_reviewed(gid_list)
            isvalid_list = [reviewed == flag for flag in reviewed_list]
            gid_list = utool.filter_items(gid_list, isvalid_list)
        return gid_list

    @ider
    def get_valid_eids(ibs, min_num_gids=0):
        """ returns list of all encounter ids """
        eid_list = ibs._get_all_eids()
        if min_num_gids > 0:
            num_gids_list = ibs.get_encounter_num_gids(eid_list)
            flag_list = [num_gids >= min_num_gids for num_gids in num_gids_list]
            eid_list  = utool.filter_items(eid_list, flag_list)
        return eid_list

    @ider
    def get_valid_aids(ibs, eid=None, is_exemplar=None):
        """ returns a list of valid ANNOTATION unique ids """
        if eid is None and is_exemplar is not None:
            # Optimization Hack
            aid_list = ibs.db.get_all_rowids_where(ANNOTATION_TABLE, 'annot_exemplar_flag=?', (is_exemplar,))
            return aid_list
        if eid is None:
            aid_list = ibs._get_all_aids()
        else:
            # HACK: Check to see if you want the
            # exemplar "encounter" (image group)
            enctext = ibs.get_encounter_enctext(eid)
            if enctext == constants.EXEMPLAR_ENCTEXT:
                is_exemplar = True
            aid_list = ibs.get_encounter_aids(eid)
        if is_exemplar:
            flag_list = ibs.get_annot_exemplar_flag(aid_list)
            aid_list = utool.filter_items(aid_list, flag_list)
        return aid_list

    @ider
    def get_valid_nids(ibs, eid=None, filter_empty=False):
        """ Returns all valid names with at least one animal
            (does not include unknown names) """
        if eid is None:
            _nid_list = ibs._get_all_known_nids()
        else:
            _nid_list = ibs.get_encounter_nids(eid)
        nRois_list = ibs.get_name_num_annotations(_nid_list)
        if filter_empty:
            nid_list = [nid for nid, nRois in zip(_nid_list, nRois_list)
                        if nRois > 0]
        else:
            nid_list = _nid_list
        return nid_list

    @ider
    def get_invalid_nids(ibs):
        """ Returns all names without any animals (does not include unknown names) """
        _nid_list = ibs._get_all_known_nids()
        nRois_list = ibs.get_name_num_annotations(_nid_list)
        nid_list = [nid for nid, nRois in zip(_nid_list, nRois_list)
                    if nRois <= 0]
        return nid_list

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

    @ider
    def get_valid_configids(ibs):
        config_rowid_list = ibs.db.get_all_rowids(constants.CONFIG_TABLE)
        return config_rowid_list

    #
    #
    #---------------
    # --- ADDERS ---
    #---------------

    @adder
    def add_metadata(ibs, metadata_key_list, metadata_value_list):
        """ Adds a list of names. Returns their nids """
        if utool.VERBOSE:
            print('[ibs] adding %d metadata' % len(metadata_key_list))
        # Add encounter text names to database
        colnames = ['metadata_key', 'metadata_value']
        params_iter = zip(metadata_key_list, metadata_value_list)
        get_rowid_from_superkey = ibs.get_metadata_rowid_from_metadata_key
        metadata_id_list = ibs.db.add_cleanly(METADATA_TABLE, colnames, params_iter, get_rowid_from_superkey)
        return metadata_id_list

    @adder
    def add_images(ibs, gpath_list, as_annots=False):
        """ Adds a list of image paths to the database.  Returns gids
        Initially we set the image_uri to exactely the given gpath.
        Later we change the uri, but keeping it the same here lets
        us process images asychronously.
        >>> from ibeis.all_imports import *  # NOQA  # doctest.SKIP
        >>> gpath_list = grabdata.get_test_gpaths(ndata=7) + ['doesnotexist.jpg']
        """
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))
        #print('[ibs] gpath_list = %r' % (gpath_list,))
        # Processing an image might fail, yeilding a None instead of a tup
        gpath_list = ibsfuncs.ensure_unix_gpaths(gpath_list)
        # Create param_iter
        params_list  = list(preproc_image.add_images_params_gen(gpath_list))
        # Error reporting
        print('\n'.join(
            [' ! Failed reading gpath=%r' % (gpath,) for (gpath, params)
             in zip(gpath_list, params_list) if not params]))
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

        #ibs.cfg.other_cfg.ensure_attr('auto_localize', True)
        if ibs.cfg.other_cfg.auto_localize:
            # Move to ibeis database local cache
            ibs.localize_images(gid_list)

        if as_annots:
            # Add succesfull imports as annotations
            isnone_list = [gid is None for gid in gid_list]
            gid_list_ = utool.filterfalse_items(gid_list, isnone_list)
            aid_list = ibs.use_images_as_annotations(gid_list)
            print('[ibs] added %d annotations' % (len(aid_list),))
        return gid_list

    @adder
    def add_encounters(ibs, enctext_list):
        """ Adds a list of names. Returns their nids """
        if utool.VERBOSE:
            print('[ibs] adding %d encounters' % len(enctext_list))
        # Add encounter text names to database
        notes_list = [''] * len(enctext_list)
        encounter_uuid_list = [uuid.uuid4() for _ in range(len(enctext_list))]
        colnames = ['encounter_text', 'encounter_uuid', 'encounter_note']
        params_iter = zip(enctext_list, encounter_uuid_list, notes_list)
        get_rowid_from_superkey = partial(ibs.get_encounter_eids_from_text, ensure=False)

        eid_list = ibs.db.add_cleanly(ENCOUNTER_TABLE, colnames, params_iter, get_rowid_from_superkey)
        return eid_list

    @adder
    def add_annots(ibs, gid_list, bbox_list=None, theta_list=None,
                        species_list=None, nid_list=None, name_list=None,
                        detect_confidence_list=None, notes_list=None,
                        vert_list=None):
        """ Adds oriented ANNOTATION bounding boxes to images """
        if utool.VERBOSE:
            print('[ibs] adding annotations')
        # Prepare the SQL input
        assert name_list is None or nid_list is None, 'cannot specify both names and nids'
        # xor bbox or vert is None
        assert bool(bbox_list is None) != bool(vert_list is None), 'must specify exactly one of bbox_list or vert_list'

        if theta_list is None:
            theta_list = [0.0 for _ in range(len(gid_list))]
        if name_list is not None:
            nid_list = ibs.add_names(name_list)
        if detect_confidence_list is None:
            detect_confidence_list = [0.0 for _ in range(len(gid_list))]
        if notes_list is None:
            notes_list = ['' for _ in range(len(gid_list))]
        if vert_list is None:
            vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
        elif bbox_list is None:
            bbox_list = geometry.bboxes_from_vert_list(vert_list)

        len_bbox    = len(bbox_list)
        len_vert    = len(vert_list)
        len_gid     = len(gid_list)
        len_notes   = len(notes_list)
        len_theta   = len(theta_list)
        try:
            assert len_vert == len_bbox, 'bbox and verts are not of same size'
            assert len_gid  == len_bbox, 'bbox and gid are not of same size'
            assert len_gid  == len_theta, 'bbox and gid are not of same size'
            assert len_notes == len_gid, 'notes and gids are not of same size'
        except AssertionError as ex:
            utool.printex(ex, key_list=['len_vert', 'len_gid', 'len_bbox'
                                        'len_theta', 'len_notes'])
            raise

        if len(gid_list) == 0:
            # nothing is being added
            print('[ibs] WARNING: 0 annotations are beign added!')
            print(utool.dict_str(locals()))
            return []

        # Build ~~deterministic?~~ random and unique ANNOTATION ids
        image_uuid_list = ibs.get_image_uuids(gid_list)
        #annotation_uuid_list = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list,
        #                                                      theta_list, deterministic=False)
        annotation_uuid_list = [uuid.uuid4() for _ in range(len(image_uuid_list))]
        nVert_list = [len(verts) for verts in vert_list]
        vertstr_list = [__STR__(verts) for verts in vert_list]
        xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
        assert len(nVert_list) == len(vertstr_list)
        # Define arguments to insert
        colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                    'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                    'annot_verts', 'annot_detect_confidence',
                    'annot_note',)

        params_iter = list(zip(annotation_uuid_list, gid_list, xtl_list, ytl_list,
                                width_list, height_list, theta_list, nVert_list,
                                vertstr_list, detect_confidence_list,
                                notes_list))
        #utool.embed()

        # Execute add ANNOTATIONs SQL
        get_rowid_from_superkey = ibs.get_annot_aids_from_uuid
        aid_list = ibs.db.add_cleanly(ANNOTATION_TABLE, colnames, params_iter, get_rowid_from_superkey)

        if species_list is not None:
            species_list = [species.lower() for species in species_list]
            ibs.set_annot_species(aid_list, species_list)

        # Also need to populate annotation_lblannot_relationship table
        if nid_list is not None:
            alrid_list = ibs.add_annot_relationship(aid_list, nid_list)
            del alrid_list
        #print('alrid_list = %r' % (alrid_list,))
        # Invalidate image thumbnails
        ibs.delete_image_thumbs(gid_list)
        return aid_list

    #@adder
    # DEPRICATE
    #def add_annot_names(ibs, aid_list, name_list=None, nid_list=None):
    #    """ Sets names/nids of a list of annotations.
    #    Convenience function for add_annot_relationship"""
    #    assert name_list is None or nid_list is None, (
    #        'can only specify one type of name values (nid or name) not both')
    #    if nid_list is None:
    #        assert name_list is not None
    #        # Convert names into nids
    #        nid_list = ibs.add_names(name_list)
    #    ibs.add_annot_relationship(aid_list, nid_list)

    # Internal

    @adder
    def add_version(ibs, versiontext_list):
        """ Adds an algorithm / actor configuration as a string """
        # FIXME: Configs are still handled poorly
        params_iter = ((versiontext,) for versiontext in versiontext_list)
        get_rowid_from_superkey = ibs.get_version_rowid_from_superkey
        versionid_list = ibs.db.add_cleanly(VERSIONS_TABLE, ('version_text',),
                                            params_iter, get_rowid_from_superkey)
        return versionid_list

    @adder
    def add_config(ibs, cfgsuffix_list):
        """ Adds an algorithm / actor configuration as a string """
        # FIXME: Configs are still handled poorly
        params_iter = ((suffix,) for suffix in cfgsuffix_list)
        get_rowid_from_superkey = partial(ibs.get_config_rowid_from_suffix, ensure=False)
        config_rowid_list = ibs.db.add_cleanly(CONFIG_TABLE, ('config_suffix',),
                                               params_iter, get_rowid_from_superkey)
        return config_rowid_list

    @adder
    def add_chips(ibs, aid_list):
        """
        FIXME: This is a dirty dirty function
        Adds chip data to the ANNOTATION. (does not create ANNOTATIONs. first use add_annots
        and then pass them here to ensure chips are computed) """
        # Ensure must be false, otherwise an infinite loop occurs
        cid_list = ibs.get_annot_cids(aid_list, ensure=False)
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
            get_rowid_from_superkey = partial(ibs.get_annot_cids, ensure=False)
            cid_list = ibs.db.add_cleanly(CHIP_TABLE, colnames, params_iter, get_rowid_from_superkey)

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
            get_rowid_from_superkey = partial(ibs.get_chip_fids, ensure=False)
            fid_list = ibs.db.add_cleanly(FEATURE_TABLE, colnames, params_iter, get_rowid_from_superkey)

        return fid_list

    #
    #
    #----------------
    # --- SETTERS ---
    #----------------

    # SETTERS::METADATA

    @setter
    def set_metadata_value(ibs, metadata_key_list, metadata_value_list):
        """ Sets metadata key, value pairs
        """
        metadata_id_list = ibs.get_metadata_rowid_from_metadata_key(metadata_key_list)
        id_iter = ((metadata_id,) for metadata_id in metadata_id_list)
        val_list = ((metadata_value,) for metadata_value in metadata_value_list)
        ibs.db.set(METADATA_TABLE, ('metadata_value',), val_list, id_iter)

    def set_database_version(ibs, version):
        """ Sets metadata key, value pairs
        """
        ibs.set_metadata_value(['database_version'], [version])

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
    def set_image_reviewed(ibs, gid_list, reviewed_list):
        """ Sets the image all instances found bit """
        id_iter = ((gid,) for gid in gid_list)
        val_list = ((reviewed,) for reviewed in reviewed_list)
        ibs.db.set(IMAGE_TABLE, ('image_toggle_reviewed',), val_list, id_iter)

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
        # FIXME: Slow and weird
        if utool.VERBOSE:
            print('[ibs] setting %r image encounter ids (from text)' % len(gid_list))
        eid_list = ibs.add_encounters(enctext_list)
        ibs.set_image_eids(gid_list, eid_list)

    @setter
    def set_image_eids(ibs, gid_list, eid_list):
        """ Sets the encoutertext of each image """
        if utool.VERBOSE:
            print('[ibs] setting %r image encounter ids' % len(gid_list))
        egrid_list = ibs.add_image_relationship(gid_list, eid_list)
        del egrid_list

    @setter
    def set_image_gps(ibs, gid_list, gps_list=None, lat_list=None, lon_list=None):
        """ see get_image_gps for how the gps_list should look.
            lat and lon should be given in degrees """
        if gps_list is not None:
            assert lat_list is None
            assert lon_list is None
            lat_list = [tup[0] for tup in gps_list]
            lon_list = [tup[1] for tup in gps_list]
        colnames = ('image_gps_lat', 'image_gps_lon',)
        val_list = zip(lat_list, lon_list)
        id_iter = ((gid,) for gid in gid_list)
        ibs.db.set(IMAGE_TABLE, colnames, val_list, id_iter)

    # SETTERS::ANNOTATION

    @setter
    def set_annot_exemplar_flag(ibs, aid_list, flag_list):
        """ Sets if an annotation is an exemplar """
        id_iter = ((aid,) for aid in aid_list)
        val_iter = ((flag,) for flag in flag_list)
        ibs.db.set(ANNOTATION_TABLE, ('annot_exemplar_flag',), val_iter, id_iter)

    @setter
    def set_annot_bboxes(ibs, aid_list, bbox_list, delete_thumbs=True):
        """ Sets bboxes of a list of annotations by aid, where bbox_list is a list of
            (x, y, w, h) tuples
        NOTICE: set_annot_bboxes is a proxy for set_annot_verts
        """
        # changing the bboxes also changes the bounding polygon
        vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
        # naively overwrite the bounding polygon with a rectangle - for now trust the user!
        ibs.set_annot_verts(aid_list, vert_list, delete_thumbs=delete_thumbs)

    @setter
    def set_annot_thetas(ibs, aid_list, theta_list, delete_thumbs=True):
        """ Sets thetas of a list of chips by aid """
        id_iter = ((aid,) for aid in aid_list)
        val_list = ((theta,) for theta in theta_list)
        ibs.db.set(ANNOTATION_TABLE, ('annot_theta',), val_list, id_iter)
        if delete_thumbs:
            ibs.delete_annot_chips(aid_list)  # Changing theta redefines the chips

    @setter
    def set_annot_verts(ibs, aid_list, verts_list, delete_thumbs=True):
        """ Sets the vertices [(x, y), ...] of a list of chips by aid """
        num_params = len(aid_list)
        # Compute data to set
        num_verts_list   = list(map(len, verts_list))
        verts_as_strings = list(map(__STR__, verts_list))
        id_iter1 = ((aid,) for aid in aid_list)
        # also need to set the internal number of vertices
        val_iter1 = ((num_verts, verts) for (num_verts, verts)
                     in zip(num_verts_list, verts_as_strings))
        colnames = ('annot_num_verts', 'annot_verts',)
        # SET VERTS in ANNOTATION_TABLE
        ibs.db.set(ANNOTATION_TABLE, colnames, val_iter1, id_iter1, num_params=num_params)
        # changing the vertices also changes the bounding boxes
        bbox_list = geometry.bboxes_from_vert_list(verts_list)      # new bboxes
        xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
        val_iter2 = zip(xtl_list, ytl_list, width_list, height_list)
        id_iter2 = ((aid,) for aid in aid_list)
        colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
        # SET BBOX in ANNOTATION_TABLE
        ibs.db.set(ANNOTATION_TABLE, colnames, val_iter2, id_iter2, num_params=num_params)
        if delete_thumbs:
            ibs.delete_annot_chips(aid_list)  # INVALIDATE THUMBNAILS

    @setter
    def set_annot_notes(ibs, aid_list, notes_list):
        """ Sets annotation notes """
        id_iter = ((aid,) for aid in aid_list)
        val_iter = ((notes,) for notes in notes_list)
        ibs.db.set(ANNOTATION_TABLE, ('annot_note',), val_iter, id_iter)

    # SETTERS::ENCOUNTER

    @setter
    def set_encounter_enctext(ibs, eid_list, names_list):
        """ Sets names of encounters (groups of animals) """
        id_iter = ((eid,) for eid in eid_list)
        val_list = ((names,) for names in names_list)
        ibs.db.set(ENCOUNTER_TABLE, ('encounter_text',), val_list, id_iter)

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    @getter_1to1
    def get_metadata_value(ibs, metadata_key_list):
        params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
        where_clause = 'metadata_key=?'
        # list of relationships for each image
        metadata_value_list = ibs.db.get_where(METADATA_TABLE, ('metadata_value',), params_iter, where_clause, unpack_scalars=True)
        return metadata_value_list
        
    @getter_1to1
    def get_metadata_rowid_from_metadata_key(ibs, metadata_key_list):
        params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
        where_clause = 'metadata_key=?'
        # list of relationships for each image
        metadata_rowid_list = ibs.db.get_where(METADATA_TABLE, ('metadata_rowid',), params_iter, where_clause, unpack_scalars=True)
        return metadata_rowid_list

    @ider
    def get_database_version(ibs):
        version_list = ibs.get_metadata_value(['database_version'])
        version = version_list[0]
        if version is None:
            version = constants.BASE_DATABASE_VERSION
            ibs.add_metadata(['database_version'], [version])
        else:
            version = version_list[0]
        return version

    #
    # GETTERS::IMAGE_TABLE

    @getter_1to1
    def get_images(ibs, gid_list):
        """ Returns a list of images in numpy matrix form by gid """
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    @getter_1to1
    def get_image_thumbtup(ibs, gid_list, thumbsize=128):
        """ Returns tuple of image paths, thumb paths, bboxes and thetas """
        # print('gid_list = %r' % (gid_list,))
        aids_list = ibs.get_image_aids(gid_list)
        bboxes_list = ibsfuncs.unflat_map(ibs.get_annot_bboxes, aids_list)
        thetas_list = ibsfuncs.unflat_map(ibs.get_annot_thetas, aids_list)
        thumb_gpaths = ibs.get_image_thumbpath(gid_list, thumbsize=128)
        image_paths = ibs.get_image_paths(gid_list)
        gsize_list = ibs.get_image_sizes(gid_list)
        thumbtup_list = list(zip(thumb_gpaths, image_paths, gsize_list, bboxes_list, thetas_list))
        return thumbtup_list

    @getter_1to1
    def get_image_thumbpath(ibs, gid_list, thumbsize=128):
        """ Returns the thumbnail path of each gid """
        thumb_dpath = ibs.thumb_dpath
        img_uuid_list = ibs.get_image_uuids(gid_list)
        thumb_suffix = '_' + str(thumbsize) + constants.IMAGE_THUMB_SUFFIX
        thumbpath_list = [join(thumb_dpath, __STR__(uuid) + thumb_suffix)
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
        """ Returns a list of image uris relative to the image dir by gid """
        uri_list = ibs.db.get(IMAGE_TABLE, ('image_uri',), gid_list)
        return uri_list

    @getter_1to1
    def get_image_gids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        gid_list = ibs.db.get(IMAGE_TABLE, ('image_rowid',), uuid_list, id_colname='image_uuid')
        return gid_list

    get_image_rowid_from_uuid = get_image_gids_from_uuid

    @getter_1to1
    def get_image_paths(ibs, gid_list):
        """ Returns a list of image absolute paths to img_dir """
        utool.assert_all_not_None(gid_list, 'gid_list', key_list=['gid_list'])
        uri_list = ibs.get_image_uris(gid_list)
        # Images should never have null uris
        utool.assert_all_not_None(uri_list, 'uri_list', key_list=['uri_list', 'gid_list'])
        gpath_list = [join(ibs.imgdir, uri) for uri in uri_list]
        return gpath_list

    # TODO make this actually return a uri format
    get_image_absolute_uri = get_image_paths

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
    def get_image_reviewed(ibs, gid_list):
        """ Returns "All Instances Found" flag, true if all objects of interest
        (animals) have an ANNOTATION in the image """
        reviewed_list = ibs.db.get(IMAGE_TABLE, ('image_toggle_reviewed',), gid_list)
        return reviewed_list

    @getter_1to1
    def get_image_detect_confidence(ibs, gid_list):
        """ Returns image detection confidence as the max of ANNOTATION confidences """
        aids_list = ibs.get_image_aids(gid_list)
        confs_list = ibsfuncs.unflat_map(ibs.get_annot_detect_confidence, aids_list)
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
        nids_list = ibs.get_annot_nids(aids_list)
        return nids_list

    @getter_1toM
    def get_image_eids(ibs, gid_list):
        """ Returns a list of encounter ids for each image by gid """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        colnames = ('encounter_rowid',)
        eids_list = ibs.db.get(EG_RELATION_TABLE, colnames, gid_list,
                               id_colname='image_rowid', unpack_scalars=False)
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
        # print('gid_list = %r' % (gid_list,))
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        colnames = ('annot_rowid',)
        aids_list = ibs.db.get(ANNOTATION_TABLE, colnames, gid_list,
                               id_colname='image_rowid', unpack_scalars=False)
        #print('aids_list = %r' % (aids_list,))
        return aids_list

    @getter_1to1
    def get_image_num_annotations(ibs, gid_list):
        """ Returns the number of chips in each image """
        return list(map(len, ibs.get_image_aids(gid_list)))

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
    def get_annot_exemplar_flag(ibs, aid_list):
        annotation_uuid_list = ibs.db.get(ANNOTATION_TABLE, ('annot_exemplar_flag',), aid_list)
        return annotation_uuid_list

    @getter_1to1
    def get_annot_uuids(ibs, aid_list):
        """ Returns a list of image uuids by gid """
        annotation_uuid_list = ibs.db.get(ANNOTATION_TABLE, ('annot_uuid',), aid_list)
        return annotation_uuid_list

    @getter_1to1
    def get_annot_aids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        aids_list = ibs.db.get(ANNOTATION_TABLE, ('annot_rowid',), uuid_list, id_colname='annot_uuid')
        return aids_list

    get_annot_rowid_from_uuid = get_annot_aids_from_uuid

    @getter_1to1
    def get_annot_detect_confidence(ibs, aid_list):
        """ Returns a list confidences that the annotations is a valid detection """
        annotation_detect_confidence_list = ibs.db.get(ANNOTATION_TABLE, ('annot_detect_confidence',), aid_list)
        return annotation_detect_confidence_list

    @getter_1to1
    def get_annot_notes(ibs, aid_list):
        """ Returns a list of annotation notes """
        annotation_notes_list = ibs.db.get(ANNOTATION_TABLE, ('annot_note',), aid_list)
        return annotation_notes_list

    @utool.accepts_numpy
    @getter_1toM
    def get_annot_bboxes(ibs, aid_list):
        """ returns annotation bounding boxes in image space """
        colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
        bbox_list = ibs.db.get(ANNOTATION_TABLE, colnames, aid_list)
        return bbox_list

    @getter_1to1
    def get_annot_thetas(ibs, aid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.db.get(ANNOTATION_TABLE, ('annot_theta',), aid_list)
        return theta_list

    @getter_1to1
    def get_annot_num_verts(ibs, aid_list):
        """ Returns the number of vertices that form the polygon of each chip """
        num_verts_list = ibs.db.get(ANNOTATION_TABLE, ('annot_num_verts',), aid_list)
        return num_verts_list

    @getter_1to1
    def get_annot_verts(ibs, aid_list):
        """ Returns the vertices that form the polygon of each chip """
        vertstr_list = ibs.db.get(ANNOTATION_TABLE, ('annot_verts',), aid_list)
        # TODO: Sanatize input for eval
        #print('vertstr_list = %r' % (vertstr_list,))
        vert_list = [eval(vertstr) for vertstr in vertstr_list]
        return vert_list

    @utool.accepts_numpy
    @getter_1to1
    @cache_getter(ANNOTATION_TABLE, 'image_rowid')
    def get_annot_gids(ibs, aid_list):
        """ returns annotation bounding boxes in image space """
        gid_list = ibs.db.get(ANNOTATION_TABLE, ('image_rowid',), aid_list)
        return gid_list

    @getter_1to1
    def get_annot_images(ibs, aid_list):
        """ Returns the images of each annotation """
        gid_list = ibs.get_annot_gids(aid_list)
        image_list = ibs.get_images(gid_list)
        return image_list

    @getter_1to1
    def get_annot_image_uuids(ibs, aid_list):
        gid_list = ibs.get_annot_gids(aid_list)
        image_uuid_list = ibs.get_image_uuids(gid_list)
        return image_uuid_list

    @getter_1to1
    def get_annot_gnames(ibs, aid_list):
        """ Returns the image names of each annotation """
        gid_list = ibs.get_annot_gids(aid_list)
        gname_list = ibs.get_image_gnames(gid_list)
        return gname_list

    @getter_1to1
    def get_annot_gpaths(ibs, aid_list):
        """ Returns the image names of each annotation """
        gid_list = ibs.get_annot_gids(aid_list)
        try:
            utool.assert_all_not_None(gid_list, 'gid_list')
        except AssertionError:
            print('[!get_annot_gpaths] ' + utool.list_dbgstr('aid_list'))
            print('[!get_annot_gpaths] ' + utool.list_dbgstr('gid_list'))
            raise
        gpath_list = ibs.get_image_paths(gid_list)
        utool.assert_all_not_None(gpath_list, 'gpath_list')
        return gpath_list

    @getter_1to1
    def get_annot_cids(ibs, aid_list, ensure=True, all_configs=False):
        # FIXME:
        if ensure:
            try:
                ibs.add_chips(aid_list)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.get_annot_cids]')
                print('[!ibs.get_annot_cids] aid_list = %r' % (aid_list,))
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
    def get_annot_chips(ibs, aid_list, ensure=True):
        utool.assert_all_not_None(aid_list, 'aid_list')
        cid_list = ibs.get_annot_cids(aid_list, ensure=ensure)
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
    def get_annot_chip_thumbtup(ibs, aid_list, thumbsize=128):
        # HACK TO MAKE CHIPS COMPUTE
        #cid_list = ibs.get_annot_cids(aid_list, ensure=True)  # NOQA
        thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=128)
        chip_paths = ibs.get_annot_cpaths(aid_list)
        chipsize_list = ibs.get_annot_chipsizes(aid_list)
        thumbtup_list = [(thumb_path, chip_path, chipsize, [], [])
                         for (thumb_path, chip_path, chipsize) in
                         zip(thumb_gpaths, chip_paths, chipsize_list,)]
        return thumbtup_list

    @getter_1to1
    def get_annot_chip_thumbpath(ibs, aid_list, thumbsize=128):
        thumb_dpath = ibs.thumb_dpath
        thumb_suffix = '_' + str(thumbsize) + constants.CHIP_THUMB_SUFFIX
        annotation_uuid_list = ibs.get_annot_uuids(aid_list)
        thumbpath_list = [join(thumb_dpath, __STR__(uuid) + thumb_suffix)
                          for uuid in annotation_uuid_list]
        return thumbpath_list

    @getter_1to1
    @cache_getter(ANNOTATION_TABLE, 'chipsizes')
    def get_annot_chipsizes(ibs, aid_list, ensure=True):
        """ Returns the imagesizes of computed annotation chips """
        cid_list  = ibs.get_annot_cids(aid_list, ensure=ensure)
        chipsz_list = ibs.get_chip_sizes(cid_list)
        return chipsz_list

    @getter_1to1
    def get_annot_cpaths(ibs, aid_list):
        """ Returns cpaths defined by ANNOTATIONs """
        #utool.assert_all_not_None(aid_list, 'aid_list')
        #assert all([aid is not None for aid in aid_list])
        cfpath_list = preproc_chip.get_annot_cfpath_list(ibs, aid_list)
        return cfpath_list

    @getter_1to1
    def get_annot_fids(ibs, aid_list, ensure=False):
        cid_list = ibs.get_annot_cids(aid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        return fid_list

    @utool.accepts_numpy
    @getter_1toM
    @cache_getter(ANNOTATION_TABLE, 'kpts')
    def get_annot_kpts(ibs, aid_list, ensure=True):
        """ Returns chip keypoints """
        fid_list  = ibs.get_annot_fids(aid_list, ensure=ensure)
        kpts_list = ibs.get_feat_kpts(fid_list)
        return kpts_list

    @getter_1toM
    def get_annot_desc(ibs, aid_list, ensure=True):
        """ Returns chip descriptors """
        fid_list  = ibs.get_annot_fids(aid_list, ensure=ensure)
        desc_list = ibs.get_feat_desc(fid_list)
        return desc_list

    @getter_1to1
    def get_annot_num_feats(ibs, aid_list, ensure=False):
        cid_list = ibs.get_annot_cids(aid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        nFeats_list = ibs.get_num_feats(fid_list)
        return nFeats_list

    @getter_1toM
    def get_annot_groundfalse(ibs, aid_list, is_exemplar=None, valid_aids=None,
                              filter_unknowns=True):
        """ Returns a list of aids which are known to be different for each
        input aid """
        if valid_aids is None:
            # get all valid aids if not specified
            valid_aids = ibs.get_valid_aids(is_exemplar=is_exemplar)
        if filter_unknowns:
            # Remove aids which do not have a name
            isunknown_list = ibs.is_aid_unknown(valid_aids)
            valid_aids_ = utool.filterfalse_items(valid_aids, isunknown_list)
        else:
            valid_aids_ = valid_aids
        # Build the set of groundfalse annotations
        nid_list = ibs.get_annot_nids(aid_list)
        aids_list = ibs.get_name_aids(nid_list)
        aids_setlist = map(set, aids_list)
        valid_aids = set(valid_aids_)
        groundfalse_list = [list(valid_aids - aids) for aids in aids_setlist]
        return groundfalse_list

    @getter_1toM
    def get_annot_groundtruth(ibs, aid_list, is_exemplar=None, noself=True):
        """ Returns a list of aids with the same name foreach aid in aid_list.
        a set of aids belonging to the same name is called a groundtruth. A list
        of these is called a groundtruth_list. """
        # TODO: Optimize
        nid_list = ibs.get_annot_nids(aid_list)
        aids_list = ibs.get_name_aids(nid_list)
        if is_exemplar is None:
            groundtruth_list_ = aids_list
        else:
            # Filter out non-exemplars
            exemplar_flags_list = ibsfuncs.unflat_map(ibs.get_annot_exemplar_flag, aids_list)
            isvalids_list = [[flag == is_exemplar for flag in flags] for flags in exemplar_flags_list]
            groundtruth_list_ = [utool.filter_items(aids, isvalids)
                                 for aids, isvalids in zip(aids_list, isvalids_list)]
        if noself:
            # Remove yourself from the set
            groundtruth_list = [list(set(aids) - {aid})
                                for aids, aid in zip(groundtruth_list_, aid_list)]
        else:
            groundtruth_list = groundtruth_list_
        return groundtruth_list

    @getter_1to1
    def get_annot_num_groundtruth(ibs, aid_list, noself=True):
        """ Returns number of other chips with the same name """
        # TODO: Optimize
        return list(map(len, ibs.get_annot_groundtruth(aid_list, noself=noself)))

    @getter_1to1
    def get_annot_has_groundtruth(ibs, aid_list):
        # TODO: Optimize
        numgts_list = ibs.get_annot_num_groundtruth(aid_list)
        has_gt_list = [num_gts > 0 for num_gts in numgts_list]
        return has_gt_list

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
    #@cache_getter('CHIP_TABLE', 'chip_size')
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
        config_rowid_list = ibs.db.get(CHIP_TABLE, ('config_rowid',), cid_list)
        return config_rowid_list

    #
    # GETTERS::FEATURE_TABLE

    @getter_1toM
    #@cache_getter(FEATURE_TABLE, 'feature_keypoints')
    def get_feat_kpts(ibs, fid_list):
        """ Returns chip keypoints in [x, y, iv11, iv21, iv22, ori] format """
        kpts_list = ibs.db.get(FEATURE_TABLE, ('feature_keypoints',), fid_list)
        return kpts_list

    @getter_1toM
    #@cache_getter(FEATURE_TABLE, 'feature_sifts')
    def get_feat_desc(ibs, fid_list):
        """ Returns chip SIFT descriptors """
        desc_list = ibs.db.get(FEATURE_TABLE, ('feature_sifts',), fid_list)
        return desc_list

    @getter_1to1
    #@cache_getter(FEATURE_TABLE, 'feature_num_feats')
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
        config_rowid_list = ibs.db.get(CONFIG_TABLE, ('config_rowid',), cfgsuffix_list, id_colname='config_suffix')

        # executeone always returns a list
        #if config_rowid_list is not None and len(config_rowid_list) == 1:
        #    config_rowid_list = config_rowid_list[0]
        return config_rowid_list

    @getter_1to1
    def get_config_suffixes(ibs, config_rowid_list):
        """ Gets suffixes for algorithm configs """
        cfgsuffix_list = ibs.db.get(CONFIG_TABLE, ('config_suffix',), config_rowid_list)
        return cfgsuffix_list

    #
    # GETTERS::ENCOUNTER
    @getter_1to1
    def get_encounter_num_gids(ibs, eid_list):
        """ Returns number of images in each encounter """
        return list(map(len, ibs.get_encounter_gids(eid_list)))

    @getter_1toM
    def get_encounter_aids(ibs, eid_list):
        """ returns a list of list of aids in each encounter """
        gids_list = ibs.get_encounter_gids(eid_list)
        aids_list_ = ibsfuncs.unflat_map(ibs.get_image_aids, gids_list)
        aids_list = list(map(utool.flatten, aids_list_))
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

    # @getter_1to1
    def get_encounter_egrids(ibs, eid_list=None, gid_list=None):
        # WEIRD FUNCTION FIXME
        assert eid_list is not None or gid_list is not None, "Either eid_list or gid_list must be None"
        """ Gets a list of encounter-image-relationship rowids for each encouterid """
        if eid_list is not None and gid_list is None:
            # TODO: Group type
            params_iter = ((eid,) for eid in eid_list)
            where_clause = 'encounter_rowid=?'
            # list of relationships for each encounter
            egrids_list = ibs.db.get_where(EG_RELATION_TABLE, ('egr_rowid',),
                                           params_iter, where_clause, unpack_scalars=False)
        elif gid_list is not None and eid_list is None:
            # TODO: Group type
            params_iter = ((gid,) for gid in gid_list)
            where_clause = 'image_rowid=?'
            # list of relationships for each encounter
            egrids_list = ibs.db.get_where(EG_RELATION_TABLE, ('egr_rowid',),
                                           params_iter, where_clause, unpack_scalars=False)
        else:
            # TODO: Group type
            params_iter = ((eid, gid,) for eid, gid in zip(eid_list, gid_list))
            where_clause = 'encounter_rowid=? AND image_rowid=?'
            # list of relationships for each encounter
            egrids_list = ibs.db.get_where(EG_RELATION_TABLE, ('egr_rowid',),
                                           params_iter, where_clause, unpack_scalars=False)
        return egrids_list

    @getter_1toM
    def get_encounter_nids(ibs, eid_list):
        """ returns a list of list of nids in each encounter """
        aids_list = ibs.get_encounter_aids(eid_list)
        nids_list = ibsfuncs.unflat_map(ibs.get_annot_lblannot_rowids_oftype, aids_list,
                                        _lbltype=constants.INDIVIDUAL_KEY)
        nids_list_ = [[nid[0] for nid in nids if len(nid) > 0] for nids in nids_list]

        nids_list = list(map(utool.unique_ordered, nids_list_))
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
        #enctext_list = list(map(__STR__, enctext_list))
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
    @cache_invalidator(ANNOTATION_TABLE)
    def delete_annots(ibs, aid_list):
        """ deletes annotations from the database """
        if utool.VERBOSE:
            print('[ibs] deleting %d annotations' % len(aid_list))
        # Delete chips and features first
        ibs.delete_annot_chips(aid_list)
        ibs.db.delete_rowids(ANNOTATION_TABLE, aid_list)
        ibs.delete_annot_relations(aid_list)

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes images from the database that belong to gids"""
        if utool.VERBOSE:
            print('[ibs] deleting %d images' % len(gid_list))
        # TODO: Move localized images to a trash folder
        # Delete annotations first
        aid_list = utool.flatten(ibs.get_image_aids(gid_list))
        ibs.delete_annots(aid_list)
        ibs.db.delete_rowids(IMAGE_TABLE, gid_list)
        #egrid_list = utool.flatten(ibs.get_image_egrids(gid_list))
        #ibs.db.delete_rowids(EG_RELATION_TABLE, egrid_list)
        ibs.db.delete(EG_RELATION_TABLE, gid_list, id_colname='image_rowid')

    @deleter
    @cache_invalidator(FEATURE_TABLE)
    def delete_features(ibs, fid_list):
        """ deletes images from the database that belong to fids"""
        if utool.VERBOSE:
            print('[ibs] deleting %d features' % len(fid_list))
        ibs.db.delete_rowids(FEATURE_TABLE, fid_list)

    @deleter
    def delete_annot_chips(ibs, aid_list):
        """ Clears annotation data but does not remove the annotation """
        _cid_list = ibs.get_annot_cids(aid_list, ensure=False)
        cid_list = utool.filter_Nones(_cid_list)
        ibs.delete_chips(cid_list)
        gid_list = ibs.get_annot_gids(aid_list)
        ibs.delete_image_thumbs(gid_list)
        ibs.delete_annot_chip_thumbs(aid_list)

    @deleter
    def delete_image_thumbs(ibs, gid_list):
        """ Removes image thumbnails from disk """
        # print('gid_list = %r' % (gid_list,))
        thumbpath_list = ibs.get_image_thumbpath(gid_list)
        utool.remove_file_list(thumbpath_list)

    @deleter
    def delete_annot_chip_thumbs(ibs, aid_list):
        """ Removes chip thumbnails from disk """
        thumbpath_list = ibs.get_annot_chip_thumbpath(aid_list)
        utool.remove_file_list(thumbpath_list)

    @deleter
    @cache_invalidator(CHIP_TABLE)
    def delete_chips(ibs, cid_list, verbose=utool.VERBOSE):
        """ deletes images from the database that belong to gids"""
        if verbose:
            print('[ibs] deleting %d annotation-chips' % len(cid_list))
        # Delete chip-images from disk
        preproc_chip.delete_chips(ibs, cid_list, verbose=verbose)
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
        ibs.db.delete_rowids(ENCOUNTER_TABLE, eid_list)
        # Optimization hack, less SQL calls
        #egrid_list = utool.flatten(ibs.get_encounter_egrids(eid_list=eid_list))
        #ibs.db.delete_rowids(EG_RELATION_TABLE, egrid_list)
        #ibs.db.delete(EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')
        ibs.unrelate_encounter_from_images(eid_list)

    @deleter
    def unrelate_encounter_from_images(ibs, eid_list):
        """ Removes relationship between input encounters and all images """
        ibs.db.delete(EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')

    @deleter
    def unrelate_image_from_encounter(ibs, gid_list):
        """ Removes relationship between input images and all encounters """
        ibs.db.delete(EG_RELATION_TABLE, gid_list, id_colname='image_rowid')

    @deleter
    def delete_image_eids(ibs, gid_list, eid_list):
        # WHAT IS THIS FUNCTION? FIXME CALLS WEIRD FUNCTION
        """ Sets the encoutertext of each image """
        if utool.VERBOSE:
            print('[ibs] deleting %r image\'s encounter ids' % len(gid_list))
        egrid_list = utool.flatten(ibs.get_encounter_egrids(eid_list=eid_list, gid_list=gid_list))
        ibs.db.delete_rowids(EG_RELATION_TABLE, egrid_list)

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
    #-----------------------------
    # --- ENCOUNTER CLUSTERING ---
    #-----------------------------

    #@default_decorator
    @utool.indent_func('[ibs.compute_encounters]')
    def compute_encounters(ibs):
        """ Clusters images into encounters """
        print('[ibs] Computing and adding encounters.')
        gid_list = ibs.get_valid_gids(require_unixtime=False, reviewed=False)
        enctext_list, flat_gids = preproc_encounter.ibeis_compute_encounters(ibs, gid_list)
        print('[ibs] Finished computing, about to add encounter.')
        ibs.set_image_enctext(flat_gids, enctext_list)
        print('[ibs] Finished computing and adding encounters.')

    #
    #
    #------------------
    # --- DETECTION ---
    #------------------

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
        ibs.cfg.other_cfg.ensure_attr('detect_add_after', 1)
        ADD_AFTER_THRESHOLD = ibs.cfg.other_cfg.detect_add_after

        def commit_detections(detected_gids, detected_bboxes, detected_confidences, img_confs):
            """ helper to commit detections on the fly """
            if len(detected_gids) == 0:
                return
            notes_list = ['rfdetect' for _ in range(len(detected_gid_list))]
            # Ideally, species will come from the detector with confidences that actually mean something
            species_list = [ibs.cfg.detect_cfg.species] * len(notes_list)
            ibs.add_annots(detected_gids, detected_bboxes,
                                notes_list=notes_list,
                                species_list=species_list,
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

    #
    #
    #-----------------------
    # --- IDENTIFICATION ---
    #-----------------------

    @default_decorator
    def get_recognition_database_aids(ibs):
        """ DEPRECATE: returns persistent recognition database annotations """
        # TODO: Depricate, use exemplars instead
        daid_list = ibs.get_valid_aids()
        return daid_list

    #@default_decorator
    #def _init_query_requestor(ibs):
    #from ibeis.model.hots import match_chips3 as mc3
    #from ibeis.model.hots import hots_query_request
    #    # DEPRICATE
    #    # Create query request object
    #    ibs.qreq = hots_query_request.QueryRequest(ibs.qresdir, ibs.bigcachedir)
    #    ibs.qreq.set_cfg(ibs.cfg.query_cfg)

    #@default_decorator
    #def _prep_qreq(ibs, qaid_list, daid_list, **kwargs):
    #    # DEPRICATE
    #    if ibs.qreq is None:
    #        ibs._init_query_requestor()
    #    qreq = mc3.prep_query_request(qreq=ibs.qreq,
    #                                  qaids=qaid_list,
    #                                  daids=daid_list,
    #                                  query_cfg=ibs.cfg.query_cfg,
    #                                  **kwargs)
    #    return qreq

    #@default_decorator
    #def _query_chips3(ibs, qaid_list, daid_list, safe=True,
    #                  use_cache=mc3.USE_CACHE,
    #                  use_bigcache=mc3.USE_BIGCACHE,
    #                  **kwargs):
    #    # DEPRICATE
    #    """
    #    qaid_list - query chip ids
    #    daid_list - database chip ids
    #    kwargs modify query_cfg
    #    """
    #    qreq = ibs._prep_qreq(qaid_list, daid_list, **kwargs)
    #    # TODO: Except query error
    #    # NOTE: maybe kwargs should not be passed here, or the previous
    #    # kwargs should become querycfgkw
    #    process_qreqkw = {
    #        'safe'         : safe,
    #        'use_cache'    : use_cache,
    #        'use_bigcache' : use_bigcache,
    #    }
    #    qaid2_qres = mc3.process_query_request(ibs, qreq, **process_qreqkw)
    #    return qaid2_qres

    def _query_chips4(ibs, qaid_list, daid_list, use_cache=mc4.USE_CACHE,
                      use_bigcache=mc4.USE_BIGCACHE):
        """
        >>> from ibeis.all_imports import *  # NOQA
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> mc3.USE_CACHE = False
        >>> mc4.USE_CACHE = False
        >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
        >>> qres1 = ibs._query_chips3(qaid_list, daid_list, use_cache=False)[1]
        >>> qres2 = ibs._query_chips4(qaid_list, daid_list, use_cache=False)[1]
        >>> qreq_ = mc4.get_ibeis_query_request(ibs, qaid_list, daid_list)
        >>> qreq_.load_indexer(ibs)
        >>> qreq_.load_query_vectors(ibs)
        >>> qreq = ibs.qreq

        """
        qaid2_qres = mc4.submit_query_request(ibs,  qaid_list, daid_list,
                                              use_cache, use_bigcache)
        return qaid2_qres

    #_query_chips = _query_chips3
    _query_chips = _query_chips4

    @default_decorator
    def query_encounter(ibs, qaid_list, eid, **kwargs):
        """ _query_chips wrapper """
        daid_list = ibs.get_encounter_aids(eid)  # encounter database chips
        qaid2_qres = ibs._query_chips(qaid_list, daid_list, **kwargs)
        for qres in six.itervalues(qaid2_qres):
            qres.eid = eid
        return qaid2_qres

    @default_decorator
    def query_exemplars(ibs, qaid_list, **kwargs):
        """ Queries vs the exemplars """
        daid_list = ibs.get_valid_aids(is_exemplar=True)
        assert len(daid_list) > 0, 'there are no exemplars'
        qaid2_qres = ibs._query_chips(qaid_list, daid_list, **kwargs)
        return qaid2_qres

    #
    #
    #--------------
    # --- MISC ---
    #--------------
    # See ibeis/ibsfuncs.py
    # there is some sneaky stuff happening there

    # Hacky code for rosemary
    # DO NOT USE ANYWHERE ELSE IN CONTROLLER

    #--------------
    # IN QUESTION
    #--------------

    # SETTERS::ALR

    @setter
    def set_alr_lblannot_rowids(ibs, alrid_list, lblannot_rowid_list):
        """ Associates whatever annotation is at row(alrid) with a new
        lblannot_rowid. (effectively changes the label value of the rowid)
        """
        id_iter = ((alrid,) for alrid in alrid_list)
        val_iter = ((lblannot_rowid,) for lblannot_rowid in lblannot_rowid_list)
        colnames = ('lblannot_rowid',)
        ibs.db.set(AL_RELATION_TABLE, colnames, val_iter, id_iter)

    @setter
    def set_alr_confidence(ibs, alrid_list, confidence_list):
        """ sets annotation-lblannot-relationship confidence """
        id_iter = ((alrid,) for alrid in alrid_list)
        val_iter = ((confidence,) for confidence in confidence_list)
        colnames = ('alr_confidence',)
        ibs.db.set(AL_RELATION_TABLE, colnames, val_iter, id_iter)

    # ADDERS::ALR

    @adder
    def add_annot_relationship(ibs, aid_list, lblannot_rowid_list, config_rowid_list=None,
                                    alr_confidence_list=None):
        """ Adds a relationship between annots and lblannots
            (annotations and labels of annotations) """
        if config_rowid_list is None:
            config_rowid_list = [ibs.MANUAL_CONFIGID] * len(aid_list)
        if alr_confidence_list is None:
            alr_confidence_list = [0.0] * len(aid_list)
        colnames = ('annot_rowid', 'lblannot_rowid', 'config_rowid', 'alr_confidence',)
        params_iter = list(zip(aid_list, lblannot_rowid_list, config_rowid_list, alr_confidence_list))
        get_rowid_from_superkey = ibs.get_alrid_from_superkey
        superkey_paramx = (0, 1, 2)  # TODO HAVE SQL GIVE YOU THESE NUMBERS
        alrid_list = ibs.db.add_cleanly(AL_RELATION_TABLE, colnames, params_iter,
                                        get_rowid_from_superkey, superkey_paramx)
        return alrid_list

    @getter_1to1
    def get_alrid_from_superkey(ibs, aid_list, lblannot_rowid_list, config_rowid_list):
        """
        Input: lblannotid_list (label id) + config_rowid_list
        Output: annot-label relationship id list
        """
        colnames = ('annot_rowid',)
        params_iter = zip(aid_list, lblannot_rowid_list, config_rowid_list)
        where_clause = 'annot_rowid=? AND lblannot_rowid=? AND config_rowid=?'
        alrid_list = ibs.db.get_where(AL_RELATION_TABLE, colnames, params_iter, where_clause)
        return alrid_list

    #
    # GETTERS::ALR

    @getter_1to1
    def get_alr_confidence(ibs, alrid_list):
        """ returns confidence in an annotation relationship """
        alr_confidence_list = ibs.db.get(AL_RELATION_TABLE, ('alr_confidence',), alrid_list)
        return alr_confidence_list

    @getter_1to1
    def get_alr_lblannot_rowids(ibs, alrid_list):
        """ get the lblannot_rowid belonging to each relationship """
        lblannot_rowids_list = ibs.db.get(AL_RELATION_TABLE, ('lblannot_rowid',), alrid_list)
        return lblannot_rowids_list

    @getter_1to1
    def get_alr_annot_rowids(ibs, alrid_list):
        """ get the annot_rowid belonging to each relationship """
        annot_rowids_list = ibs.db.get(AL_RELATION_TABLE, ('annot_rowid',), alrid_list)
        return annot_rowids_list

    # ADDERS::IMAGE->ENCOUNTER

    @adder
    def add_image_relationship(ibs, gid_list, eid_list):
        """ Adds a relationship between an image and and encounter """
        colnames = ('image_rowid', 'encounter_rowid',)
        params_iter = list(zip(gid_list, eid_list))
        get_rowid_from_superkey = ibs.get_egr_rowid_from_superkey
        superkey_paramx = (0, 1)
        egrid_list = ibs.db.add_cleanly(EG_RELATION_TABLE, colnames, params_iter,
                                        get_rowid_from_superkey, superkey_paramx)
        return egrid_list

    # SETTERS::LBLANNOT(NAME)

    @setter
    def set_name_notes(ibs, nid_list, notes_list):
        """ Sets notes of names (groups of animals) """
        ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        ibs.set_lblannot_notes(nid_list, notes_list)

    @setter
    def set_name_names(ibs, nid_list, name_list):
        """ Changes the name text. Does not affect the animals of this name.
        Effectively an alias.
        """
        ibsfuncs.assert_valid_names(name_list)
        ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        ibs.set_lblannot_values(nid_list, name_list)

    # SETTERS::LBLANNOT

    def set_lblannot_values(ibs, lblannot_rowid_list, value_list):
        """ Updates the value for lblannots. Note this change applies to
        all annotations related to this lblannot_rowid """
        id_iter = ((rowid,) for rowid in lblannot_rowid_list)
        val_list = ((value,) for value in value_list)
        ibs.db.set(LBLANNOT_TABLE, ('lblannot_value',), val_list, id_iter)

    def set_lblannot_notes(ibs, lblannot_rowid_list, value_list):
        """ Updates the value for lblannots. Note this change applies to
        all annotations related to this lblannot_rowid """
        id_iter = ((rowid,) for rowid in lblannot_rowid_list)
        val_list = ((value,) for value in value_list)
        ibs.db.set(LBLANNOT_TABLE, ('lblannot_note',), val_list, id_iter)

    # SETTERS::ANNOT->LBLANNOT(NAME)

    @setter
    def set_annot_names(ibs, aid_list, name_list):
        """ Sets the attrlbl_value of type(INDIVIDUAL_KEY) Sets names/nids of a
        list of annotations.  Convenience function for
        set_annot_lblannot_from_value"""
        ibs.set_annot_lblannot_from_value(aid_list, name_list, constants.INDIVIDUAL_KEY)

    @setter
    def set_annot_species(ibs, aid_list, species_list):
        """ Sets species/speciesids of a list of annotations.
        Convenience function for set_annot_lblannot_from_value """
        species_list = [species.lower() for species in species_list]
        ibsfuncs.assert_valid_species(ibs, species_list, iswarning=True)
        ibs.set_annot_lblannot_from_value(aid_list, species_list, constants.SPECIES_KEY)

    @setter
    def set_annot_lblannot_from_value(ibs, aid_list, value_list, _lbltype, ensure=True):
        """ Associates the annot and lblannot of a specific type and value
        Adds the lblannot if it doesnt exist.
        Wrapper around convenience function for set_annot_from_lblannot_rowid
        """
        assert value_list is not None
        assert _lbltype is not None
        if ensure:
            pass
        # a value consisting of an empty string or all spaces is set to the default
        DEFAULT_VALUE = constants.KEY_DEFAULTS[_lbltype]
        EMPTY_KEY = constants.EMPTY_KEY
        # setting a name to DEFAULT_VALUE or EMPTY is equivalent to unnaming it
        value_list_ = [DEFAULT_VALUE if value.strip() == EMPTY_KEY else value for value in value_list]
        notdefault_list = [value != DEFAULT_VALUE for value in value_list_]
        aid_list_to_delete = utool.get_dirty_items(aid_list, notdefault_list)
        # Set all the valid valids
        aids_to_set   = utool.filter_items(aid_list, notdefault_list)
        values_to_set = utool.filter_items(value_list_, notdefault_list)
        ibs.delete_annot_relations_oftype(aid_list_to_delete, _lbltype)
        # remove the relationships that have now been unnamed
        # Convert names into lblannot_rowid
        # FIXME: This function should not be able to set label realationships
        # to labels that have not been added!!
        # This is an inefficient way of getting lblannot_rowids!
        lbltype_rowid_list = [ibs.lbltype_ids[_lbltype]] * len(values_to_set)
        # auto ensure
        lblannot_rowid_list = ibs.add_lblannots(lbltype_rowid_list, values_to_set)
        # Call set_annot_from_lblannot_rowid to finish the conditional adding
        ibs.set_annot_lblannot_from_rowid(aids_to_set, lblannot_rowid_list, _lbltype)

    @setter
    def set_annot_nids(ibs, aid_list, nid_list):
        """ Sets names/nids of a list of annotations.
        Convenience function for set_annot_lblannot_from_rowid """
        ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        ibs.set_annot_lblannot_from_rowid(aid_list, nid_list, constants.INDIVIDUAL_KEY)

    @setter
    def set_annot_speciesids(ibs, aid_list, speciesid_list):
        """ Sets species/speciesids of a list of annotations.
        Convenience function for set_annot_lblannot_from_rowid"""
        ibsfuncs.assert_lblannot_rowids_are_type(ibs, speciesid_list, ibs.lbltype_ids[constants.SPECIES_KEY])
        ibs.set_annot_lblannot_from_rowid(aid_list, speciesid_list, constants.SPECIES_KEY)

    @setter
    def set_annot_lblannot_from_rowid(ibs, aid_list, lblannot_rowid_list, _lbltype):
        """ Sets items/lblannot_rowids of a list of annotations."""
        # Get the alrids_list for the aids, using the lbltype as a filter
        alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
        # Find the aids which already have relationships (of _lbltype)
        setflag_list = [len(alrids) > 0 for alrids in alrids_list]
        # Add the relationship if it doesn't exist
        aid_list_to_add = utool.get_dirty_items(aid_list, setflag_list)
        lblannot_rowid_list_to_add = utool.get_dirty_items(lblannot_rowid_list, setflag_list)
        # set the existing relationship if one already exists
        alrids_list_to_set = utool.filter_items(alrids_list, setflag_list)
        lblannot_rowid_list_to_set = utool.filter_items(lblannot_rowid_list, setflag_list)
        # Assert each annot has only one relationship of this type
        ibsfuncs.assert_singleton_relationship(ibs, alrids_list_to_set)
        alrid_list_to_set = utool.flatten(alrids_list_to_set)
        # Add the new relationships
        ibs.add_annot_relationship(aid_list_to_add, lblannot_rowid_list_to_add)
        # Set the old relationships
        ibs.set_alr_lblannot_rowids(alrid_list_to_set, lblannot_rowid_list_to_set)

    # ADDERS::LBLANNOT

    @adder
    def add_lblannots(ibs, lbltype_rowid_list, value_list, note_list=None):
        """ Adds new lblannots (labels of annotations)
        creates a new uuid for any new pair(type, value)
        #TODO: reverse order of rowid_list value_list in input
        """
        if note_list is None:
            note_list = [''] * len(value_list)
        # Get random uuids
        lblannot_uuid_list = [uuid.uuid4() for _ in range(len(value_list))]
        colnames = ['lblannot_uuid', 'lbltype_rowid', 'lblannot_value', 'lblannot_note']
        params_iter = list(zip(lblannot_uuid_list, lbltype_rowid_list, value_list, note_list))
        get_rowid_from_superkey = ibs.get_lblannot_rowid_from_superkey
        superkey_paramx = (1, 2)
        lblannot_rowid_list = ibs.db.add_cleanly(LBLANNOT_TABLE, colnames, params_iter,
                                                 get_rowid_from_superkey, superkey_paramx)
        return lblannot_rowid_list

    @adder
    def add_names(ibs, name_list, note_list=None):
        """ Adds a list of names. Returns their nids
        """
        # nid_list_ = [namenid_dict[name] for name in name_list_]
        # ibsfuncs.assert_valid_names(name_list)
        # All names are individuals and so may safely receive the INDIVIDUAL_KEY lblannot
        lbltype_rowid = ibs.lbltype_ids[constants.INDIVIDUAL_KEY]
        lbltype_rowid_list = [lbltype_rowid] * len(name_list)
        nid_list = ibs.add_lblannots(lbltype_rowid_list, name_list, note_list)
        return nid_list

    @adder
    def add_species(ibs, species_list, note_list=None):
        """ Adds a list of species. Returns their nids
        """
        species_list = [species.lower() for species in species_list]
        lbltype_rowid = ibs.lbltype_ids[constants.SPECIES_KEY]
        lbltype_rowid_list = [lbltype_rowid] * len(species_list)
        speciesid_list = ibs.add_lblannots(lbltype_rowid_list, species_list, note_list)
        return speciesid_list

    # INVESTIGATE::adders

    @adder
    def add_lbltype(ibs, text_list, default_list):
        """ Adds a label type and its default value
        Should only be called at the begining of the program.
        """
        params_iter = zip(text_list, default_list)
        colnames = ('lbltype_text', 'lbltype_default',)
        get_rowid_from_superkey = ibs.get_lbltype_rowid_from_text
        lbltype_rowid_list = ibs.db.add_cleanly(LBLTYPE_TABLE, colnames, params_iter,
                                                get_rowid_from_superkey)
        return lbltype_rowid_list

    # INVEST::getters

    @getter_1toM
    def get_annot_alrids(ibs, aid_list, configid=None):
        """ FIXME: __name__
        Get all the relationship ids belonging to the input annotations
        if lblannot lbltype is specified the relationship ids are filtered to
        be only of a specific lbltype/category/type
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
    def get_annot_alrids_oftype(ibs, aid_list, lbltype_rowid, configid=None):
        """
        Get all the relationship ids belonging to the input annotations where the
        relationship ids are filtered to be only of a specific lbltype/category/type
        """
        alrids_list = ibs.get_annot_alrids(aid_list, configid=configid)
        # Get lblannot_rowid of each relationship
        lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
        # Get the type of each lblannot
        lbltype_rowids_list = ibsfuncs.unflat_map(ibs.get_lblannot_lbltypes_rowids, lblannot_rowids_list)
        # only want the nids of individuals, not species, for example
        valids_list = [[typeid == lbltype_rowid for typeid in rowids] for rowids in lbltype_rowids_list]
        alrids_list = [utool.filter_items(alrids, valids) for alrids, valids in zip(alrids_list, valids_list)]
        assert all([len(alrid_list) < 2 for alrid_list in alrids_list]),\
            ("More than one type per lbltype.  ALRIDS: " + str(alrids_list) +
             ", ROW: " + str(lbltype_rowid) + ", KEYS:" + str(ibs.lbltype_ids))
        return alrids_list

    @getter_1toM
    def get_annot_lblannot_rowids(ibs, aid_list):
        """ Returns the name id of each annotation. """
        # Get all the annotation lblannot relationships
        # filter out only the ones which specify names
        alrids_list = ibs.get_annot_alrids(aid_list)
        lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
        return lblannot_rowids_list

    @getter_1toM
    def get_annot_lblannot_rowids_oftype(ibs, aid_list, _lbltype=None):
        """ Returns the name id of each annotation. """
        # Get all the annotation lblannot relationships
        # filter out only the ones which specify names
        assert _lbltype is not None, 'should be using lbltype_rowids anyway'
        alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
        lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
        return lblannot_rowids_list

    @utool.accepts_numpy
    @getter_1to1
    def get_annot_nids(ibs, aid_list, distinguish_unknowns=True):
        """ Returns the name id of each annotation. """
        # Get all the annotation lblannot relationships
        # filter out only the ones which specify names
        alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
        # Get a single nid from the list of lblannot_rowids of type INDIVIDUAL
        # TODO: get index of highest confidence name
        nid_list_ = [lblannot_rowids[0] if len(lblannot_rowids) else ibs.UNKNOWN_LBLANNOT_ROWID for
                     lblannot_rowids in lblannot_rowids_list]
        if distinguish_unknowns:
            nid_list = [-aid if nid == ibs.UNKNOWN_LBLANNOT_ROWID else nid
                        for nid, aid in zip(nid_list_, aid_list)]
        else:
            nid_list = nid_list_
        return nid_list

    # DELTERS

    @deleter
    #@cache_invalidator(LBLANNOT_TABLE)
    def delete_names(ibs, nid_list):
        """ deletes names from the database (CAREFUL. YOU PROBABLY DO NOT WANT
        TO USE THIS ENSURE THAT NONE OF THE NIDS HAVE ANNOTATION_TABLE) """
        ibs.delete_lblannots(nid_list)

    @deleter
    def delete_lblannots(ibs, lblannot_rowid_list):
        """ deletes lblannots from the database """
        if utool.VERBOSE:
            print('[ibs] deleting %d lblannots' % len(lblannot_rowid_list))
        ibs.db.delete_rowids(LBLANNOT_TABLE, lblannot_rowid_list)

    @deleter
    def delete_annot_relations_oftype(ibs, aid_list, _lbltype):
        """ Deletes the relationship between an annotation and a label """
        alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
        alrid_list = utool.flatten(alrids_list)
        ibs.db.delete_rowids(AL_RELATION_TABLE, alrid_list)

    @deleter
    def delete_annot_relations(ibs, aid_list):
        """ Deletes the relationship between an annotation and a label """
        alrids_list = ibs.get_annot_alrids(aid_list)
        alrid_list = utool.flatten(alrids_list)
        ibs.db.delete_rowids(AL_RELATION_TABLE, alrid_list)

    @deleter
    def delete_annot_nids(ibs, aid_list):
        """ Deletes nids of a list of annotations """
        # FIXME: This should be implicit by setting the anotation name to the
        # unknown name
        ibs.delete_annot_relations_oftype(aid_list, constants.INDIVIDUAL_KEY)

    @deleter
    def delete_annot_speciesids(ibs, aid_list):
        """ Deletes nids of a list of annotations """
        # FIXME: This should be implicit by setting the anotation name to the
        # unknown species
        ibs.delete_annot_relations_oftype(aid_list, constants.SPECIES_KEY)

    # MORE GETTERS

    @getter_1to1
    def get_annot_lblannot_value_of_lbltype(ibs, aid_list, _lbltype, lblannot_value_getter):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        lbltype_dict_list = ibs.get_annot_lblannot_rowids_oftype(aid_list, _lbltype)
        DEFAULT_VALUE = constants.KEY_DEFAULTS[_lbltype]
        # FIXME: Use filters and unflat maps
        lblannot_value_list = [lblannot_value_getter(lblannot_rowids)[0]
                               if len(lblannot_rowids) > 0 else DEFAULT_VALUE
                               for lblannot_rowids in lbltype_dict_list]
        return lblannot_value_list

    @getter_1to1
    def get_annot_names(ibs, aid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the individual """
        return ibs.get_annot_lblannot_value_of_lbltype(aid_list, constants.INDIVIDUAL_KEY, ibs.get_name_text)

    @getter_1to1
    def get_annot_species(ibs, aid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the species """
        return ibs.get_annot_lblannot_value_of_lbltype(aid_list, constants.SPECIES_KEY, ibs.get_species)

    #
    # GETTERS::EG_RELATION_TABLE

    @getter_1to1
    def get_egr_rowid_from_superkey(ibs, gid_list, eid_list):
        """ Gets eg-relate-ids from info constrained to be unique (eid, gid) """
        colnames = ('image_rowid',)
        params_iter = zip(gid_list, eid_list)
        where_clause = 'image_rowid=? AND encounter_rowid=?'
        egrid_list = ibs.db.get_where(EG_RELATION_TABLE, colnames, params_iter, where_clause)
        return egrid_list

    #
    # GETTERS::LBLTYPE

    @getter_1to1
    def get_lbltype_rowid_from_text(ibs, text_list):
        """ Returns lbltype_rowid where the lbltype_text is given """
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        # FIXME: Use unique SUPERKEYS instead of specifying id_colname
        lbltype_rowid = ibs.db.get(LBLTYPE_TABLE, ('lbltype_rowid',), text_list, id_colname='lbltype_text')
        return lbltype_rowid

    @getter_1to1
    def get_lbltype_default(ibs, lbltype_rowid_list):
        lbltype_default_list = ibs.db.get(LBLTYPE_TABLE, ('lbltype_default',), lbltype_rowid_list)
        return lbltype_default_list

    @getter_1to1
    def get_lbltype_text(ibs, lbltype_rowid_list):
        lbltype_text_list = ibs.db.get(LBLTYPE_TABLE, ('lbltype_text',), lbltype_rowid_list)
        return lbltype_text_list

    #
    # GETTERS::LBLANNOT_TABLE

    @getter_1to1
    def get_lblannot_rowid_from_superkey(ibs, lbltype_rowid_list, value_list):
        """
        Gets lblannot_rowid_list from the superkey (lbltype, value)
        """
        colnames = ('lblannot_rowid',)
        params_iter = zip(lbltype_rowid_list, value_list)
        where_clause = 'lbltype_rowid=? AND lblannot_value=?'
        lblannot_rowid_list = ibs.db.get_where(LBLANNOT_TABLE, colnames, params_iter, where_clause)
        return lblannot_rowid_list

    @getter_1to1
    def get_lblannot_rowid_from_uuid(ibs, lblannot_uuid_list):
        """
        Gets lblannot_rowid_list from the superkey (lbltype, value)
        """
        colnames = ('lblannot_rowid',)
        params_iter = lblannot_uuid_list
        id_colname = 'lblannot_uuid'
        lblannot_rowid_list = ibs.db.get(LBLANNOT_TABLE, colnames, params_iter, id_colname=id_colname)
        return lblannot_rowid_list

    @getter_1to1
    def get_lblannot_uuids(ibs, lblannot_rowid_list):
        lblannotuuid_list = ibs.db.get(LBLANNOT_TABLE, ('lblannot_uuid',), lblannot_rowid_list)
        return lblannotuuid_list

    @getter_1to1
    def get_lblannot_lbltypes_rowids(ibs, lblannot_rowid_list):
        lbltype_rowid_list = ibs.db.get(LBLANNOT_TABLE, ('lbltype_rowid',), lblannot_rowid_list)
        return lbltype_rowid_list

    @getter_1to1
    def get_lblannot_notes(ibs, lblannot_rowid_list):
        lblannotnotes_list = ibs.db.get(LBLANNOT_TABLE, ('lblannot_note',), lblannot_rowid_list)
        return lblannotnotes_list

    @getter_1to1
    def get_lblannot_values(ibs, lblannot_rowid_list, _lbltype=None):
        """ Returns text lblannots """
        #TODO: Remove keyword argument
        #ibsfuncs.assert_lblannot_rowids_are_type(ibs, lblannot_rowid_list,  ibs.lbltype_ids[_lbltype])
        lblannot_value_list = ibs.db.get(LBLANNOT_TABLE, ('lblannot_value',), lblannot_rowid_list)
        return lblannot_value_list

    @default_decorator
    def get_lblannot_aids(ibs, lblannot_rowid_list):
        #verbose = len(lblannot_rowid_list) > 20
        # TODO: Optimize IF POSSIBLE
        # FIXME: SLOW
        #if verbose:
        #    print(utool.get_caller_name(N=list(range(0, 20))))
        where_clause = 'lblannot_rowid=?'
        params_iter = [(lblannot_rowid,) for lblannot_rowid in lblannot_rowid_list]
        aids_list = ibs.db.get_where(AL_RELATION_TABLE, ('annot_rowid',), params_iter,
                                     where_clause, unpack_scalars=False)
        return aids_list

    #
    # GETTERS::LBLANNOTS_SUBSET

    @getter_1to1
    def get_species(ibs, speciesid_list):
        """ Returns text names """
        species_list = ibs.get_lblannot_values(speciesid_list, constants.SPECIES_KEY)
        return species_list

    @getter_1to1
    def get_name_nids(ibs, name_list, ensure=True):
        """ Returns nid_list. Creates one if it doesnt exist """
        if ensure:
            nid_list = ibs.add_names(name_list)
            return nid_list
        lbltype_rowid = ibs.lbltype_ids[constants.INDIVIDUAL_KEY]
        lbltype_rowid_list = [lbltype_rowid] * len(name_list)
        nid_list = ibs.get_lblannot_rowid_from_superkey(lbltype_rowid_list, name_list)
        return nid_list

    @getter_1to1
    def get_name_text(ibs, nid_list):
        """ Returns text names """
        # TODO:
        # Change the temporary negative indexes back to the unknown NID for the
        # SQL query. Then augment the lblannot list to distinguish unknown lblannots
        name_list = ibs.get_lblannot_values(nid_list, constants.INDIVIDUAL_KEY)
        return name_list

    @getter_1toM
    def get_name_aids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        # TODO: Optimize
        nid_list_ = [constants.UNKNOWN_LBLANNOT_ROWID if nid <= 0 else nid for nid in nid_list]
        #ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list_, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        aids_list = ibs.get_lblannot_aids(nid_list_)
        return aids_list

    @getter_1toM
    def get_name_exemplar_aids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        nid_list_ = [constants.UNKNOWN_LBLANNOT_ROWID if nid <= 0 else nid for nid in nid_list]
        #ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list_, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        aids_list = ibs.get_lblannot_aids(nid_list_)
        flags_list = ibsfuncs.unflat_map(ibs.get_annot_exemplar_flag, aids_list)
        exemplar_aids_list = [utool.filter_items(aids, flags) for aids, flags in
                              zip(aids_list, flags_list)]
        return exemplar_aids_list

    @getter_1to1
    def get_name_num_annotations(ibs, nid_list):
        """ returns the number of annotations for each name """
        # TODO: Optimize
        return list(map(len, ibs.get_name_aids(nid_list)))

    @getter_1to1
    def get_name_num_exemplar_annotations(ibs, nid_list):
        """ returns the number of annotations, which are exemplars for each name """
        return list(map(len, ibs.get_name_exemplar_aids(nid_list)))

    @getter_1to1
    def get_name_notes(ibs, nid_list):
        """ Returns name notes """
        notes_list = ibs.get_lblannot_notes(nid_list)
        return notes_list

    @getter_1toM
    def get_name_gids(ibs, nid_list):
        """ Returns the image ids associated with name ids"""
        # TODO: Optimize
        aids_list = ibs.get_name_aids(nid_list)
        gids_list = ibsfuncs.unflat_map(ibs.get_annot_gids, aids_list)
        return gids_list

    ## NEW

    @getter_1to1
    def get_annot_meta_lblannot_rowids(ibs, aid_list, _lbltype):
        """ ugg """
        #get_lblannot_values
        getter = partial(ibs.get_lblannot_values, _lbltype=constants.INDIVIDUAL_KEY)
        return ibs.get_annot_lblannot_value_of_lbltype(aid_list, _lbltype, getter)

    @adder
    def add_meta_lblannots(ibs, value_list, note_list=None, _lbltype=None):
        """ docstr """
        assert _lbltype is not None, 'bad lbltype'
        lbltype_rowid_list = [ibs.lbltype_ids[_lbltype]] * len(value_list)
        lblannot_rowid_list = ibs.add_lblannots(lbltype_rowid_list, value_list, note_list)
        return lblannot_rowid_list

    @setter
    def set_annot_meta_lblannot_values(ibs, aid_list, value_list, _lbltype):
        """ docstr """
        return ibs.set_annot_lblannot_from_value(aid_list, value_list, _lbltype)
