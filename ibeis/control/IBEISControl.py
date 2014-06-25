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
# TODO: rename roi annotations
# TODO: make all names consistent
from __future__ import absolute_import, division, print_function
# Python
import atexit
import requests
import uuid
from itertools import izip, imap
from functools import partial
from os.path import join, split
# Science
import numpy as np
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
                            ANNOT_TABLE,
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


__USTRCAST__ = str  # change to unicode if needed
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
        rid   - region of interest unique id
        roi   - region of interest for a chip
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
        ibs.db = sqldbc.SQLDatabaseController(ibs.get_ibsdir(), ibs.sqldb_fname)
        DB_SCHEMA.define_IBEIS_schema(ibs)
        ibs.MANUAL_CONFIG_SUFFIX = '_MANUAL_' + utool.get_computer_name()
        ibs.INDIVIDUAL_KEY = ibs.add_key('INDIVIDUAL_KEY')
        ibs.SPECIES_KEY = ibs.add_key('SPECIES_KEY')
        ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)
        assert ibs.INDIVIDUAL_KEY == 1, 'INDIVIDUAL_KEY = %r' % ibs.INDIVIDUAL_KEY
        assert ibs.SPECIES_KEY == 2, 'SPECIES_KEY = %r' % ibs.SPECIES_KEY
        ibs.UNKNOWN_NAME = constants.UNKNOWN_NAME
        ibs.UNKNOWN_NID = ibs.get_name_nids((ibs.UNKNOWN_NAME,), ensure=True)[0]
        try:
            assert ibs.UNKNOWN_NID == 1
        except AssertionError:
            print('[!ibs] ERROR: ibs.UNKNOWN_NID = %r' % ibs.UNKNOWN_NID)
            raise

    @getter_1to1
    def get_key_rowid_from_text(ibs, text_list):
        key_rowid = ibs.db.get(KEY_TABLE, ('key_rowid',), text_list, id_colname='key_text')
        return key_rowid

    @adder
    def add_key(ibs, text_list):
        params_iter = ((text,) for text in text_list)
        key = ibs.db.add_cleanly(KEY_TABLE, ('key_text',), params_iter, ibs.get_key_rowid_from_text)
        return key

    @default_decorator
    def clone_handle(ibs, **kwargs):
        ibs2 = IBEISController(dbdir=ibs.get_dbdir(), ensure=False)
        if len(kwargs) > 0:
            ibs2.update_query_cfg(**kwargs)
        if ibs.qreq is not None:
            ibs2._prep_qreq(ibs.qreq.qrids, ibs.qreq.drids)
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
            # HACK: FORCING DEFAULTS FOR NOW
            if True or utool.get_flag(('--noprefload', '--noprefload')):
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

    def get_num_rois(ibs, **kwargs):
        rid_list = ibs.get_valid_rids(**kwargs)
        return len(rid_list)

    def get_num_names(ibs, **kwargs):
        nid_list = ibs.get_valid_nids(**kwargs)
        return len(nid_list)

    @ider
    def _get_all_gids(ibs):
        all_gids = ibs.db.get_executeone(IMAGE_TABLE, ('image_rowid',))
        return sorted(all_gids)

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
    def _get_all_rids(ibs):
        """ returns a all ROI ids """
        all_rids = ibs.db.get_executeone(ANNOT_TABLE, ('annot_rowid',))
        return sorted(all_rids)

    @ider
    def get_valid_cids(ibs):
        chip_config_rowid = ibs.get_chip_config_rowid()
        params = (chip_config_rowid,)
        cid_list = ibs.db.get_executeone_where(CHIP_TABLE, ('chip_rowid',), 'config_rowid=?', params)
        return sorted(cid_list)

    @ider
    def _get_all_cids(ibs):
        """ Returns computed chips for every configuration
            (you probably should not use this) """
        all_cids = ibs.db.get_executeone(CHIP_TABLE, ('chip_rowid',))
        return sorted(all_cids)
    @ider
    def get_valid_fids(ibs):
        feat_config_rowid = ibs.get_feat_config_rowid()
        params = (feat_config_rowid,)
        fid_list = ibs.db.get_executeone_where(FEATURE_TABLE, ('feature_rowid',), 'config_rowid=?', params)
        return sorted(fid_list)

    @ider
    def _get_all_fids(ibs):
        """ Returns computed features for every configuration
        (you probably should not use this)"""
        all_fids = ibs.db.get_executeone(FEATURE_TABLE, ('feature_rowid',))
        return sorted(all_fids)
    @ider
    def _get_all_known_nids(ibs):
        """ Returns all nids of known animals
            (does not include unknown names) """
        where_clause = 'label_value!=?'
        params = (ibs.UNKNOWN_NAME,)
        all_nids = ibs.db.get_executeone_where(LABEL_TABLE, ('label_rowid',), where_clause, params)
        return sorted(all_nids)

    @ider
    def get_valid_nids(ibs, eid=None):
        """ Returns all valid names with at least one animal
            (does not include unknown names) """
        if eid is None:
            _nid_list = ibs._get_all_known_nids()
        else:
            _nid_list = ibs.get_encounter_nids(eid)
        nRois_list = ibs.get_name_num_rois(_nid_list)
        nid_list = [nid for nid, nRois in izip(_nid_list, nRois_list)
                    if nRois > 0]
        return sorted(nid_list)

    @ider
    def get_invalid_nids(ibs):
        """ Returns all names without any animals (does not include unknown names) """
        _nid_list = ibs._get_all_known_nids()
        nRois_list = ibs.get_name_num_rois(_nid_list)
        nid_list = [nid for nid, nRois in izip(_nid_list, nRois_list)
                    if nRois <= 0]
        return sorted(nid_list)

    @ider
    def _get_all_eids(ibs):
        all_eids = ibs.db.get_executeone(ENCOUNTER_TABLE, ('encounter_rowid',))
        return sorted(all_eids)

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
    def get_valid_rids(ibs, eid=None, is_exemplar=False):
        """ returns a list of valid ROI unique ids """
        if eid is None:
            rid_list = ibs._get_all_rids()
        else:
            rid_list = ibs.get_encounter_rids(eid)
        if is_exemplar:
            flag_list = ibs.get_roi_exemplar_flag(rid_list)
            rid_list = utool.filter_items(rid_list, flag_list)
        return sorted(rid_list)

    #
    #
    #---------------
    # --- ADDERS ---
    #---------------

    @adder
    def add_config(ibs, cfgsuffix_list):
        """ Adds an algorithm configuration as a string """
        # FIXME: Configs are still handled poorly
        configid_list = ibs.get_config_rowid_from_suffix(cfgsuffix_list, ensure=False)
        #print('configid_list %r' % (configid_list,))
        #print('cfgsuffix_list %r' % (cfgsuffix_list,))
        try:
            # [Jon]FIXME: This check is really weird? Why is it here?
            isdirty_list = [
                rowid is None or (isinstance(rowid, list) and len(rowid) == 0)
                for rowid in configid_list]
            if any(isdirty_list):
                params_iter = ((suffix,) for suffix in cfgsuffix_list)
                colnames = ('config_suffix',)
                get_rowid_from_uuid = partial(ibs.get_config_rowid_from_suffix, ensure=False)
                configid_list = ibs.db.add_cleanly(CONFIG_TABLE, colnames, params_iter, get_rowid_from_uuid)
        except Exception as ex:
            utool.printex(ex)
            print('FATAL ERROR')
            utool.sys.exit(1)
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
        print('[ibs] gpath_list = %r' % (gpath_list,))
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
    def add_rois(ibs, gid_list, bbox_list=None, theta_list=None, nid_list=None,
                 name_list=None, detect_confidence_list=None, notes_list=None,
                 vert_list=None):
        """ Adds oriented ROI bounding boxes to images """
        print('[ibs] adding rois')
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
        if nid_list is None:
            nid_list = [ibs.UNKNOWN_NID for _ in xrange(len(gid_list))]
        if detect_confidence_list is None:
            detect_confidence_list = [0.0 for _ in xrange(len(gid_list))]
        if notes_list is None:
            notes_list = ['' for _ in xrange(len(gid_list))]
        if vert_list is None:
            vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
        elif bbox_list is None:
            bbox_list = geometry.bboxes_from_vert_list(vert_list)

        # Build ~~deterministic?~~ random and unique ROI ids
        image_uuid_list = ibs.get_image_uuids(gid_list)
        roi_uuid_list = ibsfuncs.make_roi_uuids(image_uuid_list, bbox_list,
                                                theta_list, deterministic=False)
        nVert_list = [len(verts) for verts in vert_list]
        vertstr_list = [str(verts) for verts in vert_list]
        xtl_list, ytl_list, width_list, height_list = list(izip(*bbox_list))
        assert len(nVert_list) == len(vertstr_list)
        # Define arguments to insert
        colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                    'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                    'annot_verts', 'annot_detect_confidence',
                    'annot_note',)

        params_iter = list(izip(roi_uuid_list, gid_list, xtl_list, ytl_list,
                                width_list, height_list, theta_list, nVert_list,
                                vertstr_list, detect_confidence_list,
                                notes_list))
        #utool.embed()

        # Execute add ROIs SQL
        get_rowid_from_uuid = ibs.get_roi_rids_from_uuid
        rid_list = ibs.db.add_cleanly(ANNOT_TABLE, colnames, params_iter, get_rowid_from_uuid)

        # Also need to populate roi_label_relationship table
        alrid_list = ibs.add_roi_relationship(rid_list, nid_list)
        #print('alrid_list = %r' % (alrid_list,))

        # Invalidate image thumbnails
        ibs.delete_image_thumbtups(gid_list)
        return rid_list

    @adder
    def add_roi_relationship(ibs, rid_list, labelid_list, configid_list=None,
                             alr_confidence_list=None):
        if configid_list is None:
            configid_list = [ibs.MANUAL_CONFIGID] * len(rid_list)
        if alr_confidence_list is None:
            alr_confidence_list = [0.0] * len(rid_list)
        colnames = ('annot_rowid', 'label_rowid', 'config_rowid', 'alr_confidence')
        params_iter = list(izip(rid_list, labelid_list, configid_list,
                                alr_confidence_list))
        alrid_list = ibs.db.add_cleanly(AL_RELATION_TABLE, colnames, params_iter,
                                        ibs.get_alr_rowid_from_valtup, range(0, 3))
        return alrid_list

    @getter_1to1
    def get_alr_rowid_from_valtup(ibs, rid_list, labelid_list, configid_list):
        colnames = ('annot_rowid',)
        params_iter = izip(rid_list, labelid_list, configid_list)
        where_clause = 'annot_rowid=? AND label_rowid=? AND config_rowid=?'
        alrid_list = ibs.db.get_where(AL_RELATION_TABLE, colnames, params_iter, where_clause)
        return alrid_list

    @adder
    def add_chips(ibs, rid_list):
        """ Adds chip data to the ROI. (does not create ROIs. first use add_rois
        and then pass them here to ensure chips are computed)
        return cid_list
        """
        # Ensure must be false, otherwise an infinite loop occurs
        cid_list = ibs.get_roi_cids(rid_list, ensure=False)
        dirty_rids = utool.get_dirty_items(rid_list, cid_list)
        if len(dirty_rids) > 0:
            print('[ibs] adding chips')
            try:
                # FIXME: Cant be lazy until chip config / delete issue is fixed
                preproc_chip.compute_and_write_chips(ibs, rid_list)
                #preproc_chip.compute_and_write_chips_lazy(ibs, rid_list)
                params_iter = preproc_chip.add_chips_params_gen(ibs, dirty_rids)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.add_chips]')
                print('[!ibs.add_chips] ' + utool.list_dbgstr('rid_list'))
                raise
            colnames = ('annot_rowid', 'chip_uri', 'chip_width', 'chip_height',
                        'config_rowid',)
            get_rowid_from_uuid = partial(ibs.get_roi_cids, ensure=False)
            cid_list = ibs.db.add_cleanly(CHIP_TABLE, colnames, params_iter, get_rowid_from_uuid)

        return cid_list

    @adder
    def add_feats(ibs, cid_list, force=False):
        """ Computes the features for every chip without them """
        fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        dirty_cids = utool.get_dirty_items(cid_list, fid_list)
        if len(dirty_cids) > 0:
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
        # Ensure input list is unique
        # name_list = tuple(set(name_list_))
        # HACKY, the adder decorator should specify this

        nid_list = ibs.get_name_nids(name_list, ensure=False)
        dirty_names = utool.get_dirty_items(name_list, nid_list)
        if len(dirty_names) > 0:
            print('[ibs] adding %d names' % len(dirty_names))
            ibsfuncs.assert_valid_names(name_list)
            notes_list = ['' for _ in xrange(len(dirty_names))]
            # All names are individuals and so may safely receive the INDIVIDUAL_KEY label
            key_rowid_list = [ibs.INDIVIDUAL_KEY for name in name_list]
            new_nid_list = ibs.add_labels(key_rowid_list, dirty_names, notes_list)
            print('new_nid_list = %r' % (new_nid_list,))
            #get_rowid_from_uuid = partial(ibs.get_name_nids, ensure=False)
            #new_nid_list = ibs.db.add_cleanly(LABEL_TABLE, colnames, params_iter, get_rowid_from_uuid)
            new_nid_list  # this line silences warnings

            # All the names should have been ensured
            # this nid list should correspond to the input
            nid_list = ibs.get_name_nids(name_list, ensure=False)
            print('nid_list = %r' % (new_nid_list,))

        # # Return nids in input order
        # namenid_dict = {name: nid for name, nid in izip(name_list, nid_list)}
        # nid_list_ = [namenid_dict[name] for name in name_list_]
        return nid_list

    def add_labels(ibs, key_list, value_list, note_list):
        #label_uuid_list = [uuid.uuid4() for _ in xrange(len(value_list))]
        # FIXME: This should actually be a random uuid, but (key, vals) should be
        # enforced as unique as well
        label_uuid_list = [utool.deterministic_uuid(repr((key, value))) for key, value in
                           izip(key_list, value_list)]
        colnames = ['label_uuid', 'key_rowid', 'label_value', 'label_note']
        params_iter = list(izip(label_uuid_list, key_list, value_list, note_list))
        labelid_list = ibs.db.add_cleanly(LABEL_TABLE, colnames, params_iter,
                                          ibs.get_labelid_from_uuid)
        return labelid_list

    @adder
    def add_encounters(ibs, enctext_list):
        """ Adds a list of names. Returns their nids """
        print('[ibs] adding %d encounters' % len(enctext_list))
        # Add encounter text names to database
        notes_list = ['' for _ in xrange(len(enctext_list))]
        encounter_uuid_list = [uuid.uuid4() for _ in xrange(len(enctext_list))]
        colnames = ['encounter_text', 'encounter_uuid', 'encounter_note']
        params_iter = izip(enctext_list, encounter_uuid_list, notes_list)
        get_rowid_from_uuid = partial(ibs.get_encounter_eids, ensure=False)

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
        id_list = ((gid,) for gid in gid_list)
        val_list = ((new_gpath,) for new_gpath in new_gpath_list)
        ibs.db.set(IMAGE_TABLE, ('image_uri',), val_list, id_list)

    @setter
    def set_image_aifs(ibs, gid_list, aif_list):
        """ Sets the image all instances found bit """
        id_list = ((gid,) for gid in gid_list)
        val_list = ((aif,) for aif in aif_list)
        ibs.db.set(IMAGE_TABLE, ('image_toggle_aif',), val_list, id_list)

    @setter
    def set_image_notes(ibs, gid_list, notes_list):
        """ Sets the image all instances found bit """
        id_list = ((gid,) for gid in gid_list)
        val_list = ((notes,) for notes in notes_list)
        ibs.db.set(IMAGE_TABLE, ('image_note',), val_list, id_list)

    @setter
    def set_image_unixtime(ibs, gid_list, unixtime_list):
        """ Sets the image unixtime (does not modify exif yet) """
        id_list = ((gid,) for gid in gid_list)
        val_list = ((unixtime,) for unixtime in unixtime_list)
        ibs.db.set(IMAGE_TABLE, ('image_time_posix',), val_list, id_list)

    @setter
    def set_image_enctext(ibs, gid_list, enctext_list):
        """ Sets the encoutertext of each image """
        print('[ibs] Setting %r image encounter ids' % len(gid_list))
        eid_list = ibs.add_encounters(enctext_list)
        egrid_list = ibs.add_image_relationship(gid_list, eid_list)
        # ibs.db.executemany(
        #     operation='''
        #     INSERT OR IGNORE INTO encounter_image_relationship(
        #         egpair_rowid,
        #         image_rowid,
        #         encounter_rowid
        #     ) VALUES (NULL, ?, ?)
        #     ''',
        #     params_iter=izip(gid_list, eid_list))
        # DOES NOT WORK
        #gid_list = ibs.db.add_cleanly(tblname, colnames, params_iter,
        #                              get_rowid_from_uuid=(lambda gid: gid))

    @adder
    def add_image_relationship(ibs, gid_list, eid_list):
        colnames = ('image_rowid', 'encounter_rowid')
        params_iter = list(izip(gid_list, eid_list))
        egrid_list = ibs.db.add_cleanly(EG_RELATION_TABLE, colnames, params_iter,
                                        ibs.get_egr_rowid_from_valtup, range(0, 2))
        return egrid_list

    @setter
    def set_image_gps(ibs, gid_list, gps_list):
        id_list = ((gid,) for gid in gid_list)
        # see get_image_gps for how the gps_list should look
        lat_list = [tup[0] for tup in gps_list]
        lon_list = [tup[1] for tup in gps_list]
        colnames = ('image_gps_lat', 'image_gps_lon',)
        val_list = izip(lat_list, lon_list)
        ibs.db.set(IMAGE_TABLE, colnames, val_list, id_list)

    # SETTERS::ROI

    @setter
    def set_roi_exemplar_flag(ibs, rid_list, flag_list):
        id_list = ((rid,) for rid in rid_list)
        val_list = ((flag,) for flag in flag_list)
        ibs.db.set(ANNOT_TABLE, ('annot_exemplar_flag',), val_list, id_list)

    @setter
    def set_roi_bboxes(ibs, rid_list, bbox_list):
        """ Sets ROIs of a list of rois by rid, where roi_list is a list of
            (x, y, w, h) tuples """
        # changing the bboxes also changes the bounding polygon
        vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
        # naively overwrite the bounding polygon with a rectangle - for now trust the user!
        ibs.set_roi_verts(rid_list, vert_list)
        colnames = ['annot_xtl', 'annot_ytl', 'annot_width', 'annot_height']
        ibs.db.set(ANNOT_TABLE, colnames, bbox_list, rid_list)

    @setter
    def set_roi_thetas(ibs, rid_list, theta_list):
        """ Sets thetas of a list of chips by rid """
        ibs.delete_roi_chips(rid_list)  # Changing theta redefines the chips
        id_list = ((rid,) for rid in rid_list)
        val_list = ((theta,) for theta in theta_list)
        ibs.db.set(ANNOT_TABLE, ('annot_theta',), val_list, id_list)

    @setter
    def set_roi_verts(ibs, rid_list, verts_list):
        """ Sets the vertices [(x, y), ...] of a list of chips by rid """
        ibs.delete_roi_chips(rid_list)
        num_verts_list = [len(verts) for verts in verts_list]
        verts_as_strings = [str(verts) for verts in verts_list]
        # need a list comprehension because we want to re-use id_list
        id_list = [(rid,) for rid in rid_list]
        # also need to set the internal number of vertices
        val_list = ((num_verts, verts) for (num_verts, verts)
        			in izip(num_verts_list, verts_as_strings))
        colnames = ('annot_num_verts', 'annot_verts',)
        ibs.db.set(ANNOT_TABLE, colnames, val_list, id_list)

        # changing the vertices also changes the bounding boxes
        bbox_list = geometry.bboxes_from_vert_list(verts_list)	# new bboxes
        xtl_list, ytl_list, width_list, height_list = list(izip(*bbox_list))

        colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
        val_list2 = ((xtl, ytl, width, height)
        				for (xtl, ytl, width, height) in
        				izip(xtl_list, ytl_list, width_list, height_list))
        ibs.db.set(ANNOT_TABLE, colnames, val_list2, id_list)

    @setter
    def set_roi_notes(ibs, rid_list, notes_list):
        """ Sets roi notes """
        id_list = ((rid,) for rid in rid_list)
        val_list = ((notes,) for notes in notes_list)
        ibs.db.set(ANNOT_TABLE, ('annot_note',), val_list, id_list)

    @setter
    def set_roi_names(ibs, rid_list, name_list=None, nid_list=None):
        """ Sets names/nids of a list of rois.
        Convenience function for set_roi_nids"""
        assert name_list is None or nid_list is None, (
            'can only specify one type of name values (nid or name) not both')
        if nid_list is None:
            assert name_list is not None
            # Convert names into nids
            nid_list = ibs.add_names(name_list)
        ibs.set_roi_nids(rid_list, nid_list)

    @setter
    def set_roi_nids(ibs, rid_list, nid_list):
        """ Sets nids of a list of rois """
        # Ensure we are setting true nids (not temporary distinguished nids)
        # nids are really special labelids
        labelid_list = [nid if nid > 0 else ibs.UNKNOWN_NID for nid in nid_list]
        INDIVIDUAL_KEY = ibs.INDIVIDUAL_KEY
        alrid_list = ibs.get_roi_filtered_relationship_ids(rid_list, INDIVIDUAL_KEY)
        # SQL Setter arguments
        # Cannot use set_table_props for cross-table setters.
        ibs.db.set(AL_RELATION_TABLE, ('label_rowid',), labelid_list, alrid_list)

    # SETTERS::NAME

    @setter
    def set_name_notes(ibs, nid_list, notes_list):
        """ Sets notes of names (groups of animals) """
        id_list = ((nid,) for nid in nid_list)
        val_list = ((notes,) for notes in notes_list)
        ibs.db.set(LABEL_TABLE, ('label_note',), val_list, id_list)

    @setter
    def set_name_names(ibs, nid_list, name_list):
        """ Changes the name text. Does not affect the animals of this name """
        ibsfuncs.assert_valid_names(name_list)
        id_list = ((nid,) for nid in nid_list)
        val_list = ((name,) for name in name_list)
        ibs.db.set(LABEL_TABLE, ('label_value',), val_list, id_list)

    @setter
    def set_encounter_props(ibs, eid_list, key, value_list):
        print('[ibs] set_encounter_props')
        id_list = ((eid,) for eid in eid_list)
        val_list = ((value,) for value in value_list)
        ibs.db.set(ENCOUNTER_TABLE, key, val_list, id_list)

    @setter
    def set_encounter_enctext(ibs, eid_list, names_list):
        """ Sets names of encounters (groups of animals) """
        id_list = ((eid,) for eid in eid_list)
        val_list = ((names,) for names in names_list)
        ibs.db.set(ENCOUNTER_TABLE, ('encounter_text',), val_list, id_list)

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    #
    # GETTERS::GENERAL

    def get_valid_ids(ibs, tblname, eid=None):
        get_valid_tblname_ids = {
            'gids': ibs.get_valid_gids,
            'rids': ibs.get_valid_rids,
            'nids': ibs.get_valid_nids,
        }[tblname]
        return get_valid_tblname_ids(eid=eid)

    #
    # GETTERS::IMAGE

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
        img_uuid_list = ibs.get_image_uuids(gid_list)
        rids_list = ibs.get_image_rids(gid_list)
        bboxes_list = ibsfuncs.unflat_map(ibs.get_roi_bboxes, rids_list)
        thumb_dpath = ibs.thumb_dpath
        thumb_gpaths = [join(thumb_dpath, str(uuid) + 'thumb.png')
                        for uuid in img_uuid_list]
        image_paths = ibs.get_image_paths(gid_list)
        thumbtup_list = list(izip(thumb_gpaths, image_paths, bboxes_list))
        return thumbtup_list

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
        gid_list = ibs.db.get(IMAGE_TABLE, ('image_rowid',), uuid_list,
                              id_colname='image_uuid')
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
        lat_list = ibs.db.get(IMAGE_TABLE, ('image_gps_lat',), gid_list)
        lon_list = ibs.db.get(IMAGE_TABLE, ('image_gps_lon',), gid_list)
        gps_list = [(lat, lon) for (lat, lon) in izip(lat_list, lon_list)]
        return gps_list

    @getter_1to1
    def get_image_aifs(ibs, gid_list):
        """ Returns "All Instances Found" flag, true if all objects of interest
        (animals) have an ROI in the image """
        aif_list = ibs.db.get(IMAGE_TABLE, ('image_toggle_aif',), gid_list)
        return aif_list

    @getter_1to1
    def get_image_detect_confidence(ibs, gid_list):
        """ Returns image detection confidence as the max of ROI confidences """
        rids_list = ibs.get_image_rids(gid_list)
        confs_list = ibsfuncs.unflat_map(ibs.get_roi_detect_confidence, rids_list)
        maxconf_list = [max(confs) if len(confs) > 0 else -1
                        for confs in confs_list]
        return maxconf_list

    @getter_1to1
    def get_image_notes(ibs, gid_list):
        """ Returns image notes """
        notes_list = ibs.db.get(IMAGE_TABLE, ('image_note',), gid_list)
        return notes_list

    @getter_1to1
    def get_image_nids(ibs, gid_list):
        """ Returns the name ids associated with an image id """
        rids_list = ibs.get_image_rids(gid_list)
        nids_list = ibsfuncs.unflat_map(ibs.get_roi_nids, rids_list)
        return nids_list

    @getter_1toM
    def get_name_gids(ibs, nid_list):
        """ Returns the image ids associated with name ids"""
        rids_list = ibs.get_name_rids(nid_list)
        gids_list = ibsfuncs.unflat_map(ibs.get_roi_gids, rids_list)
        return gids_list

    @getter_1toM
    def get_image_eids(ibs, gid_list):
        """ Returns a list of encounter ids for each image by gid """
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
    def get_image_rids(ibs, gid_list):
        """ Returns a list of rids for each image by gid """
        #print('gid_list = %r' % (gid_list,))
        rids_list = ibs.db.get(ANNOT_TABLE, ('annot_rowid',), gid_list,
                                id_colname='image_rowid',
                                unpack_scalars=False)
        #print('rids_list = %r' % (rids_list,))
        return rids_list

    @getter_1to1
    def get_image_num_rois(ibs, gid_list):
        """ Returns the number of chips in each image """
        return list(imap(len, ibs.get_image_rids(gid_list)))

    #
    # GETTERS::ROI

    @getter_1to1
    def get_roi_exemplar_flag(ibs, rid_list):
        roi_uuid_list = ibs.db.get(ANNOT_TABLE, ('annot_exemplar_flag',), rid_list)
        return roi_uuid_list

    @getter_1to1
    def get_roi_uuids(ibs, rid_list):
        """ Returns a list of image uuids by gid """
        roi_uuid_list = ibs.db.get(ANNOT_TABLE, ('annot_uuid',), rid_list)
        return roi_uuid_list

    @getter_1to1
    def get_roi_rids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        rids_list = ibs.db.get(ANNOT_TABLE, ('annot_rowid',), uuid_list,
                               id_colname='annot_uuid')
        return rids_list

    @getter_1to1
    def get_roi_detect_confidence(ibs, rid_list):
        """ Returns a list confidences that the rois is a valid detection """
        roi_detect_confidence_list = ibs.db.get(ANNOT_TABLE,
                                                ('annot_detect_confidence',),
                                                rid_list)
        return roi_detect_confidence_list

    @getter_1to1
    def get_roi_notes(ibs, rid_list):
        """ Returns a list of roi notes """
        roi_notes_list = ibs.db.get(ANNOT_TABLE, ('annot_note',), rid_list)
        return roi_notes_list

    @utool.accepts_numpy
    @getter_1toM
    def get_roi_bboxes(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height')
        bbox_list = ibs.db.get(ANNOT_TABLE, colnames, rid_list)
        return bbox_list

    @getter_1to1
    def get_roi_thetas(ibs, rid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.db.get(ANNOT_TABLE, ('annot_theta',), rid_list)
        return theta_list

    @getter_1to1
    def get_roi_num_verts(ibs, rid_list):
        """ Returns the number of vertices that form the polygon of each chip """
        num_verts_list = ibs.db.get(ANNOT_TABLE, ('annot_num_verts',), rid_list)
        return num_verts_list

    @getter_1to1
    def get_roi_verts(ibs, rid_list):
        """ Returns the vertices that form the polygon of each chip """
        vertstr_list = ibs.db.get(ANNOT_TABLE, ('annot_verts',), rid_list)
        # TODO: Sanatize input for eval
        #print('vertstr_list = %r' % (vertstr_list,))
        return [eval(vertstr) for vertstr in vertstr_list]

    @utool.accepts_numpy
    @getter_1to1
    def get_roi_gids(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        gid_list = ibs.db.get(ANNOT_TABLE, ('image_rowid',), rid_list,
                              id_colname='annot_rowid')
        #try:
        #    utool.assert_all_not_None(gid_list, 'gid_list')
        #except AssertionError as ex:
        #    ibsfuncs.assert_valid_rids(ibs, rid_list)
        #    utool.printex(ex, 'Rids must have image ids!', key_list=[
        #        'gid_list', 'rid_list'])
        #    raise
        return gid_list

    @getter_1to1
    def get_roi_cids(ibs, rid_list, ensure=True, all_configs=False):
        if ensure:
            try:
                ibs.add_chips(rid_list)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.get_roi_cids]')
                print('[!ibs.get_roi_cids] rid_list = %r' % (rid_list,))
                raise
        if all_configs:
            cid_list = ibs.db.get(CHIP_TABLE, ('chip_rowid',), rid_list, id_colname='annot_rowid')
        else:
            chip_config_rowid = ibs.get_chip_config_rowid()
            #print(chip_config_rowid)
            where_clause = 'annot_rowid=? AND config_rowid=?'
            params_iter = ((rid, chip_config_rowid) for rid in rid_list)
            cid_list = ibs.db.get_where(CHIP_TABLE,  ('chip_rowid',), params_iter, where_clause)
        if ensure:
            try:
                utool.assert_all_not_None(cid_list, 'cid_list')
            except AssertionError as ex:
                valid_cids = ibs.get_valid_cids()  # NOQA
                utool.printex(ex, 'Ensured cids returned None!',
                              key_list=['rid_list', 'cid_list', 'valid_cids'])
                raise
        return cid_list

    @getter_1to1
    def get_roi_fids(ibs, rid_list, ensure=False):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        return fid_list

    # <LABEL_GETTERS>

    def get_labelid_from_uuid(ibs, label_uuid_list):
        labelid_list = ibs.db.get(LABEL_TABLE, ('label_rowid',), label_uuid_list,
                                  id_colname='label_uuid')
        return labelid_list

    @getter_1to1
    def get_labelids_from_values(ibs, value_list, key_rowid):
        params_iter = [(value, key_rowid) for value in value_list]
        where_clause = 'label_value=? AND key_rowid=?'
        labelid_list = ibs.db.get_where(LABEL_TABLE, ('label_rowid',), params_iter,
                                        where_clause)
        #print('[ibs] labelid_list = %r' % (labelid_list,))
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

    # </LABEL_GETTERS>

    @getter_1toM
    def get_roi_relationship_ids(ibs, rid_list):
        """ FIXME: func_name
        Get all the relationship ids belonging to the input rois
        if label key is specified the realtionship ids are filtered to
        be only of a specific key/category/type
        """
        alrids_list = ibs.db.get(AL_RELATION_TABLE, ('alr_rowid',), rid_list,
                                 id_colname='annot_rowid', unpack_scalars=False)
        assert all([x > 0 for x in map(len, alrids_list)]), 'annotations must have at least one relationship'
        return alrids_list

    @getter_1to1
    def get_roi_filtered_relationship_ids(ibs, rid_list, key_rowid):
        """ FIXME: func_name
        Get all the relationship ids belonging to the input rois where the
        realtionship ids are filtered to be only of a specific key/category/type
        """
        alrids_list = ibs.get_roi_relationship_ids(rid_list)
        # Get labelid of each relationship
        labelids_list = ibsfuncs.unflat_map(ibs.get_relationship_labelids, alrids_list)
        # Get the type of each label
        labelkeys_list = ibsfuncs.unflat_map(ibs.get_label_keys, labelids_list)
        try:
            # only want the nids of individuals, not species, for example
            index_list = [keys.index(key_rowid) for keys in labelkeys_list]
            alrid_list = [ids[index] for (ids, index)  in izip(alrids_list, index_list)]
        except Exception as ex:
            utool.printex(ex, key_list=['rid_list', 'alrid_list', 'alrids_list', 'key_rowid', 'labelkeys_list',
                                        'labelids_list', 'index_list'])
            raise
        return alrid_list

    @getter_1to1
    def get_relationship_labelids(ibs, alrid_list):
        """ get the labelid belonging to each relationship """
        labelids_list = ibs.db.get(AL_RELATION_TABLE, ('label_rowid',), alrid_list)
        return labelids_list

    @utool.accepts_numpy
    @getter_1to1
    def get_roi_nids(ibs, rid_list, distinguish_unknowns=True):
        """
            Returns the name id of each roi.
            If distinguish_uknowns is True, returns negative roi rowids
            instead of unknown name id
        """
        # Get all the roi label relationships
        # filter out only the ones which specify names
        alrid_list = ibs.get_roi_filtered_relationship_ids(rid_list, ibs.INDIVIDUAL_KEY)
        labelid_list = ibs.get_relationship_labelids(alrid_list)
        nid_list = labelid_list
        if distinguish_unknowns:
            tnid_list = [nid if nid != ibs.UNKNOWN_NID else -rid
                         for (nid, rid) in izip(nid_list, rid_list)]
            return tnid_list
        else:
            return nid_list

    @getter_1to1
    def get_roi_gnames(ibs, rid_list):
        """ Returns the image names of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        gname_list = ibs.get_image_gnames(gid_list)
        return gname_list

    @getter_1to1
    def get_roi_images(ibs, rid_list):
        """ Returns the images of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        image_list = ibs.get_images(gid_list)
        return image_list

    @getter_1to1
    def get_roi_image_uuids(ibs, rid_list):
        gid_list = ibs.get_roi_gids(rid_list)
        image_uuid_list = ibs.get_image_uuids(gid_list)
        return image_uuid_list

    @getter_1to1
    def get_roi_gpaths(ibs, rid_list):
        """ Returns the image names of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        try:
            utool.assert_all_not_None(gid_list, 'gid_list')
        except AssertionError:
            print('[!get_roi_gpaths] ' + utool.list_dbgstr('rid_list'))
            print('[!get_roi_gpaths] ' + utool.list_dbgstr('gid_list'))
            raise
        gpath_list = ibs.get_image_paths(gid_list)
        utool.assert_all_not_None(gpath_list, 'gpath_list')
        return gpath_list

    @getter_1to1
    def get_roi_chips(ibs, rid_list, ensure=True):
        utool.assert_all_not_None(rid_list, 'rid_list')
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
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
    def get_roi_chip_thumbtup(ibs, rid_list):
        roi_uuid_list = ibs.get_roi_uuids(rid_list)
        # PSA: Do not use ensurepath here. Move to an initialcheck on creation
        # and save thumb_dpath as an IBEIS path
        thumb_gpaths = [join(ibs.thumb_dpath, str(uuid) + 'chip_thumb.png')
                        for uuid in roi_uuid_list]
        image_paths = ibs.get_roi_cpaths(rid_list)
        thumbtup_list = [(thumb_path, img_path, [])
                         for (thumb_path, img_path) in
                         izip(thumb_gpaths, image_paths)]
        return thumbtup_list

    @utool.accepts_numpy
    @getter_1toM
    def get_roi_kpts(ibs, rid_list, ensure=True):
        """ Returns chip keypoints """
        fid_list  = ibs.get_roi_fids(rid_list, ensure=ensure)
        kpts_list = ibs.get_feat_kpts(fid_list)
        return kpts_list

    @getter_1to1
    def get_roi_chipsizes(ibs, rid_list, ensure=True):
        cid_list  = ibs.get_roi_cids(rid_list, ensure=ensure)
        chipsz_list = ibs.get_chip_sizes(cid_list)
        return chipsz_list

    @getter_1toM
    def get_roi_desc(ibs, rid_list, ensure=True):
        """ Returns chip descriptors """
        fid_list  = ibs.get_roi_fids(rid_list, ensure=ensure)
        desc_list = ibs.get_feat_desc(fid_list)
        return desc_list

    @getter_1to1
    def get_roi_cpaths(ibs, rid_list):
        """ Returns cpaths defined by ROIs """
        utool.assert_all_not_None(rid_list, 'rid_list')
        cfpath_list = preproc_chip.get_roi_cfpath_list(ibs, rid_list)
        return cfpath_list

    @getter_1to1
    def get_roi_names(ibs, rid_list, distinguish_unknowns=True):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        nid_list  = ibs.get_roi_nids(rid_list)
        name_list = ibs.get_names(nid_list, distinguish_unknowns=distinguish_unknowns)
        return name_list

    @getter_1toM
    def get_roi_groundtruth(ibs, rid_list):
        """ Returns a list of rids with the same name foreach rid in rid_list.
        a set of rids belonging to the same name is called a groundtruth. A list
        of these is called a groundtruth_list. """
        nid_list  = ibs.get_roi_nids(rid_list)
        colnames = ('annot_rowid',)
        where_clause = 'label_rowid=? AND label_rowid!=? AND annot_rowid!=?'
        params_iter = [(nid, ibs.UNKNOWN_NID, rid) for nid, rid in izip(nid_list, rid_list)]
        groundtruth_list = ibs.db.get_where(AL_RELATION_TABLE, colnames, params_iter,
                                            where_clause,
                                            unpack_scalars=False)

        return groundtruth_list

    @getter_1to1
    def get_roi_num_groundtruth(ibs, rid_list):
        """ Returns number of other chips with the same name """
        return list(imap(len, ibs.get_roi_groundtruth(rid_list)))

    @getter_1to1
    def get_roi_num_feats(ibs, rid_list, ensure=False):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        nFeats_list = ibs.get_num_feats(fid_list)
        return nFeats_list

    @getter_1to1
    def get_roi_has_groundtruth(ibs, rid_list):
        numgts_list = ibs.get_roi_num_groundtruth(rid_list)
        has_gt_list = [num_gts > 0 for num_gts in numgts_list]
        return has_gt_list

    #
    # GETTERS::CHIP_TABLE

    @getter_1to1
    def get_chips(ibs, cid_list, ensure=True):
        """ Returns a list cropped images in numpy array form by their cid """
        rid_list = ibs.get_chip_rids(cid_list)
        chip_list = preproc_chip.compute_or_read_roi_chips(ibs, rid_list, ensure=ensure)
        return chip_list

    @getter_1to1
    def get_chip_rids(ibs, cid_list):
        rid_list = ibs.db.get(CHIP_TABLE, ('annot_rowid',), cid_list)
        return rid_list

    @getter_1to1
    def get_chip_paths(ibs, cid_list):
        """ Returns a list of chip paths by their rid """
        chip_fpath_list = ibs.db.get(CHIP_TABLE, ('chip_uri',), cid_list)
        return chip_fpath_list

    @getter_1to1
    def get_chip_sizes(ibs, cid_list):
        width_list  = ibs.db.get(CHIP_TABLE, ('chip_width',), cid_list)
        height_list = ibs.db.get(CHIP_TABLE, ('chip_height',), cid_list)
        chipsz_list = [size_ for size_ in izip(width_list, height_list)]
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
    # GETTERS::FEATS

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

    def get_num_feats(ibs, fid_list):
        """ Returns the number of keypoint / descriptor pairs """
        nFeats_list = ibs.db.get(FEATURE_TABLE, ('feature_num_feats',), fid_list)
        nFeats_list = [ (-1 if nFeats is None else nFeats) for nFeats in nFeats_list]
        return nFeats_list

    #
    # GETTERS: CONFIG
    @getter_1to1
    def get_config_rowid_from_suffix(ibs, cfgsuffix_list, ensure=True):
        """
        Adds an algorithm configuration as a string
        """
        # FIXME: cfgsuffix should be renamed cfgstr? cfgtext?
        if ensure:
            return ibs.add_config(cfgsuffix_list)
        configid_list = ibs.db.get(CONFIG_TABLE, ('config_rowid',), cfgsuffix_list,
                                   id_colname='config_suffix')

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
    # GETTERS::MASK

    @getter_1to1
    def get_roi_masks(ibs, rid_list, ensure=True):
        """ Returns segmentation masks for an roi """
        roi_list  = ibs.get_roi_bboxes(rid_list)
        mask_list = [np.empty((w, h)) for (x, y, w, h) in roi_list]
        raise NotImplementedError('FIXME!')
        return mask_list

    #
    # GETTERS::NAME
    @getter_1to1
    def get_name_nids(ibs, name_list, ensure=True):
        """ Returns nid_list. Creates one if it doesnt exist """
        if ensure:
            ibs.add_names(name_list)
        nid_list = ibs.get_labelids_from_values(name_list, ibs.INDIVIDUAL_KEY)

        return nid_list

    @getter_1to1
    def get_names(ibs, nid_list, distinguish_unknowns=True):
        """ Returns text names """
        #print('get_names: %r' % nid_list)
        # Change the temporary negative indexes back to the unknown NID for the
        # SQL query. Then augment the name list to distinguish unknown names
        nid_list_  = [nid if nid is not None and nid > 0
                      else ibs.UNKNOWN_NID
                      for nid in nid_list]
        # <TESTS>
        key_rowid_list = ibs.get_label_keys(nid_list_)
        assert all([key == ibs.INDIVIDUAL_KEY
                    for key in key_rowid_list]), 'label_rowids are not individual_ids'
        # </TESTS>
        name_list = ibs.db.get(LABEL_TABLE, ('label_value',), nid_list_)
        #name_list = ibs.get_name_props('name_text', nid_list_)
        if distinguish_unknowns:
            name_list  = [name if nid is not None and nid > 0
                          else name + str(-nid) if nid is not None else ibs.UNKNOWN_NAME
                          for (name, nid) in izip(name_list, nid_list)]
        name_list  = list(imap(__USTRCAST__, name_list))

        return name_list

    @getter_1toM
    def get_name_rids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        where_clause = 'label_rowid=?'
        params_iter = [(nid,) for nid in nid_list]
        rids_list = ibs.db.get_where(AL_RELATION_TABLE, ('annot_rowid',), params_iter,
                                     where_clause, unpack_scalars=False)
        return rids_list

    @getter_1toM
    def get_name_roi_bboxes(ibs, nid_list):
        rids_list = ibs.get_name_rids(nid_list)
        bboxes_list = ibsfuncs.unflat_map(ibs.get_roi_bboxes, rids_list)
        return bboxes_list

    @getter_1to1
    def get_name_thumbtups(ibs, nid_list):
        rids_list = ibs.get_name_rids(nid_list)
        thumbtups_list_ = ibsfuncs.unflat_map(ibs.get_roi_chip_thumbtup, rids_list)
        thumbtups_list = utool.flatten(thumbtups_list_)
        return thumbtups_list

    @getter_1to1
    def get_name_num_rois(ibs, nid_list):
        """ returns the number of detections for each name """
        return list(imap(len, ibs.get_name_rids(nid_list)))

    @getter_1to1
    def get_name_notes(ibs, nid_list):
        """ Returns name notes """
        notes_list = ibs.get_label_notes(nid_list)
        return notes_list

    #
    # GETTERS::ENCOUNTER
    @getter_1to1
    def get_encounter_num_gids(ibs, eid_list):
        """ Returns number of images in each encounter """
        return list(imap(len, ibs.get_encounter_gids(eid_list)))

    @getter_1toM
    def get_encounter_rids(ibs, eid_list):
        print('get_encounter_rids')
        print('eid_list = %r' % (eid_list,))
        """ returns a list of list of rids in each encounter """
        gids_list = ibs.get_encounter_gids(eid_list)
        print('gids_list = %r' % (gids_list,))
        rids_list_ = ibsfuncs.unflat_map(ibs.get_image_rids, gids_list)
        print('rids_list_ = %r' % (rids_list_,))
        rids_list = list(imap(utool.flatten, rids_list_))
        print('rids_list = %r' % (rids_list,))
        return rids_list

    @getter_1toM
    def get_encounter_gids(ibs, eid_list):
        #print('get_encounter_gids')
        #print('eid_list = %r' % (eid_list,))
        """ returns a list of list of gids in each encounter """
        gids_list = ibs.db.get(
            EG_RELATION_TABLE, ('image_rowid',), eid_list,
            id_colname='encounter_rowid', unpack_scalars=False)
        #print('gids_list = %r' % (gids_list,))
        return gids_list

    @getter_1toM
    def get_encounter_nids(ibs, eid_list):

        """ returns a list of list of nids in each encounter """
        print('get_encounter_nids')
        print('eid_list = %r' % (eid_list,))
        rids_list = ibs.get_encounter_rids(eid_list)
        print('rids_list = %r' % (rids_list,))
        nids_list_ = ibsfuncs.unflat_map(ibs.get_roi_nids, rids_list)
        print('nids_list_ = %r' % (nids_list_,))
        nids_list = list(imap(utool.unique_ordered, nids_list_))
        print('nids_list = %r' % (nids_list,))
        return nids_list

    @getter_1to1
    def get_encounter_enctext(ibs, eid_list):
        """ Returns encounter_text of each eid in eid_list """
        enctext_list = ibs.db.get(ENCOUNTER_TABLE, ('encounter_text',), eid_list,
                                  id_colname='encounter_rowid')
        enctext_list = list(imap(__USTRCAST__, enctext_list))
        return enctext_list

    @getter_1to1
    def get_encounter_eids(ibs, enctext_list, ensure=True):
        """ Returns a list of eids corresponding to each encounter enctext
        #TODO: make new naming scheme for non-primary-key-getters
        """
        if ensure:
            ibs.add_encounters(enctext_list)
        colnames = ('encounter_rowid',)
        eid_list = ibs.db.get(ENCOUNTER_TABLE, colnames, enctext_list,
                              id_colname='encounter_text')
        return eid_list

    @getter_1to1
    def get_egr_rowid_from_valtup(ibs, gid_list, eid_list):
        colnames = ('image_rowid',)
        params_iter = izip(gid_list, eid_list)
        where_clause = 'image_rowid=? AND encounter_rowid=?'
        egrid_list = ibs.db.get_where(EG_RELATION_TABLE, colnames, params_iter, where_clause)
        return egrid_list

    #
    #
    #-----------------
    # --- DELETERS ---
    #-----------------

    @deleter
    def delete_names(ibs, nid_list):
        """ deletes names from the database
        (CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
        ENSURE THAT NONE OF THE NIDS HAVE ANNOT_TABLE)
        """
        print('[ibs] deleting %d names' % len(nid_list))
        ibs.db.delete(LABEL_TABLE, nid_list)

    @deleter
    def delete_rois(ibs, rid_list):
        """ deletes rois from the database """
        print('[ibs] deleting %d rois' % len(rid_list))
        # Delete chips and features first
        ibs.delete_roi_chips(rid_list)
        ibs.db.delete(ANNOT_TABLE, rid_list)

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes images from the database that belong to gids"""
        print('[ibs] deleting %d images' % len(gid_list))
        # Delete rois first
        rid_list = utool.flatten(ibs.get_image_rids(gid_list))
        ibs.delete_rois(rid_list)
        ibs.db.delete(IMAGE_TABLE, gid_list)
        ibs.db.delete(EG_RELATION_TABLE, gid_list, id_colname='image_rowid')

    @deleter
    def delete_features(ibs, fid_list):
        """ deletes images from the database that belong to gids"""
        print('[ibs] deleting %d features' % len(fid_list))
        ibs.db.delete(FEATURE_TABLE, fid_list)

    @deleter
    def delete_roi_chips(ibs, rid_list):
        """ Clears roi data but does not remove the roi """
        _cid_list = ibs.get_roi_cids(rid_list, ensure=False)
        cid_list = utool.filter_Nones(_cid_list)
        ibs.delete_chips(cid_list)
        gid_list = ibs.get_roi_gids(rid_list)
        ibs.delete_image_thumbtups(gid_list)
        ibs.delete_roi_chip_thumbs(rid_list)

    @deleter
    def delete_image_thumbtups(ibs, gid_list):
        thumbtup_list = ibs.get_image_thumbtup(gid_list)
        thumbpath_list = [tup[0] for tup in thumbtup_list]
        utool.remove_file_list(thumbpath_list)

    @deleter
    def delete_roi_chip_thumbs(ibs, rid_list):
        thumbtup_list = ibs.get_roi_chip_thumbtup(rid_list)
        thumbpath_list = [tup[0] for tup in thumbtup_list]
        utool.remove_file_list(thumbpath_list)

    @deleter
    def delete_chips(ibs, cid_list):
        """ deletes images from the database that belong to gids"""
        print('[ibs] deleting %d roi-chips' % len(cid_list))
        # Delete chip-images from disk
        preproc_chip.delete_chips(ibs, cid_list)
        # Delete chip features from sql
        _fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        fid_list = utool.filter_Nones(_fid_list)
        ibs.delete_features(fid_list)
        # Delete chips from sql
        ibs.db.delete(CHIP_TABLE, cid_list)

    @deleter
    def delete_encounters(ibs, eid_list):
        """ Removes encounters (but not any other data) """
        print('[ibs] deleting %d encounters' % len(eid_list))
        ibs.db.delete(ENCOUNTER_TABLE, eid_list)
        ibs.db.delete(EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')

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
        # get ROIs by encounter id
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
            ibs.add_rois(detected_gids, detected_bboxes,
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
    def get_recognition_database_rids(ibs):
        """ returns persistent recognition database rois """
        drid_list = ibs.get_valid_rids()
        return drid_list

    @default_decorator
    def query_intra_encounter(ibs, qrid_list, **kwargs):
        """ _query_chips wrapper """
        drid_list = qrid_list
        qres_list = ibs._query_chips(qrid_list, drid_list, **kwargs)
        return qres_list

    @default_decorator
    def prep_qreq_encounter(ibs, qrid_list):
        """ Puts IBEIS into intra-encounter mode """
        drid_list = qrid_list
        ibs._prep_qreq(qrid_list, drid_list)

    @default_decorator('[querydb]')
    def query_database(ibs, qrid_list, **kwargs):
        """ _query_chips wrapper """
        drid_list = ibs.get_recognition_database_rids()
        qrid2_qres = ibs._query_chips(qrid_list, drid_list, **kwargs)
        return qrid2_qres

    @default_decorator
    def query_encounter(ibs, qrid_list, eid, **kwargs):
        """ _query_chips wrapper """
        drid_list = ibs.get_encounter_rids(eid)  # encounter database chips
        qrid2_qres = ibs._query_chips(qrid_list, drid_list, **kwargs)
        for qres in qrid2_qres.itervalues():
            qres.eid = eid
        return qrid2_qres

    @default_decorator
    def prep_qreq_db(ibs, qrid_list):
        """ Puts IBEIS into query database mode """
        drid_list = ibs.get_recognition_database_rids()
        ibs._prep_qreq(qrid_list, drid_list)

    @default_decorator
    def _init_query_requestor(ibs):
        # Create query request object
        ibs.qreq = QueryRequest.QueryRequest(ibs.qresdir, ibs.bigcachedir)
        ibs.qreq.set_cfg(ibs.cfg.query_cfg)

    @default_decorator
    def _prep_qreq(ibs, qrid_list, drid_list, **kwargs):
        if ibs.qreq is None:
            ibs._init_query_requestor()
        qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                      qrids=qrid_list,
                                      drids=drid_list,
                                      query_cfg=ibs.cfg.query_cfg,
                                      **kwargs)
        return qreq

    @default_decorator
    def _query_chips(ibs, qrid_list, drid_list, **kwargs):
        """
        qrid_list - query chip ids
        drid_list - database chip ids
        """
        qreq = ibs._prep_qreq(qrid_list, drid_list, **kwargs)
        # TODO: Except query error
        qrid2_qres = mc3.process_query_request(ibs, qreq)
        return qrid2_qres

    #
    #
    #--------------
    # --- MISC ---
    #--------------
    # See ibeis/dev/ibsfuncs.py
    # there is some sneaky stuff happening there
