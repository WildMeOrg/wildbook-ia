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
from __future__ import absolute_import, division, print_function
# Python
import atexit
import requests
import uuid
from itertools import izip, imap
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
from ibeis.control.accessor_decors import (adder, setter, getter,
                                           getter_numpy,
                                           getter_numpy_vector_output,
                                           getter_vector_output,
                                           getter_general, setter_general,
                                           deleter,
                                           default_decorator)
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibs]', DEBUG=False)


__USTRCAST__ = str  # change to unicode if needed
__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers


IMAGE_TABLE = 'images'


@atexit.register
def __cleanup():
    """ prevents flann errors (not for cleaning up individual objects) """
    global __ALL_CONTROLLERS__
    del __ALL_CONTROLLERS__


def IBEIS_ThumbnailCacheContext(ibs, uuid_list):
    """ Wrapper around vtool.image.ThumbnailCacheContext """
    thumb_size = ibs.cfg.other_cfg.thumb_size
    return gtool.ThumbnailCacheContext(uuid_list, thumb_size, appname='ibeis')


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
        ibs.flanndir    = join(ibs.cachedir, PATH_NAMES.flann)
        ibs.qresdir     = join(ibs.cachedir, PATH_NAMES.qres)
        ibs.bigcachedir = join(ibs.cachedir,  PATH_NAMES.bigcache)
        ibs.thumb_dpath = utool.get_app_resource_dir('ibeis', 'thumbs')
        if ensure:
            _verbose = True
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

    def _init_sql(ibs):
        """ Load or create sql database """
        ibs.db = sqldbc.SQLDatabaseController(ibs.get_ibsdir(), ibs.sqldb_fname)
        DB_SCHEMA.define_IBEIS_schema(ibs)
        ibs.UNKNOWN_NAME = constants.UNKNOWN_NAME
        ibs.UNKNOWN_NID = ibs.get_name_nids((ibs.UNKNOWN_NAME,), ensure=True)[0]
        try:
            assert ibs.UNKNOWN_NID == 1
        except AssertionError:
            print('[!ibs] ERROR: ibs.UNKNOWN_NID = %r' % ibs.UNKNOWN_NID)
            raise

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

    def get_num_images(ibs, **kwargs):
        gid_list = ibs.get_valid_gids(**kwargs)
        return len(gid_list)

    def get_num_rois(ibs, **kwargs):
        rid_list = ibs.get_valid_rids(**kwargs)
        return len(rid_list)

    def get_num_names(ibs, **kwargs):
        nid_list = ibs.get_valid_nids(**kwargs)
        return len(nid_list)

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
    # --- ADDERS ---
    #---------------

    @adder
    def add_config(ibs, cfgsuffix_list):
        """
        Adds an algorithm configuration as a string
        """
        # FIXME: Configs are still handled poorly
        config_rowid_list = ibs.get_config_rowid_from_suffix(cfgsuffix_list, ensure=False)
        #print('config_rowid_list %r' % (config_rowid_list,))
        #print('cfgsuffix_list %r' % (cfgsuffix_list,))
        try:
            if any([x is None or (isinstance(x, list) and len(x) == 0) for x in config_rowid_list]):
                params_iter = ((_,) for _ in cfgsuffix_list)
                tblname = 'configs'
                colname_list = ['config_suffix']
                config_rowid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
                                                     ibs.get_config_rowid_from_suffix, ensure=False)
        except Exception as ex:
            utool.printex(ex)
            utool.sys.exit(1)
        return config_rowid_list

    @adder
    def add_images(ibs, gpath_list):
        """
        Adds a list of image paths to the database.  Returns gids

        Initially we set the image_uri to exactely the given gpath.
        Later we change the uri, but keeping it the same here lets
        us process images asychronously.

        TEST CODE:
            from ibeis.dev.all_imports import *
            gpath_list = grabdata.get_test_gpaths(ndata=7) + ['doesnotexist.jpg']
        """
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))
        # Processing an image might fail, yeilding a None instead of a tup
        gpath_list = ibsfuncs.assert_and_fix_gpath_slashes(gpath_list)
        # Create param_iter
        params_list  = list(preproc_image.add_images_params_gen(gpath_list))
        # Error reporting
        print('\n'.join(
            [' ! Failed reading gpath=%r' % (gpath,) for (gpath, params)
             in izip(gpath_list, params_list) if not params]))
        # Add any unadded images
        tblname = 'images'
        colname_list = ('image_uuid', 'image_uri', 'image_original_name',
                        'image_ext', 'image_width', 'image_height',
                        'image_exif_time_posix', 'image_exif_gps_lat',
                        'image_exif_gps_lon', 'image_notes',)
        # Execute SQL Add
        gid_list = ibs.db.add_cleanly(tblname, colname_list, params_list,
                                      ibs.get_image_gids_from_uuid)
        return gid_list

    @adder
    def add_rois(ibs, gid_list, bbox_list=None, theta_list=None, viewpoint_list=None,
                 nid_list=None, name_list=None, confidence_list=None, notes_list=None,
                 roi_verts_list=None):
        """ Adds oriented ROI bounding boxes to images """
        print('[ibs] adding rois')
        # Prepare the SQL input
        assert name_list is None or nid_list is None,\
            'cannot specify both names and nids'
        assert (bbox_list is     None and roi_verts_list is not None) or \
               (bbox_list is not None and roi_verts_list is     None) ,\
            'must specify exactly one of bbox_list or vert_list'
        if theta_list is None:
            theta_list = [0.0 for _ in xrange(len(gid_list))]
        if viewpoint_list is None:
            viewpoint_list = ['UNKNOWN' for _ in xrange(len(gid_list))]
        if name_list is not None:
            nid_list = ibs.add_names(name_list)
        if nid_list is None:
            nid_list = [ibs.UNKNOWN_NID for _ in xrange(len(gid_list))]
        if confidence_list is None:
            confidence_list = [0.0 for _ in xrange(len(gid_list))]
        if notes_list is None:
            notes_list = ['' for _ in xrange(len(gid_list))]
        if roi_verts_list is None:
            roi_verts_list = geometry.verts_list_from_bboxes_list(bbox_list)
        elif bbox_list is None:
            bbox_list = geometry.bboxes_from_vert_list(roi_verts_list)

        # Build ~~deterministic?~~ random and unique ROI ids
        image_uuid_list = ibs.get_image_uuids(gid_list)
        roi_uuid_list = ibsfuncs.make_roi_uuids(image_uuid_list, bbox_list,
                                                theta_list, deterministic=False)
        roi_num_verts_list = [len(verts) for verts in roi_verts_list]
        verts_as_strings = [str(r) for r in roi_verts_list]
        assert len(roi_num_verts_list) == len(verts_as_strings)
        # Define arguments to insert
        params_iter = utool.flattenize(izip(roi_uuid_list, gid_list, nid_list,
                                            bbox_list, theta_list,
                                            roi_num_verts_list, verts_as_strings,
                                            viewpoint_list, confidence_list,
                                            notes_list))

        tblname = 'rois'
        colname_list = ['roi_uuid', 'image_rowid', 'name_rowid', 'roi_xtl',
                        'roi_ytl', 'roi_width', 'roi_height', 'roi_theta',
                        'roi_num_verts', 'roi_verts', 'roi_viewpoint',
                        'roi_detect_confidence', 'roi_notes']
        # Execute add ROIs SQL
        rid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
                                      ibs.get_roi_rids_from_uuid)

        return rid_list

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
            tblname = 'chips'
            colname_list = ['roi_rowid', 'chip_uri', 'chip_width',
                            'chip_height', 'config_rowid']
            cid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
                                            ibs.get_roi_cids, ensure=False)

        return cid_list

    @adder
    def add_feats(ibs, cid_list, force=False):
        """ Computes the features for every chip without them """
        fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        dirty_cids = utool.get_dirty_items(cid_list, fid_list)
        if len(dirty_cids) > 0:
            print('[ibs] adding %d / %d features' % (len(dirty_cids), len(cid_list)))
            params_iter = preproc_feat.add_feat_params_gen(ibs, dirty_cids)
            tblname = 'features'
            colname_list = ['chip_rowid', 'feature_num_feats', 'feature_keypoints',
                            'feature_sifts', 'config_rowid']
            fid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
                                            ibs.get_chip_fids, ensure=False)

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
            params_iter = izip(dirty_names, notes_list)
            tblname = 'names'
            colname_list = ['name_text', 'name_notes']
            new_nid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
                                              ibs.get_name_nids, ensure=False)
            new_nid_list  # this line silences warnings

            # All the names should have been ensured
            # this nid list should correspond to the input
            nid_list = ibs.get_name_nids(name_list, ensure=False)

        # # Return nids in input order
        # namenid_dict = {name: nid for name, nid in izip(name_list, nid_list)}
        # nid_list_ = [namenid_dict[name] for name in name_list_]
        return nid_list

    @adder
    def add_encounters(ibs, enctext_list):
        """ Adds a list of names. Returns their nids """
        print('[ibs] adding %d encounters' % len(enctext_list))
        # Add encounter text names to database
        notes_list = ['' for _ in xrange(len(enctext_list))]
        encounter_uuid_list = [uuid.uuid4() for _ in xrange(len(enctext_list))]
        tblname = 'encounters'
        colname_list = ['encounter_text', 'encounter_uuid', 'encounter_notes']
        params_iter = izip(enctext_list, encounter_uuid_list, notes_list)
        eid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
                                      ibs.get_encounter_eids, ensure=False)
        return eid_list

    #
    #
    #----------------
    # --- SETTERS ---
    #----------------

    # SETTERS::General
    @setter_general
    def set_table_props(ibs, table, prop_key, rowid_list, val_list):
        #OFF printDBG('------------------------')
        #OFF printDBG('set_(table=%r, prop_key=%r)' % (table, prop_key))
        #OFF printDBG('set_(rowid_list=%r, val_list=%r)' % (rowid_list, val_list))
        # Sanatize input to be only lowercase alphabet and underscores
        from operator import xor
        assert not xor(utool.isiterable(rowid_list),
                       utool.isiterable(val_list)), 'invalid mixing of iterable and scalar inputs'

        if not utool.isiterable(rowid_list) and not utool.isiterable(val_list):
            rowid_list = (rowid_list,)
            val_list = (val_list,)
        table, (prop_key,) = ibs.db.sanatize_sql(table, (prop_key,))
        ibs.db.set(table, [prop_key], val_list, rowid_list)

    # SETTERS::IMAGE

    @setter
    def set_image_uris(ibs, gid_list, new_gpath_list):
        """ Sets the image URIs to a new local path.
        This is used when localizing or unlocalizing images.
        An absolute path can either be on this machine or on the cloud
        A relative path is relative to the ibeis image cache on this machine.
        """
        ibs.set_table_props('images', 'image_uri', gid_list, new_gpath_list)

    @setter
    def set_image_aifs(ibs, gid_list, aif_list):
        """ Sets the image all instances found bit """
        ibs.set_table_props('images', 'image_toggle_aif', gid_list, aif_list)

    @setter
    def set_image_notes(ibs, gid_list, notes_list):
        """ Sets the image all instances found bit """
        ibs.set_table_props('images', 'image_notes', gid_list, notes_list)

    @setter
    def set_image_unixtime(ibs, gid_list, unixtime_list):
        """ Sets the image unixtime (does not modify exif yet) """
        ibs.set_table_props('images', 'image_exif_time_posix', gid_list, unixtime_list)

    # @setter
    # def set_image_confidence(ibs, gid_list, confidence_list):
    #     """ Sets the image detection confidence """
    #     ibs.set_table_props('images', 'image_confidence', gid_list, confidence_list)

    @setter
    def set_image_enctext(ibs, gid_list, enctext_list):
        """ Sets the encoutertext of each image """
        print('[ibs] Setting %r image encounter ids' % len(gid_list))
        eid_list = ibs.add_encounters(enctext_list)
        ibs.db.executemany(
            operation='''
            INSERT OR IGNORE INTO encounter_image_relationship(
                egpair_rowid,
                image_rowid,
                encounter_rowid
            ) VALUES (NULL, ?, ?)
            ''',
            params_iter=izip(gid_list, eid_list))
        # DOES NOT WORK
        #gid_list = ibs.db.add_cleanly(tblname, colname_list, params_iter,
        #                              get_rowid_from_uuid=(lambda gid: gid))
        return gid_list

    # SETTERS::ROI

    @setter
    def set_roi_bboxes(ibs, rid_list, bbox_list):
        """ Sets ROIs of a list of rois by rid, where roi_list is a list of
            (x, y, w, h) tuples """
        ibs.delete_roi_chips(rid_list)
        colnames = ['roi_xtl', 'roi_ytl', 'roi_width', 'roi_height']
        ibs.db.set('rois', colnames, bbox_list, rid_list)

    @setter
    def set_roi_exemplar_flag(ibs, rid_list, flag_list):
        ibs.set_table_props('rois', 'roi_exemplar_flag', rid_list, flag_list)

    @setter
    def set_roi_thetas(ibs, rid_list, theta_list):
        """ Sets thetas of a list of chips by rid """
        ibs.delete_roi_chips(rid_list)  # Changing theta redefines the chips
        ibs.set_table_props('rois', 'roi_theta', rid_list, theta_list)

    @setter
    def set_roi_num_verts(ibs, rid_list, num_verts_list):
        """ Sets the number of vertices of a chip by rid """
        ibs.set_table_props('rois', 'roi_num_verts', rid_list, num_verts_list)

    @setter
    def set_roi_verts(ibs, rid_list, verts_list):
        """ Sets the vertices [(x, y), ...] of a list of chips by rid """
        ibs.set_table_props('rois', 'roi_verts', rid_list, verts_list)

    @setter
    def set_roi_viewpoints(ibs, rid_list, viewpoint_list):
        """ Sets viewpoints of a list of chips by rid """
        ibs.set_table_props('rois', 'roi_viewpoint', rid_list, viewpoint_list)

    @setter
    def set_roi_notes(ibs, rid_list, notes_list):
        """ Sets viewpoints of a list of chips by rid """
        ibs.set_table_props('rois', 'roi_notes', rid_list, notes_list)

    @setter
    def set_roi_names(ibs, rid_list, name_list=None, nid_list=None):
        """ Sets names of a list of chips by cid """
        assert name_list is None or nid_list is None, (
            'can only specify one type of name values (nid or name) not both')
        if nid_list is None:
            assert name_list is not None
            nid_list = ibs.add_names(name_list)
        # Cannot use set_table_props for cross-table setters.
        ibs.db.set('rois', ['name_rowid'], nid_list, rid_list)

    # SETTERS::NAME

    @setter
    def set_name_notes(ibs, nid_list, notes_list):
        """ Sets notes of names (groups of animals) """
        ibs.set_table_props('names', 'name_notes', nid_list, notes_list)

    @setter
    def set_name_names(ibs, nid_list, name_list):
        """ Changes the name text. Does not affect the animals of this name """
        ibsfuncs.assert_valid_names(name_list)
        ibs.set_table_props('names', 'name_text', nid_list, name_list)

    @setter
    def set_encounter_props(ibs, eid_list, key, value_list):
        print('[ibs] set_encounter_props')
        ibs.set_table_props('encounters', key, eid_list, value_list)

    @setter
    def set_encounter_enctext(ibs, eid_list, names_list):
        """ Sets names of encounters (groups of animals) """
        ibs.set_table_props('encounters', 'encounter_text', eid_list, names_list)

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    #
    # GETTERS::GENERAL

    def get_table_props(ibs, table, prop_key, rowid_list, **kwargs):
        #OFF printDBG('get_(table=%r, prop_key=%r)' % (table, prop_key))
        # Input to table props must be a list
        if isinstance(prop_key, (str, unicode)):
            prop_key = (prop_key,)
        # Sanatize input to be only lowercase alphabet and underscores
        table, prop_key = ibs.db.sanatize_sql(table, prop_key)
        #errmsg = '[ibs.get_table_props] ERROR (table=%r, prop_key=%r)' % (table, prop_key)
        tblname = table
        colname_list = prop_key
        where_col = table[:-1] + '_rowid'
        property_list = ibs.db.get(tblname, colname_list, rowid_list,
                                    where_col=where_col,
                                    unpack_scalars=True)
        return property_list

    def get_valid_ids(ibs, tblname, eid=None):
        get_valid_tblname_ids = {
            'gids': ibs.get_valid_gids,
            'rids': ibs.get_valid_rids,
            'nids': ibs.get_valid_nids,
        }[tblname]
        return get_valid_tblname_ids(eid=eid)

    def get_chip_props(ibs, prop_key, cid_list, **kwargs):
        """ general chip property getter """
        return ibs.get_table_props('chips', prop_key, cid_list, **kwargs)

    def get_image_props(ibs, prop_key, gid_list, **kwargs):
        """ general image property getter """
        return ibs.get_table_props('images', prop_key, gid_list, **kwargs)

    def get_roi_props(ibs, prop_key, rid_list, **kwargs):
        """ general image property getter """
        return ibs.get_table_props('rois', prop_key, rid_list, **kwargs)

    def get_name_props(ibs, prop_key, nid_list, **kwargs):
        """ general name property getter """
        return ibs.get_table_props('names', prop_key, nid_list, **kwargs)

    def get_feat_props(ibs, prop_key, fid_list, **kwargs):
        """ general feature property getter """
        return ibs.get_table_props('features', prop_key, fid_list, **kwargs)

    def get_encounter_props(ibs, prop_key, gid_list, **kwargs):
        """ general image property getter """
        return ibs.get_table_props('encounters`', prop_key, gid_list, **kwargs)
    #
    # GETTERS::IMAGE

    @getter_general
    def _get_all_gids(ibs):
        tblname = IMAGE_TABLE
        colname_list = ('image_rowid',)
        all_gids = ibs.db.get_executeone(tblname, colname_list)
        return all_gids

    @getter_general
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
        return gid_list

    @getter
    def get_images(ibs, gid_list):
        """ Returns a list of images in numpy matrix form by gid """
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    @getter
    def get_image_thumbs(ibs, gid_list):
        """ Does thumbnailing """
        # Cache thumbnails
        img_uuid_list = ibs.get_image_uuids(gid_list)
        # Pass unique ids to thumbnail context
        with IBEIS_ThumbnailCacheContext(ibs, img_uuid_list) as context:
            if context.needs_compute:
                dirty_gids = context.filter_dirty_items(gid_list)
                dirty_imgs = ibs.get_images(dirty_gids)
                # Pass in any images, whos thumbnails are dirty
                context.save_dirty_thumbs_from_images(dirty_imgs)
        # The context populates thumb_list on exit
        thumb_list = context.thumb_list
        return thumb_list

    @getter
    def get_image_thumbtup(ibs, gid_list):
        """ Returns tuple of image paths, where the thumb path should go, and any bboxes """
        img_uuid_list = ibs.get_image_uuids(gid_list)
        rids_list = ibs.get_image_rids(gid_list)
        #bboxes_list = ibs.get_unflat_roi_bboxes(rids_list)  # Convinience, use full mapping instead
        bboxes_list = ibsfuncs.unflat_lookup(ibs.get_roi_bboxes, rids_list)
        # PSA: Add thumb_dpath to the ibeis dirs, so you only preform the
        # ensuredir check once.
        #thumb_dpath = utool.get_app_resource_dir('ibeis', 'thumbs')
        #utool.ensuredir(thumb_dpath)
        thumb_dpath = ibs.thumb_dpath
        thumb_gpaths = [join(thumb_dpath, str(uuid) + 'thumb.png') for uuid in img_uuid_list]
        image_paths = ibs.get_image_paths(gid_list)
        thumbtup_list = list(izip(thumb_gpaths, image_paths, bboxes_list))
        #paths_list = [(thumb_path, img_path, bbox)
        #              for (thumb_path, img_path, bbox) in
        #              izip(thumb_gpaths, image_paths, bboxes_list)]
        return thumbtup_list

    @getter
    def get_image_uuids(ibs, gid_list):
        """ Returns a list of image uuids by gid """
        image_uuid_list = ibs.get_table_props('images', 'image_uuid', gid_list)
        return image_uuid_list

    @getter
    def get_image_exts(ibs, gid_list):
        """ Returns a list of image uuids by gid """
        image_uuid_list = ibs.get_table_props('images', 'image_ext', gid_list)
        return image_uuid_list

    @getter
    def get_image_uris(ibs, gid_list):
        """ Returns a list of image uris by gid """
        tblname = 'images'
        colname_list = ('image_uri',)
        uri_list = ibs.db.get(tblname, colname_list, id_iter=gid_list)
        return uri_list

    @getter
    def get_image_gids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        tblname = 'images'
        colname_list = ('image_rowid',)
        gid_list = ibs.db.get(tblname, colname_list, uuid_list,
                                where_col='image_uuid',
                                unpack_scalars=True)
        return gid_list

    @getter
    def get_image_paths(ibs, gid_list):
        """ Returns a list of image paths relative to img_dir? by gid """
        uri_list = ibs.get_image_uris(gid_list)
        utool.assert_all_not_None(uri_list, 'uri_list', key_list=['uri_list', 'gid_list'])
        gpath_list = [join(ibs.imgdir, uri) for uri in uri_list]
        return gpath_list

    @getter
    def get_image_detectpaths(ibs, gid_list):
        """ Returns a list of image paths resized to a constant area for detection """
        new_gfpath_list = preproc_detectimg.compute_and_write_detectimg_lazy(ibs, gid_list)
        return new_gfpath_list

    @getter
    def get_image_gnames(ibs, gid_list):
        """ Returns a list of original image names """
        gname_list = ibs.get_table_props('images', 'image_original_name', gid_list)
        return gname_list

    @getter
    def get_image_sizes(ibs, gid_list):
        """ Returns a list of (width, height) tuples """
        gwidth_list = ibs.get_image_props('image_width', gid_list)
        gheight_list = ibs.get_image_props('image_height', gid_list)
        gsize_list = [(w, h) for (w, h) in izip(gwidth_list, gheight_list)]
        return gsize_list

    @getter_numpy
    def get_image_unixtime(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        return ibs.get_image_props('image_exif_time_posix', gid_list)

    @getter
    def get_image_gps(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        lat_list = ibs.get_image_props('image_exif_gps_lat', gid_list)
        lon_list = ibs.get_image_props('image_exif_gps_lon', gid_list)
        gps_list = [(lat, lon) for (lat, lon) in izip(lat_list, lon_list)]
        return gps_list

    @getter
    def get_image_aifs(ibs, gid_list):
        """ Returns "All Instances Found" flag, true if all objects of interest
        (animals) have an ROI in the image """
        aif_list = ibs.get_image_props('image_toggle_aif', gid_list)
        return aif_list

    @getter
    def get_image_confidence(ibs, gid_list):
        """ Returns image detection confidence as the max of ROI confidences """
        rids_list = ibs.get_image_rids(gid_list)
        confs_list = ibsfuncs.unflat_lookup(ibs.get_roi_confidence, rids_list)
        maxconf_list = [max(confs) if len(confs) > 0 else -1
                        for confs in confs_list]
        return maxconf_list

    @getter
    def get_image_notes(ibs, gid_list):
        """ Returns image notes """
        notes_list = ibs.get_image_props('image_notes', gid_list)
        return notes_list

    @getter
    def get_image_nids(ibs, gid_list):
        """ Returns the name ids associated with an image id """
        rids_list = ibs.get_image_rids(gid_list)
        nids_list = ibsfuncs.unflat_lookup(ibs.get_roi_nids, rids_list)
        return nids_list

    @getter
    def get_name_gids(ibs, nid_list):
        """ Returns the image ids associated with name ids"""
        rids_list = ibs.get_name_rids(nid_list)
        gids_list = ibsfuncs.unflat_lookup(ibs.get_roi_gids, rids_list)
        return gids_list

    @getter
    def get_image_eids(ibs, gid_list):
        """ Returns a list of encounter ids for each image by gid """
        tblname = 'encounter_image_relationship'
        colname_list = ('encounter_rowid',)
        eids_list = ibs.db.get(tblname, colname_list, gid_list,
                                where_col='image_rowid',
                                unpack_scalars=False)
        return eids_list

    @getter
    def get_image_enctext(ibs, gid_list):
        """ Returns a list of enctexts for each image by gid """
        eids_list = ibs.get_image_eids(gid_list)
        # TODO: maybe incorporate into a decorator?
        enctext_list = ibsfuncs.unflat_lookup(ibs.get_encounter_enctext, eids_list)
        return enctext_list

    @getter_vector_output
    def get_image_rids(ibs, gid_list):
        """ Returns a list of rids for each image by gid """
        tblname = 'rois'
        colname_list = ('roi_rowid',)
        rids_list = ibs.db.get(tblname, colname_list, gid_list,
                                where_col='image_rowid',
                                unpack_scalars=False)
        return rids_list

    @getter
    def get_image_num_rois(ibs, gid_list):
        """ Returns the number of chips in each image """
        return list(imap(len, ibs.get_image_rids(gid_list)))

    #
    # GETTERS::ROI

    @getter_general
    def _get_all_rids(ibs):
        """ returns a all ROI ids """
        tblname = 'rois'
        colname_list = ('roi_rowid',)
        all_rids = ibs.db.get_executeone(tblname, colname_list)
        return all_rids

    def get_valid_rids(ibs, eid=None, is_exemplar=False):
        """ returns a list of valid ROI unique ids """
        if eid is None:
            rid_list = ibs._get_all_rids()
        else:
            rid_list = ibs.get_encounter_rids(eid)
        if is_exemplar:
            flag_list = ibs.get_roi_exemplar_flag(rid_list)
            rid_list = utool.filter_items(rid_list, flag_list)
        return rid_list

    @getter
    def get_roi_exemplar_flag(ibs, rid_list):
        roi_uuid_list = ibs.get_table_props('rois', 'roi_exemplar_flag', rid_list)
        return roi_uuid_list

    @getter
    def get_roi_uuids(ibs, rid_list):
        """ Returns a list of image uuids by gid """
        roi_uuid_list = ibs.get_table_props('rois', 'roi_uuid', rid_list)
        return roi_uuid_list

    @getter
    def get_roi_rids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        tblname = 'rois'
        colname_list = ('roi_rowid',)
        rids_list = ibs.db.get(tblname, colname_list, uuid_list,
                                where_col='roi_uuid',
                                unpack_scalars=True)
        return rids_list

    @getter
    def get_roi_confidence(ibs, rid_list):
        """ Returns a list of roi notes """
        roi_confidence_list = ibs.get_roi_props('roi_detect_confidence', rid_list)
        return roi_confidence_list

    @getter
    def get_roi_notes(ibs, rid_list):
        """ Returns a list of roi notes """
        roi_notes_list = ibs.get_roi_props('roi_notes', rid_list)
        return roi_notes_list

    @getter_numpy_vector_output
    def get_roi_bboxes(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        prop_keys = ('roi_xtl', 'roi_ytl', 'roi_width', 'roi_height')
        bbox_list = ibs.get_roi_props(prop_keys, rid_list)
        return bbox_list

    @getter
    def get_roi_thetas(ibs, rid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.get_roi_props('roi_theta', rid_list)
        return theta_list

    @getter
    def get_roi_num_verts(ibs, rid_list):
        """ Returns the number of vertices that form the polygon of each chip"""
        num_verts_list = ibs.get_roi_props('roi_num_verts', rid_list)
        return num_verts_list

    @getter
    def get_roi_verts(ibs, rid_list):
        """ Returns the vertices that form the polygon of each chip """
        verts_list = ibs.get_roi_props('roi_verts', rid_list)
        return [eval(v) for v in verts_list]

    @getter_numpy
    def get_roi_gids(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        try:
            tblname = 'rois'
            colname_list = ('image_rowid',)
            gid_list = ibs.db.get(tblname, colname_list, rid_list,
                                    where_col='roi_rowid',
                                    unpack_scalars=True)
            utool.assert_all_not_None(gid_list, 'gid_list')
        except AssertionError as ex:
            ibsfuncs.assert_valid_rids(ibs, rid_list)
            utool.printex(ex, 'Rids must have image ids!', key_list=[
                'gid_list', 'rid_list'])
            raise
        return gid_list

    @getter
    def get_roi_cids(ibs, rid_list, ensure=True, all_configs=False):
        if ensure:
            try:
                ibs.add_chips(rid_list)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.get_roi_cids]')
                print('[!ibs.get_roi_cids] rid_list = %r' % (rid_list,))
                raise
        if all_configs:
            tblname = 'chips'
            colname_list = ('chip_rowid',)
            cid_list = ibs.db.get(tblname, colname_list, rid_list, where_col='roi_rowid')
        else:
            chip_config_rowid = ibs.get_chip_config_rowid()
            #print(chip_config_rowid)
            tblname = 'chips'
            colname_list = ('chip_rowid',)
            where_clause = 'roi_rowid=? AND config_rowid=?'
            params_iter = ((rid, chip_config_rowid) for rid in rid_list)
            cid_list = ibs.db.get(tblname, colname_list, params_iter, where_clause=where_clause)
        if ensure:
            try:
                utool.assert_all_not_None(cid_list, 'cid_list')
            except AssertionError as ex:
                valid_cids = ibs.get_valid_cids()  # NOQA
                utool.printex(ex, 'Ensured cids returned None!',
                              key_list=['rid_list', 'cid_list', 'valid_cids'])
                raise
        return cid_list

    @getter
    def get_roi_fids(ibs, rid_list, ensure=False):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        return fid_list

    @getter_numpy
    def get_roi_nids(ibs, rid_list, distinguish_uknowns=True):
        """
            Returns the name id of each roi.
            If distinguish_uknowns is True, returns negative roi rowids
            instead of unknown name id
        """
        nid_list = ibs.get_roi_props('name_rowid', rid_list)
        if distinguish_uknowns:
            tnid_list = [nid if nid != ibs.UNKNOWN_NID else -rid
                         for (nid, rid) in izip(nid_list, rid_list)]
            return tnid_list
        else:
            return nid_list

    @getter
    def get_roi_gnames(ibs, rid_list):
        """ Returns the image names of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        gname_list = ibs.get_image_gnames(gid_list)
        return gname_list

    @getter
    def get_roi_images(ibs, rid_list):
        """ Returns the images of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        image_list = ibs.get_images(gid_list)
        return image_list

    @getter
    def get_roi_image_uuids(ibs, rid_list):
        gid_list = ibs.get_roi_gids(rid_list)
        image_uuid_list = ibs.get_image_uuids(gid_list)
        return image_uuid_list

    @getter
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

    @getter
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

    @getter
    def get_roi_chip_thumbs(ibs, rid_list, ensure=False, asrgb=True):
        """ TODO: CACHING """
        roi_uuid_list = ibs.get_roi_uuids(rid_list)
        # Cache thumbnails
        with IBEIS_ThumbnailCacheContext(ibs, roi_uuid_list) as context:
            #print('len(dirty_gpaths): %r' % len(dirty_gpaths))
            if len(context.dirty_gpaths) > 0:
                dirty_rids = utool.filter_items(rid_list, context.dirty_list)
                dirty_chips = ibs.get_roi_chips(dirty_rids, ensure=ensure)
                context.save_dirty_thumbs_from_images(dirty_chips)
        thumb_list = context.thumb_list
        return thumb_list

    @getter
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

    @getter_numpy_vector_output
    def get_roi_kpts(ibs, rid_list, ensure=True):
        """ Returns chip keypoints """
        fid_list  = ibs.get_roi_fids(rid_list, ensure=ensure)
        kpts_list = ibs.get_feat_kpts(fid_list)
        return kpts_list

    def get_roi_chipsizes(ibs, rid_list, ensure=True):
        cid_list  = ibs.get_roi_cids(rid_list, ensure=ensure)
        chipsz_list = ibs.get_chip_sizes(cid_list)
        return chipsz_list

    @getter_vector_output
    def get_roi_desc(ibs, rid_list, ensure=True):
        """ Returns chip descriptors """
        fid_list  = ibs.get_roi_fids(rid_list, ensure=ensure)
        desc_list = ibs.get_feat_desc(fid_list)
        return desc_list

    @getter
    def get_roi_cpaths(ibs, rid_list):
        """ Returns cpaths defined by ROIs """
        utool.assert_all_not_None(rid_list, 'rid_list')
        cfpath_list = preproc_chip.get_roi_cfpath_list(ibs, rid_list)
        return cfpath_list

    @getter
    def get_roi_names(ibs, rid_list, distinguish_unknowns=True):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        nid_list  = ibs.get_roi_nids(rid_list)
        name_list = ibs.get_names(nid_list, distinguish_unknowns=distinguish_unknowns)
        return name_list

    @getter_vector_output
    def get_roi_groundtruth(ibs, rid_list):
        """ Returns a list of rids with the same name foreach rid in rid_list.
        a set of rids belonging to the same name is called a groundtruth. A list
        of these is called a groundtruth_list. """
        nid_list  = ibs.get_roi_nids(rid_list)
        tblname = 'rois'
        colname_list = ('roi_rowid',)
        where_clause = 'name_rowid=? AND name_rowid!=? AND roi_rowid!=?'
        params_iter = ((nid, ibs.UNKNOWN_NID, rid) for nid, rid in izip(nid_list, rid_list))
        groundtruth_list = ibs.db.get(tblname, colname_list, params_iter,
                                        where_clause=where_clause,
                                        unpack_scalars=False)

        return groundtruth_list

    @getter
    def get_roi_num_groundtruth(ibs, rid_list):
        """ Returns number of other chips with the same name """
        return list(imap(len, ibs.get_roi_groundtruth(rid_list)))

    @getter
    def get_roi_num_feats(ibs, rid_list, ensure=False):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        nFeats_list = ibs.get_num_feats(fid_list)
        return nFeats_list

    @getter
    def get_roi_has_groundtruth(ibs, rid_list):
        numgts_list = ibs.get_roi_num_groundtruth(rid_list)
        has_gt_list = [num_gts > 0 for num_gts in numgts_list]
        return has_gt_list

    #
    # GETTERS::CHIPS

    @getter_general
    def get_valid_cids(ibs):
        chip_config_rowid = ibs.get_chip_config_rowid()
        tblname = 'chips'
        colname_list = ('chip_rowid',)
        cid_list = ibs.db.get(tblname, colname_list, chip_config_rowid, where_col='config_rowid')
        return cid_list

    @getter_general
    def _get_all_cids(ibs):
        """ Returns computed chips for every configuration
            (you probably should not use this)
        """
        tblname = 'chips'
        colnames = ('chip_rowid',)
        all_cids = ibs.db.get_executeone(tblname, colnames)
        return all_cids

    @getter
    def get_chips(ibs, cid_list, ensure=True):
        """ Returns a list cropped images in numpy array form by their cid """
        rid_list = ibs.get_chip_rids(cid_list)
        chip_list = preproc_chip.compute_or_read_roi_chips(ibs, rid_list, ensure=ensure)
        return chip_list

    @getter
    def get_chip_rids(ibs, cid_list):
        rid_list = ibs.get_chip_props('roi_rowid', cid_list)
        return rid_list

    @getter
    def get_chip_paths(ibs, cid_list):
        """ Returns a list of chip paths by their rid """
        tblname = 'chips'
        colname_list = ('chip_uri',)
        chip_fpath_list = ibs.db.get(tblname, colname_list, cid_list)
        return chip_fpath_list

    @getter
    def get_chip_sizes(ibs, cid_list):
        width_list  = ibs.get_chip_props('chip_width', cid_list)
        height_list = ibs.get_chip_props('chip_height', cid_list)
        chipsz_list = [size_ for size_ in izip(width_list, height_list)]
        return chipsz_list

    @getter
    def get_chip_fids(ibs, cid_list, ensure=True):
        if ensure:
            ibs.add_feats(cid_list)
        feat_config_rowid = ibs.get_feat_config_rowid()
        tblname = 'features'
        colname_list = ('feature_rowid',)
        where_clause = 'chip_rowid=? AND config_rowid=?'
        params_iter = ((cid, feat_config_rowid) for cid in cid_list)
        fid_list = ibs.db.get(tblname, colname_list, params_iter,
                                         where_clause=where_clause)
        return fid_list

    @getter
    def get_chip_cfgids(ibs, cid_list):
        tblname = 'chips'
        colname_list = ('config_rowid',)
        cfgid_list = ibs.db.get(tblname, colname_list, cid_list)
        return cfgid_list

    @getter_numpy
    def get_chip_nids(ibs, cid_list):
        """ Returns name ids. (negative roi rowids if UNKONWN_NAME) """
        rid_list = ibs.get_chip_rids(cid_list)
        nid_list = ibs.get_roi_nids(rid_list)
        return nid_list

    def get_chip_names(ibs, cid_list):
        nid_list = ibs.get_chip_nids(cid_list)
        name_list = ibs.get_names(nid_list)
        return name_list

    #
    # GETTERS::FEATS
    @getter_general
    def get_valid_fids(ibs):
        feat_config_rowid = ibs.get_feat_config_rowid()
        tblname = 'features'
        colname_list = ('feature_rowid',)
        fid_list = ibs.db.get(tblname, colname_list, [feat_config_rowid], where_col='config_rowid')
        return fid_list

    @getter_general
    def _get_all_fids(ibs):
        """ Returns computed features for every configuration
        (you probably should not use this)"""
        tblname = 'features'
        colname_list = ('feature_rowid',)
        all_fids = ibs.db.get_executeone(tblname, colname_list)
        return all_fids

    @getter_vector_output
    def get_feat_kpts(ibs, fid_list):
        """ Returns chip keypoints in [x, y, iv11, iv21, iv22, ori] format """
        kpts_list = ibs.get_feat_props('feature_keypoints', fid_list)
        return kpts_list

    @getter_vector_output
    def get_feat_desc(ibs, fid_list):
        """ Returns chip SIFT descriptors """
        desc_list = ibs.get_feat_props('feature_sifts', fid_list)
        return desc_list

    def get_num_feats(ibs, fid_list):
        """ Returns the number of keypoint / descriptor pairs """
        nFeats_list = ibs.get_feat_props('feature_num_feats', fid_list)
        nFeats_list = [ (-1 if nFeats is None else nFeats) for nFeats in nFeats_list]
        return nFeats_list

    #
    # GETTERS: CONFIG
    @getter
    def get_config_rowid_from_suffix(ibs, cfgsuffix_list, ensure=True):
        """
        Adds an algorithm configuration as a string
        """
        if ensure:
            return ibs.add_config(cfgsuffix_list)
        tblname = 'configs'
        colname_list = ('config_rowid',)
        config_rowid_list = ibs.db.get(tblname, colname_list, cfgsuffix_list,
                                        where_col='config_suffix',
                                        unpack_scalars=True)

        # executeone always returns a list
        #if config_rowid_list is not None and len(config_rowid_list) == 1:
        #    config_rowid_list = config_rowid_list[0]
        return config_rowid_list

    @getter
    def get_config_suffixes(ibs, cfgid_list):
        """ Gets suffixes for algorithm configs """
        tblname = 'configs'
        colname_list = ('config_suffix',)
        cfgsuffix_list = ibs.db.get(tblname, colname_list, cfgid_list)
        return cfgsuffix_list

    #
    # GETTERS::MASK

    @getter
    def get_roi_masks(ibs, rid_list, ensure=True):
        """ Returns segmentation masks for an roi """
        roi_list  = ibs.get_roi_bboxes(rid_list)
        mask_list = [np.empty((w, h)) for (x, y, w, h) in roi_list]
        raise NotImplementedError('FIXME!')
        return mask_list

    #
    # GETTERS::NAME
    @getter_general
    def _get_all_known_nids(ibs):
        """ Returns all nids of known animals
            (does not include unknown names) """
        tblname = 'names'
        colname_list = ('name_rowid',)
        where_clause = 'name_text!=?'
        params = [ibs.UNKNOWN_NAME]
        all_nids = ibs.db.get_executeone_where(tblname, colname_list, where_clause, params)
        return all_nids

    @getter_general
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
        return nid_list

    @getter_general
    def get_invalid_nids(ibs):
        """ Returns all names without any animals (does not include unknown names) """
        _nid_list = ibs._get_all_known_nids()
        nRois_list = ibs.get_name_num_rois(_nid_list)
        nid_list = [nid for nid, nRois in izip(_nid_list, nRois_list)
                    if nRois <= 0]
        return nid_list

    @getter
    def get_name_nids(ibs, name_list, ensure=True):
        """ Returns nid_list. Creates one if it doesnt exist """
        if ensure:
            ibs.add_names(name_list)
        tblname = 'names'
        colname_list = ('name_rowid',)
        nid_list = ibs.db.get(tblname, colname_list, name_list,
                                where_col='name_text',
                                unpack_scalars=True)
        return nid_list

    @getter
    def get_names(ibs, nid_list, distinguish_unknowns=True):
        """ Returns text names """
        #print('get_names: %r' % nid_list)
        # Change the temporary negative indexes back to the unknown NID for the
        # SQL query. Then augment the name list to distinguish unknown names
        nid_list_  = [nid if nid is not None and nid > 0 else ibs.UNKNOWN_NID for nid in nid_list]
        name_list = ibs.db.get('names', ('name_text',), nid_list_,
                               unpack_scalars=True)
        #name_list = ibs.get_name_props('name_text', nid_list_)
        if distinguish_unknowns:
            name_list  = [name if nid is not None and nid > 0
                          else name + str(-nid) if nid is not None else ibs.UNKNOWN_NAME
                          for (name, nid) in izip(name_list, nid_list)]
        name_list  = list(imap(__USTRCAST__, name_list))
        return name_list

    @getter_vector_output
    def get_name_rids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        tblname = 'rois'
        colname_list = ('roi_rowid',)
        rids_list = ibs.db.get(tblname, colname_list, nid_list, where_col='name_rowid')
        return rids_list

    @getter_vector_output
    def get_name_roi_bboxes(ibs, nid_list):
        rids_list = ibs.get_name_rids(nid_list)
        bboxes_list = ibsfuncs.unflat_lookup(ibs.get_roi_bboxes, rids_list)
        return bboxes_list

    @getter
    def get_name_thumbtups(ibs, nid_list):
        rids_list = ibs.get_name_rids(nid_list)
        thumbtups_list_ = ibsfuncs.unflat_lookup(ibs.get_roi_chip_thumbtup, rids_list)
        thumbtups_list = utool.flatten(thumbtups_list_)
        return thumbtups_list

    @getter
    def get_name_num_rois(ibs, nid_list):
        """ returns the number of detections for each name """
        return list(imap(len, ibs.get_name_rids(nid_list)))

    @getter
    def get_name_notes(ibs, gid_list):
        """ Returns name notes """
        notes_list = ibs.get_name_props('name_notes', gid_list)
        return notes_list

    #
    # GETTERS::ENCOUNTER

    @getter_general
    def _get_all_eids(ibs):
        tblname = 'encounters'
        colname_list = ('encounter_rowid',)
        all_eids = ibs.db.get_executeone(tblname, colname_list)
        return all_eids

    @getter_general
    def get_valid_eids(ibs, min_num_gids=0):
        """ returns list of all encounter ids """
        eid_list = ibs._get_all_eids()
        if min_num_gids > 0:
            num_gids_list = ibs.get_encounter_num_gids(eid_list)
            flag_list = [num_gids >= min_num_gids for num_gids in num_gids_list]
            eid_list  = utool.filter_items(eid_list, flag_list)
        return eid_list

    @getter
    def get_encounter_num_gids(ibs, eid_list):
        """ Returns number of images in each encounter """
        return list(imap(len, ibs.get_encounter_gids(eid_list)))

    @getter_vector_output
    def get_encounter_rids(ibs, eid_list):
        """ returns a list of list of rids in each encounter """
        gids_list = ibs.get_encounter_gids(eid_list)
        rids_list_ = ibsfuncs.unflat_lookup(ibs.get_image_rids, gids_list)
        rids_list = list(imap(utool.flatten, rids_list_))
        return rids_list

    @getter_vector_output
    def get_encounter_gids(ibs, eid_list):
        """ returns a list of list of gids in each encounter """
        tblname = 'encounter_image_relationship'
        colname_list = ('image_rowid',)
        gids_list = ibs.db.get(tblname, colname_list, eid_list, where_col='encounter_rowid')
        return gids_list

    @getter_vector_output
    def get_encounter_nids(ibs, eid_list):
        """ returns a list of list of nids in each encounter """
        rids_list = ibs.get_encounter_rids(eid_list)
        nids_list_ = ibsfuncs.unflat_lookup(ibs.get_roi_nids, rids_list)
        nids_list = list(imap(utool.unique_unordered, nids_list_))
        return nids_list

    @getter
    def get_encounter_enctext(ibs, eid_list):
        """ Returns encounter_text of each eid in eid_list """
        tblname = 'encounters'
        colname_list = ('encounter_text',)
        enctext_list = ibs.db.get(tblname, colname_list, eid_list,
                                  where_col='encounter_rowid',
                                  unpack_scalars=True)
        enctext_list = list(imap(__USTRCAST__, enctext_list))
        return enctext_list

    @getter
    def get_encounter_eids(ibs, enctext_list, ensure=True):
        """ Returns a list of eids corresponding to each encounter enctext"""
        if ensure:
            ibs.add_encounters(enctext_list)
        tblname = 'encounters'
        colname_list = ('encounter_rowid',)
        eid_list = ibs.db.get(tblname, colname_list, enctext_list,
                                where_col='encounter_text',
                                unpack_scalars=True)
        return eid_list

    #
    #
    #-----------------
    # --- DELETERS ---
    #-----------------

    @deleter
    def delete_names(ibs, nid_list):
        """ deletes names from the database
        (CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
        ENSURE THAT NONE OF THE NIDS HAVE ROIS)
        """
        print('[ibs] deleting %d names' % len(nid_list))
        ibs.db.delete('names', nid_list)

    @deleter
    def delete_rois(ibs, rid_list):
        """ deletes rois from the database """
        print('[ibs] deleting %d rois' % len(rid_list))
        # Delete chips and features first
        ibs.delete_roi_chips(rid_list)
        ibs.db.delete('rois', rid_list)

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes images from the database that belong to gids"""
        print('[ibs] deleting %d images' % len(gid_list))
        # Delete rois first
        rid_list = utool.flatten(ibs.get_image_rids(gid_list))
        ibs.delete_rois(rid_list)
        ibs.db.delete('images', gid_list)
        ibs.db.delete('encounter_image_relationship', gid_list, where_col='image_rowid')

    @deleter
    def delete_features(ibs, fid_list):
        """ deletes images from the database that belong to gids"""
        print('[ibs] deleting %d features' % len(fid_list))
        ibs.db.delete('features', fid_list)

    @deleter
    def delete_roi_chips(ibs, rid_list):
        """ Clears roi data but does not remove the roi """
        _cid_list = ibs.get_roi_cids(rid_list, ensure=False)
        cid_list = utool.filter_Nones(_cid_list)
        gid_list = ibs.get_roi_gids(rid_list)
        thumbtup_list = ibs.get_image_thumbtup(gid_list)
        thumb_paths = [thumbtup[0] for thumbtup in thumbtup_list]
        delete_list = [utool.remove_file(thumb_path) for thumb_path in thumb_paths]
        ibs.delete_chips(cid_list)

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
        ibs.db.delete('chips', cid_list)

    @deleter
    def delete_encounters(ibs, eid_list):
        """ Removes encounters (but not any other data) """
        print('[ibs] deleting %d encounters' % len(eid_list))
        ibs.db.delete('encounters', eid_list)
        ibs.db.delete('encounter_image_relationship', eid_list, where_col='encounter_rowid')

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
                         confidence_list=detected_confidences)
            #ibs.set_image_confidence(detected_gids, img_confs)

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
        """ returns persitent recognition database rois """
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
        qrid2_qres = mc3.process_query_request(ibs, qreq)
        return qrid2_qres

    #
    #
    #--------------
    # --- MISC ---
    #--------------
    # See ibeis/dev/ibsfuncs.py
    # there is some sneaky stuff happening there
