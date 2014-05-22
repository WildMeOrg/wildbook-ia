"""
Module Licence and docstring

LOGIC DOES NOT LIVE HERE
THIS DEFINES THE ARCHITECTURE OF IBEIS
"""
# JON SAYS (3-24)
# I had a change of heart. I'm using tripple double quotes for comment strings
# only and tripple single quotes for python multiline strings only
from __future__ import absolute_import, division, print_function
# Python
import atexit
from itertools import izip, imap
from os.path import join, split
# Science
import numpy as np
# VTool
from vtool import image as gtool
# UTool
import utool
# IBEIS DEV
from ibeis import constants
from ibeis.dev import ibsfuncs
# IBEIS MODEL
from ibeis.model import Config
from ibeis.model.preproc import preproc_chip
from ibeis.model.preproc import preproc_image
from ibeis.model.preproc import preproc_feat
from ibeis.model.preproc import preproc_detectimg
# IBEIS
from ibeis.control import DB_SCHEMA
from ibeis.control import SQLDatabaseControl as sqldbc
from ibeis.control.accessor_decors import (adder, setter, getter,
                                           getter_numpy,
                                           getter_numpy_vector_output,
                                           getter_vector_output,
                                           getter_general, deleter)
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibs]', DEBUG=False)


__USTRCAST__ = str  # change to unicode if needed
__ALL_CONTROLLERS__ = []  # Global variable containing all created controllers


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
        import requests
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
        if ensure:
            utool.ensuredir(ibs._ibsdb)
            utool.ensuredir(ibs.cachedir, verbose=False)
            utool.ensuredir(ibs.workdir, verbose=False)
            utool.ensuredir(ibs.imgdir, verbose=False)
            utool.ensuredir(ibs.chipdir, verbose=False)
            utool.ensuredir(ibs.flanndir, verbose=False)
            utool.ensuredir(ibs.qresdir, verbose=False)
            utool.ensuredir(ibs.bigcachedir, verbose=False)
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
            ibs2.update_cfg(**kwargs)
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

    def get_workdir(ibs):
        return ibs.workdir

    def get_cachedir(ibs):
        return ibs.cachedir

    def get_detectimg_cachedir(ibs):
        return join(ibs.cachedir, constants.PATH_NAMES.detectimg)

    def get_flann_cachedir(ibs):
        return ibs.flanndir

    def get_num_images(ibs):
        gid_list = ibs.get_valid_gids()
        return len(gid_list)

    def get_num_rois(ibs):
        rid_list = ibs.get_valid_rids()
        return len(rid_list)

    def get_num_names(ibs):
        nid_list = ibs.get_valid_nids()
        return len(nid_list)

    #
    #
    #----------------
    # --- Configs ---
    #----------------

    def _init_config(ibs):
        """ Loads the database's algorithm configuration """
        try:
            ibs.cfg = Config.ConfigBase('cfg', fpath=join(ibs.dbdir, 'cfg'))
            if not ibs.cfg.load() is True:
                raise Exception('did not load')
        except Exception:
            ibs._default_config()

    def _default_config(ibs):
        """ Resets the databases's algorithm configuration """
        # TODO: Detector config
        query_cfg  = Config.default_query_cfg()
        enc_cfg    = Config.default_encounter_cfg()
        ibs.set_query_cfg(query_cfg)
        ibs.set_encounter_cfg(enc_cfg)
        ibs.set_query_cfg(query_cfg)

    @utool.indent_func
    def set_encounter_cfg(ibs, enc_cfg):
        ibs.cfg.enc_cfg = enc_cfg

    @utool.indent_func
    def set_query_cfg(ibs, query_cfg):
        if ibs.qreq is not None:
            ibs.qreq.set_cfg(query_cfg)
        ibs.cfg.query_cfg = query_cfg
        ibs.cfg.feat_cfg  = query_cfg._feat_cfg
        ibs.cfg.chip_cfg  = query_cfg._feat_cfg._chip_cfg

    @utool.indent_func
    def update_cfg(ibs, **kwargs):
        ibs.cfg.query_cfg.update_cfg(**kwargs)

    @utool.indent_func
    def get_chip_config_uid(ibs):
        chip_cfg_suffix = ibs.cfg.chip_cfg.get_uid()
        chip_cfg_uid = ibs.add_config(chip_cfg_suffix)
        return chip_cfg_uid

    @utool.indent_func
    def get_feat_config_uid(ibs):
        feat_cfg_suffix = ibs.cfg.feat_cfg.get_uid()
        feat_cfg_uid = ibs.add_config(feat_cfg_suffix)
        return feat_cfg_uid

    @utool.indent_func
    def get_query_config_uid(ibs):
        query_cfg_suffix = ibs.cfg.query_cfg.get_uid()
        query_cfg_uid = ibs.add_config(query_cfg_suffix)
        return query_cfg_uid

    @utool.indent_func
    def get_qreq_uid(ibs):
        assert ibs.qres is not None
        qreq_uid = ibs.qreq.get_uid()
        return qreq_uid

    #
    #
    #---------------
    # --- ADDERS ---
    #---------------

    def add_config(ibs, config_suffix):
        ibs.db.executeone(
            operation='''
            INSERT OR IGNORE INTO configs
            (
                config_uid,
                config_suffix
            )
            VALUES (NULL, ?)
            ''',
            params=(config_suffix,))

        config_uid = ibs.db.executeone(
            operation='''
            SELECT config_uid
            FROM configs
            WHERE config_suffix=?
            ''',
            params=(config_suffix,))
        try:
            # executeone always returns a list
            if len(config_uid) == 1:
                config_uid = config_uid[0]
        except AttributeError:
            pass
        return config_uid

    @adder
    def add_images(ibs, gpath_list):
        """
        Adds a list of image paths to the database.  Returns gids

        Initially we set the image_uri to exactely the given gpath.
        Later we change the uri, but keeping it the same here lets
        us process images asychronously.
        """
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))
        for count, gpath in enumerate(gpath_list):
            assert gpath.find('\\') == -1, (('gpath_list must not contain'
                                             'backslashes. It must be in unix'
                                             'format. Failed on %d-th gpath=%r') %
                                            (count, gpath))
        # Processing an image might fail, yeilding a None instead of a tup
        raw_param_iter = preproc_image.add_images_params_gen(gpath_list)
        # Filter out None values before passing to SQL
        param_iter = utool.ifilter_Nones(raw_param_iter)
        param_list = list(param_iter)
        #print(utool.list_str(enumerate(param_list)))
        ibs.db.executemany(
            operation='''
            INSERT or IGNORE INTO images(
                image_uid,
                image_uuid,
                image_uri,
                image_original_name,
                image_ext,
                image_width,
                image_height,
                image_exif_time_posix,
                image_exif_gps_lat,
                image_exif_gps_lon,
                image_notes
            ) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            params_iter=param_list)
        # This should solve the ordering and failure issue
        # Any gpaths that failed to insert will have None as a gid
        gid_list = ibs.db.executemany(
            operation='''
            SELECT image_uid
            FROM images
            WHERE image_uri=?
            ''',
            params_iter=[(gpath,) for gpath in gpath_list])
        num_invalid = sum((gid is None for gid in gid_list))
        if num_invalid > 0:
            print(('[ibs!] There were %d invalid gpaths ' % num_invalid) +
                  '(probably duplicates if no other error was thrown)')

        assert len(gid_list) == len(gpath_list), 'bug in add_images'
        return gid_list

    @adder
    def add_rois(ibs, gid_list, bbox_list, theta_list=None, viewpoint_list=None,
                 nid_list=None, name_list=None, notes_list=None):
        """ Adds oriented ROI bounding boxes to images """
        assert name_list is None or nid_list is None,\
            'cannot specify both names and nids'
        if theta_list is None:
            theta_list = [0.0 for _ in xrange(len(gid_list))]
        if viewpoint_list is None:
            viewpoint_list = ['UNKNOWN' for _ in xrange(len(gid_list))]
        if name_list is not None:
            nid_list = ibs.add_names(name_list)
        if nid_list is None:
            nid_list = [ibs.UNKNOWN_NID for _ in xrange(len(gid_list))]
        if notes_list is None:
            notes_list = ['' for _ in xrange(len(gid_list))]
        # Build deterministic and unique ROI ids
        image_uuid_list = ibs.get_image_uuids(gid_list)
        roi_uuid_list = ibsfuncs.make_roi_uuids(image_uuid_list, bbox_list,
                                                theta_list)
        # Define arguments to insert
        params_iter = utool.flattenize(izip(roi_uuid_list,
                                            gid_list,
                                            nid_list,
                                            bbox_list,
                                            theta_list,
                                            viewpoint_list,
                                            notes_list))
        # Insert the new ROIs into the SQL database
        ibs.db.executemany(
            operation='''
            INSERT OR REPLACE INTO rois
            (
                roi_uid,
                roi_uuid,
                image_uid,
                name_uid,
                roi_xtl,
                roi_ytl,
                roi_width,
                roi_height,
                roi_theta,
                roi_viewpoint,
                roi_notes
            )
            VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            params_iter=params_iter)
        # Get the rids of the rois that were just inserted
        rid_list = ibs.db.executemany(
            operation='''
            SELECT roi_uid
            FROM rois
            WHERE roi_uuid=?
            ''',
            params_iter=[(roi_uuid,) for roi_uuid in roi_uuid_list])
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
            try:
                # FIXME: Cant be lazy until chip config / delete issue is fixed
                preproc_chip.compute_and_write_chips(ibs, rid_list)
                #preproc_chip.compute_and_write_chips_lazy(ibs, rid_list)
                params_iter = preproc_chip.add_chips_params_gen(ibs, dirty_rids)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.add_chips]')
                print('[!ibs.add_chips] ' + utool.list_dbgstr('rid_list'))
                raise
            ibs.db.executemany(
                operation='''
                INSERT OR IGNORE
                INTO chips
                (
                    chip_uid,
                    roi_uid,
                    chip_uri,
                    chip_width,
                    chip_height,
                    config_uid
                )
                VALUES (NULL, ?, ?, ?, ?, ?)
                ''',
                params_iter=params_iter)
            # Ensure must be false, otherwise an infinite loop occurs
            cid_list = ibs.get_roi_cids(rid_list, ensure=False)
        return cid_list

    @adder
    def add_feats(ibs, cid_list, force=False):
        """ Computes the features for every chip without them """
        fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        dirty_cids = utool.get_dirty_items(cid_list, fid_list)
        if len(dirty_cids) > 0:
            params_iter = preproc_feat.add_feat_params_gen(ibs, dirty_cids)
            ibs.db.executemany(
                operation='''
                INSERT OR IGNORE
                INTO features
                (
                    feature_uid,
                    chip_uid,
                    feature_num_feats,
                    feature_keypoints,
                    feature_sifts,
                    config_uid
                )
                VALUES (NULL, ?, ?, ?, ?, ?)
                ''',
                params_iter=(tup for tup in params_iter))
            fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        return fid_list

    @adder
    def add_names(ibs, name_list):
        """ Adds a list of names. Returns their nids """
        nid_list = ibs.get_name_nids(name_list, ensure=False)
        dirty_names = utool.get_dirty_items(name_list, nid_list)
        if len(dirty_names) > 0:
            ibsfuncs.assert_valid_names(name_list)
            notes_list = ['' for _ in xrange(len(dirty_names))]
            param_iter = izip(dirty_names, notes_list)
            param_list = list(param_iter)
            ibs.db.executemany(
                operation='''
                INSERT OR IGNORE
                INTO names
                (
                    name_uid,
                    name_text,
                    name_notes
                )
                VALUES (NULL, ?, ?)
                ''',
                params_iter=param_list)
            nid_list = ibs.get_name_nids(name_list, ensure=False)
        return nid_list

    @adder
    def add_encounters(ibs, enctext_list):
        """ Adds a list of names. Returns their nids """
        notes_list = ['' for _ in xrange(len(enctext_list))]
        param_iter = izip(enctext_list, notes_list)
        param_list = list(param_iter)
        ibs.db.executemany(
            operation='''
            INSERT OR IGNORE
            INTO encounters
            (
                encounter_uid,
                encounter_text,
                encounter_notes
            )
            VALUES (NULL, ?, ?)
            ''',
            params_iter=param_list)
        eid_list = ibs.get_encounter_eids(enctext_list, ensure=False)
        return eid_list

    #
    #
    #----------------
    # --- SETTERS ---
    #----------------

    # SETTERS::General

    @setter
    def set_table_props(ibs, table, prop_key, uid_list, val_list):
        #OFF printDBG('------------------------')
        #OFF printDBG('set_(table=%r, prop_key=%r)' % (table, prop_key))
        #OFF printDBG('set_(uid_list=%r, val_list=%r)' % (uid_list, val_list))
        # Sanatize input to be only lowercase alphabet and underscores
        table, (prop_key,) = ibs.db.sanatize_sql(table, (prop_key,))
        # Potentially UNSAFE SQL
        ibs.db.executemany(
            operation='''
            UPDATE ''' + table + '''
            SET ''' + prop_key + '''=?
            WHERE ''' + table[:-1] + '''_uid=?
            ''',
            params_iter=izip(val_list, uid_list),
            errmsg='[ibs.set_table_props] ERROR (table=%r, prop_key=%r)' %
            (table, prop_key))

    # SETTERS::IMAGE

    @setter
    def set_image_props(ibs, gid_list, key, value_list):
        print('[ibs] set_image_props')
        if key == 'aif':
            return ibs.set_image_aifs(gid_list, value_list)
        if key == 'enctext':
            return ibs.set_image_enctext(gid_list, value_list)
        if key == 'notes':
            return ibs.set_image_notes(gid_list, value_list)
        else:
            raise KeyError('UNKOWN key=%r' % (key,))

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

    def set_image_unixtime(ibs, gid_list, unixtime_list):
        """ Sets the image unixtime (does not modify exif yet) """
        ibs.set_table_props('images', 'image_exif_time_posix', gid_list, unixtime_list)

    @setter
    def set_image_enctext(ibs, gid_list, enctext_list):
        """ Sets the encoutertext of each image """
        eid_list = ibs.add_encounters(enctext_list)
        ibs.db.executemany(
            operation='''
            INSERT OR IGNORE INTO egpairs(
                egpair_uid,
                image_uid,
                encounter_uid
            ) VALUES (NULL, ?, ?)
            ''',
            params_iter=izip(gid_list, eid_list))

    # SETTERS::ROI

    @setter
    def set_roi_props(ibs, rid_list, key, value_list):
        print('[ibs] set_roi_props')
        if key == 'bbox':
            return ibs.set_roi_bboxes(rid_list, value_list)
        elif key == 'theta':
            return ibs.set_roi_thetas(rid_list, value_list)
        elif key == 'name':
            return ibs.set_roi_names(rid_list, value_list)
        elif key == 'viewpoint':
            return ibs.set_roi_viewpoints(rid_list, value_list)
        elif key == 'notes':
            return ibs.set_roi_notes(rid_list, value_list)
        else:
            raise KeyError('UNKOWN key=%r' % (key,))

    @setter
    def set_roi_bboxes(ibs, rid_list, bbox_list):
        """ Sets ROIs of a list of rois by rid, where roi_list is a list of
            (x, y, w, h) tuples """
        ibs.delete_roi_chips(rid_list)
        ibs.db.executemany(
            operation='''
            UPDATE rois SET
                roi_xtl=?,
                roi_ytl=?,
                roi_width=?,
                roi_height=?
            WHERE roi_uid=?
            ''',
            params_iter=utool.flattenize(izip(bbox_list, rid_list)))

    @setter
    def set_roi_thetas(ibs, rid_list, theta_list):
        """ Sets thetas of a list of chips by rid """
        ibs.delete_roi_chips(rid_list)  # Changing theta redefines the chips
        ibs.set_table_props('rois', 'roi_theta', rid_list, theta_list)

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
        if nid_list is None:
            assert name_list is not None
            nid_list = ibs.add_names(name_list)
        # Cannot use set_table_props for cross-table setters.
        ibs.db.executemany(
            operation='''
            UPDATE rois
            SET name_uid=?
            WHERE roi_uid=?''',
            params_iter=izip(nid_list, rid_list))

    # SETTERS::NAME
    @setter
    def set_name_props(ibs, nid_list, key, value_list):
        print('[ibs] set_name_props')
        if key == 'name':
            return ibs.set_name_names(nid_list, value_list)
        elif key == 'notes':
            return ibs.set_name_notes(nid_list, value_list)
        else:
            raise KeyError('UNKOWN key=%r' % (key,))

    @setter
    def set_name_notes(ibs, nid_list, notes_list):
        """ Sets notes of names (groups of animals) """
        ibs.set_table_props('names', 'name_notes', nid_list, notes_list)

    @setter
    def set_name_names(ibs, nid_list, name_list):
        """ Changes the name text. Does not affect the animals of this name """
        ibsfuncs.assert_valid_names(name_list)
        ibs.set_table_props('names', 'name_text', nid_list, name_list)

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    #
    # GETTERS::GENERAL

    def get_table_props(ibs, table, prop_key, uid_list):
        #OFF printDBG('get_(table=%r, prop_key=%r)' % (table, prop_key))
        # Input to table props must be a list
        if isinstance(prop_key, (str, unicode)):
            prop_key = (prop_key,)
        # Sanatize input to be only lowercase alphabet and underscores
        table, prop_key = ibs.db.sanatize_sql(table, prop_key)
        # Potentially UNSAFE SQL
        property_list = ibs.db.executemany(
            operation='''
            SELECT ''' + ', '.join(prop_key) + '''
            FROM ''' + table + '''
            WHERE ''' + table[:-1] + '''_uid=?
            ''',
            params_iter=((_uid,) for _uid in uid_list),
            errmsg='[ibs.get_table_props] ERROR (table=%r, prop_key=%r)' %
            (table, prop_key))
        return list(property_list)

    def get_valid_ids(ibs, tblname, eid=None):
        get_valid_tblname_ids = {
            'gids': ibs.get_valid_gids,
            'rids': ibs.get_valid_rids,
            'nids': ibs.get_valid_nids,
        }[tblname]
        return get_valid_tblname_ids(eid=eid)

    def get_chip_props(ibs, prop_key, cid_list):
        """ general chip property getter """
        return ibs.get_table_props('chips', prop_key, cid_list)

    def get_image_props(ibs, prop_key, gid_list):
        """ general image property getter """
        return ibs.get_table_props('images', prop_key, gid_list)

    def get_roi_props(ibs, prop_key, rid_list):
        """ general image property getter """
        return ibs.get_table_props('rois', prop_key, rid_list)

    def get_name_props(ibs, prop_key, nid_list):
        """ general name property getter """
        return ibs.get_table_props('names', prop_key, nid_list)

    def get_feat_props(ibs, prop_key, fid_list):
        """ general feature property getter """
        return ibs.get_table_props('features', prop_key, fid_list)

    #
    # GETTERS::IMAGE

    @getter_general
    def _get_all_gids(ibs):
        all_gids = ibs.db.executeone(
            operation='''
            SELECT image_uid
            FROM images
            ''')
        return all_gids

    @getter_general
    def get_valid_gids(ibs, eid=None):
        if eid is None:
            gid_list = ibs._get_all_gids()
        else:
            gid_list = ibs.get_encounter_gids(eid)
        return gid_list

    @getter
    def get_images(ibs, gid_list):
        """ Returns a list of images in numpy matrix form by gid """
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

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
        uri_list = ibs.db.executemany(
            operation='''
            SELECT image_uri
            FROM images
            WHERE image_uid=?
            ''',
            params_iter=utool.tuplize(gid_list),
            unpack_scalars=True)
        return uri_list

    @getter
    def get_image_gids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        gid_list = ibs.db.executemany(
            operation='''
            SELECT image_uid
            FROM images
            WHERE image_uuid=?
            ''',
            params_iter=utool.tuplize(uuid_list),
            unpack_scalars=True)
        return gid_list

    @getter
    def get_image_paths(ibs, gid_list):
        """ Returns a list of image paths relative to img_dir? by gid """
        uri_list = ibs.get_image_uris(gid_list)
        utool.assert_all_not_None(uri_list, 'uri_list')
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
        eids_list = ibs.db.executemany(
            operation='''
            SELECT encounter_uid
            FROM egpairs
            WHERE image_uid=?
            ''',
            params_iter=utool.tuplize(gid_list),
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
        rids_list = ibs.db.executemany(
            operation='''
            SELECT roi_uid
            FROM rois
            WHERE image_uid=?
            ''',
            params_iter=utool.tuplize(gid_list),
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
        rid_list = ibs.db.executeone(
            operation='''
            SELECT roi_uid
            FROM rois
            ''')
        return rid_list

    def get_valid_rids(ibs, eid=None):
        """ returns a list of valid ROI unique ids """
        if eid is None:
            rid_list = ibs._get_all_rids()
        else:
            rid_list = ibs.get_encounter_rids(eid)
        return rid_list

    @getter
    def get_roi_uuids(ibs, rid_list):
        """ Returns a list of image uuids by gid """
        roi_uuid_list = ibs.get_table_props('rois', 'roi_uuid', rid_list)
        return roi_uuid_list

    @getter
    def get_roi_rids_from_uuid(ibs, uuid_list):
        """ Returns a list of original image names """
        rid_list = ibs.db.executemany(
            operation='''
            SELECT roi_uid
            FROM rois
            WHERE roi_uuid=?
            ''',
            params_iter=utool.tuplize(uuid_list),
            unpack_scalars=True)
        return rid_list

    @getter
    def get_roi_notes(ibs, rid_list):
        """ Returns a list of roi notes """
        roi_notes_list = ibs.get_table_props('rois', 'roi_notes', rid_list)
        return roi_notes_list

    @getter_numpy_vector_output
    def get_roi_bboxes(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        bbox_list = ibs.get_roi_props(
            ('roi_xtl', 'roi_ytl', 'roi_width', 'roi_height'), rid_list)
        return bbox_list

    @getter
    def get_roi_thetas(ibs, rid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.get_roi_props('roi_theta', rid_list)
        return theta_list

    @getter_numpy
    def get_roi_gids(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        gid_list = ibs.db.executemany(
            operation='''
            SELECT image_uid
            FROM rois
            WHERE roi_uid=?
            ''',
            params_iter=utool.tuplize(rid_list))
        try:
            utool.assert_all_not_None(gid_list, 'gid_list')
        except AssertionError as ex:
            ibsfuncs.assert_valid_rids(ibs, rid_list)
            utool.printex(ex, 'Rids must have image ids!', key_list=[
                'gid_list', 'rid_list'])
            raise
        return gid_list

    @getter
    def get_roi_cids(ibs, rid_list, ensure=True):
        if ensure:
            try:
                ibs.add_chips(rid_list)
            except AssertionError as ex:
                utool.printex(ex, '[!ibs.get_roi_cids]')
                print('[!ibs.get_roi_cids] rid_list = %r' % (rid_list,))
                raise
        chip_config_uid = ibs.get_chip_config_uid()
        cid_list = ibs.db.executemany(
            operation='''
            SELECT chip_uid
            FROM chips
            WHERE roi_uid=?
            AND config_uid=?
            ''',
            params_iter=[(rid, chip_config_uid) for rid in rid_list])
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
            If distinguish_uknowns is True, returns negative roi uids
            instead of unknown name id
        """
        nid_list = ibs.get_roi_props('name_uid', rid_list)
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
        try:
            utool.assert_all_not_None(cid_list, 'cid_list')
        except AssertionError as ex:
            utool.printex(ex, 'Invalid cid_list', key_list=[
                'ensure', 'cid_list'])
            raise
        chip_list = ibs.get_chips(cid_list)
        return chip_list

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
        """ Returns a list of rids with the same name foreach rid in rid_list"""
        nid_list  = ibs.get_roi_nids(rid_list)
        groundtruth_list = ibs.db.executemany(
            operation='''
            SELECT roi_uid
            FROM rois
            WHERE name_uid=?
            AND name_uid!=?
            AND roi_uid!=?
            ''',
            params_iter=((nid, ibs.UNKNOWN_NID, rid) for nid, rid in
                         izip(nid_list, rid_list)),
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
        chip_config_uid = ibs.get_chip_config_uid()
        cid_list = ibs.db.executeone(
            operation='''
            SELECT chip_uid
            FROM chips
            WHERE config_uid=?
            ''',
            params=(chip_config_uid,))
        return cid_list

    @getter_general
    def _get_all_cids(ibs):
        """ Returns computed chips for every configuration
            (you probably should not use this)
        """
        all_cids = ibs.db.executeone(
            operation='''
            SELECT chip_uid
            FROM chips
            ''')
        return all_cids

    @getter
    def get_chips(ibs, cid_list):
        """ Returns a list cropped images in numpy array form by their cid """
        rid_list = ibs.get_chip_rids(cid_list)
        chip_list = preproc_chip.compute_or_read_roi_chips(ibs, rid_list)
        return chip_list

    @getter
    def get_chip_rids(ibs, cid_list):
        rid_list = ibs.get_chip_props('roi_uid', cid_list)
        return rid_list

    @getter
    def get_chip_paths(ibs, cid_list):
        """ Returns a list of chip paths by their rid """
        chip_fpath_list = ibs.db.executemany(
            operation='''
            SELECT chip_uri
            FROM chips
            WHERE chip_uid=?
            ''',
            params_iter=utool.tuplize(cid_list))
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
        feat_config_uid = ibs.get_feat_config_uid()
        fid_list = ibs.db.executemany(
            operation='''
            SELECT feature_uid
            FROM features
            WHERE chip_uid=?
            AND config_uid=?
            ''',
            params_iter=((cid, feat_config_uid) for cid in cid_list))
        return fid_list

    @getter
    def get_chip_cfgids(ibs, cid_list):
        cfgid_list = ibs.db.executemany(
            operation='''
            SELECT config_uid
            FROM chips
            WHERE chip_uid=?
            ''',
            params_iter=((cid) for cid in cid_list))
        return cfgid_list

    @getter_numpy
    def get_chip_nids(ibs, cid_list):
        """ Returns name ids. (negative roi uids if UNKONWN_NAME) """
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
        feat_config_uid = ibs.get_feat_config_uid()
        fid_list = ibs.db.executeone(
            operation='''
            SELECT feature_uid
            FROM features
            WHERE config_uid=?
            ''',
            params=(feat_config_uid,))
        return fid_list

    @getter_general
    def _get_all_fids(ibs):
        """ Returns computed features for every configuration
        (you probably should not use this)"""
        all_fids = ibs.db.executeone(
            operation='''
            SELECT feature_uid
            FROM features
            ''')
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
        return nFeats_list

    #
    # GETTERS: CONFIG

    def get_config_suffixes(ibs, cfgid_list):
        """ Gets suffixes for algorithm configs """
        # TODO: This can be massively optimized with a unique if it ever gets slow
        cfgsuffix_list = ibs.db.executemany(
            operation='''
            SELECT config_suffix
            FROM configs
            WHERE config_uid=?
            ''',
            params_iter=((cfgid,) for cfgid in cfgid_list))
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
        all_nids = ibs.db.executeone(
            operation='''
            SELECT name_uid
            FROM names
            WHERE name_text != ?
            ''',
            params=(ibs.UNKNOWN_NAME,))
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
        _nid_list = ibs.db.executeone(
            operation='''
            SELECT name_uid
            FROM names
            WHERE name_text != ?
            ''',
            params=(ibs.UNKNOWN_NAME,))
        nRois_list = ibs.get_name_num_rois(_nid_list)
        nid_list = [nid for nid, nRois in izip(_nid_list, nRois_list)
                    if nRois <= 0]
        return nid_list

    @getter
    def get_name_nids(ibs, name_list, ensure=True):
        """ Returns nid_list. Creates one if it doesnt exist """
        if ensure:
            ibs.add_names(name_list)
        nid_list = ibs.db.executemany(
            operation='''
            SELECT name_uid
            FROM names
            WHERE name_text=?
            ''',
            params_iter=((name,) for name in name_list))
        return nid_list

    @getter
    def get_names(ibs, nid_list, distinguish_unknowns=True):
        """ Returns text names """
        # Change the temporary negative indexes back to the unknown NID for the
        # SQL query. Then augment the name list to distinguish unknown names
        nid_list_  = [nid if nid > 0 else ibs.UNKNOWN_NID for nid in nid_list]
        name_list  = ibs.get_name_props('name_text', nid_list_)
        if distinguish_unknowns:
            name_list  = [name if nid > 0 else name + str(-nid) for (name, nid)
                          in izip(name_list, nid_list)]
        name_list  = list(imap(__USTRCAST__, name_list))
        return name_list

    @getter_vector_output
    def get_name_rids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        rids_list = ibs.db.executemany(
            operation='''
            SELECT roi_uid
            FROM rois
            WHERE name_uid=?
            ''',
            params_iter=utool.tuplize(nid_list),
            unpack_scalars=False)
        return rids_list

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
    def get_valid_eids(ibs, min_num_gids=0):
        """ returns list of all encounter ids """
        eid_list = ibs.db.executeone(
            operation='''
            SELECT encounter_uid
            FROM encounters
            ''')
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
        gids_list = ibs.db.executemany(
            operation='''
            SELECT image_uid
            FROM egpairs
            WHERE encounter_uid=?
            ''',
            params_iter=utool.tuplize(eid_list),
            unpack_scalars=False)
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
        enctext_list = ibs.db.executemany(
            operation='''
            SELECT encounter_text
            FROM encounters
            WHERE encounter_uid=?
            ''',
            params_iter=((enctext,) for enctext in eid_list))
        enctext_list = list(imap(__USTRCAST__, enctext_list))
        return enctext_list

    @getter
    def get_encounter_eids(ibs, enctext_list, ensure=True):
        """ Returns a list of eids corresponding to each encounter enctext"""
        if ensure:
            ibs.add_encounters(enctext_list)
        eid_list = ibs.db.executemany(
            operation='''
            SELECT encounter_uid
            FROM encounters
            WHERE encounter_text=?
            ''',
            params_iter=((enctext,) for enctext in enctext_list))
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
        ibs.db.executemany(
            operation='''
            DELETE
            FROM names
            WHERE name_uid=?
            ''',
            params_iter=utool.tuplize(nid_list))

    @deleter
    def delete_rois(ibs, rid_list):
        """ deletes rois from the database """
        ibs.db.executemany(
            operation='''
            DELETE
            FROM rois
            WHERE roi_uid=?
            ''',
            params_iter=utool.tuplize(rid_list))

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes images from the database that belong to gids"""
        ibs.db.executemany(  # remove from images
            operation='''
            DELETE
            FROM images
            WHERE image_uid=?
            ''',
            params_iter=utool.tuplize(gid_list))
        ibs.db.executemany(  # remove from egpairs
            operation='''
            DELETE
            FROM egpairs
            WHERE image_uid=?
            ''',
            params_iter=utool.tuplize(gid_list))

    @deleter
    def delete_features(ibs, fid_list):
        """ deletes images from the database that belong to gids"""
        ibs.db.executemany(
            operation='''
            DELETE
            FROM features
            WHERE feature_uid=?
            ''',
            params_iter=utool.tuplize(fid_list))

    @deleter
    def delete_roi_chips(ibs, rid_list):
        """ Clears roi data but does not remove the roi """
        _cid_list = ibs.get_roi_cids(rid_list)
        cid_list = utool.filter_Nones(_cid_list)
        ibs.delete_chips(cid_list)

    @deleter
    def delete_chips(ibs, cid_list):
        """ deletes images from the database that belong to gids"""
        # Delete chip-images from disk
        preproc_chip.delete_chips(ibs, cid_list)
        # Delete chip features from sql
        _fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        fid_list = utool.filter_Nones(_fid_list)
        ibs.delete_features(fid_list)
        # Delete chips from sql
        ibs.db.executemany(
            operation='''
            DELETE
            FROM chips
            WHERE chip_uid=?
            ''',
            params_iter=utool.tuplize(cid_list))

    @deleter
    def delete_encounters(ibs, eid_list):
        """ Removes encounters (but not any other data) """
        ibs.db.executemany(  # remove from encounters
            operation='''
            DELETE
            FROM encounters
            WHERE encounter_uid=?
            ''',
            params_iter=utool.tuplize(eid_list))
        ibs.db.executemany(  # remove from egpairs
            operation='''
            DELETE
            FROM egpairs
            WHERE encounter_uid=?
            ''',
            params_iter=utool.tuplize(eid_list))

    #
    #
    #----------------
    # --- WRITERS ---
    #----------------

    @utool.indent_func
    def export_to_wildbook(ibs):
        """ Exports identified chips to wildbook """
        import ibeis.export.export_wb as wb
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

    @utool.indent_func
    def compute_encounters(ibs):
        """ Clusters images into encounters """
        from ibeis.model.preproc import preproc_encounter
        enctext_list, flat_gids = preproc_encounter.ibeis_compute_encounters(ibs)
        ibs.set_image_enctext(flat_gids, enctext_list)

    @utool.indent_func
    def detect_existence(ibs, gid_list, **kwargs):
        """ Detects the probability of animal existence in each image """
        from ibeis.model.detect import randomforest
        probexist_list = randomforest.detect_existence(ibs, gid_list, **kwargs)
        # Return for user inspection
        return probexist_list

    @utool.indent_func
    def detect_random_forest(ibs, gid_list, species, quick=True, **kwargs):
        """ Runs animal detection in each image """
        from ibeis.model.detect import randomforest
        path_list = ibs.get_image_detectpaths(gid_list)
        # TODO: Return confidence here as well
        randomforest.detect_rois(ibs, gid_list, path_list,
                                              species, quick, **kwargs)

    @utool.indent_func
    def get_recognition_database_rids(ibs):
        """ returns persitent recognition database rois """
        drid_list = ibs.get_valid_rids()
        return drid_list

    @utool.indent_func
    def query_intra_encounter(ibs, qrid_list, **kwargs):
        """ _query_chips wrapper """
        drid_list = qrid_list
        qres_list = ibs._query_chips(qrid_list, drid_list, **kwargs)
        return qres_list

    @utool.indent_func(False)
    def prep_qreq_encounter(ibs, qrid_list):
        """ Puts IBEIS into intra-encounter mode """
        drid_list = qrid_list
        ibs._prep_qreq(qrid_list, drid_list)

    @utool.indent_func((False, '[query_db]'))
    def query_database(ibs, qrid_list, **kwargs):
        """ _query_chips wrapper """
        drid_list = ibs.get_recognition_database_rids()
        qrid2_qres = ibs._query_chips(qrid_list, drid_list, **kwargs)
        return qrid2_qres

    @utool.indent_func((False, '[query_enc]'))
    def query_encounter(ibs, qrid_list, eid, **kwargs):
        """ _query_chips wrapper """
        drid_list = ibs.get_encounter_rids(eid)  # encounter database chips
        qrid2_qres = ibs._query_chips(qrid_list, drid_list, **kwargs)
        for qres in qrid2_qres.itervalues():
            qres.eid = eid
        return qrid2_qres

    @utool.indent_func(False)
    def prep_qreq_db(ibs, qrid_list):
        """ Puts IBEIS into query database mode """
        drid_list = ibs.get_recognition_database_rids()
        ibs._prep_qreq(qrid_list, drid_list)

    @utool.indent_func
    def _init_query_requestor(ibs):
        from ibeis.model.hots import QueryRequest
        # Create query request object
        ibs.qreq = QueryRequest.QueryRequest(ibs.qresdir, ibs.bigcachedir)
        ibs.qreq.set_cfg(ibs.cfg.query_cfg)

    @utool.indent_func(False)
    def _prep_qreq(ibs, qrid_list, drid_list, **kwargs):
        from ibeis.model.hots import match_chips3 as mc3
        if ibs.qreq is None:
            ibs._init_query_requestor()
        qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                      qrids=qrid_list,
                                      drids=drid_list,
                                      query_cfg=ibs.cfg.query_cfg,
                                      **kwargs)
        return qreq

    @utool.indent_func('[query]')
    def _query_chips(ibs, qrid_list, drid_list, **kwargs):
        """
        qrid_list - query chip ids
        drid_list - database chip ids
        """
        from ibeis.model.hots import match_chips3 as mc3
        qreq = ibs._prep_qreq(qrid_list, drid_list, **kwargs)
        qrid2_qres = mc3.process_query_request(ibs, qreq)
        return qrid2_qres

    #
    #
    #--------------
    # --- MISC ---
    #--------------

    def get_infostr(ibs):
        """ Returns printable database information """
        dbname = ibs.get_dbname()
        workdir = utool.unixpath(ibs.get_workdir())
        num_images = ibs.get_num_images()
        num_rois = ibs.get_num_rois()
        num_names = ibs.get_num_names()
        infostr = '''
        workdir = %r
        dbname = %r
        num_images = %r
        num_rois = %r
        num_names = %r
        ''' % (workdir, dbname, num_images, num_rois, num_names)
        return infostr

    def print_roi_table(ibs):
        """ Dumps roi table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('rois', exclude_columns=['roi_uuid']))

    def print_chip_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('chips'))

    def print_feat_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('features', exclude_columns=[
            'feature_keypoints', 'feature_sifts']))

    def print_image_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('images'))
        #, exclude_columns=['image_uid']))

    def print_name_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('names'))

    def print_config_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('configs'))

    def print_encounter_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('encounters'))

    def print_egpairs_table(ibs):
        """ Dumps chip table to stdout """
        print('\n')
        print(ibs.db.get_table_csv('egpairs'))

    def print_tables(ibs):
        ibs.print_image_table()
        ibs.print_roi_table()
        ibs.print_chip_table()
        ibs.print_feat_table()
        ibs.print_name_table()
        ibs.print_config_table()
        print('\n')

atexit.register(__cleanup)
