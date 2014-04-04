"""
Module Licence and docstring

LOGIC DOES NOT LIVE HERE
THIS DEFINES THE ARCHITECTURE OF IBEIS
"""
# JON SAYS (3-24)
# I had a change of heart. I'm using tripple double quotes for comment strings
# only and tripple single quotes for python multiline strings only
from __future__ import division, print_function
# Python
from itertools import izip
from os.path import join, realpath, split
# Science
import numpy as np
# VTool
from vtool import image as gtool
# UTool
import utool
from utool import util_hash
from utool.util_iter import iflatten
from ibeis.model import Config
from ibeis.model.preproc import preproc_chip
from ibeis.model.preproc import preproc_image
from ibeis.model.preproc import preproc_feat
# IBEIS
from ibeis.control import __IBEIS_SCHEMA__
from ibeis.control import SQLDatabaseControl
from ibeis.control.__accessor_decors import (adder, setter, getter,
                                             getter_vector_output,
                                             getter_general, deleter)
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibs]', DEBUG=False)

QUIET   = utool.get_flag('--quiet')
VERBOSE = utool.get_flag('--verbose')


#
#
#-----------------
# IBEIS CONTROLLER
#-----------------

class IBEISControl(object):
    """
    IBEISController docstring
        chip  - cropped region of interest in an image, should map to one animal
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
    # --- Constructor / Privates ---
    #-------------------------------

    def __init__(ibs, dbdir=None):
        """ Creates a new IBEIS Controller associated with one database """
        if VERBOSE:
            print('[ibs.__init__] new IBEISControl')
        ibs._init_dirs(dbdir)
        ibs._init_sql()
        ibs._init_config()

    def _init_dirs(ibs, dbdir):
        """ Define ibs directories """
        print('[ibs._init_dirs] ibs.dbdir = %r' % dbdir)
        ibs.dbdir = realpath(dbdir)
        ibs.dbfname  = '_ibeis_database.sqlite3'
        ibs.cachedir = join(ibs.dbdir, '_ibeis_cache')
        ibs.chipdir  = join(ibs.cachedir, 'chips')
        ibs.flanndir = join(ibs.cachedir, 'flann')
        printDBG('[ibs._init_dirs] ibs.dbfname = %r' % ibs.dbfname)
        printDBG('[ibs._init_dirs] ibs.cachedir = %r' % ibs.cachedir)
        assert dbdir is not None, 'must specify database directory'

    def _init_sql(ibs):
        """ Load or create sql database """
        ibs.db = SQLDatabaseControl.SQLDatabaseControl(ibs.dbdir, ibs.dbfname)
        printDBG('[ibs._init_sql] Define the schema.')
        __IBEIS_SCHEMA__.define_IBEIS_schema(ibs)
        printDBG('[ibs._init_sql] Add default names.')
        ibs.UNKNOWN_NAME = '____'
        ibs.UNKNOWN_NID = ibs.get_name_nids((ibs.UNKNOWN_NAME,), ensure=True)[0]
        try:
            assert ibs.UNKNOWN_NID == 1
        except AssertionError:
            print('[!ibs] ERROR: ibs.UNKNOWN_NID = %r' % ibs.UNKNOWN_NID)
            raise

    def _init_config(ibs):
        """ Loads the database's algorithm configuration """
        printDBG('[ibs] _load_config()')
        ibs.cfg = Config.ConfigBase('cfg', fpath=join(ibs.dbdir, 'cfg'))
        if not ibs.cfg.load() is True:
            ibs._default_config()

    def _default_config(ibs):
        """ Resets the databases's algorithm configuration """
        printDBG('[ibs] _default_config()')
        # TODO: Detector config
        ibs.cfg.chip_cfg   = Config.default_chip_cfg()
        ibs.cfg.feat_cfg   = Config.default_feat_cfg(ibs)
        ibs.cfg.query_cfg  = Config.default_query_cfg(ibs)

    #
    #
    #---------------
    # --- Adders ---
    #---------------

    @adder
    def add_images(ibs, gpath_list):
        """ Adds a list of image paths to the database. Returns gids """
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))
        # Build parameter list early so we can grab the gids
        param_list = [tup for tup in
                      preproc_image.add_images_paramters_gen(gpath_list)]
        gid_list   = [tup[0] for tup in param_list]
        ibs.db.executemany(
            operation='''
            INSERT or IGNORE INTO images(
                image_uid,
                image_uri,
                image_width,
                image_height,
                image_exif_time_posix,
                image_exif_gps_lat,
                image_exif_gps_lon
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            parameters_iter=param_list)
        return gid_list

    @adder
    def add_rois(ibs, gid_list, bbox_list, theta_list,
                 viewpoint_list=None, nid_list=None):
        """ Adds oriented ROI bounding boxes to images """
        if viewpoint_list is None:
            viewpoint_list = ['UNKNOWN' for _ in xrange(len(gid_list))]
        if nid_list is None:
            nid_list = [ibs.UNKNOWN_NID for _ in xrange(len(gid_list))]
        # Build deterministic and unique ROI ids
        rid_list = [util_hash.augment_uuid(gid, bbox, theta)
                    for gid, bbox, theta
                    in izip(gid_list, bbox_list, theta_list)]
        # Define arguments to insert
        param_iter = ((rid, gid, nid, x, y, w, h, theta, viewpoint)
                      for (rid, gid, nid, (x, y, w, h), theta, viewpoint)
                      in izip(rid_list, gid_list, nid_list, bbox_list,
                              theta_list, viewpoint_list))
        ibs.db.executemany(
            operation='''
            INSERT OR REPLACE INTO rois
            (
                roi_uid,
                image_uid,
                name_uid,
                roi_xtl,
                roi_ytl,
                roi_width,
                roi_height,
                roi_theta,
                roi_viewpoint
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            parameters_iter=param_iter)
        return rid_list

    @adder
    def add_chips(ibs, rid_list):
        """ Adds chip data to the ROI. (does not create ROIs. first use add_rois
        and then pass them here to ensure chips are computed)
        return cid_list
        """
        cid_list = ibs.get_roi_cids(rid_list, ensure=False)
        dirty_rids = utool.get_dirty_items(rid_list, cid_list)
        if len(dirty_rids) > 0:
            preproc_chip.compute_and_write_chips_lazy(ibs, rid_list)
            param_iter = preproc_chip.add_chips_parameters_gen(ibs, dirty_rids)
            ibs.db.executemany(
                operation='''
                INSERT OR IGNORE
                INTO chips
                (
                    chip_uid,
                    roi_uid,
                    chip_width,
                    chip_height
                )
                VALUES (NULL, ?, ?, ?)
                ''',
                parameters_iter=param_iter)
            cid_list = ibs.get_roi_cids(rid_list)
        return cid_list

    @adder
    def add_feats(ibs, cid_list):
        """ Computes the features for every chip without them """
        fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        dirty_cids = utool.get_dirty_items(cid_list, fid_list)
        if len(dirty_cids) > 0:
            param_iter = preproc_feat.add_feat_params_gen(ibs, dirty_cids)
            ibs.db.executemany(
                operation='''
                INSERT OR IGNORE
                INTO features
                (
                    feature_uid,
                    chip_uid,
                    feature_keypoints,
                    feature_sifts
                )
                VALUES (NULL, ?, ?, ?)
                ''',
                parameters_iter=(tup for tup in param_iter))
            fid_list = ibs.get_chip_fids(cid_list, ensure=False)
        return fid_list

    @adder
    def add_names(ibs, name_list):
        """ Adds a list of names. Returns their nids """
        nid_list = ibs.get_name_nids(name_list, ensure=False)
        dirty_names = utool.get_dirty_items(name_list, nid_list)
        if len(dirty_names) > 0:
            ibs.db.executemany(
                operation='''
                INSERT OR IGNORE
                INTO names
                (
                    name_uid,
                    name_text
                )
                VALUES (NULL, ?)
                ''',
                parameters_iter=((name,) for name in dirty_names))
            nid_list = ibs.get_name_nids(name_list, ensure=False)
        return nid_list

    #
    #
    #----------------
    # --- Setters ---
    #----------------

    # SETTERS::General

    @setter
    def set_table_properties(ibs, table, prop_key, uid_list, val_list):
        printDBG('[DEBUG] set_table_properties(table=%r, prop_key=%r)' %
                 (table, prop_key))
        # Sanatize input to be only lowercase alphabet and underscores
        table, prop_key = ibs.db.sanatize_sql(table, prop_key)
        # Potentially UNSAFE SQL
        ibs.db.executemany(
            operation='''
            UPDATE ''' + table + '''
            SET ''' + prop_key + '''=?
            WHERE ''' + table[:-1] + '''_uid=?
            ''',
            parameters_iter=(tup for tup in izip(val_list, uid_list)),
            errmsg='[ibs.set_table_properties] ERROR (table=%r, prop_key=%r)' %
            (table, prop_key))

    # SETTERS::Image

    @setter
    def set_image_paths(ibs, gid_list, gpath_list):
        """ Do we want to do caching here? """
        pass

    @setter
    def set_image_eid(ibs, gid_list, eids_list):
        """ Sets the encounter id that a list of images is tied to, deletes old
        encounters.  eid_list is a list of tuples, each represents the set of
        encounters a tuple should belong to.
        """
        ibs.db.executemany(
            operation='''
            DELETE FROM egpairs WHERE image_uid=?
            ''',
            parameters_iter=gid_list)

        ibs.db.executemany(
            operation='''
            INSERT OR IGNORE INTO egpairs(
                encounter_uid,
                image_uid
            ) VALUES (?, ?)'
            ''',
            parameters_iter=iflatten(((eid, gid) for eid in eids) for eids, gid
                                     in izip(eids_list, gid_list)))

    # SETTERS::ROI

    @setter
    def set_roi_properties(ibs, rid_list, key, value_list):
        if key == 'bbox':
            return ibs.set_roi_bbox(rid_list, value_list)
        elif key == 'theta':
            return ibs.set_roi_thetas(rid_list, value_list)
        elif key == 'name':
            return ibs.set_roi_names(rid_list, value_list)
        elif key == 'viewpoint':
            return ibs.set_roi_viewpoints(rid_list, value_list)
        else:
            raise KeyError('[ibs.set_roi_properties] UNKOWN key=%r' % (key,))

    @setter
    def set_roi_bbox(ibs, rid_list, bbox_list):
        """ Sets ROIs of a list of rois by rid, where roi_list is a list of
            (x, y, w, h) tuples """
        ibs.db.executemany(
            operation='''
            UPDATE rois SET
                roi_xtl=?,
                roi_ytl=?,
                roi_width=?,
                roi_height=?
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(bbox_list, rid_list))

    @setter
    def set_roi_thetas(ibs, rid_list, theta_list):
        """ Sets thetas of a list of chips by rid """
        ibs.db.executemany(
            operation='''
            UPDATE rois SET
                roi_theta=?,
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(theta_list, rid_list))

    @setter
    def set_roi_viewpoints(ibs, rid_list, viewpoint_list):
        """ Sets viewpoints of a list of chips by rid """
        ibs.db.executemany(
            operation='''
            UPDATE rois
            SET
                roi_viewpoint=?,
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(viewpoint_list, rid_list))

    @setter
    def set_roi_names(ibs, rid_list, name_list):
        """ Sets names of a list of chips by cid """
        nid_list = ibs.get_name_nids(name_list)
        ibs.set_table_properties('rois', 'name_uid', rid_list, nid_list)

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    #
    # GETTERS::General

    def get_valid_ids(ibs, tblname):
        get_valid_tblname_ids = {
            'images': ibs.get_valid_gids,
            'rois': ibs.get_valid_rids,
            'names': ibs.get_valid_nids,
        }[tblname]
        return get_valid_tblname_ids()

    def get_table_properties(ibs, table, prop_key, uid_list):
        printDBG('[DEBUG] get_table_properties(table=%r, prop_key=%r)' %
                 (table, prop_key))
        # Sanatize input to be only lowercase alphabet and underscores
        table, prop_key = ibs.db.sanatize_sql(table, prop_key)
        # Potentially UNSAFE SQL
        property_list = ibs.db.executemany(
            operation='''
            SELECT ''' + prop_key + '''
            FROM ''' + table + '''
            WHERE ''' + table[:-1] + '''_uid=?
            ''',
            parameters_iter=((_uid,) for _uid in uid_list),
            errmsg='[ibs.get_table_properties] ERROR (table=%r, prop_key=%r)' %
            (table, prop_key))
        return property_list

    def get_chip_properties(ibs, prop_key, cid_list):
        """ general chip property getter """
        return ibs.get_table_properties('chips', prop_key, cid_list)

    def get_image_properties(ibs, prop_key, gid_list):
        """ general image property getter """
        return ibs.get_table_properties('images', prop_key, gid_list)

    def get_roi_properties(ibs, prop_key, rid_list):
        """ general image property getter """
        return ibs.get_table_properties('rois', prop_key, rid_list)

    def get_name_properties(ibs, prop_key, nid_list):
        """ general name property getter """
        return ibs.get_table_properties('names', prop_key, nid_list)

    #
    # GETTERS::Image

    @getter_general
    def get_valid_gids(ibs):
        gid_list = ibs.db.executeone(
            operation='''
            SELECT image_uid
            FROM images
            ''')
        return gid_list

    @getter
    def get_images(ibs, gid_list):
        """ Returns a list of images in numpy matrix form by gid """
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    @getter
    def get_image_uris(ibs, gid_list):
        """ Returns a list of image uris by gid """
        uri_list = ibs.db.executemany(
            operation='''
            SELECT image_uri
            FROM images
            WHERE image_uid=?
            ''',
            parameters_iter=((gid,) for gid in gid_list),
            unpack_scalars=True)
        return uri_list

    @getter
    def get_image_paths(ibs, gid_list):
        """ Returns a list of image paths relative to img_dir? by gid """
        uri_list = ibs.get_image_uris(gid_list)
        utool.assert_all_not_None(uri_list, 'uri_list')
        img_dir = join(ibs.dbdir, 'images')
        gpath_list = [join(img_dir, uri) for uri in uri_list]
        return gpath_list

    @getter
    def get_image_gnames(ibs, gid_list):
        """ Returns a list of image names """
        gpath_list = ibs.get_image_paths(gid_list)
        gname_list = [split(gpath)[1] for gpath in gpath_list]
        return gname_list

    @getter
    def get_image_size(ibs, gid_list):
        """ Returns a list of (width, height) tuples """
        gwidth_list = ibs.get_image_properties('image_width', gid_list)
        gheight_list = ibs.get_image_properties('image_height', gid_list)
        gsize_list = [(w, h) for (w, h) in izip(gwidth_list, gheight_list)]
        return gsize_list

    @getter
    def get_image_unixtime(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        return ibs.get_image_properties('image_exif_time_posix', gid_list)

    @getter
    def get_image_gps(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        lat_list = ibs.get_image_properties('image_exif_gps_lat', gid_list)
        lon_list = ibs.get_image_properties('image_exif_gps_lon', gid_list)
        gps_list = [(lat, lon) for (lat, lon) in izip(lat_list, lon_list)]
        return gps_list

    @getter
    def get_image_aifs(ibs, gid_list):
        """ Returns "All Instances Found" flag, true if all objects of interest
        (animals) have an ROI in the image """
        aif_list = ibs.get_image_properties('image_toggle_aif', gid_list)
        return aif_list

    @getter
    def get_image_eid(ibs, gid_list):
        """ Returns a list of encounter ids for each image by gid """
        eid_list = [-1 for gid in gid_list]
        return eid_list

    @getter_vector_output
    def get_rids_in_gids(ibs, gid_list):
        """ Returns a list of rids for each image by gid """
        rids_list = ibs.db.executemany(
            operation='''
            SELECT roi_uid
            FROM rois
            WHERE image_uid=?
            ''',
            parameters_iter=((gid,) for gid in gid_list),
            unpack_scalars=False)
        return rids_list

    @getter
    def get_num_rids_in_gids(ibs, gid_list):
        """ Returns the number of chips in each image """
        return map(len, ibs.get_rids_in_gids(gid_list))

    #
    # GETTERS::ROI

    #@getter_general

    @getter_general
    def get_valid_rids(ibs):
        """ returns a list of vaoid ROI unique ids """
        rid_list = ibs.db.executeone(
            operation='''
            SELECT roi_uid
            FROM rois
            ''')
        return rid_list

    @getter
    def get_roi_bboxes(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        xtl_list    = ibs.get_roi_properties('roi_xtl', rid_list)
        ytl_list    = ibs.get_roi_properties('roi_ytl', rid_list)
        width_list  = ibs.get_roi_properties('roi_width', rid_list)
        height_list = ibs.get_roi_properties('roi_height', rid_list)
        bbox_list = [(x, y, w, h) for (x, y, w, h) in
                     izip(xtl_list, ytl_list, width_list, height_list)]
        return bbox_list

    @getter
    def get_roi_thetas(ibs, rid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.get_roi_properties('roi_theta', rid_list)
        return theta_list

    @getter
    def get_roi_gids(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        gid_list = ibs.get_roi_properties('image_uid', rid_list)
        return gid_list

    @getter
    def get_roi_cids(ibs, rid_list, ensure=True):
        if ensure:
            ibs.add_chips(rid_list)
        cid_list = ibs.db.executemany(
            operation='''
            SELECT chip_uid
            FROM chips
            WHERE roi_uid=?
            ''',
            parameters_iter=((rid,) for rid in rid_list))
        return cid_list

    def get_roi_fids(ibs, rid_list, ensure=False):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        return fid_list

    @getter
    def get_roi_nids(ibs, rid_list):
        """ Returns the name_ids of each roi """
        nid_list = ibs.get_roi_properties('name_uid', rid_list)
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
    def get_roi_gpaths(ibs, rid_list):
        """ Returns the image names of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        try:
            utool.assert_all_not_None(gid_list, 'gid_list')
        except AssertionError:
            print('rid_list = %r' % rid_list)
            print('gid_list = %r' % gid_list)
            raise
        gpath_list = ibs.get_image_paths(gid_list)
        utool.assert_all_not_None(gpath_list, 'gpath_list')
        return gpath_list

    @getter
    def get_roi_chips(ibs, rid_list, ensure=True):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        chip_list = ibs.get_chips(cid_list)
        return chip_list

    @getter_vector_output
    def get_roi_kpts(ibs, rid_list, ensure=True):
        """ Returns chip keypoints """
        fid_list  = ibs.get_roi_fids(rid_list, ensure=ensure)
        kpts_list = ibs.get_feat_kpts(fid_list)
        return kpts_list

    @getter_vector_output
    def get_roi_desc(ibs, rid_list, ensure=True):
        """ Returns chip descriptors """
        fid_list  = ibs.get_roi_fids(rid_list, ensure=ensure)
        desc_list = ibs.get_feat_desc(fid_list, ensure=ensure)
        return desc_list

    @getter
    def get_roi_cpaths(ibs, rid_list):
        """ Returns cpaths defined by ROIs """
        cfpath_list = preproc_chip.get_roi_cfpath_list(ibs, rid_list)
        return cfpath_list

    @getter
    def get_roi_names(ibs, rid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        nid_list  = ibs.get_roi_nids(rid_list)
        name_list = ibs.get_name_properties('name_text', nid_list)
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
            ''',
            parameters_iter=((nid,) for nid in nid_list),
            unpack_scalars=False)
        return groundtruth_list

    @getter
    def get_roi_num_groundtruth(ibs, cid_list):
        """ Returns number of other chips with the same name """
        return map(len, ibs.get_roi_groundtruth(cid_list))

    @getter_vector_output
    def get_rids_in_nids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        # for each name return chips in that name
        rids_list = [[] for _ in xrange(len(nid_list))]
        return rids_list

    @getter
    def get_num_rids_in_nids(ibs, nid_list):
        """ returns the number of detections for each name """
        return map(len, ibs.get_rids_in_nids(nid_list))

    @getter
    def get_roi_num_feats(ibs, rid_list, ensure=False):
        cid_list = ibs.get_roi_cids(rid_list, ensure=ensure)
        fid_list = ibs.get_chip_fids(cid_list, ensure=ensure)
        kpts_list = ibs.get_feat_kpts(fid_list)
        nFeats_list = [None if kpts is None else len(kpts)
                       for kpts in kpts_list]
        return nFeats_list

    #
    # GETTERS::Chips

    @getter
    def get_chips(ibs, cid_list):
        """ Returns a list cropped images in numpy array form by their cid """
        rid_list = ibs.get_chip_rids(cid_list)
        chip_list = preproc_chip.compute_or_read_roi_chips(ibs, rid_list)
        return chip_list

    @getter
    def get_chip_rids(ibs, cid_list):
        rid_list = ibs.get_chip_properties('roi_uid', cid_list)
        return rid_list

    @getter
    def get_chip_paths(ibs, cid_list):
        """ Returns a list of chip paths by their rid """
        cfpath_list = ibs.get_roi_cpaths(ibs.get_chip_rids(cid_list))
        return cfpath_list

    @getter
    def get_chip_size(ibs, cid_list):
        width_list  = ibs.get_chip_properties('chip_width', cid_list)
        height_list = ibs.get_chip_properties('chip_height', cid_list)
        size_list = (size_ for size_ in izip(width_list, height_list))
        return size_list

    @getter
    def get_chip_fids(ibs, cid_list, ensure=True):
        if ensure:
            ibs.add_feats(cid_list)
        fid_list = ibs.db.executemany(
            operation='''
            SELECT feature_uid
            FROM features
            WHERE chip_uid=?
            ''',
            parameters_iter=((cid,) for cid in cid_list))
        return fid_list

    def get_chip_desc(ibs, cid_list, ensure=True):
        """ Returns chip descriptors """
        fid_list = ibs.get_chip_fids(cid_list, ensure)
        desc_list = ibs.get_feat_desc(fid_list)
        return desc_list

    #
    # GETTERS::Features

    @getter_vector_output
    def get_feat_kpts(ibs, fid_list):
        """ Returns chip keypoints """
        kpts_list = ibs.db.executemany(
            operation='''
            SELECT feature_keypoints
            FROM features
            WHERE feature_uid=?
            ''',
            parameters_iter=((fid,) for fid in fid_list))
        return kpts_list

    @getter_vector_output
    def get_feat_desc(ibs, fid_list):
        """ Returns chip descriptors """
        desc_list = ibs.db.executemany(
            operation='''
            SELECT feature_sifts
            FROM features
            WHERE feature_uid=?
            ''',
            parameters_iter=((fid,) for fid in fid_list))
        return desc_list

    #
    # GETTERS::Mask

    @getter
    def get_chip_masks(ibs, rid_list):
        # Should this function exist? Yes. -Jon
        roi_list  = ibs.get_roi_bboxes(rid_list)
        mask_list = [np.empty((w, h)) for (x, y, w, h) in roi_list]
        return mask_list

    #
    # GETTERS::Name

    #@getter_general
    @getter_general
    def get_valid_nids(ibs):
        """ Returns all valid names (does not include unknown names """
        nid_list = ibs.db.executeone(
            operation='''
            SELECT name_uid
            FROM names
            WHERE name_text != ?
            ''',
            parameters=(ibs.UNKNOWN_NAME,))
        return nid_list

    @getter
    def get_names(ibs, nid_list):
        """ Returns text names """
        name_list = ibs.get_name_properties('name_text', nid_list)
        return name_list

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
            parameters_iter=((name,) for name in name_list),
            auto_commit=False)
        return nid_list

    #
    # GETTERS::Encounter

    @getter_general
    def get_valid_eids(ibs):
        """ returns list of all encounter ids """
        return []

    @getter_vector_output
    def get_rids_in_eids(ibs, eid_list):
        """ returns a list of list of rids in each encounter """
        rids_list = [[] for eid in eid_list]
        return rids_list

    @getter_vector_output
    def get_gids_in_eids(ibs, eid_list):
        """ returns a list of list of gids in each encounter """
        gids_list = [[] for eid in eid_list]
        return gids_list

    #
    #
    #-----------------
    # --- Deleters ---
    #-----------------

    @deleter
    def delete_rois(ibs, rid_list):
        """ deletes rois from the database """
        ibs.db.executemany(
            operation='''
            DELETE
            FROM rois
            WHERE roi_uid=?
            ''',
            parameters_iter=((rid,) for rid in rid_list))

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes images from the database that belong to gids"""
        ibs.db.executemany(
            operation='''
            DELETE
            FROM images
            WHERE image_uid=?
            ''',
            parameters_iter=((gid,) for gid in gid_list))

    #
    #
    #----------------
    # --- Writers ---
    #----------------

    @utool.indent_func
    def export_to_wildbook(ibs, rid_list):
        """ Exports identified chips to wildbook """
        return None

    #
    #
    #--------------
    # --- Model ---
    #--------------

    def compute_all_chips(ibs):
        print('[ibs] compute_all_chips')
        rid_list = ibs.get_valid_rids()
        cid_list = ibs.add_chips(rid_list)
        return cid_list

    def compute_all_features(ibs):
        print('[ibs] compute_all_features')
        rid_list = ibs.get_valid_rids()
        cid_list = ibs.get_roi_cids(rid_list, ensure=True)
        fid_list = ibs.add_feats(cid_list)
        return fid_list

    @utool.indent_func
    def get_empty_gids(ibs):
        gid_list = ibs.get_valid_gids()
        nRois_list = ibs.get_num_rids_in_gids(gid_list)
        empty_gids = [gid for gid, nRois in izip(gid_list, nRois_list) if nRois == 0]
        return empty_gids

    def convert_empty_images_to_rois(ibs):
        gid_list = ibs.get_empty_gids()
        rid_list = ibs.use_images_as_rois(gid_list)
        return rid_list

    @utool.indent_func
    def use_images_as_rois(ibs, gid_list):
        """ Adds an roi the size of the entire image to each image."""
        gsize_list = ibs.get_image_size(gid_list)
        bbox_list  = [(0, 0, w, h) for (w, h) in gsize_list]
        theta_list = [0.0 for _ in xrange(len(gsize_list))]
        rid_list = ibs.add_rois(gid_list, bbox_list, theta_list)
        return rid_list

    @utool.indent_func
    def cluster_encounters(ibs, gid_list):
        'Finds encounters'
        from ibeis.model import encounter_cluster
        eid_list = encounter_cluster.cluster(ibs, gid_list)
        #ibs.set_image_eids(gid_list, eid_list)
        return eid_list

    @utool.indent_func
    def detect_existence(ibs, gid_list, **kwargs):
        'Detects the probability of animal existence in each image'
        from ibeis.model import jason_detector
        probexist_list = jason_detector.detect_existence(ibs,
                                                         gid_list, **kwargs)
        # Return for user inspection
        return probexist_list

    @utool.indent_func
    def detect_rois_and_masks(ibs, gid_list, **kwargs):
        'Runs animal detection in each image'
        # Should this function just return rois and no masks???
        from ibeis.model import jason_detector
        detection_list = jason_detector.detect_rois(ibs, gid_list, **kwargs)
        # detections should be a list of [(gid, roi, theta, mask), ...] tuples
        # Return for user inspection
        return detection_list

    @utool.indent_func
    def get_recognition_database_chips(ibs):
        'returns chips which are part of the persitent recognition database'
        drid_list = None
        return drid_list

    @utool.indent_func
    def query_intra_encounter(ibs, qrid_list, **kwargs):
        """ _query_chips wrapper """
        drid_list = qrid_list
        qres_list = ibs._query_chips(ibs, qrid_list, drid_list, **kwargs)
        return qres_list

    @utool.indent_func
    def query_database(ibs, qrid_list, **kwargs):
        """ _query_chips wrapper """
        drid_list = ibs.get_recognition_database_chips()
        qres_list = ibs._query_chips(ibs, qrid_list, drid_list, **kwargs)
        return qres_list

    def _init_query_requestor(ibs):
        from ibeis.model.jon_recognition import QueryRequest
        ibs.qreq = QueryRequest.QueryRequest()  # Query Data
        ibs.qreq.set_cfg(ibs.cfg.query_cfg)

    @utool.indent_func
    def _query_chips(ibs, qcid_list, dcid_list, **kwargs):
        """
        qcid_list - query chip ids
        dcid_list - database chip ids
        """
        from ibeis.model.jon_recognition import match_chips3 as mc3
        qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                      qcids=qcid_list,
                                      dcids=dcid_list,
                                      query_cfg=ibs.cfg.query_cfg,
                                      **kwargs)
        qcid2_qres = mc3.process_query_request(ibs, qreq)
        return qcid2_qres
