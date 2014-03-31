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
import functools
import re
import sys
from itertools import izip
from os.path import join, realpath, split
# Science
import numpy as np
# IBEIS
from ibeis.control import __IBEIS_SCHEMA__
from ibeis.control import SQLDatabaseControl
# VTool
from vtool import image as gtool
# UTool
import utool
from utool import util_hash, util_time
from utool.util_iter import iflatten

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[ibs]', DEBUG=False)


#-----------------
# IBEIS GLOBAL
#-----------------


UNKNOWN_NID = utool.util_hash.get_zero_uuid()
UNKNOWN_NAME = '____'


#-----------------
# IBEIS DECORATORS
#-----------------

# DECORATORS::ADDER


def adder(func):
    @utool.indent_func
    @functools.wraps(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper


# DECORATORS::SETTER


def setter(func):
    @utool.indent_func
    @functools.wraps(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper


# DECORATORS::GETTER

def getter(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    @utool.accepts_scalar_input
    @utool.indent_decor('[' + func.func_name + ']')
    @functools.wraps(func)
    def getter_wrapper1(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper1


def getter_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a homogenous list of values """
    @utool.indent_func
    @utool.accepts_scalar_input_vector_output
    @functools.wraps(func)
    def getter_wrapper3(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper3


def getter_general(func):
    """ Getter decorator for functions which has no gaurentees """
    @utool.indent_func
    @functools.wraps(func)
    def getter_wrapper2(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper2


# DECORATORS::DELETER


def deleter(func):
    @utool.indent_func
    @functools.wraps(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper

#-----------------
# IBEIS CONTROLLER
#-----------------


class IBEISControl(object):
    """
    IBEISController docstring
        chip  - cropped region of interest from an image, should map to one animal
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
    #--------------------
    # --- Constructor ---
    #--------------------

    def __init__(ibs, dbdir=None):
        print('[ibs] __init__')
        ibs.dbdir = realpath(dbdir)
        ibs.dbfname = '__ibeisdb__.sqlite3'
        print('[ibs.__init__] Open the database')
        print('[ibs.__init__] ibs.dbdir    = %r' % ibs.dbdir)
        print('[ibs.__init__] ibs.dbfname = %r' % ibs.dbfname)
        assert dbdir is not None, 'must specify database directory'
        ibs.db = SQLDatabaseControl.SQLDatabaseControl(ibs.dbdir, ibs.dbfname)
        print('[ibs.__init__] Define the schema.')
        __IBEIS_SCHEMA__.define_IBEIS_schema(ibs)
        try:
            print('[ibs.__init__] Add default names.')
            ibs.add_names((UNKNOWN_NID,), (UNKNOWN_NAME,))
        except Exception as ex:
            print('[ibs] HACKISLY IGNORING: %s, %s:' % (type(ex), ex,))
            ibs.db.get_sql_version()
            if not '--ignore' in sys.argv:
                print('use --ignore to keep going')
                raise
    #
    #
    #---------------
    # --- Adders ---
    #---------------

    @adder
    def add_images(ibs, gpath_list):
        """ Adds a list of image paths to the database. Returns newly added gids """
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))

        tag_list = ['DateTimeOriginal', 'GPSInfo']
        _TAGKEYS = gtool.get_exif_tagids(tag_list)
        _TAGDEFAULTS = (-1, (-1, -1))

        def _get_exif(pil_img):
            """ Image EXIF helper """
            (exiftime, (lat, lon)) = gtool.read_exif_tags(pil_img, _TAGKEYS, _TAGDEFAULTS)
            time = util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
            return time, lat, lon

        def _add_images_paramters_gen(gpath_list):
            """ executes sqlcmd with generated sqlvals """
            for gpath in gpath_list:
                pil_img = gtool.open_pil_image(gpath)  # Open PIL Image
                width, height  = pil_img.size        # Read width, height
                time, lat, lon = _get_exif(pil_img)  # Read exif tags
                gid = util_hash.image_uuid(pil_img)  # Read pixels ]-hash-> guid = gid
                yield (gid, gpath, width, height, time, lat, lon)

        # Build parameter list early so we can grab the gids
        param_list = [parameters for parameters in _add_images_paramters_gen(gpath_list)]
        gid_list   = [parameters[0] for paramters in param_list]

        # TODO have to handle duplicate images better here
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
    def add_rois(ibs, gid_list, bbox_list, theta_list, viewpoint_list=None,
                 nid_list=None):
        """ Adds oriented ROI bounding boxes to images """
        if viewpoint_list is None:
            viewpoint_list = ['UNKNOWN' for _ in xrange(len(gid_list))]
        if nid_list is None:
            nid_list = [UNKNOWN_NID for _ in xrange(len(gid_list))]
        augment_uuid = util_hash.augment_uuid
        # Build deterministic and unique ROI ids
        rid_list = [augment_uuid(gid, bbox, theta)
                    for gid, bbox, theta
                    in izip(gid_list, bbox_list, theta_list)]
        # Define arguments to insert
        param_iter = ((rid, gid, nid, x, y, w, h, theta, viewpoint)
                      for (rid, gid, nid, (x, y, w, h), theta, viewpoint)
                      in izip(rid_list, gid_list, nid_list, bbox_list, theta_list, viewpoint_list))
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
    def add_names(ibs, nid_iter, name_iter):
        """ Adds a list of names
        Autoinsert the defualt-unknown name into the database
        """
        ibs.db.executemany(
            operation='''
            INSERT OR IGNORE INTO names
            (
                name_uid,
                name_text
            )
            VALUES (?, ?)
            ''',
            parameters_iter=izip(nid_iter, name_iter))
        nid_iter = [-1 for _ in xrange(len(nid_iter))]
        return nid_iter

    #
    #
    #----------------
    # --- Setters ---
    #----------------

    # SETTERS::Image

    @setter
    def set_image_paths(ibs, gid_list, gpath_list):
        """ Do we want to do caching here? """
        pass

    @setter
    def set_image_eid(ibs, gid_list, eids_list):
        """
            Sets the encounter id that a list of images is tied to, deletes old encounters.
            eid_list is a list of tuples, each represents the set of encounters a tuple
            should belong to.
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
            parameters_iter=iflatten(((eid, gid)
                                      for eid in eids)
                                     for eids, gid in izip(eids_list, gid_list)))

    # SETTERS::ROI

    @setter
    def set_roi_bbox(ibs, rid_list, bbox_list):
        """ Sets ROIs of a list of rois by rid, where roi_list is a list of (x, y, w, h) tuples"""
        ibs.db.executemany(
            operation='''
            UPDATE rois SET
                roi_xtl=?,
                roi_ytl=?,
                roi_width=?,
                roi_height=?,
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
        ibs.db.executemany(
            operation='''
            UPDATE chips
            SET
            chips.name_uid=
            (
                SELECT names.name_uid
                FROM names
                WHERE name_text=?
                ORDER BY name_uid
                LIMIT 1
            ),
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(name_list, rid_list))

    # SETTERS::Chip

    @setter
    def set_chip_shape(ibs, cid_list, shape_list):
        """ Sets shape of a list of chips by cid, a list of tuples (w, h) """
        ibs.db.executemany(
            operation='''
            UPDATE chips
            SET
                chip_width=?,
                chip_height=?,
            WHERE chip_uid=?
            ''',
            parameters_iter=((w, h, cid) for ((w, h), cid) in izip(shape_list, cid_list)))

    @setter
    def set_chip_toggle_hard(ibs, cid_list, hard_list):
        """ Sets hard toggle of a list of chips by cid """
        ibs.db.executemany(
            operation='''
            UPDATE chips
            SET
                chip_toggle_hard=?,
            WHERE chip_uid=?
            ''',
            parameters_iter=izip(hard_list, cid_list))

    #
    #
    #----------------
    # --- GETTERS ---
    #----------------

    #
    # GETTERS::General

    @getter_general
    def get_valid_ids(ibs, tblname):
        get_valid_tblname_ids = {
            'images': ibs.get_valid_gids,
            'rois': ibs.get_valid_rids,
            'names': ibs.get_valid_nids,
        }[tblname]
        return get_valid_tblname_ids()

    @getter_general
    def get_table_properties(ibs, table, prop_key, uid_list):
        printDBG('[DEBUG] get_table_properties(table=%r, prop_key=%r)' % (table, prop_key))
        # Potentially UNSAFE SQL
        # Sanatize input to be only lowercase alphabet and underscores
        table    = re.sub('[^a-z_]', '', table)
        prop_key = re.sub('[^a-z_]', '', prop_key)
        valid_tables = ibs.db.get_tables()
        valid_propkeys = ibs.db.get_column_names(table)
        if not table in valid_tables:
            raise Exception('UNSAFE TABLE: table=%r' % table)
        if not prop_key in valid_propkeys:
            raise Exception('UNSAFE KEY: prop_key=%r' % prop_key)
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

    #
    # GETTERS::Image

    @getter_general
    def get_image_properties(ibs, prop_key, gid_list):
        """ general image property getter """
        return ibs.get_table_properties('images', prop_key, gid_list)

    @getter_general
    def get_valid_gids(ibs):
        ibs.db.execute(
            operation='''
            SELECT image_uid
            FROM images
            ''')
        gid_iter = ibs.db.result_iter()
        gid_list = list(gid_iter)
        return gid_list

    @getter
    def get_images(ibs, gid_list):
        """
            Returns a list of images in numpy matrix form by gid
            NO SQL REQUIRED, DEPENDS ON get_image_paths()
        """
        printDBG('[DEBUG] get_images')
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
            errmsg='[ibs.get_image_uris] ERROR',
            unpack_scalars=True)
        return uri_list

    @getter
    def get_image_paths(ibs, gid_list):
        """ Returns a list of image paths relative to img_dir? by gid """
        uri_list = ibs.get_image_uris(gid_list)
        img_dir = join(ibs.dbdir, 'images')
        gpath_list = [join(img_dir, uri) for uri in uri_list]
        return gpath_list

    @getter
    def get_image_gnames(ibs, gid_list):
        """ Returns a list of image names """
        printDBG('[DEBUG] get_image_gnames()')
        gpath_list = ibs.get_image_paths(gid_list)
        gname_list = [split(gpath)[1] for gpath in gpath_list]
        return gname_list

    @getter
    def get_image_size(ibs, gid_list):
        """ Returns a list of image dimensions by gid in (width, height) tuples """
        img_width_list = ibs.get_table_properties('images', 'image_width', gid_list)
        img_height_list = ibs.get_table_properties('images', 'image_height', gid_list)
        gsize_list = [(w, h) for (w, h) in izip(img_width_list, img_height_list)]
        return gsize_list

    @getter
    def get_image_unixtime(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        unixtime_list = ibs.get_table_properties('images', 'image_exif_time_posix', gid_list)
        return unixtime_list

    @getter
    def get_image_gps(ibs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        lat_list = ibs.get_table_properties('images', 'image_exif_gps_lat', gid_list)
        lon_list = ibs.get_table_properties('images', 'image_exif_gps_lon', gid_list)
        gps_list = [(lat, lon) for (lat, lon) in izip(lat_list, lon_list)]
        return gps_list

    @getter
    def get_image_aifs(ibs, gid_list):
        """ Returns "All Instances Found" flag, true if all objects of interest
        (animals) have an ROI in the image """
        aif_list = ibs.get_table_properties('images', 'image_toggle_aif', gid_list)
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
            errmsg='[ibs.get_image_uris] ERROR',
            unpack_scalars=False)
        # for each image return rois in that image
        return rids_list

    @getter
    def get_num_rids_in_gids(ibs, gid_list):
        """ Returns the number of chips associated with a list of images by gid """
        return map(len, ibs.get_rids_in_gids(gid_list))

    #
    # GETTERS::ROI

    @getter_general
    def get_roi_properties(ibs, prop_key, rid_list):
        """ general image property getter """
        return ibs.get_table_properties('rois', prop_key, rid_list)

    @getter_general
    def get_valid_rids(ibs):
        """ returns a list of vaoid ROI unique ids """
        ibs.db.execute(operation='''
                       SELECT roi_uid
                       FROM rois
                       ''')
        rid_list = ibs.db.result_list()
        return rid_list

    @getter
    def get_roi_bboxes(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        xtl_list    = ibs.get_roi_properties('roi_xtl', rid_list)
        ytl_list    = ibs.get_roi_properties('roi_ytl', rid_list)
        width_list  = ibs.get_roi_properties('roi_width', rid_list)
        height_list = ibs.get_roi_properties('roi_height', rid_list)
        bbox_list = [(x, y, w, h) for (x, y, w, h) in izip(xtl_list, ytl_list, width_list, height_list)]
        return bbox_list

    @getter
    def get_roi_gids(ibs, rid_list):
        """ returns roi bounding boxes in image space """
        gid_list = ibs.get_roi_properties('image_uid', rid_list)
        return gid_list

    @getter
    def get_roi_gname(ibs, rid_list):
        """ Returns the image names of each roi """
        gid_list = ibs.get_roi_gids(rid_list)
        gname_list = ibs.get_image_gnames(gid_list)
        return gname_list

    @getter
    def get_roi_thetas(ibs, rid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = ibs.get_roi_properties('roi_theta', rid_list)
        return theta_list

    @getter
    def get_roi_nids(ibs, rid_list):
        """ Returns the name_ids of each roi """
        nid_list = ibs.get_roi_properties('name_uid', rid_list)
        return nid_list

    @getter
    def get_roi_names(ibs, rid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        nid_list  = ibs.get_roi_nids(rid_list)
        name_list = ibs.get_table_properties('names', 'name_text', nid_list)
        return name_list

    @getter_vector_output
    def get_roi_groundtruth(ibs, rid_list):
        """ Returns a list of rids with the same name foreach rid in rid_list"""
        groundtruth_list = [[] for rid in rid_list]
        return groundtruth_list

    @getter
    def get_roi_num_groundtruth(ibs, cid_list):
        """ Returns number of other chips with the same name foreach rid in rid_list """
        return map(len, ibs.get_roi_groundtruth(cid_list))

    @getter_vector_output
    def get_rids_in_nids(ibs, nid_list):
        """ returns a list of list of cids in each name """
        # for each name return chips in that name
        rids_list = [[] for _ in xrange(len(nid_list))]
        return rids_list

    @getter
    def get_num_rids_in_nids(ibs, nid_list):
        return map(len, ibs.get_rids_in_nids(nid_list))

    #
    # GETTERS::Chips

    @getter
    def get_chips(ibs, rid_list):
        """ Returns a list cropped images in numpy array form by their cid """
        pass

    @getter
    def get_chip_paths(ibs, rid_list):
        """ Returns a list of chip paths by their cid """
        fmtstr = join(ibs.dbdir, '_ibeisdb/cid%d_dummy.png')
        cpath_list = [fmtstr % cid for cid in rid_list]
        return cpath_list

    @getter_vector_output
    def get_chip_kpts(ibs, rid_list):
        """ Returns chip keypoints """
        import pyhesaff
        kpts_dim = pyhesaff.KPTS_DIM
        kpts_list = [np.empty((0, kpts_dim)) for rid in rid_list]
        return kpts_list

    @getter_vector_output
    def get_chip_desc(ibs, rid_list):
        """ Returns chip descriptors """
        import pyhesaff
        desc_dim = pyhesaff.DESC_DIM
        desc_list = [np.empty((0, desc_dim)) for rid in rid_list]
        return desc_list

    @getter
    def get_chip_num_feats(ibs, rid_list):
        nFeats_list = map(len, ibs.get_chip_kpts(rid_list))
        return nFeats_list
    #
    # GETTERS::Mask

    @getter
    def get_chip_masks(ibs, rid_list):
        # Should this function exist? Yes. -Jon
        roi_list = ibs.get_roi_bboxes(rid_list)
        mask_list = [np.empty((w, h)) for (x, y, w, h) in roi_list]
        return mask_list

    #
    # GETTERS::Name

    @getter_general
    def get_name_properties(ibs, prop_key, nid_list):
        """ general name property getter """
        return ibs.get_table_properties('names', prop_key, nid_list)

    @getter_general
    def get_valid_nids(ibs):
        """ Returns all valid names (does not include unknown names """
        ibs.db.execute(operation='''
                       SELECT name_uid
                       FROM names
                       ''')
        nid_list = ibs.db.result_list()
        return nid_list

    @getter
    def get_names(ibs, nid_list):
        """ Returns text names """
        name_list = ibs.get_name_properties('name_text', nid_list)
        return name_list

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
    def delete_rois(ibs, rid_iter):
        """ deletes all associated roi from the database that belong to the rid"""
        ibs.db.executemany(
            operation='''
            DELETE
            FROM rois
            WHERE roi_uid=?
            ''',
            parameters_iter=rid_iter)

    @deleter
    def delete_images(ibs, gid_list):
        """ deletes the images from the database that belong to gids"""
        ibs.db.executemany(
            operation='''
            DELETE
            FROM images
            WHERE image_uid=?
            ''',
            parameters_iter=gid_list)

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
        probexist_list = jason_detector.detect_existence(ibs, gid_list, **kwargs)
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
        # wrapper
        drid_list = qrid_list
        qres_list = ibs._query_chips(ibs, qrid_list, drid_list, **kwargs)
        return qres_list

    @utool.indent_func
    def query_database(ibs, qrid_list, **kwargs):
        # wrapper
        drid_list = ibs.get_recognition_database_chips()
        qres_list = ibs._query_chips(ibs, qrid_list, drid_list, **kwargs)
        return qres_list

    @utool.indent_func
    def _query_chips(ibs, qrid_list, drid_list, **kwargs):
        """
        qrid_list - query chip ids
        drid_list - database chip ids
        """
        from ibeis.model import jon_identifier
        qres_list = jon_identifier.query(ibs, qrid_list, drid_list, **kwargs)
        # Return for user inspection
        return qres_list
