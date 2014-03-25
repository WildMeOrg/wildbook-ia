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
from os.path import join, realpath
import sys
# Science
import numpy as np
# IBEIS
from ibeis.control import SQLDatabaseControl
from ibeis.control import __IBEIS_SCHEMA__
# VTool
from vtool import image as gtool
from vtool import keypoint as ktool
# UTool
from utool import util_hash
from utool import util_time
from utool.util_iter import iflatten


_TAGKEYS = gtool.get_exif_tagids([gtool.EXIF_TAG_DATETIME, gtool.EXIF_TAG_GPS])
_TAGDEFAULTS = (-1, (-1, -1))


def _get_exif(pil_img):
    """ Image EXIF helper """
    (exiftime, (lat, lon)) = gtool.read_exif_tags(pil_img, _TAGKEYS, _TAGDEFAULTS)
    time = util_time.exiftime_to_unixtime(exiftime)  # convert to unixtime
    return time, lat, lon


def _gid_guid(pil_img):
    """ Image GUID helper """
    gid = util_hash.hashstr_sha1(np.asarray(pil_img), base10=True)  # Read all pixels
    return gid


def _sql_qres_gen(gpath):
    """ executes sqlcmd with generated sqlvals """
    pil_img = gtool.open_pil_image(gpath)  # Open PIL Image
    (w, h)  = pil_img.size                 # Read width, height
    (time, lat, lon) = _get_exif(pil_img)  # Read exif tags
    (gid,)           = _gid_guid(pil_img)  # Read pixels ]-hash-> guid = gid
    yield (gid, gpath, w, h, time, lat, lon)


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

    #--------------------
    # --- Constructor ---
    #--------------------

    def __init__(ibs, dbdir='.'):
        print('[ibs] __init__')
        ibs.dbdir = realpath(dbdir)
        ibs.dbfname = '__IBEIS_DATABASE__.sqlite3'
        print('[ibs.__init__] Open the database')
        print('[ibs.__init__] ibs.dbdir    = %r' % ibs.dbdir)
        print('[ibs.__init__] ibs.dbfname = %r' % ibs.dbfname)
        ibs.db = SQLDatabaseControl.SQLDatabaseControl(ibs.dbdir, ibs.dbfname)
        print('[ibs.__init__] Define the schema.')
        __IBEIS_SCHEMA__.define_IBEIS_schema(ibs)
        try:
            print('[ibs.__init__] Add default names.')
            ibs.add_names((0, 1,), ('____', '____',))
        except Exception as ex:
            print('[ibs] HACKISLY IGNORING: %s, %s:' % (type(ex), ex,))
            ibs.db.get_sql_version()
            if not '--ignore' in sys.argv:
                print('use --ignore to keep going')
                raise

    #---------------
    # --- Adders ---
    #---------------

    def add_images(ibs, gpath_list):
        """ Adds a list of image paths to the database. Returns newly added gids """
        ibs.db.querymany(
            operation='''
            INSERT INTO images(
                image_uid,
                image_uri,
                image_width,
                image_height,
                image_exif_time_posix,
                image_exif_gps_lat,
                image_exif_gps_lon
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            parameters_iter=_sql_qres_gen(),
            errmsg='[ibs.add_images] ERROR!  inserting image. Primary key collision?')
        gid_list = [-1 for _ in xrange(len(gpath_list))]
        return gid_list

    def add_rois(ibs, gid_list, bbox_list, theta_list):
        """ add_rois docstr """
        rid_list = [-1 for _ in xrange(len(gid_list))]
        return rid_list

    def add_chips(ibs, rid_list):
        """ Adds a list of chips to the database, with ROIs & thetas.
            returns newly added chip ids
        """
        ibs.db.querymany(
            operation='''
            INSERT INTO chips
            (
                rid_list,
                name_text
            )
            VALUES (?, ?)
            ''',
            parameters_iter=rid_list,
            errmsg='[ibs.add_chips] ERROR inserting chips')
        cid_list = [-1 for _ in xrange(len(rid_list))]
        return cid_list

    def add_names(ibs, nid_iter, name_iter):
        # Autoinsert the defualt-unknown name into the database
        ibs.db.querymany(
            operation='''
            INSERT IGNORE INTO names
            (
                name_uid,
                name_text
            )
            VALUES (?, ?)
            ''',
            parameters_iter=izip(nid_iter, name_iter),
            errmsg='[ibs.add_names] ERROR inserting names')
        nid_iter = [-1 for _ in xrange(len(nid_iter))]
        return nid_iter

    #----------------------
    # --- Setters ---
    #----------------------

    # Image Setters

    def set_image_paths(ibs, gid_list, gpath_list):
        """ Do we want to do caching here? """
        pass

    def set_image_eid(ibs, gid_list, eids_list):
        """
            Sets the encounter id that a list of images is tied to, deletes old encounters.
            eid_list is a list of tuples, each represents the set of encounters a tuple
            should belong to.
        """
        ibs.db.querymany(
            operation='''
            DELETE FROM egpairs WHERE image_uid=?
            ''',
            parameters_iter=gid_list,
            errmsg='[ibs.set_image_eid[1]] ERROR! deleting egpairs')

        ibs.db.querymany(
            operation='''
            INSERT IGNORE INTO egpairs(
                encounter_uid,
                image_uid
            ) VALUES (?, ?)'
            ''',
            parameters_iter=iflatten(((eid, gid)
                                      for eid in eids)
                                     for eids, gid in izip(eids_list, gid_list)),
            errmsg='[ibs.set_image_eid[2]] ERROR on deleting old image encounters')

    # ROI Setters

    def set_roi_bbox(ibs, rid_list, bbox_list):
        """ Sets ROIs of a list of rois by rid, where roi_list is a list of (x, y, w, h) tuples"""
        ibs.db.querymany(
            operation='''
            UPDATE rois SET
                roi_xtl=?,
                roi_ytl=?,
                roi_width=?,
                roi_height=?,
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(bbox_list, rid_list),
            errmsg='[ibs.set_roi_bbox] ERROR!')

    def set_roi_thetas(ibs, rid_list, theta_list):
        """ Sets thetas of a list of chips by rid """
        ibs.db.querymany(
            operation='''
            UPDATE rois SET
                roi_theta=?,
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(theta_list, rid_list),
            errmsg='[ibs.set_roi_thetas] ERROR.')

    def set_roi_viewpoints(ibs, rid_list, viewpoint_list):
        """ Sets viewpoints of a list of chips by rid """
        ibs.db.querymany(
            operation='''
            UPDATE rois SET
                roi_viewpoint=?,
            WHERE roi_uid=?
            ''',
            parameters_iter=izip(viewpoint_list, rid_list),
            errmsg='[ibs.set_roi_viewpoints] ERROR.')

    # Chip Setters

    def set_chip_names(ibs, cid_list, name_list):
        """ Sets names of a list of chips by cid """
        ibs.db.querymany(
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
            WHERE chip_uid=?
            ''',
            parameters_iter=izip(name_list, cid_list),
            errmsg='[ibs.set_chip_names] ERROR.')

    def set_chip_shape(ibs, cid_list, shape_list):
        """ Sets shape of a list of chips by cid, a list of tuples (w, h) """
        ibs.db.querymany(
            operation='''
            UPDATE chips
            SET
                chip_width=?,
                chip_height=?,
            WHERE chip_uid=?
            ''',
            parameters_iter=((w, h, cid) for ((w, h), cid) in izip(shape_list, cid_list)),
            errmsg='[ibs.set_chip_shape] ERROR.')

    def set_chip_toggle_hard(ibs, cid_list, hard_list):
        """ Sets hard toggle of a list of chips by cid """
        ibs.db.querymany(
            operation='''
            UPDATE chips
            SET
                chip_toggle_hard=?,
            WHERE chip_uid=?
            ''',
            parameters_iter=izip(hard_list, cid_list),
            errmsg='[ibs.set_chip_toggle_hard] ERROR.')

    #----------------
    # --- Getters ---
    #----------------

    # Image Getters

    def get_images(ibs, gid_list):
        """
            Returns a list of images in numpy matrix form by gid
            NO SQL REQUIRED, DEPENDS ON get_image_paths()
        """
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    def get_image_paths(ibs, gid_list):
        """ Returns a list of image paths by gid """
        ibs.db.querymany(
            '''
            SELECT image_uri
            FROM images
            WHERE image_uid=?
            ''',
            gid_list,
            errmsg='[ibs.get_image_paths] ERROR')
        guri_iter = ibs.db.result_iter()
        img_dir = join(ibs.dbdir, 'images')
        gpath_list = [join(img_dir, guri) for guri in guri_iter]
        return gpath_list

    def get_image_size(ibs, gid_list):
        """ Returns a list of image dimensions by gid in (width, height) tuples """
        gsize_list = [(0, 0) for gid in gid_list]
        return gsize_list

    def get_image_unixtime(hs, gid_list):
        """ Returns a list of times that the images were taken by gid.
            Returns -1 if no timedata exists for a given gid
        """
        unixtime_list = [-1 for gid in gid_list]
        return unixtime_list

    def get_image_eid(ibs, gid_list):
        """ Returns a list of encounter ids for each image by gid """
        eid_list = [-1 for gid in gid_list]
        return eid_list

    def get_cids_in_gids(ibs, gid_list):
        """ Returns a list of cids for each image by gid,
            e.g. [(1, 2), (3), (), (4, 5, 6) ...] """
        # for each image return chips in that image
        cids_list = [[] for gid in gid_list]
        return cids_list

    def get_num_cids_in_gids(ibs, gid_list):
        """ Returns the number of chips associated with a list of images by gid """
        return map(len, ibs.get_cids_in_gids(gid_list))

    # Chip Getters

    def get_chips(ibs, cid_list):
        """ Returns a list cropped images in numpy array form by their cid """
        pass

    def get_chip_paths(ibs, cid_list):
        """ Returns a list of chip paths by their cid """
        fmtstr = join(ibs.dbdir, '_ibeisdb/cid%d_dummy.png')
        cpath_list = [fmtstr % cid for cid in cid_list]
        return cpath_list

    def get_chip_gids(ibs, cid_list):
        """ Returns a list of image ids associated with a list of chips ids"""
        gid_list = [-1] * len(cid_list)
        return gid_list

    def get_chip_rois(ibs, cid_list):
        """ Returns a list of (x, y, w, h) tuples describing chip geometry in
            image space.
        """
        roi_list = [(0, 0, 1, 1) for cid in cid_list]
        return roi_list

    def get_chip_thetas(ibs, cid_list):
        """ Returns a list of floats describing the angles of each chip """
        theta_list = [0 for cid in cid_list]
        return theta_list

    def get_chip_names(ibs, cid_list):
        """ Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        """
        name_list = map(str, cid_list)
        return name_list

    def get_chip_kpts(ibs, cid_list):
        kpts_list = [np.empty((0, ktool.KPTS_DIM)) for cid in cid_list]
        return kpts_list

    def get_chip_desc(ibs, cid_list):
        desc_list = [np.empty((0, ktool.DESC_DIM)) for cid in cid_list]
        return desc_list

    def get_chip_masks(ibs, cid_list):
        # Should this function exist? Yes. -Jon
        roi_list = ibs.get_chip_rois(cid_list)
        mask_list = [np.empty((w, h)) for (x, y, w, h) in roi_list]
        return mask_list

    # Name Getters

    def get_chips_in_name(ibs, name_list):
        """ returns a list of list of cids in each name """
        # for each name return chips in that name
        pass

    def get_num_cids_in_name(ibs, name_list):
        return map(len, ibs.get_chips_in_name(name_list))

    # Encounter Getters

    def get_cids_in_eids(ibs, eid_list):
        """ returns a list of list of cids in each encounter """
        cids_list = [[] for eid in eid_list]
        return cids_list

    def get_gids_in_eids(ibs, eid_list):
        """ returns a list of list of gids in each encounter """
        gids_list = [[] for eid in eid_list]
        return gids_list

    #-----------------
    # --- Deleters ---
    #-----------------

    def delete_chips(ibs, cid_iter):
        """ deletes all associated chips from the database that belong to the cid"""
        ibs.db.querymany(
            operation='''
            DELETE
            FROM
            chips
            WHERE chip_uid=?
            ''',
            parameters_iter=cid_iter,
            errmsg='[ibs.delete_chips()] ERROR.')

    def delete_images(ibs, gid_list):
        """ deletes the images from the database that belong to gids"""
        ibs.db.querymany(
            operation='''
            DELETE
            FROM images
            WHERE image_uid=?
            ''',
            parameters_iter=gid_list,
            errmsg='[ibs.delete_images()] ERROR.')

    #----------------
    # --- Writers ---
    #----------------

    def export_to_wildbook(ibs, cid_list):
        """ Exports identified chips to wildbook """
        return None

    #--------------
    # --- Model ---
    #--------------

    def cluster_encounters(ibs, gid_list):
        'Finds encounters'
        from ibeis.model import encounter_cluster
        eid_list = encounter_cluster.cluster(ibs, gid_list)
        #ibs.set_image_eids(gid_list, eid_list)
        return eid_list

    def detect_existence(ibs, gid_list, **kwargs):
        'Detects the probability of animal existence in each image'
        from ibeis.model import jason_detector
        probexist_list = jason_detector.detect_existence(ibs, gid_list, **kwargs)
        # Return for user inspection
        return probexist_list

    def detect_rois_and_masks(ibs, gid_list, **kwargs):
        'Runs animal detection in each image'
        # Should this function just return rois and no masks???
        from ibeis.model import jason_detector
        detection_list = jason_detector.detect_rois(ibs, gid_list, **kwargs)
        # detections should be a list of [(gid, roi, theta, mask), ...] tuples
        # Return for user inspection
        return detection_list

    def get_recognition_database_chips(ibs):
        'returns chips which are part of the persitent recognition database'
        dcid_list = None
        return dcid_list

    def query_intra_encounter(ibs, qcid_list, **kwargs):
        # wrapper
        dcid_list = qcid_list
        qres_list = ibs._query_chips(ibs, qcid_list, dcid_list, **kwargs)
        return qres_list

    def query_database(ibs, qcid_list, **kwargs):
        # wrapper
        dcid_list = ibs.get_recognition_database_chips()
        qres_list = ibs._query_chips(ibs, qcid_list, dcid_list, **kwargs)
        return qres_list

    def _query_chips(ibs, qcid_list, dcid_list, **kwargs):
        """
        qcid_list - query chip ids
        dcid_list - database chip ids
        """
        from ibeis.model import jon_identifier
        qres_list = jon_identifier.query(ibs, qcid_list, dcid_list, **kwargs)
        # Return for user inspection
        return qres_list
