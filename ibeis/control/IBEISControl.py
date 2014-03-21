'''
Module Licence and docstring

LOGIC DOES NOT LIVE HERE
THIS DEFINES THE ARCHITECTURE OF IBEIS
'''
from __future__ import division, print_function
from itertools import izip
from ibeis.control import DatabaseControl
from ibeis.vtool import image as gtool
from ibeis.vtool import keypoint as ktool
from ibeis.util import util_hash
from os.path import join
import numpy as np


class IBEISControl(object):
    '''
    IBEISController docstring
    chip  - cropped region of interest from an image, should map to one animal
    cid   - chip unique id
    gid   - image unique id (could just be the relative file path)
    name  - name unique id
    eid   - encounter unique id
    roi   - region of interest for a chip
    theta - angle of rotation for a chip
    '''

    # Constructor
    def __init__(ibs):
        ibs.sql_file = 'database.sqlite3'
        ibs.database = DatabaseControl.DatabaseControl('.', ibs.sql_file)

        ibs.database.schema('images',       {
            'image_uid':                    'INTEGER PRIMARY KEY',
            'image_uri':                    'TEXT NOT NULL',
            'image_width':                  'INTEGER',
            'image_height':                 'INTEGER',
            'image_exif_time_unix':         'INTEGER',
            'image_exif_gps_lat':           'REAL',
            'image_exif_gps_lon':           'REAL',
            'image_confidence':             'REAL',
            'image_toggle_enabled':         'INTEGER DEFAULT 0',
            'image_toggle_aif':             'INTEGER DEFAULT 0',
        })

        ibs.database.schema('encounters',   {
            'encounter_uid':                'INTEGER PRIMARY KEY',
            'image_uid':                    'INTEGER NOT NULL',
        })

        ibs.database.schema('chips',        {
            'chip_uid':                     'INTEGER PRIMARY KEY',
            'image_uid':                    'INTEGER NOT NULL',
            'name_uid':                     'INTEGER DEFAULT 0',
            'chip_roi_xtl':                 'INTEGER NOT NULL',
            'chip_roi_ytl':                 'INTEGER NOT NULL',
            'chip_roi_width':               'INTEGER NOT NULL',
            'chip_roi_height':              'INTEGER NOT NULL',
            'chip_roi_theta':               'REAL DEFAULT 0.0',
            'chip_viewpoint':               'TEXT',
            'chip_toggle_hard':             'INTEGER DEFAULT 0',
        })

        ibs.database.schema('names',   {
            'name_uid':                 'INTEGER PRIMARY KEY',
            'name_text':                'TEXT',
        })

        # Possibly remove???
        ibs.database.schema('segmentatons', {
            'segmentation_id':              'INTEGER PRIMARY KEY',
            'image_id':                     'INTEGER NOT NULL',
            'segmentation_pixel_map_uri':   'TEXT NOT NULL',
        })

        ibs.database.dump()
        pass

    #---------------
    # --- Adders ---
    #---------------

    def add_images(ibs, gpath_list):
        ''' Adds a list of image paths to the database. Returns newly added gids '''
        gid_list = [util_hash.hashstr_arr(gtool.imread(gpath)) for gpath in gpath_list]
        return gid_list

    def add_chips(ibs, gid_list, roi_list, theta_list):
        ''' Adds a list of chips to the database, with ROIs & thetas.
            returns newly added chip ids'''
        chip_iter = izip(gid_list, roi_list, theta_list)
        cid_list = util_hash.hashstr(str(gid) + str(roi) + str(theta)
                                     for gid, roi, theta in chip_iter)
        return cid_list

    #---------------------
    # --- Chip Setters ---
    #---------------------

    def set_chip_rois(ibs, cid_list, roi_list):
        ''' Sets ROIs of a list of chips by cid, returns a list (x, y, w, h) tuples '''
        return None

    def set_chip_thetas(ibs, cid_list, theta_list):
        ''' Sets thetas of a list of chips by cid '''
        return None

    def set_chip_names(ibs, cid_list, name_list):
        ''' Sets names of a list of chips by cid '''
        return None

    #----------------------
    # --- Image Setters ---
    #----------------------

    def set_image_eid(ibs, gid_list, eid_list):
        ''' Sets the encounter id that a list of images is tied to '''
        return None

    #----------------------
    # --- Image Getters ---
    #----------------------

    def get_images(ibs, gid_list):
        ''' Returns a list of images in numpy matrix form by gid '''
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    def get_image_paths(ibs, gid_list):
        ''' Returns a list of image paths by gid '''
        fmtstr = join(ibs.dbdir, '_ibeisdb/gid%d_dummy.jpg')
        gpath_list = [fmtstr % gid for gid in gid_list]
        return gpath_list

    def get_image_size(ibs, gid_list):
        ''' Returns a list of image dimensions by gid in (width, height) tuples '''
        gsize_list = [(0, 0) for gid in gid_list]
        return gsize_list

    def get_image_unixtime(hs, gid_list):
        ''' Returns a list of times that the images were taken by gid. Returns
            -1 if no timedata exists for a given gid
        '''
        unixtime_list = [-1 for gid in gid_list]
        return unixtime_list

    def get_image_eid(ibs, gid_list):
        ''' Returns a list of encounter ids for each image by gid '''
        eid_list = [-1 for gid in gid_list]
        return eid_list

    def get_cids_in_gids(ibs, gid_list):
        ''' Returns a list of cids for each image by gid, e.g. [(1, 2), (3), (), (4, 5, 6) ...] '''
        # for each image return chips in that image
        cids_list = [[] for gid in gid_list]
        return cids_list

    def get_num_cids_in_gids(ibs, gid_list):
        ''' Returns the number of chips associated with a list of images by gid '''
        return map(len, ibs.get_cids_in_gids(gid_list))

    #---------------------
    # --- Chip Getters ---
    #---------------------

    def get_chips(ibs, cid_list):
        'Returns a list cropped images in numpy array form by their cid'
        pass

    def get_chip_paths(ibs, cid_list):
        ''' Returns a list of chip paths by their cid '''
        fmtstr = join(ibs.dbdir, '_ibeisdb/cid%d_dummy.png')
        cpath_list = [fmtstr % cid for cid in cid_list]
        return cpath_list

    def get_chip_gids(ibs, cid_list):
        ''' Returns a list of image ids associated with a list of chips ids'''
        gid_list = [-1] * len(cid_list)
        return gid_list

    def get_chip_rois(ibs, cid_list):
        ''' Returns a list of (x, y, w, h) tuples describing chip geometry in
            image space.
        '''
        roi_list = [(0, 0, -1, -1)] * len(cid_list)
        return roi_list

    def get_chip_thetas(ibs, cid_list):
        ''' Returns a list of floats describing the angles of each chip '''
        theta_list = [0] * len(cid_list)
        return theta_list

    def get_chip_names(ibs, cid_list):
        ''' Returns a list of strings ['fred', 'sue', ...] for each chip
            identifying the animal
        '''
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
        pass

    #---------------------
    # --- Name Getters ---
    #---------------------

    def get_chips_in_name(ibs, name_list):
        'returns a list of list of cids in each name'
        # for each name return chips in that name
        pass

    def get_num_cids_in_name(ibs, name_list):
        return map(len, ibs.get_chips_in_name(name_list))

    #--------------------------
    # --- Encounter Getters ---
    #--------------------------

    def get_cids_in_eids(ibs, eid_list):
        'returns a list of list of cids in each encounter'
        #return cid_list
        pass

    def get_gids_in_eids(ibs, eid_list):
        'returns a list of list of gids in each encounter'
        pass

    #-----------------
    # --- Deleters ---
    #-----------------

    def delete_chips(ibs, cid_list):
        pass

    def delete_images(ibs, gid_list):
        pass

    #----------------
    # --- Loaders ---
    #----------------

    def load_from_sql(ibs):
        'Loads chips, images, name, and encounters'
        pass

    #----------------
    # --- Writers ---
    #----------------

    def save_to_sql(ibs, cid_list):
        'Saves chips, images, name, and encounters'
        pass

    def export_to_wildbook(ibs, cid_list):
        'Exports identified chips to wildbook'
        pass

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
        detections = jason_detector.detect_rois(ibs, gid_list, **kwargs)
        # detections should be a list of [(gid, roi, theta, mask), ...] tuples
        # Return for user inspection
        return detections

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
        '''
        qcid_list - query chip ids
        dcid_list - database chip ids
        '''
        from ibeis.model import jon_identifier
        qres_list = jon_identifier.query(ibs, qcid_list, dcid_list, **kwargs)
        # Return for user inspection
        return qres_list
