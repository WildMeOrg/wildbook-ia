'''
Module Licence and docstring

LOGIC DOES NOT LIVE HERE
THIS DEFINES THE ARCHITECTURE OF IBEIS
'''
from __future__ import division, print_function
from ibeis.control import DatabaseControl


class IBEISControl(object):
    '''
    IBEISController docstring
    cid - chip unique id
    gid - image unique id (could just be the relative file path)
    name_list - name unique id
    eid - enocunter unique id
    '''

    # Constructor
    def __init__(ibs):
        ibs.sql_file = 'database.sqlite3'
        ibs.database = DatabaseControl.DatabaseControl('.', ibs.sql_file)

        ibs.database.schema('images',       {
            'image_uid':                    'INTEGER PRIMARY KEY',
            'image_uri':                    'TEXT NOT NULL',
            'image_hash_sha1':              'TEXT NOT NULL',  # remove???
            'image_width':                  'INTEGER NOT NULL',
            'image_height':                 'INTEGER NOT NULL',
            'image_exif_time_unix':         'INTEGER',
            'image_exif_gps_lat':           'REAL',
            'image_exif_gps_lon':           'REAL',
            'image_existence':              'REAL',
            'image_toggle_context':         'INTEGER DEFAULT 0',
            'image_toggle_encounter':       'INTEGER DEFAULT 0',
            'image_toggle_test':            'INTEGER DEFAULT 0',
            'image_toggle_aif':             'INTEGER DEFAULT 0',
        })

        ibs.database.schema('chips',        {
            'chip_uid':                     'INTEGER PRIMARY KEY',
            'image_uid':                    'INTEGER NOT NULL',
            'chip_roi_xtl':                 'INTEGER NOT NULL',
            'chip_roi_ytl':                 'INTEGER NOT NULL',
            'chip_roi_width':               'INTEGER NOT NULL',
            'chip_roi_height':              'INTEGER NOT NULL',
            'chip_roi_theta':               'REAL DEFAULT 0.0',
            'chip_viewpoint':               'TEXT',
            'chip_toggle_hard':             'INTEGER DEFAULT 0',
        })

        ibs.database.schema('identities',   {
            'identity_uid':                 'INTEGER PRIMARY KEY',
            'chip_uid':                     'INTEGER NOT NULL',
            'identity_name':                'TEXT',
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
        pass

    def add_chips(ibs, gid_list, roi_list, theta_list):
        pass

    #---------------------
    # --- Chip Setters ---
    #---------------------

    def set_chip_roi(ibs, cid, roi):
        'returns a list (x, y, w, h) tuples'
        pass

    def set_chip_thetas(ibs, cid_list, theta_list):
        pass

    def set_chip_names(ibs, cid_list, name_list):
        pass

    def set_chip_properties(ibs, cid_list, key, val_list):
        pass

    #----------------------
    # --- Image Setters ---
    #----------------------

    def set_image_eid(ibs, gid_list, eid_list):
        pass

    def set_image_properties(ibs, gid_list, key, val_list):
        pass

    #----------------------
    # --- Image Getters ---
    #----------------------

    def get_images(ibs, gid_list):
        'returns a list of images (numpy arrays)'
        pass

    def get_image_paths(ibs, gid_list):
        'returns a list of strings (image paths)'
        pass

    def get_image_size(ibs, gid_list):
        'returns a list (width, height) tuples'
        pass

    def get_image_unixtime(hs, gid_list):
        pass

    def get_image_eid(ibs, gid_list):
        'returns a list of encounter ids'
        pass

    def get_cids_in_gids(ibs, gid_list):
        'returns a list of cid lists: [(1, 2), (3), (), (4, 5, 6) ...]'
        # for each image return chips in that image
        pass

    def get_num_cids_in_gids(ibs, gid_list):
        return map(len, ibs.get_cids_in_gids(gid_list))

    def get_image_properties(ibs, gid_list, key):
        pass

    #---------------------
    # --- Chip Getters ---
    #---------------------

    def get_chips(ibs, cid_list):
        'returns a list cropped images (numpy arrays)'
        pass

    def get_chip_paths(ibs, cid_list):
        'returns a list strings (cropped images paths)'
        pass

    def get_chip_gids(ibs, cid_list):
        'returns a list of image ids'
        pass

    def get_chip_rois(ibs, cid_list):
        'returns a list of (x, y, w, h) tuples'
        pass

    def get_chip_thetas(ibs, cid_list):
        pass

    def get_chip_names(ibs, cid_list):
        'returns a list of strings ["fred", "sue", ...]'
        pass

    def get_chip_masks(ibs, cid_list):
        # Should this function exist?
        pass

    def get_chip_properties(ibs, cid_list, key):
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

if __name__ == "__main__":
    ibeis = IBEISController()
