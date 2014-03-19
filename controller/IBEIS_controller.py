'''
Module Licence and docstring

LOGIC DOES NOT LIVE HERE
THIS DEFINES THE ARCHITECTURE OF IBEIS
'''
from __future__ import division, print_function
#import ibs_sql_loader


class IBEISController(object):
    '''
    IBEISController docstring
    cid - chip unique id
    gid - image unique id (could just be the relative file path)
    name_list - name unique id
    eid - enocunter unique id
    '''

    # Constructor
    def __init__(ibs):
        ibs.sql_file = None
        ibs.database = None  # some sql object
        pass

    #---------------
    # --- Adders ---
    #---------------

    def add_images(ibs, gpath_list):
        pass

    def add_chips(ibs, gid_list, roi_list, theta_list):
        pass

    #----------------
    # --- Setters ---
    #----------------

    def set_chip_roi(ibs, cid, roi):
        pass

    def set_chip_theta(ibs, cid, theta):
        pass

    def set_chip_name(ibs, cid, name):
        pass

    def set_chip_property(ibs, cid, key):
        pass

    def set_image_property(ibs, gid, key):
        pass

    #----------------------
    # --- Image Getters ---
    #----------------------

    def get_images(ibs, gid_list):
        # return raw pixel data
        pass

    def get_image_paths(ibs, gid_list):
        # return image paths
        pass

    def get_image_size(ibs, gid_list):
        pass

    def get_image_unixtime(hs, gid_list):
        pass

    def get_image_property(ibs, gid_list, key):
        pass

    def get_cids_in_gids(ibs, gid_list):
        # for each image return chips in that image
        pass

    def get_num_cids_in_gids(ibs, gid_list):
        return map(len, ibs.get_cids_in_gids(gid_list))

    #---------------------
    # --- Chip Getters ---
    #---------------------

    def get_chips(ibs, cid_list):
        # return raw pixels of chips
        pass

    def get_chip_paths(ibs, gid_list):
        # return chip paths
        pass

    def get_chip_gids(ibs, cid_list):
        pass

    def get_chip_rois(ibs, cid_list):
        pass

    def get_chip_thetas(ibs, cid_list):
        pass

    def get_chip_names(ibs, cid_list):
        pass

    def get_chip_masks(ibs, cid_list):
        pass

    def get_chip_property(ibs, cid, key):
        pass

    #---------------------
    # --- Name Getters ---
    #---------------------

    def get_chips_in_name(ibs, name_list):
        # for each name return chips in that name
        pass

    def get_num_cids_in_name(ibs, name_list):
        return map(len, ibs.get_chips_in_name(name_list))


    #--------------------------
    # --- Encounter Getters ---
    #--------------------------

    def get_encounters(ibs):
        #return eid_list
        pass

    def get_cids_in_eid(ibs, eid_list):
        pass

    def get_gids_in_eids(ibs, eid_list):
        pass

    #-----------------
    # --- Deleters ---
    #-----------------

    def delete_chips(ibs, cid_list):
        pass

    def delete_images(ibs, gid_list):
        pass

    def delete_names(ibs, name_list):
        pass

    #----------------
    # --- Loaders ---
    #----------------

    def load_from_sql(ibs, self):
        pass

    #----------------
    # --- Writers ---
    #----------------

    def export_to_wildbook(ibs, self):
        pass

    def save_to_sql(ibs, self):
        pass

    #--------------
    # --- Model ---
    #--------------

    def cluster_encounters(ibs, gid_list):
        pass

    def query_chips(ibs, qcid_list, dcid_list):
        '''
        qcid_list - query chip ids
        dcid_list - database chip ids
        '''
        #import jon_identifier
        #jon_identifier.query(ibs, qcid_list, dcid_list)
        pass

    def detect_existence(ibs, gid_list):
        #import jason_detector
        #jason_detector.detect
        #jason_detector.detect(ibs, gid_list)
        pass

    def detect_rois_and_masks(ibs, gid_list):
        #import jason_detector
        #jason_detector.detect
        #jason_detector.detect(ibs, gid_list)
        pass

    def get_recognition_database_chips(ibs):
        # ...
        dcid_list = None
        return dcid_list
