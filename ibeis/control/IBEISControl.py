'''
Module Licence and docstring

LOGIC DOES NOT LIVE HERE
THIS DEFINES THE ARCHITECTURE OF IBEIS
'''
from __future__ import division, print_function
from itertools import izip
from ibeis.control import DatabaseControl as dbc
from vtool import image as gtool
from vtool import keypoint as ktool
from utool import util_hash
from utool import util_time
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
    rid   - region of interest unique id
    roi   - region of interest for a chip
    theta - angle of rotation for a chip
    '''

    # Constructor
    def __init__(ibs):
        ibs.dbdir = '.'
        ibs.sql_file = 'database.sqlite3'
        ibs.database = dbc.DatabaseControl(ibs.dbdir, ibs.sql_file)

        # TODO: Add algoritm config column
        ibs.database.schema('images',       {
            'image_uid':                    'INTEGER PRIMARY KEY',
            'image_uri':                    'TEXT NOT NULL',
            'image_width':                  'INTEGER',
            'image_height':                 'INTEGER',
            'image_exif_time_unix':         'INTEGER',
            'image_exif_gps_lat':           'REAL',
            'image_exif_gps_lon':           'REAL',
            'image_confidence':             'REAL',  # Think about moving because this is its own algorithm with settings
            'image_toggle_enabled':         'INTEGER DEFAULT 0',
            'image_toggle_aif':             'INTEGER DEFAULT 0',
        })

        ''' Used to store the detected ROIs '''
        ibs.database.schema('rois',         {
            'roi_uid':                      'INTEGER PRIMARY KEY',
            'image_uid':                    'INTEGER NOT NULL',
            'roi_xtl':                      'INTEGER NOT NULL',
            'roi_ytl':                      'INTEGER NOT NULL',
            'roi_width':                    'INTEGER NOT NULL',
            'roi_height':                   'INTEGER NOT NULL',
            'roi_theta':                    'REAL DEFAULT 0.0',
            'roi_viewpoint':                'TEXT',
        })

        ''' Used to store *processed* ROIs as segmentations '''
        ibs.database.schema('masks', {
            'mask_uid':                     'INTEGER PRIMARY KEY',
            'roi_uid':                      'INTEGER NOT NULL',
            'mask_uri':                     'TEXT NOT NULL',
        })

        ''' Used to store *processed* ROIs as chips '''
        ibs.database.schema('chips',        {
            'chip_uid':                     'INTEGER PRIMARY KEY',
            'roi_uid':                      'INTEGER NOT NULL',
            'name_uid':                     'INTEGER DEFAULT 0',
            'chip_width':                   'INTEGER NOT NULL',
            'chip_height':                  'INTEGER NOT NULL',
            'chip_toggle_hard':             'INTEGER DEFAULT 0',
        })

        ibs.database.schema('features',     {
            'feature_uid':                  'INTEGER PRIMARY KEY',
            'chip_uid':                     'INTEGER NOT NULL',
            'feature_keypoints':            'NUMPY',
            'feature_sifts':                'NUMPY',
        })

        ibs.database.schema('names',        {
            'name_uid':                     'INTEGER PRIMARY KEY',
            'name_text':                    'TEXT NOT NULL',
        })

        '''
            Detection and identification algorithm configurations, populated
            with caching information
        '''
        ibs.database.schema('configs',      {
            'config_uid':                   'INTEGER PRIMARY KEY',
            'config_suffix':                'TEXT NOT NULL',
        })

        '''
            This table defines the pairing between an encounter and an
            image. Hence, egpairs stands for encounter-image-pairs.  This table
            exists for the sole purpose of defining multiple encounters to
            a single image without the need to duplicate an image's record
            in the images table.
        '''
        ibs.database.schema('encounters',   {
            'encounter_uid':                'INTEGER PRIMARY KEY',
            'image_uid':                    'INTEGER NOT NULL',
            'encounter_text':               'TEXT NOT NULL',
        })

        # Add default into database
        # JON SAYS SOMETHING like this would look nicer:
        # 
        # JASON SAYS: While this does indeed look nicer, it has issues with extensibility.  
        # Look at line 273.
        # We would be creating a lot of different ways to call the different commands.
        # The part that is keeping me from adopting such a structure is that the 
        # commands would be super complex for some queries and we also have a very large
        # number of kinds of querries, each with their own structure.  Generalization becomes
        # a problem.  I have done the easier querries first, but there are a lot of queries
        # that will use a LEFT JOIN operation in the getters for efficiency's sake.  So if we
        # are going to have to keep a general function to handle these super complex queries, 
        # why not make them all this form?  This way, all SQL is in the controller as opposed to 
        # being dispursed between multiple files.  I'm not terribly happy with the non-general
        # nature, but having some of the queries in my head, it is going to be a huge pain to 
        # generalize (not bringing into the fact that there are security / consistency issues).
        # I'm not terribly sold on not generalizing, but I'd like to re-evaluate after we have
        # all the IBEIS SQL done so we can evaluate if it an be done.
        #ibs.database.insert(table='names', columns=('names_uid', 'names_text'), values=[0, '____'])
        ibs.database.query('INSERT INTO names(name_uid, name_text) VALUES (?, ?)', [0, '____'])

        ibs.database.dump()

    #---------------
    # --- Adders ---
    #---------------

    def add_images(ibs, gpath_list):
        ''' Adds a list of image paths to the database. Returns newly added gids '''

        EXIF_TAGKEYS = gtool.get_exif_tagids([gtool.EXIF_TAG_DATETIME, gtool.EXIF_TAG_GPS])
        EXIF_TAGVAL_DEFAULTS = (-1, (-1, -1))
        def _image_values(gpath):
            # opens lightweight handle to the image (only does this once)
            pil_img = gtool.open_pil_image(gpath)
            (w, h) = pil_img.size
            # Reads all image data
            img = np.asarray(pil_img)
            # Hash the image into an global unique image id
            gid = util_hash.hashstr_sha1(img, base10=True)
            # Try to read exif tags. Default on failure
            exif_val_list = gtool.read_exif_tags(pil_img, EXIF_TAGKEYS, EXIF_TAGVAL_DEFAULTS)
            (exiftime, (gps_lat, gps_lon)) = exif_val_list
            # Convert exif time to unix time
            unixtime = util_time.exiftime_to_unixtime(exiftime)
            # Return values in flat tuple
            return (gid, gpath, w, h, unixtime, gps_lat, gps_lon)
        values_list = [_image_values(gpath) for gpath in iter(gpath_list)]

        sql_qstr = ('INSERT INTO images('
                    'image_uid,'
                    'image_uri,'
                    'image_width,'
                    'image_height,'
                    'image_exif_time_posix,'
                    'image_exif_gps_lat,'
                    'image_exif_gps_lon'
                    ') VALUES (?,?,?,?,?,?,?)')
        sql_qres_list = [ibs.database.query(sql_qstr, value) for value in values_list]
        # JON SAYS: I think it might be better to specify insert as a function, which
        # takes some table (images) as an arg along with a tuple of of columns
        # then have the database control build the string.
        # This would allow for cleaner more reusable code
        # The error would also be generated on the fly and be much more
        # descriptive as well as not polluting the IBEISControl
        # 
        # JASON SAYS: A generic function would be great, but we would lose a lot of 
        # extensibility in the process.  Yes, these commands are long and complex
        # but a "security" issue is also raised by passing to a generator function.
        # Furthermore, there is nothing to prevent us from doing complex error handling
        # in the commit function as it is.  The SQL object will throw errors if, for example,
        # the columns don't exist.
        ibs.database.commit('Error on inserting image, most likely primary key collision', sql_qres_list)
        return [value[0] for value in values_list]

    def add_rois(ibs, gid_list, roi_list):
        pass 

    def add_chips(ibs, roi_list):
        ''' Adds a list of chips to the database, with ROIs & thetas.
            returns newly added chip ids
        '''
        chip_iter = izip(gid_list, roi_list, theta_list)
        cid_list = util_hash.hashstr(str(gid) + str(roi) + str(theta)
                                     for gid, roi, theta in chip_iter)
        return cid_list

    #----------------------
    # --- Image Setters ---
    #----------------------

    def set_image_paths(ibs, gid_list, gpath_list):
        ''' Do we want to do caching here? '''
        pass

    def set_image_eid(ibs, gid_list, eid_list):
        ''' 
            Sets the encounter id that a list of images is tied to, deletes old encounters.
            eid_list is a list of tuples, each represents the set of encounters a tuple
            should belong to.
        '''

        ibs.database.commit('Error on deleting old image encounters',
                [ibs.database.query('DELETE FROM egpairs WHERE \
                image_uid=?', [gid]) for gid in gid_list]
            )

        ibs.database.commit('Error on inserting new image encounters',
                flatten([
                    [ibs.database.query('INSERT IGNORE INTO egpairs( \
                    encounter_uid, \
                    image_uid      \
                    ) VALUES (?,?)', [eid, gid]) for eid in eids] for eids, gid in izip(eid_list, gid_list)
                ])
            )
        return None

    #--------------------
    # --- ROI Setters ---
    #--------------------

    def set_roi_shape(ibs, rid_list, shape_list):
        ''' Sets ROIs of a list of rois by rid, where roi_list is a list of (x, y, w, h) tuples'''
        ibs.database.commit('Error on updating roi shapes',
                [ibs.database.query('UPDATE rois SET \
                roi_xtl=?,     \
                roi_ytl=?,     \
                roi_width=?,   \
                roi_height=?   \
                WHERE roi_uid=?', shape + [rid]) for shape, rid in izip(shape_list, rid_list)]
            )
        return None

    def set_roi_thetas(ibs, rid_list, theta_list):
        ''' Sets thetas of a list of chips by rid '''
        ibs.database.commit('Error on updating roi thetas',
                [ibs.database.query('UPDATE rois SET \
                roi_theta=?, \
                WHERE roi_uid=?', value) for value in izip(theta_list, rid_list)]
            )
        return None

    def set_roi_viewpoints(ibs, rid_list, viewpoint_list):
        ''' Sets viewpoints of a list of chips by rid '''
        ibs.database.commit('Error on updating roi viewpoints',
                [ibs.database.query('UPDATE rois SET \
                roi_viewpoint=?, \
                WHERE roi_uid=?', value) for value in izip(theta_list, viewpoint_list)]
            )
        return None

    #---------------------
    # --- Chip Setters ---
    #---------------------

    def set_chip_names(ibs, cid_list, name_list):
        ''' Sets names of a list of chips by cid '''
        ibs.database.commit('Error on updating chip names',
                [ibs.database.query('UPDATE chips SET \
                chips.name_uid=(SELECT names.name_uid FROM names WHERE name_text=? ORDER BY name_uid LIMIT 1), \
                WHERE chip_uid=?', value) for value in izip(name_list, cid_list)]
            )
        return None

    def set_chip_shape(ibs, cid_list, shape_list):
        ''' Sets shape of a list of chips by cid, a list of tuples (w, h) '''
        ibs.database.commit('Error on updating chip shape',
                [ibs.database.query('UPDATE chips SET \
                chip_width=?, \
                chip_height=?, \
                WHERE chip_uid=?', shape) for shape in izip(shape_list, cid_list)]
            )
        return None

    def set_chip_toggle_hard(ibs, cid_list, hard_list):
        ''' Sets hard toggle of a list of chips by cid '''
        ibs.database.commit('Error on updating chip hard toggle, a list of booleans',
                [ibs.database.query('UPDATE chips SET \
                chip_toggle_hard=?, \
                WHERE chip_uid=?', shape) for shape in izip(shape_list, map(int, cid_list))]
            )
        return None

    #----------------------
    # --- Image Getters ---
    #----------------------

    def get_images(ibs, gid_list):
        ''' 
            Returns a list of images in numpy matrix form by gid 
            NO SQL REQUIRED, DEPENDS ON get_image_paths()
        '''
        gpath_list = ibs.get_image_paths(gid_list)
        image_list = [gtool.imread(gpath) for gpath in gpath_list]
        return image_list

    def get_image_paths(ibs, gid_list):
        ''' Returns a list of image paths by gid '''
        
        db.query('SELECT image_uri FROM images',[])
        for result in db.results():
            print(result)

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
        roi_list = [(0, 0, 1, 1) for cid in cid_list]
        return roi_list

    def get_chip_thetas(ibs, cid_list):
        ''' Returns a list of floats describing the angles of each chip '''
        theta_list = [0 for cid in cid_list]
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
        roi_list = ibs.get_chip_rois(cid_list)
        mask_list = [np.empty((w, h)) for (x, y, w, h) in roi_list]
        return mask_list

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
        cids_list = [[] for eid in eid_list]
        return cids_list

    def get_gids_in_eids(ibs, eid_list):
        'returns a list of list of gids in each encounter'
        gids_list = [[] for eid in eid_list]
        return gids_list

    #-----------------
    # --- Deleters ---
    #-----------------

    def delete_chips(ibs, cid_list):
        ''' deletes all associated chips from the database that belong to the cid'''
        ibs.database.commit('Error on deleting chips with cid',
                [ibs.database.query('DELETE FROM chips WHERE \
                chip_uid=?', [cid]) for cid in cid_list]
            )
        return None

    def delete_images(ibs, gid_list):
        ''' deletes the images from the database that belong to gids'''
        ibs.database.commit('Error on deleting image with gid',
                [ibs.database.query('DELETE FROM images WHERE \
                image_uid=?', [gid]) for gid in gid_list]
            )
        return None

    #----------------
    # --- Writers ---
    #----------------

    def export_to_wildbook(ibs, cid_list):
        'Exports identified chips to wildbook'
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
        '''
        qcid_list - query chip ids
        dcid_list - database chip ids
        '''
        from ibeis.model import jon_identifier
        qres_list = jon_identifier.query(ibs, qcid_list, dcid_list, **kwargs)
        # Return for user inspection
        return qres_list

