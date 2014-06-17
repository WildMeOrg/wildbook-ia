from __future__ import absolute_import, division, print_function
import utool
from itertools import izip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[headers]', DEBUG=False)

IMAGE_TABLE = 'images'
ROI_TABLE   = 'rois'
NAME_TABLE  = 'names'
QRES_TABLE  = 'qres'
ENCOUNTER_TABLE = 'encounters'
THUMB_TABLE = 'thumbs'
NAMES_TREE = 'names_tree'

#-----------------
# Define the tables
#-----------------

# enabled tables
TABLENAME_LIST = [IMAGE_TABLE, ROI_TABLE, NAME_TABLE, ENCOUNTER_TABLE, THUMB_TABLE, NAMES_TREE]

# table nice names
TABLE_NICE = {
    IMAGE_TABLE     : 'Image Table',
    ROI_TABLE       : 'ROIs Table',
    NAME_TABLE      : 'Name Table',
    QRES_TABLE      : 'Query Results Table',
    ENCOUNTER_TABLE : 'Encounter Table',
    THUMB_TABLE     : 'Thumbnail Table',
    NAMES_TREE     : 'Tree of Names',
}

# the columns each ibeis table has
TABLE_COLNAMES = {
    #IMAGE_TABLE     : ['image_uuid', 'gid', 'gname', 'nRids', 'aif', 'enctext', 'datetime', 'notes', 'ext'],
    #IMAGE_TABLE     : ['gid', 'gname', 'nRids', 'datetime', 'notes'],
    IMAGE_TABLE     : ['gid', 'thumb', 'nRids', 'gname', 'aif', 'datetime', 'gconf', 'notes'],
    #ROI_TABLE       : ['rid', 'name', 'gname', 'nGt', 'nFeats', 'bbox', 'theta', 'notes'],
    #ROI_TABLE       : ['rid', 'thumb', 'name', 'exemplar', 'gname', 'rconf', 'notes'],
    ROI_TABLE       : ['rid', 'thumb', 'name', 'bbox', 'num', 'verts', 'exemplar', 'gname', 'rconf', 'notes'],
    NAME_TABLE      : ['nid', 'name', 'nRids', 'notes'],
    QRES_TABLE      : ['rank', 'score', 'name', 'rid'],
    ENCOUNTER_TABLE : ['eid', 'nImgs', 'enctext'],
    THUMB_TABLE     : ['thumb', 'thumb', 'thumb', 'thumb'],
    NAMES_TREE      : {('name', 'nid', 'nRids') : ['rid', 'bbox', 'thumb']}, 
    #NAMES_TREE      : ['name', 'nid', 'nRids', 'rid', 'bbox', 'thumb'], 
}

# the columns which are editable
TABLE_EDITSET = {
    IMAGE_TABLE     : set(['aif', 'notes']),
    ROI_TABLE       : set(['name', 'notes', 'exemplar']),
    NAME_TABLE      : set(['name', 'notes']),
    QRES_TABLE      : set(['name']),
    ENCOUNTER_TABLE : set([]),
    THUMB_TABLE     : set([]), 
    NAMES_TREE      : set([]), 
}

# Define the valid columns a table could have
COL_DEF = dict([
    ('image_uuid', (str,      'Image UUID')),
    ('gid',        (int,      'Image ID')),
    ('rid',        (int,      'ROI ID')),
    ('nid',        (int,      'Name ID')),
    ('eid',        (int,      'Encounter ID')),
    ('nRids',      (int,      '#ROIs')),
    ('nGt',        (int,      '#GT')),
    ('nImgs',      (int,      '#Imgs')),
    ('nFeats',     (int,      '#Features')),
    ('rank',       (str,      'Rank')),  # needs to be a string for !Query
    ('unixtime',   (float,    'unixtime')),
    ('gname',      (str,      'Image Name')),
    ('gconf',      (str,      'Detection Confidence')),
    ('rconf',      (float,    'Detection Confidence')),
    ('name',       (str,      'Name')),
    ('notes',      (str,      'Notes')),
    ('match_name', (str,      'Matching Name')),
    ('bbox',       (str,      'BBOX (x, y, w, h))')),  # Non editables are safe as strs
    ('num',        (int,      'NumVerts')),
    ('verts',      (str,      'Verts')),
    ('score',      (str,      'Confidence')),
    ('theta',      (str,      'Theta')),
    ('aif',        (bool,     'Reviewed')),
    ('exemplar',   (bool,     'Is Exemplar')),
    ('enctext',    (str,      'Encounter Text')),
    ('datetime',   (str,      'Date / Time')),
    ('ext',        (str,      'EXT')),
    ('thumb',      ('PIXMAP', 'Thumb')),
])


def make_ibeis_headers_dict(ibs):
    simap_func = utool.scalar_input_map_func
    #
    # Table Iders/Setters/Getters
    iders = {}
    setters = {}
    getters = {}
    #
    # Image Iders/Setters/Getters
    iders[IMAGE_TABLE] = ibs.get_valid_gids
    getters[IMAGE_TABLE] = {
        'gid'        : lambda gids: gids,
        'eid'        : ibs.get_image_eids,
        'enctext'    : simap_func(utool.tupstr, ibs.get_image_enctext),
        'aif'        : ibs.get_image_aifs,
        'gname'      : ibs.get_image_gnames,
        'nRids'      : ibs.get_image_num_rois,
        'unixtime'   : ibs.get_image_unixtime,
        'datetime'   : simap_func(utool.unixtime_to_datetime, ibs.get_image_unixtime),
        'gconf'      : ibs.get_image_confidence,
        'notes'      : ibs.get_image_notes,
        'image_uuid' : ibs.get_image_uuids,
        'ext'        : ibs.get_image_exts,
        'thumb'      : ibs.get_image_thumbtup,
    }
    setters[IMAGE_TABLE] = {
        'aif':   ibs.set_image_aifs,
        'notes': ibs.set_image_notes,
    }
    #
    # ROI Iders/Setters/Getters
    iders[ROI_TABLE] = ibs.get_valid_rids
    getters[ROI_TABLE] = {
        'rid'      : lambda rids: rids,
        'name'     : ibs.get_roi_names,
        'gname'    : ibs.get_roi_gnames,
        'nGt'      : ibs.get_roi_num_groundtruth,
        'theta'    : simap_func(utool.theta_str, ibs.get_roi_thetas),
        'bbox'     : simap_func(utool.bbox_str,  ibs.get_roi_bboxes),
        'num'      : ibs.get_roi_num_verts,
        'verts'    : simap_func(utool.verts_str, ibs.get_roi_verts),
        'nFeats'   : ibs.get_roi_num_feats,
        'rconf'    : ibs.get_roi_confidence,
        'notes'    : ibs.get_roi_notes,
        'thumb'    : ibs.get_roi_chip_thumbtup,
        'exemplar' : ibs.get_roi_exemplar_flag,
    }
    setters[ROI_TABLE] = {
        'name'     : ibs.set_roi_names,
        'notes'    : ibs.set_roi_notes,
        'exemplar' : ibs.set_roi_exemplar_flag,
    }
    #
    # Name Iders/Setters/Getters
    iders[NAME_TABLE] = ibs.get_valid_nids
    getters[NAME_TABLE] = {
        'nid':    lambda nids: nids,
        'name':   ibs.get_names,
        'nRids':  ibs.get_name_num_rois,
        'notes':  ibs.get_name_notes,
    }
    setters[NAME_TABLE] = {
        'name':  ibs.set_name_names,
        'notes': ibs.set_name_notes,
    }
    #
    # Encounter Iders/Setters/Getters
    iders[ENCOUNTER_TABLE] = ibs.get_valid_eids
    getters[ENCOUNTER_TABLE] = {
        'eid':     lambda eids: eids,
        'nImgs':   ibs.get_encounter_num_gids,
        'enctext': ibs.get_encounter_enctext,
    }
    setters[ENCOUNTER_TABLE] = {
        'enctext': ibs.set_encounter_enctext,
    }

    iders[THUMB_TABLE] = ibs.get_valid_gids
    getters[THUMB_TABLE] = {
        'thumb'      : ibs.get_image_thumbtup,
    }
    setters[THUMB_TABLE] = {
    }
    
    
    iders[NAMES_TREE] = ibs.get_valid_nids
    getters[NAMES_TREE] = {
        'nid':    lambda nids: nids,
        'name':   ibs.get_names,
        'nRids':  ibs.get_name_num_rois,
        'rid':    ibs.get_name_rids,
        'bbox':   ibs.get_name_roi_bboxes,
        'thumb':  ibs.get_name_thumbtups,
    }
    setters[NAMES_TREE] = {
    }

    def make_header(tblname):
        """
        Input:
            table_name - the internal table name
        """
        tblnice  = TABLE_NICE[tblname]
        colnames = TABLE_COLNAMES[tblname]
        editset  = TABLE_EDITSET[tblname]
        tblgetters = getters[tblname]
        tblsetters = setters[tblname]
        
        print("....")
        print(colnames)
        def get_column_data(colname):
            coltype   = COL_DEF[colname][0]
            colnice   = COL_DEF[colname][1]
            coledit   = colname in editset 
            colgetter = tblgetters[colname]
            colsetter = None if not coledit else tblsetters.get(colname, None)
            return (coltype, colnice, coledit, colgetter, colsetter)
        try:
            (coltypes, colnices, coledits, colgetters, colsetters) = ([], [], [], [], [])
            if utool.is_list(colnames):
                (coltypes, colnices, coledits, colgetters, colsetters) = zip(*map(get_column_data,colnames))
            elif utool.is_dict(colnames):
                #TODO
                print("not yet implemented")
            else:
                print(AssertionError("TABLE_COLNAMES[%s] must be either a list or a dict." % tblname))
        except KeyError as ex:
            utool.printex(ex,  key_list=['tblname', 'colnames'])
            raise
        header = {
            'name': tblname,
            'nice': tblnice,
            'ider': iders[tblname],
            'col_name_list': colnames,
            'col_type_list': coltypes,
            'col_nice_list': colnices,
            'col_edit_list': coledits,
            'col_getter_list': colgetters,
            'col_setter_list': colsetters,
        }
        return header

    header_dict = {tblname: make_header(tblname) for tblname in TABLENAME_LIST}
    return header_dict
