from __future__ import absolute_import, division, print_function
import utool
#from itertools import izip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[headers]', DEBUG=False)

ENCOUNTER_TABLE  = 'encounters'
IMAGE_TABLE      = 'images'
IMAGE_GRID       = 'image_grid'
ANNOTATION_TABLE = 'Annotations'
NAME_TABLE       = 'names'
NAMES_TREE       = 'names_tree'
QRES_TABLE       = 'qres'
THUMB_TABLE      = 'thumbs'

#-----------------
# Define the tables
#-----------------

# enabled tables
TABLENAME_LIST = [IMAGE_TABLE, ANNOTATION_TABLE, NAME_TABLE, ENCOUNTER_TABLE, IMAGE_GRID, THUMB_TABLE, NAMES_TREE]

# table nice names
TABLE_NICE = {
    IMAGE_TABLE      : 'Image Table',
    ANNOTATION_TABLE : 'Annotations Table',
    NAME_TABLE       : 'Name Table',
    QRES_TABLE       : 'Query Results Table',
    ENCOUNTER_TABLE  : 'Encounter Table',
    IMAGE_GRID       : 'Thumbnail Grid',
    THUMB_TABLE      : 'Thumbnail Table',
    NAMES_TREE       : 'Tree of Names',
}

# the columns each ibeis table has,
TABLE_COLNAMES = {
    IMAGE_TABLE     : [
        'gid',
        'image_uuid',
        'thumb',
        'nRids',
        'gname',
        'ext',
        'aif',
        'datetime',
        'gps',
        'gdconf',
        'notes',
    ],

    ANNOTATION_TABLE       : [
        #'annotation_uuid',
        'aid',
        'thumb',
        'name',
        'species',
        'exemplar',
        'rdconf',
        'notes',
        'gname',
        'nGt',
        'nFeats',
        'bbox',
        'theta',
        'verts',
        'num_verts',
    ],

    NAME_TABLE      : [
        'nid',
        'name',
        'nRids',
        'notes'
    ],

    QRES_TABLE      : [
        'rank',
        'score',
        'name',
        'aid'
    ],

    ENCOUNTER_TABLE : [
        'eid',
        'enctext',
        'nImgs',
    ],

    NAMES_TREE      : [
        'name',
        'nid',
        'nRids',
        'exemplar',
        'aid',
        'thumb',
    ],

    IMAGE_GRID     : [
        'thumb',
    ],

    # TEST TABLE
    THUMB_TABLE     : [
        'gname',
        'thumb',
    ],


}
#THUMB_TABLE     : ['thumb' 'thumb' 'thumb' 'thumb'],
#NAMES_TREE      : {('name' 'nid' 'nRids') : ['aid' 'bbox' 'thumb']}

# the columns which are editable
TABLE_EDITSET = {
    IMAGE_TABLE      : set(['aif', 'notes']),
    ANNOTATION_TABLE : set(['name', 'species', 'notes', 'exemplar']),
    NAME_TABLE       : set(['name', 'notes']),
    QRES_TABLE       : set(['name']),
    ENCOUNTER_TABLE  : set([]),
    IMAGE_GRID       : set([]),
    THUMB_TABLE      : set([]),
    NAMES_TREE       : set(['exemplar']),
}

TABLE_TREE_LEVELS = {
    NAMES_TREE : [0, 0, 0, 1, 1, 1],
}

TABLE_HIDDEN_LIST = {
    IMAGE_TABLE      : [False, True, False, False, False, True, False, False, False, False, False],
    ANNOTATION_TABLE : [False, False, False, False, False, False, False, True, True, True, True, True, True],
    NAMES_TREE       : [False, False, False, False, False, False],
    NAME_TABLE       : [False, False, False, False],
}

TABLE_STRIPE_LIST = {
    IMAGE_GRID : 3,
}

# Define the valid columns a table could have
COL_DEF = dict([
    ('image_uuid', (str,      'Image UUID')),
    ('gid',        (int,      'Image ID')),
    ('aid',        (int,      'Annotation ID')),
    ('nid',        (int,      'Name ID')),
    ('eid',        (int,      'Encounter ID')),
    ('nRids',      (int,      '#Annotations')),
    ('nGt',        (int,      '#GT')),
    ('nImgs',      (int,      '#Imgs')),
    ('nFeats',     (int,      '#Features')),
    ('rank',       (str,      'Rank')),  # needs to be a string for !Query
    ('unixtime',   (float,    'unixtime')),
    ('species',    (str,      'Species')),
    ('gname',      (str,      'Image Name')),
    ('gdconf',     (str,      'Detection Confidence')),
    ('rdconf',     (float,    'Detection Confidence')),
    ('name',       (str,      'Name')),
    ('notes',      (str,      'Notes')),
    ('match_name', (str,      'Matching Name')),
    ('bbox',       (str,      'BBOX (x, y, w, h))')),  # Non editables are safe as strs
    ('num_verts',  (int,      'NumVerts')),
    ('verts',      (str,      'Verts')),
    ('score',      (str,      'Confidence')),
    ('theta',      (str,      'Theta')),
    ('aif',        (bool,     'Reviewed')),
    ('exemplar',   (bool,     'Is Exemplar')),
    ('enctext',    (str,      'Encounter Text')),
    ('datetime',   (str,      'Date / Time')),
    ('ext',        (str,      'EXT')),
    ('thumb',      ('PIXMAP', 'Thumb')),
    ('gps',        (str,      'GPS')),
])


def make_ibeis_headers_dict(ibs):
    partial_imap_1to1 = utool.partial_imap_1to1
    #
    # Table Iders/Setters/Getters
    iders = {}
    setters = {}
    getters = {}
    #
    # Image Iders/Setters/Getters
    iders[IMAGE_TABLE]   = [ibs.get_valid_gids]
    getters[IMAGE_TABLE] = {
        'gid'        : lambda gids: gids,
        'eid'        : ibs.get_image_eids,
        'enctext'    : partial_imap_1to1(utool.tupstr, ibs.get_image_enctext),
        'aif'        : ibs.get_image_aifs,
        'gname'      : ibs.get_image_gnames,
        'nRids'      : ibs.get_image_num_annotations,
        'unixtime'   : ibs.get_image_unixtime,
        'datetime'   : partial_imap_1to1(utool.unixtime_to_datetime, ibs.get_image_unixtime),
        'gdconf'     : ibs.get_image_detect_confidence,
        'notes'      : ibs.get_image_notes,
        'image_uuid' : ibs.get_image_uuids,
        'ext'        : ibs.get_image_exts,
        'thumb'      : ibs.get_image_thumbtup,
        'gps'        : partial_imap_1to1(utool.tupstr, ibs.get_image_gps),
    }
    setters[IMAGE_TABLE] = {
        'aif'        : ibs.set_image_aifs,
        'notes'      : ibs.set_image_notes,
    }
    #
    # ANNOTATION Iders/Setters/Getters
    iders[ANNOTATION_TABLE]   = [ibs.get_valid_aids]
    getters[ANNOTATION_TABLE] = {
        'aid'        : lambda aids: aids,
        'name'       : ibs.get_annot_names,
        'species'    : ibs.get_annot_species,
        'gname'      : ibs.get_annot_gnames,
        'nGt'        : ibs.get_annot_num_groundtruth,
        'theta'      : partial_imap_1to1(utool.theta_str, ibs.get_annot_thetas),
        'bbox'       : partial_imap_1to1(utool.bbox_str,  ibs.get_annot_bboxes),
        'num_verts'  : ibs.get_annot_num_verts,
        'verts'      : partial_imap_1to1(utool.verts_str, ibs.get_annot_verts),
        'nFeats'     : ibs.get_annot_num_feats,
        'rdconf'     : ibs.get_annot_detect_confidence,
        'notes'      : ibs.get_annot_notes,
        'thumb'      : ibs.get_annot_chip_thumbtup,
        'exemplar'   : ibs.get_annot_exemplar_flag,
    }
    setters[ANNOTATION_TABLE] = {
        'name'       : ibs.set_annot_names,
        'species'    : ibs.set_annot_species,
        'notes'      : ibs.set_annot_notes,
        'exemplar'   : ibs.set_annot_exemplar_flag,
    }
    #
    # Name Iders/Setters/Getters
    iders[NAME_TABLE]   = [ibs.get_valid_nids]
    getters[NAME_TABLE] = {
        'nid'        : lambda nids: nids,
        'name'       : ibs.get_names,
        'nRids'      : ibs.get_name_num_annotations,
        'notes'      : ibs.get_name_notes,
    }
    setters[NAME_TABLE] = {
        'name'       : ibs.set_name_names,
        'notes'      : ibs.set_name_notes,
    }
    #
    # Encounter Iders/Setters/Getters
    iders[ENCOUNTER_TABLE]   = [ibs.get_valid_eids]
    getters[ENCOUNTER_TABLE] = {
        'eid'        : lambda eids: eids,
        'nImgs'      : ibs.get_encounter_num_gids,
        'enctext'    : ibs.get_encounter_enctext,
    }
    setters[ENCOUNTER_TABLE] = {
        'enctext'    : ibs.set_encounter_enctext,
    }

    iders[IMAGE_GRID]   = [ibs.get_valid_gids]
    getters[IMAGE_GRID] = {
        'thumb'      : ibs.get_image_thumbtup,
        'gname'      : ibs.get_image_gnames,
        'aid'        : ibs.get_image_aids,
    }
    setters[IMAGE_GRID] = {
    }

    iders[THUMB_TABLE]   = [ibs.get_valid_gids]
    getters[THUMB_TABLE] = {
        'thumb'      : ibs.get_image_thumbtup,
        'gname'      : ibs.get_image_gnames,
        'aid'        : ibs.get_image_aids,
    }
    setters[THUMB_TABLE] = {
    }

    iders[NAMES_TREE]   = [ibs.get_valid_nids, ibs.get_name_aids]
    getters[NAMES_TREE] = {
        'nid'        : lambda nids: nids,
        'name'       : ibs.get_names,
        'nRids'      : ibs.get_name_num_annotations,
        'aid'        : lambda aids: aids,
        'exemplar'   : ibs.get_annot_exemplar_flag,
        'thumb'      : ibs.get_annot_chip_thumbtup,
    }
    setters[NAMES_TREE] = {
        'exemplar'   : ibs.set_annot_exemplar_flag,
    }

    def make_header(tblname):
        """
        Input:
            table_name - the internal table name
        """
        tblnice    = TABLE_NICE[tblname]
        colnames   = TABLE_COLNAMES[tblname]
        editset    = TABLE_EDITSET[tblname]
        tblgetters = getters[tblname]
        tblsetters = setters[tblname]
        #if levels aren't found, we're not dealing with a tree, so everything is at level 0
        collevels  = TABLE_TREE_LEVELS.get(tblname, [0 for _ in xrange(len(colnames))])
        hiddencols = TABLE_HIDDEN_LIST.get(tblname, [False for _ in xrange(len(colnames))])
        numstripes = TABLE_STRIPE_LIST.get(tblname, 1)

        def get_column_data(colname):
            coltype   = COL_DEF[colname][0]
            colnice   = COL_DEF[colname][1]
            coledit   = colname in editset
            colgetter = tblgetters[colname]
            colsetter = None if not coledit else tblsetters.get(colname, None)
            return (coltype, colnice, coledit, colgetter, colsetter)
        try:
            (coltypes, colnices, coledits, colgetters, colsetters) = zip(*map(get_column_data, colnames))
        except KeyError as ex:
            utool.printex(ex,  key_list=['tblname', 'colnames'])
            raise
        header = {
            'name': tblname,
            'nice': tblnice,
            'iders': iders[tblname],
            'col_name_list': colnames,
            'col_type_list': coltypes,
            'col_nice_list': colnices,
            'col_edit_list': coledits,
            'col_getter_list': colgetters,
            'col_setter_list': colsetters,
            'col_level_list': collevels,
            'col_hidden_list' : hiddencols,
            'num_duplicates'  : numstripes,
        }
        return header

    header_dict = {tblname: make_header(tblname) for tblname in TABLENAME_LIST}
    return header_dict
