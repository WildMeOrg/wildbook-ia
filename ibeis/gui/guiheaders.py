from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[headers]', DEBUG=False)

IMAGE_TABLE = 'images'
ROI_TABLE   = 'rois'
NAME_TABLE  = 'names'
QRES_TABLE  = 'qres'
ENCOUNTER_TABLE = 'encounters'


TABLE_DEF = [

    (IMAGE_TABLE, 'Image Table',
     ['gid', 'gname', 'nRids', 'aif', 'notes', 'enctext', 'unixtime'],
     ['notes', 'aif']),

    (ROI_TABLE, 'ROIs Table',
     ['rid', 'name', 'gname', 'nGt', 'nFeats', 'bbox', 'theta', 'notes'],
     ['name', 'notes']),

    (NAME_TABLE, 'Name Table',
     ['nid', 'name', 'nRids', 'notes'],
     ['name', 'notes']),

    #(QRES_TABLE, 'Query Results Table',
    # ['rank', 'score', 'name', 'rid'],
    # ['name']),

    (ENCOUNTER_TABLE, 'Name Table',
     ['eid', 'nGt'],
     []),
]


def ibeis_gui_headers(ibs):
    # Column types and fancy names
    coldefs = dict([
        ('gid',        (int,   'Image ID')),
        ('rid',        (int,   'ROI ID')),
        ('nid',        (int,   'Name ID')),
        ('eid',        (int,   'Encounter ID')),
        ('nRids',      (int,   '#ROIs')),
        ('nGt',        (int,   '#GT')),
        ('nFeats',     (int,   '#Features')),
        ('rank',       (str,   'Rank')),  # needs to be a string for !Query
        ('unixtime',   (float, 'unixtime')),
        ('enctext',    (str,   'Encounter')),
        ('gname',      (str,   'Image Name')),
        ('name',       (str,   'Name')),
        ('notes',      (str,   'Notes')),
        ('match_name', (str,   'Matching Name')),
        ('bbox',       (str,   'BBOX (x, y, w, h))')),  # Non editables are safe as strs
        ('score',      (str,   'Confidence')),
        ('theta',      (str,   'Theta')),
        ('aif',        (bool,  'All Detected')),
        ('enc_text',   (str,   'Encounter Text')),
    ])
    # Table iders
    iders = {
        IMAGE_TABLE: ibs.get_valid_gids,
        ROI_TABLE: ibs.get_valid_rids,
        NAME_TABLE: ibs.get_valid_nids,
        ENCOUNTER_TABLE: ibs.get_valid_eids,
    }
    getters, setters = {}, {}
    # Image Setters/Getters
    getters[IMAGE_TABLE] = {
        'gid':      lambda gids: gids,
        'eid':      lambda gids: ibs.get_image_eids(gids),
        'enctext':  lambda gids: map(utool.tupstr, ibs.get_image_enctext(gids)),
        'aif':      lambda gids: ibs.get_image_aifs(gids),
        'gname':    lambda gids: ibs.get_image_gnames(gids),
        'nRids':    lambda gids: ibs.get_image_num_rois(gids),
        'unixtime': lambda gids: ibs.get_image_unixtime(gids),
        'notes':    lambda nids: ibs.get_image_notes(nids),
    }
    setters[IMAGE_TABLE] = {
        'aif':   ibs.set_image_aifs,
        'notes': ibs.set_image_notes,
    }
    # ROI Setters/Getters
    getters[ROI_TABLE] = {
        'rid':    lambda rids: rids,
        'name':   ibs.get_roi_names,
        'gname':  ibs.get_roi_gnames,
        'nGt':    ibs.get_roi_num_groundtruth,
        'theta':  lambda rids: map(utool.theta_str, ibs.get_roi_thetas(rids)),
        'bbox':   lambda rids: map(str, ibs.get_roi_bboxes(rids)),
        'nFeats': ibs.get_roi_num_feats,
        'notes':  ibs.get_roi_notes,
    }
    setters[ROI_TABLE] = {
        'names': ibs.set_roi_names,
        'notes': ibs.set_roi_notes,
    }
    # Name Setters/Getters
    getters[NAME_TABLE] = {
        'nid':    lambda nids: nids,
        'name':   ibs.get_names,
        'nRids':  ibs.get_name_num_rois,
        'notes':  ibs.get_name_notes,
    }
    setters[NAME_TABLE] = {
        'name': ibs.set_name_names,
        'notes': ibs.set_name_notes,
    }
    # Encounter Setters/Getters
    getters[ENCOUNTER_TABLE] = {
        'eid': lambda eids: eids,
        'nGt': ibs.get_encounter_num_gids,
        'enc_text': ibs.get_encounter_enctext,
    }
    setters[ENCOUNTER_TABLE] = {
        'enc_text': ibs.set_encounter_enctext,
    }

    headers = {}
    def get_coltup(tblname, colname, iseditable):
        # Components of a column tuple
        try:
            (coltype, colfancy) = coldefs[colname]
            colgetter = getters[tblname][colname]
            colsetter = setters[tblname].get(colname, None) if iseditable else None
            # the column tuple
            tup = (colname, coltype, colfancy, colgetter, colsetter)
        except KeyError as ex:
            utool.printex(ex, 'undefined column', key_list=['tblname',
                                                            'colname'])
            raise

        return tup

    def get_table_header(tbltup):
        tblname, tblfancy, colname_list, editable_cols = tbltup
        header_columns = [get_coltup(tblname, colname, colname in editable_cols) for colname in colname_list]
        return tblname, (iders[tblname], header_columns)

    headers = dict([get_table_header(tbltup) for tbltup in TABLE_DEF])
    return headers


def header_ids(header):
    return header[0]


def header_names(header):
    return [ column[0] for column in header[1] ]


def header_types(header):
    return [ column[1] for column in header[1] ]


def header_edits(header):
    return [ column[4] is not None for column in header[1] ]


def header_nices(header):
    return [ column[2] for column in header[1] ]


def getter_from_name(header, name):
    for column in header[1]:
        if column[0] == name:
            return column[3]
    return None


def setter_from_name(header, name):
    for column in header[1]:
        if column[0] == name:
            return column[4]
    return None
