from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[headers]', DEBUG=False)


def ibeis_gui_headers(ibs):
    headers = {}
    #    SQL-key     Type   Nice Name
    #       lambda getter function
    #       lambda setter function

    headers['images'] = [ibs.get_valid_gids , [
        ('gid',      int,   'Image ID',
         lambda gids        : gids,
         None),

        ('gname',    str,   'Image Name',
         lambda gids        : ibs.get_image_gnames(gids),
         None),

        ('nrids',    str,   'ROIs',
         lambda gids        : ibs.get_image_num_rois(gids),
         None),

        #('aif',      bool,  'All Detected',
        # lambda gids        : ibs.get_image_aifs(gids),
        # lambda gids, values: ibs.set_image_aifs(gids, values)),

        #('notes',    str,   'Notes',
        # lambda gids        : ibs.get_image_notes(gids),
        # lambda gids, values: ibs.set_image_notes(gids, values)),

        ('enctext',  str,   'Encounter',
         lambda gids        : map(utool.tupstr, ibs.get_image_enctext(gids)),
         None),

        #('unixtime', float, 'unixtime',
        # lambda gids        : ibs.get_image_unixtime(gids),
        # None),
    ]]

    headers['rois'] = [ibs.get_valid_rids, [
        ('rid',      int,   'ROI ID',
         lambda rids        : rids,
         None),

        ('name',     str,   'Name',
         lambda rids        : ibs.get_roi_names(rids),
         lambda rids, values: ibs.set_roi_names(rids, values)),

        ('gname',    str,   'Image Name',
         lambda rids        : ibs.get_roi_gnames(rids),
         None),

        ('nGt',      int,   '#GT',
         lambda rids        : ibs.get_roi_num_groundtruth(rids),
         None),

        #('nFeats',   int,   '#Features',
        # lambda rids        : ibs.get_roi_num_feats(rids),
        # None),

        #('bbox',     str,   'BBOX (x, y, w, h)',
        # lambda rids        : map(str, ibs.get_roi_bboxes(rids)),
        # None),

        #('theta',    str,   'Theta',
        # lambda rids        : map(utool.theta_str, ibs.get_roi_thetas(rids)),
        # None),

        #('notes',    str,   'Notes',
        # lambda rids        : ibs.get_roi_notes(rids),
        # lambda rids, values: ibs.set_roi_notes(rids, values)),
    ]]

    headers['names'] = [ibs.get_valid_nids, [
        ('nid',      int,   'Name ID',
         lambda nids        : nids,
         None),

        ('name',     str,   'Name',
         lambda nids        : ibs.get_names(nids),
         lambda nids, values: ibs.set_name_names(nids, values)),

        ('nRids',    int,   '#ROIs',
         lambda nids        : ibs.get_name_num_rois(nids),
         None),

        ('notes',    str,   'Notes',
         lambda nids        : ibs.get_name_notes(nids),
         lambda nids, values: ibs.set_name_notes(nids, values)),
    ]]

    headers['encounters'] = [ibs.get_valid_eids, [
        ('enc_name', str,   'Encounter Name',
         lambda eids        : ibs.get_encounter_enctext(eids),
         lambda eids, values: ibs.set_encounter_enctext(eids, values)),

        ('enc_gids', str,   'Images',
         lambda eids        : ibs.get_encounter_num_gids(eids),
         None),
    ]]

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
