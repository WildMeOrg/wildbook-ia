# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function
from itertools import izip
from os.path import relpath, split
import utool
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibsfuncs]', DEBUG=False)


UNKNOWN_NAMES = set(['Unassigned'])


def normalize_name(name):
    """
    Maps unknonwn names to the standard ____
    """
    if name in UNKNOWN_NAMES:
        name = '____'
    return name


def normalize_names(name_list):
    """
    Maps unknonwn names to the standard ____
    """
    return map(normalize_name, name_list)


def get_names_from_parent_folder(gpath_list, img_dir):
    """
    Input: gpath_list
    Output: names based on the parent folder of each image
    """
    relgpath_list = [relpath(gpath, img_dir) for gpath in gpath_list]
    _name_list  = [split(relgpath)[0] for relgpath in relgpath_list]
    name_list = normalize_names(_name_list)
    return name_list


def resolve_name_conflicts(gid_list, name_list):
    # Build conflict map
    conflict_gid_to_names = utool.build_conflict_dict(gid_list, name_list)

    # Check to see which gid has more than one name
    unique_gids = utool.unique_keep_order2(gid_list)
    unique_names = []
    unique_notes = []

    for gid in unique_gids:
        names = utool.unique_keep_order2(conflict_gid_to_names[gid])
        unique_name = names[0]
        unique_note = ''
        if len(names) > 1:
            if '____' in names:
                names.remove('____')
            if len(names) == 1:
                unique_name = names[0]
            else:
                unique_name = names[0]
                unique_note = 'aliases([' + ', '.join(map(repr, names[1:])) + '])'
        unique_names.append(unique_name)
        unique_notes.append(unique_note)

    return unique_gids, unique_names, unique_notes


def get_image_bboxes(ibs, gid_list):
    size_list = ibs.get_image_size(gid_list)
    bbox_list  = [(0, 0, w, h) for (w, h) in size_list]
    return bbox_list


@utool.indent_func
def compute_all_chips(ibs):
    print('[ibs] compute_all_chips')
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.add_chips(rid_list)
    return cid_list


@utool.indent_func
def compute_all_features(ibs):
    print('[ibs] compute_all_features')
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.get_roi_cids(rid_list, ensure=True)
    fid_list = ibs.add_feats(cid_list)
    return fid_list


@utool.indent_func
def get_empty_gids(ibs):
    """ returns gid list without any chips """
    gid_list = ibs.get_valid_gids()
    nRois_list = ibs.get_num_rids_in_gids(gid_list)
    empty_gids = [gid for gid, nRois in izip(gid_list, nRois_list) if nRois == 0]
    return empty_gids


def convert_empty_images_to_rois(ibs):
    """ images without chips are given an ROI over the entire image """
    gid_list = ibs.get_empty_gids()
    rid_list = ibs.use_images_as_rois(gid_list)
    return rid_list


@utool.indent_func
def use_images_as_rois(ibs, gid_list, name_list=None, nid_list=None, notes_list=None):
    """ Adds an roi the size of the entire image to each image."""
    gsize_list = ibs.get_image_size(gid_list)
    bbox_list  = [(0, 0, w, h) for (w, h) in gsize_list]
    theta_list = [0.0 for _ in xrange(len(gsize_list))]
    rid_list = ibs.add_rois(gid_list, bbox_list, theta_list,
                            name_list=name_list, nid_list=nid_list, notes_list=notes_list)
    return rid_list
