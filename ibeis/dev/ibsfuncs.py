# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function
import utool
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibsfuncs]', DEBUG=False)


UNKNOWN_NAMES = set(['Unassigned'])


def normalize_name(name):
    if name in UNKNOWN_NAMES:
        name = '____'
    return name


def normalize_names(name_list):
    return map(normalize_name, name_list)


def get_image_bboxes(ibs, gid_list):
    size_list = ibs.get_image_size(gid_list)
    bbox_list  = [(0, 0, w, h) for (w, h) in size_list]
    return bbox_list
