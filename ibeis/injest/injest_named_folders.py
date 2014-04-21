#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import split, join
import utool
import ibeis
from ibeis.dev import ibsfuncs


def injest_named_folder(ibs, dpath):
    """
    Converts a folder structure where folders = names of animals to an ibeis
    database
    """
    gname_list = utool.list_images(dpath)
    name_list  = [split(gpath)[0] for gpath in gname_list]
    gpath_list = [join(dpath, gname) for gname in gname_list]
    gpath = gpath_list[0]
    #
    gid_list = ibs.add_images(gpath_list)

    unique_gids, unique_indices, unique_inverse = np.unique(gid_list, return_index=True, return_inverse=True)
    unique_names = ibsfuncs.normalize_names(np.array(name_list)[unique_indices].tolist())
    nid_list   = ibs.add_names(unique_names)
    theta_list = [0.0 for _ in xrange(len(unique_gids))]
    bbox_list  = ibsfuncs.get_image_bboxes(ibs, unique_gids)
    rid_list   = ibs.add_rois(gid_list, bbox_list, theta_list, nid_list=nid_list)
    pass

if __name__ == '__main__':
    import utool
    import ibeis
    from ibeis.dev.all_imports import *
    dpath = r'D:\data\work\polar_bears'
    main_locals = ibeis.main(dbdir=dpath)
    ibs = main_locals['ibs']
    back = main_locals.get('back', None)
    #dpath = sys.argv[1]
    injest_named_folder()
