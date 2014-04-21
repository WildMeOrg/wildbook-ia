#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join
import utool
import ibeis
from ibeis.dev import ibsfuncs
from ibeis.dev import params


def injest_named_folder(ibs, img_dir):
    """
    Converts a folder structure where folders = names of animals to an ibeis
    database
    """
    gpath_list = utool.list_images(img_dir, fullpath=True)
    name_list = ibsfuncs.get_names_from_parent_folder(gpath_list, img_dir)
    # Add Images
    gid_list = ibs.add_images(gpath_list)
    # Resolve conflicts
    unique_gids, unique_names, unique_notes = ibsfuncs.resolve_name_conflicts(gid_list, name_list)
    # Add rois with names and notes
    rid_list = ibsfuncs.use_images_as_rois(ibs, unique_gids, name_list=unique_names, notes_list=unique_notes)
    return rid_list

if __name__ == '__main__':
    from ibeis.dev.all_imports import *  # NOQA
    # TODO: be able to injest more than polar bears
    img_dir = join(params.get_workdir(), r'polar_bears')
    main_locals = ibeis.main(dbdir=img_dir)
    ibs = main_locals['ibs']
    back = main_locals.get('back', None)
    #
    # Run injest
    injest_named_folder(ibs, img_dir)
