#!/usr/bin/env python
"""
e.g.
python ibeis/injest/injest_named_folders.py --db polar_bears
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
import os
sys.path.append(os.getcwd())  # Windows fix
from os.path import join
import ibeis
from ibeis.dev import ibsfuncs
from ibeis.dev import sysres


def injest_named_folder(ibs, img_dir):
    """
    Converts a folder structure where folders = names of animals to an ibeis
    database
    """
    gpath_list = ibsfuncs.list_images(img_dir)
    name_list = ibsfuncs.get_names_from_parent_folder(gpath_list, img_dir)
    # Add Images
    gid_list = ibs.add_images(gpath_list)
    # Resolve conflicts
    unique_gids, unique_names, unique_notes = ibsfuncs.resolve_name_conflicts(gid_list, name_list)
    # Add rois with names and notes
    rid_list = ibsfuncs.use_images_as_rois(ibs, unique_gids, name_list=unique_names, notes_list=unique_notes)
    return rid_list

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    from ibeis.dev.all_imports import *  # NOQA
    # TODO: be able to injest more than polar bears
    img_dirname = utool.get_arg('--db', str, None)
    img_dir = join(sysres.get_workdir(), img_dirname)
    main_locals = ibeis.main(dbdir=img_dir)
    ibs = main_locals['ibs']
    back = main_locals.get('back', None)
    #
    # Run injest
    injest_named_folder(ibs, img_dir)
