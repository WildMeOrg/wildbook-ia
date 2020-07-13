#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of interfacing with IBEIS getters
First run ~/code/wbia/reset_dbs.sh to ensure you have the testdata
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import multiprocessing
import wbia  # IBEIS module
import utool  # Useful Utility Tools

sys.path.append(os.path.expanduser('~/code/wbia'))  # Put IBEIS in PYTHONPATH  # NOQA

(print, rrr, profile) = utool.inject2(__name__)


def run_example(ibs):
    # Print IBEIS Database info
    print(ibs.get_infostr())
    ibs.print_tables()  # uncomment if you want to see a lot of text
    #
    #
    # Each table in the database is indexed with a unique id (rowid)
    # NOTE: This is differnt than a universal unique id (uuid)
    # rowids are ints and uuids are hex strings. Currently
    # only annotations and images have uuids
    #
    gid_list = ibs.get_valid_gids()  # Valid Image IDS
    aid_list = ibs.get_valid_aids()  # Valid ANNOTATION IDs
    nid_list = ibs.get_valid_nids()  # Valid Name IDs
    imgsetid_list = ibs.get_valid_imgsetids()  # Valid ImageSet IDs
    #
    #
    # IBEIS getter methods primarily deal with lists of rowids as input
    # and return lists of values as output
    #
    name_list = ibs.get_name_texts(nid_list)  # Animal names
    aids_in_gids = ibs.get_image_aids(gid_list)  # Rois in images
    aids_in_nids = ibs.get_name_aids(nid_list)  # Rois in images
    img_uuid_list = ibs.get_image_uuids(gid_list)  # Image uuids
    annotation_uuid_list = ibs.get_annot_uuids(aid_list)  # Roi uuids
    #
    #
    # IBEIS Getter methods can take scalars as input too,
    # in this case the output is also a scalar
    #
    gid = gid_list[0]
    gpath = ibs.get_image_paths(gid)  # Get an image path

    # Print locals to the screen
    print('locals() = ' + utool.repr2(locals()))
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    main_locals = wbia.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']  # IBEIS Controller

    # Run the example
    example_locals = run_example(ibs)
    # Add local variables to main namespace
    exec(utool.execstr_dict(example_locals, 'example_locals'))

    execstr = wbia.main_loop(main_locals)
    # Pass the --cmd flag to the program to run in IPython mode
    exec(execstr)
