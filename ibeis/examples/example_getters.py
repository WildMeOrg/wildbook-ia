#!/usr/bin/env python
"""
Example of interfacing with IBEIS getters
First run ~/code/ibeis/reset_dbs.sh to ensure you have the testdata
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import multiprocessing
sys.path.append(os.path.expanduser('~/code/ibeis'))  # Put IBEIS in PYTHONPATH
import ibeis  # IBEIS module
import utool  # Useful Utility Tools


def run_example(ibs):
    # Print IBEIS Database info
    print(ibs.get_infostr())
    ibs.print_tables()  # uncomment if you want to see a lot of text
    #
    #
    # Each table in the database is indexed with a unique id (uid)
    # NOTE: This is differnt than a universal unique id (uuid)
    # uids are ints and uuids are hex strings. Currently
    # only rois and images have uuids
    #
    gid_list = ibs.get_valid_gids()  # Valid Image IDS
    rid_list = ibs.get_valid_rids()  # Valid ROI IDs
    nid_list = ibs.get_valid_nids()  # Valid Name IDs
    eid_list = ibs.get_valid_eids()  # Valid Encounter IDs
    #
    #
    # IBEIS getter methods primarily deal with lists of uids as input
    # and return lists of values as output
    #
    name_list     = ibs.get_names(nid_list)        # Animal names
    rids_in_gids  = ibs.get_image_rids(gid_list)   # Rois in images
    rids_in_nids  = ibs.get_name_rids(nid_list)    # Rois in images
    img_uuid_list = ibs.get_image_uuids(gid_list)  # Image uuids
    roi_uuid_list = ibs.get_roi_uuids(rid_list)    # Roi uuids
    #
    #
    # IBEIS Getter methods can take scalars as input too,
    # in this case the output is also a scalar
    #
    gid = gid_list[0]
    gpath = ibs.get_image_paths(gid)  # Get an image path

    # Print locals to the screen
    print('locals() = ' + utool.dict_str(locals()))
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']  # IBEIS Controller

    # Run the example
    example_locals = run_example(ibs)
    # Add local variables to main namespace
    exec(utool.execstr_dict(example_locals, 'example_locals'))

    execstr = ibeis.main_loop(main_locals)
    # Pass the --cmd flag to the program to run in IPython mode
    exec(execstr)
