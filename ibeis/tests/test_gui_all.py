#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
import multiprocessing
import utool
import numpy as np
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_ALL]')
np.tau = 2 * np.pi
printTEST = __testing__.printTEST


def TEST_GUI_ALL(ibs, back):
    """
    Creates a new database
    Adds test images
    Creates dummy ROIS
    Selects things
    """
    # DELETE OLD
    printTEST('[TEST] DELETE_OLD_DATABASE')
    work_dir   = params.get_workdir()
    new_dbname = 'testdb_guiall'
    new_dbdir = utool.truepath(utool.join(work_dir, new_dbname))
    ibs_dbdir = utool.truepath(ibs.dbdir)
    msg = 'must start in different dir new_dbdir=%r != ibs_dbdir=%r,' % (new_dbdir, ibs_dbdir)
    assert new_dbdir != ibs_dbdir, msg
    print('passed: ' + msg)
    utool.delete(new_dbdir)
    #
    #
    # CREATE NEW
    printTEST('[TEST] CREATE_NEW_DATABASE')
    back.new_database(new_dbdir)
    ibs = back.ibs  # The backend has a new ibeis do not use the old one
    #
    #
    # IMPORT IMAGES
    printTEST('[TEST] IMPORT_TEST_GPATHS', wait=True)
    gpath_list = __testing__.get_test_gpaths(zebra=True, lena=True, jeff=True)
    gid_list = back.import_images(gpath_list=gpath_list)
    print('\n'.join('  * gid_list[%d] = %r' % (count, gid) for count, gid in enumerate(gid_list)))
    assert len(gid_list) == len(gpath_list)
    #
    #
    # ADD ROIS
    printTEST('[TEST] ADD_ROIS', wait=True)
    def add_roi(gid, bbox, theta=0.0):
        rid = back.add_roi(gid=gid, bbox=bbox, theta=theta)
        return rid

    rid1 = add_roi(gid_list[0], [50, 50, 100, 100], (np.tau / 8))
    rid2 = add_roi(gid_list[1], [50, 50, 100, 100])
    rid3 = add_roi(gid_list[2], [50, 50, 64, 64])
    rid4 = add_roi(gid_list[2], [50, 50, 200, 200])
    rid5 = add_roi(gid_list[1], [0, 0, 400, 400])
    #
    #
    # SELECT ROIS
    printTEST('[TEST] SELECT ROI / Add Chips')
    rid_list = ibs.get_valid_rids()
    print('\n'.join('  * rid_list[%d] = %r' % (count, rid) for count, rid in enumerate(rid_list)))

    #back.select_rid(rid_list[1])
    #back.select_rid(rid_list[2])
    back.select_rid(rid_list[0], show_image=True)

    #add_roi(gid_list[2], None)  # user selection
    #add_roi(None, [42, 42, 8, 8])  # back selection
    # I'm not sure how I want to integrate that IPython stuff
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0', gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = __testing__.run_test(TEST_GUI_ALL, ibs, back)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
