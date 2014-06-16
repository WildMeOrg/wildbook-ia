#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
import numpy as np
from ibeis.dev import sysres
from vtool.tests import grabdata
import ibeis
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_ALL]')
np.tau = 2 * np.pi


def TEST_GUI_ALL(ibs, back, gpath_list):
    """
    Creates a new database
    Adds test images
    Creates dummy ROIS
    Selects things
    """
    # DELETE OLD
    print('[TEST] DELETE_OLD_DATABASE')
    work_dir   = sysres.get_workdir()
    new_dbname = 'testdb_guiall'
    new_dbdir = utool.truepath(utool.join(work_dir, new_dbname))
    ibs_dbdir = utool.truepath(ibs.dbdir)
    msg = 'must start in different dir new_dbdir=%r != ibs_dbdir=%r,' % (new_dbdir, ibs_dbdir)
    assert new_dbdir != ibs_dbdir, msg
    print('passed: ' + msg)
    utool.delete(new_dbdir, ignore_errors=False)
    #
    #
    # CREATE NEW
    print('[TEST] CREATE_NEW_DATABASE')
    back.new_database(new_dbdir)
    ibs = back.ibs  # The backend has a new ibeis do not use the old one

    # Dont refresh for speed
    _kwargs = {'refresh': False}
    #
    #
    # IMPORT IMAGES
    print('[TEST] IMPORT_TEST_GPATHS')
    print('gpath_list = ' + utool.indentjoin(gpath_list))
    gid_list = back.import_images(gpath_list=gpath_list, **_kwargs)
    print('\n'.join('  * gid_list[%d] = %r' % (count, gid) for count, gid in enumerate(gid_list)))
    assert len(gid_list) == len(gpath_list)
    #
    #
    # ADD ROIS
    print('[TEST] ADD_ROIS')
    def add_roi(gid, bbox, theta=0.0):
        rid = back.add_roi(gid=gid, bbox=bbox, theta=theta, **_kwargs)
        return rid

    preadd_rids = ibs.get_valid_rids()  # this should be []
    assert len(preadd_rids) == 0, 'there are already rids in the database!'
    print('preadd_rids = %r' % preadd_rids)

    rid1 = add_roi(gid_list[0], (50, 50, 100, 100), (np.tau / 8))
    rid2 = add_roi(gid_list[1], (50, 50, 100, 100))
    rid3 = add_roi(gid_list[2], (50, 50, 64, 64))
    rid4 = add_roi(gid_list[2], (50, 50, 200, 200))
    rid5 = add_roi(gid_list[1], (0, 0, 400, 400))

    print('rid1 = %r' % rid1)
    print('rid2 = %r' % rid2)
    print('rid3 = %r' % rid3)
    print('rid4 = %r' % rid4)
    print('rid5 = %r' % rid5)
    #
    #
    # SELECT ROIS
    print('[TEST] SELECT ROI / Add Chips')
    # get_valid_rids seems to return rids in an arbitrary order, it's an SQL thing
    rid_list = sorted(ibs.get_valid_rids())
    print('\n'.join('  * rid_list[%d] = %r' % (count, rid) for count, rid in enumerate(rid_list)))

    back.select_rid(rid_list[0], show_image=True, **_kwargs)
    try:
        bbox_list = ibs.get_roi_bboxes(rid_list)
        assert bbox_list[0] == (50, 50, 100, 100)
    except AssertionError as ex:
        utool.printex(ex, key_list=['bbox_list', 'rid_list'])
        raise
    back.reselect_roi(bbox=[51, 52, 103, 104])
    assert ibs.get_roi_bboxes(rid_list[0]) == (51, 52, 103, 104)

    back.compute_encounters()

    unixtime_list = [100, 23, 24]
    ibs.set_image_unixtime(gid_list, unixtime_list)

    back.compute_encounters()

    # Change some ROIs

    #add_roi(gid_list[2], None)  # user selection
    #add_roi(None, [42, 42, 8, 8])  # back selection
    # I'm not sure how I want to integrate that IPython stuff
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = ibeis.main(defaultdb='testdb0', gui=True)
    gpath_list = grabdata.get_test_gpaths(names=['lena', 'zebra', 'jeff'])
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_GUI_ALL, ibs, back, gpath_list)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
