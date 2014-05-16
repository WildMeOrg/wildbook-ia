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

    rid1 = add_roi(gid_list[0], [50, 50, 100, 100], (np.tau / 8))
    rid2 = add_roi(gid_list[1], [50, 50, 100, 100])
    rid3 = add_roi(gid_list[2], [50, 50, 64, 64])
    rid4 = add_roi(gid_list[2], [50, 50, 200, 200])
    rid5 = add_roi(gid_list[1], [0, 0, 400, 400])
    #
    #
    # SELECT ROIS
    print('[TEST] SELECT ROI / Add Chips')
    rid_list = ibs.get_valid_rids()
    print('\n'.join('  * rid_list[%d] = %r' % (count, rid) for count, rid in enumerate(rid_list)))

    #back.select_rid(rid_list[1])
    #back.select_rid(rid_list[2])

    # Keys for propname come from uidtables.fancy_headers
    back.set_roi_prop(rid_list[0], 'name', 'testname1', **_kwargs)
    back.set_roi_prop(rid_list[1], 'name', 'testname2', **_kwargs)
    back.set_roi_prop(rid_list[0], 'name', 'testname1', **_kwargs)
    back.set_roi_prop(rid_list[2], 'name', '____', **_kwargs)

    name_list = ibs.get_roi_names(rid_list)
    target_name_list = ['testname1', 'testname2', '____3', '____4', '____5']
    try:
        assert name_list == target_name_list
    except AssertionError as ex:
        utool.printex(ex, key_list=['name_list', 'target_name_list'])
        raise

    # Test name props
    nid_list = ibs.get_valid_nids()
    back.set_name_prop(nid_list[0], 'notes', 'notes of a name', **_kwargs)
    back.set_name_prop(nid_list[0], 'name', 'aliased name', **_kwargs)

    # Test image props
    back.set_image_prop(gid_list[0], 'notes', 'notes of an image', **_kwargs)
    back.set_image_prop(gid_list[1], 'aif', True, **_kwargs)

    back.set_view(1)

    back.set_roi_prop(rid_list[1], 'notes', 'Lena', **_kwargs)
    back.set_roi_prop(rid_list[2], 'notes', 'This is, a small ROI on jeff', **_kwargs)
    assert ibs.get_roi_notes(rid_list) == [u'', u'Lena', u'This is, a small ROI on jeff', u'', u'']

    back.set_image_prop(gid_list[0], 'aif', True, **_kwargs)
    back.set_image_prop(gid_list[1], 'aif', False, **_kwargs)
    assert ibs.get_image_aifs(gid_list) == [1, 0, 0]

    back.select_rid(rid_list[0], show_image=True, **_kwargs)
    assert ibs.get_roi_bboxes(rid_list[0]) == (50, 50, 100, 100)
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
    gpath_list = grabdata.get_test_gpaths(zebra=True, lena=True, jeff=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_GUI_ALL, ibs, back, gpath_list)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
