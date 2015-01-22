#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from os.path import join
import numpy as np
from ibeis.dev import sysres
from vtool.tests import grabdata
import ibeis
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_ALL]')
TAU = 2 * np.pi


def TEST_GUI_ALL(ibs, back, gpath_list):
    """
    Creates a new database
    Adds test images
    Creates dummy ANNOTATIONS
    Selects things
    """
    # DELETE OLD
    print('[TEST] DELETE_OLD_DATABASE')
    work_dir   = sysres.get_workdir()
    new_dbname = 'testdb_guiall'
    new_dbdir = utool.truepath(join(work_dir, new_dbname))
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
    # ADD ANNOTATIONS
    print('[TEST] ADD_ANNOTATIONS')
    def add_annot(gid, bbox, theta=0.0):
        aid = back.add_annot(gid=gid, bbox=bbox, theta=theta, **_kwargs)
        return aid

    preadd_aids = ibs.get_valid_aids()  # this should be []
    assert len(preadd_aids) == 0, 'there are already aids in the database!'
    print('preadd_aids = %r' % preadd_aids)

    aid1 = add_annot(gid_list[0], (50, 50, 100, 100), (TAU / 8))
    aid2 = add_annot(gid_list[1], (50, 50, 100, 100))
    aid3 = add_annot(gid_list[2], (50, 50, 64, 64))
    aid4 = add_annot(gid_list[2], (50, 50, 200, 200))
    aid5 = add_annot(gid_list[1], (0, 0, 400, 400))

    print('aid1 = %r' % aid1)
    print('aid2 = %r' % aid2)
    print('aid3 = %r' % aid3)
    print('aid4 = %r' % aid4)
    print('aid5 = %r' % aid5)
    #
    #
    # SELECT ANNOTATIONS
    print('[TEST] SELECT ANNOTATION / Add Chips')
    # get_valid_aids seems to return aids in an arbitrary order, it's an SQL thing
    aid_list = sorted(ibs.get_valid_aids())
    print('\n'.join('  * aid_list[%d] = %r' % (count, aid) for count, aid in enumerate(aid_list)))

    back.select_aid(aid_list[0], show_image=True, **_kwargs)
    try:
        bbox_list = ibs.get_annot_bboxes(aid_list)
        assert bbox_list[0] == (50, 50, 100, 100)
    except AssertionError as ex:
        utool.printex(ex, key_list=['bbox_list', 'aid_list'])
        raise
    back.reselect_annotation(bbox=[51, 52, 103, 104])
    assert ibs.get_annot_bboxes(aid_list[0]) == (51, 52, 103, 104)

    back.compute_encounters()

    unixtime_list = [100, 23, 24]
    ibs.set_image_unixtime(gid_list, unixtime_list)

    back.compute_encounters()

    # Change some ANNOTATIONs

    #add_annot(gid_list[2], None)  # user selection
    #add_annot(None, [42, 42, 8, 8])  # back selection
    # I'm not sure how I want to integrate that IPython stuff
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = ibeis.main(defaultdb='testdb0', gui=True)
    gpath_list = grabdata.get_test_gpaths(names=['lena', 'zebra', 'jeff'])
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_GUI_ALL, ibs, back, gpath_list)
    #execstr = utool.execstr_dict(test_locals, 'test_locals')
    #exec(execstr)
