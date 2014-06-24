#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_NEW_SCHEMA]')


def TEST_NEW_SCHEMA(ibs):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    print('[TEST] gpath_list = %r' % gpath_list,)
    gid_list = ibs.add_images(gpath_list)
    print('[TEST] gid_list = %r' % gid_list,)
    bbox_list = [(0, 0, 100, 100)]*len(gid_list)
    bbox_list2 = [(0, 0, 101, 101)]*len(gid_list)
    name_list = ['a', 'b', 'a', 'd']
    rid_list = ibs.add_rois(gid_list, bbox_list=bbox_list, name_list=name_list)
    print('[TEST] rid_list = %r' % rid_list,)
    rid_list2 = ibs.add_rois(gid_list, bbox_list2)
    print('[TEST] rid_list2 = %r' % rid_list2,)
    nid_list = ibs.get_roi_nids(rid_list, distinguish_unknowns=False)
    print('[TEST] nid_list = %r' % nid_list,)

    gt_list = ibs.get_roi_groundtruth(rid_list)
    gt_list2 = ibs.get_roi_groundtruth(rid_list2)
    print('[TEST] ground_truth = %r' % gt_list)
    print('[TEST] ground_truth2 = %r' % gt_list2)
    return locals()



if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_NEW_SCHEMA, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    execstr += '\n' + utool.ipython_execstr()
    exec(execstr)
