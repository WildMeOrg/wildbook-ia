#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ROI]')


def TEST_DELETE_ROI(ibs, back):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    gid_list = ibs.add_images(gpath_list)
    bbox_list = [(0, 0, 100, 100)]*len(gid_list)
    name_list = ['a', 'b', 'a', 'd']
    rid_list = ibs.add_rois(gid_list, bbox_list=bbox_list, name_list=name_list)
    rid = rid_list[0]
    cid = ibs.get_roi_cids(rid)
    fid = ibs.get_roi_fids(rid)
    thumbtup = ibs.get_roi_chip_thumbtup(rid)
    print("thumbtup_list=%r" % (thumbtup,))
    thumbpath = thumbtup[0]    
    ibs.delete_rois(rid)
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.get_valid_cids()
    fid_list = ibs.get_valid_fids()
    assert rid not in rid_list, "RID still exists"
    assert cid not in cid_list, "CID still exists"
    assert fid not in fid_list, "FID still exists"
    assert not utool.checkpath(thumbpath), "Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ROI, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
