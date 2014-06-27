#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ROI]')


def TEST_DELETE_ROI(ibs, back):
    rid_list = ibs.get_valid_rids()
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
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ROI, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
