#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ROI_ALL]')


def TEST_DELETE_ROI_ALL(ibs, back):
    rid_list = ibs.get_valid_rids()
    thumbtup_list = ibs.get_roi_chip_thumbtup(rid_list)
    thumbpath_list = [tup[0] for tup in thumbtup_list]
    ibs.delete_rois(rid_list)
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.get_valid_cids()
    fid_list = ibs.get_valid_fids()
    assert len(rid_list) == 0, "Didn't delete all ROIs"
    assert len(cid_list) == 0, "Didn't delete all chips"
    assert len(fid_list) == 0, "Didn't delete all features"
    for thumbpath in thumbpath_list:
        assert not utool.checkpath(thumbpath), "Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ROI_ALL, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
