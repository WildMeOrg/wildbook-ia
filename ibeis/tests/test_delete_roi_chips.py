#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ROI_CHIPS]')


def TEST_DELETE_ROI_CHIPS(ibs, back):
    rid_list = ibs.get_valid_rids()
    rid = rid_list[0]
    ibs.delete_roi_chips(rid)
    rid_list = ibs.get_valid_rids()
    assert rid in rid_list, "Error: RID deleted"
    gid_list = get_roi_gids(rid)
    thumbtup_list = ibs.get_image_thumbtups(gid_list)
    assert len(thumbtup_list) == 0, "Thumbtup list not deleted"
    cid_list = ibs.get_roi_cids(rid)
    assert len(cid_list) == 0, "CID not deleted"
    roi_thumbtup_list = ibs.get_roi_chip_thumbtup(rid)
    assert len(roi_thumbtup_list) == 0, "ROI chip thumbtups not deleted"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ROI_CHIPS, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
