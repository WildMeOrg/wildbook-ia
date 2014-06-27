#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_IMAGE]')


def TEST_DELETE_IMAGE(ibs, back):
    gid_list = ibs.get_valid_gids()
    gid = gid_list[0]
    rid_list = ibs.get_image_rids(gid)
    fid_list = ibs.get_roi_fids(rid_list)
    cid_list = ibs.get_roi_cids(rid_list)
    thumbtup = ibs.get_image_thumbtup(gid)
    thumbpath = thumbtup[0]    
    roi_thumbtup_list = ibs.get_roi_chip_thumbtup(rid_list)
    roi_thumbpath_list = [tup[0] for tup in roi_thumbtup_list]
    ibs.delete_images(gid)
    gid_list = ibs.get_valid_gids()
    all_rids = ibs.get_valid_rids()
    all_cids = ibs.get_valid_cids()
    all_fids = ibs.get_valid_fids()
    assert gid not in gid_list, "GID still exists"
    for rid in rid_list:
        assert rid not in all_rids, "RID %r still exists" % rid
    for fid in fid_list:
        assert fid not in all_fids, "FID %r still exists" % fid
    for cid in cid_list:
        assert cid not in all_cids, "CID %r still exists" % cid
    assert not utool.checkpath(thumbpath), "Thumbnail still exists"
    for path in roi_thumbpath_list:
        assert not utool.checkpath(path), "ROI Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_IMAGE, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
