#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_IMAGE]')


def TEST_DELETE_IMAGE(ibs, back):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    gid_list = ibs.add_images(gpath_list)
    bbox_list = [(0, 0, 100, 100)] * len(gid_list)
    name_list = ['a', 'b', 'a', 'd']
    aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list, name_list=name_list)
    gid = gid_list[0]
    assert gid is not None, "gid is None"
    aid_list = ibs.get_image_aids(gid)
    assert len(aid_list) == 1, "Length of aid_list=%r" % (len(aid_list),)
    aid = aid_list[0]
    assert aid is not None, "aid is None"
    cid = ibs.get_annot_chip_rowids(aid, ensure=False)
    fid = ibs.get_annot_feat_rowids(aid, ensure=False)
    assert cid is None, "cid=%r should be None" % (cid,)
    assert fid is None, "fid=%r should be None" % (fid,)
    cid = ibs.get_annot_chip_rowids(aid, ensure=True)
    fid = ibs.get_annot_feat_rowids(aid, ensure=True)
    assert cid is not None, "cid should be computed"
    assert fid is not None, "fid should be computed"
    gthumbpath = ibs.get_image_thumbpath(gid)
    athumbpath = ibs.get_annot_chip_thumbpath(aid)
    ibs.delete_images(gid)
    all_gids = ibs.get_valid_gids()
    all_aids = ibs.get_valid_aids()
    all_cids = ibs.get_valid_cids()
    all_fids = ibs.get_valid_fids()
    assert gid not in all_gids, "gid still exists"
    assert aid not in all_aids, "rid %r still exists" % aid
    assert fid not in all_fids, "fid %r still exists" % fid
    assert cid not in all_cids, "cid %r still exists" % cid
    assert not utool.checkpath(gthumbpath), "Thumbnail still exists"
    assert not utool.checkpath(athumbpath), "ANNOTATION Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_IMAGE, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
