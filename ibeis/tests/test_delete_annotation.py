#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ANNOTATION]')


def TEST_DELETE_ANNOTATION(ibs, back):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    gid_list = ibs.add_images(gpath_list)
    bbox_list = [(0, 0, 100, 100)] * len(gid_list)
    name_list = ['a', 'b', 'a', 'd']
    aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list, name_list=name_list)
    aid = aid_list[0]
    assert aid is not None, "aid is None"
    cid = ibs.get_annot_chip_rowids(aid, ensure=False)
    fid = ibs.get_annot_feat_rowids(aid, ensure=False)
    assert cid is None, "cid should be None"
    assert fid is None, "fid should be None"
    cid = ibs.get_annot_chip_rowids(aid, ensure=True)
    fid = ibs.get_annot_feat_rowids(aid, ensure=True)
    assert cid is not None, "cid should be computed"
    assert fid is not None, "fid should be computed"
    thumbpath = ibs.get_annot_chip_thumbpath(aid)
    ibs.delete_annots(aid)
    aid_list = ibs.get_valid_aids()
    cid_list = ibs.get_valid_cids()
    fid_list = ibs.get_valid_fids()
    assert aid not in aid_list, "RID still exists"
    assert cid not in cid_list, "CID still exists"
    assert fid not in fid_list, "FID still exists"
    assert not utool.checkpath(thumbpath), "Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ANNOTATION, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
