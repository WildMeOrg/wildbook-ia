#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_IMAGE]')


def TEST_DELETE_IMAGE(ibs, back):
    gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    gid_list1 = ibs.add_images(gpath_list)
    bbox_list = [(0, 0, 100, 100)]*len(gid_list1)
    name_list = ['a', 'b', 'a', 'd']
    aid_list = ibs.add_annotations(gid_list1, bbox_list=bbox_list, name_list=name_list)
    fid_list = ibs.get_annotation_fids(aid_list)
    cid_list = ibs.get_annotation_cids(aid_list)
    thumbtup_list = ibs.get_image_thumbtup(gid_list1)
    thumbpath_list = [tup[0] for tup in thumbtup_list]
    annotation_thumbtup_list = ibs.get_annotation_chip_thumbtup(aid_list)
    annotation_thumbpath_list = [tup[0] for tup in annotation_thumbtup_list]
    ibs.delete_images(gid_list1)
    gid_list2 = ibs.get_valid_gids()
    all_aids = ibs.get_valid_aids()
    all_cids = ibs.get_valid_cids()
    all_fids = ibs.get_valid_fids()
    for gid in gid_list1:
        assert gid not in gid_list2, "GID still exists"
    for aid in aid_list:
        assert aid not in all_aids, "RID %r still exists" % aid
    for fid in fid_list:
        assert fid not in all_fids, "FID %r still exists" % fid
    for cid in cid_list:
        assert cid not in all_cids, "CID %r still exists" % cid
    for path in thumbpath_list:
        assert not utool.checkpath(path), "Thumbnail still exists"
    for path in annotation_thumbpath_list:
        assert not utool.checkpath(path), "ANNOTATION Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    #main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_IMAGE, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
