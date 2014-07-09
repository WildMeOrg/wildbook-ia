#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ANNOTATION_ALL]')


def TEST_DELETE_ANNOTATION_ALL(ibs, back):
    aid_list = ibs.get_valid_aids()
    thumbpath_list = ibs.get_annot_chip_thumbpath(aid_list)
    ibs.delete_annots(aid_list)
    aid_list = ibs.get_valid_aids()
    cid_list = ibs.get_valid_cids()
    fid_list = ibs.get_valid_fids()
    assert len(aid_list) == 0, "Didn't delete all ANNOTATIONs"
    assert len(cid_list) == 0, "Didn't delete all chips"
    assert len(fid_list) == 0, "Didn't delete all features"
    for thumbpath in thumbpath_list:
        assert not utool.checkpath(thumbpath), "Thumbnail still exists"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ANNOTATION_ALL, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
