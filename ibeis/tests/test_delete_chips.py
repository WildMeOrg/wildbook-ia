#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_CHIPS]')


def TEST_DELETE_CHIPS(ibs, back):
    cid_list = ibs.get_valid_cids()
    cid = cid_list[0]
    _fid_list = ibs.get_chip_fids(cid_list, ensure=False)
    fid_list1 = utool.filter_Nones(_fid_list)
    ibs.delete_chips[cid]
    cid_list = ibs.get_valid_cids()
    assert cid not in cid_list, "CID not deleted"
    fid_list2 = ibs.get_valid_fids()
    for fid in fid_list1:
        assert fid not in fid_list2, "FID not deleted"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_CHIPS, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
