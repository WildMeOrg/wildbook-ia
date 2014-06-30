#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_FEATURE]')


def TEST_DELETE_FEATURE(ibs, back):
    fid_list = ibs.get_valid_fids()
    if len(fid_list) == 0:
        return locals()
    fid = fid_list[0]
    ibs.delete_names(fid)
    fid_list = ibs.get_valid_fids()
    assert fid not in fid_list, "FID not deleted"
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_FEATURE, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
