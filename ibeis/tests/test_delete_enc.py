#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_DELETE_ENC]')


def TEST_DELETE_ENC(ibs, back):
    from ibeis import ibsfuncs
    ibsfuncs.update_special_encounters(ibs)
    eid_list = ibs.get_valid_eids()
    assert len(eid_list) != 0, "All Image encounter not created"
    eid = eid_list[0]
    ibs.delete_encounters(eid)
    eid_list = ibs.get_valid_eids()
    assert eid not in eid_list, "eid=%r still exists" % (eid,)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_DELETE_ENC, ibs, back)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
