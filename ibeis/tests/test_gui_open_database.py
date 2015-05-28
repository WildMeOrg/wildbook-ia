#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from ibeis.init import sysres
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_OPEN_DATABASE]')


def TEST_GUI_OPEN_DATABASE(ibs, back):
    testdb0 = sysres.db_to_dbdir('testdb0')
    testdb1 = sysres.db_to_dbdir('testdb1')
    print('[TEST] TEST_OPEN_DATABASE testdb1=%r' % testdb1)
    back.open_database(testdb1)
    print('[TEST] TEST_OPEN_DATABASE testdb0=%r' % testdb0)
    back.open_database(testdb0)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_GUI_OPEN_DATABASE, ibs, back)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
