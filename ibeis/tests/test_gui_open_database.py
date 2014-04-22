#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import __testing__  # Should be imported before any ibeis stuff
import multiprocessing
import utool
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_OPEN_DATABASE]')

printTEST = __testing__.printTEST


def TEST_GUI_OPEN_DATABASE(ibs, back):
    dbdir = params.db_to_dbdir('testdb')
    printTEST('[TEST] TEST_OPEN_DATABASE dbdir=%r' % dbdir)
    back.open_database(dbdir)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = __testing__.run_test(TEST_GUI_OPEN_DATABASE, ibs, back)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
