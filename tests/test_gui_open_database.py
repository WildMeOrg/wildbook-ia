#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'TEST_GUI_OPEN_DATABASE'
#------
import __testing__  # Should be imported before any ibeis stuff
import multiprocessing
import utool
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext
def TEST_GUI_OPEN_DATABASE():
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    dbdir = params.db_to_dbdir('testdb')

    printTEST('[TEST] TEST_OPEN_DATABASE dbdir=%r' % dbdir)
    back.open_database(dbdir)

    __testing__.main_loop(main_locals, rungui=RUNGUI)
TEST_GUI_OPEN_DATABASE.func_name = TEST_NAME


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    TEST_GUI_OPEN_DATABASE()
