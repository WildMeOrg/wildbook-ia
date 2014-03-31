#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import __testing__
import multiprocessing
import utool
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[test_parallel]')
printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext
def TEST_GUI_SELECTION():
    # Create a HotSpotter API (hs) and GUI backend (back)
    printTEST('[TEST] TEST_ADD_IMAGES')
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    dbdir = params.db_to_dbdir('testdb')

    valid_gids = ibs.get_valid_gids()
    valid_rids = ibs.get_valid_rids()

    print(' * len(valid_gids) = %r' % len(valid_gids))
    print(' * len(valid_rids) = %r' % len(valid_rids))
    assert len(valid_gids) > 0, 'database images cannot be empty for test'

    gid = valid_gids[0]
    back.select_gid(gid)

    printTEST('[TEST] TEST SELECT dbdir=%r' % dbdir)

    __testing__.main_loop(main_locals, rungui=RUNGUI)


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    TEST_GUI_SELECTION()
