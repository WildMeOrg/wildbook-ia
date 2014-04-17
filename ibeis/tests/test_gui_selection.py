#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#-----
TEST_NAME = 'TEST_GUI_SELECTION'
#-----
import __testing__  # Should be imported before any ibeis stuff
import multiprocessing
import utool
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)
printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext2(TEST_NAME)
def TEST_GUI_SELECTION():
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    dbdir = params.db_to_dbdir('testdb')

    printTEST('''
              get_valid_gids
              ''')
    valid_gids = ibs.get_valid_gids()
    printTEST('''
              get_valid_rids
              ''')
    valid_rids = ibs.get_valid_rids()

    printTEST('''
    * len(valid_rids) = %r
    * len(valid_gids) = %r
    ''' % (len(valid_rids), len(valid_gids)))
    assert len(valid_gids) > 0, 'database images cannot be empty for test'

    gid = valid_gids[0]
    rid_list = ibs.get_rids_in_gids(gid)
    rid = rid_list[-1]
    back.select_gid(gid, sel_rids=[rid])

    printTEST('[TEST] TEST SELECT dbdir=%r' % dbdir)

    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=RUNGUI)
    return main_locals


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = TEST_GUI_SELECTION()
    exec(test_locals['execstr'])
