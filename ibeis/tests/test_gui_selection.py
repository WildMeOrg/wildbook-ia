#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_SELECTION]')
from __testing__ import printTEST


def TEST_GUI_SELECTION(ibs, back):
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
    rid_list = ibs.get_image_rids(gid)
    rid = rid_list[-1]
    back.select_gid(gid, sel_rids=[rid])

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0', gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = __testing__.run_test(TEST_GUI_SELECTION, ibs, back)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
