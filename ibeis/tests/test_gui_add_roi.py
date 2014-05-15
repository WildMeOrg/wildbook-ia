#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.tests import __testing__
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_ADD_ROI]')


def TEST_GUI_ADD_ROI(ibs, back):
    valid_gids = ibs.get_valid_gids()
    gid = valid_gids[0]
    print('[TEST] SELECT GID=%r' % gid)
    back.select_gid(gid)
    bbox = [0, 0, 100, 100] if not __testing__.INTERACTIVE else None
    rid = back.add_roi(bbox=bbox)
    print('[TEST] NEW RID=%r' % rid)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb0', gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = __testing__.run_test(TEST_GUI_ADD_ROI, ibs, back)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
