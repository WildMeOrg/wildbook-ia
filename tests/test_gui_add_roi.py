#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'TEST_GUI_ADD_ROI'
#------
import __testing__
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext2(TEST_NAME)
def TEST_GUI_ADD_ROI():
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    valid_gids = ibs.get_valid_gids()
    gid = valid_gids[0]
    printTEST('[TEST] SELECT GID=%r' % gid)
    back.select_gid(gid)
    bbox = [0, 0, 100, 100] if not __testing__.INTERACTIVE else None
    rid = back.add_roi(bbox=bbox)
    printTEST('[TEST] NEW RID=%r' % rid)

    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=RUNGUI)
    return main_locals


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = TEST_GUI_ADD_ROI()
    exec(test_locals['execstr'])
