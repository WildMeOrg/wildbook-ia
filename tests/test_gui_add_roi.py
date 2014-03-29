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
def test_gui_add_roi():
    # Create a HotSpotter API (hs) and GUI backend (back)
    printTEST('[TEST] TEST_ADD_IMAGES')
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    back.add_roi()

    __testing__.main_loop(main_locals, rungui=RUNGUI)


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_gui_add_roi()
