#!/usr/bin/env python2.7
"""
Runs IBIES gui
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import ibeis


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    main_locals = ibeis.main()
    # <DEBUG CODE>
    ibs = main_locals['ibs']
    if 'back' in main_locals:
        from ibeis.dev.all_imports import *  # NOQA
        back = main_locals['back']
        front = getattr(back, 'front', None)
        #front = back.front
        #ui = front.ui
    # </DEBUG CODE>
    execstr = ibeis.main_loop(main_locals)
    exec(execstr)
