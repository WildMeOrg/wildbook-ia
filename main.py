#!/usr/bin/env python
'''
Runs IBIES gui
'''
from __future__ import absolute_import, division, print_function
import multiprocessing
import ibeis


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    main_locals = ibeis.main()
    ibs = main_locals['ibs']
    execstr = ibeis.main_loop(main_locals)
    exec(execstr)
