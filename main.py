#!/usr/bin/env python
'''
Runs IBIES gui
'''
from __future__ import division, print_function
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    from ibeis.dev import main_api
    main_locals = main_api.main()
    main_api.main_loop(main_locals)
