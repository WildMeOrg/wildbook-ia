#!/usr/bin/env python2.7
"""
Runs IBIES gui
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
#import utool
import ibeis
import sys
import utool

# For Pyinstaller
from ibeis.dev.all_imports import *  # NOQA


def dependencies_for_myprogram():
    # Let pyintaller find these modules
    from scipy.sparse.csgraph import _validation  # NOQA
    from scipy.special import _ufuncs_cxx  # NOQA


@utool.profile
def main():
    main_locals = ibeis.main()
    execstr = ibeis.main_loop(main_locals)
    return main_locals, execstr


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    #utool.set_process_title('IBEIS_main')
    main_locals, execstr = main()
    # <DEBUG CODE>
    if 'back' in main_locals and '--cmd' in sys.argv:
        from ibeis.dev.all_imports import *  # NOQA
        back = main_locals['back']
        front = getattr(back, 'front', None)
        #front = back.front
        #ui = front.ui
    ibs = main_locals['ibs']
    exec(execstr)
    # </DEBUG CODE>
