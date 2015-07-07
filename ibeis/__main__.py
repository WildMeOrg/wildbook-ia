#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Runs IBIES gui
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool as ut
import ibeis

CMD = ut.get_argflag('--cmd')


# For Pyinstaller
#from ibeis.all_imports import *  # NOQA


def dependencies_for_myprogram():
    """ Let pyintaller find these modules

    References:
        http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe
    """
    from guitool.__PYQT__ import QtCore, QtGui  # Pyinstaller hacks  # NOQA
    from PyQt4 import QtCore, QtGui  # NOQA
    #from PyQt4 import QtCore, QtGui  # NOQA
    from scipy.sparse.csgraph import _validation  # NOQA
    from scipy.special import _ufuncs_cxx  # NOQA
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # NOQA
    #import lru  # NOQA
    # Workaround for mpl_toolkits
    import importlib
    importlib.import_module('mpl_toolkits').__path__


def run_ibeis():
    #ut.set_process_title('IBEIS_main')
    #main_locals = ibeis.main()
    #ibeis.main_loop(main_locals)
    multiprocessing.freeze_support()  # for win32
    #ut.set_process_title('IBEIS_main')
    main_locals = ibeis.main()
    execstr = ibeis.main_loop(main_locals)
    # <DEBUG CODE>
    if 'back' in main_locals and CMD:
        #from ibeis.all_imports import *  # NOQA
        back = main_locals['back']
        front = getattr(back, 'front', None)  # NOQA
        #front = back.front
        #ui = front.ui
    ibs = main_locals['ibs']  # NOQA
    exec(execstr)
    # </DEBUG CODE>


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_ibeis()
