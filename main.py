#!/usr/bin/env python2.7
"""
Runs IBIES gui
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import ibeis
import sys
import utool as ut  # NOQA

CMD = '--cmd' in sys.argv

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
    import lru  # NOQA
    # Workaround for mpl_toolkits
    import importlib
    importlib.import_module('mpl_toolkits').__path__


#@ut.profile
def main():
    main_locals = ibeis.main()
    execstr = ibeis.main_loop(main_locals)
    return main_locals, execstr


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    #ut.set_process_title('IBEIS_main')
    main_locals, execstr = main()
    # <DEBUG CODE>
    if 'back' in main_locals and CMD:
        from ibeis.all_imports import *  # NOQA
        back = main_locals['back']
        front = getattr(back, 'front', None)
        #front = back.front
        #ui = front.ui
    ibs = main_locals['ibs']
    exec(execstr)
    # </DEBUG CODE>
