#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Runs IBIES gui

DEPRICATED

use ibeis.__main__.py instead

or more desirably

python -m ibeis

"""
from __future__ import absolute_import, division, print_function
import multiprocessing
from ibeis.__main__ import run_ibeis


def dependencies_for_myprogram():
    """ Let pyintaller find these modules

    References:
        http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe
    """
    import PyQt4  # NOQA
    from PyQt4 import QtCore, QtGui  # NOQA
    from guitool.__PYQT__ import QtCore, QtGui  # Pyinstaller hacks  # NOQA
    #from PyQt4 import QtCore, QtGui  # NOQA
    from scipy.sparse.csgraph import _validation  # NOQA
    from scipy.special import _ufuncs_cxx  # NOQA
    import mpl_toolkits.axes_grid1  # NOQA
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # NOQA
    #import lru  # NOQA
    # Workaround for mpl_toolkits
    import importlib
    import pyflann  # NOQA
    importlib.import_module('mpl_toolkits').__path__

if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_ibeis()
