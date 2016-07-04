#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
"""
Runs IBIES gui

Pyinstaller entry point
When running from non-pyinstaller source use

    python ibeis.__main__.py

instead, or more desirably

    python -m ibeis

"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool as ut
import ibeis
from ibeis.__main__ import run_ibeis


def dependencies_for_myprogram():
    """ Let pyintaller find these modules

    References:
        http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe
    """
    import PyQt4
    from PyQt4 import QtCore, QtGui
    from guitool.__PYQT__ import QtCore, QtGui  # Pyinstaller hacks
    from scipy.sparse.csgraph import _validation
    from scipy.special import _ufuncs_cxx
    import mpl_toolkits.axes_grid1
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #import lru  # NOQA
    # Workaround for mpl_toolkits
    import importlib
    import pyflann
    import gflags
    import statsmodels
    import statsmodels.nonparametric.kde
    import flask
    #import flask.ext.cors
    #from flask.ext.cors import CORS
    importlib.import_module('mpl_toolkits').__path__

if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_ibeis()
