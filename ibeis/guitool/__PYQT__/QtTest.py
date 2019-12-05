# -*- coding: utf-8 -*-
# flake8:noqa
# Wrapper around PyQt4/5
from __future__ import absolute_import, division, print_function
import utool as ut
ut.noinject(__name__, '[__PYQT__.QtTest]')
from . import _internal

if _internal.GUITOOL_PYQT_VERSION == 4:
    from PyQt4.QtTest import *
else:
    from PyQt5.QtTest import *
