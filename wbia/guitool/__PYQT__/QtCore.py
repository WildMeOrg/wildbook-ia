# -*- coding: utf-8 -*-
# flake8:noqa
# Wrapper around PyQt4/5
import utool as ut

ut.noinject(__name__, '[__PYQT__.QtCore]')
from . import _internal

if _internal.GUITOOL_PYQT_VERSION == 4:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import QItemSelection
    from PyQt4.QtGui import QItemSelectionModel
else:
    from PyQt5.QtCore import *
