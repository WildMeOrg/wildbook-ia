# flake8:noqa
# Wrapper around PyQt4/5
import sip
if hasattr(sip, 'setdestroyonexit'):
    sip.setdestroyonexit(False)  # This prevents a crash on windows
from PyQt4 import *
"""
import __PYQT__
from __PYQT__ import QtCore
from __PYQT__ import QtGui
from __PYQT__.QtCore import Qt
from __PYQT__.QtGui import QSizePolicy
"""
