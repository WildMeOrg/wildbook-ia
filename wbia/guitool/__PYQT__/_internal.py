# -*- coding: utf-8 -*-
# flake8:noqa
# Wrapper around PyQt4/5
"""
Move to PyQt5?
    pip install git+git://github.com/pyqt/python-qt5.git

Ignore:
    >>> import sys
    >>> from PyQt5 import QtWidgets
    >>> app = QtWidgets.QApplication(sys.argv)
    >>> button = QtWidgets.QPushButton("Hello")
    >>> button.setFixedSize(400, 400)
    >>> button.show()
    >>> app.exec_()
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

ut.noinject(__name__, '[__PYQT__._internal]')


# SIP must be imported before any PyQt
try:
    import sip

    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    sip.setapi('QVariant', 2)
    sip.setapi('QString', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QUrl', 2)
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    if hasattr(sip, 'setdestroyonexit'):
        sip.setdestroyonexit(False)  # This prevents a crash on windows
except ImportError as ex:
    print('Warning: Import Error: %s' % str(ex))
except ValueError as ex:
    print('Warning: Value Error: %s' % str(ex))

try:
    import PyQt5

    GUITOOL_PYQT_VERSION = 5
except ImportError:
    import PyQt4

    GUITOOL_PYQT_VERSION = 4

"""
import __PYQT__
from __PYQT__ import QtCore
from __PYQT__ import QtGui
from __PYQT__ import QtTest
from __PYQT__.QtCore import Qt
from __PYQT__.QtGui import QSizePolicy

python -c "from wbia.guitool.__PYQT__ import QtCore"
python -c "from wbia.guitool import __PYQT__"
python -c "import wbia.guitool.__PYQT__"
python -c "import __PYQT__"
from __PYQT__ import QtGui
from __PYQT__ import QtTest
from __PYQT__.QtCore import Qt
from __PYQT__.QtGui import QSizePolicy
"""
