from __future__ import absolute_import, division, print_function
# FIXME: from guitool import __PYQT__
# flake8:noqa
#print('__pyqt5__1')
# Wrapper around PyQt4/5

#raise ImportError('Cannot Import Qt')

import utool as ut
ut.noinject(__name__, '[guitool.__PYQT__]')

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
except ValueError as ex:
    print('Warning: Value Error: %s' % str(ex))
    pass


from PyQt4 import *
"""
import __PYQT__
from __PYQT__ import QtCore
from __PYQT__ import QtGui
from __PYQT__ import QtTest
from __PYQT__.QtCore import Qt
from __PYQT__.QtGui import QSizePolicy
"""


def QVariantHack(*args):
    """ Hack when sip.setapi('QVariant') is 2 """
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        raise NotImplementedError(str(args))
#print('__pyqt5__2')
