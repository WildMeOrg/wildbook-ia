# -*- coding: utf-8 -*-
# flake8:noqa
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

ut.noinject(__name__, '[__PYQT__.__init__]')
from . import _internal

GUITOOL_PYQT_VERSION = _internal.GUITOOL_PYQT_VERSION

if _internal.GUITOOL_PYQT_VERSION == 4:
    from PyQt4 import *
elif _internal.GUITOOL_PYQT_VERSION == 5:
    from PyQt5 import *
else:
    raise ValueError('Unknown version of PyQt')

from . import QtCore
from . import QtGui
from . import QtWidgets  # just a clone of QtGui in PyQt4


def QVariantHack(*args):
    """ Hack when sip.setapi('QVariant') is 2 """
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        raise NotImplementedError(str(args))


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:

    def _fromUtf8(s):
        return s


try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)


except AttributeError:
    _encoding = ut.identity

    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)


# print('__pyqt5__2')
