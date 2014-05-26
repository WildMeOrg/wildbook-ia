from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool-components]')


def newSizePolicy(widget):
    """
    input: widget - the central widget
    """
    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newHorizontalSplitter(widget):
    """
    input: widget - the central widget
    """
    hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal, widget)
    # This line makes the hsplitter resize with the widget
    sizePolicy = newSizePolicy(widget)
    hsplitter.setSizePolicy(sizePolicy)
    return hsplitter
