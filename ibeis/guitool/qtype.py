from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
import utool


def qindexinfo(index):
    variant = index.data()
    item = str(variant.toString())
    row  = index.row()
    col  = index.column()
    return (item, row, col)


def cast_into_qt(data, role=Qt.DisplayRole, flags=Qt.DisplayRole):
    """ Casts data to a QVariant """
    if role == Qt.CheckStateRole and flags & Qt.ItemIsUserCheckable:
        var = Qt.Checked if data else Qt.Unchecked
    elif role == Qt.DisplayRole:
        if utool.is_float(data):
            var = QtCore.QVariant(QtCore.QString.number(float(data), format='g', precision=8))
        elif utool.is_bool(data):
            var = QtCore.QVariant(bool(data)).toString()
        elif  utool.is_int(data):
            var = QtCore.QVariant(int(data)).toString()
        else:
            var = 'Unknown qtype: %r' % type(data)
    else:
        var = QtCore.QVariant()
    return var
