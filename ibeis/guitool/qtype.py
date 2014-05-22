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
        elif  utool.is_str(data):
            var = QtCore.QVariant(str(data)).toString()
        else:
            var = 'Unknown qtype: %r' % type(data)
    else:
        var = QtCore.QVariant()
    return var


def cast_from_qt(var, type_):
    """ Casts a QVariant to data """
    if isinstance(var, QtCore.QVariant):
        # Most cases will be qvariants
        reprstr = str(var.toString())
        data = utool.smart_cast(reprstr, type_)
    elif utool.is_int(var):
        # comboboxes return ints
        data = var
    return data


def infer_coltype(column_list):
    """ Infer Column datatypes """
    try:
        coltype_list = [type(column_data[0]) for column_data in column_list]
    except Exception:
        coltype_list = [str] * len(column_list)
    return coltype_list
