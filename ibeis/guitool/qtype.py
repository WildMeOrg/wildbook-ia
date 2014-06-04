from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
import utool
import uuid
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[qtype]', DEBUG=False)


def qindexinfo(index):
    variant = index.data()
    item = str(variant.toString())
    row  = index.row()
    col  = index.column()
    return (item, row, col)


@profile
def cast_into_qt(data, role=Qt.DisplayRole, flags=Qt.DisplayRole):
    """ Casts data to a QVariant """
    if role == Qt.CheckStateRole and flags & Qt.ItemIsUserCheckable:
        return Qt.Checked if data else Qt.Unchecked
    elif role in (Qt.DisplayRole, Qt.EditRole):
        if utool.is_str(data):
            return QtCore.QVariant(str(data)).toString()
        if utool.is_float(data):
            return QtCore.QVariant(QtCore.QString.number(float(data), format='g', precision=8))
        elif utool.is_bool(data):
            return QtCore.QVariant(bool(data)).toString()
        elif  utool.is_int(data):
            return QtCore.QVariant(int(data)).toString()
        elif isinstance(data, uuid.UUID):
            return QtCore.QVariant(str(data)).toString()
        else:
            return 'Unknown qtype: %r for data=%r' % (type(data), data)
    else:
        return QtCore.QVariant()


@profile
def cast_from_qt(var, type_=None):
    """ Casts a QVariant to data """
    #printDBG('Casting var=%r' % (var,))
    if var is None:
        return None
    if type_ is not None and isinstance(var, QtCore.QVariant):
        # Most cases will be qvariants
        reprstr = str(var.toString())
        data = utool.smart_cast(reprstr, type_)
    elif isinstance(var, QtCore.QVariant):
        if var.typeName() == 'bool':
            data = bool(var.toBool())
        if var.typeName() == 'QString':
            data = str(var.toString())
    elif isinstance(var, QtCore.QString):
        data = str(var)
    #elif isinstance(var, (int, long, str, float)):
    elif isinstance(var, (int, str, unicode)):
        # comboboxes return ints
        data = var
    else:
        raise ValueError('Unknown QtType: type(var)=%r, var=%r' %
                         (type(var), var))
    return data


def infer_coltype(column_list):
    """ Infer Column datatypes """
    try:
        coltype_list = [type(column_data[0]) for column_data in column_list]
    except Exception:
        coltype_list = [str] * len(column_list)
    return coltype_list
