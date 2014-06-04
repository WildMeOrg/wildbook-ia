from __future__ import absolute_import, division, print_function
#from PyQt4.QtCore import Qt
from PyQt4.QtCore import QLocale
import utool
import uuid
import numpy as np
from PyQt4 import QtGui
from PyQt4.QtCore import QString, QVariant
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[qtype]', DEBUG=False)

LOCALE = QLocale()

QT_PIXMAP_TYPES = set((QtGui.QPixmap, 'PIXMAP'))
QT_ICON_TYPES   = set((QtGui.QIcon, 'ICON'))
QT_IMAGE_TYPES  = set(list(QT_PIXMAP_TYPES) + list(QT_ICON_TYPES))


def qindexinfo(index):
    variant = index.data()
    item = str(variant.toString())
    row  = index.row()
    col  = index.column()
    return (item, row, col)

#def format_float(data):
#    #argument_format = {
#    #    'e':    format as [-]9.9e[+|-]999
#    #    'E':    format as [-]9.9E[+|-]999
#    #    'f':    format as [-]9.9
#    #    'g':    use e or f format, whichever is the most concise
#    #    'G':    use E or f format, whichever is the most concise
#    #}
#    data = 1000000
#    print(utool.dict_str({
#        'out1': str(QString.number(float(data), format='g', precision=8))
#    }))

#    QLocale(QLocale.English).toString(123456789, 'f', 2)


def numpy_to_qpixmap(npimg):
    data = npimg.astype(np.uint8)
    (height, width) = npimg.shape[0:2]
    format_ = QtGui.QImage.Format_RGB888
    qimg    = QtGui.QImage(data, width, height, format_)
    qpixmap = QtGui.QPixmap.fromImage(qimg)
    return qpixmap


def numpy_to_qicon(npimg):
    qpixmap = numpy_to_qpixmap(npimg)
    qicon = QtGui.QIcon(qpixmap)
    return qicon


@profile
def cast_into_qt(data):
    """ Casts data to a QVariant """
    if utool.is_str(data):
        return QVariant(str(data)).toString()
    if utool.is_float(data):
        #qnumber = QString.number(float(data), format='g', precision=8)
        return QVariant(LOCALE.toString(float(data), format='g', precision=8))
    elif utool.is_bool(data):
        return QVariant(bool(data)).toString()
    elif  utool.is_int(data):
        return QVariant(int(data)).toString()
    elif isinstance(data, uuid.UUID):
        return QVariant(str(data)).toString()
    else:
        return 'Unknown qtype: %r for data=%r' % (type(data), data)


@profile
def cast_from_qt(var, type_=None):
    """ Casts a QVariant to data """
    #printDBG('Casting var=%r' % (var,))
    if var is None:
        return None
    if type_ is not None and isinstance(var, QVariant):
        # Most cases will be qvariants
        reprstr = str(var.toString())
        data = utool.smart_cast(reprstr, type_)
    elif isinstance(var, QVariant):
        if var.typeName() == 'bool':
            data = bool(var.toBool())
        if var.typeName() == 'QString':
            data = str(var.toString())
    elif isinstance(var, QString):
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
