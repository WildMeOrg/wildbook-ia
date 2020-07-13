# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

# from wbia.guitool.__PYQT__.QtCore import Qt
import six
from wbia.guitool.__PYQT__.QtCore import QLocale
import utool as ut
import uuid
import numpy as np
from wbia.guitool.__PYQT__ import QtGui
from wbia.guitool.guitool_decorators import checks_qt_error

# if six.PY2:
#    from wbia.guitool.__PYQT__.QtCore import QString
#    from wbia.guitool.__PYQT__.QtCore import QVariant
# elif six.PY3:
QVariant = None

__STR__ = unicode if six.PY2 else str  # NOQA

QString = __STR__

(print, rrr, profile) = ut.inject2(__name__)


SIMPLE_CASTING = True


ItemDataRoles = {
    0: 'DisplayRole',  # key data to be rendered in the form of text. (QString)
    1: 'DecorationRole',  # data to be rendered as an icon. (QColor, QIcon or QPixmap)
    2: 'EditRole',  # data in a form suitable for editing in an editor. (QString)
    3: 'ToolTipRole',  # data displayed in the item's tooltip. (QString)
    4: 'StatusTipRole',  # data displayed in the status bar. (QString)
    5: 'WhatsThisRole',  # data displayed in "What's This?" mode. (QString)
    6: 'FontRole',  # font used for items rendered with default delegate. (QFont)
    7: 'TextAlignmentRole',  # text alignment of items with default delegate. (Qt::AlignmentFlag)
    # 8: 'BackgroundColorRole',  # Obsolete. Use BackgroundRole instead.
    # 9: 'TextColorRole',  # Obsolete. Use ForegroundRole instead.
    8: 'BackgroundRole',  # background brush for items with default delegate. (QBrush)
    9: 'ForegroundRole',  # foreground brush for items rendered with default delegate. (QBrush)
    10: 'CheckStateRole',  # checked state of an item. (Qt::CheckState)
    11: 'AccessibleTextRole',  # text used by accessibility extensions and plugins (QString)
    12: 'AccessibleDescriptionRole',  # accessibe description of the item for (QString)
    13: 'SizeHintRole',  # size hint for item that will be supplied to views. (QSize)
    14: 'InitialSortOrderRole',  # initial sort order of a header view (Qt::SortOrder).
    32: 'UserRole',  # first role that can be used for application-specific purposes.
}

LOCALE = QLocale()

# Custom types of data that can be displayed (usually be a delegate)
QT_PIXMAP_TYPES = set((QtGui.QPixmap, 'PIXMAP'))
QT_ICON_TYPES = set((QtGui.QIcon, 'ICON'))
QT_BUTTON_TYPES = set(('BUTTON',))
QT_COMBO_TYPES = set(('COMBO',))


QT_IMAGE_TYPES = set(list(QT_PIXMAP_TYPES) + list(QT_ICON_TYPES))
# A set of all delegate types
QT_DELEGATE_TYPES = set(
    list(QT_IMAGE_TYPES) + list(QT_BUTTON_TYPES) + list(QT_COMBO_TYPES)
)


def qindexinfo(index):
    variant = index.data()
    if SIMPLE_CASTING:
        item = __STR__(variant)
    else:
        item = __STR__(variant.toString())
    row = index.row()
    col = index.column()
    return (item, row, col)


# def format_float(data):
#    #argument_format = {
#    #    'e':    format as [-]9.9e[+|-]999
#    #    'E':    format as [-]9.9E[+|-]999
#    #    'f':    format as [-]9.9
#    #    'g':    use e or f format, whichever is the most concise
#    #    'G':    use E or f format, whichever is the most concise
#    #}
#    data = 1000000
#    print(ut.repr2({
#        'out1': __STR__(QString.number(float(data), format='g', precision=8))
#    }))

#    QLocale(QLocale.English).toString(123456789, 'f', 2)


def numpy_to_qpixmap(npimg):
    data = npimg.astype(np.uint8)
    (height, width) = npimg.shape[0:2]
    format_ = QtGui.QImage.Format_RGB888
    qimg = QtGui.QImage(data, width, height, format_)
    qpixmap = QtGui.QPixmap.fromImage(qimg)
    return qpixmap


def numpy_to_qicon(npimg):
    qpixmap = numpy_to_qpixmap(npimg)
    qicon = QtGui.QIcon(qpixmap)
    return qicon


def locale_float(float_, precision=4):
    """
    References:
        http://qt-project.org/doc/qt-4.8/qlocale.html#toString-9
    """
    return LOCALE.toString(float(float_), format='g', precision=precision)


# @profile
def cast_into_qt(data):
    """
    Casts python data into a representation suitable for QT (usually a string)
    """
    if SIMPLE_CASTING:
        if ut.is_str(data):
            return __STR__(data)
        elif ut.is_float(data):
            # qnumber = QString.number(float(data), format='g', precision=8)
            return locale_float(data)
        elif ut.is_bool(data):
            return bool(data)
        elif ut.is_int(data):
            return int(data)
        elif isinstance(data, uuid.UUID):
            return __STR__(data)
        elif ut.isiterable(data):
            return ', '.join(map(__STR__, data))
        else:
            return __STR__(data)
    if ut.is_str(data):
        return __STR__(data)
    elif ut.is_float(data):
        # qnumber = QString.number(float(data), format='g', precision=8)
        return locale_float(data)
    elif ut.is_bool(data):
        return bool(data)
    elif ut.is_int(data):
        return int(data)
    elif isinstance(data, uuid.UUID):
        return __STR__(data)
    elif ut.isiterable(data):
        return ', '.join(map(__STR__, data))
    elif data is None:
        return 'None'
    else:
        return 'Unknown qtype: %r for data=%r' % (type(data), data)


@checks_qt_error
def cast_from_qt(var, type_=None):
    """ Casts a QVariant to data """
    if SIMPLE_CASTING:
        if var is None:
            return None
        if type_ is not None:
            reprstr = __STR__(var)
            return ut.smart_cast(reprstr, type_)
        return var
    # TODO: sip api v2 should take care of this.


def infer_coltype(column_list):
    """ Infer Column datatypes """
    try:
        coltype_list = [type(column_data[0]) for column_data in column_list]
    except Exception:
        coltype_list = [__STR__] * len(column_list)
    return coltype_list


def to_qcolor(color):
    qcolor = QtGui.QColor(*color[0:3])
    return qcolor
