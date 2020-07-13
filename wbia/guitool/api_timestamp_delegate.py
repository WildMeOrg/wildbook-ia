# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import QtGui as QtWidgets

# from wbia.guitool import guitool_components
# (print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APIButtonWidget]', DEBUG=False)
import utool as ut

ut.noinject(__name__, '[api_timestamp_delegate]', DEBUG=False)


DELEGATE_BASE = QtWidgets.QItemDelegate
# DELEGATE_BASE = QtWidgets.QStyledItemDelegate


class APITimestampDelegate(DELEGATE_BASE):
    def __init__(dgt, parent=None):
        assert parent is not None, 'parent must be a view'
        DELEGATE_BASE.__init__(dgt, parent)

    def paint(dgt, painter, option, qtindex):
        painter.save()
        data = qtindex.model().data(qtindex, QtCore.Qt.DisplayRole)
        print(data)

        painter.restore()

    # def editorEvent(dgt, event, model, option, qtindex):
    #    event_type = event.type()
    #    if event_type == QtCore.QEvent.MouseButtonPress:
    #        # store the position that is clicked
    #        dgt._pressed = (qtindex.row(), qtindex.column())
    #        if utool.VERBOSE:
    #            print('store')
    #        return True
    #    elif event_type == QtCore.QEvent.MouseButtonRelease:
    #        if dgt.is_qtindex_pressed(qtindex):
    #            print('emit')
    #            dgt.button_clicked.emit(qtindex)
    #            pass
    #        elif dgt._pressed is not None:
    #            # different place.
    #            # force a repaint on the pressed cell by emitting a dataChanged
    #            # Note: This is probably not the best idea
    #            # but I've yet to find a better solution.
    #            print('repaint')
    #            oldIndex = qtindex.model().index(*dgt._pressed)
    #            dgt._pressed = None
    #            qtindex.model().dataChanged.emit(oldIndex, oldIndex)
    #            pass
    #        dgt._pressed = None
    #        #print('mouse release')
    #        return True
    #    elif event_type == QtCore.QEvent.Leave:
    #        print('leave')
    #    elif event_type == QtCore.QEvent.MouseButtonDblClick:
    #        print('doubleclick')
    #    else:
    #        print('event_type = %r' % event_type)
    #        return DELEGATE_BASE.editorEvent(dgt, event, model, option, qtindex)
