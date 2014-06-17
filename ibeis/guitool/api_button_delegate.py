from __future__ import absolute_import, division, print_function
from PyQt4 import QtGui, QtCore  # NOQA
from guitool import guitool_components
import utool
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableWidget]', DEBUG=False)


#DELEGATE_BASE = QtGui.QItemDelegate
DELEGATE_BASE = QtGui.QStyledItemDelegate


def rgb_to_qcolor(rgb):
    return QtGui.QColor(*rgb[0:3])


def rgb_to_qbrush(rgb):
    return QtGui.QBrush(rgb_to_qcolor(rgb))


def paint_button(painter, option, text='button', pressed=True, bgcolor=None,
                 fgcolor=None, clicked=None, button=None, view=None):
    #http://www.qtcentre.org/archive/index.php/t-31029.html
    opt = QtGui.QStyleOptionButton()
    #opt = QtGui.QStyleOptionViewItemV4(option)
    #opt.initFrom(button)
    opt.text = text
    opt.rect = option.rect
    opt.palette = option.palette
    # FIXME: I cant set the colors!
    if bgcolor is not None:
        opt.palette.setCurrentColorGroup(QtGui.QPalette.Normal)
        opt.palette.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Button, rgb_to_qbrush(bgcolor))
        opt.palette.setBrush(QtGui.QPalette.Base, rgb_to_qbrush(bgcolor))
        opt.palette.setBrush(QtGui.QPalette.Window, rgb_to_qbrush(bgcolor))
        opt.palette.setBrush(QtGui.QPalette.ButtonText, rgb_to_qbrush(bgcolor))
        #
        opt.palette.setColor(QtGui.QPalette.Normal, QtGui.QPalette.Button, rgb_to_qcolor(bgcolor))
        opt.palette.setColor(QtGui.QPalette.Base, rgb_to_qcolor(bgcolor))
        opt.palette.setColor(QtGui.QPalette.Window, rgb_to_qcolor(bgcolor))
        opt.palette.setColor(QtGui.QPalette.ButtonText, rgb_to_qcolor(bgcolor))
        #painter.setBrush(rgb_to_qbrush(bgcolor))
    #if fgcolor is not None:
    #    opt.palette.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.ButtonText, rgb_to_qbrush(fgcolor))
    if pressed:
        opt.state = QtGui.QStyle.State_Enabled | QtGui.QStyle.State_Sunken
    else:
        opt.state = QtGui.QStyle.State_Enabled | QtGui.QStyle.State_Raised

    #style = QtGui.QApplication.style()
    painter.drawRect(option.rect)
    style = button.style()
    #print(style)
    #if view is not None:
    #view.style
    style.drawControl(QtGui.QStyle.CE_PushButton, opt, painter, button)


class APIButtonDelegate(DELEGATE_BASE):
    buttonClicked = QtCore.pyqtSignal(int, int)

    def __init__(dgt, parent=None):
        assert parent is not None, 'parent must be a view'
        DELEGATE_BASE.__init__(dgt, parent)
        dgt._pressed = None

    def get_index_butkw(dgt, qtindex):
        """ The model data for a button should be a (text, callback) tuple.  OR
        it could be a function which accepts an qtindex and returns a button """
        #data = qtindex.data()
        data = qtindex.model().data(qtindex, QtCore.Qt.DisplayRole)
        # Get info
        if isinstance(data, tuple):
            buttontup = data
        elif utool.is_funclike(data):
            func = data
            buttontup = func(qtindex)
        else:
            raise AssertionError('bad type')
        text, callback = buttontup[0:2]
        butkw = {
            #'parent': dgt.parent(),
            'text': text,
            'clicked': callback,
        }
        if len(buttontup) > 2:
            butkw['bgcolor'] = buttontup[2]
            butkw['fgcolor'] = (0, 0, 0)
        return butkw

    def paint(dgt, painter, option, qtindex):
        painter.save()
        butkw = dgt.get_index_butkw(qtindex)
        button = guitool_components.newButton(**butkw)
        pressed = dgt.is_qtindex_pressed(qtindex)
        view = dgt.parent()
        paint_button(painter, option, button=button, pressed=pressed,
                     view=view, **butkw)
        painter.restore()
        #if not qtindex.isValid():
        #    return None
        #if view.indexWidget(qtindex):
        #    return
        #else:
        #    view.setIndexWidget(qtindex, button)
        #    # The view already has a button
        #    # NOTE: this requires model::qtindex to be overwritten
        #    # and return model.createIndex(row, col, object) where
        #    # object is specified.
        #    view.setIndexWidget(qtindex, None)
        #    button = QtGui.QPushButton(text, view, clicked=view.cellButtonClicked)

    def is_qtindex_pressed(dgt, qtindex):
        return dgt._pressed is not None and dgt._pressed == (qtindex.row(), qtindex.column())

    def editorEvent(dgt, event, model, option, qtindex):
        # http://stackoverflow.com/questions/14585575/button-delegate-issue
        #print('editor event')
        event_type = event.type()
        if event_type == QtCore.QEvent.MouseButtonPress:
            # store the position that is clicked
            dgt._pressed = (qtindex.row(), qtindex.column())
            print('store')
            return True
        elif event_type == QtCore.QEvent.MouseButtonRelease:
            if dgt.is_qtindex_pressed(qtindex):
                print('emit')
                pass
            elif dgt._pressed is not None:
                print('repaint')
                pass
            dgt._pressed = None
            #print('mouse release')
            return True
        else:
            return DELEGATE_BASE.editorEvent(dgt, event, model, option, qtindex)
    #        pass
    #    #       dgt._pressed = (qtindex.row(), qtindex.column())
    #    #       return True
    #        pass
    #    else:
    #        pass
    #    #       if dgt._pressed == (qtindex.row(), qtindex.column()):
    #    #           # we are at the same place, so emit
    #    #           dgt.buttonClicked.emit(*dgt._pressed)
    #    #       elif dgt._pressed:
    #    #           # different place.
    #    #           # force a repaint on the pressed cell by emitting a dataChanged
    #    #           # Note: This is probably not the best idea
    #    #           # but I've yet to find a better solution.
    #    #           oldIndex = qtindex.model().qtindex(*dgt._pressed)
    #    #           dgt._pressed = None
    #    #           qtindex.model().dataChanged.emit(oldIndex, oldIndex)
    #    #       dgt._pressed = None
    #    #       return True
    #    #   else:
    #    # default editor event;
