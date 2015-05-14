from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtCore, QtGui
# Python
import six
from six.moves import range
import utool
import sys
import logging
from guitool.guitool_decorators import slot_
from guitool import guitool_main
#print, print_, printDBG, rrr, profile = utool.inject(__name__, '[guitool_misc]')
import utool as ut
ut.noinject(__name__, '[guitool.misc]', DEBUG=False)

WITH_GUILOG = utool.get_argflag('--guilog')
#WITH_GUILOG = not utool.get_argflag('--noguilog')


class BlockContext(object):
    def __init__(self, widget):
        self.widget = widget
        self.was_blocked = None

    def __enter__(self):
        self.was_blocked = self.widget.blockSignals(True)

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('[BlockContext] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        self.widget.blockSignals(self.was_blocked)


# Qt object that will send messages (as signals) to the frontend gui_write slot
class GUILoggingSender(QtCore.QObject):
    write_ = QtCore.pyqtSignal(str)
    def __init__(self, write_slot):
        QtCore.QObject.__init__(self)
        self.write_.connect(write_slot)

    def write_gui(self, msg):
        self.write_.emit(str(msg))


class GUILoggingHandler(logging.StreamHandler):
    """
    A handler class which sends messages to to a connected QSlot
    """
    def __init__(self, write_slot):
        super(GUILoggingHandler, self).__init__()
        self.sender = GUILoggingSender(write_slot)

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            self.sender.write_.emit(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class QLoggedOutput(QtGui.QTextEdit):
    def __init__(self, parent=None):
        QtGui.QTextEdit.__init__(self, parent)
        if WITH_GUILOG:
            self.logging_handler = GUILoggingHandler(self.gui_write)
            utool.add_logging_handler(self.logging_handler)

    @slot_(str)
    def gui_write(outputEdit, msg_):
        # Slot for teed log output
        app = guitool_main.get_qtapp()
        # Write msg to text area
        outputEdit.moveCursor(QtGui.QTextCursor.End)
        # TODO: Find out how to do backspaces in textEdit
        msg = str(msg_)
        if msg.find('\b') != -1:
            msg = msg.replace('\b', '') + '\n'
        outputEdit.insertPlainText(msg)
        if app is not None:
            app.processEvents()

    @slot_()
    def gui_flush(outputEdit):
        app = guitool_main.get_qtapp()
        if app is not None:
            app.processEvents()


def get_cplat_tab_height():
    if sys.platform.startswith('darwin'):
        tab_height = 21
    else:
        tab_height = 30
    return tab_height


def get_view_selection_as_str(view):
    """
    Taken from here http://stackoverflow.com/questions/3135737/
        copying-part-of-qtableview
    TODO: Make this pythonic
    """
    model = view.model()
    selection_model = view.selectionModel()
    qindex_list = selection_model.selectedIndexes()
    qindex_list = sorted(qindex_list)
    print('[guitool] %d cells selected' % len(qindex_list))
    if len(qindex_list) == 0:
        return
    copy_table = []
    previous = qindex_list[0]

    def astext(data):
        """ Helper which casts model data to a string """
        if not isinstance(data, six.string_types):
            text = repr(data)
        else:
            text = str(data)
        #try:
        #    text = str(data)
        #    #if isinstance(data, QtCore.Q__Variant):
        #    #    text = str(data.toString())
        #    #elif isinstance(data, QtCore.Q__String):
        #    #    text = str(data)
        #    #else:
        #    #    text = str(data)
        #except Exception as ex:
        #    text = repr(ex)
        return text.replace('\n', '<NEWLINE>').replace(',', '<COMMA>')

    #
    for ix in range(1, len(qindex_list)):
        text = astext(model.data(previous))
        copy_table.append(text)
        qindex = qindex_list[ix]

        if qindex.row() != previous.row():
            copy_table.append('\n')
        else:
            copy_table.append(', ')
        previous = qindex

    # Do last element in list
    text = astext(model.data(qindex_list[-1]))
    copy_table.append(text)
    #copy_table.append('\n')
    copy_str = str(''.join(copy_table))
    return copy_str
