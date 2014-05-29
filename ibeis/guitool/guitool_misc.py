from __future__ import absolute_import, division, print_function
# Python
import utool
import sys
import logging
from .guitool_decorators import slot_
from . import guitool_main
# Qt
from PyQt4 import QtCore, QtGui
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[guitool_misc]')


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
