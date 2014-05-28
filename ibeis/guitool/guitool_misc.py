from __future__ import absolute_import, division, print_function
# Python
import sys
import logging
# Qt
from PyQt4 import QtCore


class GUILoggingHandler(logging.StreamHandler):
    '''
    A handler class which sends messages to to a connected QSlot
    '''
    def __init__(self, write_slot):
        super(GUILoggingHandler, self).__init__()

        # Qt object that will send messages (as signals) to the frontend gui_write slot
        class GUILoggingSender(QtCore.QObject):
            write_ = QtCore.pyqtSignal(str)
            def __init__(self, write_slot):
                super(GUILoggingSender, self).__init__()
                self.write_.connect(write_slot)

            def write_gui(self, msg):
                self.write_.emit(str(msg))

        self.sender = GUILoggingSender(write_slot)

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            self.sender.write_.emit(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_cplat_tab_height():
    if sys.platform.startswith('darwin'):
        tab_height = 21
    else:
        tab_height = 30
    return tab_height


