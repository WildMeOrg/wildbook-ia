from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__ import QtWidgets  # NOQA
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


def find_used_chars(name_list):
    """ Move to guitool """
    used_chars = []
    for name in name_list:
        index = name.find('&')
        if index == -1 or index + 1 >= len(name):
            continue
        char = name[index + 1]
        used_chars.append(char)
    return used_chars


def make_word_hotlinks(name_list, used_chars=[], after_colon=False):
    """ Move to guitool

    Args:
        name_list (list):
        used_chars (list): (default = [])

    Returns:
        list: hotlinked_name_list

    CommandLine:
        python -m guitool.guitool_misc --exec-make_word_hotlinks

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_misc import *  # NOQA
        >>> name_list = ['occlusion', 'occlusion:large', 'occlusion:medium', 'occlusion:small', 'lighting', 'lighting:shadowed', 'lighting:overexposed', 'lighting:underexposed']
        >>> used_chars = []
        >>> hotlinked_name_list = make_word_hotlinks(name_list, used_chars)
        >>> result = ('hotlinked_name_list = %s' % (str(hotlinked_name_list),))
        >>> print(result)
    """
    seen_ = set(used_chars)
    hotlinked_name_list = []
    for name in name_list:
        added = False
        if after_colon:
            split_chars = name.split(':')
            offset = len(':'.join(split_chars[:-1])) + 1
            avail_chars = split_chars[-1]
        else:
            offset = 0
            avail_chars = name
        for count, char in enumerate(avail_chars, start=offset):
            char = char.upper()
            if char not in seen_:
                added = True
                seen_.add(char)
                linked_name = name[:count] + '&' + name[count:]
                hotlinked_name_list.append(linked_name)
                break
        if not added:
            # Cannot hotlink this name
            hotlinked_name_list.append(name)
    return hotlinked_name_list


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


class QLoggedOutput(QtWidgets.QTextEdit):
    def __init__(self, parent=None, visible=True):
        super(QLoggedOutput, self).__init__(parent)
        #QtWidgets.QTextEdit.__init__(self, parent)
        self.setAcceptRichText(False)
        self.setReadOnly(True)
        self.setVisible(visible)
        self.logging_handler = None
        if visible:
            self._initialize_handler()

    def setVisible(self, flag):
        if flag and self.logging_handler is None:
            # Make sure handler is initialized on first go
            self._initialize_handler()
        super(QLoggedOutput, self).setVisible(flag)

    def _initialize_handler(self):
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
