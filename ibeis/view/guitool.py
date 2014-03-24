from __future__ import division, print_function
# Python
import sys
import logging
# Qt
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA
# HotSpotter

IS_ROOT = False
QAPP = None
DEBUG = False

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


#---------------
# SLOT DECORATORS


def slot_(*types):  # This is called at wrap time to get args
    '''
    wrapper around pyqtslot decorator
    *args = types
    '''
    def pyqtSlotWrapper(func):
        func_name = func.func_name

        @QtCore.pyqtSlot(*types, name=func.func_name)
        def slot_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result

        slot_wrapper.func_name = func_name
        return slot_wrapper
    return pyqtSlotWrapper


#/SLOT DECORATOR
#---------------


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblock(func):
    def bacblock_wrapper(back, *args, **kwargs):
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            import traceback
            back.front.blockSignals(wasBlocked_)
            print('!!!!!!!!!!!!!')
            print('[guitool] caught exception in %r' % func.func_name)
            print(traceback.format_exc())
            back.user_info('Error:\nex=%r' % ex)
            raise
        back.front.blockSignals(wasBlocked_)
        return result
    bacblock_wrapper.func_name = func.func_name
    return bacblock_wrapper


# DRAWING DECORATOR
def drawing(func):
    'Wraps a class function and draws windows on completion'
    def drawing_wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if kwargs.get('dodraw', True):
            pass
        return result
    drawing_wrapper.func_name = func.func_name
    return drawing_wrapper


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


def select_orientation():
    print('[guitool] select_orientation()')
    pass


def select_roi():
    print('[guitool] select_roi()')


def init_qtapp():
    global IS_ROOT
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT
    QAPP = QtCore.QCoreApplication.instance()
    IS_ROOT = QAPP is None
    if IS_ROOT:  # if not in qtconsole
        print('[guitool] Initializing QApplication')
        QAPP = QtGui.QApplication(sys.argv)
    try:
        # You are not root if you are in IPYTHON
        __IPYTHON__
        IS_ROOT = False
    except NameError:
        pass
    return QAPP, IS_ROOT


def qtapp_loop(back=None, **kwargs):
    print('[guitool] qtapp_loop()')
    if back is not None:
        print('[guitool.qtapp_loop()] qapp.setActiveWindow(back.front)')
        QAPP.setActiveWindow(back.front)
        back.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT:
        print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        QAPP.exec_()
    else:
        print('[guitool.qtapp_loop()] not execing')


def ping_python_interpreter(frequency=4200):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer


def exit_application():
    print('[guitool] exiting application')
    QtGui.qApp.quit()
