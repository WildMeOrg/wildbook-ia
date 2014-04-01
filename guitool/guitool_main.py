from __future__ import division, print_function
# Python
import atexit
import sys
from PyQt4 import QtCore, QtGui

import sip
if hasattr(sip, 'setdestroyonexit'):
    sip.setdestroyonexit(False)  # This prevents a crash on windows

IS_ROOT = False
QAPP = None
VERBOSE = '--verbose' in sys.argv
QUIET = '--quiet' in sys.argv


def get_qtapp():
    global QAPP
    return QAPP


def ensure_qtapp():
    global IS_ROOT
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT
    QAPP = QtCore.QCoreApplication.instance()
    IS_ROOT = QAPP is None
    if IS_ROOT:  # if not in qtconsole
        if not QUIET:
            print('[guitool] Initializing QApplication')
        QAPP = QtGui.QApplication(sys.argv)
    try:
        # You are not root if you are in IPYTHON
        __IPYTHON__
        IS_ROOT = False
    except NameError:
        pass
    return QAPP, IS_ROOT

init_qtapp = ensure_qtapp


def activate_qwindow(back):
    if not QUIET:
        print('[guitool.qtapp_loop()] qapp.setActiveWindow(back.front)')
    global QAPP
    back.front.show()
    QAPP.setActiveWindow(back.front)


def qtapp_loop_nonblocking(back=None, **kwargs):
    global QAPP
    from IPython.lib.inputhook import enable_qt4
    from IPython.lib.guisupport import start_event_loop_qt4
    if not QUIET:
        print('[guitool] Starting ipython qt4 hook')
    enable_qt4()
    start_event_loop_qt4(QAPP)


def qtapp_loop(back=None, **kwargs):
    if not QUIET:
        print('[guitool] qtapp_loop()')
    global QAPP
    if back is not None:
        activate_qwindow(back)
        back.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT:
        if not QUIET:
            print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        QAPP.exec_()
    else:
        if not QUIET:
            print('[guitool.qtapp_loop()] not execing')


def ping_python_interpreter(frequency=4200):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    if not QUIET:
        print('[guitool] pinging python interpreter for ctrl+c freq=%r' % frequency)
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer


def exit_application():
    if not QUIET:
        print('[guitool] exiting application')
    QtGui.qApp.quit()


atexit.register(exit_application)
