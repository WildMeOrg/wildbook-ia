from __future__ import division, print_function
# Python
import sys
from PyQt4 import QtCore, QtGui

IS_ROOT = False
QAPP = None


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
    print('[guitool.qtapp_loop()] qapp.setActiveWindow(back.front)')
    global QAPP
    back.front.show()
    QAPP.setActiveWindow(back.front)


def qtapp_loop_nonblocking(back=None, **kwargs):
    global QAPP
    from IPython.lib.inputhook import enable_qt4
    from IPython.lib.guisupport import start_event_loop_qt4
    print('[guitool] Starting ipython qt4 hook')
    enable_qt4()
    start_event_loop_qt4(QAPP)


def qtapp_loop(back=None, **kwargs):
    print('[guitool] qtapp_loop()')
    global QAPP
    if back is not None:
        activate_qwindow(back)
        back.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT:
        print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        QAPP.exec_()
    else:
        print('[guitool.qtapp_loop()] not execing')


def ping_python_interpreter(frequency=4200):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    print('[guitool] pinging python interpreter for ctrl+c freq=%r' % frequency)
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer


def exit_application():
    print('[guitool] exiting application')
    QtGui.qApp.quit()
