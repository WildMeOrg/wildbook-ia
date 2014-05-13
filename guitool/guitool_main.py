from __future__ import absolute_import, division, print_function
# Python
import atexit
import sys
from PyQt4 import QtCore, QtGui
import sip
if hasattr(sip, 'setdestroyonexit'):
    sip.setdestroyonexit(False)  # This prevents a crash on windows
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[guitool]', DEBUG=False)


IS_ROOT = False
QAPP = None
VERBOSE = '--verbose' in sys.argv
QUIET = '--quiet' in sys.argv


def get_qtapp():
    global QAPP
    return QAPP


@profile
def ensure_qtapp():
    global IS_ROOT
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT
    parent_qapp = QtCore.QCoreApplication.instance()
    if parent_qapp is None:  # if not in qtconsole
        if not QUIET:
            print('[guitool] Init new QApplication')
        QAPP = QtGui.QApplication(sys.argv)
        print('QAPP = %r' % QAPP)
        assert QAPP is not None
        IS_ROOT = True
    else:
        if not QUIET:
            print('[guitool] Using parent QApplication')
        QAPP = parent_qapp
        IS_ROOT = False
    try:
        # You are not root if you are in IPYTHON
        __IPYTHON__
    except NameError:
        pass
    return QAPP, IS_ROOT

init_qtapp = ensure_qtapp


@profile
def activate_qwindow(back):
    if not QUIET:
        print('[guitool.qtapp_loop()] qapp.setActiveWindow(back.front)')
    global QAPP
    back.front.show()
    QAPP.setActiveWindow(back.front)


@profile
def qtapp_loop_nonblocking(back=None, **kwargs):
    global QAPP
    from IPython.lib.inputhook import enable_qt4
    from IPython.lib.guisupport import start_event_loop_qt4
    if not QUIET:
        print('[guitool] Starting ipython qt4 hook')
    enable_qt4()
    start_event_loop_qt4(QAPP)


@profile
def qtapp_loop(back=None, ipy=False, **kwargs):
    global QAPP
    if not QUIET:
        print('[guitool.qtapp_loop()]')
    if back is not None:
        activate_qwindow(back)
        back.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT:
        if not QUIET:
            print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        if ipy:
            pass
        else:
            QAPP.exec_()
    else:
        if not QUIET:
            print('[guitool.qtapp_loop()] not execing')


@profile
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
