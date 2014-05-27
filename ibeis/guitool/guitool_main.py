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
        #print('QAPP = %r' % QAPP)
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
ensure_qapp = ensure_qtapp


def activate_qwindow(qwin):
    global QAPP
    if not QUIET:
        print('[guitool.qtapp_loop()] qapp.setActiveWindow(qwin)')
    if hasattr(qwin, 'front'):
        qwin = qwin.front
    qwin.show()
    QAPP.setActiveWindow(qwin)


def qtapp_loop_nonblocking(qwin=None, **kwargs):
    global QAPP
    from IPython.lib.inputhook import enable_qt4
    from IPython.lib.guisupport import start_event_loop_qt4
    if not QUIET:
        print('[guitool] Starting ipython qt4 hook')
    enable_qt4()
    start_event_loop_qt4(QAPP)


def qtapp_loop(qwin=None, ipy=False, **kwargs):
    global QAPP
    if not QUIET and VERBOSE:
        print('[guitool.qtapp_loop()]')
    if qwin is not None:
        activate_qwindow(qwin)
        qwin.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT:
        if not QUIET and VERBOSE:
            print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        if ipy:
            pass
        else:
            old_excepthook = sys.excepthook
            def qt_excepthook(type_, value, traceback):
                print('QT EXCEPTION HOOK')
                old_excepthook(type_, value, traceback)
                QAPP.quit()
                sys.exit(1)
            sys.excepthook = qt_excepthook
            try:
                QAPP.exec_()
            except Exception as ex:
                print('QException: %r' % ex)
                raise
    else:
        if not QUIET and VERBOSE:
            print('[guitool.qtapp_loop()] not execing')


def ping_python_interpreter(frequency=420):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    if not QUIET and VERBOSE:
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
