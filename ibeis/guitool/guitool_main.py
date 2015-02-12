from __future__ import absolute_import, division, print_function
# Python
#import atexit
import sys
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__.QtCore import pyqtRemoveInputHook
import utool
#print, print_, printDBG, rrr, profile = utool.inject(__name__, '[guitool]', DEBUG=False)
import utool as ut
ut.noinject(__name__, '[guitool.main]', DEBUG=False)


IS_ROOT_WINDOW = False
QAPP = None
VERBOSE = '--verbose' in sys.argv
QUIET = '--quiet' in sys.argv


def get_qtapp():
    global QAPP
    return QAPP


def ensure_qtapp():
    global IS_ROOT_WINDOW
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT_WINDOW
    parent_qapp = QtCore.QCoreApplication.instance()
    if parent_qapp is None:  # if not in qtconsole
        if not QUIET:
            print('[guitool] Init new QApplication')
        QAPP = QtGui.QApplication(sys.argv)
        #print('QAPP = %r' % QAPP)
        assert QAPP is not None
        IS_ROOT_WINDOW = True
    else:
        if not QUIET:
            print('[guitool] Using parent QApplication')
        QAPP = parent_qapp
        IS_ROOT_WINDOW = False
    try:
        # You are not root if you are in IPYTHON
        __IPYTHON__
    except NameError:
        pass
    return QAPP, IS_ROOT_WINDOW

try:
    init_qtapp = ensure_qtapp
    ensure_qapp = ensure_qtapp
except Exception as e:
    print(e)
    print("CAUGHT EXCEPTION")
    sys.exit(0)


def activate_qwindow(qwin):
    global QAPP
    if not QUIET:
        print('[guitool] qapp.setActiveWindow(qwin)')
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


#if '__PYQT__' in sys.modules:
    #from guitool.__PYQT__ import QtCore
    #from IPython.lib.inputhook import enable_qt4
    #from IPython.lib.guisupport import start_event_loop_qt4
    #qapp = QtCore.QCoreApplication.instance()
    ##qapp.exec_()
    #print('[utool.dbg] Starting ipython qt4 hook')
    #enable_qt4()
    #start_event_loop_qt4(qapp)


def remove_pyqt_input_hook():
    pyqtRemoveInputHook()


def qtapp_loop(qwin=None, ipy=False, enable_activate_qwin=True, **kwargs):
    global QAPP
    #if not QUIET and VERBOSE:
    if not QUIET:
        print('[guitool.qtapp_loop()] ENTERING')
    print('[guitool.qtapp_loop()] starting qt app loop: qwin=%r' % (qwin,))
    if enable_activate_qwin and (qwin is not None):
        activate_qwindow(qwin)
        qwin.timer = ping_python_interpreter(**kwargs)
    if IS_ROOT_WINDOW:
        if not QUIET:
            print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        if not ipy:
            old_excepthook = sys.excepthook
            def qt_excepthook(type_, value, traceback):
                print('QT EXCEPTION HOOK')
                old_excepthook(type_, value, traceback)
                #QAPP.quit()
                exit_application()
                sys.exit(1)
            #sys.excepthook = qt_excepthook
            try:
                QAPP.exec_()
            except Exception as ex:
                print('QException: %r' % ex)
                raise
    else:
        if not QUIET:
            print('[guitool.qtapp_loop()] not execing')
    if not QUIET:
        print('[guitool.qtapp_loop()] EXITING')


def ping_python_interpreter(frequency=420):  # 4200):
    """ Create a QTimer which lets the python catch ctrl+c """
    if not QUIET and VERBOSE:
        print('[guitool] pinging python interpreter for ctrl+c freq=%r' % frequency)
    timer = QtCore.QTimer()
    def ping_func():
        return None
    timer.ping_func = ping_func
    timer.timeout.connect(timer.ping_func)
    timer.start(frequency)
    return timer


#@atexit.register
def exit_application():
    if utool.NOT_QUIET:
        print('[guitool] exiting application')
    QtGui.qApp.quit()
