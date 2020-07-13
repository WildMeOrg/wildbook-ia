# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import QtWidgets  # NOQA
from wbia.guitool.__PYQT__ import GUITOOL_PYQT_VERSION  # NOQA
import utool as ut

ut.noinject(__name__, '[guitool.main]', DEBUG=False)


IS_ROOT_WINDOW = False
QAPP = None
VERBOSE = '--verbose' in sys.argv
QUIET = '--quiet' in sys.argv


def get_qtapp():
    global QAPP
    return QAPP


class GuitoolApplication(QtWidgets.QApplication):
    """
    http://codeprogress.com/python/libraries/pyqt/showPyQTExample.php?index=378&key=QApplicationKeyPressGlobally
    """

    def __init__(self, args):
        super(GuitoolApplication, self).__init__(args)
        self.log_keys = False
        self.keylog = []

    def notify(self, receiver, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if self.log_keys:
                key = event.text()
                print('key = %r' % (key,))
                self.keylog.append(key)
            # QtWidgets.QMessageBox.information(
            #    None, "Received Key Press Event!!", "You Pressed: " + event.text())
        # Call Base Class Method to Continue Normal Event Processing
        return super(GuitoolApplication, self).notify(receiver, event)

    def start_keylog(self):
        self.log_keys = True


def ensure_qtapp():
    global IS_ROOT_WINDOW
    global QAPP
    if QAPP is not None:
        return QAPP, IS_ROOT_WINDOW
    parent_qapp = QtCore.QCoreApplication.instance()
    if parent_qapp is None:  # if not in qtconsole
        if not QUIET:
            print('[guitool] Init new QApplication')
        QAPP = GuitoolApplication(sys.argv)
        if GUITOOL_PYQT_VERSION == 4:
            QAPP.setStyle('plastique')
        else:
            # http://stackoverflow.com/questions/38154702/how-to-install-new-qstyle-for-pyqt
            # QAPP.setStyle('Windows')
            # QAPP.setStyle('WindowsXP')
            # QAPP.setStyle('WindowsVista')
            # available_styles = QtWidgets.QStyleFactory().keys()
            # print('available_styles = %r' % (available_styles,))
            # QAPP.setStyle('Fusion')
            QAPP.setStyle('GTK+')
        # QAPP.setStyle('windows')
        # QAPP.setStyle('cleanlooks')
        # QAPP.setStyle('motif')
        # QAPP.setDesktopSettingsAware(True)
        # QAPP.setStyle('cde')
        # "windows", "motif", "cde", "plastique" and "cleanlooks" and depending on the platform, "windowsxp", "windowsvista" and "macintosh"
        # print('QAPP = %r' % QAPP)
        assert QAPP is not None
        IS_ROOT_WINDOW = True
    else:
        if not QUIET:
            print('[guitool] Using parent QApplication')
        QAPP = parent_qapp
        IS_ROOT_WINDOW = False
    return QAPP, IS_ROOT_WINDOW


ensure_qapp = ensure_qtapp


def activate_qwindow(qwin):
    global QAPP
    if not QUIET:
        print('[guitool] qapp.setActiveWindow(qwin)')
    qwin.show()
    QAPP.setActiveWindow(qwin)


def qtapp_loop_nonblocking(qwin=None, **kwargs):
    """
    Fixme:
        In order to have a non-blocking qt application then the app must have been started
        with IPython.lib.inputhook.enable_gui
        import IPython.lib.inputhook
        IPython.lib.inputhook.enable_gui('qt4')
        Actually lib.inputhook is depricated

        Maybe IPython.terminal.pt_inputhooks
        import IPython.terminal.pt_inputhooks
        inputhook = IPython.terminal.pt_inputhooks.get_inputhook_func('qt4')
    """
    global QAPP
    # from IPython.lib.inputhook import enable_qt4
    import IPython.lib.guisupport

    if not QUIET:
        print('[guitool] Starting ipython qt hook')
    # enable_qt4()
    if GUITOOL_PYQT_VERSION == 4:
        IPython.lib.guisupport.start_event_loop_qt4(QAPP)
    else:
        IPython.lib.guisupport.start_event_loop_qt5(QAPP)


# if '__PYQT__' in sys.modules:
# from wbia.guitool.__PYQT__ import QtCore
# from IPython.lib.inputhook import enable_qt4
# from IPython.lib.guisupport import start_event_loop_qt4
# qapp = QtCore.QCoreApplication.instance()
# # qapp.exec_()
# print('[ut.dbg] Starting ipython qt4 hook')
# enable_qt4()
# start_event_loop_qt4(qapp)


def remove_pyqt_input_hook():
    QtCore.pyqtRemoveInputHook()


def qtapp_loop(
    qwin=None,
    ipy=False,
    enable_activate_qwin=True,
    frequency=420,
    init_signals=True,
    **kwargs,
):
    r"""
    Args:
        qwin (None): (default = None)
        ipy (bool): set to True if running with IPython (default = False)
        enable_activate_qwin (bool): (default = True)
        frequency (int): frequency to ping python interpreter (default = 420)
        init_signals (bool): if False, handle terminal signals yourself (default = True)

    CommandLine:
        python -m wbia.guitool.guitool_main --test-qtapp_loop
    """
    global QAPP
    # if not QUIET and VERBOSE:
    if not QUIET:
        print('[guitool.qtapp_loop()] ENTERING')
    print('[guitool.qtapp_loop()] starting qt app loop: qwin=%r' % (qwin,))
    if enable_activate_qwin and (qwin is not None):
        activate_qwindow(qwin)
        qwin.timer = ping_python_interpreter(frequency=frequency)
    elif qwin is None:
        print('[guitool] Warning: need to specify qwin for ctrl+c to work')
    if init_signals:
        # allow ctrl+c to exit the program
        _init_signals()
    if IS_ROOT_WINDOW:
        if not QUIET:
            print('[guitool.qtapp_loop()] qapp.exec_()  # runing main loop')
        if not ipy:
            # old_excepthook = sys.excepthook
            # def qt_excepthook(type_, value, traceback):
            #     print('QT EXCEPTION HOOK')
            #     old_excepthook(type_, value, traceback)
            #     #QAPP.quit()
            #     exit_application()
            #     sys.exit(1)
            # sys.excepthook = qt_excepthook
            try:
                retcode = QAPP.exec_()
                print('QAPP retcode = %r' % (retcode,))
                QAPP.exit(retcode)
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
        # print('lub dub')
        return None

    timer.ping_func = ping_func
    timer.timeout.connect(timer.ping_func)
    timer.start(frequency)
    return timer


# @atexit.register
def exit_application():
    if ut.NOT_QUIET:
        print('[guitool] exiting application')
    QtWidgets.qApp.quit()


def _on_ctrl_c(signal, frame):
    print('[guitool.guitool_main] Caught ctrl+c. sys.exit(0)...')
    sys.exit(0)


# -----------------------
# private init functions


def _init_signals():
    import signal

    # print('initializing qt ctrl+c signal')
    signal.signal(signal.SIGINT, _on_ctrl_c)
