# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import time
import utool as ut
import matplotlib as mpl
from wbia.plottool import custom_figure

# from .custom_constants import golden_wh


SLEEP_TIME = 0.01
__QT4_WINDOW_LIST__ = []
ut.noinject(__name__, '[fig_presenter]')


VERBOSE = ut.get_argflag(('--verbose-fig', '--verbfig', '--verb-pt'))
# (print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[fig_presenter]', DEBUG=True)


def unregister_qt4_win(win):
    global __QT4_WINDOW_LIST__
    if win == 'all':
        __QT4_WINDOW_LIST__ = []
    else:
        try:
            # index = __QT4_WINDOW_LIST__.index(win)
            __QT4_WINDOW_LIST__.remove(win)
        except ValueError:
            pass


def register_qt4_win(win):
    global __QT4_WINDOW_LIST__
    __QT4_WINDOW_LIST__.append(win)


# ---- GENERAL FIGURE COMMANDS ----


def set_geometry(fnum, x, y, w, h):
    fig = custom_figure.ensure_fig(fnum)
    qtwin = get_figure_window(fig)
    qtwin.setGeometry(x, y, w, h)


def get_geometry(fnum):
    fig = custom_figure.ensure_fig(fnum)
    qtwin = get_figure_window(fig)
    (x1, y1, x2, y2) = qtwin.geometry().getCoords()
    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
    return (x, y, w, h)


# def get_screen_info():
#    # TODO Move dependency to guitool
#    desktop = QtWidgets.QDesktopWidget()
#    mask = desktop.mask()  # NOQA
#    layout_direction = desktop.layoutDirection()  # NOQA
#    screen_number = desktop.screenNumber()  # NOQA
#    normal_geometry = desktop.normalGeometry()  # NOQA
#    num_screens = desktop.screenCount()  # NOQA
#    avail_rect = desktop.availableGeometry()  # NOQA
#    screen_rect = desktop.screenGeometry()  # NOQA
#    QtWidgets.QDesktopWidget().availableGeometry().center()  # NOQA
#    normal_geometry = desktop.normalGeometry()  # NOQA


# @profile
def get_all_figures():
    manager_list = mpl._pylab_helpers.Gcf.get_all_fig_managers()
    all_figures = []
    # Make sure you dont show figures that this module closed
    for manager in manager_list:
        try:
            fig = manager.canvas.figure
        except AttributeError:
            continue
        if not fig.__dict__.get('df2_closed', False):
            all_figures.append(fig)
    # Return all the figures sorted by their number
    all_figures = sorted(all_figures, key=lambda fig: fig.number)
    return all_figures


def get_all_qt4_wins():
    return __QT4_WINDOW_LIST__


def all_figures_show():
    if VERBOSE:
        print('all_figures_show')
    if not ut.get_argflag('--noshow'):
        for fig in get_all_figures():
            time.sleep(SLEEP_TIME)
            show_figure(fig)
            # fig.show()
            # fig.canvas.draw()


def show_figure(fig):
    try:
        fig.show()
        fig.canvas.draw()
    except AttributeError as ex:
        if not hasattr(fig, '_no_raise_plottool'):
            ut.printex(
                ex, '[pt] probably registered made figure with Qt.', iswarning=True
            )


def all_figures_tight_layout():
    if '--noshow' not in sys.argv:
        for fig in iter(get_all_figures()):
            fig.tight_layout()
            time.sleep(SLEEP_TIME)


def get_main_win_base():
    if hasattr(mpl.backends, 'backend_qt4'):
        backend = mpl.backends.backend_qt4
    else:
        backend = mpl.backends.backend_qt5
    try:
        QMainWin = backend.MainWindow
    except Exception as ex:
        try:
            ut.printex(ex, 'warning', '[fig_presenter]')
            # from wbia.guitool.__PYQT__ import QtGui
            QMainWin = backend.QtWidgets.QMainWindow
        except Exception as ex1:
            ut.printex(ex1, 'warning', '[fig_presenter]')
            QMainWin = object
    return QMainWin


def get_all_windows():
    """ Returns all mpl figures and registered qt windows """
    try:
        all_figures = get_all_figures()
        all_qt4wins = get_all_qt4_wins()
        all_wins = all_qt4wins + [get_figure_window(fig) for fig in all_figures]
        return all_wins
    except AttributeError as ex:
        ut.printex(ex, 'probably using a windowless backend', iswarning=True)
        return []


# @profile
def all_figures_tile(
    max_rows=None,
    row_first=True,
    no_tile=False,
    monitor_num=None,
    percent_w=None,
    percent_h=None,
    hide_toolbar=True,
):
    """
    Lays out all figures in a grid. if wh is a scalar, a golden ratio is used
    """
    # print('[plottool] all_figures_tile()')
    if no_tile:
        return

    current_backend = mpl.get_backend()
    if not current_backend.startswith('Qt'):
        # print('current_backend=%r is not a Qt backend. cannot tile.' % current_backend)
        return

    all_wins = get_all_windows()
    num_wins = len(all_wins)
    if num_wins == 0:
        return

    from wbia.plottool import screeninfo

    valid_positions = screeninfo.get_valid_fig_positions(
        num_wins,
        max_rows,
        row_first,
        monitor_num,
        percent_w=percent_w,
        percent_h=percent_h,
    )

    QMainWin = get_main_win_base()
    for ix, win in enumerate(all_wins):
        isqt4_mpl = isinstance(win, QMainWin)
        from wbia.guitool.__PYQT__ import QtGui  # NOQA
        from wbia.guitool.__PYQT__ import QtWidgets  # NOQA

        isqt4_back = isinstance(win, QtWidgets.QMainWindow)
        isqt4_widget = isinstance(win, QtWidgets.QWidget)
        (x, y, w, h) = valid_positions[ix]
        # printDBG('tile %d-th win: xywh=%r' % (ix, (x, y, w, h)))
        if not isqt4_mpl and not isqt4_back and not isqt4_widget:
            raise NotImplementedError('%r-th Backend %r is not a Qt Window' % (ix, win))
        try:
            if hide_toolbar:
                toolbar = win.findChild(QtWidgets.QToolBar)
                toolbar.setVisible(False)
            win.setGeometry(x, y, w, h)
        except Exception as ex:
            ut.printex(ex)


def all_figures_bring_to_front():
    try:
        all_figures = get_all_figures()
        for fig in iter(all_figures):
            bring_to_front(fig)
    except Exception as ex:
        if not hasattr(fig, '_no_raise_plottool'):
            ut.printex(ex, iswarning=True)


def close_all_figures():
    print('[pt] close_all_figures')
    all_figures = get_all_figures()
    for fig in iter(all_figures):
        close_figure(fig)


def close_figure(fig):
    print('[pt] close_figure')
    fig.clf()
    fig.df2_closed = True
    qtwin = get_figure_window(fig)
    qtwin.close()


def get_figure_window(fig):
    try:
        qwin = fig.canvas.manager.window
    except AttributeError:
        qwin = fig.canvas.window()
    return qwin


def bring_to_front(fig):
    if VERBOSE:
        print('[pt] bring_to_front')
    # what is difference between show and show normal?
    qtwin = get_figure_window(fig)
    qtwin.raise_()
    # if not ut.WIN32:
    # NOT sure on the correct order of these
    # can cause the figure geometry to be unset
    from wbia.guitool.__PYQT__.QtCore import Qt

    qtwin.activateWindow()
    qtwin.setWindowFlags(Qt.WindowStaysOnTopHint)
    qtwin.setWindowFlags(Qt.WindowFlags(0))
    qtwin.show()


def show():
    if VERBOSE:
        print('[pt] show')
    all_figures_show()
    all_figures_bring_to_front()
    # plt.show()


def reset():
    if VERBOSE:
        print('[pt] reset')
    close_all_figures()


def draw():
    if VERBOSE:
        print('[pt] draw')
    all_figures_show()


def update():
    if VERBOSE:
        print('[pt] update')
    draw()
    all_figures_bring_to_front()


def iupdate():
    if VERBOSE:
        print('[pt] iupdate')
    if ut.inIPython():
        update()


iup = iupdate


def present(*args, **kwargs):
    """
    basically calls show if not embeded.

    Kwargs:
        max_rows, row_first, no_tile, monitor_num, percent_w, percent_h,
        hide_toolbar

    CommandLine:
        python -m wbia.plottool.fig_presenter present

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.fig_presenter import *  # NOQA
        >>> result = present()
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    if VERBOSE:
        print('[pt] present')
    if not ut.get_argflag('--noshow'):
        # print('[fig_presenter] Presenting figures...')
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        all_figures_tile(*args, **kwargs)
        # Both of these lines cause the weird non-refresh black border behavior
        all_figures_show()
        all_figures_bring_to_front()
