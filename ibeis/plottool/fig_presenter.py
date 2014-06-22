from __future__ import absolute_import, division, print_function
import utool
import sys
import textwrap
import time
import warnings
# maptlotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
# PyQt
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
# Science
from .custom_figure import get_fig
from . import screeninfo
#from .custom_constants import golden_wh


SLEEP_TIME = .05
__QT4_WINDOW_LIST__ = []
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[fig_presenter]', DEBUG=False)


def unregister_qt4_win(win):
    global __QT4_WINDOW_LIST__
    if win == 'all':
        __QT4_WINDOW_LIST__ = []
    else:
        try:
            #index = __QT4_WINDOW_LIST__.index(win)
            __QT4_WINDOW_LIST__.remove(win)
        except ValueError:
            pass


def register_qt4_win(win):
    global __QT4_WINDOW_LIST__
    __QT4_WINDOW_LIST__.append(win)


# ---- GENERAL FIGURE COMMANDS ----

def set_geometry(fnum, x, y, w, h):
    fig = get_fig(fnum)
    qtwin = fig.canvas.manager.window
    qtwin.setGeometry(x, y, w, h)


def get_geometry(fnum):
    fig = get_fig(fnum)
    qtwin = fig.canvas.manager.window
    (x1, y1, x2, y2) = qtwin.geometry().getCoords()
    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
    return (x, y, w, h)


def get_screen_info():
    from PyQt4 import Qt, QtGui  # NOQA
    desktop = QtGui.QDesktopWidget()
    mask = desktop.mask()  # NOQA
    layout_direction = desktop.layoutDirection()  # NOQA
    screen_number = desktop.screenNumber()  # NOQA
    normal_geometry = desktop.normalGeometry()  # NOQA
    num_screens = desktop.screenCount()  # NOQA
    avail_rect = desktop.availableGeometry()  # NOQA
    screen_rect = desktop.screenGeometry()  # NOQA
    QtGui.QDesktopWidget().availableGeometry().center()  # NOQA
    normal_geometry = desktop.normalGeometry()  # NOQA


#@profile
def get_all_figures():
    all_figures_ = [manager.canvas.figure for manager in
                    mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    all_figures = []
    # Make sure you dont show figures that this module closed
    for fig in iter(all_figures_):
        if 'df2_closed' not in fig.__dict__.keys() or not fig.df2_closed:
            all_figures.append(fig)
    # Return all the figures sorted by their number
    all_figures = sorted(all_figures, key=lambda fig: fig.number)
    return all_figures


def get_all_qt4_wins():
    return __QT4_WINDOW_LIST__


def all_figures_show():
    if '--noshow' not in sys.argv:
        for fig in iter(get_all_figures()):
            time.sleep(SLEEP_TIME)
            fig.show()
            fig.canvas.draw()


def all_figures_tight_layout():
    if '--noshow' not in sys.argv:
        for fig in iter(get_all_figures()):
            fig.tight_layout()
            #adjust_subplots()
            time.sleep(SLEEP_TIME)


def get_main_win_base():
    try:
        QMainWin = mpl.backends.backend_qt4.MainWindow
    except Exception as ex:
        try:
            utool.printex(ex, 'warning', '[df2]')
            QMainWin = mpl.backends.backend_qt4.QtGui.QMainWindow
        except Exception as ex1:
            utool.printex(ex1, 'warning', '[df2]')
            QMainWin = object
    return QMainWin


def get_all_windows():
    """ Returns all mpl figures and registered qt windows """
    try:
        all_figures = get_all_figures()
        all_qt4wins = get_all_qt4_wins()
        all_wins = all_qt4wins + [fig.canvas.manager.window for fig in all_figures]
        return all_wins
    except AttributeError as ex:
        utool.printex(ex, 'probably using a windowless backend',
                      iswarning=True)
        return []


#@profile
def all_figures_tile(max_rows=None,
                     row_first=True,
                     no_tile=False,
                     monitor_num=None,
                     **kwargs):
    """
    Lays out all figures in a grid. if wh is a scalar, a golden ratio is used
    """
    print('[plottool] all_figures_tile()')
    if no_tile:
        return

    all_wins = get_all_windows()
    num_wins = len(all_wins)
    if num_wins == 0:
        return

    valid_positions = screeninfo.get_valid_fig_positions(num_wins, max_rows,
                                                         row_first, monitor_num,
                                                         adaptive=True)

    for ix, win in enumerate(all_wins):
        QMainWin = get_main_win_base()
        isqt4_mpl = isinstance(win, QMainWin)
        isqt4_back = isinstance(win, QtGui.QMainWindow)
        isqt4_widget = isinstance(win, QtGui.QWidget)
        (x, y, w, h) = valid_positions[ix]
        printDBG('tile %d-th win: xywh=%r' % (ix, (x, y, w, h)))
        if not isqt4_mpl and not isqt4_back and not isqt4_widget:
            raise NotImplementedError('%r-th Backend %r is not a Qt Window' %
                                      (ix, win))
        try:
            win.setGeometry(x, y, w, h)
        except Exception as ex:
            print(ex)


def all_figures_bring_to_front():
    try:
        all_figures = get_all_figures()
        for fig in iter(all_figures):
            bring_to_front(fig)
    except Exception as ex:
        print(ex)


def close_all_figures():
    all_figures = get_all_figures()
    for fig in iter(all_figures):
        close_figure(fig)


def close_figure(fig):
    fig.clf()
    fig.df2_closed = True
    qtwin = fig.canvas.manager.window
    qtwin.close()


def bring_to_front(fig):
    #what is difference between show and show normal?
    qtwin = fig.canvas.manager.window
    qtwin.raise_()
    #if not utool.WIN32:
    # NOT sure on the correct order of these
    # can cause the figure geometry to be unset
    qtwin.activateWindow()
    qtwin.setWindowFlags(Qt.WindowStaysOnTopHint)
    qtwin.setWindowFlags(Qt.WindowFlags(0))
    qtwin.show()


def show():
    all_figures_show()
    all_figures_bring_to_front()
    plt.show()


def reset():
    close_all_figures()


def draw():
    all_figures_show()


def update():
    draw()
    all_figures_bring_to_front()


def iupdate():
    if utool.inIPython():
        update()

iup = iupdate


def present(*args, **kwargs):
    'execing present should cause IPython magic'
    if '--noshow' not in sys.argv:
        #print('[df2] Presenting figures...')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_figures_tile(*args, **kwargs)
            all_figures_show()
            all_figures_bring_to_front()
        # Return an exec string
    execstr = utool.ipython_execstr()
    execstr += textwrap.dedent('''
    if not embedded:
        if '--quiet' not in sys.argv:
            print('[df2] Presenting in normal shell.')
            print('[df2] ... plt.show()')
        import matplotlib.pyplot as plt
        if '--noshow' not in sys.argv:
            plt.show()
    ''')
    return execstr
