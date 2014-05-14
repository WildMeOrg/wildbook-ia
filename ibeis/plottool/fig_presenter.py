from __future__ import absolute_import, division, print_function
import utool
import sys
import textwrap
import time
import warnings
# maptlotlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# PyQt
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
# Science
#import numpy as np
from .custom_figure import get_fig
#from .custom_constants import golden_wh


DEFAULT_MAX_ROWS = 3
AVAIL_PERCENT_W = .5
AVAIL_PERCENT_H = 1
QT4_WINS = []
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[df2]', DEBUG=False)


def unregister_qt4_win(win):
    global QT4_WINS
    if win == 'all':
        QT4_WINS = []


def register_qt4_win(win):
    global QT4_WINS
    QT4_WINS.append(win)


def OooScreen2():
    nRows = 1
    nCols = 1
    x_off = 30 * 4
    y_off = 30 * 4
    x_0 = -1920
    y_0 = 30
    w = (1912 - x_off) / nRows
    h = (1080 - y_off) / nCols
    return dict(num_rc=(1, 1), wh=(w, h), xy_off=(x_0, y_0), wh_off=(0, 10),
                row_first=True, no_tile=False)


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
    return QT4_WINS


SLEEP_TIME = .05


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


def ensure_app_is_running():
    import guitool
    app, is_root = guitool.init_qtapp()


def get_monitor_geom(monitor_num=0):
    from PyQt4 import QtGui  # NOQA
    ensure_app_is_running()
    desktop = QtGui.QDesktopWidget()
    rect = desktop.availableGeometry(screen=monitor_num)
    geom = (rect.x(), rect.y(), rect.width(), rect.height())
    return geom


def get_monitor_geometries():
    from PyQt4 import QtGui  # NOQA
    ensure_app_is_running()
    monitor_geometries = {}
    desktop = QtGui.QDesktopWidget()
    for screenx in xrange(desktop.numScreens()):
        rect = desktop.availableGeometry(screen=screenx)
        geom = (rect.x(), rect.y(), rect.width(), rect.height())
        monitor_geometries[screenx] = geom
    return monitor_geometries


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

# Win7 Areo
WIN7_SIZES = {
    'os_border_x':   20,
    'os_border_y':   35,
    'os_border_h':   30,
    'win_border_x':  17,
    'win_border_y':  10,
    'mpl_toolbar_y': 10,
}

# Ubuntu (Medeterrainian Dark)
GNOME3_SIZES = {
    'os_border_x':    0,
    'os_border_y':   35,  # for gnome3 title bar
    'os_border_h':    0,
    'win_border_x':   5,
    'win_border_y':  30,
    'mpl_toolbar_y':  0,
}


def get_stdpxls():
    if sys.platform.startswith('win32'):
        stdpxls = WIN7_SIZES
    if sys.platform.startswith('linux'):
        stdpxls = GNOME3_SIZES
    return stdpxls


def get_xywh_pads():
    stdpxls = get_stdpxls()
    w_pad =  stdpxls['win_border_x']
    y_pad =  stdpxls['win_border_y'] + stdpxls['mpl_toolbar_y']
    # Pads are applied to all windows
    x_pad =  stdpxls['os_border_x']
    y_pad =  stdpxls['os_border_y']
    return (x_pad, y_pad, w_pad, y_pad)


def get_avail_geom(monitor_num=None):
    stdpxls = get_stdpxls()
    if monitor_num is None:
        if utool.get_computer_name() == 'Ooo':
            monitor_num = 1
        else:
            monitor_num = 0
    monitor_geometries = get_monitor_geometries()
    (startx, starty, availw, availh) = monitor_geometries[monitor_num]
    available_geom = (startx, starty,
                      availw * AVAIL_PERCENT_W,
                      (availh - stdpxls['os_border_h']) * AVAIL_PERCENT_H)
    return available_geom


#@profile
def all_figures_tile(max_rows=None, row_first=True, no_tile=False, override1=False,
                     adaptive=False, monitor_num=None, **kwargs):
    """
    Lays out all figures in a grid. if wh is a scalar, a golden ratio is used
    """
    if max_rows is None:
        max_rows = DEFAULT_MAX_ROWS
    #print('[df2] all_figures_tile()')
    if no_tile:
        return
    #print('tile_figures_in_avail_geom')

    all_figures = get_all_figures()
    all_qt4wins = get_all_qt4_wins()
    all_wins = all_qt4wins + [fig.canvas.manager.window for fig in all_figures]

    num_wins = len(all_wins)
    if num_wins == 0:
        return

    startx, starty, avail_width, avail_height = get_avail_geom()

    nRows = num_wins if num_wins < max_rows else max_rows
    nCols = int(np.ceil(num_wins / nRows))

    win_height = avail_height / nRows
    win_width  = avail_width  / nCols

    (x_pad, y_pad, w_pad, h_pad) = get_xywh_pads()

    printDBG('startx = %r' % startx)
    printDBG('starty = %r' % starty)
    printDBG('avail_width = %r' % avail_width)
    printDBG('avail_height = %r' % avail_height)
    printDBG('win_width = %r' % win_width)
    printDBG('win_height = %r' % win_height)
    printDBG('nRows = %r' % nRows)
    printDBG('nCols = %r' % nCols)

    def position_window(ix, win):
        QMainWin = get_main_win_base()
        isqt4_mpl = isinstance(win, QMainWin)
        isqt4_back = isinstance(win, QtGui.QMainWindow)
        if not isqt4_mpl and not isqt4_back:
            raise NotImplementedError('%r-th Backend %r is not a Qt Window' %
                                      (ix, win))
        if row_first:
            rowx = ix % nRows
            colx = int(ix // nRows)
        else:
            colx = (ix % nCols)
            rowx = int(ix // nCols)
        w = win_width  - w_pad
        h = win_height - h_pad
        x = startx + colx * (win_width)  + x_pad
        y = starty + rowx * (win_height) + y_pad
        try:
            printDBG('tile %d-th win: rc=%r xywh=%r' % (ix, (rowx, colx), (x, y, w, h)))
            win.setGeometry(x, y, w, h)
        except Exception as ex:
            print(ex)
    for ix, win in enumerate(all_wins):
        position_window(ix, win)


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
