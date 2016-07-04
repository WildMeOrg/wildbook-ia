from __future__ import absolute_import, division, print_function
import six
from six.moves import range
import sys
import utool as ut
import numpy as np
try:
    import guitool
    from guitool.__PYQT__ import QtWidgets
except ImportError:
    try:
        from PyQt4 import QtGui as QtWidgets
    except ImportError:
        pass
    try:
        from PyQt5 import QtWidgets  # NOQA
    except ImportError:
        pass
    print('Warning: guitool did not import correctly')
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[screeninfo]', DEBUG=True)
ut.noinject(__name__, '[screeninfo]')


DEFAULT_MAX_ROWS = 3


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

for key in GNOME3_SIZES:
    GNOME3_SIZES[key] += 5


def ensure_app_is_running():
    #print('[screeninfo] Ensuring that qtapp is running')
    app, is_root = guitool.init_qtapp()


def get_resolution_info(monitor_num=0):
    r"""
    Args:
        monitor_num (int): (default = 0)

    Returns:
        dict: info

    CommandLine:
        python -m plottool.screeninfo get_resolution_info --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.screeninfo import *  # NOQA
        >>> monitor_num = 0
        >>> info = get_resolution_info(monitor_num)
        >>> print('info = %s' % (ut.repr2(info, nl=True),))
        >>> info = get_resolution_info(1)
        >>> print('info = %s' % (ut.repr2(info, nl=True),))
    """
    ensure_app_is_running()
    desktop = QtWidgets.QDesktopWidget()
    screen = desktop.screen(monitor_num)
    ppi_x = screen.logicalDpiX()
    ppi_y = screen.logicalDpiY()
    dpi_x = screen.physicalDpiX()
    dpi_y = screen.physicalDpiY()
    rect = desktop.availableGeometry(screen=monitor_num)
    pixels_w = rect.width()
    pixels_h = rect.height()
    inches_w = (pixels_w / dpi_x)
    inches_h = (pixels_h / dpi_y)
    #pixel_density = dpi_x / ppi_x
    info = {
        'monitor_num': monitor_num,
        'ppi_x': ppi_x,
        'ppi_y': ppi_y,
        'dpi_x': dpi_x,
        'dpi_y': dpi_y,
        #'pixel_density': pixel_density,
        'inches_w': inches_w,
        'inches_h': inches_h,
        'pixels_w': pixels_w,
        'pixels_h': pixels_h,
    }
    return info


def get_monitor_geom(monitor_num=0):
    r"""
    Args:
        monitor_num (int): (default = 0)

    Returns:
        tuple: geom

    CommandLine:
        python -m plottool.screeninfo get_monitor_geom --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.screeninfo import *  # NOQA
        >>> monitor_num = 0
        >>> geom = get_monitor_geom(monitor_num)
        >>> result = ('geom = %s' % (ut.repr2(geom),))
        >>> print(result)
    """
    ensure_app_is_running()
    desktop = QtWidgets.QDesktopWidget()
    rect = desktop.availableGeometry(screen=monitor_num)
    geom = (rect.x(), rect.y(), rect.width(), rect.height())
    return geom


def get_monitor_geometries():
    ensure_app_is_running()
    monitor_geometries = {}
    desktop = QtWidgets.QDesktopWidget()
    for screenx in range(desktop.numScreens()):
        rect = desktop.availableGeometry(screen=screenx)
        geom = (rect.x(), rect.y(), rect.width(), rect.height())
        monitor_geometries[screenx] = geom
    return monitor_geometries


def get_stdpxls():
    if sys.platform.startswith('win32'):
        stdpxls = WIN7_SIZES
    elif sys.platform.startswith('linux'):
        stdpxls = GNOME3_SIZES
    else:
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


def get_avail_geom(monitor_num=None, percent_w=1.0, percent_h=1.0):
    stdpxls = get_stdpxls()
    if monitor_num is None:
        if ut.get_computer_name() == 'Ooo':
            monitor_num = 1
        else:
            monitor_num = 0
    monitor_geometries = get_monitor_geometries()
    try:
        (startx, starty, availw, availh) = monitor_geometries[monitor_num]
    except KeyError:
        (startx, starty, availw, availh) = six.itervalues(monitor_geometries).next()
    available_geom = (startx,
                      starty,
                      availw * percent_w,
                      (availh - stdpxls['os_border_h']) * percent_h)
    return available_geom


def get_valid_fig_positions(num_wins, max_rows=None, row_first=True,
                            monitor_num=None, adaptive=False):
    """ Computes which figure positions are valid given args """
    if max_rows is None:
        max_rows = DEFAULT_MAX_ROWS

    percent_w = 1.0
    percent_h = 1.0

    if adaptive:
        if num_wins <= DEFAULT_MAX_ROWS:
            percent_w = .5
        else:
            percent_w = 1.0

    available_geom = get_avail_geom(monitor_num, percent_w=percent_w, percent_h=percent_h)
    startx, starty, avail_width, avail_height = available_geom

    nRows = num_wins if num_wins < max_rows else max_rows
    nCols = int(np.ceil(num_wins / nRows))

    win_height = avail_height / nRows
    win_width  = avail_width  / nCols

    (x_pad, y_pad, w_pad, h_pad) = get_xywh_pads()

    #printDBG('startx = %r' % startx)
    #printDBG('starty = %r' % starty)
    #printDBG('avail_width = %r' % avail_width)
    #printDBG('avail_height = %r' % avail_height)
    #printDBG('win_width = %r' % win_width)
    #printDBG('win_height = %r' % win_height)
    #printDBG('nRows = %r' % nRows)
    #printDBG('nCols = %r' % nCols)

    def get_position_ix(ix):
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
        return (x, y, w, h)
    valid_positions = [get_position_ix(ix) for ix in range(num_wins)]
    return valid_positions


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m plottool.screeninfo
        python -m plottool.screeninfo --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
