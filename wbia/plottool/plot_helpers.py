# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np

# from . import draw_func2 as df2
from wbia.plottool import fig_presenter

# from wbia.plottool import custom_figure
# from wbia.plottool import custom_constants
# from os.path import join
import utool as ut

ut.noinject(__name__, '[plot_helpers]')
# (print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[plot_helpers]', DEBUG=False)


SIFT_OR_VECFIELD = ut.get_argval('--vecfield', type_=bool)


def draw():
    fig_presenter.draw()


# def dump_figure(dumpdir, subdir=None, quality=False, overwrite=False, verbose=2,
#                   reset=True):
#    """ Dumps figure to disk based on the figurename """
#    if quality is True:
#        custom_constants.FIGSIZE = custom_constants.golden_wh2(14)
#        #custom_constants.DPI = 120
#        custom_constants.DPI = 120
#        #custom_constants.FIGSIZE = custom_constants.golden_wh2(12)
#        #custom_constants.DPI = 120
#        custom_constants.FONTS.figtitle = custom_constants.FONTS.small
#    elif quality is False:
#        #custom_constants.FIGSIZE = custom_constants.golden_wh2(8)
#        #custom_constants.FIGSIZE = custom_constants.golden_wh2(14)
#        #custom_constants.DPI = 100
#        custom_constants.FIGSIZE = custom_constants.golden_wh2(8)
#        custom_constants.DPI = 90
#        custom_constants.FONTS.figtitle = custom_constants.FONTS.smaller
#    fpath = dumpdir
#    if subdir is not None:
#        fpath = join(fpath, subdir)
#        ut.ensurepath(fpath)
#    fpath_clean = custom_figure.save_figure(fpath=fpath, usetitle=True, overwrite=overwrite, verbose=verbose)
#    return fpath_clean


def get_square_row_cols(nSubplots, max_cols=None, fix=False, inclusive=True):
    r"""
    Args:
        nSubplots (?):
        max_cols (None):

    Returns:
        tuple: (None, None)

    CommandLine:
        python -m wbia.plottool.plot_helpers --test-get_square_row_cols

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.plot_helpers import *  # NOQA
        >>> # build test data
        >>> nSubplots = 9
        >>> nSubplots_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        >>> max_cols = None
        >>> # execute function
        >>> rc_list = [get_square_row_cols(nSubplots, fix=True) for nSubplots in nSubplots_list]
        >>> # verify results
        >>> result = repr(np.array(rc_list).T)
        >>> print(result)
        array([[1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
               [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4]])
    """
    if nSubplots == 0:
        return 0, 0
    if inclusive:
        rounder = np.ceil
    else:
        rounder = np.floor
    if fix:
        # This function is very broken, but it might have dependencies
        # this is the correct version
        nCols = int(rounder(np.sqrt(nSubplots)))
        nRows = int(rounder(nSubplots / nCols))
        return nRows, nCols
    else:
        # This is the clamped num cols version
        # probably used in wbia.viz
        if max_cols is None:
            max_cols = 5
            if nSubplots in [4]:
                max_cols = 2
            if nSubplots in [5, 6, 7]:
                max_cols = 3
            if nSubplots in [8]:
                max_cols = 4
        nCols = int(min(nSubplots, max_cols))
        # nCols = int(min(rounder(np.sqrt(nrids)), 5))
        nRows = int(rounder(nSubplots / nCols))
    return nRows, nCols


def get_plotdat(ax, key, default=None):
    """ returns internal property from a matplotlib axis """
    _plotdat = get_plotdat_dict(ax)
    val = _plotdat.get(key, default)
    return val


def set_plotdat(ax, key, val):
    """ sets internal property to a matplotlib axis """
    _plotdat = get_plotdat_dict(ax)
    _plotdat[key] = val


def del_plotdat(ax, key):
    """ sets internal property to a matplotlib axis """
    _plotdat = get_plotdat_dict(ax)
    if key in _plotdat:
        del _plotdat[key]


def get_plotdat_dict(ax):
    """ sets internal property to a matplotlib axis """
    if '_plotdat' not in ax.__dict__:
        ax.__dict__['_plotdat'] = {}
    plotdat_dict = ax.__dict__['_plotdat']
    return plotdat_dict


def get_bbox_centers(bbox_list):
    bbox_centers = np.array(
        [np.array([x + (w / 2), y + (h / 2)]) for (x, y, w, h) in bbox_list]
    )
    return bbox_centers


def qt4ensure():
    qtensure()
    # if ut.inIPython():
    #     import IPython
    #     #IPython.get_ipython().magic('pylab qt4')
    #     IPython.get_ipython().magic('pylab qt4 --no-import-all')


def qtensure():
    import wbia.guitool as gt

    if ut.inIPython():
        import IPython

        ipython = IPython.get_ipython()
        if ipython is None:
            # we must have exited ipython at some point
            return
        if gt.__PYQT__.GUITOOL_PYQT_VERSION == 5:
            """
            sudo apt-get install python3-pyqt5.qtsvg
            """
            # import os
            # os.environ['QT_API'] = 'pyqt5'
            # import matplotlib
            # matplotlib.use('Qt5Agg')
            # IPython.get_ipython().magic('matplotlib qt5')
            # IPython.get_ipython().magic('pylab qt4')
            ipython.magic('pylab qt5 --no-import-all')
        else:
            # IPython.get_ipython().magic('pylab qt4')
            ipython.magic('pylab qt4 --no-import-all')


ensureqt = qt4ensure

# ==========================#
#  --- TESTING FUNCS ---   #
# ==========================#


def kp_info(kp):
    import vtool.keypoint as ktool

    kpts = np.array([kp])
    xy_str = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str
