from __future__ import absolute_import, division, print_function
import numpy as np
import plottool.draw_func2 as df2
from plottool import custom_figure
from plottool import fig_presenter
from plottool import custom_constants
from os.path import join
import utool as ut
ut.noinject(__name__, '[plot_helpers]')
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[plot_helpers]', DEBUG=False)


SIFT_OR_VECFIELD = ut.get_argval('--vecfield', type_=bool)


def draw():
    df2.adjust_subplots_safe()
    fig_presenter.draw()


def dump_figure(dumpdir, subdir=None, quality=False, overwrite=False, verbose=2,
                   reset=True):
    """ Dumps figure to disk based on the figurename """
    if quality is True:
        custom_constants.FIGSIZE = custom_constants.golden_wh2(14)
        custom_constants.DPI = 120
        #custom_constants.FIGSIZE = custom_constants.golden_wh2(12)
        #custom_constants.DPI = 120
        custom_constants.FONTS.figtitle = custom_constants.FONTS.small
    elif quality is False:
        #custom_constants.FIGSIZE = custom_constants.golden_wh2(8)
        #custom_constants.FIGSIZE = custom_constants.golden_wh2(14)
        #custom_constants.DPI = 100
        custom_constants.FIGSIZE = custom_constants.golden_wh2(8)
        custom_constants.DPI = 90
        custom_constants.FONTS.figtitle = custom_constants.FONTS.smaller
    fpath = dumpdir
    if subdir is not None:
        fpath = join(fpath, subdir)
        ut.ensurepath(fpath)
    fpath_clean = custom_figure.save_figure(fpath=fpath, usetitle=True, overwrite=overwrite, verbose=verbose)
    if reset:
        try:
            fig_presenter.reset()
        except Exception as ex:
            if ut.VERBOSE:
                ut.prinex(ex)
            pass
    return fpath_clean


def get_square_row_cols(nSubplots, max_cols=None, fix=False):
    r"""
    Args:
        nSubplots (?):
        max_cols (None):

    Returns:
        tuple: (None, None)

    CommandLine:
        python -m plottool.plot_helpers --test-get_square_row_cols

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.plot_helpers import *  # NOQA
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
    if fix:
        # This function is very broken, but it might have dependencies
        # this is the correct version
        nCols = int(np.ceil(np.sqrt(nSubplots)))
        nRows = int(np.ceil(nSubplots / nCols))
        return nRows, nCols
    else:
        # This is the clamped num cols version
        # probably used in ibeis.viz
        if max_cols is None:
            max_cols = 5
            if nSubplots in [4]:
                max_cols = 2
            if nSubplots in [5, 6, 7]:
                max_cols = 3
            if nSubplots in [8]:
                max_cols = 4
        nCols = int(min(nSubplots, max_cols))
        #nCols = int(min(np.ceil(np.sqrt(nrids)), 5))
        nRows = int(np.ceil(nSubplots / nCols))
    return nRows, nCols


def get_plotdat(ax, key, default=None):
    """ returns internal property from a matplotlib axis """
    _plotdat = ax.__dict__.get('_plotdat', None)
    if _plotdat is None:
        return default
    val = _plotdat.get(key, default)
    return val


def set_plotdat(ax, key, val):
    """ sets internal property to a matplotlib axis """
    if '_plotdat' not in ax.__dict__:
        ax.__dict__['_plotdat'] = {}
    _plotdat = ax.__dict__['_plotdat']
    _plotdat[key] = val


def get_bbox_centers(bbox_list):
    bbox_centers = np.array([np.array([x + (w / 2), y + (h / 2)])
                             for (x, y, w, h) in bbox_list])
    return bbox_centers


#==========================#
#  --- TESTING FUNCS ---   #
#==========================#

def kp_info(kp):
    import vtool.keypoint as ktool
    kpts = np.array([kp])
    xy_str    = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str
