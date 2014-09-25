from __future__ import absolute_import, division, print_function
import numpy as np
import plottool.draw_func2 as df2
from plottool import custom_figure
import utool
import vtool.keypoint as ktool
from os.path import join
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[plot_helpers]', DEBUG=False)


SIFT_OR_VECFIELD = utool.get_argval('--vecfield', type_=bool)


def draw():
    df2.adjust_subplots_safe()
    df2.draw()


def dump_figure(dumpdir, subdir=None, quality=False, overwrite=False, verbose=2):
    """ Dumps figure to disk based on the figurename """
    if quality is True:
        custom_figure.FIGSIZE = df2.golden_wh2(12)
        custom_figure.DPI = 120
        custom_figure.FONTS.figtitle = df2.FONTS.small
    elif quality is False:
        custom_figure.FIGSIZE = df2.golden_wh2(8)
        custom_figure.DPI = 90
        custom_figure.FONTS.figtitle = df2.FONTS.smaller
    #print('[viz] Dumping Image')
    fpath = dumpdir
    if subdir is not None:
        fpath = join(fpath, subdir)
        utool.ensurepath(fpath)
    fpath_clean = df2.save_figure(fpath=fpath, usetitle=True, overwrite=overwrite, verbose=verbose)
    try:
        df2.reset()
    except Exception as ex:
        if utool.VERBOSE:
            utool.prinex(ex)
        pass
    return fpath_clean


def get_square_row_cols(nSubplots, max_cols=None):
    if nSubplots == 0:
        return 0, 0
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
    kpts = np.array([kp])
    xy_str    = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str
