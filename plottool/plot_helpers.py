from __future__ import absolute_import, division, print_function
import numpy as np
import plottool.draw_func2 as df2
import utool
import vtool.keypoint as ktool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[plot_helpers]', DEBUG=False)


SIFT_OR_VECFIELD = utool.get_arg('--vecfield', type_=bool)


def draw():
    df2.adjust_subplots_safe()
    df2.draw()


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
    _ibsdat = ax.__dict__.get('_ibsdat', None)
    if _ibsdat is None:
        return default
    val = _ibsdat.get(key, default)
    return val


def set_plotdat(ax, key, val):
    """ sets internal property to a matplotlib axis """
    if not '_ibsdat' in ax.__dict__:
        ax.__dict__['_ibsdat'] = {}
    _ibsdat = ax.__dict__['_ibsdat']
    _ibsdat[key] = val


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
