# LICENCE
from __future__ import absolute_import, division, print_function
# Science
import numpy as np
import numpy.linalg as npl
# VTool
from vtool import linalg as ltool
from vtool import image as gtool
from vtool import image_filters as gfilt_tool
import utool as ut
import cv2
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[chip]', DEBUG=False)


@profile
def get_image_to_chip_transform(bbox, chipsz, theta):
    """
    transforms image space into chipspace

    Args:
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box

    Sympy:
        # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
        from vtool.patch import *  # NOQA
        import sympy
        import sympy.abc
        theta = sympy.abc.theta

        x, y, w, h, target_area  = sympy.symbols('x y w h, a')
        gx, gy  = sympy.symbols('gx, gy')

        round = sympy.floor  # hack

        ht = sympy.sqrt(target_area * h / w)
        wt = w * ht / h
        cw_, ch_ = round(wt), round(ht)

        from vtool import ltool
        T1 = ltool.translation_mat3x3(tx1, ty1, dtype=None)
        S  = ltool.scale_mat3x3(sx, sy, dtype=None)
        R  = ltool.rotation_mat3x3(-theta, sympy.sin, sympy.cos)
        T2 = ltool.translation_mat3x3(tx2, ty2, dtype=None)

        def add_matmul_hold_prop(mat):
            #import functools
            mat = sympy.Matrix(mat)
            def matmul_hold(other, hold=False):
                new = sympy.MatMul(mat, other, hold=hold)
                add_matmul_hold_prop(new)
                return new
            setattr(mat, 'matmul_hold', matmul_hold)
            return mat

        T1 = add_matmul_hold_prop(T1)
        T2 = add_matmul_hold_prop(T2)
        R = add_matmul_hold_prop(R)
        S = add_matmul_hold_prop(S)

        C = T2.multiply(R.multiply(S.multiply(T1)))
        sympy.simplify(C)

    """
    (x, y, w, h) = bbox
    (cw_, ch_) = chipsz
    tx1 = -(x + (w / 2.0))
    ty1 = -(y + (h / 2.0))
    sx = (cw_ / w)
    sy = (ch_ / h)
    tx2 = (cw_ / 2.0)
    ty2 = (ch_ / 2.0)
    # Translate from bbox center to (0, 0)
    T1 = ltool.translation_mat3x3(tx1, ty1)
    # Scale to chip height
    S  = ltool.scale_mat3x3(sx, sy)
    # Rotate to chip orientation
    R  = ltool.rotation_mat3x3(-theta)
    # Translate from (0, 0) to chip center
    T2 = ltool.translation_mat3x3(tx2, ty2)
    # Merge into single transformation (operate left-to-right aka data on left)
    C = T2.dot(R.dot(S.dot(T1)))
    return C


@profile
def _get_chip_to_image_transform(bbox, chipsz, theta):
    """ transforms chip space into imgspace
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box
    """
    C    = get_image_to_chip_transform(bbox, chipsz, theta)
    invC = npl.inv(C)
    return invC


@profile
def extract_chip_from_gpath(gfpath, bbox, theta, new_size):
    imgBGR = gtool.imread(gfpath)  # Read parent image
    chipBGR = extract_chip_from_img(imgBGR, bbox, theta, new_size)
    return chipBGR


def extract_chip_into_square(imgBGR, bbox, theta, target_size):
    bbox_size = bbox[2:4]
    unpadded_dsize, ratio = gtool.resized_dims_and_ratio(bbox_size, target_size)
    chipBGR = extract_chip_from_img(imgBGR, bbox, theta, unpadded_dsize)
    chipBGR_square = gtool.embed_in_square_image(chipBGR, target_size)
    return chipBGR_square


def extract_chip_from_gpath_into_square(args):
    gfpath, bbox, theta, target_size = args
    imgBGR = gtool.imread(gfpath)  # Read parent image
    return extract_chip_into_square(imgBGR, bbox, theta, target_size)


@profile
def extract_chip_from_img(imgBGR, bbox, theta, new_size):
    """ Crops chip from image ; Rotates and scales;

    ibs.show_annot_image(aid)[0].pt_save_and_view()

    Args:
        gfpath (str):
        bbox (tuple):  xywh
        theta (float):
        new_size (tuple): wy

    Returns:
        ndarray: chipBGR

    CommandLine:
        python -m vtool.chip --test-extract_chip_from_img
        python -m vtool.chip --test-extract_chip_from_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.chip import *  # NOQA
        >>> # build test data
        >>> imgBGR = gtool.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> bbox = (100, 3, 100, 100)
        >>> theta = 0.0
        >>> new_size = (58, 34)
        >>> # execute function
        >>> chipBGR = extract_chip_from_img(imgBGR, bbox, theta, new_size)
        >>> # verify results
        >>> assert chipBGR.shape[0:2] == new_size[::-1], 'did not resize correctly'
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(chipBGR)
        >>> pt.show_if_requested()
    """
    M = get_image_to_chip_transform(bbox, new_size, theta)  # Build transformation
    # THE CULPRIT FOR MULTIPROCESSING FREEZES
    chipBGR = cv2.warpAffine(imgBGR, M[0:2], tuple(new_size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    #chipBGR = gtool.warpAffine(imgBGR, M, new_size)  # Rotate and scale
    return chipBGR


def get_scaled_size_with_width(target_width, w, h):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.chip import *  # NOQA
        >>> # build test data
        >>> target_width = 128
        >>> w = 600
        >>> h = 400
        >>> # execute function
        >>> new_size = get_scaled_size_with_width(target_width, w, h)
        >>> # verify results
        >>> result = str(new_size)
        >>> print(result)
        (128, 85)
    """
    wt = target_width
    sf = wt / w
    ht = sf * h
    new_size = (int(round(wt)), int(round(ht)))
    return new_size


def get_scaled_size_with_area(target_area, w, h):
    """
    returns new_size which scales (w, h) as close to target_area as possible and
    maintains aspect ratio

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.chip import *  # NOQA
        >>> # build test data
        >>> target_area = 800 ** 2
        >>> w = 600
        >>> h = 400
        >>> # execute function
        >>> new_size = get_scaled_size_with_area(target_area, w, h)
        >>> # verify results
        >>> result = str(new_size)
        >>> print(result)
        (980, 653)
    """
    ht = np.sqrt(target_area * h / w)
    wt = w * ht / h
    new_size = (int(round(wt)), int(round(ht)))
    return new_size


def get_scaled_sizes_with_area(target_area, size_list):
    return [get_scaled_size_with_area(target_area, w, h) for (w, h) in size_list]


#@profile
def compute_chip(gfpath, bbox, theta, new_size, filter_list=[]):
    """ Extracts a chip and applies filters

    Args:
        gfpath (str):  image file path string
        bbox (tuple):  bounding box in the format (x, y, w, h)
        theta (float):  angle in radians
        new_size (tuple): must maintain the same aspect ratio or else you will get weirdness
        filter_list (list):

    Returns:
        ndarray: chipBGR -  cropped image

    CommandLine:
        python -m vtool.chip --test-compute_chip --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.chip import *  # NOQA
        >>> # build test data
        >>> gfpath = ut.grab_test_imgpath('carl.jpg')
        >>> bbox = (100, 3, 100, 100)
        >>> TAU = 2 * np.pi
        >>> theta = TAU / 8
        >>> new_size = (32, 32)
        >>> filter_list = []  # gfilt_tool.adapteq_fn]
        >>> # execute function
        >>> chipBGR = compute_chip(gfpath, bbox, theta, new_size, filter_list)
        >>> # verify results
        >>> assert chipBGR.shape[0:2] == new_size[::-1], 'did not resize correctly'
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> import vtool as vt
        >>> pt.imshow(vt.draw_verts(vt.imread(gfpath), vt.scaled_verts_from_bbox(bbox, theta, 1, 1)), pnum=(1, 2, 1))
        >>> pt.imshow(chipBGR, pnum=(1, 2, 2))
        >>> pt.show_if_requested()
    """
    chipBGR = extract_chip_from_gpath(gfpath, bbox, theta, new_size)
    chipBGR = apply_filter_funcs(chipBGR, filter_list)
    return chipBGR


def apply_filter_funcs(chipBGR, filter_funcs):
    """ applies a list of preprocessing filters to a chip """
    chipBGR_ = chipBGR
    for func in filter_funcs:
        chipBGR_ = func(chipBGR)
    return chipBGR_


def get_filter_list(chipcfg_dict):
    filter_list = []
    if chipcfg_dict.get('adapteq'):
        filter_list.append(gfilt_tool.adapteq_fn)
    if chipcfg_dict.get('histeq'):
        filter_list.append(gfilt_tool.histeq_fn)
    #if chipcfg_dict.get('maxcontrast'):
        #filter_list.append(maxcontr_fn)
    #if chipcfg_dict.get('rank_eq'):
        #filter_list.append(rankeq_fn)
    #if chipcfg_dict.get('local_eq'):
        #filter_list.append(localeq_fn)
    if chipcfg_dict.get('grabcut'):
        filter_list.append(gfilt_tool.grabcut_fn)
    return filter_list

if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.chip
        python -m vtool.chip --allexamples
        python -m vtool.chip --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
