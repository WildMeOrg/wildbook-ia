# LICENCE
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
from vtool_ibeis import linalg as ltool
from vtool_ibeis import image as gtool
import utool as ut
try:
    import cv2
except ImportError as ex:
    print('ERROR: import cv2 is failing!')
    cv2 = ut.DynStruct()
    cv2.INTER_LANCZOS4 = None


def get_image_to_chip_transform(bbox, chipsz, theta):
    """
    transforms image space into chipspace

    Args:
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box

    Sympy:
        # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
        from vtool_ibeis.patch import *  # NOQA
        import sympy
        import sympy.abc
        theta = sympy.abc.theta

        x, y, w, h, target_area  = sympy.symbols('x y w h, a')
        gx, gy  = sympy.symbols('gx, gy')

        round = sympy.floor  # hack

        ht = sympy.sqrt(target_area * h / w)
        wt = w * ht / h
        cw_, ch_ = round(wt), round(ht)

        from vtool_ibeis import ltool
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


def _get_chip_to_image_transform(bbox, chipsz, theta):
    """ transforms chip space into imgspace
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box
    """
    C    = get_image_to_chip_transform(bbox, chipsz, theta)
    invC = npl.inv(C)
    return invC


def extract_chip_from_gpath(gfpath, bbox, theta, new_size, interpolation=cv2.INTER_LANCZOS4):
    imgBGR = gtool.imread(gfpath)  # Read parent image
    chipBGR = extract_chip_from_img(imgBGR, bbox, theta, new_size, interpolation)
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


def extract_chip_from_img(imgBGR, bbox, theta, new_size, interpolation=cv2.INTER_LANCZOS4):
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
        python -m vtool_ibeis.chip --test-extract_chip_from_img
        python -m vtool_ibeis.chip --test-extract_chip_from_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.chip import *  # NOQA
        >>> # build test data
        >>> imgBGR = gtool.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> bbox = (100, 3, 100, 100)
        >>> theta = 0.0
        >>> new_size = (58, 34)
        >>> # execute function
        >>> chipBGR = extract_chip_from_img(imgBGR, bbox, theta, new_size)
        >>> # verify results
        >>> assert chipBGR.shape[0:2] == new_size[::-1], 'did not resize correctly'
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> pt.imshow(chipBGR)
        >>> pt.show_if_requested()
    """
    # THE CULPRIT FOR MULTIPROCESSING FREEZES
    flags = interpolation
    #if True:
    M = get_image_to_chip_transform(bbox, new_size, theta)  # Build transformation
    chipBGR = cv2.warpAffine(imgBGR, M[0:2], tuple(new_size), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    #else:
    #    # if theta == 0, not sure if this is better. Certainly not more general
    #    x, y, w, h = bbox
    #    roiBGR = imgBGR[y:y + h, x:x + w, :]
    #    chipBGR = cv2.resize(roiBGR, tuple(new_size), interpolation=interpolation)
    #chipBGR = gtool.warpAffine(imgBGR, M, new_size)  # Rotate and scale
    return chipBGR


def gridsearch_chipextract():
    r"""
    CommandLine:
        xdoctest -m ~/code/vtool_ibeis/vtool_ibeis/chip.py gridsearch_chipextract --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # GRIDSEARCH
        >>> from vtool_ibeis.chip import *  # NOQA
        >>> gridsearch_chipextract()
        >>> ut.show_if_requested()
    """
    import cv2
    test_func = extract_chip_from_img
    if False:
        gpath = ut.grab_test_imgpath('carl.jpg')
        bbox = (100, 3, 100, 100)
        theta = 0.0
        new_size = (58, 34)
    else:
        gpath = '/media/raid/work/GZ_Master1/_ibsdb/images/1524525d-2131-8770-d27c-3a5f9922e9e9.jpg'
        bbox = (450, 373, 2062, 1124)
        theta = 0.0
        old_size = bbox[2:4]
        #target_area = 700 ** 2
        target_area = 1200 ** 2
        new_size = ScaleStrat.area(target_area, old_size)
        print('old_size = %r' % (old_size,))
        print('new_size = %r' % (new_size,))
        #new_size = (677, 369)
    imgBGR = gtool.imread(gpath)
    args = (imgBGR, bbox, theta, new_size)
    param_info = ut.ParamInfoList('extract_params', [
        ut.ParamInfo('interpolation', cv2.INTER_LANCZOS4,
                     varyvals=[
                         cv2.INTER_LANCZOS4,
                         cv2.INTER_CUBIC,
                         cv2.INTER_LINEAR,
                         cv2.INTER_NEAREST,
                         #cv2.INTER_AREA
                     ],)
    ])
    show_func = None
    # Generalize
    import plottool_ibeis as pt
    pt.imshow(imgBGR)  # HACK
    cfgdict_list, cfglbl_list = param_info.get_gridsearch_input(defaultslice=slice(0, 10))
    fnum = pt.ensure_fnum(None)
    if show_func is None:
        show_func = pt.imshow
    lbl = ut.get_funcname(test_func)
    cfgresult_list = [
        test_func(*args, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl=lbl)
    ]
    onclick_func = None
    ut.interact_gridsearch_result_images(
        show_func, cfgdict_list, cfglbl_list,
        cfgresult_list, fnum=fnum,
        figtitle=lbl, unpack=False,
        max_plots=25, onclick_func=onclick_func)
    pt.iup()


class ScaleStrat(object):
    """
    Scaling strategies
    """

    @staticmethod
    def maxwh(target, orig_wh, tol=0):
        r"""
        The maximum dimension becomes target

        Args:
            target (int): target size

        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> ut.assert_eq(ScaleStrat.maxwh(800, (190, 220)), (691, 800))
            >>> ut.assert_eq(ScaleStrat.maxwh(800, (220, 190)), (800, 691))
        """
        max_idx = np.argmax(orig_wh)
        orig_dim_size = orig_wh[max_idx]
        low, high = (target - tol, target + tol)
        if low <= orig_dim_size and orig_dim_size <= high:
            new_size = orig_wh
        else:
            scale_factor = target / orig_dim_size
            wt = int(round(orig_wh[0] * scale_factor))
            ht = int(round(orig_wh[1] * scale_factor))
            new_size = (wt, ht)
        return new_size

    @staticmethod
    def width(target, orig_wh, tol=0):
        r"""
        The width becomes target

        Args:
            target (int): target size

        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> ut.assert_eq(ScaleStrat.width(800, (190, 220)), (800, 926))
            >>> ut.assert_eq(ScaleStrat.width(800, (220, 190)), (800, 691))
        """
        orig_dim_size = orig_wh[0]
        low, high = (target - tol, target + tol)
        if low <= orig_dim_size and orig_dim_size <= high:
            new_size = orig_wh
        else:
            scale_factor = target / orig_dim_size
            wt = int(round(orig_wh[0] * scale_factor))
            ht = int(round(orig_wh[1] * scale_factor))
            new_size = (wt, ht)
        return new_size

    @staticmethod
    def area(target, orig_wh, tol=0):
        r"""
        The area becomes target

        Args:
            target (int): target size

        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> ut.assert_eq(ScaleStrat.area(800 ** 2, (190, 220)), (743, 861))
            >>> ut.assert_eq(ScaleStrat.area(800 ** 2, (220, 190)), (861, 743))
        """
        w, h = orig_wh
        area = w * h
        low, high = (target - tol, target + tol)
        if low <= area and area <= high:
            new_size = orig_wh
        else:
            ht = np.sqrt(target * h / w)
            wt = w * ht / h
            new_size = (int(round(wt)), int(round(ht)))
        return new_size


def get_scaled_size_with_dlen(target_dlen, w, h):
    r"""
    returns new_size which scales (w, h) as close to target_dlen as possible
    and maintains aspect ratio
    """
    #ht = np.sqrt(target_area * h / w)
    #wt = w * ht / h
    #new_size = (int(round(wt)), int(round(ht)))
    raise NotImplementedError()
    #return new_size


def compute_chip(gfpath, bbox, theta, new_size, filter_list=[],
                 interpolation=cv2.INTER_LANCZOS4):
    r""" Extracts a chip and applies filters

    DEPRICATE

    Args:
        gfpath (str):  image file path string
        bbox (tuple):  bounding box in the format (x, y, w, h)
        theta (float):  angle in radians
        new_size (tuple): must maintain the same aspect ratio or else you will get weirdness
        filter_list (list):

    Returns:
        ndarray: chipBGR -  cropped image

    CommandLine:
        python -m vtool_ibeis.chip --test-compute_chip --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.chip import *  # NOQA
        >>> from vtool_ibeis.util_math import TAU
        >>> # build test data
        >>> gfpath = ut.grab_test_imgpath('carl.jpg')
        >>> bbox = (100, 3, 100, 100)
        >>> theta = TAU / 8
        >>> new_size = (32, 32)
        >>> filter_list = []
        >>> # execute function
        >>> chipBGR = compute_chip(gfpath, bbox, theta, new_size, filter_list)
        >>> # verify results
        >>> assert chipBGR.shape[0:2] == new_size[::-1], 'did not resize correctly'
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool_ibeis as pt
        >>> import vtool_ibeis as vt
        >>> pt.imshow(vt.draw_verts(vt.imread(gfpath), vt.scaled_verts_from_bbox(bbox, theta, 1, 1)), pnum=(1, 2, 1))
        >>> pt.imshow(chipBGR, pnum=(1, 2, 2))
        >>> pt.show_if_requested()
    """
    chipBGR = extract_chip_from_gpath(gfpath, bbox, theta, new_size, interpolation)
    chipBGR = apply_filter_funcs(chipBGR, filter_list)
    return chipBGR


def apply_filter_funcs(chipBGR, filter_funcs):
    """ applies a list of preprocessing filters to a chip

    DEPRICATE

    """
    chipBGR_ = chipBGR
    for func in filter_funcs:
        chipBGR_ = func(chipBGR)
    return chipBGR_


def get_extramargin_measures(bbox_gs, new_size, halfoffset_ms=(64, 64)):
    r"""
    Computes a detection chip with a bit of spatial context so the detection
    algorithm doesn't clip boundaries

    Returns:
        mbbox_gs, margin_size -
            margin bounding box in image size,
            size of entire margined chip,

    CommandLine:
        python -m vtool_ibeis.chip --test-get_extramargin_measures --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool_ibeis.chip import *  # NOQA
        >>> gfpath = ut.grab_test_imgpath('carl.jpg')
        >>> bbox_gs = [40, 40, 150, 150]
        >>> theta = .15 * (np.pi * 2)
        >>> new_size = (150, 150)
        >>> halfoffset_ms = (32, 32)
        >>> mbbox_gs, margin_size = get_extramargin_measures(bbox_gs, new_size, halfoffset_ms)
        >>> # xdoctest: +REQUIRES(--show)
        >>> testshow_extramargin_info(gfpath, bbox_gs, theta, new_size, halfoffset_ms, mbbox_gs, margin_size)
    """
    # _ex denotes an expanded version
    # There are three spaces we are working in here
    # chip _cs (the space of the original chip)
    # margin _ms (the margin chip has the scale of chip space with padding)
    # imagespace _gs (the space using in bbox_gs specification)
    x_gs, y_gs, w_gs, h_gs = bbox_gs
    if w_gs == 0 or  h_gs == 0:
        raise ValueError('Bounding box has no area')
    w_cs, h_cs = new_size
    # Extra margin in chip space
    xo_ms, yo_ms = halfoffset_ms
    # Compute size of margin chip
    mw, mh = (w_cs + (2 * xo_ms), h_cs + (2 * yo_ms))
    margin_size = (mw, mh)
    # Get the conversion from chip to image space
    sx, sy = (w_gs / w_cs, h_gs / h_cs)
    # Convert the chip offsets to image space
    halfoffset_gs = ((sx * xo_ms), (sy * yo_ms))
    xo_gs, yo_gs = halfoffset_gs
    # Find the size of the expanded margin bbox in image space
    mbbox_gs = (x_gs - xo_gs, y_gs - yo_gs,
                w_gs + (2 * xo_gs), h_gs + (2 * yo_gs))
    return mbbox_gs, margin_size


def testshow_extramargin_info(gfpath, bbox_gs, theta, new_size, halfoffset_ms, mbbox_gs, margin_size):
    import plottool_ibeis as pt
    import vtool_ibeis as vt

    imgBGR = vt.imread(gfpath)
    chipBGR = compute_chip(gfpath, bbox_gs, theta, new_size, [])
    mchipBGR = compute_chip(gfpath, mbbox_gs, theta, margin_size, [])

    #index = 0
    w_cs, h_cs = new_size
    xo_ms, yo_ms = halfoffset_ms
    bbox_ms = [xo_ms, yo_ms, w_cs, h_cs]

    verts_gs = vt.scaled_verts_from_bbox(bbox_gs, theta, 1, 1)
    expanded_verts_gs = vt.scaled_verts_from_bbox(mbbox_gs, theta, 1, 1)
    expanded_verts_ms = vt.scaled_verts_from_bbox(bbox_ms, 0, 1, 1)
    # topheavy
    imgBGR = vt.draw_verts(imgBGR, verts_gs)
    imgBGR = vt.draw_verts(imgBGR, expanded_verts_gs)

    mchipBGR = vt.draw_verts(mchipBGR, expanded_verts_ms)

    fnum = 1
    pt.imshow(imgBGR, pnum=(1, 3, 1), fnum=fnum, title='original image')
    pt.gca().set_xlabel(str(imgBGR.shape))
    pt.imshow(chipBGR, pnum=(1, 3, 2), fnum=fnum, title='original chip')
    pt.gca().set_xlabel(str(chipBGR.shape))
    pt.imshow(mchipBGR, pnum=(1, 3, 3), fnum=fnum,
              title='scaled chip with expanded margin.\n(orig margin drawn in orange)')
    pt.gca().set_xlabel(str(mchipBGR.shape))

    pt.show_if_requested()
    #pt.imshow(chipBGR)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool_ibeis.chip
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
