# -*- coding: utf-8 -*-
r"""
Keypoints are stored in the invA format by default.
Unfortunately many places in the code reference this as A instead of invA
because I was confused when I first started writing this.

to rectify this I am changing terminology.

Variables:
    invV : maps from ucircle onto an ellipse (perdoch.invA)
       V : maps from ellipse to ucircle      (perdoch.A)
       Z : the conic matrix                  (perdoch.E)

Representation:
    kpts (ndarray) : [x, y, iv11, iv21, iv22, ori]
       a flat on disk representation of the keypoint


    invV (ndarray): [(iv11, iv12, x),
                     (iv21, iv22, y),
                     (   0,    0, 1),]
        a more conceptually useful representation mapp;ing a
        unit circle onto an ellipse (without any rotation)

    invVR (ndarray): [(iv11, iv12, x),
                      (iv21, iv22, y),
                      (   0,    0, 1),].dot(R)
         same as invV but it is rotated before warping a unit circle
         into an ellipse.

Sympy:
    >>> # DISABLE_DOCTEST
    >>> # xdoctest: +SKIP
    >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
    >>> from vtool.patch import *  # NOQA
    >>> import sympy
    >>> from sympy.abc import theta
    >>> ori = theta
    >>> x, y, iv11, iv21, iv22, patch_size = sympy.symbols('x y iv11 iv21 iv22 S')
    >>> sx, sy, w1, w2, tx, ty = sympy.symbols('sx, sy, w1, w2, tx, ty')
    >>> kpts = np.array([[x, y, iv11, iv21, iv22, ori]])
    >>> kp = ktool.get_invV_mats(kpts, with_trans=True)[0]
    >>> invV = sympy.Matrix(kp)
    >>> V = invV.inv()
    >>> #
    >>> print(ub.hzcat('invV = ', repr(invV)))
    >>> invV = sympy.Matrix([
    >>>        [iv11,  0.0,   x],
    >>>        [iv21, iv22,   y],
    >>>        [ 0.0,  0.0, 1.0]])
    >>> R = vt.sympy_mat(vt.rotation_mat3x3(theta, sin=sympy.sin, cos=sympy.cos))
    >>> invVR = invV.multiply(R)
    >>> trans = sympy.Matrix([
    >>>        [  1,  0.0,   x],
    >>>        [  0,    1,   y],
    >>>        [ 0.0,  0.0, 1.0]])
    >>> #
    >>> Hypoth = sympy.Matrix([
    >>>        [    sx,    w1,   tx],
    >>>        [    w2,    sy,   ty],
    >>>        [     0,     0,    1],
    >>>        ])
    >>> #
    >>> xyz = sympy.Matrix([[x], [y], [1]])
    >>> #
    >>> invV_2x2 = invV[0:2, 0:2]
    >>> Hypoth_2x2 = Hypoth[0:2, 0:2]
    >>> #
    >>> invV_t = sympy.simplify(Hypoth.multiply(invV))
    >>> xyz_t = sympy.simplify(Hypoth.multiply(xyz))
    >>> invV_2x2_t = Hypoth_2x2.multiply(invV_2x2)
    >>> print('\n----')
    >>> vt.evalprint('invV_t')
    >>> vt.evalprint('xyz_t')
    >>> vt.evalprint('invV_2x2_t')
    >>> print('-----')
    >>> #
    >>> print('\n--- CHECKING 3x3 ---')
    >>> vt.check_expr_eq(invV_t[:, 2], xyz_t)
    >>> print('\n--- CHECKING 2x2 ---')
    >>> vt.check_expr_eq(invV_t[0:2, 0:2], invV_2x2_t)
    >>> #
    >>> # CHeck with rotation component as well (probably ok)
    >>> invVR_2x2 = invVR[0:2, 0:2]
    >>> invVR_t = sympy.simplify(Hypoth.multiply(invVR))
    >>> invVR_2x2_t = sympy.simplify(Hypoth_2x2.multiply(invVR_2x2))
    >>> print('\n----')
    >>> vt.evalprint('invVR_t')
    >>> print('\n----')
    >>> vt.evalprint('invVR_2x2_t')
    >>> print('-----')
    >>> #
    >>> print('\n--- CHECKING ROTATION + TRANSLATION 3x3 ---')
    >>> vt.check_expr_eq(invVR_t[:, 2], xyz_t)
    >>> print('\n--- CHECKING ROTATION 2x2 ---')
    >>> vt.check_expr_eq(invVR_t[0:2, 0:2], invVR_2x2_t)
    >>> ####
    >>> ####
    >>> ####
    >>> # Checking orientation property
    >>> [[ivr11, ivr12, ivr13], [ivr21, ivr22, ivr23], [ivr31, ivr32, ivr33],] = invVR.tolist()
    >>> ori = sympy.atan2(ivr12, ivr11)  # outputs from -TAU/2 to TAU/2
    >>> z = ori.subs(dict(iv11=1, theta=1))
    >>> sympy.trigsimp(sympy.simplify(sympy.trigsimp(z)))

    #_oris = np.arctan2(_iv12s, _iv11s)  # outputs from -TAU/2 to TAU/2
    >>> # xdoctest: +SKIP
    >>> # OLD STUFF
    >>> #
    >>> print(ub.hzcat('V = ', repr(V)))
    V = Matrix([
        [          1/iv11,     0,                -1.0*x/iv11],
        [-iv21/(iv11*iv22), 1/iv22, -1.0*(y - iv21*x/iv11)/iv22],
        [               0,     0,                        1.0]])
    >>> print(ub.hzcat('V = ', repr(sympy.simplify(invV.inv()))))
    V = Matrix([
        [          1/iv11,     0,                       -1.0*x/iv11],
        [-iv21/(iv11*iv22), 1/iv22, 1.0*(-iv11*y + iv21*x)/(iv11*iv22)],
        [               0,     0,                               1.0]])



Efficiency Notes:
    single index indexing is very fast

    slicing seems to be very fast.

    fancy indexing with __getitem__ is very slow
    using np.take is a better idea, but its a bit harder
    to use with multidimensional arrays (nope use axis=x)
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range, reduce
import numpy as np
import numpy.linalg as npl
from numpy.core.umath_tests import matrix_multiply
from vtool import linalg as linalgtool
from vtool import chip as chiptool
from vtool import distance
from vtool import trig
import ubelt as ub
import utool as ut
from .util_math import TAU


GRAVITY_THETA = TAU / 4
KPTS_DTYPE = np.float32

XDIM = 0
YDIM = 1
SCAX_DIM = 2
SKEW_DIM = 3
SCAY_DIM = 4
ORI_DIM = 5
LOC_DIMS   = np.array([XDIM, YDIM])
SHAPE_DIMS = np.array([SCAX_DIM, SKEW_DIM, SCAY_DIM])


def get_grid_kpts(wh=(300, 300), wh_stride=None, scale=20, wh_num=None,
                  dtype=np.float32, **kwargs):
    """ Returns a regular grid of keypoints

    Args:
        wh (tuple): (default = (300, 300))
        wh_stride (tuple): stride of keypoints (defaults to (50, 50))
        scale (int): (default = 20)
        wh_num (tuple): desired number of keypoints in x and y direction.
            (incompatible with stride).
        dtype (type): (default = <type 'numpy.float32'>)

    Returns:
        ndarray[float32_t, ndim=2]: kpts -  keypoints

    CommandLine:
        python -m vtool.keypoint get_grid_kpts --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> wh = (300, 300)
        >>> wh_stride = None
        >>> scale = 20
        >>> wh_num = (3, 3)
        >>> dtype = np.float32
        >>> kpts = get_grid_kpts(wh, wh_num=wh_num, dtype=dtype)
        >>> assert len(kpts) == np.prod(wh_num)
        >>> result = ('kpts = %s' % (ub.repr2(kpts.shape),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.show_kpts(kpts)
        >>> pt.dark_background()
        >>> ut.show_if_requested()
    """
    (w, h) = wh
    padding = scale * 1.5
    inner_width = w - 2 * padding
    inner_height = h - 2 * padding
    if wh_num is not None:
        #assert wh_stride is None, 'cannot specify both stride and wh_num'
        nx, ny = wh_num
        wh_stride = (inner_width / nx, inner_height / ny)
    elif wh_stride is None:
        wh_stride = (50, 50)
    (wstride, hstride) = wh_stride
    xbasis = np.arange(padding, (w - padding), wstride)
    ybasis = np.arange(padding, (h - padding), hstride)
    xs_grid, ys_grid = np.meshgrid(xbasis, ybasis)
    _xs = xs_grid.flatten()
    _ys = ys_grid.flatten()
    nKpts = len(_xs)
    _zeros = np.zeros(nKpts, dtype=dtype)
    _iv11s = _zeros + scale
    _iv21s = _zeros
    _iv22s = _zeros + scale
    _oris = _zeros
    kpts = np.vstack((_xs, _ys, _iv11s, _iv21s, _iv22s, _oris)).astype(dtype).T
    return kpts


# --- raw keypoint components ---
def get_xys(kpts):
    """ Keypoint locations in chip space """
    _xys = kpts.T[0:2]
    return _xys


def get_invVs(kpts):
    """ Keypoint shapes (oriented with the gravity vector) """
    _invVs = kpts.T[2:5]
    return _invVs


def get_oris(kpts):
    """ Extracts keypoint orientations for kpts array

    (in isotropic guassian space relative to the gravity vector)
    (in simpler words: the orientation is is taken from keypoints warped to the unit circle)

    Args:
        kpts (ndarray): (N x 6) [x, y, a, c, d, theta]

    Returns:
        (ndarray) theta
    """
    if kpts.shape[1] == 5:
        _oris = np.zeros(len(kpts), dtype=kpts.dtype)
    elif kpts.shape[1] == 6:
        _oris = kpts.T[5]
    else:
        raise AssertionError('[ktool] Invalid kpts.shape = %r' % (kpts.shape,))
    return _oris


# --- keypoint properties ---

def get_sqrd_scales(kpts):
    """
    gets average squared scale (does not take into account elliptical shape

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        np.ndarray

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> _scales_sqrd = get_sqrd_scales(kpts)
        >>> result = (ub.repr2(_scales_sqrd, precision=2))
        >>> print(result)
        np.array([125.98,  56.88, 128.62, 188.37, 188.38])

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> _scales_sqrd = get_sqrd_scales([])
        >>> result = (ub.repr2(_scales_sqrd, precision=2))
        >>> print(result)
        np.array([])
    """
    if len(kpts) == 0:
        return np.empty(0)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _scales_sqrd = np.multiply(_iv11s, _iv22s)
    return _scales_sqrd


def get_scales(kpts):
    """ Gets average scale (does not take into account elliptical shape """
    _scales = np.sqrt(get_sqrd_scales(kpts))
    return _scales


# --- keypoint matrixes ---

def get_ori_mats(kpts):
    """ Returns keypoint orientation matrixes """
    _oris = get_oris(kpts)
    R_mats = [linalgtool.rotation_mat2x2(ori)
              for ori in _oris]
    return R_mats


def convert_kptsZ_to_kpts(kpts_Z):
    """
    Convert keypoints in Z format to invV format
    """
    import vtool as vt
    x, y, e11, e12, e22 = kpts_Z.T
    #import numpy as np
    Z_mats2x2 = np.array([[e11, e12],
                         [e12, e22]])
    Z_mats2x2 = np.rollaxis(Z_mats2x2, 2)
    invV_mats2x2 = vt.decompose_Z_to_invV_mats2x2(Z_mats2x2)
    invV_mats2x2 = invV_mats2x2.astype(np.float32)
    a = invV_mats2x2[:, 0, 0]
    c = invV_mats2x2[:, 1, 0]
    d = invV_mats2x2[:, 1, 1]
    kpts = np.vstack([x, y, a, c, d]).T
    return kpts


# def test_kpts_type(kpts):
#     import vtool as vt
#     invV_mats2x2 = vt.get_invV_mats2x2(kpts)
#     # Test if it is in Z format
#     e11, e12, e22 = kpts.T[[2, 3, 4]]
#     det = e11 * e22 - (e12 ** 2)
#     Z_neg_evidence = (det < 0).sum() / len(det)
#     kpts_invV = vt.convert_kptsZ_to_kpts(kpts)


def get_invV_mats2x2(kpts):
    """
    Returns the keypoint shape (from unit circle to ellipse)
    Ignores translation and rotation component

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[float32_t, ndim=3]: invV_mats

    CommandLine:
        python -m vtool.keypoint --test-get_invV_mats2x2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> invV_mats2x2 = get_invV_mats2x2(kpts)
        >>> # verify results
        >>> result = kpts_repr(invV_mats2x2)
        >>> print(result)
        array([[[1., 0.],
                [2., 3.]],
               [[1., 0.],
                [2., 3.]]])
    """
    nKpts = len(kpts)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _zeros = np.zeros(nKpts)
    invV_arrs2x2 = np.array([[_iv11s, _zeros],
                             [_iv21s, _iv22s]])  # R x C x N
    invV_mats2x2 = np.rollaxis(invV_arrs2x2, 2)  # N x R x C
    return invV_mats2x2


def get_invVR_mats2x2(kpts):
    r"""
    Returns the keypoint shape+rotation matrix (from unit circle to ellipse)
    Ignores translation component

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints

    Returns:
        ndarray: invVR_mats

    CommandLine:
        python -m vtool.keypoint --test-get_invVR_mats2x2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> invVR_mats2x2 = get_invVR_mats2x2(kpts)
        >>> # verify results
        >>> result = kpts_repr(invVR_mats2x2)
        >>> print(result)
        array([[[ 1.,  0.],
                [ 2.,  3.]],
               [[ 0., -1.],
                [ 3., -2.]]])

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts = np.empty((0, 6))
        >>> invVR_mats2x2 = get_invVR_mats2x2(kpts)
        >>> assert invVR_mats2x2.shape == (0, 2, 2)
    """
    if len(kpts) == 0:
        return np.empty((0, 2, 2))
    invV_mats2x2 = get_invV_mats2x2(kpts)
    # You must apply rotations before you apply shape
    # This is because we are dealing with \emph{inv}(V).
    # numpy operates with data on the right (operate right-to-left)
    R_mats2x2  = get_ori_mats(kpts)
    invVR_mats2x2 = matrix_multiply(invV_mats2x2, R_mats2x2)
    return invVR_mats2x2


def augment_2x2_with_translation(kpts, _mat2x2):
    """
    helper function to augment shape matrix with a translation component.
    """
    nKpts = len(kpts)
    # Unpack shape components
    _11s = _mat2x2.T[0, 0]
    _12s = _mat2x2.T[1, 0]
    _21s = _mat2x2.T[0, 1]
    _22s = _mat2x2.T[1, 1]
    # Get translation components
    _13s, _23s = get_xys(kpts)
    # Use homogenous coordinates
    _zeros = np.zeros(nKpts)
    _ones = np.ones(nKpts)
    _arrs3x3 =  np.array([[_11s, _12s, _13s],
                          [_21s, _22s, _23s],
                          [_zeros, _zeros, _ones]])  # R x C x N
    _mats3x3 = np.rollaxis(_arrs3x3, 2)  # N x R x C
    return _mats3x3


def get_invV_mats3x3(kpts):
    r"""
    NEWER FUNCTION

    Returns full keypoint transform matricies from a unit circle to an
    ellipse that has been scaled, skewed, and translated. Into
    the image keypoint position.

    DOES NOT INCLUDE ROTATION

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[float32_t, ndim=3]: invVR_mats -  keypoint shape and rotations (possibly translation)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> invV_arrs3x3 = get_invV_mats3x3(kpts)
        >>> # verify results
        >>> result = kpts_repr(invV_arrs3x3)
        >>> print(result)
        array([[[1., 0., 0.],
                [2., 3., 0.],
                [0., 0., 1.]],
               [[1., 0., 0.],
                [2., 3., 0.],
                [0., 0., 1.]]])
    """
    #nKpts = len(kpts)
    invV_mats2x2 = get_invV_mats2x2(kpts)
    invV_mats3x3 = augment_2x2_with_translation(kpts, invV_mats2x2)
    ## Unpack shape components
    #_iv11s = invV_mats2x2.T[0, 0]
    #_iv12s = invV_mats2x2.T[1, 0]
    #_iv21s = invV_mats2x2.T[0, 1]
    #_iv22s = invV_mats2x2.T[1, 1]
    ## Get translation components
    #_iv13s, _iv23s = get_xys(kpts)
    ## Use homogenous coordinates
    #_zeros = np.zeros(nKpts)
    #_ones = np.ones(nKpts)
    #invV_arrs3x3 =  np.array([[_iv11s, _iv12s, _iv13s],
    #                          [_iv21s, _iv22s, _iv23s],
    #                          [_zeros, _zeros, _ones]])  # R x C x N
    #invV_mats3x3 = np.rollaxis(invV_arrs3x3, 2)  # N x R x C
    return invV_mats3x3


def get_RV_mats_3x3(kpts):
    """
    prefered over get_invV_mats

    Returns:
        V_mats (ndarray) : sequence of matrices that transform an ellipse to unit circle
    """
    invVR_mats = get_invVR_mats3x3(kpts)
    RV_mats = invert_invV_mats(invVR_mats)
    return RV_mats


def get_invVR_mats3x3(kpts):
    r"""
    NEWER FUNCTION

    Returns full keypoint transform matricies from a unit circle to an
    ellipse that has been rotated, scaled, skewed, and translated. Into
    the image keypoint position.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[float32_t, ndim=3]: invVR_mats

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts = np.array([
        ...    [10, 20, 1, 2, 3, 0],
        ...    [30, 40, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> invVR_mats3x3 = get_invVR_mats3x3(kpts)
        >>> # verify results
        >>> result = kpts_repr(invVR_mats3x3)
        >>> print(result)
        array([[[ 1.,  0., 10.],
                [ 2.,  3., 20.],
                [ 0.,  0.,  1.]],
               [[ 0., -1., 30.],
                [ 3., -2., 40.],
                [ 0.,  0.,  1.]]])
    """
    #nKpts = len(kpts)
    invVR_mats2x2 = get_invVR_mats2x2(kpts)
    invVR_mats3x3 = augment_2x2_with_translation(kpts, invVR_mats2x2)
    # Unpack shape components
    #_iv11s = invVR_mats2x2.T[0, 0]
    #_iv12s = invVR_mats2x2.T[1, 0]
    #_iv21s = invVR_mats2x2.T[0, 1]
    #_iv22s = invVR_mats2x2.T[1, 1]
    ## Get translation components
    #_iv13s, _iv23s = get_xys(kpts)
    ## Use homogenous coordinates
    #_zeros = np.zeros(nKpts)
    #_ones = np.ones(nKpts)
    #invVR_arrs =  np.array([[_iv11s, _iv12s, _iv13s],
    #                        [_iv21s, _iv22s, _iv23s],
    #                        [_zeros, _zeros, _ones]])  # R x C x N
    #invVR_mats = np.rollaxis(invVR_arrs, 2)  # N x R x C
    return invVR_mats3x3


def get_invV_mats(kpts, with_trans=False, with_ori=False, ashomog=False, ascontiguous=False):
    """
    TODO: DEPRICATE. too many conditionals

    packs keypoint shapes into affine invV matrixes
    (default is just the 2x2 shape. But translation, orientation,
    homogonous, and contiguous flags can be set.)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts = np.array([[10, 20, 1, 2, 3, 0]])
        >>> with_trans=True
        >>> with_ori=True
        >>> ashomog=True
        >>> ascontiguous=False
        >>> innVR_mats = get_invV_mats(kpts, with_trans, with_ori, ashomog, ascontiguous)
        >>> result = kpts_repr(innVR_mats)
        >>> print(result)
        array([[[ 1.,  0., 10.],
                [ 2.,  3., 20.],
                [ 0.,  0.,  1.]]])
    """
    nKpts = len(kpts)
    if with_ori:
        # these are actually invVR mats
        invV_mats = get_invVR_mats2x2(kpts)
    else:
        invV_mats = get_invV_mats2x2(kpts)
    if with_trans or ashomog:
        #_iv11s = invV_mats[:, 0, 0]
        #_iv12s = invV_mats[:, 0, 1]
        #_iv21s = invV_mats[:, 1, 0]
        #_iv22s = invV_mats[:, 1, 1]
        _iv11s = invV_mats.T[0, 0]
        _iv12s = invV_mats.T[1, 0]
        _iv21s = invV_mats.T[0, 1]
        _iv22s = invV_mats.T[1, 1]
        # Use homogenous coordinates
        _zeros = np.zeros(nKpts)
        _ones = np.ones(nKpts)
        if with_trans:
            _iv13s, _iv23s = get_xys(kpts)
        else:
            _iv13s = _iv23s = _zeros
        invV_arrs =  np.array([[_iv11s, _iv12s, _iv13s],
                               [_iv21s, _iv22s, _iv23s],
                               [_zeros, _zeros, _ones]])  # R x C x N
        invV_mats = np.rollaxis(invV_arrs, 2)  # N x R x C
    if ascontiguous:
        invV_mats = np.ascontiguousarray(invV_mats)
    return invV_mats

# --- scaled and offset keypoint components ---


def get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor=1.0):
    r"""
    Given some patch (like a gaussian patch) transforms a patch to be overlayed
    on top of each keypoint in the image (adjusted for a scale factor)

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch_shape (?):
        scale_factor (float):

    Returns:
        M_list: a list of 3x3 tranformation matricies for each keypoint

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> patch_shape = (7, 7)
        >>> scale_factor = 1.0
        >>> M_list = get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)
        >>> # verify results
        >>> result = kpts_repr(M_list)
        >>> print(result)
        array([[[ 1.49,  0.  , 15.53],
                [-1.46,  6.9 ,  8.68],
                [ 0.  ,  0.  ,  1.  ]],
               [[ 0.67,  0.  , 26.98],
                [-1.46,  6.9 ,  8.68],
                [ 0.  ,  0.  ,  1.  ]],
               [[ 3.49,  0.  , 19.53],
                [ 3.43,  3.01, 10.67],
                [ 0.  ,  0.  ,  1.  ]],
               [[ 3.82,  0.  , 19.55],
                [ 5.04,  4.03,  1.8 ],
                [ 0.  ,  0.  ,  1.  ]],
               [[ 4.59,  0.  , 18.24],
                [ 0.97,  3.35, 18.02],
                [ 0.  ,  0.  ,  1.  ]]])

    Ignore:
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> invVR_aff2Ds = [np.array(((a, 0, x),
        >>>                           (c, d, y),
        >>>                           (0, 0, 1),))
        >>>                 for (x, y, a, c, d, ori) in kpts]
        >>> invVR_3x3 = vt.get_invVR_mats3x3(kpts)
        >>> invV_3x3 = vt.get_invV_mats3x3(kpts)
        >>> assert np.all(np.array(invVR_aff2Ds) == invVR_3x3)
        >>> assert np.all(np.array(invVR_aff2Ds) == invV_3x3)

    Timeit:
        %timeit [np.array(((a, 0, x), (c, d, y), (0, 0, 1),)) for (x, y, a, c, d, ori) in kpts]
        %timeit vt.get_invVR_mats3x3(kpts)
        %timeit vt.get_invV_mats3x3(kpts) <- THIS IS ACTUALLY MUCH FASTER

    Ignore::
        %pylab qt4
        import plottool as pt
        pt.imshow(chip)
        pt.draw_kpts2(kpts)
        pt.update()

    Timeit:
        sa_list1 = np.array([S2.dot(A) for A in invVR_aff2Ds])
        sa_list2 = matrix_multiply(S2, invVR_aff2Ds)
        assert np.all(sa_list1 == sa_list2)
        %timeit np.array([S2.dot(A) for A in invVR_aff2Ds])
        %timeit matrix_multiply(S2, invVR_aff2Ds)

        from six.moves import reduce
        perspective_list2 = np.array([S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds])
        perspective_list = reduce(matrix_multiply, (S2, invVR_aff2Ds, S1, T1))
        assert np.all(perspective_list == perspective_list2)
        %timeit np.array([S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds])
        %timeit reduce(matrix_multiply, (S2, invVR_aff2Ds, S1, T1))
    """
    (patch_h, patch_w) = patch_shape
    half_width  = (patch_w / 2.0)  # - .5
    half_height = (patch_h / 2.0)  # - .5
    # Center src image
    T1 = linalgtool.translation_mat3x3(-half_width + .5, -half_height + .5)
    # Scale src to the unit circle
    #S1 = linalgtool.scale_mat3x3(1.0 / patch_w, 1.0 / patch_h)
    S1 = linalgtool.scale_mat3x3(1.0 / half_width, 1.0 / half_height)
    # Transform the source image to the keypoint ellipse
    invVR_aff2Ds = get_invVR_mats3x3(kpts)
    # Adjust for the requested scale factor
    S2 = linalgtool.scale_mat3x3(scale_factor, scale_factor)
    #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
    M_list = reduce(matrix_multiply, (S2, invVR_aff2Ds, S1.dot(T1)))
    return M_list


def transform_kpts_to_imgspace(kpts, bbox, bbox_theta, chipsz):
    """ Transforms keypoints so they are plotable in imagespace
        kpts   - xyacdo keypoints
        bbox   - chip bounding boxes in image space
        theta  - chip rotationsinvC
        chipsz - chip extent (in keypoint / chip space)
    """
    # Get keypoints in matrix format
    invV_mats = get_invV_mats(kpts, with_trans=True, with_ori=True)
    # Get chip to imagespace transform
    invC = chiptool._get_chip_to_image_transform(bbox, chipsz, bbox_theta)
    # Apply transform to keypoints
    invCinvV_mats = matrix_multiply(invC, invV_mats)
    # Flatten back into keypoint (x, y, a, c, d, o) format
    imgkpts = flatten_invV_mats_to_kpts(invCinvV_mats)
    return imgkpts


def get_kpts_eccentricity(kpts):
    """

    SeeAlso:
        pyhesaff.tests.test_ellipse

    References:
        https://en.wikipedia.org/wiki/Eccentricity_(mathematics)

    Ascii:
        Connic marix is
        Z_mat = np.array((('    A', 'B / 2', 'D / 2'),
                          ('B / 2', '    C', 'E / 2'),
                          ('D / 2', 'E / 2', '    F')))
        ----------------------------------
        The eccentricity is determined by:
        [A, B, C, D] = kpts_mat

                    (2 * np.sqrt((A - C) ** 2 + B ** 2))
        ecc = -----------------------------------------------
              (nu * (A + C) + np.sqrt((A - C) ** 2 + B ** 2))

        nu = 1 if det(Z) > 0, -1 if det(Z) < 0, and 0 if det(Z) == 0

        (nu is always 1 for ellipses.)

    Notes:
        For an ellipse/hyperbola the eccentricity is
        sqrt(1 - (b ** 2 / a ** 2))

        Eccentricity is undefined for parabolas

        where a is the lenth of the semi-major axis and b is the length of the
        semi minor axis. The length of the semi-major axis is 2 time the
        largest eigenvalue.  And the length of the semi-minor axis is 2 times
        the smallest eigenvalue.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        offset (tuple): (default = (0.0, 0.0))
        scale_factor (float): (default = 1.0)

    CommandLine:
        python -m vtool.keypoint --exec-get_kpts_eccentricity --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts_ = vt.demodata.get_dummy_kpts()
        >>> kpts = np.append(kpts_, [[10, 10, 5, 0, 5, 0]], axis=0)
        >>> ecc = get_kpts_eccentricity(kpts)
        >>> result = 'ecc = %s' % (ub.repr2(ecc, precision=2))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> colors = pt.scores_to_color(ecc)
        >>> pt.draw_kpts2(kpts, color=colors, ell_linewidth=6)
        >>> extent = vt.get_kpts_image_extent(kpts)
        >>> ax = pt.gca()
        >>> pt.set_axis_extent(extent, ax)
        >>> pt.dark_background()
        >>> pt.colorbar(ecc, colors)
        >>> ut.show_if_requested()
        ecc = np.array([ 0.96, 0.99, 0.87, 0.91, 0.55, 0.  ])
    """
    RV_mats2x2 = get_RV_mats2x2(kpts)
    Z_mats2x2 = get_Z_mats(RV_mats2x2)
    A = Z_mats2x2[:, 0, 0]
    B = Z_mats2x2[:, 0, 1] * 2
    C = Z_mats2x2[:, 1, 1]
    nu = 1
    numer = (2 * np.sqrt((A - C) ** 2 + B ** 2))
    denom = (nu * (A + C) + np.sqrt((A - C) ** 2 + B ** 2))
    ecc = numer / denom
    return ecc


def offset_kpts(kpts, offset=(0.0, 0.0), scale_factor=1.0):
    r"""
    Transfoms keypoints by a scale factor and a translation

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        offset (tuple):
        scale_factor (float):

    Returns:
        ndarray[float32_t, ndim=2]: kpts -  keypoints

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts().astype(np.float64)
        >>> offset = (0.0, 0.0)
        >>> scale_factor = (1.5, 0.5)
        >>> kpts_ = offset_kpts(kpts, offset, scale_factor)
        >>> # verify results (hack + 0. to fix negative 0)
        >>> result = ut.repr3((kpts, kpts_ + 0.), precision=2, nobr=True, with_dtype=True)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_kpts2(kpts, color=pt.ORANGE, ell_linewidth=6)
        >>> pt.draw_kpts2(kpts_, color=pt.LIGHT_BLUE, ell_linewidth=4)
        >>> extent1 = np.array(vt.get_kpts_image_extent(kpts))
        >>> extent2 = np.array(vt.get_kpts_image_extent(kpts_))
        >>> extent = vt.union_extents([extent1, extent2])
        >>> ax = pt.gca()
        >>> pt.set_axis_extent(extent)
        >>> pt.dark_background()
        >>> ut.show_if_requested()
        np.array([[20.  , 25.  ,  5.22, -5.11, 24.15,  0.  ],
                  [29.  , 25.  ,  2.36, -5.11, 24.15,  0.  ],
                  [30.  , 30.  , 12.22, 12.02, 10.53,  0.  ],
                  [31.  , 29.  , 13.36, 17.63, 14.1 ,  0.  ],
                  [32.  , 31.  , 16.05,  3.41, 11.74,  0.  ]], dtype=np.float64),
        np.array([[30.  , 12.5 ,  7.82, -2.56, 12.07,  0.  ],
                  [43.5 , 12.5 ,  3.53, -2.56, 12.07,  0.  ],
                  [45.  , 15.  , 18.32,  6.01,  5.26,  0.  ],
                  [46.5 , 14.5 , 20.03,  8.82,  7.05,  0.  ],
                  [48.  , 15.5 , 24.08,  1.7 ,  5.87,  0.  ]], dtype=np.float64),
    """
    if (np.all(offset == (0.0, 0.0)) and
        (np.all(scale_factor == 1.0) or
         np.all(scale_factor == (1.0, 1.0)))):
        return kpts
    try:
        sfx, sfy = scale_factor
    except TypeError:
        sfx = sfy = scale_factor
    tx, ty = offset
    T = linalgtool.translation_mat3x3(tx, ty)
    S = linalgtool.scale_mat3x3(sfx, sfy)
    M = T.dot(S)
    #M = linalgtool.scaleedoffset_mat3x3(offset, scale_factor)
    kpts_ = transform_kpts(kpts, M)
    return kpts_


def transform_kpts(kpts, M):
    r"""
    returns M.dot(kpts_mat)
    Currently, only works if M is affine.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        M (ndarray): affine transform matrix

    Returns:
        ndarray: kpts_

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> M = np.array([[10, 0, 0], [10, 10, 0], [0, 0, 1]], dtype=np.float64)
        >>> kpts = transform_kpts(kpts, M)
        >>> # verify results
        >>> result = ub.repr2(kpts, precision=3, with_dtype=True).replace('-0. ', ' 0. ')
        >>> print(result)
        np.array([[200.   , 450.   ,  52.166,   1.056, 241.499,   0.   ],
                  [290.   , 540.   ,  23.551, -27.559, 241.499,   0.   ],
                  [300.   , 600.   , 122.166, 242.357, 105.287,   0.   ],
                  [310.   , 600.   , 133.556, 309.899, 141.041,   0.   ],
                  [320.   , 630.   , 160.527, 194.6  , 117.354,   0.   ]], dtype=np.float64)

    IGNORE:
        >>> # HOW DO WE KEEP SHAPE AFTER HOMOGRAPHY?
        >>> # DISABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> M = np.array([[ 3.,  3.,  5.],
        ...               [ 2.,  3.,  6.],
        ...               [ 1.,  1.,  2.]])
        >>> invVR_mats3x3 = get_invVR_mats3x3(kpts)
        >>> MinvVR_mats3x3 = matrix_multiply(M, invVR_mats3x3)
        >>> MinvVR_mats3x3 = np.divide(MinvVR_mats3x3, MinvVR_mats3x3[:, None, None, 2, 2])  # 2.6 us
        >>> MinvVR = MinvVR_mats3x3[0]
        >>> result = kpts_repr(MinvVR)
        >>> print(result)

        # Inspect matrix decompositions
        import numpy.linalg as npl
        print(ub.hzcat('MinvVR = ', kpts_repr(MinvVR)))
        U, s, Vt = npl.svd(MinvVR)
        S = np.diagflat(s)
        print(ub.hzcat('SVD: U * S * Vt = ', U, ' * ', S, ' * ', Vt, precision=3))
        Q, R = npl.qr(MinvVR)
        print(ub.hzcat('QR: Q * R = ', Q, ' * ', R, precision=3))
        #print('cholesky = %r' % (npl.cholesky(MinvVR),))
        #npl.cholesky(MinvVR)


        print(ub.hzcat('MinvVR = ', kpts_repr(MinvVR)))
        MinvVR_ = MinvVR / MinvVR[None, 2, :]
        print(ub.hzcat('MinvVR_ = ', kpts_repr(MinvVR_)))

    """
    invVR_mats3x3 = get_invVR_mats3x3(kpts)
    MinvVR_mats3x3 = matrix_multiply(M, invVR_mats3x3)
    try:
        assert np.all(MinvVR_mats3x3[:, 2, 0:2] == 0)
        assert np.all(MinvVR_mats3x3[:, 2, 2] == 1)
    except AssertionError as ex:  # NOQA
        # THERE IS NO WAY TO GET KEYPOINTS TRANFORMED BY A HOMOGENOUS
        # TRANSFORM MATRIX INTO THE 6 COMPONENT KEYPOINT VECTOR.
        #print(ex)
        #oris = get_invVR_mats_oris(MinvVR_mats3x3)
        #Lmats = [linalgtool.rotation_mat3x3(-ori) for ori in oris]
        #matrix_multiply(MinvVR_mats3x3, Lmats)
        #matrix_multiply(Lmats, MinvVR_mats3x3)
        #scipy.linalg.lu(MinvVR_mats3x3[0])
        #scipy.linalg.qr(MinvVR_mats3x3[0])
        import warnings
        warnings.warn('WARNING: [vtool.keypoint] transform produced non-affine keypoint')
        # We can approximate it very very roughly
        MinvVR_mats3x3 = np.divide(MinvVR_mats3x3, MinvVR_mats3x3[:, None, None, 2, 2])  # 2.6 us
        raise
        #MinvVR_mats3x3 / MinvVR_mats3x3[:, None, None, 2, :]
    kpts_ = flatten_invV_mats_to_kpts(MinvVR_mats3x3)
    return kpts_


def transform_kpts_xys(H, kpts):
    r"""
    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix

    Returns:
        ndarray: xy_t

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> H = np.array([[ 3.,  3.,  5.],
        ...               [ 2.,  3.,  6.],
        ...               [ 1.,  1.,  2.]])
        >>> xy_t = transform_kpts_xys(H, kpts)
        >>> # verify results
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> result = ub.repr2(xy_t, precision=3, with_dtype=True)
        >>> print(result)
        np.array([[ 2.979, 2.982, 2.984, 2.984, 2.985],
                  [ 2.574, 2.482, 2.516, 2.5  , 2.508]], dtype=np.float64)

    Ignore::
        %pylab qt4
        import plottool as pt
        pt.imshow(chip)
        pt.draw_kpts2(kpts)
        pt.update()
    """
    xy = get_xys(kpts)
    xy_t = linalgtool.transform_points_with_homography(H, xy)
    return xy_t
    #xyz   = get_homog_xyzs(kpts)
    #xyz_t = matrix_multiply(H, xyz)
    #xy_t  = linalgtool.add_homogenous_coordinate(xyz_t)
    #return xy_t

#---------------------
# invV_mats functions
#---------------------


def get_invVR_mats_sqrd_scale(invVR_mats):
    """ Returns the squared scale of the invVR keyponts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(7, 3, 3).astype(np.float64)
        >>> det_arr = get_invVR_mats_sqrd_scale(invVR_mats)
        >>> result = ub.repr2(det_arr, precision=2, with_dtype=True)
        >>> print(result)
        np.array([-0.16, -0.09, -0.34, 0.59, -0.2 , 0.18, 0.06], dtype=np.float64)
    """
    det_arr = npl.det(invVR_mats[:, 0:2, 0:2])
    return det_arr


def get_invVR_mats_shape(invVR_mats):
    """ Extracts keypoint shape components

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
        >>> output = get_invVR_mats_shape(invVR_mats)
        >>> result = ut.hash_data(output)
        >>> print(result)
        pibujdiaimwcnmomserkcytyyikahjmp

    References:
        TODO
        (a.ravel()[(cols + (rows * a.shape[1]).reshape((-1,1))).ravel()]).reshape(rows.size, cols.size)
        http://stackoverflow.com/questions/14386822/fast-numpy-fancy-indexing
        # So, this doesn't work
        # Try this instead
        http://docs.cython.org/src/userguide/memoryviews.html#memoryviews
    """
    _iv11s = invVR_mats[:, 0, 0]
    _iv12s = invVR_mats[:, 0, 1]
    _iv21s = invVR_mats[:, 1, 0]
    _iv22s = invVR_mats[:, 1, 1]
    return (_iv11s, _iv12s, _iv21s, _iv22s)


def get_invVR_mats_xys(invVR_mats):
    r"""
    extracts locations
    extracts xys from matrix encoding, Its just the (0, 2), and (1, 2) components

    Args:
        invVR_mats (ndarray) : list of matrices mapping ucircles to ellipses

    Returns:
        ndarray: the xy location

    Timeit:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> setup = ut.codeblock(
        ...     '''
                import numpy as np
                np.random.seed(0)
                invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
                ''')
        >>> stmt_list = ut.codeblock(
        ...     '''
                invVR_mats[:, 0:2, 2].T
                invVR_mats.T[2, 0:2]
                invVR_mats.T.take(2, axis=0).take([0, 1], axis=0)
                invVR_mats.T.take(2, axis=0)[0:2]
                '''
        ... ).split('\n')
        >>> ut.util_dev.timeit_compare(stmt_list, setup, int(1E5))

    Example:
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
        >>> invVR_mats.T[2, 0:2]
    """
    # ORIG NUMPY
    #_xys = invVR_mats[:, 0:2, 2].T
    # BETTER NUMPY
    _xys = invVR_mats.T[2, 0:2]
    return _xys


def get_invVR_mats_oris(invVR_mats):
    r""" extracts orientation from matrix encoding, this is a bit tricker
    can use -arctan2 or (0, 0) and (0, 1), but then have to normalize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(7, 2, 2).astype(np.float64)
        >>> output = get_invVR_mats_oris(invVR_mats)
        >>> result = ub.repr2(output, precision=2, with_dtype=True)
        np.array([5.37, 5.29, 5.9 , 5.26, 4.74, 5.6 , 4.9 ], dtype=np.float64)

    Sympy:
        >>> # DISABLE_DOCTEST
        >>> # BEST PROOF SO FAR OF EXTRACTION FROM ARBITRARY COMPOMENTS
        >>> from vtool.keypoint import *
        >>> import vtool as vt
        >>> import sympy
        >>> symkw = dict(real=True, finite=True)
        >>> #x, y, v21  = sympy.symbols('x, y, v21', **symkw)
        >>> #v11, v22    = sympy.symbols('v11, v22', positive=True, **symkw)
        >>> x, y, v21  = sympy.symbols('x, y, c', **symkw)
        >>> v11, v22    = sympy.symbols('a, d', positive=True, **symkw)
        >>> theta       = sympy.symbols('theta', **symkw)
        >>> symtau = 2 * sympy.pi
        >>> # Forward rotation
        >>> keypoint_terms = [x, y, v11, v21, v22, theta]
        >>> # Ell to ucircle
        >>> V = vt.sympy_mat([
        >>>         [v11, 0.0,  0],
        >>>         [v21, v22,  0],
        >>>         [0.0, 0.0, 1.0]])
        >>> # Backwards rotation
        >>> R = vt.sympy_mat([
        >>>         [sympy.cos(-theta), -sympy.sin(-theta), 0],
        >>>         [sympy.sin(-theta), sympy.cos(-theta), 0],
        >>>         [               0,                 0, 1]])
        >>> # Backwards translation
        >>> T = vt.sympy_mat([
        >>>        [  1, 0.0,  -x],
        >>>        [  0,   1,  -y],
        >>>        [ 0.0, 0.0, 1.0]])
        >>> # Scale is the inverse square root determinant of the shape matrix.
        >>> scale = 1 / sympy.sqrt(sympy.det(V))
        >>> # Inverse of components
        >>> invT = T.inv_()
        >>> invR = vt.sympy_mat(sympy.simplify(R.inv_()))
        >>> invV = V.inv_()  # TODO: figure out how to make -theta say inside sin and cos
        >>> # -----------
        >>> # Build the B matrix
        >>> RVT_held_full = R.matmul(V, hold=True).matmul(T, hold=True)
        >>> RVT_full = RVT_held_full.as_mutable()
        >>> # Build the inv(B) matrix
        >>> invTVR_held_full = invT.matmul(invV, hold=True).matmul(invR, hold=True)
        >>> invTVR_full = invTVR_held_full.as_mutable()
        >>> # ------------------------
        >>> # Build the invTVR_full in arbitrary terms
        >>> iv11, iv12, iv13, iv21, iv22, iv23 = sympy.symbols('iv11, iv12, iv13, iv21, iv22, iv23', **symkw)
        >>> arb_symbols = [iv11, iv12, iv13, iv21, iv22, iv23]
        >>> invVR_arb = vt.sympy_mat([
        >>>        [  iv11, iv12, iv13],
        >>>        [  iv21, iv22, iv23],
        >>>        [   0.0, 0.0, 1.0]])
        >>> # Set set terms equal to the construction from the inverse
        >>> arb_expr1 = sympy.Eq(invVR_arb, invTVR_full)
        >>> arb_assign = sympy.solve(arb_expr1, arb_symbols)
        >>> # Solve for keypoint varibles in terms of the arbitrary invVR_arb mat
        >>> solutions = sympy.solve(arb_expr1, x, y)
        >>> solutions[theta] = sympy.solve(arb_expr1, theta)
        >>> # Solutions for scale is not well defined, but can be taken through the determ
        >>> #solutions_ad = sympy.solve(arb_expr1, v11, v22)
        >>> #solutions_scale = sympy.solve(arb_expr1, scale)
        >>> #solutions = sympy.solve(arb_expr1, *keypoint_terms)
        >>> # ------------------------
        >>> # Print review info (ell to ucirc)
        >>> print('Keypoint review (RVT):')
        >>> indenter = ut.Indenter('[RVT] ')
        >>> print = indenter.start()
        >>> print('Translate keypoint to origin')
        >>> vt.evalprint('T')
        >>> print('Warp from ellipse to unit circle shape:')
        >>> vt.evalprint('V')
        >>> print('Orientation normalize by -theta radians')
        >>> vt.evalprint('R')
        >>> print('These can be combined as such:')
        >>> vt.evalprint('RVT_held_full')
        >>> print('This simplifies to a matrix which tranlates, scales, skews, and rotates an ellipse into a unit circle')
        >>> print('(B) = RVT')
        >>> vt.evalprint('RVT_full')
        >>> print = indenter.stop()
        >>> # ------------------------
        >>> # Print review info (ucirc to ell)
        >>> print('\nNow backwards:')
        >>> print('Unorient')
        >>> indenter = ut.Indenter('[invTVR] ')
        >>> print = indenter.start()
        >>> vt.evalprint('invR')
        >>> print('Warp from unit circle to ellipse')
        >>> vt.evalprint('invV')
        >>> print('Translate to point in annot space')
        >>> vt.evalprint('invT')
        >>> print('These can be combined as such:')
        >>> vt.evalprint('invTVR_held_full')
        >>> print('This simplifies to a matrix which rotates, skews, scales, and translates a unit circle into an ellipse')
        >>> print('inv(B) = inv(T) inv(V) inv(R)')
        >>> vt.evalprint('invTVR_full')
        >>> print = indenter.stop()
        >>> # ------------------------
        >>> # Now we will solve for keypoint componts given an arbitrary shape matrix
        >>> print('\n')
        >>> print('Given an arbitrary invVRT shape matrix')
        >>> vt.evalprint('invVR_arb')
        >>> print('The keypoint components can be extracte as such')
        >>> print('The position is easy')
        >>> print('Scale is not found through symbolic manipulation but can be taken through linear algebra properites')
        >>> print('Orientation is a bit more involved')
        >>> print(ub.repr2(solutions, sorted_=True))
        >>> # PROOVE ORIENTATION EQUATION IS CORRECT
        >>> #ivr11 must be positive for this to work
        >>> ori_arb = (-sympy.atan2(iv12, iv11)) % (symtau)
        >>> ori_arb_nomod = sympy.atan2(iv12, iv11)  # outputs from -TAU/2 to TAU/2
        >>> scale_arb = sympy.sqrt(sympy.det(invVR_arb))
        >>> print('\n CLAIM:')
        >>> print('Scale is be computed as:')
        >>> vt.evalprint('scale_arb')
        >>> vt.evalprint('scale_arb.subs(arb_assign)')
        >>> vt.evalprint('scale_arb.subs(arb_assign)', simplify=True)
        >>> print('\n CLAIM:')
        >>> print('Orientation is be computed as:')
        >>> vt.evalprint('ori_arb')
        >>> vt.evalprint('ori_arb.subs(arb_assign)')
        >>> vt.evalprint('ori_arb.subs(arb_assign)', simplify=True)
        >>> ori_subs = ori_arb.subs(arb_assign)
        >>> print('Consider only the arctan2 part')
        >>> # hack aroung resolve atan2
        >>> ori_arb_nomod = sympy.atan2(iv12, iv11)
        >>> from sympy.assumptions.refine import refine_atan2
        >>> # There are 3 cases we need to wory about for atan2(y, x)
        >>> # Case where x is positive
        >>> atan2_case1 = refine_atan2(ori_arb_nomod,
        >>>      sympy.Q.real(iv12) & sympy.Q.positive(iv11))
        >>> # Case where x is negative and y is non-negative
        >>> atan2_case2 = refine_atan2(ori_arb_nomod,
        >>>     sympy.Q.negative(iv11) & sympy.Q.positive(iv12))
        >>> # Case where x is negative and y is negative
        >>> atan2_case3 = refine_atan2(ori_arb_nomod,
        >>>     sympy.Q.negative(iv11) & sympy.Q.negative(iv12))
        >>> atan2_case_strs = ['QI, QIV', 'QII', 'QIII']
        >>> theta_ranges = [(-TAU / 4, TAU / 4, False, False), (TAU / 4, TAU / 2, True, True), (TAU / 2, 3 * TAU / 4, False, True)]
        >>> atan2_case_list = [atan2_case1, atan2_case2, atan2_case3]
        >>> for caseno, atan2_case in enumerate(atan2_case_list):
        >>>     print('\n----\ncaseno = %r' % (caseno,))
        >>>     print('Quadrent: %r'  % (atan2_case_strs[caseno]))
        >>>     print('theta_ranges: %r'  % (theta_ranges[caseno],))
        >>>     atan2_case_subs = atan2_case.subs(arb_assign)
        >>>     vt.evalprint('atan2_case_subs')
        >>>     atan2_case_subs = sympy.simplify(atan2_case_subs)
        >>>     atan2_case_subs = sympy.trigsimp(atan2_case_subs)
        >>>     vt.evalprint('atan2_case_subs')
        >>>     ori_arb_case = (-atan2_case) % (symtau)
        >>>     ori_arb_case_subs = ori_arb_case.subs(arb_assign)
        >>>     ori_arb_case_subs = sympy.simplify(ori_arb_case_subs)
        >>>     ori_arb_case_subs = sympy.trigsimp(ori_arb_case_subs)
        >>>     vt.evalprint('ori_arb_case_subs')
        >>> #

        nptheta = np.linspace(0, 2 * np.pi, 32, endpoint=False)

        mapping = np.arctan(np.tan(nptheta))
        print(ub.repr2(zip(nptheta / (2 * np.pi), nptheta, mapping, nptheta == mapping), precision=3))
        print(ub.repr2(zip(nptheta / (2 * np.pi), nptheta, mapping  % (np.pi * 2), nptheta == mapping % (np.pi * 2)), precision=3))

        >>> # NUMPY CHECKS

        >>> nptheta_special = [ np.arccos(0), -np.arccos(0), -np.arcsin(0), np.arcsin(0) ]
        >>> nptheta = np.array(np.linspace(0, 2 * np.pi, 64, endpoint=False).tolist() + nptheta_special)
        >>> # Case 1
        >>> #\modfn{\paren{-\atan{\tan{(-\theta)}}} }{\TAU}           &\text{if } \cos{(-\theta )} > 0 \\
        >>> flags = np.cos(-nptheta) > 0
        >>> case1_theta  = nptheta.compress(flags)
        >>> case1_result = (-np.arctan(np.tan(-case1_theta)) % TAU)
        >>> case1_theta == case1_result

        >>> print(ub.repr2(zip(case1_theta, case1_result, vt.ori_distance(case1_theta, case1_result) ), precision=3))
        >>> #
        >>> # Case 2
        >>> #\modfn{\paren{-\atan{\tan{(-\theta)}} - \pi }}{\TAU}     &\text{if } \cos{(-\theta )} < 0 \AND \sin{(-\theta )} \ge 0 \\
        >>> flags = (np.cos(-nptheta) < 0) * (np.sin(-nptheta) >= 0)
        >>> case2_theta =  nptheta.compress(flags)
        >>> case2_result = (-np.arctan(np.tan(-case2_theta)) - np.pi) % TAU
        >>> print(ub.repr2(zip(case2_theta, case2_result, vt.ori_distance(case2_theta, case2_result) ), precision=3))
        >>> # Case 3
        >>> #\modfn{\paren{-\atan{\tan{(-\theta)}} + \pi }}{\TAU} &\text{if } \cos{(-\theta )} < 0 \AND \sin{(-\theta )} < 0 \\
        >>> flags = (np.cos(-nptheta) < 0) * (np.sin(-nptheta) < 0)
        >>> case3_theta =  nptheta.compress(flags)
        >>> case3_result = (-np.arctan(np.tan(-case3_theta)) + np.pi) % TAU
        >>> print(ub.repr2(zip(case3_theta, case3_result, vt.ori_distance(case3_theta, case3_result)), precision=3))
        >>> # Case 4
        >>> #\modfn{\paren{-\frac{\pi}{2} }}{\TAU}                &\text{if } \cos{(-\theta )} = 0 \AND \sin{(-\theta )} > 0 \\
        >>> # There are 2 locations with cos(-theta) = 0 and sing(-theta) > 0
        >>> # case4_theta = [ 3 * TAU / 4,   -TAU / 4]
        >>> cosine0_theta = np.array([TAU / 4, TAU * 3 / 4, -TAU / 4, -TAU * 3 / 4]) # positions with cosine = 0
        >>> flags = (np.isclose(np.cos(-cosine0_theta), 0) * (np.sin(-cosine0_theta) > 0))
        >>> case4_theta =  cosine0_theta.compress(flags)
        >>> print('case4_theta = %r =? %r' % (case4_theta, (-TAU / 4) % TAU))
        >>> # Case 5
        >>> # There are 2 locations with cos(-theta) = 0 and sing(-theta) < 0
        >>> # case4_theta = [ -3 * TAU / 4,   TAU / 4]
        >>> #\modfn{\paren{\frac{\pi}{2} }}{\TAU}                &\text{if } \cos{(-\theta )} = 0 \AND \sin{(-\theta )} < 0 \\
        >>> flags = (np.isclose(np.cos(-cosine0_theta), 0) * (np.sin(-cosine0_theta) < 0))
        >>> case5_theta =  cosine0_theta.compress(flags)
        >>> print('case5_theta = %r =? %r' % (case5_theta, (TAU / 4) % TAU))

        # numpy check


        >>> # LATEX PART
        >>> expr1_repr = vt.sympy_latex_repr(invTVR_held_full)
        >>> print(expr1_repr)
        >>> ut.copy_text_to_clipboard(expr1_repr)
        >>>
        >>> expr1_repr = vt.sympy_latex_repr(invTVR_full)
        >>> print(expr1_repr)
        >>> ut.copy_text_to_clipboard(expr1_repr)


        >>> from sympy import Symbol, Q, refine, atan2
        >>> from sympy.assumptions.refine import refine_atan2
        >>> from sympy.abc import x, y
        >>> print(refine_atan2(atan2(y,x), Q.real(y) & Q.positive(x)))
        >>> print(refine_atan2(atan2(y,x), Q.negative(y) & Q.negative(x)))
        >>> print(refine_atan2(atan2(y,x), Q.positive(y) & Q.negative(x)))
        atan(y/x)
        atan(y/x) - pi
        atan(y/x) + pi

        >>> negtheta = sympy.symbols('negtheta', **symkw)
        >>> ori_subs2 = sympy.simplify(sympy.trigsimp(ori_subs))

        >>> ori_subs3 = ori_subs2.subs({theta:-negtheta})
        >>> ori_subs4 = sympy.simplify(ori_subs3)
        Out[45]: Mod(-atan2(sin(negtheta)/a, cos(negtheta)/a), 2*pi)

        SimpleError:
            import sympy
            from sympy.assumptions.refine import refine_atan2
            symkw = dict(real=True, finite=True)
            a = sympy.symbols('a', positive=True, **symkw)
            theta  = sympy.symbols('theta', **symkw)

            iv11, iv12  = sympy.symbols('iv11, iv12', **symkw)
            arb_assign = {
                 iv12: -sympy.sin(theta)/a,
                 iv11: sympy.cos(theta)/a,
            }

            ori_subs_nomod = sympy.atan2(-sympy.sin(theta)/a, sympy.cos(theta)/a)
            atan2_case1 = refine_atan2(ori_subs_nomod,
                sympy.Q.real(arb_assign[iv12]) & sympy.Q.positive(arb_assign[iv11])
            )



        >>> ori_subs3 = ori_subs2.subs({theta:0})
        >>> ori_subs3 = ori_subs2.subs(dict(theta=0), simultanious=True)
        for sym in ori_subs2.free_symbols:
            print('%r.assumptions0 = %s' % (sym, ub.repr2(sym.assumptions0),))



        >>> #invTVR = sympy.simplify(RVT_full.inv())
        >>> expr1_repr = vt.sympy_latex_repr(invTVR_full)
        >>> print(expr1_repr)
        >>> ut.copy_text_to_clipboard(expr1_repr)

    Sympy:
        >>> import sympy
        >>> import vtool as vt
        >>> # First orient a unit circle
        >>> symkw = dict(real=True, finite=True)
        >>> theta       = sympy.symbols('theta', **symkw)
        >>> x, y, iv21  = sympy.symbols('x y iv21', **symkw)
        >>> vx, vy, v21 = sympy.symbols('vx, vy, v21', **symkw)
        >>> iv11, iv22  = sympy.symbols('iv11 iv12', positive=True, **symkw)
        >>> v11, v22    = sympy.symbols('v11 v22', positive=True, **symkw)
        >>> # Forward rotation
        >>> invR = vt.sympy_mat([
        >>>         [sympy.cos(theta), -sympy.sin(theta), 0],
        >>>         [sympy.sin(theta), sympy.cos(theta), 0],
        >>>         [              0,          0,      1]])
        >>> # Warps a unit circle at (0, 0) onto an ellipse at (x, y)
        >>> invV = vt.sympy_mat([
        >>>         [iv11, 0.0,  x],
        >>>         [iv21, iv22,  y],
        >>>         [ 0.0, 0.0, 1.0]])
        >>> V = vt.sympy_mat([
        >>>         [v11, 0.0, vx],
        >>>         [v21, v22, vy],
        >>>         [0.0, 0.0, 1.0]])
        veq = sympy.Eq(invVR, VR.inv())
        print('iv11 = ' + str(sympy.solve(veq, iv11)))
        print('iv21 = ' + str(sympy.solve(veq, iv21)))
        print('iv22 = ' + str(sympy.solve(veq, iv22)))
        print('x = ' + str(sympy.solve(veq, x)))
        print('y = ' + str(sympy.solve(veq, y)))
        inveq = sympy.Eq(V, invV.inv())
        print('v11 = ' + str(sympy.solve(inveq, v11)))
        print('v12 = ' + str(sympy.solve(inveq, v21)))
        print('v22 = ' + str(sympy.solve(inveq, v22)))
        >>> invVR = invV.multiply(R)
        >>> invV.matmul(R, hold=True)
        >>> ut.eval
        >>> print(invVR)
        >>> print(repr(invVR))
        >>> vt.rrrr()
        >>> other_repr = vt.sympy_latex_repr(invV.matmul(R, hold=True))
        >>> print(other_repr)
        >>> ut.copy_text_to_clipboard(other_repr)
        >>> expr1_repr = vt.sympy_latex_repr(invVR)
        >>> print(expr1_repr)
        >>> ut.copy_text_to_clipboard(expr1_repr)


    Sympy:
        >>> # Show orientation property
        >>> import sympy
        >>> import vtool as vt
        >>> # First orient a unit circle
        >>> theta = sympy.symbols('theta', real=True)
        >>> x, y, iv21 = sympy.symbols('x y g', real=True, finite=True)
        >>> vx, vy, v21 = sympy.symbols('vx, vy, c', real=True, finite=True)
        >>> iv11, iv22 = sympy.symbols('e h', real=True, finite=True, positive=True)
        >>> v11, v22 = sympy.symbols('a d', positive=True, real=True, finite=True)
        >>> # Forward rotation
        >>> invR = vt.sympy_mat([
        >>>         [sympy.cos(theta), -sympy.sin(theta), 0],
        >>>         [sympy.sin(theta), sympy.cos(theta), 0],
        >>>         [              0,          0,      1]])
        >>> # Warps a unit circle at (0, 0) onto an ellipse at (x, y)
        >>> invV = vt.sympy_mat([
        >>>         [iv11, 0.0,  x],
        >>>         [iv21, iv22,  y],
        >>>         [ 0.0, 0.0, 1.0]])
        >>> V = vt.sympy_mat([
        >>>         [v11, 0.0, vx],
        >>>         [v21, v22, vy],
        >>>         [0.0, 0.0, 1.0]])
        veq = sympy.Eq(invVR, VR.inv())
        print('iv11 = ' + str(sympy.solve(veq, iv11)))
        print('iv21 = ' + str(sympy.solve(veq, iv21)))
        print('iv22 = ' + str(sympy.solve(veq, iv22)))
        print('x = ' + str(sympy.solve(veq, x)))
        print('y = ' + str(sympy.solve(veq, y)))
        inveq = sympy.Eq(V, invV.inv())
        print('v11 = ' + str(sympy.solve(inveq, v11)))
        print('v12 = ' + str(sympy.solve(inveq, v21)))
        print('v22 = ' + str(sympy.solve(inveq, v22)))
        >>> invVR = invV.multiply(R)
        >>> invV.matmul(R, hold=True)
        >>> ut.eval
        >>> print(invVR)
        >>> print(repr(invVR))
        >>> vt.rrrr()
        >>> other_repr = vt.sympy_latex_repr(invV.matmul(R, hold=True))
        >>> print(other_repr)
        >>> ut.copy_text_to_clipboard(other_repr)
        >>> expr1_repr = vt.sympy_latex_repr(invVR)
        >>> print(expr1_repr)
        >>> ut.copy_text_to_clipboard(expr1_repr)
        Matrix([
        [                 iv11*cos(theta),                  -iv11*sin(theta),  x],
        [iv21*cos(theta) + iv22*sin(theta), -iv21*sin(theta) + iv22*cos(theta),  y],
        [                               0,                                 0, 1.0]])
        >>> print(sympy.latex(invVR))
        >>> # Now extract the orientation from any invVR formated matrix
        >>> [[ivr11, ivr12, ivr13], [ivr21, ivr22, ivr23], [ivr31, ivr32, ivr33],] = invVR.tolist()
        >>> # tan = sin / cos
        >>> symtau = 2 * sympy.pi
        >>> #ivr11 must be positive for this to work
        >>> ori = (-sympy.atan2(ivr12, ivr11)) % (symtau)  # outputs from -TAU/2 to TAU/2
        >>> # Check Equality with a domain
        >>> expr1 = ori
        >>> expr2 = theta
        >>> domain = {theta: (0, 2 * np.pi)}
        >>> truth_list, results_list, input_list = vt.symbolic_randcheck(ori, theta, domain, n=7)
        >>> print(ub.repr2(truth_list, precision=2))
        >>> print(ub.repr2(results_list, precision=2))
        >>> print(ub.repr2(input_list, precision=2))
        >>> difference = results_list.T[1] - results_list.T[0]
        >>> print('diff = ' + ub.repr2(difference))
        >>> print('ori diff = ' + ub.repr2(vt.ori_distance(results_list.T[1], results_list.T[0])))

        truth_list, results_list, input_list =
        check_random_points(sympy.sin(theta) / sympy.cos(theta),
        sympy.tan(theta))
        _oris = (-trig.atan2(_iv12s, _iv11s)) % TAU
        ori.evalf(subs=dict(iv11=1, theta=3), verbose=True)
        sympy.trigsimp(sympy.simplify(sympy.trigsimp(z)))
        ori = np.arctan2(_iv12s, _iv11s)
        z = ori.subs(dict(iv11=1, theta=1))

    Timeit:
        >>> import utool as ut
        >>> setup = ut.codeblock(
        ...     '''
                import numpy as np
                np.random.seed(0)
                invVR_mats = np.random.rand(10000, 2, 2).astype(np.float64)
                ''')
        >>> stmt_list = ut.codeblock(
        ...     '''
                invVR_mats[:, 0, 1]
                invVR_mats.T[1, 0]
                '''
        ... ).split('\n')
        >>> ut.util_dev.rrr()
        >>> ut.util_dev.timeit_compare(stmt_list, setup, int(1E3))
    """
    # Extract only the needed shape components
    #_iv11s = invVR_mats[:, 0, 0]
    #_iv12s = invVR_mats[:, 0, 1]
    _iv11s = invVR_mats.T[0, 0]
    _iv12s = invVR_mats.T[1, 0]
    # Solve for orientations. Adjust gravity vector pointing down
    _oris = (-trig.atan2(_iv12s, _iv11s)) % TAU
    return _oris
    "#endif"


def rectify_invV_mats_are_up(invVR_mats):
    """
    Useful if invVR_mats is no longer lower triangular
    rotates affine shape matrixes into downward (lower triangular) position

    CommandLine:
        python -m vtool.keypoint --exec-rectify_invV_mats_are_up --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> rng = np.random.RandomState(0)
        >>> kpts = vt.demodata.get_dummy_kpts()[0:2]
        >>> # Shrink x and y scales a bit
        >>> kpts.T[2:4] /= 2
        >>> kpts[1][3] *= 3  # increase skew
        >>> # Set random orientation
        >>> kpts.T[5] = TAU * np.array([.2, .6])
        >>> invVR_mats = get_invVR_mats3x3(kpts)
        >>> invVR_mats2, oris = rectify_invV_mats_are_up(invVR_mats)
        >>> kpts2 = flatten_invV_mats_to_kpts(invVR_mats2)
        >>> # Scale down in y a bit
        >>> kpts2.T[1] += 100
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.show_kpts(np.vstack([kpts, kpts2]), ori=1, eig=True,
        >>>              ori_color='green', rect=True)
        >>> # Redraw oriented to show difference
        >>> pt.draw_kpts2(kpts2, color='red', ell_linewidth=2, ori=1,
        >>>               eig=True, ori_color='green', rect=True)
        >>> ax = pt.gca()
        >>> ax.set_aspect('auto')
        >>> pt.dark_background()
        >>> ut.show_if_requested()

        pt.figure(doclf=True, fnum=pt.ensure_fnum(None))
        ax = pt.gca()
        #ax.invert_yaxis()
        #pt.draw_kpts2(kpts, color='blue', ell_linewidth=3, ori=1, eig=True, ori_color='green', rect=True)
        pt.draw_kpts2(kpts2, color='red', ell_linewidth=2, ori=1, eig=True, ori_color='green', rect=True)
        extents = np.array(vt.get_kpts_image_extent(np.vstack([kpts, kpts2])))
        pt.set_axis_extent(extent, ax)
        pt.dark_background()
        ut.show_if_requested()

    Example1:
        >>> from vtool.keypoint import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> invVR_mats = rng.rand(1000, 2, 2).astype(np.float64)
        >>> output = rectify_invV_mats_are_up(invVR_mats)
        >>> print(ut.hash_data(output))
        nbgarvieipbyfihtrhmeouosgehswvcr

    Ignore:
        _invRs_2x2 = invVR_mats[:, 0:2, 0:2][0:1]
        A = _invRs_2x2[0]
        Q, R = np.linalg.qr(A)

        invVR_mats2, oris = rectify_invV_mats_are_up(_invRs_2x2[0:1])
        L2, ori2 = invVR_mats2[0], oris[0]
        Q2 = vt.rotation_mat2x2(ori2)

        np.linalg.det(Q)

        vecs = np.random.rand(2, 4)
        Q2.dot(vecs)
        Q.dot(vecs)

        np.linalg.cholesky(_invR_2x2)


    """
    # Get orientation encoded in the matrix
    _oris = get_invVR_mats_oris(invVR_mats)
    # Extract keypoint shape components
    (_a, _b, _c, _d) = get_invVR_mats_shape(invVR_mats)
    # Convert to lower triangular (rectify orientation downwards)
    # I believe this is an LQ decomposition
    det_ = np.sqrt(np.abs((_a * _d) - (_b * _c)))
    b2a2 = np.sqrt((_b ** 2) + (_a ** 2))
    iv11 = b2a2 / det_
    iv21 = ((_d * _b) + (_c * _a)) / (b2a2 * det_)
    iv22 = det_ / b2a2
    # Rebuild the matrixes
    invV_mats = invVR_mats.copy()
    invV_mats[:, 0, 0] = iv11 * det_
    invV_mats[:, 0, 1] = 0
    invV_mats[:, 1, 0] = iv21 * det_
    invV_mats[:, 1, 1] = iv22 * det_
    return invV_mats, _oris


def flatten_invV_mats_to_kpts(invV_mats):
    """ flattens invV matrices into kpts format """
    invV_mats, _oris = rectify_invV_mats_are_up(invV_mats)
    _xs    = invV_mats[:, 0, 2]
    _ys    = invV_mats[:, 1, 2]
    _iv11s = invV_mats[:, 0, 0]
    _iv21s = invV_mats[:, 1, 0]
    assert np.all(invV_mats[:, 0, 1] == 0), 'expected lower triangular matrix'
    _iv22s = invV_mats[:, 1, 1]
    kpts = np.vstack((_xs, _ys, _iv11s, _iv21s, _iv22s, _oris)).T
    return kpts


def get_V_mats(kpts, **kwargs):
    """
    Returns:
        V_mats (ndarray) : sequence of matrices that transform an ellipse to unit circle
    """
    invV_mats = get_invV_mats(kpts, **kwargs)
    V_mats = invert_invV_mats(invV_mats)
    return V_mats


def get_RV_mats2x2(kpts):
    """
    Returns:
        V_mats (ndarray) : sequence of matrices that transform an ellipse to unit circle
    """
    invVR_mats2x2 = get_invVR_mats2x2(kpts)
    RV_mats2x2 = invert_invV_mats(invVR_mats2x2)
    return RV_mats2x2


def get_Z_mats(V_mats):
    """
    transform into conic matrix Z
    Z = (V.T).dot(V)

    Returns:
        Z_mats (ndarray): Z is a conic representation of an ellipse
    """
    Vt_mats = np.array(list(map(np.transpose, V_mats)))
    Z_mats = matrix_multiply(Vt_mats, V_mats)
    return Z_mats


# def assert_Z_mat(Z_mats2x2):
#     for Z in Z_mats2x2:
#         A, B, _, C = Z.ravel()
#         X, Y = 0, 0
#         theta = np.linspace(0, np.pi * 2)
#         circle_xy = np.vstack([np.cos(theta), np.sin(theta)])
#         invV = invV_mats[0, 0:2, 0:2]
#         x, y = invV.dot(circle_xy)
#         # V = np.linalg.inv(invV)
#         # E = V.T.dot(V)
#         ans = (A * (x - X) ** 2 + 2 * B * (x - X) * (y - Y) + C * (y - Y) ** 2)
#         np.all(np.isclose(ans, 1))


def decompose_Z_to_invV_2x2(Z_2x2):
    import vtool as vt
    import scipy.linalg
    RV_2x2 = scipy.linalg.sqrtm(Z_2x2)
    invVR_2x2 = np.linalg.inv(RV_2x2)
    invV_2x2, ori_ = vt.rectify_invV_mats_are_up(invVR_2x2[None, :, :])
    invV_2x2 = invV_2x2[0]
    return invV_2x2


def decompose_Z_to_V_2x2(Z_2x2):
    invV_2x2 = decompose_Z_to_invV_2x2(Z_2x2)
    V_2x2 = np.linalg.inv(invV_2x2)
    return V_2x2


def decompose_Z_to_invV_mats2x2(Z_mats2x2):
    RV_mats2x2 = decompose_Z_to_RV_mats2x2(Z_mats2x2)
    invVR_mats2x2 = np.linalg.inv(RV_mats2x2)
    invV_2x2, ori_ = rectify_invV_mats_are_up(invVR_mats2x2)
    return invV_2x2


def decompose_Z_to_RV_mats2x2(Z_mats2x2):
    """
    A, B, C = [0.016682, 0.001693, 0.014927]
    #A, B, C = [0.010141, -1.1e-05, 0.02863]
    Z = np.array([[A, B], [B, C]])

    A, B, C = 0.010141, -1.1e-05, 0.02863

    Ignore:
        # Working on figuring relationship between us and VGG
        A, B, _, C = Z_mats2x2[0].ravel()
        X, Y = 0, 0
        theta = np.linspace(0, np.pi * 2)
        circle_xy = np.vstack([np.cos(theta), np.sin(theta)])
        invV = invV_mats[0, 0:2, 0:2]
        x, y = invV.dot(circle_xy)
        V = np.linalg.inv(invV)
        E = V.T.dot(V)
        [[A, B], [_, C]] = E
        [[A_, B_], [_, C_]] = E
        print(A*(x-X) ** 2 + 2*B*(x-X)*(y-Y) + C*(y-Y) ** 2)

    Z_mats2x2 = np.array([
        [[ .016682, .001693],
        [ .001693, .014927]],
        [[ .01662, .001693],
        [ .001693, .014927]],
        [[ .016682, .00193],
        [ .00193, .01492]],
        ])

    import scipy.linalg
    %timeit np.array([scipy.linalg.sqrtm(Z) for Z in Z_mats2x2])
    %timeit decompose_Z_to_VR_mats2x2(Z_mats2x2)
    """
    # explicit 2x2 square root matrix case
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    tr = np.trace(Z_mats2x2, axis1=1, axis2=2)
    det = np.linalg.det(Z_mats2x2)
    s = np.sqrt(det)
    t = np.sqrt(tr + 2 * s)
    a = Z_mats2x2[:, 0, 0]
    b = Z_mats2x2[:, 0, 1]
    # FIXME; Z is symmetric, so c is not really needed
    # should make another function that takes 3 args.
    c = Z_mats2x2[:, 1, 0]
    d = Z_mats2x2[:, 1, 1]
    RV_mats2x2 = np.array([[a + s, b], [c, d + s]]) / t
    RV_mats2x2 = np.rollaxis(RV_mats2x2, 2)
    return RV_mats2x2


def invert_invV_mats(invV_mats):
    r"""
    Args:
        invV_mats (ndarray[float32_t, ndim=3]):  keypoint shapes (possibly translation)

    Returns:
        ndarray[float32_t, ndim=3]: V_mats

    # Ignore:
    #     >>> from vtool.keypoint import *
    #     >>> invV_mats  = np.array([[[ 18.00372824,   1.86434161,  32.        ],
    #     >>>                         [ -0.61356842,  16.02202028,  27.2       ],
    #     >>>                         [  0.        ,   0.        ,   1.        ]],
    #     >>> #
    #     >>>                         [[ 17.41989015,   2.51145917,  61.        ],
    #     >>>                         [ -2.94649591,  24.02540959,  22.9       ],
    #     >>>                         [  0.        ,   0.        ,   1.        ]],
    #     >>> #
    #     >>>                         [[ 20.38098025,   0.88070646,  93.1       ],
    #     >>>                         [ -0.93778675,  24.78261982,  23.6       ],
    #     >>>                         [  0.        ,   0.        ,   1.        ]],
    #     >>> #
    #     >>>                         [[ 16.25114793,  -5.93213207, 120.        ],
    #     >>>                         [  4.71295477,  21.80597527,  29.5       ],
    #     >>>                         [  0.        ,   0.        ,   1.        ]],
    #     >>> #
    #     >>>                         [[ 19.60863253, -11.43641248, 147.        ],
    #     >>>                         [  8.45128003,  10.69925072,  42.        ],
    #     >>>                         [  0.        ,   0.        ,   1.        ]]])
    #     >>> ut.hash_data(invV_mats)
    #     hcnoknyxgeecfyfrygblbvdeezmiulws
    #     >>> V_mats = npl.inv(invV_mats)
    #     >>> ut.hash_data(V_mats)
    #     yooneahjgcifojzpovddeyhtkkyypldd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> invV_mats = vt.get_invVR_mats3x3(kpts)
        >>> V_mats = invert_invV_mats(invV_mats)
        >>> test = vt.matrix_multiply(invV_mats, V_mats)
        >>> # This should give us identity
        >>> assert np.allclose(test, np.eye(3))
    """
    try:
        V_mats = npl.inv(invV_mats)
    except npl.LinAlgError:
        # FIXME: !!!
        # Debug inverse
        V_mats_list = [None for _ in range(len(invV_mats))]
        for ix, invV in enumerate(invV_mats):
            try:
                V_mats_list[ix] = npl.inv(invV)
            except npl.LinAlgError:
                print(ub.hzcat('ERROR: invV_mats[%d] = ' % ix, invV))
                V_mats_list[ix] = np.nan(invV.shape)
        if ut.SUPER_STRICT:
            raise
        V_mats = np.array(V_mats_list)
    return V_mats


def get_kpts_wh(kpts, outer=True):
    r"""
    Gets the width / height diameter of a keypoint
    ie the diameter of the xaxis and yaxis of the keypoint.

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        outer (bool): if True returns wh of bounding box.
           This is useful because extracting a patch needs a rectangle.
           If false it returns the otherwise gets the extent of the ellipse.

    Returns:
        ndarray: (2xN) column1 is X extent and column2 is Y extent

    Ignore:
        # Determine formula for min/maxing x and y
        import sympy
        x, y = sympy.symbols('x, y', real=True)
        a, d = sympy.symbols('a, d', real=True, positive=True)
        c = sympy.symbols('c', real=True)
        theta = sympy.symbols('theta', real=True, nonnegative=True)
        xeqn = sympy.Eq(x, a * sympy.cos(theta))
        yeqn = sympy.Eq(y, c * sympy.sin(theta) + v * d)
        dxdt = sympy.solve(sympy.diff(xeqn, theta), 0)
        dydt = sympy.solve(sympy.diff(yeqn, theta), 0)

        # Ugg, cant get sympy to do trig derivative, do it manually
        dxdt = -a * sin(theta)
        dydt = d * cos(theta) - c * sin(theta)
        critical_thetas = solve(Eq(dxdt, 0), theta)
        critical_thetas += solve(Eq(dydt, 0), theta)
        [a, _, c, d] = invV.ravel()
        critical_thetas = [
            0, np.pi,
            -2 * np.arctan((c + np.sqrt(c ** 2 + d ** 2)) / d),
            -2 * np.arctan((c - np.sqrt(c ** 2 + d ** 2)) / d),
        ]
        critical_uvs = np.vstack([np.cos(critical_thetas),
                                  np.sin(critical_thetas)])
        critical_xys = invV.dot(critical_uvs)


    SeeAlso:
        get_kpts_major_minor

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()[0:5]
        >>> kpts[:, 0] += np.arange(len(kpts)) * 30
        >>> kpts[:, 1] += np.arange(len(kpts)) * 30
        >>> xyexnts = get_kpts_wh(kpts)
        >>> result = ub.repr2(xyexnts)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.cla()
        >>> pt.draw_kpts2(kpts, color='red', ell_linewidth=6, rect=True)
        >>> ax = pt.gca()
        >>> extent = np.array(get_kpts_image_extent(kpts))
        >>> extent = vt.scale_extents(extent, 1.1)
        >>> pt.set_axis_extent(extent, ax)
        >>> xs, ys = vt.get_xys(kpts)
        >>> radii = xyexnts / 2
        >>> horiz_pts1 = np.array([(xs - radii.T[0]), ys]).T
        >>> horiz_pts2 = np.array([(xs + radii.T[0]), ys]).T
        >>> vert_pts1 = np.array([xs, (ys - radii.T[1])]).T
        >>> vert_pts2 = np.array([xs, (ys + radii.T[1])]).T
        >>> pt.draw_line_segments2(horiz_pts1, horiz_pts2, color='g')
        >>> pt.draw_line_segments2(vert_pts1, vert_pts2, color='b')
        >>> ut.show_if_requested()
        np.array([[10.43315411, 58.5216589 ],
                  [ 4.71017647, 58.5216589 ],
                  [24.43314171, 45.09558868],
                  [26.71114159, 63.47679138],
                  [32.10540009, 30.28536987]])
    """
    if outer:
        # Either use bbox or elliptical points
        invV_mats2x2 = get_invVR_mats2x2(kpts)
        corners = np.array([
            [-1, 1, 1, -1],
            [-1, -1, 1, 1],
        ])
        warped_corners = np.array([invV.dot(corners)
                                   for invV in invV_mats2x2])
        maxx = warped_corners[:, 0, :].max(axis=1)
        minx = warped_corners[:, 0, :].min(axis=1)
        maxy = warped_corners[:, 1, :].max(axis=1)
        miny = warped_corners[:, 1, :].min(axis=1)
    else:
        # Find minimum and maximum points on the ellipse
        a = kpts.T[2]
        c = kpts.T[3]
        d = kpts.T[4]
        # x_crit_thetas = np.array([[0, np.pi]])
        # x_crit_u = np.cos(x_crit_thetas)
        # x_crit_v = np.sin(x_crit_thetas)
        x_crit_u = np.array([[1], [-1]])
        x_crit_v = np.array([[0], [0]])
        x_crit_x = a * x_crit_u
        x_crit_y = c * x_crit_u + d * x_crit_v

        part = np.sqrt(c ** 2 + d ** 2)
        y_crit_thetas1 = -2 * np.arctan((c + part) / d)
        y_crit_thetas2 = -2 * np.arctan((c - part) / d)
        y_crit_thetas = np.vstack(
            (y_crit_thetas1, y_crit_thetas2))
        y_crit_u = np.cos(y_crit_thetas)
        y_crit_v = np.sin(y_crit_thetas)
        y_crit_x = a * y_crit_u
        y_crit_y = c * y_crit_u + d * y_crit_v

        crit_x = np.vstack([y_crit_x, x_crit_x])
        crit_y = np.vstack([y_crit_y, x_crit_y])
        maxx = crit_x.max(axis=0)
        minx = crit_x.min(axis=0)
        maxy = crit_y.max(axis=0)
        miny = crit_y.min(axis=0)

    w = maxx - minx
    h = maxy - miny
    wh_list = np.vstack([w, h]).T
    return wh_list


def get_kpts_image_extent(kpts, outer=False, only_xy=False):
    """
    returns the width and height of keypoint bounding box
    This combines xy and shape information
    Does not take into account if keypoint extent goes under (0, 0)

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        outer: uses outer rectangle if True. Set to false for a
            tighter extent.

    Returns:
        tuple: (minx, maxx, miny, maxy)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> extent = get_kpts_image_extent(kpts, outer=False)
        >>> result = ub.repr2(np.array(extent), precision=2)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_kpts2(kpts, bbox=True)
        >>> ax = pt.gca()
        >>> pt.set_axis_extent(extent, ax)
        >>> ut.show_if_requested()
        np.array([ 14.78, 48.05,  0.32, 51.58])
    """
    if len(kpts) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    xs, ys = get_xys(kpts)
    if only_xy:
        minx = xs.min()
        maxx = xs.max()
        miny = ys.min()
        maxy = ys.max()
    else:
        wh_list = get_kpts_wh(kpts, outer=outer)
        radii = np.divide(wh_list, 2, out=wh_list)
        minx = (xs - radii.T[0]).min()
        maxx = (xs + radii.T[0]).max()
        miny = (ys - radii.T[1]).min()
        maxy = (ys + radii.T[1]).max()
    extent = (minx, maxx, miny, maxy)
    return extent


def get_kpts_dlen_sqrd(kpts, outer=False):
    r"""
    returns diagonal length squared of keypoint extent

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        outer (bool): loose if False tight if True

    Returns:
        float: dlen_sqrd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> dlen_sqrd = get_kpts_dlen_sqrd(kpts)
        >>> result = '%.2f' % dlen_sqrd
        >>> print(result)
        3735.01
    """
    if len(kpts) == 0:
        return 0.0
    extent = get_kpts_image_extent(kpts, outer=outer)
    x1, x2, y1, y2 = extent
    w = x2 - x1
    h = y2 - y1
    dlen_sqrd = (w ** 2) + (h ** 2)
    return dlen_sqrd


def cast_split(kpts, dtype=KPTS_DTYPE):
    """ breakup keypoints into location, shape, and orientation """
    kptsT = kpts.T
    _xs   = np.array(kptsT[0], dtype=dtype)
    _ys   = np.array(kptsT[1], dtype=dtype)
    _invVs = np.array(kptsT[2:5], dtype=dtype)
    if kpts.shape[1] == 6:
        _oris = np.array(kptsT[5:6], dtype=dtype)
    else:
        _oris = np.zeros(len(kpts))
    return _xs, _ys, _invVs, _oris


# --- strings ---

def get_xy_strs(kpts):
    """ strings debugging and output """
    _xs, _ys   = get_xys(kpts)
    xy_strs = [('xy=(%.1f, %.1f)' % (x, y,)) for x, y, in zip(_xs, _ys)]
    return xy_strs


def get_shape_strs(kpts):
    """ strings debugging and output """
    invVs = get_invVs(kpts)
    shape_strs  = [(('[(%3.1f, 0.00),\n' +
                     ' (%3.1f, %3.1f)]') % (iv11, iv21, iv22,))
                   for iv11, iv21, iv22 in zip(*invVs)]
    shape_strs = ['invV=\n' +  _str for _str in shape_strs]
    return shape_strs


def get_ori_strs(kpts):
    _oris = get_oris(kpts)
    ori_strs = ['ori=' + ut.theta_str(ori) for ori in _oris]
    return ori_strs


def get_kpts_strs(kpts):
    xy_strs = get_xy_strs(kpts)
    shape_strs = get_shape_strs(kpts)
    ori_strs = get_ori_strs(kpts)
    kpts_strs = ['\n---\n'.join(tup) for tup in zip(xy_strs, shape_strs, ori_strs)]
    return kpts_strs


def kpts_repr(arr, precision=2, suppress_small=True, linebreak=False):
    # TODO replace with ub.repr2
    repr_kw = dict(precision=precision, suppress_small=suppress_small)
    reprstr = np.array_repr(arr, **repr_kw)
    if not linebreak:
        reprstr = reprstr.replace('\n\n', '\n')
    return reprstr


def kp_cpp_infostr(kp):
    """ mirrors c++ debug code """
    x, y = kp[0:2]
    a11, a21, a22 = kp[2:5]
    a12 = 0.0
    ori = kp[5]
    s = np.sqrt(a11 * a22)
    a11 /= s
    a12 /= s
    a21 /= s
    a22 /= s
    infostr_list = [
        ('+---'),
        ('|     xy = (%s, %s)' % (x, y)),
        ('| hat{invV} = [(%s, %s),' % (a11, a12,)),
        ('|              (%s, %s)]' % (a21, a22,)),
        ('|    sc  = %s' % (s,)),
        ('|    ori = %s' % (ori,)),
        ('L___'),
    ]
    return '\n'.join(infostr_list)


def kpts_docrepr(arr, name='arr', indent=True, *args, **kwargs):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> arr = np.random.rand(3, 3)
        >>> args = tuple()
        >>> kwargs = dict()
        >>> result = kpts_docrepr(arr)
        >>> # verify results
        >>> print(result)
    """
    reprstr_ = kpts_repr(arr, *args, **kwargs)
    eq = ' = '
    if len(name) == 0:
        eq = ''
    prefix = name + eq + 'np.'
    docrepr_ = ut.indent(prefix + reprstr_, ' ' * len(prefix))[len(prefix):]
    if indent:
        docrepr = ut.indent('>>> ' + ut.indent(docrepr_, '... ')[4:], ' ' * 8)
    else:
        docrepr = docrepr_
    return docrepr


def get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1):
    """ transforms img2 to img2 and finds squared spatial error

    Args:
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix mapping image 1 to image 2 space
        fx2_to_fx1 (ndarray): has shape (nMatch, K)

    Returns:
        ndarray: fx2_to_xyerr_sqrd has shape (nMatch, K)

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts1 = np.array([[ 129.83,  46.97,  15.84,   4.66,   7.24,   0.  ],
        ...                   [ 137.88,  49.87,  20.09,   5.76,   6.2 ,   0.  ],
        ...                   [ 115.95,  53.13,  12.96,   1.73,   8.77,   0.  ],
        ...                   [ 324.88, 172.58, 127.69,  41.29,  50.5 ,   0.  ],
        ...                   [ 285.44, 254.61, 136.06,  -4.77,  76.69,   0.  ],
        ...                   [ 367.72, 140.81, 172.13,  12.99,  96.15,   0.  ]], dtype=np.float64)
        >>> kpts2 = np.array([[ 318.93,  11.98,  12.11,   0.38,   8.04,   0.  ],
        ...                   [ 509.47,  12.53,  22.4 ,   1.31,   5.04,   0.  ],
        ...                   [ 514.03,  13.04,  19.25,   1.74,   4.72,   0.  ],
        ...                   [ 490.19, 185.49,  95.67,  -4.84,  88.23,   0.  ],
        ...                   [ 316.97, 206.07,  90.87,   0.07,  80.45,   0.  ],
        ...                   [ 366.07, 140.05, 161.27, -47.01,  85.62,   0.  ]], dtype=np.float64)
        >>> H = np.array([[ -0.70098,  0.12273,  5.18734],
        >>>               [ 0.12444, -0.63474, 14.13995],
        >>>               [ 0.00004,  0.00025, -0.64873]])
        >>> fx2_to_fx1 = np.array([[5, 4, 1, 0],
        >>>                        [0, 1, 5, 4],
        >>>                        [0, 1, 5, 4],
        >>>                        [2, 3, 1, 5],
        >>>                        [5, 1, 0, 4],
        >>>                        [3, 1, 5, 0]], dtype=np.int32)
        >>> fx2_to_xyerr_sqrd = get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
        >>> fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
        >>> # verify results
        >>> result = ub.repr2(fx2_to_xyerr, precision=3)
        >>> print(result)
        np.array([[ 82.848, 186.238, 183.979, 192.639],
                  [382.988, 374.356, 122.179, 289.16 ],
                  [387.563, 378.93 , 126.389, 292.391],
                  [419.246, 176.668, 400.175, 167.411],
                  [174.269, 274.289, 281.03 ,  33.521],
                  [ 54.083, 269.645,  94.711, 277.706]])


    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> kpts1 = np.array([[ 6.,  4.,  15.84,   4.66,   7.24,   0.  ],
        ...                   [ 9.,  3.,  20.09,   5.76,   6.2 ,   0.  ],
        ...                   [ 1.,  1.,  12.96,   1.73,   8.77,   0.  ],])
        >>> kpts2 = np.array([[ 2.,  1.,  12.11,   0.38,   8.04,   0.  ],
        ...                   [ 5.,  1.,  22.4 ,   1.31,   5.04,   0.  ],
        ...                   [ 6.,  1.,  19.25,   1.74,   4.72,   0.  ],])
        >>> H = np.array([[ 2, 0, 0],
        >>>               [ 0, 1, 0],
        >>>               [ 0, 0, 1]])
        >>> fx2_to_fx1 = np.array([[2, 1, 0],
        >>>                        [0, 1, 2],
        >>>                        [2, 1, 0]], dtype=np.int32)
        >>> fx2_to_xyerr_sqrd = get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
        >>> fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
        >>> # verify results
        >>> result = ub.repr2(fx2_to_xyerr, precision=3)
        >>> print(result)
        np.array([[ 0.   , 16.125, 10.44 ],
                  [ 7.616, 13.153,  3.   ],
                  [ 4.   , 12.166,  6.708]])
    """
    DEBUG = True
    if DEBUG:
        try:
            assert kpts2.shape[0] == fx2_to_fx1.shape[0]
            assert kpts1.shape[0] >= fx2_to_fx1.max()
        except AssertionError as ex:
            ut.printex(ex, 'bad shape', keys=[
                'kpts2.shape',
                'kpts1.shape',
                'fx2_to_fx1.shape',
                'fx2_to_fx1.max()']
            )
            raise
    # Transform img1 xy-keypoints into img2 space
    xy1_t = transform_kpts_xys(H, kpts1)
    # Get untransformed image 2 xy-keypoints
    xy2   = get_xys(kpts2)
    # get spatial keypoint distance to all neighbor candidates
    bcast_xy2   = xy2[:, None, :].T
    bcast_xy1_t = xy1_t.T[fx2_to_fx1]
    fx2_to_xyerr_sqrd = distance.L2_sqrd(bcast_xy2, bcast_xy1_t)
    return fx2_to_xyerr_sqrd


def get_uneven_point_sample(kpts):
    """
    for each keypoint returns an uneven sample of points along the ellipical
      boundries.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    SeeAlso:
        pyhesaff.tests.test_ellipse
        python -m pyhesaff.tests.test_ellipse --test-in_depth_ellipse --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()[0:2]
        >>> ellipse_pts1 = get_uneven_point_sample(kpts)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_line_segments(ellipse_pts1)
        >>> pt.set_title('uneven sample points')
        >>> pt.show_if_requested()
    """
    # Define points on a unit circle
    nSamples = 32
    invV_mats = get_invVR_mats3x3(kpts)
    theta_list = np.linspace(0, TAU, nSamples)
    circle_pts = np.array([(np.cos(t_), np.sin(t_), 1) for t_ in theta_list])
    # Transform those points to the ellipse using invV
    ellipse_pts1 = matrix_multiply(invV_mats, circle_pts.T).transpose(0, 2, 1)
    return ellipse_pts1


def get_even_point_sample(kpts):
    """
    gets even points sample along the boundary of the ellipse

    SeeAlso:
        pyhesaff.tests.test_ellipse

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()[0:2]
        >>> ell_border_pts_list = get_even_point_sample(kpts)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_line_segments(ell_border_pts_list)
        >>> pt.set_title('even sample points')
        >>> pt.show_if_requested()
    """
    # BROKEN
    from vtool import ellipse
    nSamples = 32
    ell_border_pts_list = ellipse.sample_uniform(kpts, nSamples)
    return ell_border_pts_list


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.keypoint
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
