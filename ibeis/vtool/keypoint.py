# -*- coding: utf-8 -*-
"""
python -c "import vtool, doctest; print(doctest.testmod(vtool.keypoint))"

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
    # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
    from vtool.patch import *  # NOQA
    import sympy
    from sympy.abc import theta
    ori = theta
    x, y, iv11, iv21, iv22, patch_size = sympy.symbols('x y iv11 iv21 iv22 S')
    kpts = np.array([[x, y, iv11, iv21, iv22, ori]])
    kp = ktool.get_invV_mats(kpts, with_trans=True)[0]
    invV = sympy.Matrix(kp)
    V = invV.inv()

    print(ut.hz_str('invV = ', repr(invV)))
    invV = Matrix([
           [iv11,  0.0,   x],
           [iv21, iv22,   y],
           [ 0.0,  0.0, 1.0]])

    invV2 = sympy.Matrix([
           [iv11,  0.0,   0],
           [iv21, iv22,   0],
           [ 0.0,  0.0, 1.0]])

    trans = sympy.Matrix([
           [   1,  0.0,   x],
           [   0,    1,   y],
           [ 0.0,  0.0, 1.0]])


    print(ut.hz_str('V = ', repr(V)))
    V = Matrix([
        [           1/iv11,      0,                 -1.0*x/iv11],
        [-iv21/(iv11*iv22), 1/iv22, -1.0*(y - iv21*x/iv11)/iv22],
        [                0,      0,                         1.0]])



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
from vtool import linalg as ltool
from vtool import chip as ctool
from vtool import distance as dtool
from vtool import trig
import utool as ut
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[kpts]')


"""
// These are cython style comments used for maintaining python compatibility
#if CYTH
from vtool.keypoint import get_invVR_mats_shape, get_invVR_mats_sqrd_scale, get_invVR_mats_oris

cdef np.float64_t TAU = 2 * np.pi
#endif
"""
#:%s/numpy_floatarray_\([13]\)dimension/np.ndarray[np.float64_t, ndim=\1]/gc
#:%s/np.ndarray\[np.float64_t, ndim=\([13]\)\]/numpy_floatarray_\1dimension/gc


TAU = 2 * np.pi  # References: tauday.com
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


def get_grid_kpts(wh=(300, 300), wh_stride=(50, 50), scale=20, dtype=np.float32, **kwargs):
    """ Returns a regular grid of keypoints """
    (w, h) = wh
    (wstride, hstride) = wh_stride
    padding = scale * 1.5
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
    """ gets average squared scale (does not take into account elliptical shape """
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _scales_sqrd = np.multiply(_iv11s, _iv22s)
    return _scales_sqrd


def get_scales(kpts):
    """  Gets average scale (does not take into account elliptical shape """
    _scales = np.sqrt(get_sqrd_scales(kpts))
    return _scales


# --- keypoint matrixes ---

def get_ori_mats(kpts):
    """ Returns keypoint orientation matrixes """
    _oris = get_oris(kpts)
    R_mats = [ltool.rotation_mat2x2(ori) for ori in _oris]
    return R_mats


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
        >>> # build test data
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> # execute function
        >>> invV_mats2x2 = get_invV_mats2x2(kpts)
        >>> # verify results
        >>> result = kpts_repr(invV_mats2x2)
        >>> print(result)
        array([[[ 1.,  0.],
                [ 2.,  3.]],
               [[ 1.,  0.],
                [ 2.,  3.]]])
    """
    nKpts = len(kpts)
    #try:
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _zeros = np.zeros(nKpts)
    invV_arrs2x2 = np.array([[_iv11s, _zeros],
                             [_iv21s, _iv22s]])  # R x C x N
    invV_mats2x2 = np.rollaxis(invV_arrs2x2, 2)  # N x R x C
    #except ValueError as ex:
    #    ut.printex(ex, keys=['kpts', '_zeros', '_iv11s', '_iv21s', '_iv22s'])
    #    #ut.embed()
    #    #print(kpts)
    #    raise
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
        >>> # build test data
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> # execute function
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
    with ut.EmbedOnException():
        invVR_mats2x2 = matrix_multiply(invV_mats2x2, R_mats2x2)
    return invVR_mats2x2


def augment_2x2_with_translation(kpts, _mat2x2):
    """ helper function to augment shape matrix with a translation component """
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
                          [_zeros, _zeros,  _ones]])  # R x C x N
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

    CommandLine:
        python -m vtool.keypoint --test-get_invV_mats3x3

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> # build test data
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> # execute function
        >>> invV_arrs3x3 = get_invV_mats3x3(kpts)
        >>> # verify results
        >>> result = kpts_repr(invV_arrs3x3)
        >>> print(result)
        array([[[ 1.,  0.,  0.],
                [ 2.,  3.,  0.],
                [ 0.,  0.,  1.]],
               [[ 1.,  0.,  0.],
                [ 2.,  3.,  0.],
                [ 0.,  0.,  1.]]])
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
    #                          [_zeros, _zeros,  _ones]])  # R x C x N
    #invV_mats3x3 = np.rollaxis(invV_arrs3x3, 2)  # N x R x C
    return invV_mats3x3


#@profile
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

    CommandLine:
        python -m vtool.keypoint --test-get_invVR_mats3x3

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> # build test data
        >>> kpts = np.array([
        ...    [10, 20, 1, 2, 3, 0],
        ...    [30, 40, 1, 2, 3, TAU / 4.0],
        ... ])
        >>> # execute function
        >>> invVR_mats3x3 = get_invVR_mats3x3(kpts)
        >>> # verify results
        >>> result = kpts_repr(invVR_mats3x3)
        >>> print(result)
        array([[[  1.,   0.,  10.],
                [  2.,   3.,  20.],
                [  0.,   0.,   1.]],
               [[  0.,  -1.,  30.],
                [  3.,  -2.,  40.],
                [  0.,   0.,   1.]]])
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
    #                        [_zeros, _zeros,  _ones]])  # R x C x N
    #invVR_mats = np.rollaxis(invVR_arrs, 2)  # N x R x C
    return invVR_mats3x3


#@profile
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
        array([[[  1.,   0.,  10.],
                [  2.,   3.,  20.],
                [  0.,   0.,   1.]]])
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
                               [_zeros, _zeros,  _ones]])  # R x C x N
        invV_mats = np.rollaxis(invV_arrs, 2)  # N x R x C
    if ascontiguous:
        invV_mats = np.ascontiguousarray(invV_mats)
    return invV_mats

# --- scaled and offset keypoint components ---


def get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor=1.0):
    """
    Given some patch (like a gaussian patch) transforms a patch to be overlayed
    on top of each keypoint in the image (adjusted for a scale factor)

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch_shape (?):
        scale_factor (float):

    Returns:
        M_list: a list of 3x3 tranformation matricies for each keypoint

    CommandLine:
        python -m vtool.keypoint --test-get_transforms_from_patch_image_kpts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> patch_shape = (7, 7)
        >>> scale_factor = 1.0
        >>> # execute function
        >>> M_list = get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)
        >>> # verify results
        >>> result = kpts_repr(M_list)
        >>> print(result)
        array([[[  1.49,   0.  ,  15.53],
                [ -1.46,   6.9 ,   8.68],
                [  0.  ,   0.  ,   1.  ]],
               [[  0.67,   0.  ,  26.98],
                [ -1.46,   6.9 ,   8.68],
                [  0.  ,   0.  ,   1.  ]],
               [[  3.49,   0.  ,  19.53],
                [  3.43,   3.01,  10.67],
                [  0.  ,   0.  ,   1.  ]],
               [[  3.82,   0.  ,  19.55],
                [  5.04,   4.03,   1.8 ],
                [  0.  ,   0.  ,   1.  ]],
               [[  4.59,   0.  ,  18.24],
                [  0.97,   3.35,  18.02],
                [  0.  ,   0.  ,   1.  ]]])

    Ignore:
        >>> from vtool.coverage_kpts import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.dummy.get_dummy_kpts()
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
    T1 = ltool.translation_mat3x3(-half_width + .5, -half_height + .5)
    # Scale src to the unit circle
    #S1 = ltool.scale_mat3x3(1.0 / patch_w, 1.0 / patch_h)
    S1 = ltool.scale_mat3x3(1.0 / half_width, 1.0 / half_height)
    # Transform the source image to the keypoint ellipse
    invVR_aff2Ds = get_invVR_mats3x3(kpts)
    # Adjust for the requested scale factor
    S2 = ltool.scale_mat3x3(scale_factor, scale_factor)
    #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
    M_list = reduce(matrix_multiply, (S2, invVR_aff2Ds, S1.dot(T1)))
    return M_list


#@profile
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
    invC = ctool._get_chip_to_image_transform(bbox, chipsz, bbox_theta)
    # Apply transform to keypoints
    invCinvV_mats = matrix_multiply(invC, invV_mats)
    # Flatten back into keypoint (x, y, a, c, d, o) format
    imgkpts = flatten_invV_mats_to_kpts(invCinvV_mats)
    return imgkpts


#@profile
def offset_kpts(kpts, offset=(0.0, 0.0), scale_factor=1.0):
    r"""
    Transfoms keypoints by a scale factor and a translation

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        offset (tuple):
        scale_factor (float):

    Returns:
        ndarray[float32_t, ndim=2]: kpts -  keypoints

    CommandLine:
        python -m vtool.keypoint --test-offset_kpts
        python -m vtool.keypoint --test-offset_kpts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> offset = (0.0, 0.0)
        >>> scale_factor = (1.5, 0.5)
        >>> # execute function
        >>> kpts_ = offset_kpts(kpts, offset, scale_factor).astype(np.float32)
        >>> # verify results
        >>> result = ut.list_str((kpts, kpts_), label_list=['orig', 'new'], precision=2)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_kpts2(kpts, color=pt.ORANGE, ell_linewidth=6)
        >>> pt.draw_kpts2(kpts_, color=pt.LIGHT_BLUE, ell_linewidth=4)
        >>> wh1 = np.array(vt.get_kpts_image_extent(kpts))
        >>> wh2 = np.array(vt.get_kpts_image_extent(kpts_))
        >>> wh = np.maximum(wh1, wh2)
        >>> ax = pt.gca()
        >>> ax.set_xlim(0, wh[0])
        >>> ax.set_ylim(0, wh[1])
        >>> pt.dark_background()
        >>> ut.show_if_requested()
        orig = np.array([[ 20.  ,  25.  ,   5.22,  -5.11,  24.15,   0.  ],
                         [ 29.  ,  25.  ,   2.36,  -5.11,  24.15,   0.  ],
                         [ 30.  ,  30.  ,  12.22,  12.02,  10.53,   0.  ],
                         [ 31.  ,  29.  ,  13.36,  17.63,  14.1 ,   0.  ],
                         [ 32.  ,  31.  ,  16.05,   3.41,  11.74,   0.  ]], dtype=np.float32)
        new = np.array([[ 30.  ,  12.5 ,   7.82,  -2.56,  12.07,  -0.  ],
                        [ 43.5 ,  12.5 ,   3.53,  -2.56,  12.07,  -0.  ],
                        [ 45.  ,  15.  ,  18.32,   6.01,   5.26,  -0.  ],
                        [ 46.5 ,  14.5 ,  20.03,   8.82,   7.05,  -0.  ],
                        [ 48.  ,  15.5 ,  24.08,   1.7 ,   5.87,  -0.  ]], dtype=np.float32)
    """
    if np.all(offset == (0.0, 0.0)) and (np.all(scale_factor == 1.0) or np.all(scale_factor == (1.0, 1.0))):
        return kpts
    try:
        sfx, sfy = scale_factor
    except TypeError:
        sfx = sfy = scale_factor
    #with ut.embed_on_exception_context:
    tx, ty = offset
    T = ltool.translation_mat3x3(tx, ty)
    S = ltool.scale_mat3x3(sfx, sfy)
    M = T.dot(S)
    #M = ltool.scaleedoffset_mat3x3(offset, scale_factor)
    kpts_ = transform_kpts(kpts, M)
    return kpts_


#@profile
def transform_kpts(kpts, M):
    r"""
    returns M.dot(kpts_mat)

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        M (ndarray): transform matrix

    Returns:
        tuple: (_iv11s, _iv12s, _iv21s, _iv22s)

    CommandLine:
        python -m vtool.keypoint --test-transform_kpts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> M = np.array([[10, 0, 0], [10, 10, 0], [0, 0, 1]], dtype=np.float64)
        >>> # execute function
        >>> kpts = transform_kpts(kpts, M)
        >>> # verify results
        >>> result = ut.numpy_str(kpts, precision=3).replace('-0. ', ' 0. ')
        >>> print(result)
        np.array([[ 200.   ,  450.   ,   52.166,    1.056,  241.499,    0.   ],
                  [ 290.   ,  540.   ,   23.551,  -27.559,  241.499,    0.   ],
                  [ 300.   ,  600.   ,  122.166,  242.357,  105.287,    0.   ],
                  [ 310.   ,  600.   ,  133.556,  309.899,  141.041,    0.   ],
                  [ 320.   ,  630.   ,  160.527,  194.6  ,  117.354,    0.   ]], dtype=np.float64)

    IGNORE:
        >>> # HOW DO WE KEEP SHAPE AFTER HOMOGRAPHY?
        >>> # DISABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> M = np.array([[  3.,   3.,   5.],
        ...               [  2.,   3.,   6.],
        ...               [  1.,   1.,   2.]])
        >>> invVR_mats3x3 = get_invVR_mats3x3(kpts)
        >>> MinvVR_mats3x3 = matrix_multiply(M, invVR_mats3x3)
        >>> MinvVR_mats3x3 = np.divide(MinvVR_mats3x3, MinvVR_mats3x3[:, None, None, 2, 2])  # 2.6 us
        >>> MinvVR = MinvVR_mats3x3[0]
        >>> result = kpts_repr(MinvVR)
        >>> print(result)

        # Inspect matrix decompositions
        import numpy.linalg as npl
        print(ut.hz_str('MinvVR = ', kpts_repr(MinvVR)))
        U, s, Vt = npl.svd(MinvVR)
        S = np.diagflat(s)
        print(ut.hz_str('SVD: U * S * Vt = ', U, ' * ', S, ' * ', Vt, precision=3))
        Q, R = npl.qr(MinvVR)
        print(ut.hz_str('QR: Q * R = ', Q, ' * ', R, precision=3))
        #print('cholesky = %r' % (npl.cholesky(MinvVR),))
        #npl.cholesky(MinvVR)


        print(ut.hz_str('MinvVR = ', kpts_repr(MinvVR)))
        MinvVR_ = MinvVR / MinvVR[None, 2, :]
        print(ut.hz_str('MinvVR_ = ', kpts_repr(MinvVR_)))

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
        #Lmats = [ltool.rotation_mat3x3(-ori) for ori in oris]
        #matrix_multiply(MinvVR_mats3x3, Lmats)
        #matrix_multiply(Lmats, MinvVR_mats3x3)
        #scipy.linalg.lu(MinvVR_mats3x3[0])
        #scipy.linalg.qr(MinvVR_mats3x3[0])
        #ut.embed()
        import warnings
        warnings.warn('WARNING: [vtool.keypoint] transform produced non-affine keypoint')
        #ut.printex(ex, )
        #raise
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

    CommandLine:
        python -m vtool.keypoint --test-transform_kpts_xys

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> H = np.array([[  3.,   3.,   5.],
        ...               [  2.,   3.,   6.],
        ...               [  1.,   1.,   2.]])
        >>> # execute function
        >>> xy_t = transform_kpts_xys(H, kpts)
        >>> # verify results
        >>> result = ut.numpy_str(xy_t, precision=3)
        >>> print(result)
        np.array([[ 2.979,  2.982,  2.984,  2.984,  2.985],
                  [ 2.574,  2.482,  2.516,  2.5  ,  2.508]], dtype=np.float64)

    Ignore::
        %pylab qt4
        import plottool as pt
        pt.imshow(chip)
        pt.draw_kpts2(kpts)
        pt.update()
    """
    xy = get_xys(kpts)
    xy_t = ltool.transform_points_with_homography(H, xy)
    return xy_t
    #xyz   = get_homog_xyzs(kpts)
    #xyz_t = matrix_multiply(H, xyz)
    #xy_t  = ltool.add_homogenous_coordinate(xyz_t)
    #return xy_t

#---------------------
# invV_mats functions
#---------------------


#@profile
def get_invVR_mats_sqrd_scale(invVR_mats):
    """ Returns the squared scale of the invVR keyponts

    CommandLine:
        python -m vtool.keypoint --test-get_invVR_mats_sqrd_scale

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(7, 3, 3).astype(np.float64)
        >>> det_arr = get_invVR_mats_sqrd_scale(invVR_mats)
        >>> result = ut.numpy_str(det_arr, precision=2)
        >>> print(result)
        np.array([-0.16, -0.09, -0.34,  0.59, -0.2 ,  0.18,  0.06], dtype=np.float64)

    #CYTH_INLINE
    #CYTH_RETURNS np.ndarray[np.float64_t, ndim=1]
    #CYTH_PARAM_TYPES:
        np.ndarray[np.float64_t, ndim=3] invVR_mats
    #if CYTH
    cdef unsigned int nMats = invVR_mats.shape[0]
    # Prealloc output
    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros((nMats,), dtype=np.float64)
    #cdef size_t ix
    cdef Py_ssize_t ix
    for ix in range(nMats):
        # simple determinant: ad - bc
        out[ix] = ((invVR_mats[ix, 0, 0] * invVR_mats[ix, 1, 1]) -
                   (invVR_mats[ix, 0, 1] * invVR_mats[ix, 1, 0]))
    return out
    #else
    """
    det_arr = npl.det(invVR_mats[:, 0:2, 0:2])
    return det_arr
    "#endif"


#@profile
def get_invVR_mats_shape(invVR_mats):
    """ Extracts keypoint shape components

    CommandLine:
        python -m vtool.keypoint --test-get_invVR_mats_shape

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
        >>> output = get_invVR_mats_shape(invVR_mats)
        >>> result = ut.hashstr(output)
        >>> print(result)
        oq9o@yqhtgloy58!

    References:
        TODO
        (a.ravel()[(cols + (rows * a.shape[1]).reshape((-1,1))).ravel()]).reshape(rows.size, cols.size)
        http://stackoverflow.com/questions/14386822/fast-numpy-fancy-indexing
        # So, this doesn't work
        # Try this instead
        http://docs.cython.org/src/userguide/memoryviews.html#memoryviews

    Cyth::
        #CYTH_INLINE
        #if CYTH
        #CYTH_PARAM_TYPES:
            np.ndarray[np.float64_t, ndim=3] invVR_mats
        cdef:
            np.ndarray[np.float64_t, ndim=1] _iv11s
            np.ndarray[np.float64_t, ndim=1] _iv12s
            np.ndarray[np.float64_t, ndim=1] _iv21s
            np.ndarray[np.float64_t, ndim=1] _iv22s
            #double [:] _iv11s
            #double [:] _iv12s
            #double [:] _iv21s
            #double [:] _iv22s
        #endif
    """
    pass
    ###
    '''
    #if cyth
    #m_acro numpy_fancy_index_macro
    #e_ndmacro
    _iv11s = invVR_mats.take(:, axis=1)
    _iv12s = invVR_mats[:, 0, 1]
    _iv21s = invVR_mats[:, 1, 0]
    _iv22s = invVR_mats[:, 1, 1]
    #else
    #cols, rows, dims = invVR_mats.shape
    #invVR_mats.ravel()[(cols + (rows * a.shape[1]).reshape((-1, 1))).ravel()])
    '''
    _iv11s = invVR_mats[:, 0, 0]
    _iv12s = invVR_mats[:, 0, 1]
    _iv21s = invVR_mats[:, 1, 0]
    _iv22s = invVR_mats[:, 1, 1]
    '#endif'
    #'#pragma cyth numpy_fancy_index_assign'
    return (_iv11s, _iv12s, _iv21s, _iv22s)


#@profile
def get_invVR_mats_xys(invVR_mats):
    r"""
    extracts locations
    extracts xys from matrix encoding

    Args:
        invVR_mats (ndarray) : list of matrices mapping ucircles to ellipses

    Returns:
        ndarray: the xy location

    Cyth:
        #if CYTH
        #CYTH_PARAM_TYPES:
            np.ndarray[np.float64_t, ndim=3] invVR_mats
        cdef:
            np.ndarray[np.float64_t, ndim=2] _xys
        #endif

    Timeit:
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
        #>>> ut.util_dev.rrr()

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


#@profile
def get_invVR_mats_oris(invVR_mats):
    r""" extracts orientation from matrix encoding

    CommandLine:
        python -m vtool.keypoint --test-get_invVR_mats_oris

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> np.random.seed(0)
        >>> invVR_mats = np.random.rand(7, 2, 2).astype(np.float64)
        >>> output = get_invVR_mats_oris(invVR_mats)
        >>> result = ut.numpy_str(output, precision=2)
        np.array([ 5.37,  5.29,  5.9 ,  5.26,  4.74,  5.6 ,  4.9 ], dtype=np.float64)

    Cyth:
        #CYTH_INLINE
        #CYTH_RETURNS np.ndarray[np.float64_t, ndim=1]
        #CYTH_PARAMS:
            np.ndarray[np.float64_t, ndim=3] invVR_mats
        #if CYTH
        cdef:
            np.ndarray[np.float64_t, ndim=1] _oris
            np.ndarray[np.float64_t, ndim=1] _iv12s
            np.ndarray[np.float64_t, ndim=1] _iv11s

        _iv11s = invVR_mats[:, 0, 0]
        _iv12s = invVR_mats[:, 0, 1]
        _oris = np.arctan2(_iv12s, _iv11s)  # outputs from -TAU/2 to TAU/2
        _oris[_oris < 0] = _oris[_oris < 0] + TAU  # map to 0 to TAU (keep coords)
        _oris = (-_oris) % TAU
        return _oris
        #else

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


#@profile
def rectify_invV_mats_are_up(invVR_mats):
    """
    Useful if invVR_mats is no longer lower triangular
    rotates affine shape matrixes into downward (lower triangular) position

    >>> from vtool.keypoint import *  # NOQA
    >>> np.random.seed(0)
    >>> invVR_mats = np.random.rand(1000, 2, 2).astype(np.float64)
    >>> output = rectify_invV_mats_are_up(invVR_mats)
    >>> print(ut.hashstr(output))
    2wir&6ybcga0bpvz

    #if CYTH
    # TODO: Template this for [float64_t, float32_t]
    #CYTH_PARAM_TYPES:
        np.ndarray[np.float64_t, ndim=3] invVR_mats
    cdef:
        np.ndarray[np.float64_t, ndim=3] invV_mats
        np.ndarray[np.float64_t, ndim=1] _oris
        np.ndarray[np.float64_t, ndim=1] _a
        np.ndarray[np.float64_t, ndim=1] _b
        np.ndarray[np.float64_t, ndim=1] _c
        np.ndarray[np.float64_t, ndim=1] _d
        np.ndarray[np.float64_t, ndim=1] det_
        np.ndarray[np.float64_t, ndim=1] b2a2
        np.ndarray[np.float64_t, ndim=1] iv11
        np.ndarray[np.float64_t, ndim=1] iv21
        np.ndarray[np.float64_t, ndim=1] iv22
    #endif
    """
    # Get orientation encoded in the matrix
    #_oris = get_invVR_mats_oris_cyth(invVR_mats)
    _oris = get_invVR_mats_oris(invVR_mats)
    # Extract keypoint shape components
    #(_a, _b, _c, _d) = get_invVR_mats_shape_cyth(invVR_mats)
    (_a, _b, _c, _d) = get_invVR_mats_shape(invVR_mats)
    #
    # Convert to lower triangular (rectify orientation downwards)
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


#@profile
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


#@profile
def get_V_mats(kpts, **kwargs):
    """
    Returns:
        V_mats (ndarray) : sequence of matrices that transform an ellipse to unit circle
    """
    invV_mats = get_invV_mats(kpts, **kwargs)
    V_mats = invert_invV_mats(invV_mats)
    return V_mats


#@profile
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


#@profile
def invert_invV_mats(invV_mats):
    r"""
    Args:
        invV_mats (ndarray[float32_t, ndim=3]):  keypoint shapes (possibly translation)

    Returns:
        ndarray[float32_t, ndim=3]: V_mats

    CommandLine:
        python -m vtool.keypoint --test-invert_invV_mats

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> invV_mats = vt.get_invVR_mats3x3(kpts)
        >>> # execute function
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
                print(ut.hz_str('ERROR: invV_mats[%d] = ' % ix, invV))
                V_mats_list[ix] = np.nan(invV.shape)
        if ut.SUPER_STRICT:
            raise
        V_mats = np.array(V_mats_list)
    return V_mats


#@profile
def get_invV_xy_axis_extents(invV_mats):
    """ gets the scales of the major and minor elliptical axis.
        from invV_mats (faster)
    """
    if invV_mats.shape[1] == 3:
        # Take the SVD of only the shape part
        invV_mats = invV_mats[:, 0:2, 0:2]
    Us_list = [ltool.svd(invV)[0:2] for invV in invV_mats]
    def Us_axis_extent(U, s):
        """ Columns of U.dot(S) are in principle scaled directions """
        return np.sqrt(U.dot(np.diag(s)) ** 2).T.sum(0)
    xyexnts = np.array([Us_axis_extent(U, s) for U, s in Us_list])
    return xyexnts


#@profile
def get_xy_axis_extents(kpts):
    """
    gets the scales of the major and minor elliptical axis from kpts
    (slower due to conversion to invV_mats)

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints

    Returns:
        xyexnts:

    CommandLine:
        python -m vtool.keypoint --test-get_xy_axis_extents

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> # execute function
        >>> xyexnts = get_xy_axis_extents(kpts)
        >>> # verify results
        >>> result = str(xyexnts)
        >>> print(result)
        [[  6.2212909   24.91645859]
         [  2.79504602  24.7306281 ]
         [ 16.43837149  19.39813418]
         [ 18.23215582  25.76692184]
         [ 19.78704902  16.82756301]]
    """
    #invV_mats = get_invV_mats(kpts, ashomog=False)
    invV_mats2x2 = get_invVR_mats2x2(kpts)
    xyexnts = get_invV_xy_axis_extents(invV_mats2x2)
    return xyexnts


#@profile
def get_kpts_image_extent(kpts):
    """
    returns the width and height of keypoint bounding box
    This combines xy and shape information
    Does not take into account if keypoint extent goes under (0, 0)

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints

    Returns:
        tuple: wh_bound

    CommandLine:
        python -m vtool.keypoint --test-get_kpts_image_extent

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> # execute function
        >>> wh_bound = get_kpts_image_extent(kpts)
        >>> # verify results
        >>> result = kpts_repr(np.array(wh_bound))
        >>> print(result)
        array([ 51.79,  54.77])
    """
    xs, ys = get_xys(kpts)
    xyexnts = get_xy_axis_extents(kpts)
    width = (xs + xyexnts.T[0]).max()
    height = (ys + xyexnts.T[1]).max()
    wh_bound = (width, height)
    return wh_bound


def get_kpts_dlen_sqrd(kpts):
    r"""
    returns diagonal length squared of keypoint extent

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        float: dlen_sqrd

    CommandLine:
        python -m vtool.keypoint --test-get_kpts_dlen_sqrd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> # execute function
        >>> dlen_sqrd = get_kpts_dlen_sqrd(kpts)
        >>> # verify results
        >>> result = '%.2f' % dlen_sqrd
        >>> print(result)
        5681.31
    """
    if len(kpts) == 0:
        return 0.0
    w, h = get_kpts_image_extent(kpts)
    dlen_sqrd = (w ** 2) + (h ** 2)
    return dlen_sqrd


#@profile
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

#@profile
def get_xy_strs(kpts):
    """ strings debugging and output """
    _xs, _ys   = get_xys(kpts)
    xy_strs = [('xy=(%.1f, %.1f)' % (x, y,)) for x, y, in zip(_xs, _ys)]
    return xy_strs


#@profile
def get_shape_strs(kpts):
    """ strings debugging and output """
    invVs = get_invVs(kpts)
    shape_strs  = [(('[(%3.1f,  0.00),\n' +
                     ' (%3.1f, %3.1f)]') % (iv11, iv21, iv22,))
                   for iv11, iv21, iv22 in zip(*invVs)]
    shape_strs = ['invV=\n' +  _str for _str in shape_strs]
    return shape_strs


#@profile
def get_ori_strs(kpts):
    _oris = get_oris(kpts)
    ori_strs = ['ori=' + ut.theta_str(ori) for ori in _oris]
    return ori_strs


#@profile
def get_kpts_strs(kpts):
    xy_strs = get_xy_strs(kpts)
    shape_strs = get_shape_strs(kpts)
    ori_strs = get_ori_strs(kpts)
    kpts_strs = ['\n---\n'.join(tup) for tup in zip(xy_strs, shape_strs, ori_strs)]
    return kpts_strs


def kpts_repr(arr, precision=2, suppress_small=True, linebreak=False):
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
    CommandLine:
        python -m vtool.keypoint --test-kpts_docrepr

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> # build test data
        >>> np.random.seed(0)
        >>> arr = np.random.rand(3, 3)
        >>> args = tuple()
        >>> kwargs = dict()
        >>> # execute function
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

    CommandLine:
        python -m vtool.keypoint --test-get_match_spatial_squared_error:0
        python -m vtool.keypoint --test-get_match_spatial_squared_error:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> # build test data
        >>> kpts1 = np.array([[ 129.83,   46.97,   15.84,    4.66,    7.24,    0.  ],
        ...                   [ 137.88,   49.87,   20.09,    5.76,    6.2 ,    0.  ],
        ...                   [ 115.95,   53.13,   12.96,    1.73,    8.77,    0.  ],
        ...                   [ 324.88,  172.58,  127.69,   41.29,   50.5 ,    0.  ],
        ...                   [ 285.44,  254.61,  136.06,   -4.77,   76.69,    0.  ],
        ...                   [ 367.72,  140.81,  172.13,   12.99,   96.15,    0.  ]], dtype=np.float32)
        >>> kpts2 = np.array([[ 318.93,   11.98,   12.11,    0.38,    8.04,    0.  ],
        ...                   [ 509.47,   12.53,   22.4 ,    1.31,    5.04,    0.  ],
        ...                   [ 514.03,   13.04,   19.25,    1.74,    4.72,    0.  ],
        ...                   [ 490.19,  185.49,   95.67,   -4.84,   88.23,    0.  ],
        ...                   [ 316.97,  206.07,   90.87,    0.07,   80.45,    0.  ],
        ...                   [ 366.07,  140.05,  161.27,  -47.01,   85.62,    0.  ]], dtype=np.float32)
        >>> H = np.array([[ -0.70098,   0.12273,   5.18734],
        >>>               [  0.12444,  -0.63474,  14.13995],
        >>>               [  0.00004,   0.00025,  -0.64873]])
        >>> fx2_to_fx1 = np.array([[5, 4, 1, 0],
        >>>                        [0, 1, 5, 4],
        >>>                        [0, 1, 5, 4],
        >>>                        [2, 3, 1, 5],
        >>>                        [5, 1, 0, 4],
        >>>                        [3, 1, 5, 0]], dtype=np.int32)
        >>> # execute function
        >>> fx2_to_xyerr_sqrd = get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
        >>> fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
        >>> # verify results
        >>> result = ut.numpy_str(fx2_to_xyerr, precision=3)
        >>> print(result)
        np.array([[  82.848,  186.238,  183.979,  192.639],
                  [ 382.988,  374.356,  122.179,  289.16 ],
                  [ 387.563,  378.93 ,  126.389,  292.391],
                  [ 419.246,  176.668,  400.175,  167.411],
                  [ 174.269,  274.289,  281.03 ,   33.521],
                  [  54.083,  269.645,   94.711,  277.706]], dtype=np.float32)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> # build test data
        >>> kpts1 = np.array([[  6.,   4.,   15.84,    4.66,    7.24,    0.  ],
        ...                   [  9.,   3.,   20.09,    5.76,    6.2 ,    0.  ],
        ...                   [  1.,   1.,   12.96,    1.73,    8.77,    0.  ],])
        >>> kpts2 = np.array([[  2.,   1.,   12.11,    0.38,    8.04,    0.  ],
        ...                   [  5.,   1.,   22.4 ,    1.31,    5.04,    0.  ],
        ...                   [  6.,   1.,   19.25,    1.74,    4.72,    0.  ],])
        >>> H = np.array([[ 2,  0, 0],
        >>>               [ 0,  1, 0],
        >>>               [ 0,  0, 1]])
        >>> fx2_to_fx1 = np.array([[2, 1, 0],
        >>>                        [0, 1, 2],
        >>>                        [2, 1, 0]], dtype=np.int32)
        >>> # execute function
        >>> fx2_to_xyerr_sqrd = get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
        >>> fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
        >>> # verify results
        >>> result = ut.numpy_str(fx2_to_xyerr, precision=3)
        >>> print(result)
        np.array([[  0.   ,  16.125,  10.44 ],
                  [  7.616,  13.153,   3.   ],
                  [  4.   ,  12.166,   6.708]], dtype=np.float32)
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
    fx2_to_xyerr_sqrd = dtool.L2_sqrd(bcast_xy2, bcast_xy1_t)
    return fx2_to_xyerr_sqrd


def get_uneven_point_sample(kpts):
    """
    for each keypoint returns an uneven sample of points along the ellipical
      boundries.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        python -m vtool.keypoint --test-get_uneven_point_sample --show

    SeeAlso:
        pyhesaff.tests.test_ellipse
        python -m pyhesaff.tests.test_ellipse --test-in_depth_ellipse --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.dummy.get_dummy_kpts()[0:2]
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
    #ellipse_pts1 = [invV.dot(cicrle_pts) for invV in invV_mats]
    ellipse_pts1 = matrix_multiply(invV_mats, circle_pts.T).transpose(0, 2, 1)
    return ellipse_pts1


def get_even_point_sample(kpts):
    """
    gets even points sample along the boundary of the ellipse

    SeeAlso:
        pyhesaff.tests.test_ellipse

    CommandLine:
        python -m vtool.keypoint --test-get_even_point_sample --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.keypoint import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.dummy.get_dummy_kpts()[0:2]
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


#try:
#    import cyth
#    if cyth.DYNAMIC:
#        exec(cyth.import_cyth_execstr(__name__))
#    else:
#        # <AUTOGEN_CYTH>
#        # Regen command: python -c "import vtool.keypoint" --cyth-write
#        pass
#        # </AUTOGEN_CYTH>
#except ImportError as ex:
#    pass

if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.keypoint
        python -m vtool.keypoint --allexamples
        python -m vtool.keypoint --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
