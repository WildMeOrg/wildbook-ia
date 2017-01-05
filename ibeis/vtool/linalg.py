# -*- coding: utf-8 -*-
"""
TODO: Look at this file
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

Sympy:
    >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
    >>> import vtool as vt
    >>> import sympy
    >>> from sympy.abc import theta
    >>> x, y, a, c, d, sx, sy  = sympy.symbols('x y a c d, sx, sy')
    >>> R = vt.sympy_mat(vt.rotation_mat3x3(theta, sin=sympy.sin, cos=sympy.cos))
    >>> vt.evalprint('R')
    >>> #evalprint('R.inv()')
    >>> vt.evalprint('sympy.simplify(R.inv())')
    >>> #evalprint('sympy.simplify(R.inv().subs(theta, 4))')
    >>> #print('-------')
    >>> #invR = sympy_mat(vt.rotation_mat3x3(-theta, sin=sympy.sin, cos=sympy.cos))
    >>> #evalprint('invR')
    >>> #evalprint('invR.inv()')
    >>> #evalprint('sympy.simplify(invR)')
    >>> #evalprint('sympy.simplify(invR.subs(theta, 4))')
    >>> print('-------')
    >>> T = vt.sympy_mat(vt.translation_mat3x3(x, y, None))
    >>> vt.evalprint('T')
    >>> vt.evalprint('T.inv()')
    >>> print('-------')
    >>> S = vt.sympy_mat(vt.scale_mat3x3(sx, sy, dtype=None))
    >>> vt.evalprint('S')
    >>> vt.evalprint('S.inv()')
    >>> print('-------')
    >>> print('LaTeX')
    >>> print(ut.align('\\\\\n'.join(sympy.latex(R).split(r'\\')).replace('{matrix}', '{matrix}\n'), '&')

"""
from __future__ import absolute_import, division, print_function
#import sys
#sys.exit(1)
# THE NUMPY ERROR HAPPENS BECAUSE OF OPENCV
try:
    import cv2
except ImportError as ex:
    print('WARNING: import cv2 is failing!')
import numpy as np
import numpy.linalg as npl
import utool as ut
import warnings  # NOQA
(print, rrr, profile) = ut.inject2(__name__, '[linalg]')

TAU = 2 * np.pi  # References: tauday.com

TRANSFORM_DTYPE = np.float64


# Function which multiplies many matrices at once
from numpy.core.umath_tests import matrix_multiply  # NOQA


def svd(M):
    r"""
    Args:
        M (ndarray): must be either float32 or float64

    Returns:
        tuple: (U, s, Vt)

    CommandLine:
        python -m vtool.linalg --test-svd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> M = np.array([1, 2, 3], dtype=np.float32)
        >>> M = np.array([[20.5812, 0], [3.615, 17.1295]], dtype=np.float64)
        >>> # execute function
        >>> (U, s, Vt) = svd(M)

    Timeit::
        flags = cv2.SVD_FULL_UV
        %timeit cv2.SVDecomp(M, flags=flags)
        %timeit npl.svd(M)
    """
    # V is actually Vt
    flags = cv2.SVD_FULL_UV
    S, U, Vt = cv2.SVDecomp(M, flags=flags)
    s = S.flatten()
    #U, s, Vt = npl.svd(M)
    return U, s, Vt


def OLD_pdf_norm2d(x_, y_):
    """  DEPRICATED """
    import math
    x = np.array([x_, y_])
    sigma = np.eye(2)
    mu = np.array([0, 0])
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError('The covariance matrix cant be singular')
    TAU = 2 * np.pi
    norm_const = 1.0 / ( math.pow(TAU, float(size) / 2) * math.pow(det, 1.0 / 2))
    x_mu = np.matrix(x - mu)
    inv = np.linalg.inv(sigma)
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result


def gauss2d_pdf(x_, y_, sigma=None, mu=None):
    """
    Input: x and y coordinate of a 2D gaussian
           sigma, mu - covariance and mean vector
    Output: The probability density at that point
    """
    if sigma is None:
        sigma = np.eye(2)
    else:
        if not isinstance(sigma, np.ndarray):
            sigma = np.eye(2) * sigma
    if mu is None:
        mu = np.array([0, 0])
    x = np.array([x_, y_])
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = npl.det(sigma)
        if det == 0:
            raise NameError('The covariance matrix cant be singular')
    denom1 = TAU ** (size / 2.0)
    denom2 = np.sqrt(det)
    #norm_const = 1.0 / (denom1 * denom2)
    norm_const = np.reciprocal(denom1 * denom2)
    x_mu = x - mu  # deviation from mean
    invSigma = npl.inv(sigma)  # inverse covariance
    exponent = -0.5 * (x_mu.dot(invSigma).dot(x_mu.T))
    result = norm_const * np.exp(exponent)
    return result


def rotation_mat3x3(radians, sin=np.sin, cos=np.cos):
    """
    References:
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    # TODO: handle array impouts
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = np.array(((cos_, -sin_,  0),
                  (sin_,  cos_,  0),
                  (   0,     0,  1),))
    return R


def rotation_mat2x2(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    rot_ = np.array(((cos_, -sin_),
                     (sin_,  cos_),))
    return rot_


def transform_around(M, x, y):
    """ translates to origin, applies transform and then translates back """
    tr1_ = translation_mat3x3(-x, -y)
    tr2_ = translation_mat3x3(x, y)
    M_ = tr2_.dot(M).dot(tr1_)
    return M_


def rotation_around_mat3x3(theta, x, y):
    # rot = rotation_mat3x3(theta)
    # return transform_around(rot, x, y)
    tr1_ = translation_mat3x3(-x, -y)
    rot_ = rotation_mat3x3(theta)
    tr2_ = translation_mat3x3(x, y)
    rot = tr2_.dot(rot_).dot(tr1_)
    return rot


def scale_around_mat3x3(sx, sy, x, y):
    scale_ = scale_mat3x3(sx, sy)
    return transform_around(scale_, x, y)


def rotation_around_bbox_mat3x3(theta, bbox):
    x, y, w, h = bbox
    centerx = x + (w / 2)
    centery = y + (h / 2)
    return rotation_around_mat3x3(theta, centerx, centery)


def translation_mat3x3(x, y, dtype=TRANSFORM_DTYPE):
    T = np.array([[1, 0,  x],
                  [0, 1,  y],
                  [0, 0,  1]], dtype=dtype)
    return T


def scale_mat3x3(sx, sy=None, dtype=TRANSFORM_DTYPE):
    sy = sx if sy is None else sy
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0,  0, 1]], dtype=dtype)
    return S


def shear_mat3x3(shear_x, shear_y, dtype=TRANSFORM_DTYPE):
    shear = np.array([[      1, shear_x, 0],
                      [shear_y,       1, 0],
                      [      0,       0, 1]], dtype=dtype)
    return shear


#def affine_mat3x3_orig(sx=1, sy=1, theta=0, shear=0, tx=0, ty=0):
#    """
#    Args:
#        shear is angle in counterclockwise direction

#    References:
#        https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
#    """
#    sin1_ = np.sin(theta)
#    cos1_ = np.cos(theta)
#    sin2_ = np.sin(theta + shear)
#    cos2_ = np.cos(theta + shear)
#    Aff = np.array([
#        [sx * cos1_, -sy * sin2_, tx],
#        [sx * sin1_,  sy * cos2_, ty],
#        [        0,            0,  1]
#    ])
#    return Aff


def affine_mat3x3(sx=1, sy=1, theta=0, shear=0, tx=0, ty=0, trig=np):
    """
    Args:
        shear is angle in counterclockwise direction

    References:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
    """
    sin1_ = trig.sin(theta)
    cos1_ = trig.cos(theta)
    sin2_ = trig.sin(theta + shear)
    cos2_ = trig.cos(theta + shear)
    Aff = np.array([
        [sx * cos1_, -sy * sin2_, tx],
        [sx * sin1_,  sy * cos2_, ty],
        [        0,            0,  1]
    ])
    return Aff


def affine_around_mat3x3_old(x, y, sx=1, sy=1, theta=0, shear=0, tx=0, ty=0,
                             x2=None, y2=None, trig=np, dtype=TRANSFORM_DTYPE):
    """
    Executes an affine transform around point (x, y) in the input coordinate
    system.

    Specify x2 and y2 if change of coordinates (but not scale)?

    >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
    >>> from vtool.linalg import *  # NOQA
    >>> import vtool as vt
    >>> import sympy
    >>> from sympy.abc import theta
    >>> x, y, sx, sy, theta, shear, tx, ty, x2, y2 = sympy.symbols(
    >>>     'x, y, sx, sy, theta, shear, tx, ty, x2, y2')
    >>> trig = sympy
    >>> Aff = vt.sympy_mat(vt.affine_around_mat3x3_old(
    >>>     x, y, sx, sy, theta, shear, tx, ty, x2, y2, trig=trig, dtype=None))
    >>> print('-------')
    >>> Aff = sympy.simplify(Aff)
    >>> vt.evalprint('Aff')
    >>> print('-------')
    >>> print('Numpy')
    >>> vt.sympy_numpy_repr(Aff)
    """
    x2 = x if x2 is None else x2
    y2 = y if y2 is None else y2
    # move to center location
    tr1_ = translation_mat3x3(-x, -y, dtype=dtype)
    # apply affine transform
    Aff_ = affine_mat3x3(sx, sy, theta, shear, tx, ty, trig=trig)
    # move to original location
    tr2_ = translation_mat3x3(x2, y2, dtype=dtype)
    # combine transformations
    Aff = tr2_.dot(Aff_).dot(tr1_)
    return Aff


def affine_around_mat3x3(x, y, sx=1, sy=1, theta=0, shear=0, tx=0, ty=0,
                         x2=None, y2=None):
    """
    Combines a translate with affine3x3 with untranslate

    Args:
        x (float): center x location in input space
        y (float): center y location in input space
        x2 (float, optional): center x location in output space if different
            from input space (assumed the same if unspecified)
        y2 (float, optional): center y location in output space if different
            from input space (assumed the same if unspecified)

    Example:
        >>> from vtool.linalg import *  # NOQA
        >>> tup = (256.0, 256.0, 1.5, 1.0, 0.7853981633974483, 0.2, 0, 100, 500.0, 500.0)
        >>> x, y, sx, sy, theta, shear, tx, ty, x2, y2 = tup
        >>> Aff1 = affine_around_mat3x3(x, y, sx, sy, theta, shear, tx, ty, x2, y2)
        >>> Aff2 = affine_around_mat3x3_old(x, y, sx, sy, theta, shear, tx, ty, x2, y2)
        >>> assert np.all(Aff2 == Aff1)

    %timeit affine_around_mat3x3_old(x, y, sx, sy, theta, shear, tx, ty, x2, y2)
    %timeit affine_around_mat3x3(x, y, sx, sy, theta, shear, tx, ty, x2, y2)
    """
    x2 = x if x2 is None else x2
    y2 = y if y2 is None else y2
    # Make auxially varables to reduce the number of sin/cosine calls
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_shear_p_theta = np.cos(shear + theta)
    sin_shear_p_theta = np.sin(shear + theta)
    tx_ = -sx * x * cos_theta + sy * y * sin_shear_p_theta + tx + x2
    ty_ = -sx * x * sin_theta - sy * y * cos_shear_p_theta + ty + y2
    # Sympy compiled expression
    Aff = np.array([[sx * cos_theta, -sy * sin_shear_p_theta, tx_],
                    [sx * sin_theta,  sy * cos_shear_p_theta, ty_],
                    [             0,                       0, 1]])
    return Aff


#@ut.on_exception_report_input(force=True)
#def scaleedoffset_mat3x3(offset, scale_factor):
#    r"""
#    Args:
#        offset (tuple):
#        scale_factor (scalar or tuple):
#
#    Returns:
#        ndarray[ndims=2]: M
#
#    CommandLine:
#        python -m vtool.linalg --test-scaleedoffset_mat3x3
#
#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from vtool.linalg import *  # NOQA
#        >>> # build test data
#        >>> offset = (11, 13)
#        >>> scale_factor = (.3, .5)
#        >>> # execute function
#        >>> M = scaleedoffset_mat3x3(offset, scale_factor)
#        >>> # verify results
#        >>> result = ut.numpy_str(M, precision=2)
#        >>> print(result)
#        np.array([[  0.3,   0. ,  11. ],
#                  [  0. ,   0.5,  13. ],
#                  [  0. ,   0. ,   1. ]], dtype=np.float64)
#    """
#    try:
#        sfx, sfy = scale_factor
#    except TypeError:
#        sfx = sfy = scale_factor
#    #with ut.embed_on_exception_context:
#    tx, ty = offset
#    T = translation_mat3x3(tx, ty)
#    S = scale_mat3x3(sfx, sfy)
#    M = T.dot(S)
#    return M


# Ensure that a feature doesn't have multiple assignments
# --------------------------------
# Linear algebra functions on lower triangular matrices


#PYX DEFINE
def det_ltri(ltri):
    #cdef det_ltri(FLOAT_2D ltri):
    """ Lower triangular determinant """
    #PYX CDEF FLOAT_1D det
    det = ltri[0] * ltri[2]
    return det


#PYX DEFINE
def inv_ltri(ltri, det):
    #cdef inv_ltri(FLOAT_2D ltri, FLOAT_1D det):
    """ Lower triangular inverse """
    # PYX CDEF FLOAT_2D inv_ltri
    inv_ltri = np.array((ltri[2], -ltri[1], ltri[0]), dtype=ltri.dtype) / det
    return inv_ltri


#PYX BEGIN
def dot_ltri(ltri1, ltri2):
    """ Lower triangular dot product """
    #cdef dot_ltri(FLOAT_2D ltri1, FLOAT_2D ltri2):
    # PYX FLOAT_1D m11, m21, m22
    # PYX FLOAT_1D n11, n21, n22
    # PYX FLOAT_1D o11, o21, o22
    # PYX FLOAT_2D ltri3
    # use m, n, and o as temporary matrixes
    m11, m21, m22 = ltri1
    n11, n21, n22 = ltri2
    o11 = (m11 * n11)
    o21 = (m21 * n11) + (m22 * n21)
    o22 = (m22 * n22)
    ltri3 = np.array((o11, o21, o22), dtype=ltri1.dtype)
    return ltri3
# PYX END CDEF


def whiten_xy_points(xy_m):
    """
    whitens points to mean=0, stddev=1 and returns transformation

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> from vtool.tests import dummy
        >>> xy_m = dummy.get_dummy_xy()
        >>> tup = whiten_xy_points(xy_m)
        >>> xy_norm, T = tup
        >>> result = (ut.hashstr(tup))
        >>> print(result)
        wg%mpai0hxvil4p2
    """
    mu_xy  = xy_m.mean(1)  # center of mass
    std_xy = xy_m.std(1)
    std_xy[std_xy == 0] = 1  # prevent divide by zero
    tx, ty = -mu_xy / std_xy
    sx, sy = 1 / std_xy
    T = np.array([(sx, 0, tx),
                  (0, sy, ty),
                  (0,  0,  1)])
    xy_norm = ((xy_m.T - mu_xy) / std_xy).T
    return xy_norm, T


def add_homogenous_coordinate(_xys):
    r"""
    CommandLine:
        python -m vtool.linalg --test-add_homogenous_coordinate

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> _xys = np.array([[ 2.,  0.,  0.,  2.],
        ...                  [ 2.,  2.,  0.,  0.]], dtype=np.float32)
        >>> # execute function
        >>> _xyzs = add_homogenous_coordinate(_xys)
        >>> # verify results
        >>> assert np.all(_xys == remove_homogenous_coordinate(_xyzs))
        >>> result = ut.numpy_str(_xyzs)
        >>> print(result)
        np.array([[ 2.,  0.,  0.,  2.],
                  [ 2.,  2.,  0.,  0.],
                  [ 1.,  1.,  1.,  1.]], dtype=np.float32)
    """
    assert _xys.shape[0] == 2
    _zs = np.ones((1, _xys.shape[1]), dtype=_xys.dtype)
    _xyzs = np.vstack((_xys, _zs))
    return _xyzs


def remove_homogenous_coordinate(_xyzs):
    """
    normalizes 3d homogonous coordinates into 2d coordinates

    Args:
        _xyzs (ndarray): of shape (3, N)

    Returns:
        ndarray: _xys of shape (2, N)

    CommandLine:
        python -m vtool.linalg --test-remove_homogenous_coordinate

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> _xyzs = np.array([[ 2.,   0.,  0.,  2.],
        ...                   [ 2.,   2.,  0.,  0.],
        ...                   [ 1.2,  1.,  1.,  2.]], dtype=np.float32)
        >>> # execute function
        >>> _xys = remove_homogenous_coordinate(_xyzs)
        >>> # verify results
        >>> result = ut.numpy_str(_xys, precision=3)
        >>> print(result)
        np.array([[ 1.667,  0.   ,  0.   ,  1.   ],
                  [ 1.667,  2.   ,  0.   ,  0.   ]], dtype=np.float32)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> _xyzs = np.array([[ 140.,  167.,  185.,  185.,  194.],
        ...                   [ 121.,  139.,  156.,  155.,  163.],
        ...                   [  47.,   56.,   62.,   62.,   65.]])
        >>> # execute function
        >>> _xys = remove_homogenous_coordinate(_xyzs)
        >>> # verify results
        >>> result = np.array_repr(_xys, precision=3)
        >>> print(result)
        array([[ 2.979,  2.982,  2.984,  2.984,  2.985],
               [ 2.574,  2.482,  2.516,  2.5  ,  2.508]])
    """
    assert _xyzs.shape[0] == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xys = np.divide(_xyzs[0:2], _xyzs[None, 2])
    return _xys


def transform_points_with_homography(H, _xys):
    """
    Args:
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        _xys (ndarray[ndim=2]): (2 x N) array
    """
    xyz  = add_homogenous_coordinate(_xys)
    xyz_t = matrix_multiply(H, xyz)
    xy_t  = remove_homogenous_coordinate(xyz_t)
    return xy_t


def normalize_rows(arr, out=None):
    """ DEPRICATE """
    assert len(arr.shape) == 2
    return normalize(arr, axis=1, out=out)


def normalize(arr, ord=None, axis=None, out=None):
    r"""
    Returns all row vectors normalized by their magnitude.

    Args:
        arr (ndarray): row vectors to normalize
        ord (int): type of norm to use (defaults to 2-norm)
            {non-zero int, inf, -inf}
        axis (int): axis to normalize
        out (ndarray): preallocated output

    SeeAlso:
        np.linalg.norm

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> arr = np.array([[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]])
        >>> arr_normed = normalize(arr, axis=1)
        >>> result = ut.hz_str('arr_normed = ', ut.numpy_str(arr_normed, precision=2))
        >>> assert np.allclose((arr_normed ** 2).sum(axis=1), [1, 1])
        >>> print(result)
        arr_normed = np.array([[ 0.13,  0.27,  0.4 ,  0.54,  0.67],
                               [ 0.45,  0.45,  0.45,  0.45,  0.45]], dtype=np.float64)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> arr = np.array([ 0.6,  0.1, -0.6])
        >>> arr_normed = normalize(arr)
        >>> result = ut.hz_str('arr_normed = ', ut.numpy_str(arr_normed, precision=2))
        >>> assert np.allclose((arr_normed ** 2).sum(), [1])
        >>> print(result)

    Example:
        >>> from vtool.linalg import *  # NOQA
        >>> ord_list = [0, 1, 2, np.inf, -np.inf]
        >>> arr = np.array([ 0.6,  0.1, -0.5])
        >>> normed = [(ord, normalize(arr, ord=ord)) for ord in ord_list]
        >>> result = ut.repr4(normed, precision=2)
        >>> print(result)
        [
            (0, np.array([ 0.2 ,  0.03, -0.17], dtype=np.float64)),
            (1, np.array([ 0.5 ,  0.08, -0.42], dtype=np.float64)),
            (2, np.array([ 0.76,  0.13, -0.64], dtype=np.float64)),
            (inf, np.array([ 1.  ,  0.17, -0.83], dtype=np.float64)),
            (-inf, np.array([ 6.,  1., -5.], dtype=np.float64)),
        ]

    """
    norm_ = np.linalg.norm(arr, ord=ord, axis=axis, keepdims=True)
    arr_normed = np.divide(arr, norm_, out=out)
    return arr_normed


def random_affine_args(zoom_pdf=None,
                       tx_pdf=None,
                       ty_pdf=None,
                       shear_pdf=None,
                       theta_pdf=None,
                       enable_flip=False,
                       enable_stretch=False,
                       default_distribution='uniform',
                       scalar_anchor='reflect',  # 0
                       txy_pdf=None,
                       rng=np.random):
    r"""

    TODO: allow for a pdf of ranges for each dimension

    If pdfs are tuples it is interpreted as a default (uniform) distribution between the
    two points. A single scalar is a default distribution between -scalar and
    scalar.

    Args:
        zoom_range (tuple): (default = (1.0, 1.0))
        tx_range (tuple): (default = (0.0, 0.0))
        ty_range (tuple): (default = (0.0, 0.0))
        shear_range (tuple): (default = (0, 0))
        theta_range (tuple): (default = (0, 0))
        enable_flip (bool): (default = False)
        enable_stretch (bool): (default = False)
        rng (module):  random number generator(default = numpy.random)

    Returns:
        tuple: affine_args

    CommandLine:
        python -m vtool.linalg --exec-random_affine_args --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> import vtool as vt
        >>> zoom_range = (0.9090909090909091, 1.1)
        >>> tx_pdf = (0.0, 4.0)
        >>> ty_pdf = (0.0, 4.0)
        >>> shear_pdf = (0, 0)
        >>> theta_pdf = (0, 0)
        >>> enable_flip = False
        >>> enable_stretch = False
        >>> rng = np.random.RandomState(0)
        >>> affine_args = random_affine_args(
        >>>     zoom_range, tx_pdf, ty_pdf, shear_pdf, theta_pdf,
        >>>     enable_flip, enable_stretch, rng=rng)
        >>> print('affine_args = %s' % (ut.repr2(affine_args),))
        >>> (sx, sy, theta, shear, tx, ty) = affine_args
        >>> Aff = vt.affine_mat3x3(sx, sy, theta, shear, tx, ty)
        >>> result = ut.numpy_str2(Aff)
        >>> print(result)
        np.array([[ 1.009, -0.   ,  1.695],
                  [ 0.   ,  1.042,  2.584],
                  [ 0.   ,  0.   ,  1.   ]])
    """
    if zoom_pdf is None:
        sx = sy = 1.0
    else:
        log_zoom_range = [np.log(z) for z in zoom_pdf]

        if enable_stretch:
            sx = sy = np.exp(rng.uniform(*log_zoom_range))
        else:
            sx = np.exp(rng.uniform(*log_zoom_range))
            sy = np.exp(rng.uniform(*log_zoom_range))

    def param_distribution(param_pdf, rng=rng):
        if param_pdf is None:
            param = 0
        elif not ut.isiterable(param_pdf):
            max_param = param_pdf
            if scalar_anchor == 'reflect':
                min_param = -max_param
            elif scalar_anchor == 0:
                min_param = 0
            param = rng.uniform(min_param, max_param)
        elif isinstance(param_pdf, tuple):
            min_param, max_param = param_pdf
            param = rng.uniform(min_param, max_param)
        else:
            assert False
        return param

    theta = param_distribution(theta_pdf)
    shear = param_distribution(shear_pdf)
    if txy_pdf is not None:
        assert tx_pdf is None, 'cannot specify both'
        assert ty_pdf is None, 'cannot specify both'
        xy_locs, xy_probs = txy_pdf
        tx = param_distribution(tx_pdf)
    else:
        tx = param_distribution(tx_pdf)
        ty = param_distribution(ty_pdf)

    flip = enable_flip and (rng.randint(2) > 0)  # flip half of the time
    if flip:
        # shear 180 degrees + rotate 180 == flip
        theta += np.pi
        shear += np.pi

    affine_args = (sx, sy, theta, shear, tx, ty)
    return affine_args
    #Aff = vt.affine_mat3x3(sx, sy, theta, shear, tx, ty)
    #return Aff


def random_affine_transform(*args, **kwargs):
    affine_args = random_affine_args(**kwargs)
    Aff = affine_mat3x3(*affine_args)
    return Aff


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.linalg
        python -m vtool.linalg --allexamples
        python -m vtool.linalg --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
