"""
TODO: Look at this file
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
"""
from __future__ import absolute_import, division, print_function
#import sys
#sys.exit(1)
# THE NUMPY ERROR HAPPENS BECAUSE OF OPENCV
import cv2
#import six
#import functools
import numpy as np
import numpy.linalg as npl
import utool as ut
import warnings  # NOQA

profile = ut.profile
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[linalg]', DEBUG=False)

'''
#if CYTH
cdef np.float64_t TAU = 2 * np.pi
#endif
'''
TAU = 2 * np.pi  # References: tauday.com

TRANSFORM_DTYPE = np.float64


# Function which multiplies many matrices at once
from numpy.core.umath_tests import matrix_multiply  # NOQA


@profile
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


@profile
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


@profile
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


@profile
def rotation_mat3x3(radians):
    # TODO: handle array impouts
    sin_ = np.sin(radians)
    cos_ = np.cos(radians)
    R = np.array(((cos_, -sin_,  0),
                  (sin_,  cos_,  0),
                  (   0,     0,  1),))
    return R


@profile
def rotation_mat2x2(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    rot_ = np.array(((cos_, -sin_),
                     (sin_,  cos_),))
    return rot_


@profile
def rotation_around_mat3x3(theta, x, y):
    tr1_ = translation_mat3x3(-x, -y)
    rot_ = rotation_mat3x3(theta)
    tr2_ = translation_mat3x3(x, y)
    rot = tr2_.dot(rot_).dot(tr1_)
    return rot


@profile
def rotation_around_bbox_mat3x3(theta, bbox):
    x, y, w, h = bbox
    centerx = x + (w / 2)
    centery = y + (h / 2)
    return rotation_around_mat3x3(theta, centerx, centery)


@profile
def translation_mat3x3(x, y, dtype=TRANSFORM_DTYPE):
    T = np.array([[1, 0,  x],
                  [0, 1,  y],
                  [0, 0,  1]], dtype=dtype)
    return T


@profile
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


def affine_mat3x3(sx=1, sy=1, theta=0, shear=0, tx=0, ty=0):
    """
    Args:
        shear is angle in counterclockwise direction

    References:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
    """
    sin1_ = np.sin(theta)
    cos1_ = np.cos(theta)
    sin2_ = np.sin(theta + shear)
    cos2_ = np.cos(theta + shear)
    Aff = np.array([
        [sx * cos1_, -sy * sin2_, tx],
        [sx * sin1_,  sy * cos2_, ty],
        [        0,            0,  1]
    ])
    return Aff


def affine_around_mat3x3(x, y, sx=1, sy=1, theta=0, shear=0, tx=0, ty=0):
    # move to center location
    tr1_ = translation_mat3x3(-x, -y)
    # apply affine transform
    Aff_ = affine_mat3x3(sx, sy, theta, shear, tx, ty)
    # move to original location
    tr2_ = translation_mat3x3(x, y)
    # combine transformations
    Aff = tr2_.dot(Aff_).dot(tr1_)
    return Aff


@profile
#@ut.on_exception_report_input(force=True)
def scaleedoffset_mat3x3(offset, scale_factor):
    r"""
    Args:
        offset (tuple):
        scale_factor (scalar or tuple):

    Returns:
        ndarray[ndims=2]: M

    CommandLine:
        python -m vtool.linalg --test-scaleedoffset_mat3x3

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> offset = (11, 13)
        >>> scale_factor = (.3, .5)
        >>> # execute function
        >>> M = scaleedoffset_mat3x3(offset, scale_factor)
        >>> # verify results
        >>> result = ut.numpy_str(M, precision=2)
        >>> print(result)
        np.array([[  0.3,   0. ,  11. ],
                  [  0. ,   0.5,  13. ],
                  [  0. ,   0. ,   1. ]], dtype=np.float64)
    """
    try:
        sfx, sfy = scale_factor
    except TypeError:
        sfx = sfy = scale_factor
    #with ut.embed_on_exception_context:
    tx, ty = offset
    T = translation_mat3x3(tx, ty)
    S = scale_mat3x3(sfx, sfy)
    M = T.dot(S)
    return M


# Ensure that a feature doesn't have multiple assignments
# --------------------------------
# Linear algebra functions on lower triangular matrices


#PYX DEFINE
@profile
def det_ltri(ltri):
    #cdef det_ltri(FLOAT_2D ltri):
    """ Lower triangular determinant """
    #PYX CDEF FLOAT_1D det
    det = ltri[0] * ltri[2]
    return det


#PYX DEFINE
@profile
def inv_ltri(ltri, det):
    #cdef inv_ltri(FLOAT_2D ltri, FLOAT_1D det):
    """ Lower triangular inverse """
    # PYX CDEF FLOAT_2D inv_ltri
    inv_ltri = np.array((ltri[2], -ltri[1], ltri[0]), dtype=ltri.dtype) / det
    return inv_ltri


#PYX BEGIN
@profile
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


@profile
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

    #CYTH_INLINE
    #if CYTH
    cdef:
        np.ndarray[np.float64_t, ndim=2] xy_m
        np.ndarray[np.float64_t, ndim=1] mu_xy
        np.ndarray[np.float64_t, ndim=1] std_xy
        np.ndarray[np.float64_t, ndim=2] T
        np.float64_t tx, ty, sx, sy
        np.ndarray[np.float64_t, ndim=2] xy_norm
    #endif
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
    xyz  = add_homogenous_coordinate(_xys)
    xyz_t = matrix_multiply(H, xyz)
    xy_t  = remove_homogenous_coordinate(xyz_t)
    return xy_t


def normalize_rows(arr1):  # , out=None):
    """
    from vtool.linalg import *
    """
    #norm_ = npl.norm(arr1, axis=-1)
    # actually this is a colwise op
    #arr1_normed = rowwise_operation(arr1, norm_, np.divide)
    assert len(arr1.shape) == 2
    norm_ = npl.norm(arr1, axis=1)
    norm_.shape = (norm_.size, 1)
    arr1_normed = np.divide(arr1, norm_)  # , out=out)
    return arr1_normed


def normalize_vecs2d_inplace(vecs):
    """
    Args:
        vecs (ndarray): row vectors to normalize in place

    CommandLine:
        python -m vtool.linalg --test-normalize_vecs2d_inplace

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> import numpy as np
        >>> vecs1 = np.random.rand(1, 10)
        >>> normalize_vecs2d_inplace(vecs1)
        >>> res1 = str(vecs1)
        >>> #vecs2 = np.random.rand(10)
        >>> #normalize_vecs2d_inplace(vecs2)
        >>> #res2 = str(vecs2)
    """
    # Normalize residuals
    # this can easily be sped up by cyth
    assert len(vecs.shape) == 2
    norm_ = npl.norm(vecs, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(vecs, norm_.reshape(norm_.size, 1), out=vecs)
    return vecs


#try:
#    import cyth
#    if cyth.DYNAMIC:
#        exec(cyth.import_cyth_execstr(__name__))
#    else:
#        # <AUTOGEN_CYTH>
#        # Regen command: python -c "import vtool.linalg" --cyth-write
#        pass
#        # </AUTOGEN_CYTH>
#except Exception as ex:
#    pass


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
