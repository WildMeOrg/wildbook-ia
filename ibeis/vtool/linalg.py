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
from numpy import (array, sin, cos)
import utool

profile = utool.profile
(print, print_, printDBG, rrr, profile) = utool.inject(
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
    x = array([x_, y_])
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
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = array(((cos_, -sin_,  0),
               (sin_,  cos_,  0),
               (   0,     0,  1),))
    return R


@profile
def rotation_mat2x2(theta):
    sin_ = sin(theta)
    cos_ = cos(theta)
    rot_ = array(((cos_, -sin_),
                  (sin_,  cos_),))
    return rot_


@profile
def rotation_around_mat3x3(theta, x, y):
    sin_ = sin(theta)
    cos_ = cos(theta)
    tr1_ = array([[1, 0, -x],
                  [0, 1, -y],
                  [0, 0, 1]])
    rot_ = array([[cos_, -sin_, 0],
                  [sin_, cos_,  0],
                  [   0,    0,  1]])
    tr2_ = array([[1, 0, x],
                  [0, 1, y],
                  [0, 0, 1]])
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
    T = array([[1, 0,  x],
               [0, 1,  y],
               [0, 0,  1]], dtype=dtype)
    return T


@profile
def scale_mat3x3(sx, sy=None, dtype=TRANSFORM_DTYPE):
    sy = sx if sy is None else sy
    S = array([[sx, 0, 0],
               [0, sy, 0],
               [0,  0, 1]], dtype=dtype)
    return S


@profile
def scaleedoffset_mat3x3(offset, scale_factor):
    sfy = sfx = scale_factor
    T = translation_mat3x3(*offset)
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
    inv_ltri = array((ltri[2], -ltri[1], ltri[0]), dtype=ltri.dtype) / det
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
    ltri3 = array((o11, o21, o22), dtype=ltri1.dtype)
    return ltri3
# PYX END CDEF


@profile
def nearest_point(x, y, pts, mode='random'):
    """ finds the nearest point(s) in pts to (x, y) """
    dists = (pts.T[0] - x) ** 2 + (pts.T[1] - y) ** 2
    fx = dists.argmin()
    mindist = dists[fx]
    other_fx = np.where(mindist == dists)[0]
    if len(other_fx) > 0:
        if mode == 'random':
            np.random.shuffle(other_fx)
            fx = other_fx[0]
        if mode == 'all':
            fx = other_fx
        if mode == 'first':
            fx = fx
    return fx, mindist


def intersect2d_indicies(A, B):
    r"""
    Args:
        A (ndarray[ndims=2]):
        B (ndarray[ndims=2]):

    Returns:
        tuple: (ax_list, bx_list)

    CommandLine:
        python -m vtool.linalg --test-intersect2d_indicies

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> A = np.array([[ 158,  171], [ 542,  297], [ 955, 1113], [ 255, 1254], [ 976, 1255], [ 170, 1265]])
        >>> B = array([[ 117,  211], [ 158,  171], [ 255, 1254], [ 309,  328], [ 447, 1148], [ 750,  357], [ 976, 1255]])
        >>> # execute function
        >>> (ax_list, bx_list) = intersect2d_indicies(A, B)
        >>> # verify results
        >>> result = str((ax_list, bx_list))
        >>> print(result)
        (array([0, 3, 4]), array([1, 2, 6]))
    """
    flag_list1, flag_list2 = intersect2d_flags(A, B)
    ax_list = np.flatnonzero(flag_list1)
    bx_list = np.flatnonzero(flag_list2)
    return ax_list, bx_list


def intersect2d_flags(A, B):
    r"""
    Args:
        A (ndarray[ndims=2]):
        B (ndarray[ndims=2]):

    Returns:
        tuple: (flag_list1, flag_list2)

    CommandLine:
        python -m vtool.linalg --test-intersect2d_flags

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> A = array([[609, 307], [ 95, 344], [  1, 690]])
        >>> B = array([[ 422, 1148], [ 422,  968], [ 481, 1148], [ 750, 1132], [ 759,  159]])
        >>> # execute function
        >>> (flag_list1, flag_list2) = intersect2d_flags(A, B)
        >>> # verify results
        >>> result = str((flag_list1, flag_list2))
        >>> print(result)
        (array([False, False, False], dtype=bool), array([False, False, False, False, False], dtype=bool))
    """
    A_, B_, C_  = intersect2d_structured_numpy(A, B)
    flag_list1 = flag_intersection(A_, C_)
    flag_list2 = flag_intersection(B_, C_)
    return flag_list1, flag_list2


def flag_intersection(X_, C_):
    if X_.size == 0 or C_.size == 0:
        flags = np.full(X_.shape[0], False, dtype=np.bool)
        #return np.empty((0,), dtype=np.bool)
    else:
        flags = np.logical_or.reduce([X_ == c for c in C_]).T[0]
    return flags


def intersect2d_structured_numpy(A, B, assume_unique=False):
    nrows, ncols = A.shape
    assert A.dtype == B.dtype, ('A and B must have the same dtypes.'
                                'A.dtype=%r, B.dtype=%r' % (A.dtype, B.dtype))
    dtype = np.dtype([('f%d' % i, A.dtype) for i in range(ncols)])
    #try:
    A_ = np.ascontiguousarray(A).view(dtype)
    B_ = np.ascontiguousarray(B).view(dtype)
    C_ = np.intersect1d(A_, B_, assume_unique=assume_unique)
    #C = np.intersect1d(A.view(dtype),
    #                   B.view(dtype),
    #                   assume_unique=assume_unique)
    #except ValueError:
    #    C = np.intersect1d(A.copy().view(dtype),
    #                       B.copy().view(dtype),
    #                       assume_unique=assume_unique)
    return A_, B_, C_


def intersect2d_numpy(A, B, assume_unique=False, return_indicies=False):
    """
    References::
        http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays/8317155#8317155

    Args:
        A (ndarray[ndims=2]):
        B (ndarray[ndims=2]):
        assume_unique (bool):

    Returns:
        ndarray[ndims=2]: C

    CommandLine:
        python -m vtool.linalg --test-intersect2d_numpy

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> A = np.array([[  0,  78,  85, 283, 396, 400, 403, 412, 535, 552],
        ...               [152,  98,  32, 260, 387, 285,  22, 103,  55, 261]]).T
        >>> B = np.array([[403,  85, 412,  85, 815, 463, 613, 552],
        ...                [ 22,  32, 103, 116, 188, 199, 217, 254]]).T
        >>> assume_unique = False
        >>> # execute function
        >>> C, Ax, Bx = intersect2d_numpy(A, B, return_indicies=True)
        >>> # verify results
        >>> result = str((C.T, Ax, Bx))
        >>> print(result)
        (array([[ 85, 403, 412],
               [ 32,  22, 103]]), array([2, 6, 7]), array([0, 1, 2]))

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> A = np.array([[1, 2, 3], [1, 1, 1]])
        >>> B = np.array([[1, 2, 3], [1, 2, 14]])
        >>> C, Ax, Bx = intersect2d_numpy(A, B, return_indicies=True)
        >>> result = str((C, Ax, Bx))
        >>> print(result)
        (array([[1, 2, 3]]), array([0]), array([0]))
    """
    nrows, ncols = A.shape
    A_, B_, C_ = intersect2d_structured_numpy(A, B, assume_unique)
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C_.view(A.dtype).reshape(-1, ncols)
    if return_indicies:
        ax_list = np.flatnonzero(flag_intersection(A_, C_))
        bx_list = np.flatnonzero(flag_intersection(B_, C_))
        return C, ax_list, bx_list
    else:
        return C


def get_uncovered_mask(covered_array, covering_array):
    r"""
    Args:
        covered_array (ndarray):
        covering_array (ndarray):

    Returns:
        ndarray: flags

    CommandLine:
        python -m vtool.linalg --test-get_mask_cover

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> covered_array = [1, 2, 3, 4, 5]
        >>> covering_array = [2, 4, 5]
        >>> flags = get_uncovered_mask(covered_array, covering_array)
        >>> result = str(flags)
        >>> print(result)
        [ True False  True False False]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> covered_array = [1, 2, 3, 4, 5]
        >>> covering_array = []
        >>> flags = get_uncovered_mask(covered_array, covering_array)
        >>> result = str(flags)
        >>> print(result)
        [ True  True  True  True  True]

    """
    if len(covering_array) == 0:
        return np.ones(np.shape(covered_array), dtype=np.bool)
    else:
        flags_list = (np.not_equal(covered_array, item) for item in covering_array)
        mask_array = and_lists(*flags_list)
        return mask_array


def get_covered_mask(covered_array, covering_array):
    return ~get_uncovered_mask(covered_array, covering_array)


def mult_lists(*args):
    return np.multiply.reduce(args)


def or_lists(*args):
    """
    Like np.logical_and, but can take more than 2 arguments

    SeeAlso:
        and_lists
    """
    flags = np.logical_or.reduce(args)
    return flags


def and_lists(*args):
    """
    Like np.logical_and, but can take more than 2 arguments

    CommandLine:
        python -m vtool.linalg --test-and_lists

    SeeAlso:
       or_lists

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> arg1 = np.array([1, 1, 1, 1,])
        >>> arg2 = np.array([1, 1, 0, 1,])
        >>> arg3 = np.array([0, 1, 0, 1,])
        >>> args = (arg1, arg2, arg3)
        >>> flags = and_lists(*args)
        >>> result = str(flags)
        >>> print(result)
        [False  True False  True]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> size = 10000
        >>> np.random.seed(0)
        >>> arg1 = np.random.randint(2, size=size)
        >>> arg2 = np.random.randint(2, size=size)
        >>> arg3 = np.random.randint(2, size=size)
        >>> args = (arg1, arg2, arg3)
        >>> flags = and_lists(*args)
        >>> # ensure equal division
        >>> segments = 5
        >>> validx = np.where(flags)[0]
        >>> endx = int(segments * (validx.size // (segments)))
        >>> parts = np.split(validx[:endx], segments)
        >>> result = str(list(map(np.sum, parts)))
        >>> print(result)
        [243734, 714397, 1204989, 1729375, 2235191]

    %timeit reduce(np.logical_and, args)
    %timeit np.logical_and.reduce(args)  # wins with more data
    """
    flags = np.logical_and.reduce(args)
    return flags
    # TODO: Cython
    # TODO: remove reduce statement (bleh)
    #if six.PY2:
    #    flags =  reduce(np.logical_and, args)
    #elif six.PY3:
    #    flags =  functools.reduce(np.logical_and, args)
    #return flags


def and_3lists(arr1, arr2, arr3):
    """
    >>> from vtool.linalg import *  # NOQA
    >>> np.random.seed(53)
    >>> arr1 = (np.random.rand(1000) > .5).astype(np.bool)
    >>> arr2 = (np.random.rand(1000) > .5).astype(np.bool)
    >>> arr3 = (np.random.rand(1000) > .5).astype(np.bool)
    >>> output = and_3lists(arr1, arr2, arr3)
    >>> print(utool.hashstr(output))
    prxuyy1w%ht57jaf

    #if CYTH
    #CYTH_INLINE
    cdef:
        np.ndarray arr1
        np.ndarray arr2
        np.ndarray arr3
    #endif
    """
    return np.logical_and(np.logical_and(arr1, arr2), arr3)


@profile
def ori_distance(ori1, ori2):
    """ Returns how far off determinants are from one another
    >>> from vtool.linalg import *  # NOQA
    >>> np.random.seed(53)
    >>> ori1 = (np.random.rand(10) * TAU) - np.pi
    >>> ori2 = (np.random.rand(10) * TAU) - np.pi
    >>> output = utool.hashstr(utool.hashstr(ori_distance(ori1, ori2)))
    >>> print(utool.hashstr(output))
    !755pt!alrfgshiu

    Cyth:
        #if CYTH
        #CYTH_INLINE
        #CYTH_PARAM_TYPES:
            np.ndarray ori1
            np.ndarray ori2
        #endif

    Timeit:
        >>> import utool
        >>> setup = utool.codeblock(
        ...     '''
                import numpy as np
                TAU = np.pi * 2
                np.random.seed(53)
                ori1 = (np.random.rand(100000) * TAU) - np.pi
                ori2 = (np.random.rand(100000) * TAU) - np.pi

                def func_outvars():
                    ori_dist = np.abs(ori1 - ori2)
                    np.mod(ori_dist, TAU, out=ori_dist)
                    np.minimum(ori_dist, np.subtract(TAU, ori_dist), out=ori_dist)
                    return ori_dist

                def func_orig():
                    ori_dist = np.abs(ori1 - ori2) % TAU
                    ori_dist = np.minimum(ori_dist, TAU - ori_dist)
                    return ori_dist
                ''')
        >>> stmt_list = utool.codeblock(
        ...     '''
                func_outvars()
                func_orig()
                '''
        ... ).split('\n')
        >>> utool.util_dev.rrr()
        >>> utool.util_dev.timeit_compare(stmt_list, setup, int(1E3))

    """
    # TODO: Cython
    # TODO: Outvariable
    ori_dist = np.abs(ori1 - ori2)
    np.mod(ori_dist, TAU, out=ori_dist)
    np.minimum(ori_dist, np.subtract(TAU, ori_dist), out=ori_dist)
    return ori_dist


@profile
def det_distance(det1, det2):
    """ Returns how far off determinants are from one another
    >>> from vtool.linalg import *  # NOQA
    >>> np.random.seed(53)
    >>> det1 = np.random.rand(1000)
    >>> det2 = np.random.rand(1000)
    >>> output = det_distance(det1, det2)
    >>> print(utool.hashstr(output))
    pfce!exwvqz8e1n!

    #CYTH_INLINE
    #CYTH_RETURNS np.ndarray[np.float64_t, ndim=1]
    #CYTH_PARAM_TYPES:
        np.ndarray[np.float64_t, ndim=1] det1
        np.ndarray[np.float64_t, ndim=1] det2
    #if CYTH
    # TODO: Move to ktool?
    cdef unsigned int nDets = det1.shape[0]
    # Prealloc output
    out = np.zeros((nDets,), dtype=det1.dtype)
    cdef size_t ix
    for ix in range(nDets):
        # simple determinant: ad - bc
        if det1[ix] > det2[ix]:
            out[ix] = det1[ix] / det2[ix]
        else:
            out[ix] = det2[ix] / det1[ix]
    return out
    #else
    """
    # TODO: Cython
    det_dist = det1 / det2
    # Flip ratios that are less than 1
    _flip_flag = det_dist < 1
    #det_dist[_flip_flag] = (1.0 / det_dist[_flip_flag])
    det_dist[_flip_flag] = np.reciprocal(det_dist[_flip_flag])
    return det_dist


@profile
def L1(hist1, hist2):
    """ returns L1 (aka manhatten or grid) distance between two histograms """
    return (np.abs(hist1 - hist2)).sum(-1)


@profile
def L2_sqrd(hist1, hist2):
    """ returns the squared L2 distance
    seealso L2
    Test:
    >>> from vtool.linalg import *  # NOQA
    >>> np.random.seed(53)
    >>> hist1 = np.random.rand(1000, 2)
    >>> hist2 = np.random.rand(1000, 2)
    >>> output = L2_sqrd(hist1, hist2)
    >>> print(utool.hashstr(output))
    v9wc&brmvjy1as!z

    #CYTH_INLINE
    #CYTH_RETURNS np.ndarray[np.float64_t, ndim=1]
    #CYTH_PARAM_TYPES:
        np.ndarray[np.float64_t, ndim=2] hist1
        np.ndarray[np.float64_t, ndim=2] hist2
    #if CYTH
    cdef:
        size_t cx, rx
    cdef unsigned int rows = hist1.shape[0]
    cdef unsigned int cols = hist1.shape[1]
    # Prealloc output
    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros((rows,), dtype=hist1.dtype)
    for rx in range(rows):
        for cx in range(cols):
            out[rx] += (hist1[rx, cx] - hist2[rx, cx]) ** 2
    return out
    #else
    """
    # TODO: np.ufunc
    # TODO: Cython
    # temp memory
    #temp = np.empty(hist1.shape, dtype=hist1.dtype)
    #np.subtract(hist1, hist2, temp)
    #np.abs(temp, temp)
    #np.power(temp, 2, temp)
    #out = temp.sum(-1)
    return ((hist1 - hist2) ** 2).sum(-1)  # this is faster
    #return out


@profile
def L2(hist1, hist2):
    """ returns L2 (aka euclidean or standard) distance between two histograms """
    return np.sqrt((np.abs(hist1 - hist2) ** 2).sum(-1))


@profile
def hist_isect(hist1, hist2):
    """ returns histogram intersection distance between two histograms """
    numer = (np.dstack([hist1, hist2])).min(-1).sum(-1)
    denom = hist2.sum(-1)
    hisect_dist = 1 - (numer / denom)
    if len(hisect_dist) == 1:
        hisect_dist = hisect_dist[0]
    return hisect_dist


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
        >>> result = (utool.hashstr(tup))
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
    assert _xys.shape[0] == 2
    _zs = np.ones((1, _xys.shape[1]), dtype=_xys.dtype)
    _xyzs = np.vstack((_xys, _zs))
    return _xyzs


def remove_homogenous_coordinate(_xyzs):
    assert _xyzs.shape[0] == 3
    _xys = np.divide(_xyzs[0:2], _xyzs[None, 2])
    return _xys


def transform_points_with_homography(H, _xys):
    xyz  = add_homogenous_coordinate(_xys)
    xyz_t = matrix_multiply(H, xyz)
    xy_t  = remove_homogenous_coordinate(xyz_t)
    return xy_t


def homogonize(_xyzs):
    """
    DEPRICATE in favor of remove_homogenous_coordinate

    normalizes 3d homogonous coordinates into 2d coordinates

    Args:
        _xyzs (ndarray): of shape (3, N)

    Returns:
        ndarray: _xys of shape (2, N)

    CommandLine:
        python -m vtool.linalg --test-homogonize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> # build test data
        >>> _xyzs = np.array([[ 140.,  167.,  185.,  185.,  194.],
        ...                   [ 121.,  139.,  156.,  155.,  163.],
        ...                   [  47.,   56.,   62.,   62.,   65.]])
        >>> # execute function
        >>> _xys = homogonize(_xyzs)
        >>> # verify results
        >>> result = np.array_repr(_xys, precision=3)
        >>> print(result)
        array([[ 2.979,  2.982,  2.984,  2.984,  2.985],
               [ 2.574,  2.482,  2.516,  2.5  ,  2.508]])
    """
    _xys = np.divide(_xyzs[0:2], _xyzs[np.newaxis, 2])
    #_xs, _ys, _zs = _xyzs
    #_xys = np.vstack((_xs / _zs, _ys / _zs))
    return _xys


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


@profile
def axiswise_operation2(arr1, arr2, op, axis=0):
    """
    Apply opperation to each row

    >>> arr1 = (255 * np.random.rand(5, 128)).astype(np.uint8)
    >>> arr2 = vecs.mean(axis=0)
    >>> op = np.subtract
    >>> axis = 0

    performs an operation between an
    (N x A x B ... x Z) array with an
    (N x 1) array

    %timeit op(arr1, arr2[np.newaxis, :])
    %timeit op(arr1, arr2[None, :])
    %timeit op(arr1, arr2.reshape(1, arr2.shape[0]))
    arr2.shape = (1, arr2.shape[0])
    %timeit op(arr1, arr2)
    """


@profile
def rowwise_operation(arr1, arr2, op):
    """
    DEPRICATE THIS IS POSSIBLE WITH STRICTLY BROADCASTING AND
    USING np.newaxis

    DEPRICATE, numpy has better ways of doing this.
    Is the rowwise name correct? Should it be colwise?

    performs an operation between an
    (N x A x B ... x Z) array with an
    (N x 1) array
    """
    # FIXME: not sure this is the correct terminology
    assert arr1.shape[0] == arr2.shape[0]
    broadcast_dimensions = arr1.shape[1:]  # need padding for
    tileshape = tuple(list(broadcast_dimensions) + [1])
    arr2_ = np.rollaxis(np.tile(arr2, tileshape), -1)
    rowwise_result = op(arr1, arr2_)
    return rowwise_result


def colwise_operation(arr1, arr2, op):
    arr1T = arr1.T
    arr2T = arr2.T
    rowwise_result = rowwise_operation(arr1T, arr2T, op)
    colwise_result = rowwise_result.T
    return colwise_result


def rowwise_oridist(arr1, arr2):
    op = ori_distance
    return rowwise_operation(arr1, arr2, op)


@profile
def rowwise_division(arr1, arr2):
    """ DEPRICATE THIS IS POSSIBLE WITH STRICTLY BROADCASTING """
    return rowwise_operation(arr1, arr2, np.divide)


def compare_matrix_columns(matrix, columns):
    # FIXME: Generalize
    #row_matrix = matrix.T
    #row_list   = columns.T
    return compare_matrix_to_rows(matrix.T, columns.T).T


@profile
def compare_matrix_to_rows(row_matrix, row_list, comp_op=np.equal, logic_op=np.logical_or):
    # FIXME: Generalize
    """
    Compares each row in row_list to each row in row matrix using comp_op
    Both must have the same number of columns.
    Performs logic_op on the results of each individual row

    compop   = np.equal
    logic_op = np.logical_or
    """
    row_result_list = [array([comp_op(matrow, row) for matrow in row_matrix])
                       for row in row_list]
    output = row_result_list[0]
    for row_result in row_result_list[1:]:
        output = logic_op(output, row_result)
    return output

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
