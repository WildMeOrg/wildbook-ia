from __future__ import absolute_import, division, print_function
# Science
import cv2
import six
import functools
import numpy as np
import numpy.linalg as npl
from numpy import (array, sin, cos)
import utool

profile = utool.profile
#(print, print_, printDBG, rrr, profile) = utool.inject(
#    __name__, '[linalg]', DEBUG=False)

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
    # V is actually Vt
    flags = cv2.SVD_FULL_UV
    S, U, Vt = cv2.SVDecomp(M, flags=flags)
    s = S.flatten()
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


@profile
def and_lists(*args):
    """ Like np.logical_and, but can take more than 2 arguments
    """
    # TODO: Cython
    # TODO: remove reduce statement (bleh)
    if six.PY2:
        flags =  reduce(np.logical_and, args)
    elif six.PY3:
        flags =  functools.reduce(np.logical_and, args)
    return flags


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
    >>> from vtool.linalg import *  # NOQA
    >>> from vtool.tests import dummy
    >>> xy_m = dummy.get_dummy_xy()
    >>> tup = whiten_xy_points(xy_m)
    >>> xy_norm, T = tup
    >>> print(utool.hashstr(tup))
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


def homogonize(_xyzs):
    """ normalizes 3d homogonous coordinates into 2d coordinates """
    _xs, _ys, _zs = _xyzs
    _xys = np.vstack((_xs / _zs, _ys / _zs))
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
