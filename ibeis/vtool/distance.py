# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import warnings  # NOQA # TODO enable these warnings in strict mode
from collections import OrderedDict
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[dist]', DEBUG=False)
#profile = utool.profile


TAU = 2 * np.pi  # References: tauday.com


@profile
def ori_distance(ori1, ori2):
    r""" Returns how far off determinants are from one another

    Cyth:
        #if CYTH
        #CYTH_INLINE
        #CYTH_PARAM_TYPES:
            np.ndarray ori1
            np.ndarray ori2
        #endif

    Timeit:
        >>> import utool as ut
        >>> setup = ut.codeblock(
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
        >>> stmt_list = ut.codeblock(
        ...     '''
                func_outvars()
                func_orig()
                '''
        ... ).split('\n')
        >>> ut.util_dev.rrr()
        >>> ut.util_dev.timeit_compare(stmt_list, setup, int(1E3))

    CommandLine:
        python -m vtool.distance --test-ori_distance

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> np.random.seed(0)
        >>> ori1 = (np.random.rand(10) * TAU) - np.pi
        >>> ori2 = (np.random.rand(10) * TAU) - np.pi
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ut.numpy_str(ori1, precision=1)
        >>> result += '\n' + ut.numpy_str(ori2, precision=1)
        >>> result += '\n' + ut.numpy_str(dist_, precision=1)
        >>> print(result)
        np.array([ 0.3,  1.4,  0.6,  0.3, -0.5,  0.9, -0.4,  2.5,  2.9, -0.7], dtype=np.float64)
        np.array([ 1.8,  0.2,  0.4,  2.7, -2.7, -2.6, -3. ,  2.1,  1.7,  2.3], dtype=np.float64)
        np.array([ 1.5,  1.2,  0.2,  2.4,  2.2,  2.8,  2.6,  0.4,  1.2,  3.1], dtype=np.float64)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> ori1 = np.array([ 0.3,  7.0,  0.0,  3.1], dtype=np.float64)
        >>> ori2 = np.array([ 6.8, -1.0,  0.0, -3.1], dtype=np.float64)
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ut.numpy_str(dist_, precision=1)
        >>> print(result)
        np.array([ 0.2,  1.7,  0. ,  0.1], dtype=np.float64)
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> np.random.seed(53)
        >>> det1 = np.random.rand(1000)
        >>> det2 = np.random.rand(1000)
        >>> output = det_distance(det1, det2)
        >>> result = ut.hashstr(output)
        >>> print(result)
        pfce!exwvqz8e1n!

    Cyth::
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
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
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

    SeeAlso:
        L2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.linalg import *  # NOQA
        >>> np.random.seed(53)
        >>> hist1 = np.random.rand(1000, 2)
        >>> hist2 = np.random.rand(1000, 2)
        >>> output = L2_sqrd(hist1, hist2)
        >>> result = ut.hashstr(output)
        >>> print(result)
        v9wc&brmvjy1as!z

    Cyth::
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


def nearest_point(x, y, pts, conflict_mode='next', __next_counter=[0]):
    """ finds the nearest point(s) in pts to (x, y)
    """
    with ut.embed_on_exception_context:
        dists = (pts.T[0] - x) ** 2 + (pts.T[1] - y) ** 2
        fx = dists.argmin()
        mindist = dists[fx]
        other_fx = np.where(mindist == dists)[0]
        if len(other_fx) > 0:
            if conflict_mode == 'random':
                np.random.shuffle(other_fx)
                fx = other_fx[0]
            elif conflict_mode == 'next':
                __next_counter[0] += 1
                idx = __next_counter[0] % len(other_fx)
                fx = other_fx[idx]
            elif conflict_mode == 'all':
                fx = other_fx
            elif conflict_mode == 'first':
                fx = fx
            else:
                raise AssertionError('unknown conflict_mode=%r' % (conflict_mode,))
    return fx, mindist

#from six.moves import zip
#from utool import util_inject
#print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dist]')
VALID_DISTS = [
    'L1',
    'L2',
    'L2_sift',
    'L2_sqrd',
    'bar_L2_sift',
    'bar_cos_sift',
    'cos_sift',
    'det_distance',
    'emd',
    'hist_isect',
    'nearest_point',
    'ori_distance',
]


def compute_distances(hist1, hist2, dist_list=['L1', 'L2']):
    r"""
    Args:
        hist1 (ndarray):
        hist2 (ndarray):
        dist_list (list): (default = ['L1', 'L2'])

    Returns:
        dict: dist_dict

    Ignore:
        # Build valid dist list programtically
        import vtool
        func_list = ut.get_module_owned_functions(vtool.distance)
        funcname_list = [ut.get_funcname(x) for x in func_list]
        funcname_list = [n for n in funcname_list if n not in ['compute_distances']]
        print('VALID_DISTS = ' + ut.list_str(sorted(funcname_list)))

    CommandLine:
        python -m vtool.distance --test-compute_distances

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1 = np.array([[1, 2], [2, 1], [0, 0]])
        >>> hist2 = np.array([[1, 2], [3, 1], [2, 2]])
        >>> dist_list = ['L1', 'L2']
        >>> dist_dict = compute_distances(hist1, hist2, dist_list)
        >>> result = ut.dict_str(dist_dict, precision=3)
        >>> print(result)
        {
            'L1': np.array([ 0.,  1.,  4.], dtype=np.float64),
            'L2': np.array([ 0.   ,  1.   ,  2.828], dtype=np.float64),
        }
    """
    dtype_ = np.float64
    hist1 = np.array(hist1, dtype=dtype_)
    hist2 = np.array(hist2, dtype=dtype_)
    # TODO: enumerate value distances
    dist_funcs = [globals()[type_] for type_ in dist_list]
    val_list = [func(hist1, hist2) for func in dist_funcs]
    dist_dict = OrderedDict(list(zip(dist_list, val_list)))
    return dist_dict


def bar_L2_sift(hist1, hist2):
    """  1 - Normalized SIFT L2 """
    return 1.0 - L2_sift(hist1, hist2)


def bar_cos_sift(hist1, hist2):
    """  1 - Normalized SIFT L2 """
    return 1.0 - cos_sift(hist1, hist2)


def L2_sift(hist1, hist2):
    """  1 - Normalized SIFT L2 """
    psuedo_max = 512.0
    sift1 = hist1 / psuedo_max
    sift2 = hist2 / psuedo_max
    sift1 /= np.linalg.norm(sift1)
    sift2 /= np.linalg.norm(sift2)
    return L2(sift1, sift2)


def cos_sift(hist1, hist2):
    """ returns the squared L2 distance
    seealso L2
    """
    psuedo_max = 512.0
    sift1 = hist1 / psuedo_max
    sift2 = hist2 / psuedo_max
    sift1 /= np.linalg.norm(sift1)
    sift2 /= np.linalg.norm(sift2)
    #import utool as ut
    #ut.embed()
    return (sift1 * sift2).sum(-1)


def emd(hist1, hist2):
    """
    earth mover's distance by robjects(lpSovle::lp.transport)
    require: lpsolve55-5.5.0.9.win32-py2.7.exe

    Example:
        >>> from vtool.distances import *   # NOQA
        >>> import numpy as np
        >>> hist1 = np.random.rand(128)
        >>> hist2 = np.random.rand(128)
        >>> result = emd(hist1, hist2)

    References:
        https://github.com/andreasjansson/python-emd
        http://stackoverflow.com/questions/15706339/how-to-compute-emd-for-2-numpy-arrays-i-e-histogram-using-opencv
        http://www.cs.huji.ac.il/~ofirpele/FastEMD/code/
        http://www.cs.huji.ac.il/~ofirpele/publications/ECCV2008.pdf
    """
    try:
        from cv2 import cv
    except ImportError as ex:
        print(repr(ex))
        print('Cannot import cv. Is opencv 2.4.9?')
        return -1

    # Stack weights into the first column
    def add_weight(hist):
        weights = np.ones(len(hist))
        stacked = np.ascontiguousarray(np.vstack([weights, hist]).T)
        return stacked

    def convertCV32(stacked):
        hist64 = cv.fromarray(stacked)
        hist32 = cv.CreateMat(hist64.rows, hist64.cols, cv.CV_32FC1)
        cv.Convert(hist64, hist32)
        return hist32

    def emd_(a32, b32):
        return cv.CalcEMD2(a32, b32, cv.CV_DIST_L2)

    # HACK
    if len(hist1.shape) == 1 and len(hist2.shape) == 1:
        a, b = add_weight(hist1), add_weight(hist2)
        a32, b32 = convertCV32(a), convertCV32(b)
        emd_dist = emd_(a32, b32)
        return emd_dist
    else:
        ab_list   = [(add_weight(a), add_weight(b)) for a, b in zip(hist1, hist2)]
        ab32_list = [(convertCV32(a), convertCV32(b)) for a, b in ab_list]
        emd_dists = [emd_(a32, b32) for a32, b32, in ab32_list]
        return emd_dists


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.distance
        python -m vtool.distance --allexamples
        python -m vtool.distance --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
