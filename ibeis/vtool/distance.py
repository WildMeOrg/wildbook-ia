# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import warnings  # NOQA # TODO enable these warnings in strict mode
from collections import OrderedDict
(print, rrr, profile) = ut.inject2(__name__, '[dist]', DEBUG=False)
#profile = utool.profile

#TEMP_VEC_DTYPE = np.float32
TEMP_VEC_DTYPE = np.float64

#DEBUG_DIST = __debug__
DEBUG_DIST = False

TAU = 2 * np.pi  # References: tauday.com


def testdata_hist():
    import vtool as vt
    rng = np.random.RandomState(0)
    hist1 = vt.tests.dummy.testdata_dummy_sift(rng=rng)
    hist2 = vt.tests.dummy.testdata_dummy_sift(rng=rng)
    return hist1, hist2


def signed_ori_distance(ori1, ori2):
    r"""
    Args:
        ori1 (ndarray):
        ori2 (ndarray):

    Returns:
        ndarray: ori_dist

    CommandLine:
        python -m vtool.distance --exec-signed_ori_distance

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> ori1 = np.array([0,  0, 3, 4, 0, 0])
        >>> ori2 = np.array([3,  4, 0, 0, np.pi, np.pi - .1])
        >>> ori_dist = signed_ori_distance(ori1, ori2)
        >>> result = ('ori_dist = %s' % (ut.repr2(ori_dist, precision=3),))
        >>> print(result)
        ori_dist = np.array([ 3.   , -2.283, -3.   ,  2.283, -3.142,  3.042])
    """
    ori_dist = ori2 - ori1
    ori_dist = (ori_dist + np.pi) % TAU - np.pi
    return ori_dist


@profile
def ori_distance(ori1, ori2, out=None):
    r"""
    Returns the unsigned distance between two angles

    References:
        http://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles

    Timeit:
        >>> import utool as ut
        >>> setup = ut.codeblock(
        >>>     r'''
                # STARTBLOCK
                import numpy as np
                TAU = np.pi * 2
                rng = np.random.RandomState(53)
                ori1 = (rng.rand(100000) * TAU) - np.pi
                ori2 = (rng.rand(100000) * TAU) - np.pi

                def func_outvars():
                    ori_dist = np.abs(ori1 - ori2)
                    np.mod(ori_dist, TAU, out=ori_dist)
                    np.minimum(ori_dist, np.subtract(TAU, ori_dist), out=ori_dist)
                    return ori_dist

                def func_orig():
                    ori_dist = np.abs(ori1 - ori2) % TAU
                    ori_dist = np.minimum(ori_dist, TAU - ori_dist)
                    return ori_dist
                # ENDBLOCK
                ''')
        >>> stmt_list = ut.codeblock(
        >>>    '''
                func_outvars()
                func_orig()
                '''
        >>> ).split('\n')
        >>> ut.util_dev.rrr()
        >>> ut.util_dev.timeit_compare(stmt_list, setup, int(1E3))

    CommandLine:
        python -m vtool.distance --test-ori_distance

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> ori1 = (rng.rand(10) * TAU) - np.pi
        >>> ori2 = (rng.rand(10) * TAU) - np.pi
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ut.repr2(ori1, precision=1)
        >>> result += '\n' + ut.repr2(ori2, precision=1)
        >>> result += '\n' + ut.repr2(dist_, precision=1)
        >>> print(result)
        np.array([ 0.3,  1.4,  0.6,  0.3, -0.5,  0.9, -0.4,  2.5,  2.9, -0.7])
        np.array([ 1.8,  0.2,  0.4,  2.7, -2.7, -2.6, -3. ,  2.1,  1.7,  2.3])
        np.array([ 1.5,  1.2,  0.2,  2.4,  2.2,  2.8,  2.6,  0.4,  1.2,  3.1])

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> ori1 = np.array([ 0.3,  7.0,  0.0,  3.1], dtype=np.float64)
        >>> ori2 = np.array([ 6.8, -1.0,  0.0, -3.1], dtype=np.float64)
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ut.repr2(dist_, precision=2)
        >>> print(result)
        np.array([ 0.22,  1.72,  0.  ,  0.08])

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> ori1 = .3
        >>> ori2 = 6.8
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ut.repr2(dist_, precision=2)
        >>> print(result)
        0.21681469282041377

    Ignore:
        # This also works
        ori_dist = np.abs(np.arctan2(np.sin(ori1 - ori2), np.cos(ori1 - ori2)))
        %timeit np.abs(np.arctan2(np.sin(ori1 - ori2), np.cos(ori1 - ori2)))
    """
    # TODO: Cython
    #if out is None:
    #    out = np.empty(ori1.shape, dtype=np.float64)
    #ori_diff  = np.subtract(ori1, ori2, out=out)
    #abs_diff  = np.abs(ori_diff, out=out)
    #mod_diff1 = np.mod(abs_diff, TAU, out=out)
    #mod_diff2 = np.subtract(TAU, mod_diff1)
    #ori_dist  = np.minimum(mod_diff1, mod_diff2, out=out)
    ori_diff  = np.subtract(ori1, ori2)
    abs_diff  = np.abs(ori_diff)
    mod_diff1 = np.mod(abs_diff, TAU)
    mod_diff2 = np.subtract(TAU, mod_diff1)
    ori_dist  = np.minimum(mod_diff1, mod_diff2)
    if out is not None:
        out[:] = ori_dist
    return ori_dist


@profile
def det_distance(det1, det2):
    """ Returns how far off determinants are from one another

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> rng = np.random.RandomState(53)
        >>> det1 = rng.rand(1000)
        >>> det2 = rng.rand(1000)
        >>> scaledist = det_distance(det1, det2)
        >>> result = ut.repr2(scaledist, precision=2, threshold=2)
        >>> print(result)
        np.array([ 1.03,  1.19,  1.21, ...,  1.25,  1.83,  1.43])
    """
    det_dist = det1 / det2
    # Flip ratios that are less than 1
    _flip_flag = det_dist < 1
    det_dist[_flip_flag] = np.reciprocal(det_dist[_flip_flag])
    return det_dist


def L1(hist1, hist2, dtype=TEMP_VEC_DTYPE):
    """ returns L1 (aka manhatten or grid) distance between two histograms """
    return (np.abs(hist1 - hist2)).sum(-1)


def L2_sqrd(hist1, hist2, dtype=TEMP_VEC_DTYPE):
    """ returns the squared L2 distance

    # FIXME:
        if hist1.shape = (0,) and hist.shape = (0,) then result=0.0

    SeeAlso:
        L2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> rng = np.random.RandomState(53)
        >>> hist1 = rng.rand(1000, 2)
        >>> hist2 = rng.rand(1000, 2)
        >>> l2dist = L2_sqrd(hist1, hist2)
        >>> result = ut.repr2(l2dist, precision=2, threshold=2)
        >>> print(result)
        np.array([ 0.77,  0.27,  0.11, ...,  0.14,  0.3 ,  0.66])

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
    # Carefull, this will not return the correct result if the types are unsigned.
    #return ((hist1 - hist2) ** 2).sum(-1)  # this is faster
    return ((hist1.astype(dtype) - hist2.astype(dtype)) ** 2).sum(-1)  # this is faster


def understanding_pseudomax_props(mode=2):
    """
    Function showing some properties of distances between normalized pseudomax vectors

    CommandLine:
        python -m vtool.distance --test-understanding_pseudomax_props

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> for mode in [0, 1, 2, 3]:
        ...     print('+---')
        ...     print('mode = %r' % (mode,))
        ...     result = understanding_pseudomax_props(mode)
        ...     print('L___')
        >>> print(result)
    """
    import vtool as vt
    pseudo_max = 512
    rng = np.random.RandomState(0)
    num = 10
    if mode == 0:
        dim = 2
        p1_01 = (vt.normalize_rows(rng.rand(num, dim)))
        p2_01 = (vt.normalize_rows(rng.rand(num, dim)))
    elif mode == 1:
        p1_01 = vt.dummy.testdata_dummy_sift(num, rng) / pseudo_max
        p2_01 = vt.dummy.testdata_dummy_sift(num, rng) / pseudo_max
    elif mode == 2:
        # Build theoretically maximally distant normalized vectors (type 1)
        dim = 128
        p1_01 = np.zeros((1, dim))
        p2_01 = np.zeros((1, dim))
        p2_01[:, 0::2] = 1
        p1_01[:, 1::2] = 1
        p1_01 = vt.normalize_rows(p1_01)
        p2_01 = vt.normalize_rows(p2_01)
    elif mode == 3:
        # Build theoretically maximally distant vectors (type 2)
        # This mode will clip if cast to uint8, thus failing the test
        dim = 128
        p1_01 = np.zeros((1, dim))
        p2_01 = np.zeros((1, dim))
        p2_01[:, 0] = 1
        p1_01[:, 1:] = 1
        p1_01 = vt.normalize_rows(p1_01)
        p2_01 = vt.normalize_rows(p2_01)
        pass
    print('ndims = %r' % (p1_01.shape[1]))

    p1_01 = p1_01.astype(TEMP_VEC_DTYPE)
    p2_01 = p2_01.astype(TEMP_VEC_DTYPE)

    p1_256 = p1_01 * pseudo_max
    p2_256 = p2_01 * pseudo_max

    dist_sqrd_01 = vt.L2_sqrd(p1_01, p2_01)
    dist_sqrd_256 = vt.L2_sqrd(p1_256, p2_256)

    dist_01 = np.sqrt(dist_sqrd_01)
    dist_256 = np.sqrt(dist_sqrd_256)

    print('dist_sqrd_01  = %s' % (ut.numpy_str(dist_sqrd_01, precision=2),))
    print('dist_sqrd_256 = %s' % (ut.numpy_str(dist_sqrd_256, precision=2),))
    print('dist_01       = %s' % (ut.numpy_str(dist_01, precision=2),))
    print('dist_256      = %s' % (ut.numpy_str(dist_256, precision=2),))

    print('--')
    print('sqrt(2)       = %f' % (np.sqrt(2)))
    print('--')

    assert np.all(dist_01 == vt.L2(p1_01, p2_01))
    assert np.all(dist_256 == vt.L2(p1_256, p2_256))

    const_sqrd = dist_sqrd_256 / dist_sqrd_01
    const = dist_256 / dist_01

    print('const = %r' % (const[0],))
    print('const_sqrd = %r' % (const_sqrd[0],))
    print('1 / const = %r' % (1 / const[0],))
    print('1 / const_sqrd = %r' % (1 / const_sqrd[0],))

    assert ut.list_allsame(const)
    assert ut.list_allsame(const_sqrd)

    assert np.all(const == np.sqrt(const_sqrd))

    # Assert that distance conversions work
    assert np.all(dist_256 / const == dist_01)
    assert np.all(dist_sqrd_256 / const_sqrd == dist_sqrd_01)
    print('Conversions work')

    print('Maximal L2 distance between any two NON-NEGATIVE L2-NORMALIZED'
          ' vectors should always be sqrt(2)')


def L2(hist1, hist2):
    """ returns L2 (aka euclidean or standard) distance between two histograms """
    #return np.sqrt((np.abs(hist1 - hist2) ** 2).sum(-1))
    #((hist1.astype(TEMP_VEC_DTYPE) - hist2.astype(TEMP_VEC_DTYPE)) ** 2).sum(-1))
    return np.sqrt(L2_sqrd(hist1, hist2))


@profile
def hist_isect(hist1, hist2):
    """ returns histogram intersection distance between two histograms """
    numer = (np.dstack([hist1, hist2])).min(-1).sum(-1)
    denom = hist2.sum(-1)
    hisect_dist = 1 - (numer / denom)
    if len(hisect_dist) == 1:
        hisect_dist = hisect_dist[0]
    return hisect_dist

#from six.moves import zip
#from utool import util_inject
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
    """
    Args:
        hist1 (ndarray): Nx128 array of uint8 with pseudomax trick
        hist2 (ndarray): Nx128 array of uint8 with pseudomax trick

    1 - Normalized SIFT L2

    CommandLine:
        python -m vtool.distance --test-bar_L2_sift

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1, hist2 = testdata_hist()
        >>> barl2_dist = bar_L2_sift(hist1, hist2)
        >>> result = ut.repr2(barl2_dist, precision=2)
        >>> print(result)
        np.array([ 0.55,  0.51,  0.49,  0.51,  0.49,  0.52,  0.48,  0.48,  0.51,  0.45])
    """
    return 1.0 - L2_sift(hist1, hist2)


def L2_sift(hist1, hist2):
    """
    Normalized SIFT L2

    Args:
        hist1 (ndarray): Nx128 array of uint8 with pseudomax trick
        hist2 (ndarray): Nx128 array of uint8 with pseudomax trick

    Returns:
        ndarray: euclidean distance between 0-1 normalized sift descriptors

    CommandLine:
        python -m vtool.distance --test-L2_sift

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1, hist2 = testdata_hist()
        >>> l2_dist = L2_sift(hist1, hist2)
        >>> result = ut.repr2(l2_dist, precision=2)
        >>> print(result)
        np.array([ 0.45,  0.49,  0.51,  0.49,  0.51,  0.48,  0.52,  0.52,  0.49,  0.55])
    """
    # remove the pseudo max hack
    psuedo_max = 512.0
    max_l2_dist = np.sqrt(2)  # maximum L2 distance should always be sqrt 2
    sift1 = hist1.astype(TEMP_VEC_DTYPE) / psuedo_max
    sift2 = hist2.astype(TEMP_VEC_DTYPE) / psuedo_max
    l2_dist = L2(sift1, sift2)
    return l2_dist / max_l2_dist


def L2_root_sift(hist1, hist2):
    """
    Normalized Root-SIFT L2

    Args:
        hist1 (ndarray): Nx128 array of uint8 with pseudomax trick
        hist2 (ndarray): Nx128 array of uint8 with pseudomax trick

    Returns:
        ndarray: euclidean distance between 0-1 normalized sift descriptors
    """
    # remove the pseudo max hack
    psuedo_max = 512.0
    max_root_l2_dist = 2  # This is a guess
    sift1 = hist1.astype(TEMP_VEC_DTYPE) / psuedo_max
    sift2 = hist2.astype(TEMP_VEC_DTYPE) / psuedo_max
    root_sift1 = np.sqrt(sift1)
    root_sift2 = np.sqrt(sift2)
    l2_dist = L2(root_sift1, root_sift2)
    # Usure if correct;
    l2_root_dist =  l2_dist / max_root_l2_dist
    return l2_root_dist


def L2_sift_sqrd(hist1, hist2):
    """
    Normalized SIFT L2**2

    Args:
        hist1 (ndarray): Nx128 array of uint8 with pseudomax trick
        hist2 (ndarray): Nx128 array of uint8 with pseudomax trick

    Returns:
        ndarray: squared euclidean distance between 0-1 normalized sift descriptors
    """
    # remove the pseudo max hack
    psuedo_max = 512.0
    max_l2_dist_sqrd = 2
    sift1 = hist1.astype(TEMP_VEC_DTYPE) / psuedo_max
    sift2 = hist2.astype(TEMP_VEC_DTYPE) / psuedo_max
    l2_sqrd_dist = L2_sqrd(sift1, sift2)
    return l2_sqrd_dist / max_l2_dist_sqrd


def bar_cos_sift(hist1, hist2):
    """ 1 - cos dist  """
    return 1.0 - cos_sift(hist1, hist2)


def cos_sift(hist1, hist2):
    """
    cos dist

    CommandLine:
        python -m vtool.distance --test-cos_sift

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1, hist2 = testdata_hist()
        >>> l2_dist = cos_sift(hist1, hist2)
        >>> result = ut.repr2(l2_dist, precision=2)
        >>> print(result)
        np.array([ 0.77,  0.74,  0.72,  0.74,  0.72,  0.75,  0.71,  0.72,  0.74,  0.68])
    """
    psuedo_max = 512.0
    sift1 = hist1.astype(TEMP_VEC_DTYPE) / psuedo_max
    sift2 = hist2.astype(TEMP_VEC_DTYPE) / psuedo_max
    if DEBUG_DIST:
        _assert_siftvec(sift1)
        _assert_siftvec(sift2)
    #sift1 /= np.linalg.norm(sift1, axis=-1)
    #sift2 /= np.linalg.norm(sift2, axis=-1)
    return (sift1 * sift2).sum(-1)


def cosine_dist(hist1, hist2):
    return (hist1 * hist2).sum(-1)


def _assert_siftvec(sift):
    import vtool as vt
    assert vt.check_sift_validity(sift)
    #norm = (sift ** 2).sum(axis=-1)
    #isvalid = np.allclose(norm, 1.0, rtol=.05)
    #assert np.all(isvalid), norm[np.logical_not(isvalid)]


def emd(hist1, hist2, cost_matrix='sift'):
    """
    earth mover's distance by robjects(lpSovle::lp.transport)
    require: lpsolve55-5.5.0.9.win32-py2.7.exe

    Ignore:
        #http://docs.opencv.org/modules/imgproc/doc/histograms.html
        import re
        [x for x in cv2.__dict__.keys() if x.find('emd') > -1 or x.find('EMD') > -1]
        import re
        [x for x in cv2.__dict__.keys() if re.search('emd|earth', x, flags=re.IGNORECASE)]
        [x for x in cv2.__dict__.keys() if re.search('dist', x, flags=re.IGNORECASE)]
        CV_COMP_CORREL Correlation
        CV_COMP_CHISQR Chi-Square
        CV_COMP_INTERSECT Intersection
        CV_COMP_BHATTACHARYYA Bhattacharyya distance
        CV_COMP_HELLINGER

    CommandLine:
        python -m vtool.distance --test-emd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1, hist2 = testdata_hist()
        >>> emd_dists = emd(hist1, hist2)
        >>> result = ut.repr2(emd_dists, precision=2)
        >>> print(result)
        np.array([ 2063.99,  2078.02,  2109.03,  2011.99,  2130.99,  2089.01,
                   2030.99,  2294.98,  2026.02,  2426.01])

    References:
        pip install pyemd
        https://github.com/andreasjansson/python-emd
        http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf
        http://stackoverflow.com/questions/15706339/how-to-compute-emd-for-2-numpy-arrays-i-e-histogram-using-opencv
        http://www.cs.huji.ac.il/~ofirpele/FastEMD/code/
        http://www.cs.huji.ac.il/~ofirpele/publications/ECCV2008.pdf
    """
    import pyemd
    if cost_matrix == 'sift':
        # Build cost matrix where bin-to-bin cost is 0,
        # neighbor cost is 1, and other cost is 2
        N = 8
        cost_matrix = np.full((128, 128), 2)
        i, j = np.meshgrid(np.arange(128), np.arange(128))
        cost_matrix[i == j] = 0
        absdiff = np.abs(i - j)
        is_neighbor = np.abs(np.minimum(absdiff, N - absdiff)) == 1
        cost_matrix[is_neighbor] = 1.0
        #print(cost_matrix[0:16, 0:16])

    if len(hist1.shape) == 2:
        dist = np.array([
            pyemd.emd(hist1_.astype(np.float), hist2_.astype(np.float), cost_matrix)
            for hist1_, hist2_ in zip(hist1, hist2)])
    else:
        dist = pyemd.emd(hist1.astype(np.float), hist2.astype(np.float), cost_matrix)
    return dist

    if False:
        import cv2
        try:
            from cv2 import cv
        except ImportError as ex:
            #cv2.histComparse(
            print(repr(ex))
            print('Cannot import cv. Is opencv 2.4.9?. cv2.__version__=%r' % (cv2.__version__,))
            raise
            #return -1

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


def nearest_point(x, y, pts, conflict_mode='next', __next_counter=[0]):
    """ finds the nearest point(s) in pts to (x, y)

    TODO: depricate
    """
    #with ut.embed_on_exception_context:
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


def closest_point(pt, pt_arr, distfunc=L2_sqrd):
    """ finds the nearest point(s) in pts to (x, y)
    pt = np.array([1])
    pt_arr = np.array([1.1, 2, .95, 20])[:, None]
    distfunc = vt.L2_sqrd
    """
    #import vtool as vt
    assert len(pt_arr) > 0
    dists = distfunc(pt, pt_arr)
    xlist = dists.argsort()
    if len(xlist) > 1:
        if dists[xlist[0]] == dists[xlist[1]]:
            print('conflict')
    index = xlist[0]
    dist = dists[index]
    return index, dist


def haversine(latlon1, latlon2):
    r"""
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        latlon1 (ndarray):
        latlon2 (ndarray):

    References:
        en.wikipedia.org/wiki/Haversine_formula
        gis.stackexchange.com/questions/81551/matching-gps-tracks
        stackoverflow.com/questions/4913349/haversine-distance-gps-points

    CommandLine:
        python -m vtool.distance --exec-haversine

    LaTeX:
        from sympy import *
        import vtool as vt
        source = ut.get_func_sourcecode(vt.haversine, stripdef=True, strip_docstr=True, strip_comments=False, stripret=True)
        source = source[source.find('# haversine formula'):]
        source = source.replace('np.', '')
        source = source.replace('arcsin', 'asin')
        print(source)
        lon1, lon2, lat1, lat2 = symbols('\lon_i, \lon_j, \lat_i, \lat_j')
        locals_ = globals()
        locals_.update(locals())
        exec(source, locals_)
        c = locals_['c']
        print(vt.sympy_latex_repr(c))

2 \operatorname{asin}{(\sqrt{\sin^{2}{(\frac{\lat_i}{2} - \frac{\lat_j}{2} )} + \sin^{2}{(\frac{\lon_i}{2} - \frac{\lon_j}{2} )} \cos{(\lat_i )} \cos{(\lat_j )}} )}


    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> import scipy.spatial.distance as spdist
        >>> import vtool as vt
        >>> import functools
        >>> gpsarr_track_list_ = [
        ...    np.array([[ -80.21895315, -158.81099213],
        ...              [ -12.08338926,   67.50368014],
        ...              [ -11.08338926,   67.50368014],
        ...              [ -11.08338926,   67.50368014],]
        ...    ),
        ...    np.array([[   9.77816711,  -17.27471498],
        ...              [ -51.67678814, -158.91065495],])
        ...    ]
        >>> latlon1 = gpsarr_track_list_[0][0]
        >>> latlon2 = gpsarr_track_list_[0][1]
        >>> kilometers = vt.haversine(latlon1, latlon2)
        >>> haversin_pdist = functools.partial(spdist.pdist, metric=vt.haversine)
        >>> dist_vector_list = list(map(haversin_pdist, gpsarr_track_list_))
        >>> dist_matrix_list = list(map(spdist.squareform, dist_vector_list))
        >>> result = ('dist_matrix_list = %s' % (ut.repr3(dist_matrix_list, precision=2),))
        >>> print(result)
        dist_matrix_list = [
            np.array([[    0.  ,  9417.52,  9527.8 ,  9527.8 ],
                      [ 9417.52,     0.  ,   111.13,   111.13],
                      [ 9527.8 ,   111.13,     0.  ,     0.  ],
                      [ 9527.8 ,   111.13,     0.  ,     0.  ]], dtype=np.float64),
            np.array([[     0.  ,  14197.57],
                      [ 14197.57,      0.  ]], dtype=np.float64),
        ]
    """
    # convert decimal degrees to radians
    lat1, lon1 = np.radians(latlon1)
    lat2, lon2 = np.radians(latlon2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    EARTH_RADIUS_KM = 6367
    kilometers = EARTH_RADIUS_KM * c
    return kilometers


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
