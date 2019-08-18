# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import ubelt as ub
import itertools
from six.moves import range, zip
from collections import OrderedDict
import scipy.spatial.distance as spdist
from .util_math import TAU

TEMP_VEC_DTYPE = np.float64


def testdata_hist():
    import vtool as vt
    rng = np.random.RandomState(0)
    hist1 = vt.demodata.testdata_dummy_sift(rng=rng)
    hist2 = vt.demodata.testdata_dummy_sift(rng=rng)
    return hist1, hist2


def testdata_sift2():
    sift1 = np.zeros(128)
    sift2 = np.ones(128)
    sift3 = np.zeros(128)
    sift4 = np.zeros(128)
    sift5 = np.zeros(128)
    sift1[0] = 1
    sift3[-1] = 1
    sift4[0::2] = 1
    sift5[1::2] = 1

    def normalize_sift(sift):
        # normalize
        sift_norm = sift / np.linalg.norm(sift)
        # clip
        sift_norm = np.clip(sift_norm, 0, .2)
        # re-normalize
        sift_norm = sift_norm / np.linalg.norm(sift_norm)
        # cast hack
        sift_norm = np.clip(sift_norm * 512.0, 0, 255).astype(np.uint8)
        return sift_norm
    sift1 = normalize_sift(sift1)
    sift2 = normalize_sift(sift2)
    sift3 = normalize_sift(sift3)
    sift4 = normalize_sift(sift4)
    sift5 = normalize_sift(sift5)

    return sift1, sift2, sift3, sift4, sift5


def wrapped_distance(arr1, arr2, base, out=None):
    """
    base = TAU corresponds to ori diff
    """
    arr_diff  = np.subtract(arr1, arr2)
    abs_diff  = np.abs(arr_diff)
    mod_diff1 = np.mod(abs_diff, base)
    mod_diff2 = np.subtract(base, mod_diff1)
    arr_dist  = np.minimum(mod_diff1, mod_diff2)
    if out is not None:
        out[:] = arr_dist
    return arr_dist


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
        >>> result = ('ori_dist = %s' % (ub.repr2(ori_dist, precision=3),))
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)
    """
    ori_dist = ori2 - ori1
    ori_dist = (ori_dist + np.pi) % TAU - np.pi
    return ori_dist


def ori_distance(ori1, ori2, out=None):
    r"""
    Returns the unsigned distance between two angles

    References:
        http://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles

    Timeit:
        >>> #xdoctest: +SKIP
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> import utool as ut
        >>> setup = ub.codeblock(
        >>>     r'''
                # STARTBLOCK
                import numpy as np
                tau = np.pi * 2
                rng = np.random.RandomState(53)
                ori1 = (rng.rand(100000) * tau) - np.pi
                ori2 = (rng.rand(100000) * tau) - np.pi

                def func_outvars():
                    ori_dist = np.abs(ori1 - ori2)
                    np.mod(ori_dist, tau, out=ori_dist)
                    np.minimum(ori_dist, np.subtract(tau, ori_dist), out=ori_dist)
                    return ori_dist

                def func_orig():
                    ori_dist = np.abs(ori1 - ori2) % tau
                    ori_dist = np.minimum(ori_dist, tau - ori_dist)
                    return ori_dist
                # ENDBLOCK
                ''')
        >>> stmt_list = ub.codeblock(
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
        >>> result = ub.repr2(ori1, precision=1)
        >>> result += '\n' + ub.repr2(ori2, precision=1)
        >>> result += '\n' + ub.repr2(dist_, precision=1)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> ori1 = np.array([ 0.3,  7.0,  0.0,  3.1], dtype=np.float64)
        >>> ori2 = np.array([ 6.8, -1.0,  0.0, -3.1], dtype=np.float64)
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ub.repr2(dist_, precision=2)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> ori1 = .3
        >>> ori2 = 6.8
        >>> dist_ = ori_distance(ori1, ori2)
        >>> result = ub.repr2(dist_, precision=2)
        >>> print(result)

    Ignore:
        # This also works
        ori_dist = np.abs(np.arctan2(np.sin(ori1 - ori2), np.cos(ori1 - ori2)))
        %timeit np.abs(np.arctan2(np.sin(ori1 - ori2), np.cos(ori1 - ori2)))
    """
    return cyclic_distance(ori1, ori2, modulo=TAU, out=out)


def cyclic_distance(arr1, arr2, modulo, out=None):
    r"""
    returns an unsigned distance

    Args:
        arr1 (ndarray):
        arr2 (ndarray):
        modulo (float or int):
        out (ndarray): (default = None)

    Returns:
        ndarray: arr_dist

    CommandLine:
        python -m vtool.distance cyclic_distance

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> out = None
        >>> modulo = 8
        >>> offset = 0  # doesnt matter what offset is
        >>> arr1 = np.hstack([np.arange(offset, modulo + offset), np.nan])
        >>> arr2 = arr1[:, None]
        >>> arr_dist = cyclic_distance(arr1, arr2, modulo, out)
        >>> result = ('arr_dist =\n%s' % (ub.repr2(arr_dist),))
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)
    """
    arr_diff = np.subtract(arr1, arr2, out=out)
    abs_diff = np.abs(arr_diff, out=out)
    mod_diff1 = np.mod(abs_diff, modulo, out=out)
    mod_diff2 = np.subtract(modulo, mod_diff1)
    arr_dist  = np.minimum(mod_diff1, mod_diff2, out=out)
    return arr_dist


def signed_cyclic_distance(arr1, arr2, modulo, out=None):
    arr_diff = np.subtract(arr1, arr2, out=out)
    half_mod = modulo / 2
    arr_dist = (arr_diff + half_mod) % modulo - half_mod
    return arr_dist


def det_distance(det1, det2):
    """ Returns how far off determinants are from one another

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> rng = np.random.RandomState(53)
        >>> det1 = rng.rand(5)
        >>> det2 = rng.rand(5)
        >>> scaledist = det_distance(det1, det2)
        >>> result = ub.repr2(scaledist, precision=2, threshold=2)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)
    """
    det_dist = det1 / det2
    # Flip ratios that are less than 1
    _flip_flag = det_dist < 1
    det_dist[_flip_flag] = np.reciprocal(det_dist[_flip_flag])
    return det_dist


def L1(hist1, hist2, dtype=TEMP_VEC_DTYPE):
    """ returns L1 (aka manhatten or grid) distance between two histograms """
    return (np.abs(np.asarray(hist1, dtype) - np.asarray(hist2, dtype))).sum(-1)


def L2_sqrd(hist1, hist2, dtype=TEMP_VEC_DTYPE):
    """ returns the squared L2 distance

    # FIXME:
        if hist1.shape = (0,) and hist.shape = (0,) then result=0.0

    SeeAlso:
        L2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> import numpy
        >>> ut.exec_funckw(L2_sqrd, globals())
        >>> rng = np.random.RandomState(53)
        >>> hist1 = rng.rand(5, 2)
        >>> hist2 = rng.rand(5, 2)
        >>> l2dist = L2_sqrd(hist1, hist2)
        >>> result = ub.repr2(l2dist, precision=2, threshold=2)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1 = 3
        >>> hist2 = 0
        >>> result = L2_sqrd(hist1, hist2)
        >>> print(result)
    """
    # Carefull, this will not return the correct result if the types are unsigned.
    hist1_ = np.asarray(hist1, dtype)
    hist2_ = np.asarray(hist2, dtype)
    return ((hist1_ - hist2_) ** 2).sum(-1)  # this is faster


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
        p1_01 = vt.demodata.testdata_dummy_sift(num, rng) / pseudo_max
        p2_01 = vt.demodata.testdata_dummy_sift(num, rng) / pseudo_max
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

    print('dist_sqrd_01  = %s' % (ub.repr2(dist_sqrd_01, precision=2),))
    print('dist_sqrd_256 = %s' % (ub.repr2(dist_sqrd_256, precision=2),))
    print('dist_01       = %s' % (ub.repr2(dist_01, precision=2),))
    print('dist_256      = %s' % (ub.repr2(dist_256, precision=2),))

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

    assert ub.allsame(const)
    assert ub.allsame(const_sqrd)

    assert np.all(const == np.sqrt(const_sqrd))

    # Assert that distance conversions work
    assert np.all(dist_256 / const == dist_01)
    assert np.all(dist_sqrd_256 / const_sqrd == dist_sqrd_01)
    print('Conversions work')

    print('Maximal L2 distance between any two NON-NEGATIVE L2-NORMALIZED'
          ' vectors should always be sqrt(2)')


def L2(hist1, hist2):
    """ returns L2 (aka euclidean or standard) distance between two histograms """
    return np.sqrt(L2_sqrd(hist1, hist2))


def hist_isect(hist1, hist2):
    """ returns histogram intersection distance between two histograms """
    numer = (np.dstack([hist1, hist2])).min(-1).sum(-1)
    denom = hist2.sum(-1)
    hisect_dist = 1 - (numer / denom)
    if len(hisect_dist) == 1:
        hisect_dist = hisect_dist[0]
    return hisect_dist

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

    CommandLine:
        python -m vtool.distance --test-compute_distances

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1 = np.array([[1, 2], [2, 1], [0, 0]])
        >>> hist2 = np.array([[1, 2], [3, 1], [2, 2]])
        >>> dist_list = ['L1', 'L2']
        >>> dist_dict = compute_distances(hist1, hist2, dist_list)
        >>> result = ub.repr2(dist_dict, precision=3)
        >>> print(result)
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
    Normalized SIFT L2

    Args:
        hist1 (ndarray): Nx128 array of uint8 with pseudomax trick
        hist2 (ndarray): Nx128 array of uint8 with pseudomax trick

    CommandLine:
        python -m vtool.distance --test-bar_L2_sift

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1, hist2 = testdata_hist()
        >>> barl2_dist = bar_L2_sift(hist1, hist2)
        >>> result = ub.repr2(barl2_dist, precision=2)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)
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
        >>> sift1, sift2, sift3, sift4, sift5 = testdata_sift2()
        >>> l2_dist = L2_sift(hist1, hist2)
        >>> max_dist = L2_sift(sift4, sift5)
        >>> assert np.isclose(max_dist, 1.0)
        >>> result = ub.repr2(l2_dist, precision=2)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)
    """
    # The corret number is 512, because thats what is used in siftdesc.cpp
    # remove the pseudo max hack
    psuedo_max = 512.0
    max_l2_dist = np.sqrt(2)  # maximum L2 distance should always be sqrt 2
    sift1 = hist1.astype(TEMP_VEC_DTYPE) / psuedo_max
    sift2 = hist2.astype(TEMP_VEC_DTYPE) / psuedo_max
    l2_dist = L2(sift1, sift2)
    sift_dist = l2_dist / max_l2_dist
    return sift_dist


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
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> result = ub.repr2(l2_dist, precision=2)
        >>> print(result)
    """
    psuedo_max = 512.0
    sift1 = hist1.astype(TEMP_VEC_DTYPE) / psuedo_max
    sift2 = hist2.astype(TEMP_VEC_DTYPE) / psuedo_max
    return (sift1 * sift2).sum(-1)


def cosine_dist(hist1, hist2):
    return (hist1 * hist2).sum(-1)


def _assert_siftvec(sift):
    import vtool as vt
    assert vt.check_sift_validity(sift)


def emd(hist1, hist2, cost_matrix='sift'):
    """
    earth mover's distance by robjects(lpSovle::lp.transport)
    require: lpsolve55-5.5.0.9.win32-py2.7.exe

    CommandLine:
        python -m vtool.distance --test-emd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> hist1, hist2 = testdata_hist()
        >>> emd_dists = emd(hist1, hist2)
        >>> result = ub.repr2(emd_dists, precision=2)
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> print(result)
        np.array([ 2063.99,  2078.02,  2109.03,  2011.99,  2130.99,  2089.01,
                   2030.99,  2294.98,  2026.02,  2426.01])

    References:
        pip install pyemd
        https://github.com/andreasjansson/python-emd
        http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf
        http://stackoverflow.com/questions/15706339/compute-emd-2umpy-arrays-using-opencv
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
        >>> #xdoctest: +IGNORE_WHITESPACE
        >>> result = ('dist_matrix_list = %s' % (ut.repr3(dist_matrix_list, precision=2, with_dtype=True),))
        >>> print(result)
    """
    # FIXME; lat, lon should be different columns not different rows
    # convert decimal degrees to radians
    lat1, lon1 = np.radians(latlon1)
    lat2, lon2 = np.radians(latlon2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    EARTH_RADIUS_KM = 6367.0
    kilometers = EARTH_RADIUS_KM * c
    return kilometers


def safe_pdist(arr, *args, **kwargs):
    """
    Kwargs:
        metric = ut.absdiff

    SeeAlso:
        scipy.spatial.distance.pdist
    """
    if arr is None or len(arr) < 2:
        return None
    else:
        if len(arr.shape) == 1:
            return spdist.pdist(arr[:, None], *args, **kwargs)
        else:
            return spdist.pdist(arr, *args, **kwargs)


def pdist_indicies(num):
    return list(itertools.combinations(range(num), 2))


def pdist_argsort(x):
    """
    Sorts 2d indicies by their distnace matrix output from scipy.spatial.distance
    x = np.array([  3.05555556e-03,   1.47619797e+04,   1.47619828e+04])

    Args:
        x (ndarray):

    Returns:
        ndarray: sortx_2d

    CommandLine:
        python -m vtool.distance --test-pdist_argsort

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.distance import *  # NOQA
        >>> x = np.array([ 21695.78, 10943.76, 10941.44, 25867.64, 10752.03,
        >>>               10754.35, 4171.86, 2.32, 14923.89, 14926.2 ],
        >>>              dtype=np.float64)
        >>> sortx_2d = pdist_argsort(x)
        >>> result = ('sortx_2d = %s' % (str(sortx_2d),))
        >>> print(result)
        sortx_2d = [(2, 3), (1, 4), (1, 2), (1, 3), (0, 3), (0, 2), (2, 4), (3, 4), (0, 1), (0, 4)]
    """
    OLD = True
    #compare_idxs = [(r, c) for r, c in itertools.product(range(len(x) / 2),
    #range(len(x) / 2)) if (c > r)]
    if OLD:
        mat = spdist.squareform(x)
        matu = np.triu(mat)
        sortx_row, sortx_col = np.unravel_index(matu.ravel().argsort(), matu.shape)
        # only take where col is larger than row due to upper triu
        sortx_2d = [(r, c) for r, c in zip(sortx_row, sortx_col) if (c > r)]
    else:
        num_rows = len(x) // 2
        compare_idxs = ut.flatten([[(r, c)  for c in range(r + 1, num_rows)]
                                   for r in range(num_rows)])
        sortx = x.argsort()
        sortx_2d = ut.take(compare_idxs, sortx)
    return sortx_2d


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m vtool.distance all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
