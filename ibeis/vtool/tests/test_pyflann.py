#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import ubelt as ub
# import utool
import numpy as np
from numpy.random import randint
try:
    import pyflann
except ImportError:
    pass
# (print, print_, printDBG, rrr, profile) = utool.inject(
#     __name__, '[test_pyflann]', DEBUG=False)

"""
remove_points does not currently have bindings
nn_radius has incorrect binindgs

class FLANN:
   __del__(self)
   __init__(self, **kwargs)

   build_index(self, pts, **kwargs)
   delete_index(self, **kwargs)
   add_points(self, pts, rebuild_threshold=2)

   hierarchical_kmeans(self, pts, branch_size, num_branches,
                       max_iterations=None, dtype=None, **kwargs)
   kmeans(self, pts, num_clusters, max_iterations=None, dtype=None, **kwargs)

   nn(self, pts, qpts, num_neighbors=1, **kwargs)
   nn_index(self, qpts, num_neighbors=1, **kwargs)
   nn_radius(self, qpts, radius, **kwargs)

   save_index(self, filename)
   load_index(self, filename, pts)

# in c++ but missing from python docs
removePoint(size_t, point_id)



# Look at /flann/algorithms/dist.h for distance clases

distance_translation = {
    "euclidean"        : 1,
    "manhattan"        : 2,
    "minkowski"        : 3,
    "max_dist"         : 4,
    "hik"              : 5,
    "hellinger"        : 6,
    "chi_square"       : 7,
    "cs"               : 7,
    "kullback_leibler" : 8,
    "kl"               : 8,
    "hamming"          : 9,
    "hamming_lut"      : 10,
    "hamming_popcnt"   : 11,
    "l2_simple"        : 12,
    }

# MAKE SURE YOU EDIT index.py in pyflann
flann_algos = {
    'linear'        : 0,
    'kdtree'        : 1,
    'kmeans'        : 2,
    'composite'     : 3,
    'kdtree_single' : 4,
    'hierarchical'  : 5,
    'lsh'           : 6, # locality sensitive hashing
    'kdtree_cuda'   : 7,
    'saved'         : 254, # dont use
    'autotuned'     : 255,
    }

multikey_dists = {
    #
    # Huristic distances
    ('euclidian', 'l2')        :  1,
    ('manhattan', 'l1')        :  2,
    ('minkowski', 'lp')        :  3, # order=p: lp could be l1, l2, l3, ...
    ('max_dist' , 'linf')      :  4,
    ('hellinger')              :  6,
    ('l2_simple')              : 12, # For low dimensional points
    #
    # Nonparametric test statistics
    ('hik','histintersect')    :  5,
    ('chi_square', 'cs')       :  7,
    #
    # Information-thoery divergences
    ('kullback_leibler', 'kl') :  8,
    ('hamming')                :  9, # xor and bitwise sum
    ('hamming_lut')            : 10, # xor (sums with lookup table;if nosse2)
    ('hamming_popcnt')         : 11, # population count (number of 1 bits)
    }

#Hamming distance functor - counts the bit differences between two strings -
#useful for the Brief descriptor
#bit count of A exclusive XOR'ed with B


pyflann.set_distance_type('hellinger', order=0)
"""


def testdata_points(nPts=53, nDims=11, dtype=np.float64):
    pts = np.array(randint(0, 255, (nPts, nDims)), dtype=dtype)
    return pts


def test_pyflann_hkmeans():
    """
    hkmeans:
        Clusters the data by using multiple runs of kmeans to
        recursively partition the dataset.  The number of resulting
        clusters is given by (branch_size-1)*num_branches+1.
        This method can be significantly faster when the number of
        desired clusters is quite large (e.g. a hundred or more).
        Higher branch sizes are slower but may give better results.
        If dtype is None (the default), the array returned is the same
        type as pts.  Otherwise, the returned array is of type dtype.

        #>>> from vtool.tests.test_pyflann import * # NOQA
        #>>> test_pyflann_hkmeans()  #doctest: +ELLIPSIS
        #HKmeans...

    CommandLine:
        python -m vtool.tests.test_pyflann --test-test_pyflann_hkmeans

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.tests.test_pyflann import *  # NOQA
        >>> result = test_pyflann_hkmeans()
        >>> print(result)
    """

    # Test parameters
    flann = pyflann.FLANN()

    branch_size = 5
    num_branches = 7
    print('HKmeans')
    pts = testdata_points(nPts=1009)
    hkmean_centroids = flann.hierarchical_kmeans(pts, branch_size, num_branches,
                                                 max_iterations=1000, dtype=None)
    # print(utool.truncate_str(str(hkmean_centroids)))
    print('hkmean_centroids.shape = %r' % (hkmean_centroids.shape,))
    nHKMeansCentroids = (branch_size - 1) * num_branches + 1
    target_shape = (nHKMeansCentroids, pts.shape[1])
    test_shape = hkmean_centroids.shape
    assert test_shape == target_shape, repr(test_shape) + ' != ' + repr(target_shape)


def test_pyflann_kmeans():
    """
    kmeans:
        (self, pts, num_clusters, max_iterations=None, dtype=None, **kwargs)
        Runs kmeans on pts with num_clusters centroids.  Returns a
        numpy array of size num_clusters x dim.
        If max_iterations is not None, the algorithm terminates after
        the given number of iterations regardless of convergence.  The
        default is to run until convergence.
        If dtype is None (the default), the array returned is the same
        type as pts.  Otherwise, the returned array is of type dtype.

    CommandLine:
        python -m vtool.tests.test_pyflann --test-test_pyflann_kmeans

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.tests.test_pyflann import *  # NOQA
        >>> result = test_pyflann_kmeans()
        >>> print(result)
    """
    print('Kmeans')
    flann = pyflann.FLANN()
    num_clusters = 7
    pts = testdata_points(nPts=1009)
    kmeans_centroids = flann.kmeans(pts, num_clusters, max_iterations=None,
                                    dtype=None)
    # print(utool.truncate_str(str(kmeans_centroids)))
    print('kmeans_centroids.shape = %r' % (kmeans_centroids.shape,))
    target_shape = (num_clusters, pts.shape[1])
    test_shape = kmeans_centroids.shape
    assert test_shape == target_shape, repr(test_shape) + ' != ' + repr(target_shape)


def test_pyflann_add_point():
    """
    CommandLine:
        python -m vtool.tests.test_pyflann --test-test_pyflann_add_point

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.tests.test_pyflann import *  # NOQA
        >>> result = test_pyflann_add_point()
        >>> print(result)
    """
    # Test parameters
    num_neighbors = 3
    pts = testdata_points(nPts=1009)
    qpts = testdata_points(nPts=7)
    newpts = testdata_points(nPts=1013)

    # build index
    print('Build Index')
    flann = pyflann.FLANN()
    _build_params = flann.build_index(pts)
    print(_build_params)

    print('NN_Index')
    indices1, dists1 = flann.nn_index(qpts, num_neighbors=num_neighbors)
    assert np.all(indices1 < pts.shape[0]), 'indicies should be less than num pts'
    print(ub.hzcat('indices1, dists1 = ', indices1,  dists1))

    print('Adding points')
    flann.add_points(newpts, rebuild_threshold=2)

    print('NN_Index')
    indices2, dists2 = flann.nn_index(qpts, num_neighbors=num_neighbors)
    print(ub.hzcat('indices2, dists2 = ', indices2,  dists2))
    assert np.any(indices2 > pts.shape[0]), 'should be some indexes into new points'
    assert np.all(indices2 < pts.shape[0] + newpts.shape[0]), 'but not more than the points being added'


def test_pyflann_searches():
    """
    CommandLine:
        python -m vtool.tests.test_pyflann --test-test_pyflann_searches

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.tests.test_pyflann import *  # NOQA
        >>> result = test_pyflann_searches()
        >>> print(result)
    """
    try:
        num_neighbors = 3
        pts = testdata_points(nPts=5743, nDims=2)
        qpts = testdata_points(nPts=7, nDims=2)
        import vtool as vt
        # sample a radius
        radius = vt.L2(pts[0:1], qpts[0:1])[0] * 2 + 1

        flann = pyflann.FLANN()

        print('NN_OnTheFly')
        # build nn_index on the fly
        indices1, dists1 = flann.nn(pts, qpts, num_neighbors, algorithm='hierarchical')
        print(ub.hzcat('indices1, dists1 = ', indices1,  dists1))

        _build_params = flann.build_index(pts, algorithm='kmeans')
        del _build_params

        print('NN_Index')
        indices2, dists2 = flann.nn_index(qpts, num_neighbors=num_neighbors)
        print(ub.hzcat('indices2, dists2 = ', indices2,  dists2))

        # this can only be called on one query point at a time
        # because the output size is unknown
        print('NN_Radius, radius=%r' % (radius,))
        indices3, dists3  = flann.nn_radius(pts[0], radius)
        print('indices3 = %r ' % (indices3,))
        print('dists3 = %r ' % (dists3,))

        assert np.all(dists3 < radius)
    except Exception as ex:
        utool.printex(ex, key_list=[
            'query',
            'query.shape',
            'pts.shape',
        ], pad_stdout=True)
        #utool.embed()
        raise


def test_pyflann_tune():
    """
    CommandLine:
        python -m vtool.tests.test_pyflann --test-test_pyflann_tune

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.tests.test_pyflann import *  # NOQA
        >>> result = test_pyflann_tune()
        >>> print(result)
    """
    print('Create random qpts and database data')
    pts = testdata_points(nPts=1009)
    qpts = testdata_points(nPts=7)
    num_neighbors = 3
    #num_data = len(data)
    # untuned query

    flann = pyflann.FLANN()
    index_untuned, dist_untuned = flann.nn(pts, qpts, num_neighbors)

    # tuned query
    flannkw = dict(
        algorithm='autotuned',
        target_precision=.01,
        build_weight=0.01,
        memory_weight=0.0,
        sample_fraction=0.001
    )
    flann_tuned = pyflann.FLANN()
    tuned_params = flann_tuned.build_index(pts, **flannkw)
    index_tuned, dist_tuned = flann_tuned.nn_index(qpts, num_neighbors=num_neighbors)

    print(ub.hzcat('index_tuned, dist_tuned     = ', index_tuned,  dist_tuned))
    print('')
    print(ub.hzcat('index_untuned, dist_untuned = ', index_untuned,  dist_untuned))

    print(dist_untuned >= dist_tuned)

    return tuned_params


def test_pyflann_io():
    """
    CommandLine:
        python -m vtool.tests.test_pyflann --test-test_pyflann_io

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.tests.test_pyflann import *  # NOQA
        >>> result = test_pyflann_io()
        >>> print(result)
    """
    # Create qpts and database data
    print('Create random qpts and database data')
    num_neighbors = 3
    nPts = 1009
    nQPts = 31
    qpts = testdata_points(nPts=nQPts)
    pts = testdata_points(nPts=nPts)

    # Create flann object
    print('Create flann object')
    flann = pyflann.FLANN()

    # Build kd-tree index over the data
    print('Build the kd tree')
    with ub.Timer('Buliding the kd-tree with %d pts' % (len(pts),)):
        _build_params = flann.build_index(pts)  # noqa

    # Find the closest few points to num_neighbors
    print('Find nn_index nearest neighbors')
    indices1, dists1 = flann.nn_index(qpts, num_neighbors=num_neighbors)

    # Save the data to disk
    print('Save the data to the disk')
    np.savez('test_pyflann_ptsdata.npz', pts)
    npload_pts = np.load('test_pyflann_ptsdata.npz')
    pts2 = npload_pts['arr_0']

    print('Save and delete the FLANN index')
    flann.save_index('test_pyflann_index.flann')
    flann.delete_index()

    print('Reload the data')
    flann2 = pyflann.FLANN()
    flann2.load_index('test_pyflann_index.flann', pts2)
    indices2, dists2 = flann2.nn_index(qpts, num_neighbors=num_neighbors)
    #print(ub.hzcat('indices2, dists2 = ', indices2,  dists2))

    print('Find the same nearest neighbors?')

    if np.all(indices1 == indices2) and np.all(dists1 == dists2):
        print('...data is the same! SUCCESS!')
    else:
        raise AssertionError('...data is the different! FAILURE!')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/vtool/vtool/tests/test_pyflann.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
