#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[test_pyflann]', DEBUG=False)

"""
class FLANN
 |  Methods defined here:
 |  __del__(self)
 |  __init__(self, **kwargs)
 |
 |  build_index(self, pts, **kwargs)
 |  delete_index(self, **kwargs)
 |  add_points(self, pts, rebuild_threshold=2)
 |
 |  hierarchical_kmeans(self, pts, branch_size, num_branches,
 |                      max_iterations=None, dtype=None, **kwargs)
 |  kmeans(self, pts, num_clusters, max_iterations=None, dtype=None, **kwargs)
 |
 |  nn(self, pts, qpts, num_neighbors=1, **kwargs)
 |  nn_index(self, qpts, num_neighbors=1, **kwargs)
 |  nn_radius(self, qpts, radius, **kwargs)
 |
 |  save_index(self, filename)
 |  load_index(self, filename, pts)


# Look at /flann/algorithms/dist.h for distance clases

#distance_translation = {"euclidean"        : 1,
                        #"manhattan"        : 2,
                        #"minkowski"        : 3,
                        #"max_dist"         : 4,
                        #"hik"              : 5,
                        #"hellinger"        : 6,
                        #"chi_square"       : 7,
                        #"cs"               : 7,
                        #"kullback_leibler" : 8,
                        #"kl"               : 8,
                        #"hamming"          : 9,
                        #"hamming_lut"      : 10,
                        #"hamming_popcnt"   : 11,
                        #"l2_simple"        : 12,}

# MAKE SURE YOU EDIT index.py in pyflann

#flann_algos = {
    #'linear'        : 0,
    #'kdtree'        : 1,
    #'kmeans'        : 2,
    #'composite'     : 3,
    #'kdtree_single' : 4,
    #'hierarchical'  : 5,
    #'lsh'           : 6, # locality sensitive hashing
    #'kdtree_cuda'   : 7,
    #'saved'         : 254, # dont use
    #'autotuned'     : 255,
#}

#multikey_dists = {

    ## Huristic distances

    #('euclidian', 'l2')        :  1,
    #('manhattan', 'l1')        :  2,
    #('minkowski', 'lp')        :  3, # I guess p is the order?
    #('max_dist' , 'linf')      :  4,
    #('l2_simple')              : 12, # For low dimensional points
    #('hellinger')              :  6,

    ## Nonparametric test statistics

    #('hik','histintersect')    :  5,
    #('chi_square', 'cs')       :  7,
    ## Information-thoery divergences
    #('kullback_leibler', 'kl') :  8,
    #('hamming')                :  9, # xor and bitwise sum
    #('hamming_lut')            : 10, # xor (sums with lookup table;if nosse2)
    #('hamming_popcnt')         : 11, # population count (number of 1 bits)
#}


 #Hamming distance functor - counts the bit differences between two strings -
 #useful for the Brief descriptor
 #bit count of A exclusive XOR'ed with B

#flann_distances = {"euclidean"        : 1,
                   #"manhattan"        : 2,
                   #"minkowski"        : 3,
                   #"max_dist"         : 4,
                   #"hik"              : 5,
                   #"hellinger"        : 6,
                   #"chi_square"       : 7,
                   #"cs"               : 7,
                   #"kullback_leibler" : 8,
                   #"kl"               : 8 }

#pyflann.set_distance_type('hellinger', order=0)
"""


def test_pyflann_clustering():
    """
    >>> from vtool.tests.test_pyflann import * # NOQA
    """
    import pyflann
    import numpy as np

    # Test parameters
    nIndexed = 10000
    nDims = 11  # 128
    dtype = np.float64
    randint = np.random.randint
    print('Create random qpts and database data')
    pts  = np.array(randint(0, 255, (nIndexed, nDims)), dtype=dtype)
    print('Create flann object')
    flann = pyflann.FLANN()

    # hkmeans
    #Clusters the data by using multiple runs of kmeans to
    #recursively partition the dataset.  The number of resulting
    #clusters is given by (branch_size-1)*num_branches+1.
    #This method can be significantly faster when the number of
    #desired clusters is quite large (e.g. a hundred or more).
    #Higher branch sizes are slower but may give better results.
    #If dtype is None (the default), the array returned is the same
    #type as pts.  Otherwise, the returned array is of type dtype.
    branch_size = 5
    num_branches = 7
    print('HKmeans')
    hkmean_centroids = flann.hierarchical_kmeans(pts, branch_size, num_branches,
                                                 max_iterations=1000, dtype=None)
    print(utool.truncate_str(str(hkmean_centroids)))
    print('hkmean_centroids.shape')
    print(hkmean_centroids.shape)
    print('')
    HKMeansCentroids = (branch_size - 1) * num_branches + 1
    assert hkmean_centroids.shape[0] == HKMeansCentroids, nDims

    # kmeans
    #kmeans(self, pts, num_clusters, max_iterations=None, dtype=None, **kwargs)
        #Runs kmeans on pts with num_clusters centroids.  Returns a
        #numpy array of size num_clusters x dim.
        #If max_iterations is not None, the algorithm terminates after
        #the given number of iterations regardless of convergence.  The
        #default is to run until convergence.
        #If dtype is None (the default), the array returned is the same
        #type as pts.  Otherwise, the returned array is of type dtype.
    num_clusters = 7
    print('Kmeans')
    kmeans_centroids = flann.kmeans(pts, num_clusters, max_iterations=None,
                                    dtype=None)
    print(utool.truncate_str(str(kmeans_centroids)))
    print('kmeans_centroids.shape')
    print(kmeans_centroids.shape)
    print('')
    target = num_clusters, nDims
    assert kmeans_centroids.shape == target, repr(target)

    print('\n...done testing clustering')
    print('==============================')


def test_pyflann_add_point():
    """
    >>> from vtool.tests.test_pyflann import * # NOQA
    """
    import pyflann
    import utool
    import numpy as np

    #radius = 200000
    nQueries = 3
    num_neighbors = 5
    # Test parameters
    nIndexed = 10000
    nNewPoints = 10000
    nDims = 11  # 128
    dtype = np.float64
    randint = np.random.randint
    print('Create random qpts and database data')
    pts  = np.array(randint(0, 255, (nIndexed, nDims)), dtype=dtype)
    qpts = np.array(randint(0, 255, (nQueries, nDims)), dtype=dtype)
    newpts = np.array(randint(0, 255, (nNewPoints, nDims)), dtype=dtype)
    print('Create flann object')
    flann = pyflann.FLANN()
    #______________________________

    # build index
    print('Build Index')
    build_params = flann.build_index(pts)
    print(build_params)

    index1, dist1 = flann.nn_index(qpts, num_neighbors=num_neighbors)
    print(utool.hz_str('index1, dist1 = ', index1.T,  dist1.T))

    print('Adding points')
    flann.add_points(newpts, rebuild_threshold=2)

    print('NN_Index')
    rindex2, rdist2 = flann.nn_index(qpts, num_neighbors=num_neighbors)
    print(utool.hz_str('rindex2, rdist2 = ', rindex2,  rdist2))

    # Broken ??
    #print('NN_Radius')
    #query = qpts
    #radius_neigbhs  = flann.nn_radius(query, radius)
    #print(utool.hz_str('radius_neigbhs = ', radius_neigbhs))

    print('NN_Num')
    # Returns the num_neighbors nearest points in dataset for each point in testset.
    num_nearby      = flann.nn(pts, qpts, num_neighbors)
    print(utool.hz_str('num_nearby = ', num_nearby))

    print('\n...done testing add points')
    print('==============================')


def tune_flann(**kwargs):
    import pyflann
    import numpy as np
    nIndexed = 10000
    nQuery = 10000
    num_neighbors = 3
    nDims = 11  # 128
    dtype = np.float64
    randint = np.random.randint
    print('Create random qpts and database data')
    pts   = np.array(randint(0, 255, (nIndexed, nDims)), dtype=dtype)
    qpts  = np.array(randint(0, 255, (nQuery, nDims)), dtype=dtype)
    #num_data = len(data)
    flannkw = dict(
        algorithm='autotuned',
        target_precision=.01,
        build_weight=0.01,
        memory_weight=0.0,
        sample_fraction=0.001
    )
    flannkw.update(kwargs)
    flann = pyflann.FLANN()
    tuned_params = flann.build_index(pts, **flannkw)
    index1, dist1 = flann.nn_index(qpts, num_neighbors=num_neighbors)
    print(utool.hz_str('index1, dist1 = ', index1.T,  dist1.T))
    return tuned_params


def test_pyflann_io():
    import pyflann
    import numpy as np
    #from six.moves import range

    #alpha = range(0,128)
    #pts  = np.random.dirichlet(alpha,size=10000, dtype=np.uint8)
    #qpts = np.random.dirichlet(alpha,size=100, dtype=np.uint8)

    # Test parameters
    num_neighbors = 3
    nump = 10000
    numq = 100
    dims = 128
    dtype = np.float32

    # Create qpts and database data
    print('Create random qpts and database data')
    pts  = np.array(np.random.randint(0, 255, (nump, dims)), dtype=dtype)
    qpts = np.array(np.random.randint(0, 255, (numq, dims)), dtype=dtype)

    # Create flann object
    print('Create flann object')
    flann = pyflann.FLANN()

    # Build kd-tree index over the data
    print('Build the kd tree')
    build_params = flann.build_index(pts)  # noqa
    # Find the closest few points to num_neighbors
    print('Find nn_index nearest neighbors')
    rindex, rdist = flann.nn_index(qpts, num_neighbors=3)

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
    rindex2, rdist2 = flann2.nn_index(qpts, num_neighbors=num_neighbors)
    #print(utool.hz_str('rindex2, rdist2 = ', rindex2,  rdist2))

    print('Find the same nearest neighbors?')

    if np.all(rindex == rindex2) and np.all(rdist == rdist2):
        print('...SUCCESS!')
    else:
        print('...FAILURE!')


if __name__ == '__main__':
    """
    build_index(self, pts, **kwargs) method of pyflann.index.FLANN instance
        This builds and internally stores an index to be used for
        future nearest neighbor matchings.  It erases any previously
        stored indexes, so use multiple instances of this class to
        work with multiple stored indices.  Use nn_index(...) to find
        the nearest neighbors in this index.

        pts is a 2d numpy array or matrix. All the computation is done
        in float32 type, but pts may be any type that is convertable
        to float32.

    delete_index(self, **kwargs) method of pyflann.index.FLANN instance
        Deletes the current index freeing all the momory it uses.
        The memory used by the dataset that was indexed is not freed.

    """
    import utool
    passed = 0
    passed += not (False is utool.run_test(test_pyflann_io))
    passed += not (False is utool.run_test(test_pyflann_clustering))
    passed += not (False is utool.run_test(test_pyflann_add_point))
    print('%d/3 passed in test_pyflann' % passed)
