#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import pyflann
import numpy as np

if __name__ == '__main__':
    print('==============================')
    print('____ Running test pyflann ____')

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

    #alpha = xrange(0,128)
    #pts  = np.random.dirichlet(alpha,size=10000, dtype=np.uint8)
    #qpts = np.random.dirichlet(alpha,size=100, dtype=np.uint8)

    # Test parameters
    nump = 10000
    numq = 100
    dims = 128
    dtype = np.float32

    # Create query and database data
    print('Create random query and database data')
    pts  = np.array(np.random.randint(0, 255, (nump, dims)), dtype=dtype)
    qpts = np.array(np.random.randint(0, 255, (nump, dims)), dtype=dtype)

    # Create flann object
    print('Create flann object')
    flann = pyflann.FLANN()

    # Build kd-tree index over the data
    print('Build the kd tree')
    build_params = flann.build_index(pts)
    # Find the closest few points to num_neighbors
    print('Find some nearest neighbors')
    rindex, rdist = flann.nn_index(qpts, num_neighbors=3)
    print((rindex, rdist))

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
    rindex2, rdist2 = flann2.nn_index(qpts, num_neighbors=3)

    print('Find the same nearest neighbors?')
    print((rindex2, rdist2))

    if np.all(rindex == rindex2) and np.all(rdist == rdist2):
        print('...SUCCESS!')
    else:
        print('...FAILURE!')
    print('\n...done testing pyflann')
    print('==============================')
