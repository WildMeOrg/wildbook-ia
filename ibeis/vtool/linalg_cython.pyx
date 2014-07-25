from __future__ import absolute_import, division, print_function
cimport numpy as np
cimport cython
import numpy as np
import cython


float32 = np.float32
float64 = np.float64

ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def L2_sqrd_float32(np.ndarray[float32_t, ndim=2] hist1, 
                    np.ndarray[float32_t, ndim=2] hist2):
    """ returns the squared L2 distance """
    cdef unsigned int rows = hist1.shape[0]
    cdef unsigned int cols = hist1.shape[1]
    # Prealloc output
    cdef np.ndarray[float32_t, ndim=1] out = np.zeros((rows,), dtype=float32)

    for rx in xrange(rows):
        for cx in xrange(cols):
            out[rx] += (hist1[rx, cx] - hist2[rx, cx]) ** 2
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def L2_sqrd_float64(np.ndarray[float64_t, ndim=2] hist1, 
                    np.ndarray[float64_t, ndim=2] hist2):
    """ returns the squared L2 distance """
    cdef unsigned int rows = hist1.shape[0]
    cdef unsigned int cols = hist1.shape[1]
    # Prealloc output
    cdef np.ndarray[float64_t, ndim=1] out = np.zeros((rows,), dtype=float64)

    for rx in xrange(rows):
        for cx in xrange(cols):
            out[rx] += (hist1[rx, cx] - hist2[rx, cx]) ** 2
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def det_distance_float64(np.ndarray[float64_t, ndim=1] det1,
                         np.ndarray[float64_t, ndim=1] det2):
    # TODO: Move to ktool_cython
    cdef unsigned int nDets = det1.shape[0]
    # Prealloc output
    cdef np.ndarray[float64_t, ndim=1] out = np.zeros((nDets,), dtype=float64)

    for ix in xrange(nDets):
        # simple determinant: ad - bc
        if det1[ix] > det2[ix]:
            out[ix] = det1[ix] / det2[ix]
        else:
            out[ix] = det2[ix] / det1[ix]
    return out
