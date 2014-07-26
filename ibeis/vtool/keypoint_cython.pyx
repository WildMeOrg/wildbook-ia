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
def get_invVR_mats_det_float64(np.ndarray[float64_t, ndim=3] invVRs):
    # TODO: Move to ktool_cython
    cdef unsigned int nMats = invVRs.shape[0]
    # Prealloc output
    cdef np.ndarray[float64_t, ndim=1] out = np.zeros((nMats,), dtype=float64)
    cdef size_t ix
    for ix in range(nMats):
        # simple determinant: ad - bc
        out[ix] = (invVRs[ix, 0, 0] * invVRs[ix, 1, 1]) - (invVRs[ix, 0, 1] * invVRs[ix, 1, 0])
    return out
