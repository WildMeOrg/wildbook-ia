from __future__ import absolute_import, division, print_function
import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
#@cython.wraparound(False)
def L2_sqrd(np.ndarray hist1, np.ndarray hist2):
    """ returns the squared L2 distance
    seealso L2
    Test:
    hist1 = np.random.rand(4, 2)
    hist2 = np.random.rand(4, 2)
    out = np.empty(hist1.shape, dtype=hist1.dtype)
    """
    return (np.abs(hist1 - hist2) ** 2).sum(-1)  # this is faster

