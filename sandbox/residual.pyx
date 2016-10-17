"""
smk
cython --compile-args=-fopenmp --link-args=-fopenmp --force -a residual.pyx
cython residual.pyx
"""
#from __future__ import division
#cimport cython

#import numpy as np
#from cython.parallel import prange, parallel
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
#from libc.stdlib cimport malloc, calloc, free, abort
#from libc.stdio cimport puts

#import os
#import sys

#try:
#    from builtins import next # Py3k
#except ImportError:
#    def next(it):
#        return it.next()

#@cython.test_assert_path_exists(
#    "//ParallelWithBlockNode//ParallelRangeNode[@schedule = 'dynamic']",
#    "//GILStatNode[@state = 'nogil]//ParallelRangeNode")
#def test_prange():
#    """
#    >>> test_prange()
#    (9, 9, 45, 45)
#    """
#    cdef Py_ssize_t i, j, sum1 = 0, sum2 = 0

#    with nogil, cython.parallel.parallel():
#        for i in prange(10, schedule='dynamic'):
#            sum1 += i

#    for j in prange(10, nogil=True):
#        sum2 += j

#    return i, j, sum1, sum2


cimport cython.parallel
cimport numpy as np
#from cython.parallel import prange, threadid
from cython.parallel import prange


cdef extern:
    void c_multiply_dbl(double* array, double multiplier, int m, int n)
    void c_multiply_flt(float* array, double multiplier, int m, int n)

ctypedef np.int_t int_t
ctypedef np.ndarray nd

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _par_agg_phi_worker_int(
    nd[int_t, ndim=1] wx_list, 
    nd[int_t, ndim=2] word_list, 
    fxs_list, 
    maws_list, 
    nd[int_t, ndim=2] fx_to_vecs,
    int int_rvec
):
    """
    https://github.com/cmccully/lacosmicx/blob/master/lacosmicx/_lacosmicx.pyx
    http://stackoverflow.com/questions/17811958/iterating-over-a-list-in-parallel-with-cython
    http://docs.cython.org/en/latest/src/userguide/parallelism.html
    http://kbarbary.github.io/blog/cython-and-multiple-numpy-dtypes/
    """
    pass
    #wx_list, word_list, fxs_list, maws_list, fx_to_vecs, int_rvec = argtup

    ##agg_rvecs = np.empty((len(wx_list), fx_to_vecs.shape[1]), dtype=np.float)

    cdef Py_ssize_t idx
    cdef Py_ssize_t num_words = wx_list.shape[0]
    cdef Py_ssize_t feat_dim = fx_to_vecs.shape[1]

    cdef np.uint8_t[:, :] buffer
    buffer = input.view(np.uint8)

    cdef nd[np.int8_t, ndim=2] agg_rvecs = np.zeros((num_words, feat_dim), dtype=np.int8)
    cdef nd[np.int8_t, ndim=2] agg_flags = np.empty((num_words, 1), dtype=np.np.int8)

    #cdef np.ndarray[np.float_t, ndim=2] _agg_rvecs
    #cdef np.ndarray[np.int8_t, ndim=2] _agg_flags
    #cdef np.ndarray[np.float_t, ndim=1] word

    #with nogil:
    for idx in range(num_words):
        #with gil:
        word = word_list[idx]
        fxs = fxs_list[idx]
        maws = maws_list[idx]
        vecs = fx_to_vecs.take(fxs, axis=0)

        rvecs = np.subtract(word.astype(np.float), vecs.astype(np.float))
        # If a vec is a word then the residual is 0 and it cant be L2 noramlized.
        is_zero = np.all(rvecs == 0, 1)

        norm_ = np.sqrt((rvecs.astype(np.float) ** 2).sum(1))
        rvecs = np.divide(rvecs, norm_, rvecs)

        # reset these values back to zero
        #with gil:
        if np.any(is_zero):
            rvecs[is_zero, :] = 0
        # Determine if any errors occurred
        # FIXME: zero will drive the score of a match to 0 even though if they
        # are both 0, then it is an exact match and should be scored as a 1.
        error_flags = is_zero
                            
#        #_agg_rvecs, _agg_flags = aggregate_rvecs(_rvecs, maws, _flags)
#        # Cast to integers for storage
#        #if int_rvec:
#        #_agg_rvecs = cast_residual_integer(_agg_rvecs)
#        #agg_rvecs[idx] = _agg_rvecs[0]
#        #agg_flags[idx] = _agg_flags[0]

#    #fxs_list = ut.take(wx_to_fxs, wx_list)
#    #maws_list = ut.take(wx_to_maws, wx_list)

#    #tup = (wx_list, fxs_list, maws_list, agg_rvecs, agg_flags)
#    #return tup


#cdef cast_residual_integer(np.ndarray[np.float_t, ndim=2] rvecs):
#    # same trunctation hack as in SIFT. values will typically not reach the
#    # maximum, so we can multiply by a higher number for better fidelity.
#    cdef np.ndarray[np.int8_t, ndim=2] out
#    rvecs = np.clip(np.round(rvecs * 255.0), -127, 127)
#    out = rvecs.astype(np.int8)
#    return out
#    #return np.clip(np.round(rvecs * 255.0), -127, 127).astype(np.int8)


#cdef uncast_residual_integer(rvecs):
#    return rvecs.astype(np.float) / 255.0


cdef compute_rvec(vecs, word):
    """
    Compute residual vectors phi(x_c)

    Subtract each vector from its quantized word to get the resiudal, then
    normalize residuals to unit length.
    """
    rvecs = np.subtract(word.astype(np.float), vecs.astype(np.float))
    # If a vec is a word then the residual is 0 and it cant be L2 noramlized.
    is_zero = np.all(rvecs == 0, axis=1)

    norm_ = np.sqrt((rvecs.astype(np.float) ** 2).sum(axis=1))
    rvecs = np.divide(rvecs, norm_[:, None], out=rvecs)
    
    # reset these values back to zero
    if np.any(is_zero):
        rvecs[is_zero, :] = 0
    # Determine if any errors occurred
    # FIXME: zero will drive the score of a match to 0 even though if they
    # are both 0, then it is an exact match and should be scored as a 1.
    error_flags = is_zero
#    return rvecs, error_flags


#cdef aggregate_rvecs(rvecs, maws, error_flags):
#    r"""
#    Compute aggregated residual vectors Phi(X_c)
#    """
#    # Propogate errors from previous step
#    cdef np.ndarray[np.float_t, ndim=2] rvecs_agg
#    cdef np.ndarray[np.int8_t, ndim=2] flags_agg

#    flags_agg = np.any(error_flags, axis=0, keepdims=True)
#    if rvecs.shape[0] == 0:
#        rvecs_agg = np.empty((0, rvecs.shape[1]), dtype=np.float)
#    if rvecs.shape[0] == 1:
#        rvecs_agg = rvecs
#    else:
#        # Prealloc sum output (do not assign the result of sum)
#        rvecs_agg = np.empty((1, rvecs.shape[1]), dtype=np.float)
#        out = rvecs_agg[0]
#        # Take weighted average of multi-assigned vectors
#        weighted_sum = (maws[:, None] * rvecs).sum(axis=0, out=out)
#        total_weight = maws.sum()
#        is_zero = np.all(rvecs_agg == 0, axis=1)
#        rvecs_agg = np.divide(weighted_sum, total_weight, out=rvecs_agg)

#        norm_ = np.sqrt((rvecs_agg.astype(np.float) ** 2).sum(axis=1))
#        rvecs_agg = np.divide(rvecs_agg, norm_[:, None], out=rvecs_agg)
        
#        if np.any(is_zero):
#            # Add in errors from this step
#            rvecs_agg[is_zero, :] = 0
#            flags_agg[is_zero] = True
#    return rvecs_agg, flags_agg
