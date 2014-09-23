"""
#if CYTH
    cdef:
        dict daid_fm, daid_fs, daid_fk
        tuple item, chipmatch
        object scoremat
        list fm_accum, fs_accum, fk_accum
        Py_ssize_t count
        np.ndarray[np.int64_t, ndim=2] fm, fm_
        np.ndarray[np.float64_t, ndim=1] fs_, fs
        np.ndarray[np.int32_t, ndim=1] fk
        np.ndarray[np.int64_t, ndim=1] qfxs
        np.ndarray[int, ndim=1] dfxs
        np.ndarray[np.int64_t, ndim=1] scoremat_column_values
        np.ndarray[np.float64_t, ndim=2] scoremat_values
        np.ndarray[np.uint8_t, cast=True] valid
        np.float64_t thresh
    #endif
"""
