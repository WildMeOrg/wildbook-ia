# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import vtool as vt
from vtool import clustering2 as clustertool
from ibeis.algo.hots import hstypes
(print, rrr, profile) = ut.inject2(__name__, '[smk_residuals]')


#@ut.cached_func('nonagg_rvecs', appname='smk_cachedir', key_argx=[1, 3, 4])
@profile
def compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list):
    """
    Driver function for nonagg residual computation

    Args:
        words (ndarray): array of words
        idx2_vec (dict): stacked vectors
        wx_sublist (list): words of interest
        idxs_list (list): list of idxs grouped by wx_sublist

    Returns:
        tuple : (rvecs_list, flags_list)

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals --test-compute_nonagg_rvecs:0
        python -m ibeis.algo.hots.smk.smk_residuals --test-compute_nonagg_rvecs:1

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> from ibeis.algo.hots.smk import smk_residuals
        >>> words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list = smk_debug.testdata_nonagg_rvec()
        >>> rvecs_list, flags_list = smk_residuals.compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)
        >>> print('Computed size(rvecs_list) = %r' % ut.get_object_size_str(rvecs_list))
        >>> print('Computed size(flags_list) = %r' % ut.get_object_size_str(flags_list))

    Example2:
        >>> # ENABLE_DOCTEST
        >>> # The case where vecs == words
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> vecs = (hstypes.VEC_MAX * rng.rand(4, 128)).astype(hstypes.VEC_TYPE)
        >>> word = vecs[1]
        >>> words = word.reshape(1, 128)
        >>> idx2_vec = vecs
        >>> idxs_list = [np.arange(len(vecs), dtype=np.int32)]
        >>> wx_sublist = [0]
        >>> rvecs_list, flags_list = compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)
        >>> rvecs = rvecs_list[0]
        >>> flags = flags_list[0]
        >>> maws  = (np.ones(rvecs.shape[0])).astype(hstypes.FLOAT_TYPE)
        >>> maws_list = np.array([maws])
        >>> aids_list = np.array([np.arange(len(vecs))])

    Timeit:
        %timeit [~np.any(vecs, axis=1) for vecs in vecs_list]
        %timeit [vecs.sum(axis=1) == 0 for vecs in vecs_list]

    """
    # Pick out corresonding lists of residuals and words
    words_list = [words[wx:wx + 1] for wx in wx_sublist]
    vecs_list  = [idx2_vec.take(idxs, axis=0) for idxs in idxs_list]
    # Compute nonaggregated normalized residuals
    rvecs_list = [get_norm_residuals(vecs, word)
                  for vecs, word in zip(vecs_list, words_list)]
    # Extract flags (rvecs which are all zeros) and rvecs
    flags_list = [~np.any(rvecs, axis=1) for rvecs in rvecs_list]
    return rvecs_list, flags_list


@profile
def compute_agg_rvecs(rvecs_list, idxs_list, aids_list, maws_list):
    """
    Driver function for agg residual computation

    Sums and normalizes all rvecs that belong to the same word and the same
    annotation id

    Args:
        rvecs_list (list): residual vectors grouped by word
        idxs_list (list): stacked descriptor indexes grouped by word
        aids_list (list): annotation rowid for each stacked descriptor index
        maws_list (list): multi assign weights

    Returns:
        tuple : (aggvecs_list, aggaids_list, aggidxs_list, aggmaws_list)

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals --test-compute_agg_rvecs

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> from ibeis.algo.hots.smk import smk_residuals
        >>> words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list = smk_debug.testdata_nonagg_rvec()
        >>> rvecs_list, flags_list = smk_residuals.compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)
        >>> tup = compute_agg_rvecs(rvecs_list, idxs_list, aids_list, maws_list)
        >>> aggvecs_list, aggaids_list, aggidxs_list, aggmaws_list, aggflags_list = tup
        >>> ut.assert_eq(len(wx_sublist), len(rvecs_list))

    """
    #assert len(idxs_list) == len(rvecs_list)
    # group members of each word by aid, we will collapse these groups
    grouptup_list = [clustertool.group_indices(aids) for aids in aids_list]
    # Agg aids
    aggaids_list = [tup[0] for tup in grouptup_list]
    groupxs_list = [tup[1] for tup in grouptup_list]
    # Aggregate vecs that belong to the same aid, for each word
    # (weighted aggregation with multi-assign-weights)
    aggvecs_list = [
        np.vstack([aggregate_rvecs(rvecs.take(xs, axis=0), maws.take(xs)) for xs in groupxs])
        if len(groupxs) > 0 else
        np.empty((0, hstypes.VEC_DIM), dtype=hstypes.FLOAT_TYPE)
        for rvecs, maws, groupxs in zip(rvecs_list, maws_list, groupxs_list)]
    # Agg idxs
    aggidxs_list = [[idxs.take(xs) for xs in groupxs]
                    for idxs, groupxs in zip(idxs_list, groupxs_list)]
    aggmaws_list = [np.array([maws.take(xs).prod() for xs in groupxs])
                    for maws, groupxs in zip(maws_list, groupxs_list)]
    # Need to recompute flags for consistency
    # flag is true when aggvec is all zeros
    aggflags_list = [~np.any(aggvecs, axis=1) for aggvecs in aggvecs_list]
    return aggvecs_list, aggaids_list, aggidxs_list, aggmaws_list, aggflags_list


@profile
def compress_normvec_float16(arr_float):
    """
    CURRENTLY THIS IS NOT USED. WE ARE WORKING WITH INT8 INSTEAD

    compresses 8 or 4 bytes of information into 2 bytes
    Assumes RVEC_TYPE is float16

    Args:
        arr_float (ndarray):

    Returns:
        ndarray[dtype=np.float16]

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals --test-compress_normvec_float16

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> rng = np.random.RandomState(0)
        >>> arr_float = smk_debug.get_test_float_norm_rvecs(2, 5, rng=rng)
        >>> vt.normalize_rows(arr_float, out=arr_float)
        >>> arr_float16 = compress_normvec_float16(arr_float)
        >>> result = ut.numpy_str(arr_float16, precision=4)
        >>> print(result)
        np.array([[ 0.4941,  0.1121,  0.2742,  0.6279,  0.5234],
                  [-0.6812,  0.6621, -0.1055, -0.0719,  0.2861]], dtype=np.float16)
    """
    return arr_float.astype(np.float16)


@profile
def compress_normvec_uint8(arr_float):
    """
    compresses 8 or 4 bytes of information into 1 byte
    Assumes RVEC_TYPE is int8

    Takes a normalized float vectors in range -1 to 1 with l2norm=1 and
    compresses them into 1 byte. Takes advantage of the fact that
    rarely will a component of a vector be greater than 64, so we can extend the
    range to double what normally would be allowed. This does mean there is a
    slight (but hopefully negligable) information loss. It will be negligable
    when nDims=128, when it is lower, you may want to use a different function.

    Args:
        arr_float (ndarray): normalized residual vector of type float in range -1 to 1 (with l2 norm of 1)

    Returns:
        (ndarray): residual vector of type int8 in range -128 to 128

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals --test-compress_normvec_uint8

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> rng = np.random.RandomState(0)
        >>> arr_float = smk_debug.get_test_float_norm_rvecs(2, 5, rng=rng)
        >>> vt.normalize_rows(arr_float, out=arr_float)
        >>> arr_int8 = compress_normvec_uint8(arr_float)
        >>> result = arr_int8
        >>> print(result)
        [[ 126   29   70  127  127]
         [-127  127  -27  -18   73]]
    """
    # Trick / hack: use 2 * max (psuedo_max), and clip because most components
    # will be less than 2 * max. This will reduce quantization error
    # rvec_max = 128
    # rvec_pseudo_max = rvec_max * 2 = 256
    # TODO: not sure if rounding or floor is the correct operation
    return np.clip(np.round(arr_float * 255.0), -127, 127).astype(np.int8)
    #return np.clip(np.round((arr_float * (hstypes.RVEC_PSEUDO_MAX))),
    #               hstypes.RVEC_MIN, hstypes.RVEC_MAX).astype(np.int8)


# Choose appropriate compression function based on the RVEC_TYPE
# currently its np.int8
if hstypes.RVEC_TYPE == np.float16:
    compress_normvec = compress_normvec_float16
elif hstypes.RVEC_TYPE == np.int8:
    compress_normvec = compress_normvec_uint8
else:
    raise AssertionError('unsupported RVEC_TYPE = %r' % hstypes.RVEC_TYPE)


@profile
def aggregate_rvecs(rvecs, maws):
    r"""
    helper for compute_agg_rvecs

    Args:
        rvecs (ndarray): residual vectors
        maws (ndarray): multi assign weights

    Returns:
        rvecs_agg : aggregated residual vectors

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals --test-aggregate_rvecs
        ./run_tests.py --exclude-doctest-patterns pipeline neighbor score coverage automated_helpers name automatch chip_match multi_index automated special_query scoring automated nn_weights distinctive match_chips4 query_request devcases hstypes params ibsfuncs smk_core, smk_debug control

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> rvecs = (hstypes.RVEC_MAX * rng.rand(4, 128)).astype(hstypes.RVEC_TYPE)
        >>> maws  = (rng.rand(rvecs.shape[0])).astype(hstypes.FLOAT_TYPE)
        >>> rvecs_agg = aggregate_rvecs(rvecs, maws)
        >>> result = ut.numpy_str2(rvecs_agg, linewidth=70)
        >>> print(result)
        np.array([[28, 27, 32, 16, 16, 16, 12, 31, 27, 29, 19, 27, 21, 24, 15,
                   21, 17, 37, 13, 40, 38, 33, 17, 30, 13, 23,  9, 25, 19, 15,
                   20, 17, 19, 18, 13, 25, 37, 29, 21, 16, 20, 21, 34, 11, 28,
                   19, 17, 12, 14, 24, 21, 11, 27, 11, 24, 10, 23, 20, 28, 12,
                   16, 14, 30, 22, 18, 26, 21, 20, 18,  9, 29, 20, 25, 19, 23,
                   20,  7, 13, 22, 22, 15, 20, 22, 16, 27, 10, 16, 20, 25, 25,
                   26, 28, 22, 38, 24, 16, 14, 19, 24, 14, 22, 19, 19, 33, 21,
                   22, 18, 22, 25, 25, 22, 23, 32, 16, 25, 15, 29, 21, 25, 20,
                   22, 31, 29, 24, 24, 25, 20, 14]], dtype=np.int8)

    """
    if rvecs.shape[0] == 1:
        return rvecs
    # Prealloc sum output (do not assign the result of sum)
    arr_float = np.empty((1, rvecs.shape[1]), dtype=hstypes.FLOAT_TYPE)
    # Take weighted average of multi-assigned vectors
    (maws[:, np.newaxis] * rvecs.astype(hstypes.FLOAT_TYPE)).sum(axis=0, out=arr_float[0])
    # Jegou uses mean instead. Sum should be fine because we normalize
    #rvecs.mean(axis=0, out=rvecs_agg[0])
    vt.normalize_rows(arr_float, out=arr_float)
    rvecs_agg = compress_normvec(arr_float)
    return rvecs_agg


@profile
def get_norm_residuals(vecs, word):
    """
    computes normalized residuals of vectors with respect to a word

    Args:
        vecs (ndarray):
        word (ndarray):

    Returns:
        tuple : (rvecs_n, rvec_flag)

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals --test-get_norm_residuals

    Example:
        >>> # ENABLE_DOCTEST
        >>> # The case where vecs != words
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> vecs = (hstypes.VEC_MAX * rng.rand(4, 128)).astype(hstypes.VEC_TYPE)
        >>> word = (hstypes.VEC_MAX * rng.rand(1, 128)).astype(hstypes.VEC_TYPE)
        >>> rvecs_n = get_norm_residuals(vecs, word)
        >>> result = ut.numpy_str2(rvecs_n)
        >>> print(result)

    Example:
        >>> # ENABLE_DOCTEST
        >>> # The case where vecs == words
        >>> from ibeis.algo.hots.smk.smk_residuals import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> vecs = (hstypes.VEC_MAX * rng.rand(4, 128)).astype(hstypes.VEC_TYPE)
        >>> word = vecs[1]
        >>> rvecs_n = get_norm_residuals(vecs, word)
        >>> result = ut.numpy_str2(rvecs_n)
        >>> print(result)

    IGNORE
        rvecs_agg8 = compress_normvec_uint8(arr_float)
        rvecs_agg16 = compress_normvec_float16(arr_float)
        ut.print_object_size(rvecs_agg16, 'rvecs_agg16: ')
        ut.print_object_size(rvecs_agg8,  'rvecs_agg8:  ')
        ut.print_object_size(rvec_flag,   'rvec_flag:   ')

        %timeit np.isnan(_rvec_sums)
        %timeit  _rvec_sums == 0
        %timeit  np.equal(rvec_sums, 0)
        %timeit  rvec_sums == 0
        %timeit  np.logical_or(np.isnan(_rvec_sums), _rvec_sums == 0)
    """
    # Compute residuals of assigned vectors
    #rvecs_n = word.astype(dtype=FLOAT_TYPE) - vecs.astype(dtype=FLOAT_TYPE)
    arr_float = np.subtract(word.astype(hstypes.FLOAT_TYPE), vecs.astype(hstypes.FLOAT_TYPE))
    # Faster, but doesnt work with np.norm
    #rvecs_n = np.subtract(word.view(hstypes.FLOAT_TYPE), vecs.view(hstypes.FLOAT_TYPE))
    vt.normalize_rows(arr_float, out=arr_float)
    # Mark null residuals
    #_rvec_sums = arr_float.sum(axis=1)
    #rvec_flag = np.isnan(_rvec_sums)
    # Converts normvec to a smaller type like float16 or int8
    rvecs_n = compress_normvec(arr_float)
    # IF FLOAT16 WE NEED TO FILL NANS
    # (but we should use int8, and in that case it is implicit)
    # rvecs_n = np.nan_to_num(rvecs_n)
    return rvecs_n


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.smk.smk_residuals
        python -m ibeis.algo.hots.smk.smk_residuals --allexamples
        python -m ibeis.algo.hots.smk.smk_residuals --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
