from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import numpy.linalg as npl
import numpy as np
from vtool import clustering2 as clustertool
from ibeis.model.hots import hstypes
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_residuals]')


#@utool.cached_func('nonagg_rvecs', appname='smk_cachedir', key_argx=[1, 3, 4])
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

    Example:
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> from ibeis.model.hots.smk import smk_residuals
        >>> words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list = smk_debug.testdata_nonagg_rvec()
        >>> rvecs_list, flags_list = smk_residuals.compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)
        >>> print('Computed size(rvecs_list) = %r' % utool.get_object_size_str(rvecs_list))
        >>> print('Computed size(flags_list) = %r' % utool.get_object_size_str(flags_list))

    Example2:
        >>> # The case where vecs == words
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> vecs = (hstypes.VEC_MAX * np.random.rand(4, 128)).astype(hstypes.VEC_TYPE)
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

    Example:
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> from ibeis.model.hots.smk import smk_residuals
        >>> words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list = smk_debug.testdata_nonagg_rvec()
        >>> rvecs_list, flags_list = smk_residuals.compute_nonagg_rvecs(words, idx2_vec, wx_sublist, idxs_list)

    """
    #assert len(idxs_list) == len(rvecs_list)
    # group members of each word by aid, we will collapse these groups
    grouptup_list = [clustertool.group_indicies(aids) for aids in aids_list]
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

    Example:
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> np.random.seed(0)
        >>> arr_float = smk_debug.get_test_float_norm_rvecs(2, 5)
        >>> normalize_vecs2d_inplace(arr_float)
        >>> arr_float16 = compress_normvec_float16(arr_float)
        >>> print(arr_float16)
        [[ 0.49414062  0.11212158  0.27416992  0.62792969  0.5234375 ]
         [-0.68115234  0.66210938 -0.10546875 -0.07189941  0.28613281]]
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

    Example:
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> np.random.seed(0)
        >>> arr_float = smk_debug.get_test_float_norm_rvecs(2, 5)
        >>> normalize_vecs2d_inplace(arr_float)
        >>> arr_int8 = compress_normvec_uint8(arr_float)
        >>> print(arr_int8)
        [[ 127   29   70 -128 -128]
         [-128 -128  -27  -18   73]]
    """
    # Trick / hack: use 2 * max (psuedo_max), and clip because most components
    # will be less than 2 * max. This will reduce quantization error
    # rvec_max = 128
    # rvec_pseudo_max = rvec_max * 2 = 256
    # TODO: not sure if rounding or floor is the correct operation
    return np.clip(np.round(arr_float * 256.0), -128, 128).astype(np.int8)
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
def normalize_vecs2d_inplace(vecs):
    """
    Args:
        vecs (ndarray): row vectors to normalize in place

    Example:
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> import numpy as np
        >>> vecs = np.random.rand(1, 10)
        >>> normalize_vecs2d_inplace(vecs)
        >>> print(vecs)
        >>> vecs = np.random.rand(10)
        >>> normalize_vecs2d_inplace(vecs)
        >>> print(vecs)
    """
    if utool.DEBUG2:
        try:
            assert vecs.ndim == 2, 'vecs.shape = %r' % (vecs.shape,)
        except AssertionError as ex:
            ut.printex(ex, keys=['vecs'])
            raise
    # Normalize residuals
    # this can easily be sped up by cyth
    norm_ = npl.norm(vecs, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(vecs, norm_.reshape(norm_.size, 1), out=vecs)


@profile
def aggregate_rvecs(rvecs, maws):
    """
    helper for compute_agg_rvecs

    Args:
        rvecs (ndarray): residual vectors
        maws (ndarray): multi assign weights

    Returns:
        rvecs_agg : aggregated residual vectors

    Example:
        >>> #rvecs = (hstypes.RVEC_MAX * np.random.rand(4, 4)).astype(hstypes.RVEC_TYPE)
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> rvecs = (hstypes.RVEC_MAX * np.random.rand(4, 128)).astype(hstypes.RVEC_TYPE)
        >>> maws  = (np.random.rand(rvecs.shape[0])).astype(hstypes.FLOAT_TYPE)
    """
    if rvecs.shape[0] == 1:
        return rvecs
    # Prealloc sum output (do not assign the result of sum)
    arr_float = np.empty((1, rvecs.shape[1]), dtype=hstypes.FLOAT_TYPE)
    # Take weighted average of multi-assigned vectors
    (maws[:, np.newaxis] * rvecs.astype(hstypes.FLOAT_TYPE)).sum(axis=0, out=arr_float[0])
    # Jegou uses mean instead. Sum should be fine because we normalize
    #rvecs.mean(axis=0, out=rvecs_agg[0])
    normalize_vecs2d_inplace(arr_float)
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

    Example:
        >>> # The case where vecs != words
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> vecs = (hstypes.VEC_MAX * np.random.rand(4, 128)).astype(hstypes.VEC_TYPE)
        >>> word = (hstypes.VEC_MAX * np.random.rand(1, 128)).astype(hstypes.VEC_TYPE)

    Example:
        >>> # The case where vecs == words
        >>> from ibeis.model.hots.smk.smk_residuals import *  # NOQA
        >>> vecs = (hstypes.VEC_MAX * np.random.rand(4, 128)).astype(hstypes.VEC_TYPE)
        >>> word = vecs[1]

    IGNORE
        rvecs_agg8 = compress_normvec_uint8(arr_float)
        rvecs_agg16 = compress_normvec_float16(arr_float)
        utool.print_object_size(rvecs_agg16, 'rvecs_agg16: ')
        utool.print_object_size(rvecs_agg8,  'rvecs_agg8:  ')
        utool.print_object_size(rvec_flag,   'rvec_flag:   ')

        %timeit np.isnan(_rvec_sums)
        %timeit  _rvec_sums == 0
        %timeit  np.equal(rvec_sums, 0)
        %timeit  rvec_sums == 0
        %timeit  np.logical_or(np.isnan(_rvec_sums), _rvec_sums == 0)
    IGNORE
    """
    # Compute residuals of assigned vectors
    #rvecs_n = word.astype(dtype=FLOAT_TYPE) - vecs.astype(dtype=FLOAT_TYPE)
    arr_float = np.subtract(word.astype(hstypes.FLOAT_TYPE), vecs.astype(hstypes.FLOAT_TYPE))
    # Faster, but doesnt work with np.norm
    #rvecs_n = np.subtract(word.view(hstypes.FLOAT_TYPE), vecs.view(hstypes.FLOAT_TYPE))
    normalize_vecs2d_inplace(arr_float)
    # Mark null residuals
    #_rvec_sums = arr_float.sum(axis=1)
    #rvec_flag = np.isnan(_rvec_sums)
    # Converts normvec to a smaller type like float16 or int8
    rvecs_n = compress_normvec(arr_float)
    # IF FLOAT16 WE NEED TO FILL NANS
    # (but we should use int8, and in that case it is implicit)
    # rvecs_n = np.nan_to_num(rvecs_n)
    return rvecs_n
