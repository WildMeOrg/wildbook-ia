from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from ibeis.model.hots.smk import smk_core
from ibeis.model.hots.hstypes import FLOAT_TYPE, VEC_DIM
from vtool import clustering2 as clustertool
from six.moves import zip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_speed]')


@profile
def compute_agg_rvecs(rvecs_list, idxs_list, aids_list, maws_list):
    """
    Sums and normalizes all rvecs that belong to the same word and the same
    annotation id

    Example:
        >>> from ibeis.model.hots.smk.smk_speed import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list = smk_debug.testdata_nonagg_rvec()
        >>> rvecs_list = compute_nonagg_rvec_listcomp(words, wx_sublist, idxs_list, idx2_vec)

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
        np.vstack([smk_core.aggregate_rvecs(rvecs.take(xs, axis=0), maws.take(xs)) for xs in groupxs])
        if len(groupxs) > 0 else
        np.empty((0, VEC_DIM), dtype=FLOAT_TYPE)
        for rvecs, maws, groupxs in zip(rvecs_list, maws_list, groupxs_list)]
    # Agg idxs
    aggidxs_list = [[idxs.take(xs) for xs in groupxs]
                    for idxs, groupxs in zip(idxs_list, groupxs_list)]
    aggmaws_list = [np.array([maws.take(xs).prod() for xs in groupxs])
                    for maws, groupxs in zip(maws_list, groupxs_list)]
    return aggvecs_list, aggaids_list, aggidxs_list, aggmaws_list


#def group_and_aggregate(rvecs, aids):
#    """
#    assumes rvecs are all from the same word
#    Returns aggregated vecs, the aids they came from, and the invertable feature
#    map

#    >>> from ibeis.model.hots.smk.smk_speed import *
#    >>> rvecs = np.random.rand(5, 128) * 255
#    >>> aids  = np.array([1, 1, 2, 3, 2])
#    """
#    assert len(aids) == len(rvecs)
#    if len(aids) == 0:
#        group_aids = np.empty((0), dtype=INTEGER_TYPE)
#        groupxs = np.empty((0), dtype=INTEGER_TYPE)
#        group_aggvecs = np.empty((0, 128), dtype=FLOAT_TYPE)
#        return group_aids, group_aggvecs
#    else:
#        group_aids, groupxs = group_indicies(aids)  # 35us
#        group_vecs = [rvecs.take(xs, axis=0) for xs in groupxs]
#        aggvec_list = [smk_core.aggregate_rvecs(vecs) for vecs in group_vecs]  # 25us
#        group_aggvecs = np.vstack(aggvec_list)  # 6.53us
#        return group_aids, group_aggvecs, groupxs
#    #with utool.Timer('tew'):
#    #    agg_list = [group_and_aggregate(rvecs, aids)
#    #                for rvecs, aids in zip(rvecs_list, aids_list)]  # 233 ms


@profile
def compute_nonagg_rvec_listcomp(words, wx_sublist, idxs_list, idx2_vec):
    """
    PREFERED METHOD - 110ms

    Example:
        >>> from ibeis.model.hots.smk import smk_debug
        >>> words, wx_sublist, aids_list, idxs_list, idx2_vec, maws_list = smk_debug.testdata_nonagg_rvec()

    Timeit:
        %timeit words_list = [words[np.newaxis, wx] for wx in wx_sublist]  # 5 ms
        %timeit words_list = [words[wx:wx + 1] for wx in wx_sublist]  # 1.6 ms
    """
    #with utool.Timer('compute_nonagg_rvec_listcomp'):
    #vecs_list  = [idx2_vec[idxs] for idxs in idxs_list]  # 23 ms
    words_list = [words[wx:wx + 1] for wx in wx_sublist]  # 1 ms
    vecs_list  = [idx2_vec.take(idxs, axis=0) for idxs in idxs_list]  # 5.3 ms
    rvecs_list = [smk_core.get_norm_rvecs(vecs, word)
                  for vecs, word in zip(vecs_list, words_list)]  # 103 ms  # 90%
    return rvecs_list


def compute_nonagg_residuals_forloop(words, wx_sublist, idxs_list, idx2_vec):
    """
    OK, but slower than listcomp method - 140ms

    Timeit:
        idxs = idxs.astype(np.int32)
        %timeit idx2_vec.take(idxs, axis=0)  # 1.27
        %timeit idx2_vec.take(idxs.astype(np.int32), axis=0)  # 1.94
        %timeit idx2_vec[idxs]  # 7.8
    """
    #with utool.Timer('compute_nonagg_residuals_forloop'):
    num = wx_sublist.size
    rvecs_list = np.empty(num, dtype=np.ndarray)
    for count, wx in enumerate(wx_sublist):
        idxs = idxs_list[count]
        vecs = idx2_vec[idxs]
        word = words[wx:wx + 1]
        rvecs_n = smk_core.get_norm_rvecs(vecs, word)
        rvecs_list[count] = rvecs_n
    return rvecs_list


def compute_nonagg_residuals_pandas(words, wx_sublist, wx2_idxs, idx2_vec):
    """
    VERY SLOW. DEBUG USE ONLY

    Ignore:
        words = words.values
        wxlist = [wx]
        ### index test
        %timeit words[wx:wx + 1]      # 0.334 us
        %timeit words[wx, np.newaxis] # 1.05 us
        %timeit words[np.newaxis, wx] # 1.05 us
        %timeit words.take(wxlist, axis=0) # 1.6 us
        ### pandas test
        %timeit words.values[wx:wx + 1]      # 7.6 us
        %timeit words[wx:wx + 1].values      # 84.9 us
    """
    #with utool.Timer('compute_nonagg_residuals_pandas'):
    #mark, end_ = utool.log_progress('compute residual: ', len(wx_sublist), flushfreq=500, writefreq=50)
    num = wx_sublist.size
    rvecs_arr = np.empty(num, dtype=np.ndarray)
    # Compute Residuals
    for count, wx in enumerate(wx_sublist):
        #mark(count)
        idxs = wx2_idxs[wx].values
        vecs = idx2_vec.take(idxs).values
        word = words.values[wx:wx + 1]
        rvecs_n = smk_core.get_norm_rvecs(vecs, word)
        rvecs_arr[count] = rvecs_n
    return rvecs_arr
