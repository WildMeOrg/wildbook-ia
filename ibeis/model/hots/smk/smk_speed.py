from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from ibeis.model.hots.smk import smk_core
from ibeis.model.hots.smk.hstypes import FLOAT_TYPE
from six.moves import zip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_speed]')


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
def compute_agg_rvecs(rvecs_list, idxs_list, idx2_aid):
    """
    Total time: 4.24612 s
    >>> from ibeis.model.hots.smk.smk_speed import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> words_values, wx_sublist, idxs_list, idx2_vec_values = smk_debug.testdata_nonagg_rvec()
    >>> rvecs_list = compute_nonagg_rvec_listcomp(words_values, wx_sublist, idxs_list, idx2_vec_values)

    """
    idx2_aid_values = idx2_aid.values
    assert len(idxs_list) == len(rvecs_list)
    aids_list = [idx2_aid_values.take(idxs) for idxs in idxs_list]  # 2.84 ms
    grouptup_list = [group_indicies(aids) for aids in aids_list]  # 44%
    # Agg aids
    aggaids_list = [tup[0] for tup in grouptup_list]
    groupxs_list = [tup[1] for tup in grouptup_list]
    # Agg vecs
    aggvecs_list = [
        np.vstack([smk_core.aggregate_rvecs(rvecs.take(xs, axis=0))
                   for xs in groupxs])
        if len(groupxs) > 0 else
        np.empty((0, 128), dtype=FLOAT_TYPE)
        for rvecs, groupxs in zip(rvecs_list, groupxs_list)]  # 49.8%
    # Agg idxs
    aggidxs_list = [[idxs.take(xs) for xs in groupxs]
                    for idxs, groupxs in zip(idxs_list, groupxs_list)]  # 4.2%
    return aggvecs_list, aggaids_list, aggidxs_list


@profile
def compute_nonagg_rvec_listcomp(words_values, wx_sublist, idxs_list,
                                      idx2_vec_values):
    """
     Total time: 1.29423 s
    >>> from ibeis.model.hots.smk import smk_debug
    >>> words_values, wx_sublist, idxs_list, idx2_vec_values = smk_debug.testdata_nonagg_rvec()
    PREFERED METHOD - 110ms
    %timeit words_list = [words_values[np.newaxis, wx] for wx in wx_sublist]  # 5 ms
    %timeit words_list = [words_values[wx:wx + 1] for wx in wx_sublist]  # 1.6 ms
    """
    #with utool.Timer('compute_nonagg_rvec_listcomp'):
    words_list = [words_values[wx:wx + 1] for wx in wx_sublist]  # 1 ms
    #vecs_list  = [idx2_vec_values[idxs] for idxs in idxs_list]  # 23 ms
    vecs_list  = [idx2_vec_values.take(idxs, axis=0) for idxs in idxs_list]  # 5.3 ms
    rvecs_list = [smk_core.get_norm_rvecs(vecs, word)
                  for vecs, word in zip(vecs_list, words_list)]  # 103 ms  # 90%
    return rvecs_list


@profile
def compute_nonagg_residuals_forloop(words_values, wx_sublist, idxs_list, idx2_vec_values):
    """
    OK, but slower than listcomp method - 140ms

    idxs = idxs.astype(np.int32)
    %timeit idx2_vec_values.take(idxs, axis=0)  # 1.27
    %timeit idx2_vec_values.take(idxs.astype(np.int32), axis=0)  # 1.94
    %timeit idx2_vec_values[idxs]  # 7.8

    idx2_vec_values
    """
    #with utool.Timer('compute_nonagg_residuals_forloop'):
    num = wx_sublist.size
    rvecs_list = np.empty(num, dtype=np.ndarray)
    for count, wx in enumerate(wx_sublist):
        idxs = idxs_list[count]
        vecs = idx2_vec_values[idxs]
        word = words_values[wx:wx + 1]
        rvecs_n = smk_core.get_norm_rvecs(vecs, word)
        rvecs_list[count] = rvecs_n
    return rvecs_list


@profile
def compute_nonagg_residuals_pandas(words, wx_sublist, wx2_idxs, idx2_vec):
    """
    VERY SLOW. DEBUG USE ONLY

    pico  = ps = 1E-12
    nano  = ns = 1E-9
    micro = us = 1E-6
    mili  = ns = 1E-3
    words_values = words.values
    wxlist = [wx]
    ### index test
    %timeit words_values[wx:wx + 1]      # 0.334 us
    %timeit words_values[wx, np.newaxis] # 1.05 us
    %timeit words_values[np.newaxis, wx] # 1.05 us
    %timeit words_values.take(wxlist, axis=0) # 1.6 us
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


@profile
def group_indicies2(groupids):
    """
    >>> groupids = np.array(np.random.randint(0, 4, size=100))

    #http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
    """
    # Sort items and groupids by groupid
    sortx = groupids.argsort()
    groupids_sorted = groupids[sortx]
    num_items = groupids.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, groupids.dtype)
    diff[1:(num_items)] = np.diff(groupids_sorted)
    idxs = np.where(diff > 0)[0]
    num_groups = idxs.size - 1
    # Groups are between bounding indexes
    lrx_pairs = np.vstack((idxs[0:num_groups], idxs[1:num_groups + 1])).T
    group_idxs = [sortx[lx:rx] for lx, rx in lrx_pairs]
    return group_idxs


@profile
def group_indicies(groupids):
    """
    Total time: 1.29423 s
    >>> groupids = np.array(np.random.randint(0, 4, size=100))

    #http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance
    """
    # Sort items and groupids by groupid
    sortx = groupids.argsort()  # 2.9%
    groupids_sorted = groupids[sortx]  # 3.1%
    num_items = groupids.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, groupids.dtype)  # 8.6%
    diff[1:(num_items)] = np.diff(groupids_sorted)  # 22.5%
    idxs = np.where(diff > 0)[0]  # 8.8%
    num_groups = idxs.size - 1  # 1.3%
    # Groups are between bounding indexes
    lrx_pairs = np.vstack((idxs[0:num_groups], idxs[1:num_groups + 1])).T  # 28.8%
    group_idxs = [sortx[lx:rx] for lx, rx in lrx_pairs]  # 17.5%
    # Unique group keys
    keys = groupids_sorted[idxs[0:num_groups]]  # 4.7%
    #items_sorted = items[sortx]
    #vals = [items_sorted[lx:rx] for lx, rx in lrx_pairs]
    return keys, group_idxs


@profile
def groupby(items, groupids):
    """
    >>> items    = np.array(np.arange(100))
    >>> groupids = np.array(np.random.randint(0, 4, size=100))
    >>> items = groupids
    """
    keys, group_idxs = group_indicies(groupids)
    vals = [items[idxs] for idxs in group_idxs]
    return keys, vals


@profile
def groupby_gen(items, groupids):
    """
    >>> items    = np.array(np.arange(100))
    >>> groupids = np.array(np.random.randint(0, 4, size=100))
    """
    for key, val in zip(*groupby(items, groupids)):
        yield (key, val)


@profile
def groupby_dict(items, groupids):
    # Build a dict
    grouped = {key: val for key, val in groupby_gen(items, groupids)}
    return grouped
