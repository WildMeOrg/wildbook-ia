"""
smk core
"""
from __future__ import absolute_import, division, print_function
#import six
from six.moves import zip
from itertools import product
import utool
#import pandas as pd
import numpy as np
import scipy.sparse as spsparse
from ibeis.model.hots import hstypes
from ibeis.model.hots.smk import smk_scoring
from vtool import clustering2 as clustertool

(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_core]')

DEBUG_SMK = utool.DEBUG2 or utool.get_argflag('--debug-smk')


@profile
def accumulate_scores(dscores_list, daids_list):
    """ helper to accumulate grouped scores for database annotations """
    daid2_aggscore = utool.ddict(lambda: 0)
    ### Weirdly iflatten was slower here
    for dscores, daids in zip(dscores_list, daids_list):
        for daid, score in zip(daids, dscores):
            daid2_aggscore[daid] += score
    daid_agg_keys   = np.array(list(daid2_aggscore.keys()))
    daid_agg_scores = np.array(list(daid2_aggscore.values()))
    return daid_agg_keys, daid_agg_scores


@profile
def match_kernel_L0(qrvecs_list, drvecs_list, qflags_list, dflags_list,
                    qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list,
                    daids_list, daid2_sccw, query_sccw):
    """
    Computes smk kernels

    Args:
        qrvecs_list (list):
        drvecs_list (list):
        qflags_list (list):
        dflags_list (list):
        qmaws_list (list):
        dmaws_list (list):
        smk_alpha (float): selectivity power
        smk_thresh (float): selectivity threshold
        idf_list (list):
        daids_list (list):
        daid2_sccw (dict):
        query_sccw (float): query self-consistency-criterion

    Returns:
        retL0 : (daid2_totalscore, scores_list, daid_agg_keys,)

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> smk_debug.rrr()
        >>> core1, core2, extra = smk_debug.testdata_match_kernel_L0()
        >>> smk_alpha, smk_thresh, query_sccw, daids_list, daid2_sccw = core1
        >>> qrvecs_list, drvecs_list, qmaws_list, dmaws_list, idf_list = core2
        >>> qaid2_sccw, qaids_list = extra
        >>> retL0 = match_kernel_L0(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list, daids_list, daid2_sccw, query_sccw)
        >>> # Test Asymetric Matching
        >>> (daid2_totalscore, scores_list, daid_agg_keys,) = retL0
        >>> print(daid2_totalscore[5])
        0.336434201301
        >>> # Test Self Consistency
        >>> qret = match_kernel_L0(qrvecs_list, qrvecs_list, qmaws_list, qmaws_list, smk_alpha, smk_thresh, idf_list, qaids_list, qaid2_sccw, query_sccw)
        >>> (qaid2_totalscore, qscores_list, qaid_agg_keys,) = qret
        >>> print(qaid2_totalscore[42])
        1.0000000000000007
    """
    # Residual vector scores
    scores_list = smk_scoring.score_matches(qrvecs_list, drvecs_list,
                                            qflags_list, dflags_list,
                                            qmaws_list, dmaws_list,
                                            smk_alpha, smk_thresh,
                                            idf_list)
    # Summation over query features (resulting in scores over daids)
    dscores_list = [scores.sum(axis=0) for scores in scores_list]
    # Accumulate scores over daids (database annotation ids)
    daid_agg_keys, daid_agg_scores = accumulate_scores(dscores_list, daids_list)
    # Apply database-side sccw (self consistency criterion weight)
    daid_sccw_list = [daid2_sccw[daid] for daid in daid_agg_keys]
    # Apply query-side sccw (self consistency criterion weight )
    daid_total_list = np.multiply(np.multiply(daid_sccw_list, daid_agg_scores), query_sccw)
    # Group scores by daid using a dictionary
    daid2_totalscore = dict(zip(daid_agg_keys, daid_total_list))
    retL0 = (daid2_totalscore, scores_list, daid_agg_keys,)
    return retL0


@profile
def match_kernel_L1(qindex, invindex, qparams):
    """ Builds up information and does verbosity before going to L0 """
    # Unpack Query
    (wx2_qrvecs, wx2_qflags, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw) = qindex
    # Unpack Database
    wx2_drvecs     = invindex.wx2_drvecs
    wx2_idf        = invindex.wx2_idf
    wx2_daid       = invindex.wx2_aids
    wx2_dflags     = invindex.wx2_dflags
    daid2_sccw     = invindex.daid2_sccw

    smk_alpha  = qparams.smk_alpha
    smk_thresh = qparams.smk_thresh

    # for each word compute the pairwise scores between matches
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))
    # Build lists over common word indexes
    qrvecs_list = [ wx2_qrvecs[wx] for wx in common_wxs]
    drvecs_list = [ wx2_drvecs[wx] for wx in common_wxs]
    daids_list  = [   wx2_daid[wx] for wx in common_wxs]
    idf_list    = [    wx2_idf[wx] for wx in common_wxs]
    qmaws_list  = [  wx2_qmaws[wx] for wx in common_wxs]  # NOQA
    dflags_list = [ wx2_dflags[wx] for wx in common_wxs]  # NOQA
    qflags_list = [ wx2_qflags[wx] for wx in common_wxs]
    dmaws_list  = None
    if utool.VERBOSE:
        mark, end_ = utool.log_progress('[smk_core] query word: ', len(common_wxs),
                                        flushfreq=100, writefreq=25,
                                        with_totaltime=True)
    #--------
    retL0 = match_kernel_L0(qrvecs_list, drvecs_list, qflags_list, dflags_list,
                            qmaws_list, dmaws_list, smk_alpha, smk_thresh,
                            idf_list, daids_list, daid2_sccw, query_sccw)
    (daid2_totalscore, scores_list, daid_agg_keys) = retL0
    #print('[smk_core] Matched %d daids' % daid2_totalscore.keys())
    #utool.embed()

    retL1 = (daid2_totalscore, common_wxs, scores_list, daids_list)
    #--------
    if utool.VERBOSE:
        end_()
        print('[smk_core] Matched %d daids. nAssign=%r' %
              (len(daid2_totalscore.keys()), qparams.nAssign))
    return retL1


@profile
def match_kernel_L2(qindex, invindex, qparams, withinfo=True):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, invindex, qindex, qparams = smk_debug.testdata_match_kernel_L2()
        >>> withinfo = True  # takes an 11s vs 2s
        >>> smk_debug.rrr()
        >>> smk_debug.invindex_dbgstr(invindex)
        >>> daid2_totalscore, daid2_wx2_scoremat = match_kernel_L2(qindex, invindex, qparams, withinfo)
    """
    if DEBUG_SMK:
        from ibeis.model.hots.smk import smk_debug
        assert smk_debug.check_wx2_rvecs2(invindex), 'bad invindex'
        smk_debug.dbstr_qindex()  # UNSAFE FUNC STACK INSPECTOR
    # Unpack qindex
    # Call match kernel logic
    retL1 =  match_kernel_L1(qindex, invindex, qparams)
    # Unpack
    (daid2_totalscore, common_wxs, scores_list, daids_list)  = retL1
    if withinfo:
        # Build up chipmatch if requested TODO: Only build for a shortlist
        daid2_chipmatch = build_daid2_chipmatch3(qindex, invindex, common_wxs, scores_list, daids_list)
    else:
        daid2_chipmatch = None

    return daid2_totalscore, daid2_chipmatch


@profile
def build_daid2_chipmatch3(qindex, invindex, common_wxs, scores_list,
                           daids_list):
    """

    Args:
        invindex (InvertedIndex): object for fast vocab lookup
        common_wxs (list): list of word intersections
        wx2_qfxs (dict):
        scores_list (list):
        daids_list (list):
        query_sccw (float): query self-consistency-criterion

    Returns:
        daid2_chipmatch

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, invindex, qindex, qparams = smk_debug.testdata_match_kernel_L2(aggregate=True)
        >>> args = (qindex, invindex, qparms)
        >>> retL1 =  match_kernel_L1(*args)
        >>> (daid2_totalscore, common_wxs, scores_list, daids_list, idf_list, daid_agg_keys,)  = retL1
        >>> daid2_chipmatch_new = build_daid2_chipmatch3(invindex, common_wxs, wx2_qfxs, scores_list, daids_list, query_sccw)
        >>> daid2_chipmatch_old = build_daid2_chipmatch2(invindex, common_wxs, wx2_qfxs, scores_list, daids_list, query_sccw)
        >>> print(utool.is_dicteq(daid2_chipmatch_old[0], daid2_chipmatch_new[0]))
        >>> print(utool.is_dicteq(daid2_chipmatch_old[2], daid2_chipmatch_new[2]))
        >>> print(utool.is_dicteq(daid2_chipmatch_old[1], daid2_chipmatch_new[1]))

    Notation::
         The Format of Feature Index Lists are:
         fxs_list ~ [ ... list_per_word ... ]
         list_per_word ~ [ ... list_per_rvec ... ]
         list_per_rvec ~ [ features contributing to rvec (only one if agg=False)]
    """
    """
    CommandLine::
        python dev.py -t smk0 --allgt --db GZ_ALL --index 2:5
        python dev.py -t smk --allgt --db PZ_Mothers --index 1:3 --noqcache --va --vf

    Timeit:
        num_matches = sum(map(len, daid_nestlist))

        %timeit np.array(list(utool.iflatten(daid_nestlist)), dtype=hstypes.INDEX_TYPE)
        %timeit num_matches = sum(map(len, daid_nestlist))
        %timeit np.fromiter(utool.iflatten(daid_nestlist), hstypes.INDEX_TYPE, num_matches)

        This function is still a tiny bit slower than the other one.
        There are probably faster ways to do a few things

        %timeit build_daid2_chipmatch2(invindex, common_wxs, wx2_qfxs, scores_list, daids_list, query_sccw)
        %timeit build_daid2_chipmatch3(invindex, common_wxs, wx2_qfxs, scores_list, daids_list, query_sccw)
    """
    if utool.VERBOSE:
        print(' +--- START BUILD CHIPMATCH3')
    wx2_qfxs = qindex.wx2_qfxs
    query_sccw = qindex.query_sccw
    daid2_sccw = invindex.daid2_sccw
    wx2_dfxs = invindex.wx2_fxs
    # For each word the query feature indexes mapped to it
    qfxs_list = [wx2_qfxs[wx] for wx in common_wxs]
    # For each word the database feature indexes mapped to it
    dfxs_list = [wx2_dfxs[wx] for wx in common_wxs]
    # There are a lot of 0 scores, represent sparsely
    # 117 ms
    sparse_list = [spsparse.coo_matrix(scores) for scores in scores_list]

    if DEBUG_SMK:
        assert len(sparse_list) == len(qfxs_list), 'words to not corresond'
        assert len(dfxs_list) == len(qfxs_list), 'words to not corresond'
        assert len(daids_list) == len(qfxs_list), 'words to not corresond'
        for scores, qfxs, dfxs, daids in zip(sparse_list, qfxs_list, dfxs_list, daids_list):
            assert scores.shape == (len(qfxs), len(dfxs)), 'indicies do not correspond'
            assert len(daids) == len(dfxs), 'data indicies do not corresond'
        print('[smk_core] checked build_chipmatch input ...ok')

    # 47ms
    nest_ret = build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list)
    fm_nestlist, fs_nestlist, daid_nestlist = nest_ret

    # 7ms
    flat_ret = flatten_correspondences(fm_nestlist, fs_nestlist, daid_nestlist, query_sccw)
    all_matches, all_scores, all_daids = flat_ret

    # 3.61ms
    daid2_chipmatch = group_correspondences(all_matches, all_scores, all_daids, daid2_sccw)
    if utool.VERBOSE:
        print(' L___ END BUILD CHIPMATCH3')

    return daid2_chipmatch


@profile
def build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list):
    """ helper
    these list comprehensions replace the prevous for loop
    they still need to be optimized a little bit (and made clearer)
    can probably unnest the list comprehensions as well
    """

    """
    IGNORE
    Legacy::
        def old_build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list):
            fm_nestlist_ = []
            fs_nestlist_ = []
            daid_nestlist_ = []
            for scores, qfxs, dfxs, daids in zip(sparse_list, qfxs_list, dfxs_list, daids_list):
                for rx, cx, score in zip(scores.row, scores.col, scores.data):
                    _fm = tuple(product(qfxs[rx], dfxs[cx]))
                    _fs = [score / len(_fm)] * len(_fm)
                    _daid = [daids[cx]] * len(_fm)
                    fm_nestlist_.append(_fm)
                    fs_nestlist_.append(_fs)
                    daid_nestlist_.append(_daid)
            return fm_nestlist_, fs_nestlist_, daid_nestlist_

        oldtup_ = old_build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list)
        fm_nestlist_, fs_nestlist_, daid_nestlist_ = oldtup_
        newtup_ = build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list)
        fm_nestlist, fs_nestlist, daid_nestlist = newtup_

        assert fm_nestlist == fm_nestlist_
        assert fs_nestlist == fs_nestlist_
        assert daid_nestlist == daid_nestlist_

        47ms
        %timeit build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list)

        59ms
        %timeit old_build_correspondences(sparse_list, qfxs_list, dfxs_list, daids_list)
    IGNORE
    """
    # FIXME: rewrite double comprehension as a flat comprehension

    # Build nested feature matches (a single match might have many members)
    fm_nestlist = [
        tuple(product(qfxs[rx], dfxs[cx]))
        for scores, qfxs, dfxs in zip(sparse_list, qfxs_list, dfxs_list)
        for rx, cx in zip(scores.row, scores.col)
    ]
    nFm_list = [len(fm) for fm in fm_nestlist]
    #fs_unsplit = (score
    #              for scores in sparse_list
    #              for score in scores.data)
    #daid_unsplit = (daids[cx]
    #                for scores, daids in zip(sparse_list, daids_list)
    #                for cx in scores.col)
    # Build nested feature scores
    fs_unsplit = utool.iflatten(
        (scores.data for scores in sparse_list))
    # Build nested feature matches (a single match might have many members)
    daid_unsplit = utool.iflatten(
        (daids.take(scores.col)
         for scores, daids in zip(sparse_list, daids_list)))
    # Expand feature scores and daids splitting scores amongst match members
    fs_nestlist = [
        [score / nFm] * nFm
        for score, nFm in zip(fs_unsplit, nFm_list)
    ]
    daid_nestlist = [
        [daid] * nFm
        for daid, nFm in zip(daid_unsplit, nFm_list)
    ]

    if DEBUG_SMK:
        assert len(fm_nestlist) == len(fs_nestlist), 'inconsistent len'
        assert len(fm_nestlist) == len(nFm_list), 'inconsistent len'
        assert len(daid_nestlist) == len(fs_nestlist), 'inconsistent len'
        min_ = min(2, len(nFm_list))
        max_ = min(15, len(nFm_list))
        print('nFm_list[_min:_max]      = ' + utool.list_str(nFm_list[min_:max_]))
        print('fm_nestlist[_min:_max]   = ' + utool.list_str(fm_nestlist[min_:max_]))
        print('fs_nestlist[_min:_max]   = ' + utool.list_str(fs_nestlist[min_:max_]))
        print('daid_nestlist[_min:_max] = ' + utool.list_str(daid_nestlist[min_:max_]))
        for fm_, fs_, daid_ in zip(fm_nestlist, fs_nestlist, daid_nestlist):
            assert len(fm_) == len(fs_), 'inconsistent len'
            assert len(fm_) == len(daid_), 'inconsistent len'
        print('[smk_core] checked build_chipmatch correspondence ...ok')
    return fm_nestlist, fs_nestlist, daid_nestlist


@profile
def flatten_correspondences(fm_nestlist, fs_nestlist, daid_nestlist, query_sccw):
    """
    helper
    """
    iflat_ = utool.iflatten
    DAID_DTYPE = hstypes.INDEX_TYPE
    FS_DTYPE = hstypes.FS_DTYPE
    FM_DTYPE = hstypes.FM_DTYPE

    #_all_daids = np.array(list(utool.iflatten(daid_nestlist)), dtype=hstypes.INDEX_TYPE)
    #_all_scores = np.array(list(utool.iflatten(fs_nestlist)), dtype=hstypes.FS_DTYPE) * query_sccw
    #_all_matches = np.array(list(utool.iflatten(fm_nestlist)), dtype=hstypes.FM_DTYPE)

    #count1 = sum(map(len, daid_nestlist))
    count = sum(map(len, fs_nestlist))
    #count3 = sum(map(len, fm_nestlist))
    all_daids   = np.fromiter(iflat_(daid_nestlist), DAID_DTYPE, count)
    all_scores  = np.fromiter(iflat_(fs_nestlist), FS_DTYPE, count) * query_sccw
    # Shape hack so we can use fromiter which outputs a 1D array
    all_matches = np.fromiter(iflat_(iflat_(fm_nestlist)), FM_DTYPE, 2 * count)
    all_matches.shape = (all_matches.size / 2, 2)

    if utool.DEBUG2:
        assert len(all_daids) == len(all_scores), 'inconsistent len'
        assert len(all_matches) == len(all_scores), 'inconsistent len'
        print('[smk_core] checked build_chipmatch flatten ...ok')

    return all_matches, all_scores, all_daids


@profile
def group_correspondences(all_matches, all_scores, all_daids, daid2_sccw):
    daid_keys, groupxs = clustertool.group_indicies(all_daids)
    fs_list = clustertool.apply_grouping(all_scores, groupxs)
    fm_list = clustertool.apply_grouping(all_matches, groupxs)
    daid2_fm = {daid: fm for daid, fm in zip(daid_keys, fm_list)}
    daid2_fs = {daid: fs * daid2_sccw[daid] for daid, fs in zip(daid_keys, fs_list)}
    # FIXME: generalize to when nAssign > 1
    daid2_fk = {daid: np.ones(fs.size, dtype=hstypes.FK_DTYPE) for daid, fs in zip(daid_keys, fs_list)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)
    return daid2_chipmatch


@profile
def build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids, wx2_qfxs,
                           scores_list, daids_list, query_sccw):
    """
    Builds explicit chipmatches that the rest of the pipeline plays nice with

    Notation:
        An explicit chipmatch is a tuple (fm, fs, fk) feature_matches,
        feature_scores, and feature_ranks.

        Let N be the number of matches

        A feature match, fm{shape=(N, 2), dtype=int32}, is an array where the first
        column corresponds to query_feature_indexes (qfx) and the second column
        corresponds to database_feature_indexes (dfx).

        A feature score, fs{shape=(N,), dtype=float64} is an array of scores

        A feature rank, fk{shape=(N,), dtype=int16} is an array of ranks

    Returns:
        daid2_chipmatch (dict) : (daid2_fm, daid2_fs, daid2_fk)
        Return Format::
            daid2_fm (dict): {daid: fm, ...}
            daid2_fs (dict): {daid: fs, ...}
            daid2_fk (dict): {daid: fk, ...}

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, invindex, qindex, qparams = smk_debug.testdata_match_kernel_L2()
        >>> wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw = qindex
        >>> smk_alpha = ibs.cfg.query_cfg.smk_cfg.smk_alpha
        >>> smk_thresh = ibs.cfg.query_cfg.smk_cfg.smk_thresh
        >>> withinfo = True  # takes an 11s vs 2s
        >>> args = (wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex, withinfo, smk_alpha, smk_thresh)
        >>> retL1 =  match_kernel_L1(*args)
        >>> (daid2_totalscore, common_wxs, scores_list, daids_list, idf_list, daid_agg_keys,)  = retL1
        >>> daid2_chipmatch_old = build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids, wx2_qfxs, scores_list, daids_list, query_sccw)
        >>> daid2_chipmatch_new = build_daid2_chipmatch3(invindex, common_wxs, wx2_qaids, wx2_qfxs, scores_list, daids_list, query_sccw)
        >>> print(utool.is_dicteq(daid2_chipmatch_old[0], daid2_chipmatch_new[0]))
        >>> print(utool.is_dicteq(daid2_chipmatch_old[2], daid2_chipmatch_new[2]))
        >>> print(utool.is_dicteq(daid2_chipmatch_old[1],  daid2_chipmatch_new[1]))

    %timeit build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids, wx2_qfxs, scores_list, daids_list, query_sccw)
    %timeit build_daid2_chipmatch3(invindex, common_wxs, wx2_qaids, wx2_qfxs, scores_list, daids_list, query_sccw)
    """
    # FIXME: move groupby to vtool
    if utool.VERBOSE:
        print('[smk_core] build chipmatch')

    wx2_dfxs  = invindex.wx2_fxs
    daid2_sccw = invindex.daid2_sccw

    qfxs_list = [wx2_qfxs[wx] for wx in common_wxs]
    dfxs_list = [wx2_dfxs[wx] for wx in common_wxs]

    shapes_list  = [scores.shape for scores in scores_list]  # 51us
    shape_ranges = [(mem_arange(w), mem_arange(h)) for (w, h) in shapes_list]  # 230us
    ijs_list = [mem_meshgrid(wrange, hrange) for (wrange, hrange) in shape_ranges]  # 278us
    # Normalize scores for words, nMatches, and query sccw (still need daid sccw)
    nscores_iter = (scores * query_sccw for scores in scores_list)

    # FIXME: Preflatten all of these lists
    out_ijs = [
        list(zip(_is.flat, _js.flat))
        for (_is, _js) in ijs_list
    ]
    out_qfxs = [
        [qfxs[ix] for (ix, jx) in ijs]
        for (qfxs, ijs) in zip(qfxs_list, out_ijs)
    ]
    out_dfxs = [
        [dfxs[jx] for (ix, jx) in ijs]
        for (dfxs, ijs) in zip(dfxs_list, out_ijs)
    ]
    out_daids = (
        [daids[jx] for (ix, jx) in ijs]
        for (daids, ijs) in zip(daids_list, out_ijs)
    )
    out_scores = (
        [nscores[ijx] for ijx in ijs]
        for (nscores, ijs) in zip(nscores_iter, out_ijs)
    )
    nested_fm_iter = [
        [
            tuple(product(qfxs_, dfxs_))
            for qfxs_, dfxs_ in zip(qfxs, dfxs)
        ]
        for qfxs, dfxs in zip(out_qfxs, out_dfxs)
    ]
    all_fms = np.array(list(utool.iflatten(utool.iflatten(nested_fm_iter))), dtype=hstypes.FM_DTYPE)
    nested_nmatch_list = [[len(fm) for fm in fms] for fms in nested_fm_iter]
    nested_daid_iter = (
        [
            [daid] * nMatch
            for nMatch, daid in zip(nMatch_list, daids)
        ]
        for nMatch_list, daids in zip(nested_nmatch_list, out_daids)
    )
    nested_score_iter = (
        [
            [score / nMatch] * nMatch
            for nMatch, score in zip(nMatch_list, scores)
        ]
        for nMatch_list, scores in zip(nested_nmatch_list, out_scores)
    )
    all_daids_ = np.array(list(utool.iflatten(utool.iflatten(nested_daid_iter))), dtype=hstypes.INDEX_TYPE)
    all_fss = np.array(list(utool.iflatten(utool.iflatten(nested_score_iter))), dtype=hstypes.FS_DTYPE)

    # Filter out 0 scores
    keep_xs = np.where(all_fss > 0)[0]
    all_fss = all_fss.take(keep_xs)
    all_fms = all_fms.take(keep_xs, axis=0)
    all_daids_ = all_daids_.take(keep_xs)

    daid_keys, groupxs = clustertool.group_indicies(all_daids_)
    fs_list = clustertool.apply_grouping(all_fss, groupxs)
    fm_list = clustertool.apply_grouping(all_fms, groupxs)
    daid2_fm = {daid: fm for daid, fm in zip(daid_keys, fm_list)}
    daid2_fs = {daid: fs * daid2_sccw[daid] for daid, fs in zip(daid_keys, fs_list)}
    # FIXME: generalize to when nAssign > 1
    daid2_fk = {daid: np.ones(fs.size, dtype=hstypes.FK_DTYPE) for daid, fs in zip(daid_keys, fs_list)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)

    return daid2_chipmatch
    #if False:
    #    np.all(all_fms == all_matches)
    #    np.all(all_fss == all_scores)
    #    np.all(np.abs(all_scores - all_fss) < .00001)

    #assert len(all_daids_) == len(all_fss)
    #assert len(all_fms) == len(all_fss)
    #with utool.EmbedOnException():
    #    try:
    #    except Exception as ex:
    #        utool.printex(ex)
    #        #utool.list_depth(dfxs_list, max)
    #        #utool.list_depth(dfxs_list, min)
    #        raise

    # This code is incomprehensable. I feel ashamed.

    # Number of times to duplicate scores
    #nested_nmatch_list = [
    #    [
    #        qfxs_.size * dfxs_.size
    #        for qfxs_, dfxs_ in zip(dfxs, qfxs)
    #    ]
    #    for dfxs, qfxs in zip(out_dfxs, out_qfxs)
    #]
    #test = [[(qfxs_, dfxs_) for (qfxs_, dfxs_) in zip(qfxs, dfxs)] for qfxs, dfxs in zip(out_qfxs, out_dfxs)]
    #flatqfxs = utool.flatten([[(qfxs_, dfxs_) for (qfxs_, dfxs_) in zip(qfxs, dfxs)] for qfxs, dfxs in zip(out_qfxs, out_dfxs)])


@profile
def mem_arange(num, cache={}):
    # TODO: weakref cache
    if num not in cache:
        cache[num] = np.arange(num)
    return cache[num]


@profile
def mem_meshgrid(wrange, hrange, cache={}):
    # TODO: weakref cache
    key = (id(wrange), id(hrange))
    if key not in cache:
        cache[key] = np.meshgrid(wrange, hrange, indexing='ij')
    return cache[key]
