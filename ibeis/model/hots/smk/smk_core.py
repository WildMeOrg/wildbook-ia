"""
smk core
"""
from __future__ import absolute_import, division, print_function
#import six
from six.moves import zip
import numpy as np
#import pandas as pd
import utool
import numpy.linalg as npl
from ibeis.model.hots.smk.hstypes import FLOAT_TYPE, INDEX_TYPE
from ibeis.model.hots.smk import pandas_helpers as pdh
from itertools import product
from vtool import clustering2 as clustertool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_core]')


#@profile
def normalize_vecs_inplace(vecs):
    # Normalize residuals
    # this can easily be sped up by cyth
    norm_ = npl.norm(vecs, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(vecs, norm_.reshape(norm_.size, 1), out=vecs)


#@profile
def aggregate_rvecs(rvecs, maws):
    """
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> rvecs = (255 * np.random.rand(4, 128)).astype(FLOAT_TYPE)
    >>> rvecs = (255 * np.random.rand(4, 4)).astype(FLOAT_TYPE)
    """
    if rvecs.shape[0] == 1:
        return rvecs
    rvecs_agg = np.empty((1, rvecs.shape[1]), dtype=rvecs.dtype)
    # Take weighted average of multi-assigned vectors
    (maws[:, np.newaxis] * rvecs).sum(axis=0, out=rvecs_agg[0])
    # Jegou uses mean instead. Sum should be fine because we normalize
    #rvecs.mean(axis=0, out=rvecs_agg[0])
    normalize_vecs_inplace(rvecs_agg)
    return rvecs_agg


#@profile
def get_norm_rvecs(vecs, word):
    """
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> vecs = (255 * np.random.rand(4, 128)).astype(VEC_TYPE)
    >>> word = (255 * np.random.rand(1, 128)).astype(VEC_TYPE)
    """
    # Compute residuals of assigned vectors
    rvecs_n = word.astype(dtype=FLOAT_TYPE) - vecs.astype(dtype=FLOAT_TYPE)
    normalize_vecs_inplace(rvecs_n)
    return rvecs_n


#@profile
def selectivity_function(simmat_list, alpha, thresh):
    """ Selectivity function - sigma from SMK paper rscore = residual score """
    alpha = 3
    thresh = 0
    scores_iter = [
        np.multiply(np.sign(simmat), np.power(np.abs(simmat), alpha))
        for simmat in simmat_list
    ]
    scores_list = [np.multiply(scores, np.greater(scores, thresh))
                   for scores in scores_iter]
    if utool.DEBUG2:
        assert len(scores_list) == len(simmat_list)
    return scores_list


def apply_maws(simmat_list, qmaws_list, dmaws_list):
    """
    Applys multi-assign weights to rvec similarty matrices

    """
    if dmaws_list is None and qmaws_list is None:
        mawsim_list = simmat_list
    elif dmaws_list is not None and qmaws_list is not None:
        #mawsim_list = [qmaws.reshape((qmaws.size, 1)) * simmat * dmaws.reshape((1, dmaws.size))
        mawsim_list = [qmaws[:, np.newaxis] * simmat * dmaws[np.newaxis, :]
                       for simmat, qmaws, dmaws in
                       zip(simmat_list, qmaws_list, dmaws_list)]
    elif qmaws_list is not None and dmaws_list is None:
        #mawsim_list = [qmaws.reshape((qmaws.size, 1)) * simmat
        mawsim_list = [qmaws[:, np.newaxis] * simmat
                       for simmat, qmaws in
                       zip(simmat_list, qmaws_list)]
    else:
        #mawsim_list = [simmat * dmaws[np.newaxis, :]
        #               for simmat, dmaws in
        #               zip(simmat_list, dmaws_list)]
        raise NotImplementedError('cannot just do dmaws')
    return mawsim_list


def similarity_function(qrvecs_list, drvecs_list):
    """ Phi dot product. Accounts for NaN residual vectors
    qrvecs_list list of rvecs for each word
    """
    simmat_list = [
        qrvecs.dot(drvecs.T)
        for qrvecs, drvecs in zip(qrvecs_list, drvecs_list)
    ]
    if utool.DEBUG2:
        assert len(simmat_list) == len(qrvecs_list), 'bad simmat and qrvec'
        assert len(simmat_list) == len(drvecs_list), 'bad simmat and drvec'
    # Rvec is NaN implies it is a cluster center. perfect similarity
    for simmat in simmat_list:
        simmat[np.isnan(simmat)] = 1.0
    return simmat_list


def score_matches(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, alpha, thresh):
    """ Similarity + Selectivity: M(X_c, Y_c)

    qrvecs_list = query vectors for each word
    drvecs_list = database vectors for each word

    computes score matrix.

    """
    # Cosine similarity between normalized residuals
    simmat_list = similarity_function(qrvecs_list, drvecs_list)
    # Apply multi assign weights
    # THIS IS WRONG WHEN AGG=TRUE
    # QMAWS IS NOT COLLAPSED
    mawmat_list = apply_maws(simmat_list, qmaws_list, dmaws_list)
    # Apply sigma selectivity (power law)
    scores_list = selectivity_function(mawmat_list, alpha, thresh)
    return scores_list


#@profile
def sccw_summation(rvecs_list, idf_list, maws_list, alpha, thresh):
    r"""
    Computes gamma from "To Aggregate or not to aggregate"

    scc = self consistency criterion

    It is a scalar which ensure K(X, X) = 1

    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}

    >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> idf_list, rvecs_list, maws_list, alpha, thresh = smk_debug.testsdata_sccw_sum()
    >>> qmaws_list = dmaws_list = maws_list
    >>> drvecs_list = qrvecs_list = rvecs_list
    >>> scoremat = smk_core.sccw_summation(rvecs_list, idf_list, maws_list, alpha, thresh )
    >>> print(scoremat)
    0.0384477314197

    qrvecs_list = drvecs_list = rvecs_list
    """
    # Indexing with asymetric multi-assignment might get you a non 1 self score?
    scores_list = score_matches(rvecs_list, rvecs_list,
                                maws_list, maws_list,
                                alpha, thresh)
    # Summation over query features
    score_list = [scores.sum() for scores in scores_list]
    if utool.DEBUG2:
        assert len(scores_list) == len(rvecs_list), 'bad rvec and score'
        assert len(idf_list) == len(score_list), 'bad weight and score'
    # Apply idf weighting
    weighted_total = np.multiply(idf_list, score_list).sum()
    # Square root inverse
    sccw = np.reciprocal(np.sqrt(weighted_total))
    return sccw


@profile
def match_kernel(wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex,
                 withinfo=True, alpha=3, thresh=0):
    """

    >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, invindex, qindex = smk_debug.testdata_match_kernel()
    >>> wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw = qindex
    >>> alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> withinfo = True  # takes an 11s vs 2s
    >>> _args = (wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex, withinfo, alpha, thresh)
    >>> smk_debug.rrr()
    >>> smk_debug.invindex_dbgstr(invindex)
    >>> daid2_totalscore, daid2_wx2_scoremat = match_kernel(*_args)
    """
    #utool.embed()
    #idx2_daid = invindex.idx2_daid
    #wx2_idxs   = invindex.wx2_idxs
    #wx2_maws   = invindex.wx2_maws
    wx2_drvecs     = invindex.wx2_drvecs
    wx2_idf        = invindex.wx2_idf
    wx2_daid       = invindex.wx2_aids
    daid2_sccw     = invindex.daid2_sccw

    # for each word compute the pairwise scores between matches
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))
    #print('+==============')
    if utool.VERBOSE:
        mark, end_ = utool.log_progress('[smk_core] query word: ', len(common_wxs),
                                        flushfreq=100, writefreq=25,
                                        with_totaltime=True)
    # Build lists over common word indexes
    qrvecs_list = [wx2_qrvecs[wx] for wx in common_wxs]
    drvecs_list = [wx2_drvecs[wx] for wx in common_wxs]
    daids_list  = [  wx2_daid[wx] for wx in common_wxs]
    idf_list    = [   wx2_idf[wx] for wx in common_wxs]
    qmaws_list  = [ wx2_qmaws[wx] for wx in common_wxs]  # NOQA
    if utool.DEBUG2:
        assert len(qrvecs_list) == len(drvecs_list)
        assert len(daids_list)  == len(drvecs_list)
        assert len(idf_list) == len(drvecs_list)
        assert len(idf_list) == len(common_wxs)
        # Slow recursive check
        #assert utool.depth_profile(utool.depth_profile(qrvecs_list)) == utool.depth_profile(maws_list)
        #print(utool.mystats(utool.depth_profile(utool.depth_profile(qrvecs_list))))
        #print(utool.mystats(utool.depth_profile(utool.depth_profile(drvecs_list))))
        #print(utool.mystats(utool.depth_profile(maws_list)))
        #print(utool.mystats(utool.depth_profile(utool.depth_profile(scores_list))))

    # Summation over query features
    scores_list = score_matches(qrvecs_list, drvecs_list, qmaws_list, None, alpha, thresh)
    # Apply idf-weights
    wscores_list = [idf * (scores).sum(axis=0)
                    for scores, idf in zip(scores_list, idf_list)]
    # Accumulate scores over daids (database annotation ids)
    daid2_aggscore = utool.ddict(lambda: 0)
    ### Weirdly iflatten was slower here
    for wscores, daids in zip(wscores_list, daids_list):
        for daid, wscore in zip(daids, wscores):
            daid2_aggscore[daid] += wscore
    daid_agg_keys   = np.array(list(daid2_aggscore.keys()))
    daid_agg_scores = np.array(list(daid2_aggscore.values()))
    # Apply database sccw (self consistency criterion weight )
    daid_sccw_list = pdh.ensure_values_scalar_subset(daid2_sccw, daid_agg_keys)
    # Apply query sccw (self consistency criterion weight )
    daid_total_list = np.multiply(np.multiply(daid_sccw_list, daid_agg_scores), query_sccw)
    # Pack scores into a dictionary
    daid2_totalscore = dict(zip(daid_agg_keys, daid_total_list))

    #assert len(wscore) == len(daids)
    #print('L==============')
    if withinfo:
        daid2_chipmatch = build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids,
                                                 wx2_qfxs, scores_list,
                                                 idf_list, daids_list,
                                                 query_sccw, daid2_sccw)
    else:
        daid2_chipmatch = None

    if utool.VERBOSE:
        end_()

    return daid2_totalscore, daid2_chipmatch


def mem_arange(num, cache={}):
    if num not in cache:
        cache[num] = np.arange(num)
    return cache[num]


def mem_meshgrid(wrange, hrange, cache={}):
    key = (id(wrange), id(hrange))
    if key not in cache:
        cache[key] = np.meshgrid(wrange, hrange, indexing='ij')
    return cache[key]


@profile
def build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids, wx2_qfxs,
                           scores_list, idf_list, daids_list, query_sccw,
                           daid2_sccw):
    """
    this builds the structure that the rest of the pipeline plays nice with
    """
    # FIXME: move groupby to vtool
    if utool.VERBOSE:
        print('[smk_core] build chipmatch')
    wx2_dfxs  = invindex.wx2_fxs
    qfxs_list = [wx2_qfxs[wx] for wx in common_wxs]
    dfxs_list = [wx2_dfxs[wx] for wx in common_wxs]
    if isinstance(daid2_sccw, dict):
        daid2_sccw_ = daid2_sccw
    else:
        daid2_sccw_ = daid2_sccw.to_dict()

    shapes_list  = [scores.shape for scores in scores_list]  # 51us
    shape_ranges = [(mem_arange(w), mem_arange(h)) for (w, h) in shapes_list]  # 230us
    ijs_list = [mem_meshgrid(wrange, hrange) for (wrange, hrange) in shape_ranges]  # 278us
    norm_list = np.multiply(idf_list, query_sccw)
    # Normalize scores for words, nMatches, and query sccw (still need daid sccw)
    nscores_iter = (scores * norm for (scores, norm) in zip(scores_list, norm_list))

    #with utool.Timer('fsd'):
    # FIXME: Preflatten all of these lists
    def dbstr_qindex():
        qindex = utool.get_localvar_from_stack('qindex')
        qindex.query_sccw
        qmaws_list = [qindex.wx2_maws[wx] for wx in common_wxs]
        qaids_list  = [qindex.wx2_qaids[wx] for wx in common_wxs]
        qfxs_list   = [qindex.wx2_qfxs[wx] for wx in common_wxs]
        qrvecs_list = [qindex.wx2_qrvecs[wx] for wx in common_wxs]
        qaids_list  = [wx2_qaids[wx] for wx in common_wxs]
        print('-- max --')
        print('list_depth(qaids_list) = %d' % utool.list_depth(qaids_list, max))
        print('list_depth(qmaws_list) = %d' % utool.list_depth(qmaws_list, max))
        print('list_depth(qfxs_list) = %d' % utool.list_depth(qfxs_list, max))
        print('list_depth(qrvecs_list) = %d' % utool.list_depth(qrvecs_list, max))
        print('-- min --')
        print('list_depth(qaids_list) = %d' % utool.list_depth(qaids_list, min))
        print('list_depth(qmaws_list) = %d' % utool.list_depth(qmaws_list, min))
        print('list_depth(qfxs_list) = %d' % utool.list_depth(qfxs_list, min))
        print('list_depth(qrvecs_list) = %d' % utool.list_depth(qrvecs_list, min))
        print('-- sig --')
        print('list_depth(qaids_list) = %r' % utool.depth_profile(qaids_list))
        print('list_depth(qmaws_list) = %r' % utool.depth_profile(qmaws_list))
        print('list_depth(qfxs_list) = %r' % utool.depth_profile(qfxs_list))
        print('list_depth(qrvecs_list) = %r' % utool.depth_profile(utool.depth_profile(qrvecs_list)))
        print(qfxs_list[0:3])
        print(qaids_list[0:3])
        print(qmaws_list[0:3])
    dbstr_qindex()
    with utool.EmbedOnException():
        try:
            out_ijs    = [list(zip(_is.flat, _js.flat)) for (_is, _js) in ijs_list]
            out_qfxs   = [[qfxs[ix] for (ix, jx) in ijs]
                          for (qfxs, ijs) in zip(qfxs_list, out_ijs)]
            out_dfxs   = [[dfxs[jx] for (ix, jx) in ijs]
                          for (dfxs, ijs) in zip(dfxs_list, out_ijs)]
            out_daids  = ([daids[jx] for (ix, jx) in ijs]
                          for (daids, ijs) in zip(daids_list, out_ijs))
            out_scores = ([nscores[ijx] for ijx in ijs]
                          for (nscores, ijs) in zip(nscores_iter, out_ijs))
        except Exception as ex:
            utool.printex(ex)
            #utool.list_depth(dfxs_list, max)
            #utool.list_depth(dfxs_list, min)
            raise

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
    nested_fm_iter = [
        [
            tuple(product(qfxs_, dfxs_))
            for qfxs_, dfxs_ in zip(qfxs, dfxs)
        ]
        for qfxs, dfxs in zip(out_qfxs, out_dfxs)
    ]
    all_fms = np.array(list(utool.iflatten(utool.iflatten(nested_fm_iter))), dtype=INDEX_TYPE)
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
    all_daids = np.array(list(utool.iflatten(utool.iflatten(nested_daid_iter))), dtype=INDEX_TYPE)
    all_scores = np.array(list(utool.iflatten(utool.iflatten(nested_score_iter))), dtype=FLOAT_TYPE)
    #assert len(all_daids) == len(all_scores)
    #assert len(all_fms) == len(all_scores)

    daid_keys, groupxs = clustertool.group_indicies(all_daids)
    fs_list = clustertool.apply_grouping(all_scores, groupxs)
    fm_list = clustertool.apply_grouping(all_fms, groupxs)
    daid2_fm = {daid: fm for daid, fm in zip(daid_keys, fm_list)}
    daid2_fs = {daid: fs * daid2_sccw_[daid] for daid, fs in zip(daid_keys, fs_list)}
    daid2_fk = {daid: np.ones(fs.size) for daid, fs in zip(daid_keys, fs_list)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)

    return daid2_chipmatch
