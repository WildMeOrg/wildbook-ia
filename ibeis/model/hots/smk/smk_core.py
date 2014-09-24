"""
smk core
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import zip
import numpy as np
#import pandas as pd
import utool
import numpy.linalg as npl
from ibeis.model.hots.smk.hstypes import FLOAT_TYPE, INDEX_TYPE
from ibeis.model.hots.smk import pandas_helpers as pdh
from itertools import product
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_core]')


#@profile
def normalize_vecs_inplace(vecs):
    # Normalize residuals
    norm_ = npl.norm(vecs, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(vecs, norm_.reshape(norm_.size, 1), out=vecs)


#@profile
def aggregate_rvecs(rvecs):
    """
    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> rvecs = (255 * np.random.rand(4, 128)).astype(FLOAT_TYPE)
    """
    if rvecs.shape[0] == 1:
        return rvecs
    rvecs_agg = np.empty((1, rvecs.shape[1]), dtype=rvecs.dtype)
    rvecs.sum(axis=0, out=rvecs_agg[0])
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
def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


#@profile
def Match_N(vecs1, vecs2, alpha=3, thresh=0):
    simmat = vecs1.dot(vecs2.T)
    # Nanvectors were equal to the cluster center.
    # This means that point was the only one in its cluster
    # Therefore it is distinctive and should have a high score
    simmat[np.isnan(simmat)] = 1.0
    return selectivity_function(simmat, alpha=alpha, thresh=thresh)


#@profile
def gamma_summation(wx2_rvecs, wx2_weight):
    r"""
    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}

    >>> from ibeis.model.hots.smk.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> wx2_weight = invindex.wx2_weight
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_rvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> assert smk_debug.check_wx2_rvecs(wx2_rvecs)
    >>> #print(utool.dict_str(smk_debug.wx2_rvecs_stats(wx2_rvecs)))
    >>> scoremat = smk_core.gamma_summation(wx2_rvecs, wx2_weight)
    >>> print(scoremat)
    0.0384477314197
    """
    gamma_iter = [wx2_weight.get(wx, 0) * Match_N(vecs.values, vecs.values).sum()
                  for wx, vecs in six.iteritems(wx2_rvecs)]
    summation = sum(gamma_iter)
    scoremat = np.reciprocal(np.sqrt(summation))
    return scoremat


#@profile
def gamma_summation2(rvecs_list, weight_list, alpha=3, thresh=0):
    r"""
    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}
    """
    simmat_list = [rvecs.dot(rvecs.T) for rvecs in rvecs_list]  # 0.4 %
    for simmat in simmat_list:
        simmat[np.isnan(simmat)] = 1  # .2%
    # Selectivity function
    scores_iter = (np.sign(simmat) * np.power(np.abs(simmat), alpha)
                   for simmat in simmat_list)
    scores_list = [scores * (scores > thresh) for scores in scores_iter]  # 1.3%
    # Summation over query features
    score_list = [scores.sum() for scores in scores_list]
    invgamma = np.multiply(weight_list, score_list).sum()
    gamma = np.reciprocal(np.sqrt(invgamma))
    return gamma


def score_matches(qrvec_list, drvec_list, alpha, thresh):
    # Phi dot product
    simmat_list = [qrvecs.dot(drvecs.T)
                   for qrvecs, drvecs in zip(qrvec_list, drvec_list)]  # 0.4 %
    for simmat in simmat_list:
        simmat[np.isnan(simmat)] = 1  # .2%
    # Selectivity function
    scores_iter = (np.sign(simmat) * np.power(np.abs(simmat), alpha)
                   for simmat in simmat_list)
    scores_list = [scores * (scores > thresh) for scores in scores_iter]  # 1.3%
    return scores_list


@profile
def match_kernel(wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma, invindex,
                 withinfo=True, alpha=3, thresh=0):
    """
    Total time: 3.05432 s

    >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_debug
    >>> tup  = smk_debug.testdata_match_kernel()
    >>> ibs, invindex, wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma = tup
    >>> alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> withinfo = True  # takes an 11s vs 2s
    >>> _args = (wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma, invindex, withinfo, alpha, thresh)
    >>> smk_debug.invindex_dbgstr(invindex)
    >>> daid2_totalscore, daid2_wx2_scoremat = match_kernel(*_args)
    """
    #idx2_daid = invindex.idx2_daid
    #wx2_idxs   = invindex.wx2_idxs
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    wx2_daid   = invindex.wx2_aids
    daid2_gamma = invindex.daid2_gamma

    # for each word compute the pairwise scores between matches
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))  # .2
    #print('+==============')
    if utool.VERBOSE:
        mark, end_ = utool.log_progress('[smk_core] query word: ', len(common_wxs),
                                        flushfreq=100, writefreq=25,
                                        with_totaltime=False)
    #pd.Series(np.zeros(len(invindex.daids)), index=invindex.daids, name='agg_score')
    # Build lists over common word indexes
    qrvec_list  = pdh.ensure_values_subset(wx2_qrvecs, common_wxs)
    drvec_list  = pdh.ensure_values_subset(wx2_drvecs, common_wxs)
    daids_list  = pdh.ensure_values_subset(wx2_daid,   common_wxs)
    weight_list = pdh.ensure_values_subset(wx2_weight, common_wxs)
    if utool.DEBUG2:
        assert len(qrvec_list) == len(drvec_list)
        assert len(daids_list) == len(drvec_list)
        assert len(weight_list) == len(drvec_list)
        assert len(weight_list) == len(common_wxs)
    # Summation over query features
    scores_list = score_matches(qrvec_list, drvec_list, alpha, thresh)
    wscores_list = [weight * scores.sum(axis=0)
                    for scores, weight in zip(scores_list, weight_list)]
    # Accumulate daid scores
    daid2_aggscore   = utool.ddict(lambda: 0)
    # Weirdly iflatten was slower here
    for wscores, daids in zip(wscores_list, daids_list):
        for daid, wscore in zip(daids, wscores):  # 1.7%
            daid2_aggscore[daid] += wscore  # 1.8%
    daid_agg_keys   = np.array(list(daid2_aggscore.keys()))
    daid_agg_scores = np.array(list(daid2_aggscore.values()))
    daid_gamma_list = pdh.ensure_values_scalar_subset(daid2_gamma, daid_agg_keys)
    daid_total_list = np.multiply(np.multiply(daid_gamma_list, daid_agg_scores), query_gamma)
    daid2_totalscore = dict(zip(daid_agg_keys, daid_total_list))

    #assert len(wscore) == len(daids)
    #print('L==============')
    if withinfo:
        daid2_chipmatch = build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids,
                                                 wx2_qfxs, scores_list,
                                                 weight_list, daids_list,
                                                 query_gamma, daid2_gamma)  # 90.5%
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
                           scores_list, weight_list, daids_list, query_gamma,
                           daid2_gamma):
    """
    Total time: 4.43759 s
    this builds the structure that the rest of the pipeline plays nice with
    """
    # FIXME: move groupby to vtool
    from ibeis.model.hots.smk import smk_speed
    if utool.VERBOSE:
        print('[smk_core] build chipmatch')
    #qfxs_list  = [qfxs for qfxs in wx2_qfxs[common_wxs]]
    #dfxs_list  = [pdh.ensure_values(dfxs) for dfxs in wx2_dfxs[common_wxs]]
    #qaids_list = [pdh.ensure_values(qaids) for qaids in wx2_qaids[common_wxs]]
    #qaids_list = pdh.ensure_values_subset(wx2_qaids, common_wxs)
    wx2_dfxs  = invindex.wx2_fxs
    qfxs_list = pdh.ensure_values_subset(wx2_qfxs, common_wxs)
    dfxs_list = pdh.ensure_values_subset(wx2_dfxs, common_wxs)
    if isinstance(daid2_gamma, dict):
        daid2_gamma_ = daid2_gamma
    else:
        daid2_gamma_ = daid2_gamma.to_dict()

    shapes_list  = [scores.shape for scores in scores_list]  # 51us
    shape_ranges = [(mem_arange(w), mem_arange(h)) for (w, h) in shapes_list]  # 230us
    ijs_list = [mem_meshgrid(wrange, hrange) for (wrange, hrange) in shape_ranges]  # 278us
    norm_list = np.multiply(weight_list, query_gamma)
    # Normalize scores for words, nMatches, and query gamma (still need daid gamma)
    nscores_iter = (scores * norm for (scores, norm) in zip(scores_list, norm_list))

    #with utool.Timer('fsd'):
    # FIXME: Preflatten all of these lists
    out_ijs    = [list(zip(_is.flat, _js.flat)) for (_is, _js) in ijs_list]
    out_qfxs   = [[qfxs[ix] for (ix, jx) in ijs]
                          for (qfxs, ijs) in zip(qfxs_list, out_ijs)]
    out_dfxs   = [[dfxs[jx] for (ix, jx) in ijs]
                          for (dfxs, ijs) in zip(dfxs_list, out_ijs)]
    out_daids  = ([daids[jx] for (ix, jx) in ijs]
                          for (daids, ijs) in zip(daids_list, out_ijs))
    out_scores = ([nscores[ijx] for ijx in ijs]
                          for (nscores, ijs) in zip(nscores_iter, out_ijs))

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

    daid_keys, groupxs = smk_speed.group_indicies(all_daids)
    fs_list = smk_speed.apply_grouping(all_scores, groupxs)
    fm_list = smk_speed.apply_grouping(all_fms, groupxs)
    daid2_fm = {daid: fm for daid, fm in zip(daid_keys, fm_list)}
    daid2_fs = {daid: fs * daid2_gamma_[daid] for daid, fs in zip(daid_keys, fs_list)}
    daid2_fk = {daid: np.ones(fs.size) for daid, fs in zip(daid_keys, fs_list)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)

    return daid2_chipmatch


#import cyth
#if cyth.DYNAMIC:
#    exec(cyth.import_cyth_execstr(__name__))
#else:
#    pass
#    # <AUTOGEN_CYTH>
#    # </AUTOGEN_CYTH>
