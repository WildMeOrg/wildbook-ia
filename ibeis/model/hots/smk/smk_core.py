"""
smk core
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import zip
import numpy as np
import pandas as pd
import utool
import numpy.linalg as npl
from ibeis.model.hots.smk.hstypes import FLOAT_TYPE
from ibeis.model.hots.smk import pandas_helpers as pdh
from itertools import product
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_core]')


def normalize_vecs_inplace(vecs):
    # Normalize residuals
    norm_ = npl.norm(vecs, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(vecs, norm_.reshape(norm_.size, 1), out=vecs)


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


@profile
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


def gamma_summation2(rvecs_list, weight_list, alpha=3, thresh=0):
    r"""
    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}
    """
    gamma_list = [weight * Match_N(rvecs, rvecs, alpha, thresh).sum()
                  for rvecs, weight in zip(rvecs_list, weight_list)]
    summation = sum(gamma_list)
    scoremat = np.reciprocal(np.sqrt(summation))
    return scoremat


@profile
def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


@profile
def Match_N(vecs1, vecs2, alpha=3, thresh=0):
    simmat = vecs1.dot(vecs2.T)
    # Nanvectors were equal to the cluster center.
    # This means that point was the only one in its cluster
    # Therefore it is distinctive and should have a high score
    simmat[np.isnan(simmat)] = 1.0
    return selectivity_function(simmat, alpha=alpha, thresh=thresh)


@profile
def match_kernel(wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma, invindex,
                 withinfo=True, alpha=3, thresh=0):
    """
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
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    wx2_daid   = invindex.wx2_aids
    #wx2_idxs   = invindex.wx2_idxs
    daid2_gamma = invindex.daid2_gamma

    # for each word compute the pairwise scores between matches
    print('+==============')
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))
    mark, end_ = utool.log_progress('[smk_core] query word: ', len(common_wxs),
                                    flushfreq=100, writefreq=25,
                                    with_totaltime=False)
    # Accumulate scores over the entire database
    daid2_aggscore   = pd.Series(np.zeros(len(invindex.daids)), index=invindex.daids, name='agg_score')
    daid2_totalscore = pd.Series(np.zeros(len(invindex.daids)), index=invindex.daids, name='total_score')
    # Build lists over common word indexes
    qrvec_list = [qrvecs.values for qrvecs in wx2_qrvecs[common_wxs]]
    drvec_list = [drvecs.values for drvecs in wx2_drvecs[common_wxs]]
    daids_list  = [_daids.values for _daids in wx2_daid[common_wxs]]
    weight_list = wx2_weight[common_wxs].values
    # Phi dot product
    simmat_list = [qrvecs.dot(drvecs.T)
                   for qrvecs, drvecs in zip(qrvec_list, drvec_list)]
    for simmat in simmat_list:
        simmat[np.isnan(simmat)] = 1
    # Selectivity function
    scores_iter = ((np.sign(simmat) * np.abs(simmat)) ** alpha
                   for simmat in simmat_list)
    scores_list = [scores * (scores > thresh) for scores in scores_iter]
    # Summation over query features
    wscore_list = [weight * scores.sum(axis=0)
                   for scores, weight in zip(scores_list, weight_list)]
    # Accumulate daid scores
    for wscore, daids in zip(wscore_list, daids_list):
        daid2_aggscore[daids] += wscore
        #assert len(wscore) == len(daids)
    daid2_totalscore = daid2_aggscore * daid2_gamma * query_gamma
    end_()
    print('L==============')
    if withinfo:
        daid2_chipmatch = build_daid2_chipmatch(invindex, common_wxs, wx2_qaids,
                                                wx2_qfxs, scores_list,
                                                weight_list, daids_list,
                                                query_gamma, daid2_gamma)
    else:
        daid2_chipmatch = None

    return daid2_totalscore, daid2_chipmatch


def build_daid2_chipmatch(invindex, common_wxs, wx2_qaids, wx2_qfxs,
                          scores_list, weight_list, daids_list, query_gamma,
                          daid2_gamma):
    """
    this builds the structure that the rest of the pipeline plays nice with
    """
    print('[smk_core] build chipmatch')
    #start_keys = set() set(locals().keys())
    wx2_dfxs   = invindex.wx2_fxs
    qfxs_list  = [qfxs for qfxs in wx2_qfxs[common_wxs]]
    dfxs_list  = [pdh.ensure_numpy(dfxs) for dfxs in wx2_dfxs[common_wxs]]
    qaids_list = [pdh.ensure_numpy(qaids) for qaids in wx2_qaids[common_wxs]]
    daid2_chipmatch_ = utool.ddict(list)
    _iter = list(zip(scores_list, qaids_list, daids_list, qfxs_list, dfxs_list,
                weight_list))

    #def accumulate_chipmatch(
    # Accumulate all matching indicies with scores etc...
    for scores, qaids, daids, qfxs, dfxs, weight in _iter:
        _is, _js = np.meshgrid(np.arange(scores.shape[0]), np.arange(scores.shape[1]), indexing='ij')
        for i, j in zip(_is.flat, _js.flat):
            try:
                score = scores.take(i, axis=0).take(j, axis=0)
                if score == 0:
                    continue
                qaid  = qaids[i]
                qfxs_ = qfxs[i]
                daid  = daids[j]
                dfxs_ = dfxs[j]
                # Cartesian product to list all matches that gave this score
                stackable = list(product(qfxs_, dfxs_))
                # Distribute score over all words that contributed to it.
                # apply other normalizers as well so a sum will reconstruct the
                # total score
                norm  = weight * (query_gamma * daid2_gamma[daid]) / len(stackable)
                _fm   = np.vstack(stackable)
                _fs   = np.array([score] * _fm.shape[0]) * norm
                _fk   = np.ones(_fs.shape)
                assert len(_fm) == len(_fs)
                assert len(_fk) == len(_fs)
                chipmatch_ = (_fm, _fs, _fk)
                daid2_chipmatch_[daid].append(chipmatch_)
            except Exception as ex:
                local_keys = ['score', 'qfxs_', 'dfxs_', 'qaids', 'daids',
                             '_fm', '_fs', '_fm.shape', '_fs.shape', '_fk.shape']
                utool.printex(ex, keys=local_keys, separate=True)
                raise

    # Concatenate into full fmfsfk reprs
    def concat_chipmatch(cmtup):
        fm_list = [_[0] for _ in cmtup]
        fs_list = [_[1] for _ in cmtup]
        fk_list = [_[2] for _ in cmtup]
        assert len(fm_list) == len(fs_list)
        assert len(fk_list) == len(fs_list)
        chipmatch = (np.vstack(fm_list), np.hstack(fs_list), np.hstack(fk_list))
        assert len(chipmatch[0]) == len(chipmatch[1])
        assert len(chipmatch[2]) == len(chipmatch[1])
        return chipmatch
    daid2_cattup = {daid: concat_chipmatch(cmtup) for daid, cmtup in
                    six.iteritems(daid2_chipmatch_)}
    #smk_debug.check_daid2_chipmatch(daid2_cattup)
    # Qreq needs unzipped chipmatch
    daid2_fm = {daid: cattup[0] for daid, cattup in
                six.iteritems(daid2_cattup)}
    daid2_fs = {daid: cattup[1] for daid, cattup in
                six.iteritems(daid2_cattup)}
    daid2_fk = {daid: cattup[2] for daid, cattup in
                six.iteritems(daid2_cattup)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)
    return daid2_chipmatch


import cyth
if cyth.DYNAMIC:
    exec(cyth.import_cyth_execstr(__name__))
else:
    pass
    # <AUTOGEN_CYTH>
    # </AUTOGEN_CYTH>
