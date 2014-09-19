"""
smk core
"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import pandas as pd
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_core]')


@profile
def gamma_summation(wx2_rvecs, wx2_weight):
    r"""
    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}

    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> from ibeis.model.hots import smk_debug
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


@profile
def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


@profile
def Match_N(vecs1, vecs2):
    simmat = vecs1.dot(vecs2.T)
    # Nanvectors were equal to the cluster center.
    # This means that point was the only one in its cluster
    # Therefore it is distinctive and should have a high score
    simmat[np.isnan(simmat)] = 1.0
    return selectivity_function(simmat)


@profile
def match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid, withinfo=True, alpha=3,
                 thresh=0):
    """
    >>> from ibeis.model.hots.smk_core import *  # NOQA
    >>> from ibeis.model.hots.smk_index import *  # NOQA
    >>> from ibeis.model.hots import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> withinfo = True  # takes an 11s vs 2s
    >>> daid2_totalscore, daid2_wx2_scoremat = match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid, withinfo=withinfo)
    """
    _daids = invindex.daids
    idx2_daid = invindex.idx2_daid
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    daid2_gamma = invindex.daid2_gamma

    wx2_rvecs = wx2_qrvecs
    query_gamma = gamma_summation(wx2_rvecs, wx2_weight)
    assert query_gamma > 0, 'query gamma is not positive!'

    # Accumulate scores over the entire database
    daid2_aggscore = pd.Series(np.zeros(len(_daids)), index=_daids, name='total_score')
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))

    if withinfo:
        daid2_wx2_scoremat = utool.ddict(dict)

    # for each word compute the pairwise scores between matches
    print('+==============')
    mark, end_ = utool.log_progress('query word: ', len(common_wxs),
                                    flushfreq=100, writefreq=25,
                                    with_totaltime=False)
    for count, wx in enumerate(common_wxs):
        mark(count)
        qrvecs = wx2_qrvecs[wx]  # Query vectors for wx-th word
        drvecs = wx2_drvecs[wx]  # Database vectors for wx-th word
        weight = wx2_weight[wx]  # Word Weight
        qfx2_wscore = Match_N(qrvecs, drvecs)  # Compute score matrix
        simmat = qrvecs.dot(qrvecs.T)
        # Nanvectors were equal to the cluster center.
        # This means that point was the only one in its cluster
        # Therefore it is distinctive and should have a high score
        simmat[np.isnan(simmat)] = 1.0
        #""" sigma from SMK paper rscore = residual score """
        scores = (np.sign(simmat) * np.abs(simmat)) ** alpha
        scores[scores <= thresh] = 0
        # Group scores by database annotation ids
        if withinfo:
            group = qfx2_wscore.groupby(idx2_daid, axis=1)
            for daid, scoremat in group:
                daid2_wx2_scoremat[daid][wx] = scoremat
        daid2_wscore = weight * qfx2_wscore.sum(axis=0).groupby(idx2_daid).sum()
        daid2_aggscore = daid2_aggscore.add(daid2_wscore, fill_value=0)
    daid2_totalscore = daid2_aggscore * daid2_gamma * query_gamma
    end_()
    print('L==============')

    if withinfo:
        print('applying weights')
        # Correctly weight individual matches
        for daid, wx2_scoremat in six.iteritems(daid2_wx2_scoremat):
            # Adjust recoreded scores to account for normalizations
            gamma_weight = (daid2_gamma[daid] * query_gamma)
            for wx, scoremat in six.iteritems(wx2_scoremat):
                num = scoremat.values.size
                daid2_wx2_scoremat[daid][wx] = num * scoremat * wx2_weight[wx] * gamma_weight

    return daid2_totalscore, daid2_wx2_scoremat


import cyth
if cyth.DYNAMIC:
    exec(cyth.import_cyth_execstr(__name__))
else:
    pass
    # <AUTOGEN_CYTH>
    # </AUTOGEN_CYTH>
