"""
smk core
"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import pandas as pd
import utool


def gamma_summation(wx2_rvecs, wx2_weight):
    r"""
    \begin{equation}
    \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
    \end{equation}
    """
    gamma_iter = (wx2_weight.get(wx, 0) * Match_N(vecs, vecs).sum()
                  for wx, vecs in six.iteritems(wx2_rvecs))
    return np.reciprocal(np.sqrt(sum(gamma_iter)))


def selectivity_function(rscore_mat, alpha=3, thresh=0):
    """ sigma from SMK paper rscore = residual score """
    scores = (np.sign(rscore_mat) * np.abs(rscore_mat)) ** alpha
    scores[scores <= thresh] = 0
    return scores


def Match_N(vecs1, vecs2):
    simmat = vecs1.dot(vecs2.T)
    # Nanvectors were equal to the cluster center.
    # This means that point was the only one in its cluster
    # Therefore it is distinctive and should have a high score
    simmat[np.isnan(simmat)] = 1.0
    return selectivity_function(simmat)


def match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid):
    """
    >>> from smk_core import *  # NOQA
    >>> import smk
    >>> ibs, annots_df, taids, daids, qaids, nWords = smk.testdata()
    >>> words = learn_visual_words(annots_df, taids, nWords)
    >>> invindex = index_data_annots(annots_df, daids, words)
    >>> qaid = qaids[0]
    >>> wx2_qfxs, wx2_qrvecs = compute_query_repr(annots_df, qaid, invindex)
    >>> daid2_totalscore = match_kernel(wx2_qrvecs, wx2_qfxs, invindex, qaid)
    """
    _daids = invindex._daids
    idx2_daid = invindex.idx2_daid
    wx2_drvecs = invindex.wx2_drvecs
    wx2_weight = invindex.wx2_weight
    daid2_gamma = invindex.daid2_gamma

    wx2_rvecs = wx2_qrvecs
    query_gamma = gamma_summation(wx2_rvecs, wx2_weight)

    # Accumulate scores over the entire database
    daid2_aggscore = pd.Series(np.zeros(len(_daids)), index=_daids, name='total_score')
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))

    daid2_wx2_scoremat = utool.ddict(lambda: utool.ddict(list))

    # for each word compute the pairwise scores between matches
    mark, end = utool.log_progress('query word: ', len(common_wxs), flushfreq=100)
    for count, wx in enumerate(common_wxs):
        mark(count)
        # Query and database vectors for wx-th word
        qrvecs = wx2_qrvecs[wx]
        drvecs = wx2_drvecs[wx]
        # Word Weight
        weight = wx2_weight[wx]
        # Compute score matrix
        qfx2_wscore = Match_N(qrvecs, drvecs)
        qfx2_wscore.groupby(idx2_daid)
        # Group scores by database annotation ids
        group = qfx2_wscore.groupby(idx2_daid, axis=1)
        for daid, scoremat in group:
            daid2_wx2_scoremat[daid][wx] = scoremat
        #qfx2_wscore = pd.DataFrame(qfx2_wscore_, index=qfxs, columns=_idxs)
        daid2_wscore = weight * qfx2_wscore.sum(axis=0).groupby(idx2_daid).sum()
        daid2_aggscore = daid2_aggscore.add(daid2_wscore, fill_value=0)
    daid2_totalscore = daid2_aggscore * daid2_gamma * query_gamma
    end()

    return daid2_totalscore, daid2_wx2_scoremat
