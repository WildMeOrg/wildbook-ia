# -*- coding: utf-8 -*-
"""
The functions for scoring smk matches
"""
from __future__ import absolute_import, division, print_function
import utool
#import pandas as pd
import numpy as np
#import scipy.sparse as spsparse
#from ibeis.algo.hots import hstypes
from ibeis.algo.hots import hstypes
from six.moves import zip
(print, print_, printDBG, rrr, profile) = utool.inject2(__name__, '[smk_scoring]')


DEBUG_SMK = utool.DEBUG2 or utool.get_argflag('--debug-smk')


@profile
def sccw_summation(rvecs_list, flags_list, idf_list, maws_list, smk_alpha, smk_thresh):
    r"""
    Computes gamma from "To Aggregate or not to aggregate". Every component in
    each list is with repsect to a different word.

    scc = self consistency criterion
    It is a scalar which ensure K(X, X) = 1

    Args:
        rvecs_list (list of ndarrays): residual vectors for every word
        idf_list (list of floats): idf weight for each word
        maws_list (list of ndarrays): multi-assign weights for each word for each residual vector
        smk_alpha (float): selectivity power
        smk_thresh (float): selectivity threshold

    Returns:
        float: sccw self-consistency-criterion weight

    Math:
        \begin{equation}
        \gamma(X) = (\sum_{c \in \C} w_c M(X_c, X_c))^{-.5}
        \end{equation}

    Example:
        >>> from ibeis.algo.hots.smk.smk_scoring import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_scoring
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> #idf_list, rvecs_list, maws_list, smk_alpha, smk_thresh, wx2_flags = smk_debug.testdata_sccw_sum(db='testdb1')
        >>> tup = smk_debug.testdata_sccw_sum(db='PZ_MTEST', nWords=128000)
        >>> idf_list, rvecs_list, flags_list, maws_list, smk_alpha, smk_thresh = tup
        >>> sccw = smk_scoring.sccw_summation(rvecs_list, flags_list, idf_list, maws_list, smk_alpha, smk_thresh)
        >>> print(sccw)
        0.0201041835751

    CommandLine:
        python smk_match.py --db PZ_MOTHERS --nWords 128

    Ignore:
        0.0384477314197
        qmaws_list = dmaws_list = maws_list
        drvecs_list = qrvecs_list = rvecs_list
        dflags_list = qflags_list = flags_list

        flags_list = flags_list[7:10]
        maws_list  = maws_list[7:10]
        idf_list   = idf_list[7:10]
        rvecs_list = rvecs_list[7:10]

    """
    num_rvecs = len(rvecs_list)
    if DEBUG_SMK:
        assert maws_list is None or len(maws_list) == num_rvecs, 'inconsistent lengths'
        assert num_rvecs == len(idf_list), 'inconsistent lengths'
        assert maws_list is None or list(map(len, maws_list)) == list(map(len, rvecs_list)), 'inconsistent per word lengths'
        assert flags_list is None or list(map(len, maws_list)) == list(map(len, flags_list)), 'inconsistent per word lengths'
        assert flags_list is None or len(flags_list) == num_rvecs, 'inconsistent lengths'
    # Indexing with asymetric multi-assignment might get you a non 1 self score?
    # List of scores for every word.
    scores_list = score_matches(rvecs_list, rvecs_list, flags_list, flags_list,
                                maws_list, maws_list, smk_alpha, smk_thresh,
                                idf_list)
    if DEBUG_SMK:
        assert len(scores_list) == num_rvecs, 'bad rvec and score'
        assert len(idf_list) == len(scores_list), 'bad weight and score'
    # Summation over all residual vector scores
    _count = sum((scores.size for scores in  scores_list))
    _iter  = utool.iflatten(scores.ravel() for scores in scores_list)
    self_rawscore = np.fromiter(_iter, np.float64, _count).sum()
    # Square root inverse to enforce normalized self-score is 1.0
    sccw = np.reciprocal(np.sqrt(self_rawscore))
    try:
        assert not np.isinf(sccw), 'sccw cannot be infinite'
        assert not np.isnan(sccw), 'sccw cannot be nan'
    except AssertionError as ex:
        utool.printex(ex, 'problem computing self consistency criterion weight',
                      keys=['num_rvecs'], iswarning=True)
        if num_rvecs > 0:
            raise
        else:
            sccw = 1
    return sccw


@profile
def score_matches(qrvecs_list, drvecs_list, qflags_list, dflags_list,
                  qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list):
    """
    Similarity + Selectivity: M(X_c, Y_c)

    Computes the similarity matrix between word correspondences

    Args:
        qrvecs_list : query vectors for each word
        drvecs_list : database vectors for each word
        qmaws_list  : multi assigned weights for each query word
        dmaws_list  : multi assigned weights for each database word
        smk_alpha       : selectivity power
        smk_thresh      : selectivity smk_thresh

    Returns:
        list : list of score matrices

    References:
        https://lear.inrialpes.fr/~douze/enseignement/2013-2014/presentation_papers/tolias_aggregate.pdf

    Example:
        >>> from ibeis.algo.hots.smk.smk_scoring import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> smk_alpha = 3
        >>> smk_thresh = 0
        >>> qrvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> drvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> qmaws_list  = [smk_debug.get_test_maws(rvecs) for rvecs in qrvecs_list]
        >>> dmaws_list  = [np.ones(rvecs.shape[0], dtype=hstypes.FLOAT_TYPE) for rvecs in qrvecs_list]
        >>> idf_list = [1.0 for _ in qrvecs_list]
        >>> scores_list = score_matches(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list)
    """
    # Cosine similarity between normalized residuals
    simmat_list = similarity_function(qrvecs_list, drvecs_list, qflags_list, dflags_list)
    # Apply sigma selectivity (power law) (BEFORE WEIGHTING)
    scoremat_list = selectivity_function(simmat_list, smk_alpha, smk_thresh)
    # Apply Weights (AFTER SELECTIVITY)
    wscoremat_list = apply_weights(scoremat_list, qmaws_list, dmaws_list, idf_list)
    return wscoremat_list


def rvecs_dot_uint8(qrvecs, drvecs):
    return qrvecs.astype(np.float32).dot(drvecs.T.astype(np.float32)) / hstypes.RVEC_PSEUDO_MAX_SQRD


@profile
def similarity_function(qrvecs_list, drvecs_list, qflags_list, dflags_list):
    """ Phi dot product.

    Args:
        qrvecs_list (list): query residual vectors for each matching word
        drvecs_list (list): corresponding database residual vectors
        qflags_list (list): indicates if a query vector was nan
        dflags_list (list): indicates if a database vector was nan

    Returns:
        simmat_list

    qrvecs_list list of rvecs for each word

    Example:
        >>> from ibeis.algo.hots.smk.smk_scoring import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> qrvecs_list, drvecs_list = smk_debug.testdata_similarity_function()
        >>> simmat_list = similarity_function(qrvecs_list, drvecs_list)
    """
    # For int8: Downweight by the psuedo max squared, to get scores between 0 and 1
    simmat_list = [
        rvecs_dot_uint8(qrvecs, drvecs)
        for qrvecs, drvecs in zip(qrvecs_list, drvecs_list)
    ]
    if utool.DEBUG2:
        assert len(simmat_list) == len(qrvecs_list), 'bad simmat and qrvec'
        assert len(simmat_list) == len(drvecs_list), 'bad simmat and drvec'

    if qflags_list is not None and dflags_list is not None:
        # Set any scores resulting from flagged vectors to 1
        # Actually lets add .5  because we dont know if a flagged vector
        # is a good match, but if both database and query are flagged then
        # it must be a good match
        for qflags, dflags, simmat in zip(qflags_list, dflags_list, simmat_list):
            simmat[qflags] += 0.5
            simmat.T[dflags] += 0.5
    elif qflags_list is not None:
        for qflags, simmat in zip(qflags_list, simmat_list):
            simmat[qflags] += 0.5
    elif dflags_list is not None:
        for dflags, simmat in zip(dflags_list, simmat_list):
            simmat.T[dflags] += 0.5

    # for float16: just perform the calculation
    #simmat_list = [
    #    qrvecs.dot(drvecs.T)
    #    for qrvecs, drvecs in zip(qrvecs_list, drvecs_list)
    #]

    # uint8 does not have nans. We need to use flag lists
    #for simmat in simmat_list:
    #    simmat[np.isnan(simmat)] = 1.0

    return simmat_list


@profile
def apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list):
    """
    Applys multi-assign weights and idf weights to rvec similarty matrices

    TODO: Maybe should apply the sccw weights too?

    Accounts for rvecs being stored as int8's

    Example:
        >>> from ibeis.algo.hots.smk.smk_scoring import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> simmat_list, qmaws_list, dmaws_list, idf_list = smk_debug.testdata_apply_weights()
        >>> wsim_list = apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list)
    """
    word_weight_list = idf_list
    if qmaws_list is None and dmaws_list is None:
        wsim_list = [
            (word_weight * simmat)
            for simmat, word_weight in
            zip(simmat_list, word_weight_list)
        ]
    elif qmaws_list is not None and dmaws_list is not None:
        wsim_list = [
            (((word_weight * qmaws[:, None]) * simmat) * dmaws[None, :])
            for simmat, qmaws, dmaws, word_weight in
            zip(simmat_list, qmaws_list, dmaws_list, word_weight_list)
        ]
    elif qmaws_list is not None and dmaws_list is None:
        wsim_list = [
            ((word_weight * qmaws[:, None]) * simmat)
            for simmat, qmaws, word_weight in
            zip(simmat_list, qmaws_list, word_weight_list)
        ]
    else:
        raise NotImplementedError('cannot just do dmaws')
    return wsim_list


#@profile
@profile
def selectivity_function(wsim_list, smk_alpha, smk_thresh):
    r""" Selectivity function - sigma from SMK paper rscore = residual score

    Downweights weak matches using power law normalization and thresholds
    anybody that is too weak

    Example:
        >>> import numpy as np
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> smk_debug.rrr()
        >>> np.random.seed(0)
        >>> wsim_list, smk_alpha, smk_thresh = smk_debug.testdata_selectivity_function()

    Timeits:
        >>> import utool
        >>> utool.util_dev.rrr()
        >>> setup = utool.codeblock(
        ...     '''
                import numpy as np
                import scipy.sparse as spsparse
                from ibeis.algo.hots.smk import smk_debug
                np.random.seed(0)
                wsim_list, smk_alpha, smk_thresh = smk_debug.testdata_selectivity_function()
                scores_iter = [
                    np.multiply(np.sign(mawmat), np.power(np.abs(mawmat), smk_alpha))
                    for mawmat in wsim_list
                ]
                ''')
        >>> stmt_list = utool.codeblock(
        ...     '''
                scores_list0 = [np.multiply(scores, np.greater(scores, smk_thresh)) for scores in scores_iter]
                scores_list1 = [spsparse.coo_matrix(np.multiply(scores, np.greater(scores, smk_thresh))) for scores in scores_iter]
                scores_list2 = [spsparse.dok_matrix(np.multiply(scores, np.greater(scores, smk_thresh))) for scores in scores_iter]
                scores_list3 = [spsparse.lil_matrix(np.multiply(scores, np.greater(scores, smk_thresh))) for scores in scores_iter]
                '''
        ... ).split('\n')
        >>> utool.util_dev.timeit_compare(stmt_list, setup, int(1E4))

        scores0 = scores_list0[-1]
        scores1 = scores_list1[-1]
        scores2 = scores_list2[-1]
        scores3 = scores_list3[-1]
        %timeit scores0.sum()
        %timeit scores1.sum()
        %timeit scores2.sum()
        %timeit scores3.sum()
    """
    # Apply powerlaw
    scores_iter = [
        np.multiply(np.sign(mawmat), np.power(np.abs(mawmat), smk_alpha))
        for mawmat in wsim_list
    ]
    # Apply threshold
    scores_list = [
        np.multiply(scores, np.greater(scores, smk_thresh))
        for scores in scores_iter
    ]
    if utool.DEBUG2:
        assert len(scores_list) == len(wsim_list)
    return scores_list
