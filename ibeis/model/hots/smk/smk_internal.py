"""
The real core of the smk_algorithm
The smk_core module was getting crowded so I made smk_internal
"""
from __future__ import absolute_import, division, print_function
import utool
#import pandas as pd
import numpy as np
#import scipy.sparse as spsparse
#from ibeis.model.hots import hstypes
from six.moves import zip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_internal]')


@profile
def score_matches(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha,
                  smk_thresh, idf_list):
    """ Similarity + Selectivity: M(X_c, Y_c)

    Args:
        qrvecs_list : query vectors for each word
        drvecs_list : database vectors for each word
        qmaws_list  : multi assigned weights for each query word
        dmaws_list  : multi assigned weights for each database word
        smk_alpha       : selectivity power
        smk_thresh      : selectivity smk_thresh

    Returns:
        score matrix

    Example:
        >>> from ibeis.model.hots.smk.smk_internal import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> smk_alpha = 3
        >>> smk_thresh = 0
        >>> qrvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> drvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> qmaws_list  = [smk_debug.get_test_maws(rvecs) for rvecs in qrvecs_list]
        >>> dmaws_list  = [np.ones(rvecs.shape[0], dtype=hstypes.FLOAT_TYPE) for rvecs in qrvecs_list]
        >>> idf_list = [1 for _ in qrvecs_list]
        >>> scores_list = score_matches(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list)
    """
    # Cosine similarity between normalized residuals
    simmat_list = similarity_function(qrvecs_list, drvecs_list)
    # Apply Weights
    wsim_list = apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list)
    # Apply sigma selectivity (power law)
    scores_list = selectivity_function(wsim_list, smk_alpha, smk_thresh)
    return scores_list


@profile
def similarity_function(qrvecs_list, drvecs_list):
    """ Phi dot product. Accounts for NaN residual vectors
    qrvecs_list list of rvecs for each word

    Example:
        >>> from ibeis.model.hots.smk.smk_internal import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> qrvecs_list, drvecs_list = smk_debug.testdata_similarity_function()
        >>> simmat_list = similarity_function(qrvecs_list, drvecs_list)
    """
    # For int8: Downweight by the psuedo max squared, to get scores between 0 and 1
    #simmat_list = [
    #    qrvecs.astype(np.float32).dot(drvecs.T.astype(np.float32)) / hstypes.RVEC_PSEUDO_MAX_SQRD
    #    for qrvecs, drvecs in zip(qrvecs_list, drvecs_list)
    #]
    # for float16: just perform the calculation
    simmat_list = [
        qrvecs.dot(drvecs.T)
        for qrvecs, drvecs in zip(qrvecs_list, drvecs_list)
    ]
    if utool.DEBUG2:
        assert len(simmat_list) == len(qrvecs_list), 'bad simmat and qrvec'
        assert len(simmat_list) == len(drvecs_list), 'bad simmat and drvec'
    # Rvec is NaN implies it is a cluster center. perfect similarity
    # FIXME: this only works for float16, if RVEC_TYPE is int8, then we need
    # to come up with something else (like a mask for rows)
    for simmat in simmat_list:
        simmat[np.isnan(simmat)] = 1.0

    return simmat_list


@profile
def apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list):
    """
    Applys multi-assign weights and idf weights to rvec similarty matrices

    TODO: Maybe should apply the sccw weights too?

    Accounts for rvecs being stored as int8's

    Example:
        >>> from ibeis.model.hots.smk.smk_internal import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
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
    """ Selectivity function - sigma from SMK paper rscore = residual score

    Downweights weak matches using power law normalization and thresholds
    anybody that is too weak

    Example:
        >>> import numpy as np
        >>> from ibeis.model.hots.smk import smk_debug
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
                from ibeis.model.hots.smk import smk_debug
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
