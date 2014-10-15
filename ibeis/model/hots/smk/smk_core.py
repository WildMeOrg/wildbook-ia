"""
smk core
"""
from __future__ import absolute_import, division, print_function
#import six
from six.moves import zip
import numpy as np
import scipy.sparse as spsparse
#import pandas as pd
import utool
import numpy.linalg as npl
from ibeis.model.hots import hstypes
#from ibeis.model.hots.smk import pandas_helpers as pdh
from itertools import product
from vtool import clustering2 as clustertool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_core]')


RVEC_TYPE = hstypes.RVEC_TYPE
RVEC_MAX = hstypes.RVEC_MAX
RVEC_MIN = hstypes.RVEC_MIN
RVEC_MAX_SQRD = hstypes.FLOAT_TYPE(RVEC_MAX) ** 2


def quantize_normvec_to_int8(arr_float):
    """
    compresses 8 or 4 bytes of information into 1 byte

    Takes a normalized float vectors in range -1 to 1 with l2norm=1 and
    compresses them into 1 byte. Takes advantage of the fact that
    rarely will a component of a vector be greater than 64, so we can extend the
    range to double what normally would be allowed. This does mean there is a
    slight (but hopefully negligable) information loss. It will be negligable
    when nDims=128, when it is lower, you may want to use a different function.

    Args:
        arr_float (ndarray): normalized residual vector of type float in range -1 to 1 (with l2 norm of 1)
    Returns:
        (ndarray): residual vector of type int8 in range -128 to 128

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> np.random.seed(0)
        >>> arr_float = smk_debug.get_test_float_norm_rvecs(2, 5)
        >>> normalize_vecs_inplace(arr_float)
        >>> arr_int8 = quantize_normvec_to_int8(arr_float)
        >>> print(arr_int8)
        [[ 126   28   70 -128 -128]
         [-128 -128  -26  -18   73]]
    """
    # Trick / hack: use 2 * max, and clip because most components will be less
    # than 2 * max. This will reduce quantization error
    return np.clip((arr_float * (RVEC_MAX * 2)), RVEC_MIN, RVEC_MAX).astype(RVEC_TYPE)


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
    Example:
        #>>> rvecs = (hstypes.RVEC_MAX * np.random.rand(4, 4)).astype(hstypes.RVEC_TYPE)
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> rvecs = (hstypes.RVEC_MAX * np.random.rand(4, 128)).astype(hstypes.RVEC_TYPE)
        >>> maws  = (np.random.rand(rvecs.shape[0])).astype(hstypes.FLOAT_TYPE)
    """
    if rvecs.shape[0] == 1:
        return rvecs
    # Prealloc sum output
    rvecs_agg = np.empty((1, rvecs.shape[1]), dtype=hstypes.FLOAT_TYPE)
    # Take weighted average of multi-assigned vectors
    (maws[:, np.newaxis] * rvecs).sum(axis=0, out=rvecs_agg[0])
    # Jegou uses mean instead. Sum should be fine because we normalize
    #rvecs.mean(axis=0, out=rvecs_agg[0])
    normalize_vecs_inplace(rvecs_agg)
    rvecs_agg = quantize_normvec_to_int8(rvecs_agg)
    return rvecs_agg


#@profile
def get_norm_rvecs(vecs, word):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> vecs = (hstypes.VEC_MAX * np.random.rand(4, 128)).astype(hstypes.VEC_TYPE)
        >>> word = (hstypes.VEC_MAX * np.random.rand(1, 128)).astype(hstypes.VEC_TYPE)
    """
    # Compute residuals of assigned vectors
    #rvecs_n = word.astype(dtype=FLOAT_TYPE) - vecs.astype(dtype=FLOAT_TYPE)
    rvecs_n = np.subtract(word.astype(hstypes.FLOAT_TYPE), vecs.astype(hstypes.FLOAT_TYPE))
    # Faster, but doesnt work with np.norm
    #rvecs_n = np.subtract(word.view(hstypes.FLOAT_TYPE), vecs.view(hstypes.FLOAT_TYPE))
    normalize_vecs_inplace(rvecs_n)
    rvecs_n = quantize_normvec_to_int8(rvecs_n)
    return rvecs_n


#@profile
def selectivity_function(simmat_list, smk_alpha, smk_thresh):
    """ Selectivity function - sigma from SMK paper rscore = residual score """
    smk_alpha = 3
    smk_thresh = 0
    scores_iter = [
        np.multiply(np.sign(simmat), np.power(np.abs(simmat), smk_alpha))
        for simmat in simmat_list
    ]
    scores_list = [
        np.multiply(scores, np.greater(scores, smk_thresh))
        for scores in scores_iter
    ]
    if utool.DEBUG2:
        assert len(scores_list) == len(simmat_list)
    return scores_list


def apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list):
    """
    Applys multi-assign weights to rvec similarty matrices

    Accounts for rvecs being stored as int8's

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> qrvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> drvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> simmat_list = similarity_function(qrvecs_list, drvecs_list)
        >>> qmaws_list  = [smk_debug.get_test_maws(rvecs) for rvecs in qrvecs_list]
        >>> dmaws_list  = [np.ones(rvecs.shape[0], dtype=hstypes.FLOAT_TYPE) for rvecs in qrvecs_list]
        >>> idf_list = [1.0 for _ in qrvecs_list]
    """
    idf_list_ = np.divide(idf_list, RVEC_MAX_SQRD)
    if dmaws_list is None and qmaws_list is None:
        mawsim_list = [
            (idf * simmat)
            for simmat, idf in zip(simmat_list, idf_list_)
        ]
    elif dmaws_list is not None and qmaws_list is not None:
        mawsim_list = [
            (((idf * qmaws[:, np.newaxis]) * simmat) * dmaws[np.newaxis, :])
            for simmat, qmaws, dmaws, idf in
            zip(simmat_list, qmaws_list, dmaws_list, idf_list_)
        ]
    elif qmaws_list is not None and dmaws_list is None:
        mawsim_list = [
            ((idf * qmaws[:, np.newaxis]) * simmat)
            for simmat, qmaws, idf in
            zip(simmat_list, qmaws_list, idf_list)
        ]
    else:
        raise NotImplementedError('cannot just do dmaws')
    return mawsim_list


def similarity_function(qrvecs_list, drvecs_list):
    """ Phi dot product. Accounts for NaN residual vectors
    qrvecs_list list of rvecs for each word

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> qrvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> drvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> simmat_list = similarity_function(qrvecs_list, drvecs_list)
    """
    simmat_list = [
        qrvecs.astype(np.float32).dot(drvecs.T.astype(np.float32))
        for qrvecs, drvecs in zip(qrvecs_list, drvecs_list)
    ]
    if utool.DEBUG2:
        assert len(simmat_list) == len(qrvecs_list), 'bad simmat and qrvec'
        assert len(simmat_list) == len(drvecs_list), 'bad simmat and drvec'
    # Rvec is NaN implies it is a cluster center. perfect similarity
    for simmat in simmat_list:
        simmat[np.isnan(simmat)] = 1.0

    #return sklearn.preprocessing.normalize(csr_mat, norm='l2', axis=1, copy=False)
    #csr_vec = spsparse.csr_matrix(vec, copy=False)
    #csr_vec.shape = (1, csr_vec.size)
    #sparse_stack = [row.multiply(csr_vec) for row in csr_mat]
    #return spsparse.vstack(sparse_stack, format='csr')
    return simmat_list


def score_matches(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha,
                  smk_thresh, idf_list=None):
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
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> qrvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> drvecs_list = [smk_debug.get_test_rvecs(_) for _ in range(10)]
        >>> qmaws_list  = [smk_debug.get_test_maws(rvecs) for rvecs in qrvecs_list]
        >>> dmaws_list  = [np.ones(rvecs.shape[0], dtype=hstypes.FLOAT_TYPE) for rvecs in qrvecs_list]
        >>> idf_list = [1 for _ in qrvecs_list]
        >>> smk_alpha = 3
        >>> smk_thresh = 0
    """
    # Cosine similarity between normalized residuals
    simmat_list = similarity_function(qrvecs_list, drvecs_list)
    # Apply Weights
    mawmat_list = apply_weights(simmat_list, qmaws_list, dmaws_list, idf_list)
    # Apply sigma selectivity (power law)
    scores_list = selectivity_function(mawmat_list, smk_alpha, smk_thresh)
    return scores_list


#@profile
def sccw_summation(rvecs_list, idf_list, maws_list, smk_alpha, smk_thresh):
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
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> idf_list, rvecs_list, maws_list, smk_alpha, smk_thresh = smk_debug.testsdata_sccw_sum()
        >>> qmaws_list = dmaws_list = maws_list
        >>> drvecs_list = qrvecs_list = rvecs_list
        >>> scoremat = sccw_summation(rvecs_list, idf_list, maws_list, smk_alpha, smk_thresh )
        >>> print(scoremat)
        0.0201041835751

        0.0384477314197

    Ignore:
        qrvecs_list = drvecs_list = rvecs_list
    """
    if utool.DEBUG2:
        assert len(maws_list) == len(rvecs_list)
        assert len(maws_list) == len(idf_list)
        assert list(map(len, maws_list)) == list(map(len, rvecs_list))
    # Indexing with asymetric multi-assignment might get you a non 1 self score?
    # List of scores for every word.
    scores_list = score_matches(rvecs_list, rvecs_list,
                                maws_list, maws_list,
                                smk_alpha, smk_thresh, idf_list)
    if utool.DEBUG2:
        assert len(scores_list) == len(rvecs_list), 'bad rvec and score'
        assert len(idf_list) == len(scores_list), 'bad weight and score'
    # Summation over all residual vector scores
    _count = sum((scores.size for scores in  scores_list))
    _iter  = utool.iflatten(scores.ravel() for scores in scores_list)
    total  = np.fromiter(_iter, np.float64, _count).sum()
    # Square root inverse
    sccw = np.reciprocal(np.sqrt(total))
    return sccw


def match_kernel_L1(wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex,
                    withinfo=True, smk_alpha=3, smk_thresh=0):
    """ function """
    wx2_drvecs     = invindex.wx2_drvecs
    wx2_idf        = invindex.wx2_idf
    wx2_daid       = invindex.wx2_aids
    daid2_sccw     = invindex.daid2_sccw

    # for each word compute the pairwise scores between matches
    common_wxs = set(wx2_qrvecs.keys()).intersection(set(wx2_drvecs.keys()))
    # Build lists over common word indexes
    qrvecs_list = [wx2_qrvecs[wx] for wx in common_wxs]
    drvecs_list = [wx2_drvecs[wx] for wx in common_wxs]
    daids_list  = [  wx2_daid[wx] for wx in common_wxs]
    idf_list    = [   wx2_idf[wx] for wx in common_wxs]
    qmaws_list  = [ wx2_qmaws[wx] for wx in common_wxs]  # NOQA
    dmaws_list  = None
    if utool.VERBOSE:
        mark, end_ = utool.log_progress('[smk_core] query word: ', len(common_wxs),
                                        flushfreq=100, writefreq=25,
                                        with_totaltime=True)
    #--------
    ret_L0 = match_kernel_L0(qrvecs_list, drvecs_list, qmaws_list, dmaws_list,
                             smk_alpha, smk_thresh, idf_list, daids_list,
                             daid2_sccw, query_sccw)
    (daid2_totalscore, scores_list, daid_agg_keys,) = ret_L0

    ret_L1 = (daid2_totalscore, common_wxs, scores_list, daids_list, idf_list,
              daid_agg_keys,)
    #--------
    if utool.VERBOSE:
        end_()
    return ret_L1


def match_kernel_L0(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha,
                    smk_thresh, idf_list, daids_list, daid2_sccw, query_sccw):
    """
    Computes smk kernels

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> smk_debug.rrr()
        >>> core1, core2, extra = smk_debug.testdata_match_kernel_L0()
        >>> smk_alpha, smk_thresh, query_sccw, daids_list, daid2_sccw = core1
        >>> qrvecs_list, drvecs_list, qmaws_list, dmaws_list, idf_list = core2
        >>> qaid2_sccw, qaids_list = extra
        >>> ret_L0 = match_kernel_L0(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list, daids_list, daid2_sccw, query_sccw)
        >>> # Test Asymetric Matching
        >>> (daid2_totalscore, scores_list, daid_agg_keys,) = ret_L0
        >>> print(daid2_totalscore[5])
        0.336434201301
        >>> # Test Self Consistency
        >>> qret = match_kernel_L0(qrvecs_list, qrvecs_list, qmaws_list, qmaws_list, smk_alpha, smk_thresh, idf_list, qaids_list, qaid2_sccw, query_sccw)
        >>> (qaid2_totalscore, qscores_list, qaid_agg_keys,) = qret
        >>> print(qaid2_totalscore[42])
        1.0000000000000007
    """
    # Residual vector scores
    scores_list = score_matches(qrvecs_list, drvecs_list, qmaws_list, dmaws_list, smk_alpha, smk_thresh, idf_list)
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
    ret_L0 = (daid2_totalscore, scores_list, daid_agg_keys,)
    return ret_L0


def accumulate_scores(dscores_list, daids_list):
    """ helper """
    daid2_aggscore = utool.ddict(lambda: 0)
    ### Weirdly iflatten was slower here
    for dscores, daids in zip(dscores_list, daids_list):
        for daid, score in zip(daids, dscores):
            daid2_aggscore[daid] += score
    daid_agg_keys   = np.array(list(daid2_aggscore.keys()))
    daid_agg_scores = np.array(list(daid2_aggscore.values()))
    return daid_agg_keys, daid_agg_scores


@profile
def match_kernel_L2(wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex,
                    withinfo=True, smk_alpha=3, smk_thresh=0):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, invindex, qindex = smk_debug.testdata_match_kernel_L2()
        >>> wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw = qindex
        >>> smk_alpha = ibs.cfg.query_cfg.smk_cfg.smk_alpha
        >>> smk_thresh = ibs.cfg.query_cfg.smk_cfg.smk_thresh
        >>> withinfo = True  # takes an 11s vs 2s
        >>> _args = (wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex, withinfo, smk_alpha, smk_thresh)
        >>> smk_debug.rrr()
        >>> smk_debug.invindex_dbgstr(invindex)
        >>> daid2_totalscore, daid2_wx2_scoremat = match_kernel_L2(*_args)
    """
    # Pack
    args = (wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex,
            withinfo, smk_alpha, smk_thresh)
    # Call match kernel logic
    ret_L1 =  match_kernel_L1(*args)
    # Unpack
    (daid2_totalscore, common_wxs, scores_list, daids_list, idf_list,
     daid_agg_keys,)  = ret_L1
    if withinfo:
        # Build up chipmatch if requested
        # TODO: Only build for a shortlist
        # TODO: Use a sparse matrices
        daid2_chipmatch = build_daid2_chipmatch2(invindex, common_wxs,
                                                 wx2_qaids, wx2_qfxs,
                                                 scores_list, idf_list,
                                                 daids_list, query_sccw)
    else:
        daid2_chipmatch = None

    return daid2_totalscore, daid2_chipmatch


def mem_arange(num, cache={}):
    # TODO: weakref cache
    if num not in cache:
        cache[num] = np.arange(num)
    return cache[num]


def mem_meshgrid(wrange, hrange, cache={}):
    # TODO: weakref cache
    key = (id(wrange), id(hrange))
    if key not in cache:
        cache[key] = np.meshgrid(wrange, hrange, indexing='ij')
    return cache[key]


@profile
def build_daid2_chipmatch2(invindex, common_wxs, wx2_qaids, wx2_qfxs,
                           scores_list, idf_list, daids_list, query_sccw):
    """
    Builds explicit chipmatches that the rest of the pipeline plays nice with

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

    CommandLine::
        python dev.py -t smk0 --allgt --db GZ_ALL --index 2:5

    Example:
        >>> from ibeis.model.hots.smk.smk_core import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, invindex, qindex = smk_debug.testdata_match_kernel_L2()
        >>> wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw = qindex
        >>> smk_alpha = ibs.cfg.query_cfg.smk_cfg.smk_alpha
        >>> smk_thresh = ibs.cfg.query_cfg.smk_cfg.smk_thresh
        >>> withinfo = True  # takes an 11s vs 2s
        >>> _args = (wx2_qrvecs, wx2_qmaws, wx2_qaids, wx2_qfxs, query_sccw, invindex, withinfo, smk_alpha, smk_thresh)
        >>> smk_debug.rrr()
        >>> smk_debug.invindex_dbgstr(invindex)
        >>> daid2_totalscore, daid2_wx2_scoremat = match_kernel_L2(*_args)
    """
    # FIXME: move groupby to vtool
    if utool.VERBOSE:
        print('[smk_core] build chipmatch')

    wx2_dfxs  = invindex.wx2_fxs
    daid2_sccw = invindex.daid2_sccw

    qfxs_list = [wx2_qfxs[wx] for wx in common_wxs]
    dfxs_list = [wx2_dfxs[wx] for wx in common_wxs]
    if isinstance(daid2_sccw, dict):
        daid2_sccw_ = daid2_sccw
    else:
        daid2_sccw_ = daid2_sccw.to_dict()

    shapes_list  = [scores.shape for scores in scores_list]  # 51us
    shape_ranges = [(mem_arange(w), mem_arange(h)) for (w, h) in shapes_list]  # 230us
    ijs_list = [mem_meshgrid(wrange, hrange) for (wrange, hrange) in shape_ranges]  # 278us
    # Normalize scores for words, nMatches, and query sccw (still need daid sccw)
    nscores_iter = (scores * query_sccw for scores in scores_list)

    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.dbstr_qindex()

    # FIXME: Preflatten all of these lists
    out_ijs    = [
        list(zip(_is.flat, _js.flat))
        for (_is, _js) in ijs_list
    ]
    out_qfxs   = [
        [qfxs[ix] for (ix, jx) in ijs]
        for (qfxs, ijs) in zip(qfxs_list, out_ijs)
    ]
    out_dfxs   = [
        [dfxs[jx] for (ix, jx) in ijs]
        for (dfxs, ijs) in zip(dfxs_list, out_ijs)
    ]
    out_daids  = (
        [daids[jx] for (ix, jx) in ijs]
        for (daids, ijs) in zip(daids_list, out_ijs)
    )
    out_scores = (
        [nscores[ijx] for ijx in ijs]
        for (nscores, ijs) in zip(nscores_iter, out_ijs)
    )
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
    all_daids = np.array(list(utool.iflatten(utool.iflatten(nested_daid_iter))), dtype=hstypes.INDEX_TYPE)
    all_scores = np.array(list(utool.iflatten(utool.iflatten(nested_score_iter))), dtype=hstypes.FS_DTYPE)
    #assert len(all_daids) == len(all_scores)
    #assert len(all_fms) == len(all_scores)

    daid_keys, groupxs = clustertool.group_indicies(all_daids)
    fs_list = clustertool.apply_grouping(all_scores, groupxs)
    fm_list = clustertool.apply_grouping(all_fms, groupxs)
    daid2_fm = {daid: fm for daid, fm in zip(daid_keys, fm_list)}
    daid2_fs = {daid: fs * daid2_sccw_[daid] for daid, fs in zip(daid_keys, fs_list)}
    # FIXME: generalize to when nAssign > 1
    daid2_fk = {daid: np.ones(fs.size, dtype=hstypes.FK_DTYPE) for daid, fs in zip(daid_keys, fs_list)}
    daid2_chipmatch = (daid2_fm, daid2_fs, daid2_fk)

    return daid2_chipmatch
