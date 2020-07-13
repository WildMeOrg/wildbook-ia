# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import vtool as vt
import functools
from wbia.algo.hots import hstypes
from wbia.algo.hots import _pipeline_helpers as plh
from six.moves import zip, range, map  # NOQA

print, rrr, profile = ut.inject2(__name__)

"""
FIXME: qfx2_ no longer applies due to fgw_thresh. Need to change names in this file

TODO: replace testdata_pre_weight_neighbors with

        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='testdb1',
        >>>                                a=['default:qindex=0:1,dindex=0:5,hackerrors=False'],
        >>>                                p=['default:codename=vsmany,bar_l2_on=True,fg_on=False'], verbose=True)

"""


NN_WEIGHT_FUNC_DICT = {}
MISC_WEIGHT_FUNC_DICT = {}
EPS = 1e-8


def _register_nn_normalized_weight_func(func):
    r"""
    Decorator for weighting functions

    Registers a nearest neighbor normalized weighting
    Used for LNBNN
    """
    global NN_WEIGHT_FUNC_DICT
    filtkey = ut.get_funcname(func).replace('_fn', '').lower()
    if ut.VERYVERBOSE:
        print('[nn_weights] registering norm func: %r' % (filtkey,))
    filtfunc = functools.partial(nn_normalized_weight, func)
    NN_WEIGHT_FUNC_DICT[filtkey] = filtfunc
    return func


def _register_nn_simple_weight_func(func):
    """
    Used for things that dont require a normalizer like const
    """
    filtkey = ut.get_funcname(func).replace('_match_weighter', '').lower()
    if ut.VERYVERBOSE:
        print('[nn_weights] registering simple func: %r' % (filtkey,))
    NN_WEIGHT_FUNC_DICT[filtkey] = func
    return func


def _register_misc_weight_func(func):
    filtkey = ut.get_funcname(func).replace('_match_weighter', '').lower()
    if ut.VERYVERBOSE:
        print('[nn_weights] registering simple func: %r' % (filtkey,))
    MISC_WEIGHT_FUNC_DICT[filtkey] = func
    return func


@_register_nn_simple_weight_func
def const_match_weighter(nns_list, nnvalid0_list, qreq_):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> #tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='PZ_MTEST')
        >>> nns_list, nnvalid0_list = args
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> constvote_weight_list = const_match_weighter(nns_list, nnvalid0_list, qreq_)
        >>> result = ('constvote_weight_list = %s' % (str(constvote_weight_list),))
        >>> print(result)
    """
    constvote_weight_list = []
    # K = qreq_.qparams.K  # Dont use K because K is dynamic per query
    # Subtract Knorm from size instead
    Knorm = qreq_.qparams.Knorm

    for nns in nns_list:
        (neighb_idx, neighb_dist) = nns
        neighb_constvote = np.ones(
            (len(neighb_idx), len(neighb_idx.T) - Knorm), dtype=np.float
        )
        constvote_weight_list.append(neighb_constvote)
    return constvote_weight_list


@_register_nn_simple_weight_func
def fg_match_weighter(nns_list, nnvalid0_list, qreq_):
    r"""
    foreground feature match weighting

    CommandLine:
        python -m wbia.algo.hots.nn_weights --exec-fg_match_weighter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> #tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> #ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='PZ_MTEST')
        >>> nns_list, nnvalid0_list = args
        >>> print(ut.repr2(qreq_.qparams.__dict__, sorted_=True))
        >>> assert qreq_.qparams.fg_on == True, 'bug setting custom params fg_on'
        >>> fgvotes_list = fg_match_weighter(nns_list, nnvalid0_list, qreq_)
        >>> print('fgvotes_list = %r' % (fgvotes_list,))
    """
    Knorm = qreq_.qparams.Knorm
    config2_ = qreq_.get_internal_query_config2()
    # Database feature index to chip index
    fgvotes_list = []
    for nn in nns_list:
        # database forground weights
        neighb_dfgws = qreq_.indexer.get_nn_fgws(nn.neighb_idxs.T[0:-Knorm].T)
        # query forground weights
        qfx2_qfgw = qreq_.ibs.get_annot_fgweights(
            [nn.qaid], ensure=False, config2_=config2_
        )[0]
        qfgws = qfx2_qfgw.take(nn.qfx_list, axis=0)
        # feature match forground weight is geometric mean
        neighb_fgvote_weight = np.sqrt(qfgws[:, None] * neighb_dfgws)
        fgvotes_list.append(neighb_fgvote_weight)
    return fgvotes_list


def nn_normalized_weight(normweight_fn, nns_list, nnvalid0_list, qreq_):
    r"""
    Generic function to weight nearest neighbors

    ratio, lnbnn, and other nearest neighbor based functions use this

    Args:
        normweight_fn (func): chosen weight function e.g. lnbnn
        nns_list (dict): query descriptor nearest neighbors and distances.
        nnvalid0_list (list): list of neighbors preflagged as valid
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        list: weights_list

    CommandLine:
        python -m wbia.algo.hots.nn_weights nn_normalized_weight --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> #tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> #ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> qreq_, args = plh.testdata_pre('weight_neighbors',
        >>>                                defaultdb='PZ_MTEST')
        >>> nns_list, nnvalid0_list = args
        >>> normweight_fn = lnbnn_fn
        >>> weights_list1, normk_list1 = nn_normalized_weight(
        >>>     normweight_fn, nns_list, nnvalid0_list, qreq_)
        >>> weights1 = weights_list1[0]
        >>> nn_normonly_weight = NN_WEIGHT_FUNC_DICT['lnbnn']
        >>> weights_list2, normk_list2 = nn_normonly_weight(nns_list, nnvalid0_list, qreq_)
        >>> weights2 = weights_list2[0]
        >>> assert np.all(weights1 == weights2)
        >>> ut.assert_inbounds(weights1.sum(), 100, 510)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> #tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> qreq_, args = plh.testdata_pre('weight_neighbors',
        >>>                                defaultdb='PZ_MTEST')
        >>> nns_list, nnvalid0_list = args
        >>> normweight_fn = ratio_fn
        >>> weights_list1, normk_list1 = nn_normalized_weight(normweight_fn, nns_list, nnvalid0_list, qreq_)
        >>> weights1 = weights_list1[0]
        >>> nn_normonly_weight = NN_WEIGHT_FUNC_DICT['ratio']
        >>> weights_list2, normk_list2 = nn_normonly_weight(nns_list, nnvalid0_list, qreq_)
        >>> weights2 = weights_list2[0]
        >>> assert np.all(weights1 == weights2)
        >>> ut.assert_inbounds(weights1.sum(), 1500, 4500)
    """
    Knorm = qreq_.qparams.Knorm
    normalizer_rule = qreq_.qparams.normalizer_rule
    # Database feature index to chip index
    qaid_list = qreq_.get_internal_qaids()
    normk_list = [
        get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule)
        for qaid, (neighb_idx, neighb_dist) in zip(qaid_list, nns_list)
    ]
    weight_list = [
        apply_normweight(normweight_fn, neighb_normk, neighb_idx, neighb_dist, Knorm)
        for neighb_normk, (neighb_idx, neighb_dist) in zip(normk_list, nns_list)
    ]
    return weight_list, normk_list


def get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule):
    """
    Get positions of the LNBNN/ratio tests normalizers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> cfgdict = {'K':10, 'Knorm': 10, 'normalizer_rule': 'name',
        >>>            'dim_size': 450, 'resize_dim': 'area'}
        >>> #tup = plh.testdata_pre_weight_neighbors(cfgdict=cfgdict)
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='testdb1',
        >>>                                p=['default:K=10,Knorm=10,normalizer_rule=name,dim_size=450,resize_dim=area'])
        >>> nns_list, nnvalid0_list = args
        >>> (neighb_idx, neighb_dist) = nns_list[0]
        >>> qaid = qreq_.qaids[0]
        >>> K = qreq_.qparams.K
        >>> Knorm = qreq_.qparams.Knorm
        >>> neighb_normk1 = get_normk(qreq_, qaid, neighb_idx, Knorm, 'last')
        >>> neighb_normk2 = get_normk(qreq_, qaid, neighb_idx, Knorm, 'name')
        >>> assert np.all(neighb_normk1 == Knorm + K)
        >>> assert np.all(neighb_normk2 <= Knorm + K) and np.all(neighb_normk2 > K)
    """
    K = len(neighb_idx.T) - Knorm
    assert K > 0, 'K=%r cannot be 0' % (K,)
    # neighb_nndist = neighb_dist.T[0:K].T
    if normalizer_rule == 'last':
        neighb_normk = np.zeros(len(neighb_idx), hstypes.FK_DTYPE) + (K + Knorm - 1)
    elif normalizer_rule == 'name':
        neighb_normk = get_name_normalizers(qaid, qreq_, Knorm, neighb_idx)
    elif normalizer_rule == 'external':
        pass
    else:
        raise NotImplementedError('[nn_weights] no normalizer_rule=%r' % normalizer_rule)
    return neighb_normk


def apply_normweight(normweight_fn, neighb_normk, neighb_idx, neighb_dist, Knorm):
    r"""
    helper applies the normalized weight function to one query annotation

    Args:
        normweight_fn (func):  chosen weight function e.g. lnbnn
        qaid (int):  query annotation id
        neighb_idx (ndarray[int32_t, ndims=2]):  mapping from query feature
            index to db neighbor index
        neighb_dist (ndarray):  mapping from query feature index to dist
        Knorm (int):
        qreq_ (QueryRequest):  query request object with hyper-parameters

    Returns:
        ndarray: neighb_normweight

    CommandLine:
        python -m wbia.algo.hots.nn_weights --test-apply_normweight

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> from wbia.algo.hots import nn_weights
        >>> #cfgdict = {'K':10, 'Knorm': 10, 'normalizer_rule': 'name',
        >>> #           'dim_size': 450, 'resize_dim': 'area'}
        >>> #tup = plh.testdata_pre_weight_neighbors(cfgdict=cfgdict)
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='testdb1',
        >>>                                p=['default:K=10,Knorm=10,normalizer_rule=name,dim_size=450,resize_dim=area'])
        >>> nns_list, nnvalid0_list = args
        >>> qaid = qreq_.qaids[0]
        >>> Knorm = qreq_.qparams.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> normalizer_rule  = qreq_.qparams.normalizer_rule
        >>> (neighb_idx, neighb_dist) = nns_list[0]
        >>> neighb_normk = get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule)
        >>> neighb_normweight = nn_weights.apply_normweight(
        >>>   normweight_fn, neighb_normk, neighb_idx, neighb_dist, Knorm)
        >>> ut.assert_inbounds(neighb_normweight.sum(), 600, 950)
    """
    K = len(neighb_idx.T) - Knorm
    # neighb_normdist = np.array(
    #     [dists[normk] for (dists, normk) in zip(neighb_dist, neighb_normk)])
    neighb_normdist = vt.take_col_per_row(neighb_dist, neighb_normk)
    neighb_normdist.shape = (len(neighb_idx), 1)
    neighb_nndist = neighb_dist.T[0:K].T
    vdist = neighb_nndist  # voting distance
    ndist = neighb_normdist  # normalizer distance
    neighb_normweight = normweight_fn(vdist, ndist)
    return neighb_normweight


def get_name_normalizers(qaid, qreq_, Knorm, neighb_idx):
    r"""
    helper normalizers for 'name' normalizer_rule

    Args:
        qaid (int): query annotation id
        qreq_ (wbia.QueryRequest): hyper-parameters
        Knorm (int):
        neighb_idx (ndarray):

    Returns:
        ndarray : neighb_normk

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> from wbia.algo.hots import nn_weights
        >>> #cfgdict = {'K':10, 'Knorm': 10, 'normalizer_rule': 'name'}
        >>> #tup = plh.testdata_pre_weight_neighbors(cfgdict=cfgdict)
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='testdb1',
        >>>                                p=['default:K=10,Knorm=10,normalizer_rule=name'])
        >>> nns_list, nnvalid0_list = args
        >>> Knorm = qreq_.qparams.Knorm
        >>> (neighb_idx, neighb_dist) = nns_list[0]
        >>> qaid = qreq_.qaids[0]
        >>> neighb_normk = get_name_normalizers(qaid, qreq_, Knorm, neighb_idx)
    """
    assert Knorm == qreq_.qparams.Knorm, 'inconsistency in qparams'
    # Get the top names you do not want your normalizer to be from
    # qnid = qreq_.internal_qannots.loc([qaid]).nids[0]
    qnid = qreq_.get_qreq_annot_nids(qaid)
    K = len(neighb_idx.T) - Knorm
    assert K > 0, 'K cannot be 0'
    # Get the 0th - Kth matching neighbors
    neighb_topidx = neighb_idx.T[0:K].T
    # Get tke Kth - KNth normalizing neighbors
    neighb_normidx = neighb_idx.T[-Knorm:].T
    # Apply temporary uniquish name
    neighb_topaid = qreq_.indexer.get_nn_aids(neighb_topidx)
    neighb_normaid = qreq_.indexer.get_nn_aids(neighb_normidx)
    neighb_topnid = qreq_.get_qreq_annot_nids(neighb_topaid)
    neighb_normnid = qreq_.get_qreq_annot_nids(neighb_normaid)
    # Inspect the potential normalizers
    neighb_selnorm = mark_name_valid_normalizers(qnid, neighb_topnid, neighb_normnid)
    neighb_normk = neighb_selnorm + (K + Knorm)  # convert form negative to pos indexes
    return neighb_normk


def mark_name_valid_normalizers(qnid, neighb_topnid, neighb_normnid):
    r"""
    Helper func that allows matches only to the first result for a name

    Each query feature finds its K matches and Kn normalizing matches. These
    are the candidates from which it can choose a set of matches and a single
    normalizer.

    A normalizer is marked as invalid if it belongs to a name that was also in
    its feature's candidate matching set.

    Args:
        neighb_topnid (ndarray): marks the names a feature matches
        neighb_normnid (ndarray): marks the names of the feature normalizers
        qnid (int): query name id

    Returns:
        neighb_selnorm - index of the selected normalizer for each query feature

    CommandLine:
        python -m wbia.algo.hots.nn_weights --exec-mark_name_valid_normalizers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> qnid = 1
        >>> neighb_topnid = np.array([[1, 1, 1, 1, 1],
        ...                         [1, 2, 1, 1, 1],
        ...                         [1, 2, 2, 3, 1],
        ...                         [5, 8, 9, 8, 8],
        ...                         [5, 8, 9, 8, 8],
        ...                         [6, 6, 9, 6, 8],
        ...                         [5, 8, 6, 6, 6],
        ...                         [1, 2, 8, 6, 6]], dtype=np.int32)
        >>> neighb_normnid = np.array([[ 1, 1, 1],
        ...                          [ 2, 3, 1],
        ...                          [ 2, 3, 1],
        ...                          [ 6, 6, 6],
        ...                          [ 6, 6, 8],
        ...                          [ 2, 6, 6],
        ...                          [ 6, 6, 1],
        ...                          [ 4, 4, 9]], dtype=np.int32)
        >>> neighb_selnorm = mark_name_valid_normalizers(qnid, neighb_topnid, neighb_normnid)
        >>> K = len(neighb_topnid.T)
        >>> Knorm = len(neighb_normnid.T)
        >>> neighb_normk_ = neighb_selnorm + (Knorm)  # convert form negative to pos indexes
        >>> result = str(neighb_normk_)
        >>> print(result)
        [2 1 2 0 0 0 2 0]

    Ignore:
        print(ut.doctest_repr(neighb_normnid, 'neighb_normnid', verbose=False))
        print(ut.doctest_repr(neighb_topnid, 'neighb_topnid', verbose=False))
    """
    # TODO?: warn if any([np.any(flags) for flags in neighb_invalid]), (
    #    'Normalizers are potential matches. Increase Knorm')
    neighb_valid = np.logical_and.reduce(
        [col1[:, None] != neighb_normnid for col1 in neighb_topnid.T]
    )
    # Mark self as invalid, if given that information
    neighb_valid = np.logical_and(neighb_normnid != qnid, neighb_valid)
    # For each query feature find its best normalizer (using negative indices)
    Knorm = neighb_normnid.shape[1]
    neighb_validxs = [np.nonzero(normrow)[0] for normrow in neighb_valid]
    neighb_selnorm = np.array(
        [validxs[0] - Knorm if len(validxs) != 0 else -1 for validxs in neighb_validxs],
        hstypes.FK_DTYPE,
    )
    return neighb_selnorm


@_register_nn_normalized_weight_func
def lnbnn_fn(vdist, ndist):
    r"""
    Locale Naive Bayes Nearest Neighbor weighting

    References:
        http://www.cs.ubc.ca/~lowe/papers/12mccannCVPR.pdf
        http://www.cs.ubc.ca/~sanchom/local-naive-bayes-nearest-neighbor

    Sympy:
        >>> import sympy
        >>> #https://github.com/sympy/sympy/pull/10247
        >>> from sympy import log
        >>> from sympy.stats import P, E, variance, Die, Normal, FiniteRV
        >>> C, Cbar = sympy.symbols('C Cbar')
        >>> d_i = Die(sympy.symbols('di'), 6)
        >>> log(P(di, C) / P(di, Cbar))
        >>> #
        >>> PdiC, PdiCbar = sympy.symbols('PdiC, PdiCbar')
        >>> oddsC = log(PdiC / PdiCbar)
        >>> sympy.simplify(oddsC)
        >>> import vtool as vt
        >>> vt.check_expr_eq(oddsC, log(PdiC) - log(PdiCbar))


    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> out = lnbnn_fn(vdist, ndist)
        >>> result = ut.hz_str('lnbnn  = ', ut.repr2(out, precision=2))
        >>> print(result)
        lnbnn  = np.array([[0.62, 0.22, 0.03],
                           [0.35, 0.22, 0.01],
                           [0.87, 0.58, 0.27],
                           [0.67, 0.42, 0.25],
                           [0.59, 0.3 , 0.27]])
    """
    return ndist - vdist


@_register_nn_normalized_weight_func
def ratio_fn(vdist, ndist):
    r"""
    Args:
        vdist (ndarray): voting array
        ndist (ndarray): normalizing array

    Returns:
        ndarray: out

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> out = ratio_fn(vdist, ndist)
        >>> result = ut.hz_str('ratio = ', ut.repr2(out, precision=2))
        >>> print(result)
        ratio = np.array([[0.  , 0.65, 0.95],
                          [0.33, 0.58, 0.98],
                          [0.13, 0.42, 0.73],
                          [0.15, 0.47, 0.68],
                          [0.23, 0.61, 0.65]])
    """
    return np.divide(vdist, ndist)


@_register_nn_normalized_weight_func
def bar_l2_fn(vdist, ndist):
    r"""
    The feature weight is (1 - the euclidian distance
    between the features). The normalizers are unused.

    (not really a normaalized function)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> out = bar_l2_fn(vdist, ndist)
        >>> result = ut.hz_str('barl2  = ', ut.repr2(out, precision=2))
        >>> print(result)
        barl2  = np.array([[1.  , 0.6 , 0.41],
                           [0.83, 0.7 , 0.49],
                           [0.87, 0.58, 0.27],
                           [0.88, 0.63, 0.46],
                           [0.82, 0.53, 0.5 ]])
    """
    return 1.0 - vdist


@_register_nn_normalized_weight_func
def loglnbnn_fn(vdist, ndist):
    r"""
    Ignore:
        import vtool as vt
        vt.check_expr_eq('log(d) - log(n)', 'log(d / n)')   # True
        vt.check_expr_eq('log(d) / log(n)', 'log(d - n)')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> out = loglnbnn_fn(vdist, ndist)
        >>> result = ut.hz_str('loglnbnn  = ', ut.repr2(out, precision=2))
        >>> print(result)
        loglnbnn  = np.array([[0.48, 0.2 , 0.03],
                              [0.3 , 0.2 , 0.01],
                              [0.63, 0.46, 0.24],
                              [0.51, 0.35, 0.22],
                              [0.46, 0.26, 0.24]])
    """
    return np.log(ndist - vdist + 1.0)


@_register_nn_normalized_weight_func
def logratio_fn(vdist, ndist):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> out = normonly_fn(vdist, ndist)
        >>> result = ut.repr2(out)
        >>> print(result)
        np.array([[0.62, 0.62, 0.62],
                  [0.52, 0.52, 0.52],
                  [1.  , 1.  , 1.  ],
                  [0.79, 0.79, 0.79],
                  [0.77, 0.77, 0.77]])
    """
    return np.log(np.divide(ndist, vdist + EPS) + 1.0)


@_register_nn_normalized_weight_func
def normonly_fn(vdist, ndist):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> out = normonly_fn(vdist, ndist)
        >>> result = ut.repr2(out)
        >>> print(result)
        np.array([[0.62, 0.62, 0.62],
                  [0.52, 0.52, 0.52],
                  [1.  , 1.  , 1.  ],
                  [0.79, 0.79, 0.79],
                  [0.77, 0.77, 0.77]])
    """
    return np.tile(ndist[:, 0:1], (1, vdist.shape[1]))
    # return ndist[None, 0:1]


def testdata_vn_dists(nfeats=5, K=3):
    r"""
    Test voting and normalizing distances

    Returns:
        tuple : (vdist, ndist) - test voting distances and normalizer distances

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> vdist, ndist = testdata_vn_dists()
        >>> result = (ut.hz_str('vdist = ', ut.repr2(vdist))) + '\n'
        >>> result += (ut.hz_str('ndist = ', ut.repr2(ndist)))
        vdist = np.array([[0.  , 0.4 , 0.59],
                          [0.17, 0.3 , 0.51],
                          [0.13, 0.42, 0.73],
                          [0.12, 0.37, 0.54],
                          [0.18, 0.47, 0.5 ]])
        ndist = np.array([[0.62],
                          [0.52],
                          [1.  ],
                          [0.79],
                          [0.77]])
    """

    def make_precise(dist):
        prec = 100
        dist = (prec * dist).astype(np.uint8) / prec
        dist = dist.astype(hstypes.FS_DTYPE)
        return dist

    rng = np.random.RandomState(0)
    vdist = rng.rand(nfeats, K)
    ndist = rng.rand(nfeats, 1)
    # Ensure distance increases
    vdist = vdist.cumsum(axis=1)
    ndist = (ndist.T + vdist.max(axis=1)).T
    Z = ndist.max()
    vdist = make_precise(vdist / Z)
    ndist = make_precise(ndist / Z)
    vdist[0][0] = 0
    return vdist, ndist


# @_register_nn_normalized_weight_func
# def dist_fn(vdist, ndist):
#    """ the euclidian distance between the features """
#    return vdist


# @_register_nn_simple_weight_func
def gravity_match_weighter(nns_list, nnvalid0_list, qreq_):
    raise NotImplementedError('have not finished gv weighting')
    # qfx2_nnkpts = qreq_.indexer.get_nn_kpts(qfx2_nnidx)
    # qfx2_nnori = ktool.get_oris(qfx2_nnkpts)
    # qfx2_kpts  = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.get_internal_query_config2())  # FIXME: Highly inefficient
    # qfx2_oris  = ktool.get_oris(qfx2_kpts)
    # # Get the orientation distance
    # qfx2_oridist = vt.rowwise_oridist(qfx2_nnori, qfx2_oris)
    # # Normalize into a weight (close orientations are 1, far are 0)
    # qfx2_gvweight = (TAU - qfx2_oridist) / TAU
    # # Apply gravity vector weight to the score
    # qfx2_score *= qfx2_gvweight


def all_normalized_weights_test():
    r"""
    CommandLine:
        python -m wbia.algo.hots.nn_weights --exec-all_normalized_weights_test

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> all_normalized_weights_test()
    """
    from wbia.algo.hots import nn_weights
    import six

    # ibs, qreq_, nns_list, nnvalid0_list = plh.testdata_pre_weight_neighbors()

    qreq_, args = plh.testdata_pre(
        'weight_neighbors',
        defaultdb='testdb1',
        a=['default:qindex=0:1,dindex=0:5,hackerrors=False'],
        p=['default:codename=vsmany,bar_l2_on=True,fg_on=False'],
        verbose=True,
    )
    nns_list = args.nns_list
    nnvalid0_list = args.nnvalid0_list
    qaid = qreq_.qaids[0]

    def tst_weight_fn(nn_weight, nns_list, qreq_, qaid):
        normweight_fn = nn_weights.__dict__[nn_weight + '_fn']
        weight_list1, nomx_list1 = nn_weights.nn_normalized_weight(
            normweight_fn, nns_list, nnvalid0_list, qreq_
        )
        weights1 = weight_list1[0]
        # ---
        # test NN_WEIGHT_FUNC_DICT
        # ---
        nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT[nn_weight]
        weight_list2, nomx_list2 = nn_normonly_weight(nns_list, nnvalid0_list, qreq_)
        weights2 = weight_list2[0]
        assert np.all(weights1 == weights2)
        print(nn_weight + ' passed')

    for nn_weight in six.iterkeys(nn_weights.NN_WEIGHT_FUNC_DICT):
        normweight_key = nn_weight + '_fn'
        if normweight_key not in nn_weights.__dict__:
            continue
        tst_weight_fn(nn_weight, nns_list, qreq_, qaid)


if __name__ == '__main__':
    r"""
    python -m wbia.algo.hots.nn_weights --allexamples
    python -m wbia.algo.hots.nn_weights
    """
    import multiprocessing

    multiprocessing.freeze_support()
    import utool as ut  # NOQA

    ut.doctest_funcs()
