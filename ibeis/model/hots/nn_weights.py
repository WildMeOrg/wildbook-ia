from __future__ import absolute_import, division, print_function
import utool
import six
import numpy as np
import vtool.linalg as ltool
from numpy import array
from six.moves import zip
print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[nnweight]')
import functools
from ibeis.model.hots import hstypes


def testdata_nn_weights():
    import ibeis
    from ibeis.model.hots import query_request
    from ibeis.model.hots import pipeline
    ibs = ibeis.opendb('testdb1')
    aids = ibs.get_valid_aids()
    daid_list = aids[1:5]
    qaid_list = aids[0:1]
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list)
    qreq_.lazy_load(ibs)
    qaid2_nns = pipeline.nearest_neighbors(qreq_)
    return ibs, daid_list, qaid_list, qaid2_nns, qreq_


def test_all_weights():
    from ibeis.model.hots import nn_weights
    import six
    ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
    qaid = qaid_list[0]

    def test_weight_fn(nn_weight, qaid2_nns, qreq_, qaid):
        from ibeis.model.hots import nn_weights
        #----
        normweight_fn = nn_weights.__dict__[nn_weight + '_fn']
        tup1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qreq_)
        (qaid2_weight1, qaid2_selnorms1) = tup1
        weights1 = qaid2_weight1[qaid]
        selnorms1 = qaid2_selnorms1[qaid]
        #---
        # test NN_WEIGHT_FUNC_DICT
        #---
        nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT[nn_weight]
        tup2 = nn_normonly_weight(qaid2_nns, qreq_)
        (qaid2_weight2, qaid2_selnorms2) = tup2
        selnorms2 = qaid2_selnorms2[qaid]
        weights2 = qaid2_weight2[qaid]
        assert np.all(weights1 == weights2)
        assert np.all(selnorms1 == selnorms2)
        print(nn_weight + ' passed')

    for nn_weight in six.iterkeys(nn_weights.NN_WEIGHT_FUNC_DICT):
        nn_weights.test_weight_fn(nn_weight, qaid2_nns, qreq_, qaid)


def nn_normalized_weight(normweight_fn, qaid2_nns, qreq_):
    """
    Weights nearest neighbors using the chosen function

    Args:
        normweight_fn (func): chosen weight function e.g. lnbnn
        qaid2_nns (dict): query descriptor nearest neighbors and distances. qaid -> (qfx2_nnx, qfx2_dist)
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        tuple(dict, dict) : (qaid2_weight, qaid2_selnorms)

    Example:
        >>> from ibeis.model.hots.nn_weights import *
        >>> from ibeis.model.hots import nn_weights
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
        >>> qaid = qaid_list[0]
        >>> #----
        >>> normweight_fn = lnbnn_fn
        >>> tup1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qreq_)
        >>> (qaid2_weight1, qaid2_selnorms1) = tup1
        >>> weights1 = qaid2_weight1[qaid]
        >>> selnorms1 = qaid2_selnorms1[qaid]
        >>> #---
        >>> # test NN_WEIGHT_FUNC_DICT
        >>> #---
        >>> nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT['lnbnn']
        >>> tup2 = nn_normonly_weight(qaid2_nns, qreq_)
        >>> (qaid2_weight2, qaid2_selnorms2) = tup2
        >>> selnorms2 = qaid2_selnorms2[qaid]
        >>> weights2 = qaid2_weight2[qaid]
        >>> assert np.all(weights1 == weights2)
        >>> assert np.all(selnorms1 == selnorms2)

    Ignore:
        #from ibeis.model.hots import neighbor_index as hsnbrx
        #nnindexer = hsnbrx.new_ibeis_nnindexer(ibs, daid_list)
    """
    #utool.stash_testdata('qaid2_nns')
    #
    K = qreq_.qparams.K

    Knorm = qreq_.qparams.Knorm
    rule  = qreq_.qparams.normalizer_rule
    # Prealloc output
    qaid2_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    qaid2_selnorms = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        # Apply normalized weights
        (qfx2_normweight, qfx2_normmeta) = apply_normweight(
            normweight_fn, qaid, qfx2_idx, qfx2_dist, rule, K, Knorm, qreq_)
        # Output
        qaid2_weight[qaid]   = qfx2_normweight
        qaid2_selnorms[qaid] = qfx2_normmeta
    return (qaid2_weight, qaid2_selnorms)


def apply_normweight(normweight_fn, qaid, qfx2_idx, qfx2_dist, rule, K, Knorm, qreq_):
    """
    helper: applies the normalized weight function to one query annotation

    Args:
        normweight_fn (func): chosen weight function e.g. lnbnn
        qaid (int): query annotation id
        qfx2_idx (ndarray):
        qfx2_dist (ndarray):
        rule (str):
        K (int):
        Knorm (int):
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        tuple(ndarray, ndarray) : (qfx2_normweight, qfx2_normmeta)

    Example:
        >>> from ibeis.model.hots.nn_weights import *
        >>> from ibeis.model.hots import nn_weights
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
        >>> qaid = qaid_list[0]
        >>> K = ibs.cfg.query_cfg.nn_cfg.K
        >>> Knorm = ibs.cfg.query_cfg.nn_cfg.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> rule  = qreq_.qparams.normalizer_rule
        >>> (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        >>> tup = nn_weights.apply_normweight(normweight_fn, qaid, qfx2_idx, qfx2_dist, rule, K, Knorm, qreq_)

    Timeits:
        %timeit qfx2_dist.T[0:K].T
        %timeit qfx2_dist[:, 0:K]

    """

    qfx2_nndist = qfx2_dist.T[0:K].T
    if rule == 'last':
        # Normalizers for 'last' rule
        qfx2_normk = np.zeros(len(qfx2_dist), hstypes.FK_DTYPE) + (K + Knorm - 1)
    elif rule == 'name':
        # Normalizers for 'name' rule
        qfx2_normk = get_name_normalizers(qaid, qreq_, K, Knorm, qfx2_idx)
    else:
        raise NotImplementedError('[nn_weights] no rule=%r' % rule)
    qfx2_normdist = [dists[normk] for (dists, normk) in zip(qfx2_dist, qfx2_normk)]
    qfx2_normidx  = [idxs[normk] for (idxs, normk) in zip(qfx2_idx, qfx2_normk)]
    # build meta
    qfx2_normmeta = [(qreq_.indexer.get_nn_aids(idx), qreq_.indexer.get_nn_featxs(idx), normk)
                     for (normk, idx) in zip(qfx2_normk, qfx2_normidx)]
    qfx2_normdist = array(qfx2_normdist)
    qfx2_normidx  = array(qfx2_normidx)
    qfx2_normmeta = array(qfx2_normmeta)
    # Ensure shapes are valid
    qfx2_normdist.shape = (len(qfx2_idx), 1)
    qfx2_normweight = normweight_fn(qfx2_nndist, qfx2_normdist)
    return (qfx2_normweight, qfx2_normmeta)


def get_name_normalizers(qaid, qreq_, K, Knorm, qfx2_idx):
    """
    helper: normalizers for 'name' rule

    Args:
        qaid (int): query annotation id
        qreq_ (QueryRequest): hyper-parameters
        K (int):
        Knorm (int):
        qfx2_idx (ndarray):

    Returns:
        ndarray : qfx2_normk

    Example:
        >>> from ibeis.model.hots.nn_weights import *
        >>> from ibeis.model.hots import nn_weights
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
        >>> qaid = qaid_list[0]
        >>> K = ibs.cfg.query_cfg.nn_cfg.K
        >>> Knorm = ibs.cfg.query_cfg.nn_cfg.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        >>> qfx2_nndist = qfx2_dist.T[0:K].T

    """
    # Get the top names you do not want your normalizer to be from
    qnid = qreq_.get_annot_nids(qaid)
    nTop = max(1, K)
    qfx2_topidx = qfx2_idx.T[0:nTop].T
    qfx2_normidx = qfx2_idx.T[-Knorm:].T
    # Apply temporary uniquish name
    qfx2_topaid  = qreq_.indexer.get_nn_aids(qfx2_topidx)
    qfx2_normaid = qreq_.indexer.get_nn_aids(qfx2_normidx)
    qfx2_topnid  = qreq_.get_annot_nids(qfx2_topaid)
    qfx2_normnid = qreq_.get_annot_nids(qfx2_normaid)
    # Inspect the potential normalizers
    qfx2_normk = mark_name_valid_normalizers(qfx2_normnid, qfx2_topnid, qnid)
    qfx2_normk += (K + Knorm)  # convert form negative to pos indexes
    return qfx2_normk


def mark_name_valid_normalizers(qfx2_normnid, qfx2_topnid, qnid=None):
    """
    helper: Allows matches only to the first result of a given name

    Args:
        qfx2_normnid (ndarray):
        qfx2_topnid (ndarray):
        qnid (int): query name id

    Returns:
        qfx2_selnorm
    """
    #columns = qfx2_topnid
    #matrix = qfx2_normnid
    Kn = qfx2_normnid.shape[1]
    # Find the positions in the normalizers that could be valid (assumes Knorm > 1)
    # compare_matrix_columns is probably inefficient
    qfx2_valid = True - ltool.compare_matrix_columns(qfx2_normnid, qfx2_topnid)

    if qnid is not None:
        # Mark self as invalid, if given that information
        qfx2_valid = np.logical_and(qfx2_normnid != qnid, qfx2_valid)

    # For each query feature find its best normalizer (using negative indices)
    qfx2_validlist = [np.where(normrow)[0] for normrow in qfx2_valid]
    qfx2_selnorm = array([poslist[0] - Kn if len(poslist) != 0 else -1 for
                          poslist in qfx2_validlist], hstypes.FK_DTYPE)
    return qfx2_selnorm


NN_WEIGHT_FUNC_DICT = {}
EPS = 1E-8


def _regweight_decor(func):
    """
    Decorator for weighting functions

    Registers a nearest neighbor weighting
    """
    global NN_WEIGHT_FUNC_DICT
    #filtfunc = functools.partial(nn_normalized_weight, func, *args)
    nnweight = utool.get_funcname(func).replace('_fn', '').lower()
    if utool.VERBOSE:
        print('[nn_weights] registering func: %r' % (nnweight,))
    filtfunc = functools.partial(nn_normalized_weight, func)
    NN_WEIGHT_FUNC_DICT[nnweight] = filtfunc
    return func


@_regweight_decor
def lnbnn_fn(vdist, ndist):
    """
    Locale Naive Bayes Nearest Neighbor weighting

    Example:
        >>> import numpy as np
        >>> ndist = np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5], [3, 4, 5],  [9, 7, 6] ])
        >>> vdist = np.array([[3, 2, 1, 5], [3, 2, 5, 6], [3, 4, 5, 3], [3, 4, 5, 8],  [9, 7, 6, 3] ])
        >>> vdist1 = vdist[:,0:1]
        >>> vdist2 = vdist[:,0:2]
        >>> vdist3 = vdist[:,0:3]
        >>> vdist4 = vdist[:,0:4]
        >>> print(lnbnn_fn(vdist1, ndist))
        >>> print(lnbnn_fn(vdist2, ndist))
        >>> print(lnbnn_fn(vdist3, ndist))
        >>> print(lnbnn_fn(vdist4, ndist))
    """
    return (ndist - vdist)  # / 1000.0


@_regweight_decor
def loglnbnn_fn(vdist, ndist):
    return np.log(ndist - vdist + 1.0)  # / 1000.0


@_regweight_decor
def ratio_fn(vdist, ndist):
    return np.divide(ndist, vdist + EPS)


@_regweight_decor
def logratio_fn(vdist, ndist):
    return np.log(np.divide(ndist, vdist + EPS) + 1.0)


@_regweight_decor
def normonly_fn(vdist, ndist):
    return np.tile(ndist[:, 0:1], (1, vdist.shape[1]))


#nn_filt_weight_fmtstr = utool.codeblock(
#    '''
#    nn_{filt}_weight({filter}, *args):
#        return nn_normalized_weight({filter}_fn)
#    '''
#)


#import utool as ut
#filt_dict_fmtstr = ut.codeblock('''
#    NN_WEIGHT_FUNC_DICT[{filt}] = nn_{filt}_weight
#    ''')

#NN_WEIGHT_LIST = ['lograt', 'lnbnn', 'ratio' 'logdist', 'crowded']

#weight_funcstrs = [nn_filt_weight_fmtstr.format(filt=filt) for filt in NN_WEIGHT_LIST]
#weight_funcstrs = [nn_filt_weight_fmtstr.format(filt=filt) for filt in NN_WEIGHT_LIST]
#for funcstr in weight_funcstrs:
#    exec(funcstr)


#def nn_ratio_weight(*args):
#    return nn_normalized_weight(RATIO_fn, *args)


#def nn_lnbnn_weight(*args):
#    return nn_normalized_weight(LNBNN_fn, *args)


#def nn_lograt_weight(*args):
#    return nn_normalized_weight(LOGRAT_fn, *args)


#def nn_logdist_weight(*args):
#    return nn_normalized_weight(LOGDIST_fn, *args)


#def nn_crowded_weight(*args):
#    return nn_normalized_weight(CROWDED_fn, *args)


# TODO: Make a more elegant way of mapping weighting parameters to weighting
# function. A dict is better than eval, but there may be a better way.
#NN_WEIGHT_FUNC_DICT = {
    #'lograt':  nn_lograt_weight,
    #'lnbnn':   nn_lnbnn_weight,
    #'ratio':   nn_ratio_weight,
    #'logdist': nn_logdist_weight,
    #'crowded': nn_crowded_weight,
#}


# normweight_fn = LNBNN_fn
