from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import six
import numpy as np
import vtool.linalg as ltool
from six.moves import zip
print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[nnweight]')
import functools
from ibeis.model.hots import hstypes


NN_WEIGHT_FUNC_DICT = {}
EPS = 1E-8


def _register_nn_normalized_weight_func(func):
    """
    Decorator for weighting functions

    Registers a nearest neighbor normalized weighting
    """
    global NN_WEIGHT_FUNC_DICT
    nnweight = utool.get_funcname(func).replace('_fn', '').lower()
    if utool.VERBOSE:
        print('[nn_weights] registering norm func: %r' % (nnweight,))
    filtfunc = functools.partial(nn_normalized_weight, func)
    NN_WEIGHT_FUNC_DICT[nnweight] = filtfunc
    return func


def _register_nn_simple_weight_func(func):
    nnweight = utool.get_funcname(func).replace('_match_weighter', '').lower()
    if utool.VERBOSE:
        print('[nn_weights] registering simple func: %r' % (nnweight,))
    NN_WEIGHT_FUNC_DICT[nnweight] = func
    return func


@_register_nn_simple_weight_func
def dupvote_match_weighter(qaid2_nns, qreq_, metadata):
    """
    Each query feature is only allowed to vote for each name at most once.
    IE: a query feature can vote for multiple names, but it cannot vote
    for the same name twice.

    CommandLine:
        python dev.py --allgt -t best --db PZ_MTEST
        python dev.py --allgt -t nsum --db PZ_MTEST
        python dev.py --allgt -t dupvote --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = nn_weights.testdata_nn_weights('testdb1', slice(0, 1), slice(0, 11))
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = tup
        >>> metadata = {}
        >>> # Test Function Call
        >>> qaid2_dupvote_weight = dupvote_match_weighter(qaid2_nns, qreq_, metadata)
        >>> # Check consistency
        >>> qaid = qaid_list[0]
        >>> flags = qaid2_dupvote_weight[qaid] > .5
        >>> qfx2_topnid = ibs.get_annot_nids(qreq_.indexer.get_nn_aids(qaid2_nns[qaid][0]))
        >>> isunique_list = [ut.isunique(row[flag]) for row, flag in zip(qfx2_topnid, flags)]
        >>> assert all(isunique_list), 'dupvote should only allow one vote per name'
    """
    # Prealloc output
    K = qreq_.qparams.K
    qaid2_dupvote_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_topidx = qfx2_idx.T[0:K].T
        qfx2_topaid = qreq_.indexer.get_nn_aids(qfx2_topidx)
        qfx2_topnid = qreq_.get_annot_nids(qfx2_topaid)
        # Don't let current query count as a valid match
        # Change those names to the unused name
        qfx2_topnid[qfx2_topaid == qaid] = 0

        # A duplicate vote is when any vote for a name after the first
        qfx2_isdupvote =  np.array([ut.flag_unique_items(topnids) for topnids in qfx2_topnid])
        qfx2_dupvote_weight = qfx2_isdupvote.astype(np.float32) * (1 - EPS) + EPS
        qaid2_dupvote_weight[qaid] = qfx2_dupvote_weight
    return qaid2_dupvote_weight


@_register_nn_simple_weight_func
def fg_match_weighter(qaid2_nns, qreq_, metadata):
    r"""
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = nn_weights.testdata_nn_weights(custom_qparams=dict(featweight_on=True, fg_weight=1.0))
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = tup
        >>> print(ut.dict_str((qreq_.qparams.__dict__)))
        >>> assert qreq_.qparams.featweight_on == True, 'bug setting custom params featweight_on'
        >>> assert qreq_.qparams.fg_weight == 1, 'bug setting custom params fg_weight'
        >>> metadata = {}
        >>> qaid2_fgvote_weight = fg_match_weighter(qaid2_nns, qreq_, metadata)
    """
    # Prealloc output
    K = qreq_.qparams.K
    qaid2_fgvote_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        # database forground weights
        qfx2_dfgw = qreq_.indexer.get_nn_fgws(qfx2_idx.T[0:K].T)
        # query forground weights
        qfx2_qfgw = qreq_.ibs.get_annot_fgweights([qaid], ensure=False)[0]
        # feature match forground weight
        qfx2_fgvote_weight = np.sqrt(qfx2_qfgw[:, None] * qfx2_dfgw)
        qaid2_fgvote_weight[qaid] = qfx2_fgvote_weight
    return qaid2_fgvote_weight


def nn_normalized_weight(normweight_fn, qaid2_nns, qreq_, metadata):
    """
    Weights nearest neighbors using the chosen function

    Args:
        normweight_fn (func): chosen weight function e.g. lnbnn
        qaid2_nns (dict): query descriptor nearest neighbors and distances. qaid -> (qfx2_nnx, qfx2_dist)
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        dict: qaid2_weight

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
        >>> qaid = qaid_list[0]
        >>> normweight_fn = lnbnn_fn
        >>> metadata = {}
        >>> qaid2_weight1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qreq_, metadata)
        >>> weights1 = qaid2_weight1[qaid]
        >>> nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT['lnbnn']
        >>> qaid2_weight2 = nn_normonly_weight(qaid2_nns, qreq_, metadata)
        >>> weights2 = qaid2_weight2[qaid]
        >>> assert np.all(weights1 == weights2)

    Ignore:
        #from ibeis.model.hots import neighbor_index as hsnbrx
        #nnindexer = hsnbrx.new_ibeis_nnindexer(ibs, daid_list)
    """
    #utool.stash_testdata('qaid2_nns')
    #
    K = qreq_.qparams.K

    Knorm = qreq_.qparams.Knorm
    rule  = qreq_.qparams.normalizer_rule
    with_metadata = qreq_.qparams.with_metadata
    # Prealloc output
    qaid2_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    if with_metadata:
        metakey = ut.get_funcname(normweight_fn) + '_norm_meta'
        metadata[metakey] = {}
        metakey_metadata = metadata[metakey]
    else:
        metakey_metadata = None
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        # Apply normalized weights
        qfx2_normweight = apply_normweight(
            normweight_fn, qaid, qfx2_idx, qfx2_dist, rule, K, Knorm, qreq_,
            with_metadata, metakey_metadata)
        # Output
        qaid2_weight[qaid] = qfx2_normweight
    return qaid2_weight


def apply_normweight(normweight_fn, qaid, qfx2_idx, qfx2_dist, rule, K, Knorm,
                     qreq_, with_metadata, metakey_metadata):
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
        with_metadata (bool):
        metadata (dict):

    Returns:
        ndarray: qfx2_normweight

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
        >>> qaid = qaid_list[0]
        >>> K = ibs.cfg.query_cfg.nn_cfg.K
        >>> Knorm = ibs.cfg.query_cfg.nn_cfg.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> rule  = qreq_.qparams.normalizer_rule
        >>> (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        >>> with_metadata = True
        >>> metakey_metadata = {}
        >>> tup = nn_weights.apply_normweight(normweight_fn, qaid, qfx2_idx,
        ...         qfx2_dist, rule, K, Knorm, qreq_, with_metadata,
        ...         metakey_metadata)

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
    qfx2_normdist = np.array([dists[normk]
                              for (dists, normk) in zip(qfx2_dist, qfx2_normk)])
    qfx2_normidx  = np.array([idxs[normk]
                              for (idxs, normk) in zip(qfx2_idx, qfx2_normk)])
    # Ensure shapes are valid
    qfx2_normdist.shape = (len(qfx2_idx), 1)
    qfx2_normweight = normweight_fn(qfx2_nndist, qfx2_normdist)
    # build meta
    if with_metadata:
        normmeta_header = ('normalizer_metadata', ['norm_aid', 'norm_fx', 'norm_k'])
        qfx2_normmeta = np.array(
            [
                (qreq_.indexer.get_nn_aids(idx), qreq_.indexer.get_nn_featxs(idx), normk)
                for (normk, idx) in zip(qfx2_normk, qfx2_normidx)
            ]
        )
        metakey_metadata[qaid] = (normmeta_header, qfx2_normmeta)
    return qfx2_normweight


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
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
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
        qfx2_selnorm - index of the selected normalizer for each query feature
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
    qfx2_selnorm = np.array([poslist[0] - Kn if len(poslist) != 0 else -1 for
                             poslist in qfx2_validlist], hstypes.FK_DTYPE)
    return qfx2_selnorm


@_register_nn_normalized_weight_func
def lnbnn_fn(vdist, ndist):
    """
    Locale Naive Bayes Nearest Neighbor weighting

    Example:
        >>> # ENABLE_DOCTEST
        >>> import numpy as np
        >>> ndists = np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5], [3, 4, 5],  [9, 7, 6]])
        >>> ndist = ndists.T[0:1].T
        >>> vdist = np.array([[3, 2, 1, 5], [3, 2, 5, 6], [3, 4, 5, 3], [3, 4, 5, 8],  [9, 7, 6, 3] ])
        >>> vdist1 = vdist[:, 0:1]
        >>> vdist2 = vdist[:, 0:2]
        >>> vdist3 = vdist[:, 0:3]
        >>> vdist4 = vdist[:, 0:4]
        >>> print(lnbnn_fn(vdist1, ndist))
        >>> print(lnbnn_fn(vdist2, ndist))
        >>> print(lnbnn_fn(vdist3, ndist))
        >>> print(lnbnn_fn(vdist4, ndist))
    """
    return (ndist - vdist) / 1000.0


@_register_nn_normalized_weight_func
def loglnbnn_fn(vdist, ndist):
    return np.log(ndist - vdist + 1.0)  # / 1000.0


@_register_nn_normalized_weight_func
def ratio_fn(vdist, ndist):
    return np.divide(ndist, vdist + EPS)


@_register_nn_normalized_weight_func
def logratio_fn(vdist, ndist):
    return np.log(np.divide(ndist, vdist + EPS) + 1.0)


@_register_nn_normalized_weight_func
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


#def nn_ratio_weight(qaid2_nns, qreq_):
#    return nn_normalized_weight(RATIO_fn, qaid2_nns, qreq_)


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


def testdata_nn_weights(dbname='testdb1', qaid_slice=slice(0, 1), daid_slice=slice(0, 5), custom_qparams={}):
    """
    >>> # ibs.cfg.query_cfg.filt_cfg.fg_weight = 1
    >>> qaid_slice=slice(0, 1)
    >>> daid_slice=slice(0, 5)
    >>> dbname = 'testdb1'
    >>> custom_qparams = {'fg_weight': 1.0}
    """
    assert isinstance(dbname, str), 'dbname is not string. instead=%r' % (dbname,)
    import ibeis
    from ibeis.model.hots import query_request
    from ibeis.model.hots import pipeline
    ibs = ibeis.opendb(dbname)
    aids = ibs.get_valid_aids()
    daid_list = aids[daid_slice]
    qaid_list = aids[qaid_slice]
    #ibs.cfg.query_cfg.filt_cfg.fg_weight = 1
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, custom_qparams)
    qreq_.lazy_load(ibs)
    metadata = {}
    qaid2_nns = pipeline.nearest_neighbors(qreq_, metadata)
    return ibs, daid_list, qaid_list, qaid2_nns, qreq_


def test_all_normalized_weights():
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> test_all_normalized_weights()
    """
    from ibeis.model.hots import nn_weights
    import six
    ibs, daid_list, qaid_list, qaid2_nns, qreq_ = nn_weights.testdata_nn_weights()
    qaid = qaid_list[0]

    def test_weight_fn(nn_weight, qaid2_nns, qreq_, qaid):
        from ibeis.model.hots import nn_weights
        #----
        metadata = {}
        normweight_fn = nn_weights.__dict__[nn_weight + '_fn']
        qaid2_weight1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qreq_, metadata)
        weights1 = qaid2_weight1[qaid]
        #---
        # test NN_WEIGHT_FUNC_DICT
        #---
        nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT[nn_weight]
        qaid2_weight2 = nn_normonly_weight(qaid2_nns, qreq_, metadata)
        weights2 = qaid2_weight2[qaid]
        assert np.all(weights1 == weights2)
        print(nn_weight + ' passed')

    for nn_weight in six.iterkeys(nn_weights.NN_WEIGHT_FUNC_DICT):
        normweight_key = nn_weight + '_fn'
        if normweight_key not in nn_weights.__dict__:
            continue
        test_weight_fn(nn_weight, qaid2_nns, qreq_, qaid)


if __name__ == '__main__':
    """
    python utool/util_tests.py
    python -c "import utool, ibeis; utool.doctest_funcs(module=ibeis.model.hots.nn_weights, needs_enable=False)"
    python ibeis/model/hots/nn_weights.py --allexamples
    python ibeis/model/hots/nn_weights.py
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
