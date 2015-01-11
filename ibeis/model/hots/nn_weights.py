from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import six
import numpy as np
import vtool.linalg as ltool
from six.moves import zip
import functools
from ibeis.model.hots import hstypes
print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[nnweight]')


NN_WEIGHT_FUNC_DICT = {}
MISC_WEIGHT_FUNC_DICT = {}
EPS = 1E-8


def testdata_nn_weights(dbname='testdb1', qaid_list=None, daid_list=None, cfgdict={}):
    """
    >>> dbname = 'testdb1'
    >>> cfgdict = {'fg_weight': 1.0}
    """
    from ibeis.model.hots import pipeline
    ibs, qreq_ = pipeline.get_pipeline_testdata(dbname=dbname,
                                                qaid_list=qaid_list,
                                                daid_list=daid_list,
                                                cfgdict=cfgdict)
    pipeline_locals_ = pipeline.testrun_pipeline_upto(qreq_, 'weight_neighbors')
    qaid2_nns     = pipeline_locals_['qaid2_nns']
    qaid2_nnvalid0 = pipeline_locals_['qaid2_nnvalid0']
    return ibs, qreq_, qaid2_nns, qaid2_nnvalid0


def _register_nn_normalized_weight_func(func):
    """
    Decorator for weighting functions

    Registers a nearest neighbor normalized weighting
    """
    global NN_WEIGHT_FUNC_DICT
    filtkey = utool.get_funcname(func).replace('_fn', '').lower()
    if utool.VERBOSE:
        print('[nn_weights] registering norm func: %r' % (filtkey,))
    filtfunc = functools.partial(nn_normalized_weight, func)
    NN_WEIGHT_FUNC_DICT[filtkey] = filtfunc
    return func


def _register_nn_simple_weight_func(func):
    filtkey = utool.get_funcname(func).replace('_match_weighter', '').lower()
    if utool.VERBOSE:
        print('[nn_weights] registering simple func: %r' % (filtkey,))
    NN_WEIGHT_FUNC_DICT[filtkey] = func
    return func


def _register_misc_weight_func(func):
    filtkey = utool.get_funcname(func).replace('_match_weighter', '').lower()
    if utool.VERBOSE:
        print('[nn_weights] registering simple func: %r' % (filtkey,))
    MISC_WEIGHT_FUNC_DICT[filtkey] = func
    return func


@_register_nn_simple_weight_func
def dupvote_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_):
    """
    dupvotes gives duplicate name votes a weight close to 0.

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
        >>> #tup = nn_weights.testdata_nn_weights('testdb1', slice(0, 1), slice(0, 11))
        >>> dbname = 'testdb1'  # 'GZ_ALL'  # 'testdb1'
        >>> cfgdict = dict(K=10, Knorm=10, codename='nsum')
        >>> tup = nn_weights.testdata_nn_weights(dbname, cfgdict=cfgdict)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> # Test Function Call
        >>> qaid2_dupvote_weight = nn_weights.dupvote_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_)
        >>> # Check consistency
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> qfx2_dupvote_weight = qaid2_dupvote_weight[qaid]
        >>> flags = qfx2_dupvote_weight  > .5
        >>> qfx2_topnid = ibs.get_annot_name_rowids(qreq_.indexer.get_nn_aids(qaid2_nns[qaid][0]))
        >>> isunique_list = [ut.isunique(row[flag]) for row, flag in zip(qfx2_topnid, flags)]
        >>> assert all(isunique_list), 'dupvote should only allow one vote per name'

    CommandLine:
        ./dev.py -t nsum --db GZ_ALL --show --va -w --qaid 1032
        ./dev.py -t nsum_nosv --db GZ_ALL --show --va -w --qaid 1032

    """
    # Prealloc output
    K = qreq_.qparams.K
    qaid2_dupvote_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        qfx2_valid0 = qaid2_nnvalid0[qaid]
        if len(qfx2_valid0) == 0:
            # hack for empty query features (should never happen, but it
            # inevitably will)
            qaid2_dupvote_weight[qaid] = np.empty((0, K), dtype=np.float32)
            continue
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_topidx = qfx2_idx.T[0:K].T
        qfx2_topaid = qreq_.indexer.get_nn_aids(qfx2_topidx)
        qfx2_topnid = qreq_.ibs.get_annot_name_rowids(qfx2_topaid)
        # Don't let current query count as a valid match
        # Change those names to the unused name
        # qfx2_topnid[qfx2_topaid == qaid] = 0
        qfx2_invalid0 = np.bitwise_not(qfx2_valid0)
        qfx2_topnid[qfx2_invalid0] = 0
        # A duplicate vote is when any vote for a name after the first
        qfx2_isnondup = np.array([ut.flag_unique_items(topnids) for topnids in qfx2_topnid])
        # set invalids to be duplicates as well (for testing)
        qfx2_isnondup[qfx2_invalid0] = False
        qfx2_dupvote_weight = (qfx2_isnondup.astype(np.float32) * (1 - 1E-7)) + 1E-7
        qaid2_dupvote_weight[qaid] = qfx2_dupvote_weight
    return qaid2_dupvote_weight


def componentwise_uint8_dot(qfx2_qvec, qfx2_dvec):
    """ a dot product is a componentwise multiplication of
    two vector and then a sum. Do that for arbitary vectors.
    Remember to cast uint8 to float32 and then divide by 255**2.
    BUT THESE ARE SIFT DESCRIPTORS WHICH USE THE SMALL UINT8 TRICK
    DIVIDE BY 512**2 instead
    """
    arr1 = qfx2_qvec.astype(np.float32)
    arr2 = qfx2_dvec.astype(np.float32)
    cosangle = np.multiply(arr1, arr2).sum(axis=-1).T / hstypes.PSEUDO_UINT8_MAX_SQRD
    return cosangle


@_register_nn_simple_weight_func
def cos_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_):
    r"""

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> cfgdict = dict(cos_weight=1.0, K=10, Knorm=10)
        >>> tup = nn_weights.testdata_nn_weights('PZ_MTEST', cfgdict=cfgdict)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> assert qreq_.qparams.cos_weight == 1, 'bug setting custom params cos_weight'
        >>> qaid2_cos_weight = nn_weights.cos_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_)


    Dev::
        qnid = ibs.get_annot_name_rowids(qaid)
        qfx2_nids = ibs.get_annot_name_rowids(qreq_.indexer.get_nn_aids(qfx2_idx.T[0:K].T))

        # remove first match
        qfx2_nids_ = qfx2_nids.T[1:].T
        qfx2_cos_  = qfx2_cos.T[1:].T

        # flags of unverified 'correct' matches
        qfx2_samename = qfx2_nids_ == qnid

        for k in [1, None]:
            for alpha in [.01, .1, 1, 3, 10, 20, 50]:
                print('-------')
                print('alpha = %r' % alpha)
                print('k = %r' % k)
                qfx2_cosweight = np.multiply(np.sign(qfx2_cos_), np.power(qfx2_cos_, alpha))
                if k is None:
                    qfx2_weight = qfx2_cosweight
                    flag = qfx2_samename
                else:
                    qfx2_weight = qfx2_cosweight.T[0:k].T
                    flag = qfx2_samename.T[0:k].T
                #print(qfx2_weight)
                #print(flag)
                good_stats_ = ut.get_stats(qfx2_weight[flag])
                bad_stats_ = ut.get_stats(qfx2_weight[~flag])
                print('good_matches = ' + ut.dict_str(good_stats_))
                print('bad_matchees = ' + ut.dict_str(bad_stats_))
                print('diff_mean = ' + str(good_stats_['mean'] - bad_stats_['mean']))

    """
    # Prealloc output
    K = qreq_.qparams.K
    qaid2_cos_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    # Database feature index to chip index
    for qaid in six.iterkeys(qaid2_nns):
        (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        qfx2_qvec = qreq_.ibs.get_annot_vecs(qaid)[np.newaxis, :, :]
        # database forground weights
        qfx2_dvec = qreq_.indexer.get_nn_vecs(qfx2_idx.T[0:K])
        # Component-wise dot product
        qfx2_cos = componentwise_uint8_dot(qfx2_qvec, qfx2_dvec)
        # Selectivity function
        alpha = 3
        qfx2_cosweight = np.multiply(np.sign(qfx2_cos), np.power(qfx2_cos, alpha))
        qaid2_cos_weight[qaid] = qfx2_cosweight
    return qaid2_cos_weight


@_register_nn_simple_weight_func
def fg_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_):
    r"""
    foreground feature match weighting

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> cfgdict = dict(featweight_on=True, fg_weight=1.0)
        >>> tup = nn_weights.testdata_nn_weights('PZ_MTEST', cfgdict=cfgdict)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> print(ut.dict_str(qreq_.qparams.__dict__, sorted_=True))
        >>> assert qreq_.qparams.featweight_on == True, 'bug setting custom params featweight_on'
        >>> assert qreq_.qparams.fg_weight == 1, 'bug setting custom params fg_weight'
        >>> qaid2_fgvote_weight = nn_weights.fg_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_)
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


@_register_misc_weight_func
def distinctiveness_match_weighter(qreq_):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> cfgdict = dict(featweight_on=True, fg_weight=1.0, codename='vsone_dist_extern_distinctiveness')
        >>> tup = nn_weights.testdata_nn_weights('PZ_MTEST', cfgdict=cfgdict)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup

    TODO: finish intergration
    """
    dstcnvs_normer = qreq_.dstcnvs_normer
    assert dstcnvs_normer is not None
    qaid_list = qreq_.get_external_qaids()
    vecs_list = qreq_.ibs.get_annot_vecs(qaid_list)
    dstcvs_list = []
    for vecs in vecs_list:
        qfx2_vec = vecs
        dstcvs = dstcnvs_normer.get_distinctiveness(qfx2_vec)
        dstcvs_list.append(dstcvs)
    qaid2_dstcvs = dict(zip(qaid_list, dstcvs_list))
    return qaid2_dstcvs


def nn_normalized_weight(normweight_fn, qaid2_nns, qaid2_nnvalid0, qreq_):
    """
    Generic function to weight nearest neighbors

    ratio, lnbnn, and other nearest neighbor based functions use this

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
        >>> dbname = 'PZ_MTEST'
        >>> tup = nn_weights.testdata_nn_weights(dbname)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> qaid = qreq_.get_external_daids()[0]
        >>> normweight_fn = lnbnn_fn
        >>> qaid2_weight1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qaid2_nnvalid0, qreq_)
        >>> weights1 = qaid2_weight1[qaid]
        >>> nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT['lnbnn']
        >>> qaid2_weight2 = nn_normonly_weight(qaid2_nns, qaid2_nnvalid0, qreq_)
        >>> weights2 = qaid2_weight2[qaid]
        >>> assert np.all(weights1 == weights2)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> dbname = 'PZ_MTEST'
        >>> tup = nn_weights.testdata_nn_weights(dbname)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> qaid = qreq_.get_external_daids()[0]
        >>> normweight_fn = ratio_fn
        >>> qaid2_weight1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qaid2_nnvalid0, qreq_)
        >>> weights1 = qaid2_weight1[qaid]
        >>> nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT['ratio']
        >>> qaid2_weight2 = nn_normonly_weight(qaid2_nns, qaid2_nnvalid0, qreq_)
        >>> weights2 = qaid2_weight2[qaid]
        >>> assert np.all(weights1 == weights2)

    Ignore:
        #from ibeis.model.hots import neighbor_index as hsnbrx
        #nnindexer = hsnbrx.request_ibeis_nnindexer(qreq_)
    """
    #utool.stash_testdata('qaid2_nns')
    #
    K = qreq_.qparams.K

    Knorm = qreq_.qparams.Knorm
    rule  = qreq_.qparams.normalizer_rule
    with_metadata = qreq_.qparams.with_metadata
    #normweight_upper_bound = 30  # TODO:  make this specific to each normweight func

    # Prealloc output
    qaid2_weight = {qaid: None for qaid in six.iterkeys(qaid2_nns)}
    if with_metadata:
        metadata = qreq_.metadata
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

        #qfx2_normweight[qfx2_normweight > normweight_upper_bound] = normweight_upper_bound
        #qfx2_normweight /= normweight_upper_bound

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
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> cfgdict = {'K':10, 'Knorm': 10, 'normalizer_rule': 'name'}
        >>> tup = nn_weights.testdata_nn_weights(cfgdict=cfgdict)
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> K = qreq_.qparams.K
        >>> Knorm = qreq_.qparams.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> rule  = qreq_.qparams.normalizer_rule
        >>> (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        >>> with_metadata = True
        >>> metakey_metadata = {}
        >>> qfx2_normweight = nn_weights.apply_normweight(normweight_fn, qaid, qfx2_idx,
        ...         qfx2_dist, rule, K, Knorm, qreq_, with_metadata,
        ...         metakey_metadata)

    Timeits:
        %timeit qfx2_dist.T[0:K].T
        %timeit qfx2_dist[:, 0:K]

    Ignore:
        print('\n'.join(((
        '>>> ndist = np.' + np.array_repr(ndist.T).replace(' ...,', '') + '.T'),
        '>>> vdist = np.' + np.array_repr(vdist.T).replace(' ...,', '') + '.T')
        ))

    """

    qfx2_nndist = qfx2_dist.T[0:K].T
    if rule == 'last':
        # Normalizers for 'last' rule
        qfx2_normk = np.zeros(len(qfx2_dist), hstypes.FK_DTYPE) + (K + Knorm - 1)
    elif rule == 'name':
        # Normalizers for 'name' rule
        qfx2_normk = get_name_normalizers(qaid, qreq_, K, Knorm, qfx2_idx)
    elif rule == 'external':
        pass
    else:
        raise NotImplementedError('[nn_weights] no rule=%r' % rule)
    qfx2_normdist = np.array([dists[normk]
                              for (dists, normk) in zip(qfx2_dist, qfx2_normk)])
    qfx2_normidx  = np.array([idxs[normk]
                              for (idxs, normk) in zip(qfx2_idx, qfx2_normk)])
    # Ensure shapes are valid
    qfx2_normdist.shape = (len(qfx2_idx), 1)
    vdist = qfx2_nndist    # voting distance
    ndist = qfx2_normdist  # normalizer distance
    qfx2_normweight = normweight_fn(vdist, ndist)
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
        >>> tup = nn_weights.testdata_nn_weights()
        >>> ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
        >>> qaid = qreq_.get_external_daids()[0]
        >>> K = ibs.cfg.query_cfg.nn_cfg.K
        >>> Knorm = ibs.cfg.query_cfg.nn_cfg.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> (qfx2_idx, qfx2_dist) = qaid2_nns[qaid]
        >>> qfx2_nndist = qfx2_dist.T[0:K].T

    """
    # Get the top names you do not want your normalizer to be from
    qnid = qreq_.ibs.get_annot_name_rowids(qaid)
    nTop = max(1, K)
    # Get the 0th - Kth matching neighbors
    qfx2_topidx = qfx2_idx.T[0:nTop].T
    # Get tke Kth - KNth normalizing neighbors
    qfx2_normidx = qfx2_idx.T[-Knorm:].T
    # Apply temporary uniquish name
    qfx2_topaid  = qreq_.indexer.get_nn_aids(qfx2_topidx)
    qfx2_normaid = qreq_.indexer.get_nn_aids(qfx2_normidx)
    qfx2_topnid  = qreq_.ibs.get_annot_name_rowids(qfx2_topaid)
    qfx2_normnid = qreq_.ibs.get_annot_name_rowids(qfx2_normaid)
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
    return (ndist - vdist)


@_register_nn_normalized_weight_func
def loglnbnn_fn(vdist, ndist):
    return np.log(ndist - vdist + 1.0)  # / 1000.0


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
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> vdist = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.]], dtype=np.float32).T
        >>> ndist = np.array([[  60408.,   61594.,  111387., 120138., 124307.,  125625.]], dtype=np.float32).T
        >>> out = ratio_fn(vdist, ndist)
        >>> result = np.array_repr(out.T, precision=2)
        >>> print(result)
        array([[ 0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> vdist = np.array([[  79260.,  138617.,   47964.,  127839.,  123543.,  112204.]], dtype=np.float32).T
        >>> ndist = np.array([[  83370.,  146245.,  128620.,  129598.,  126165.,  124761.]], dtype=np.float32).T
        >>> out = ratio_fn(vdist, ndist)
        >>> result = np.array_repr(out.T, precision=2)
        >>> print(result)
        array([[ 0.95,  0.95,  0.37,  0.99,  0.98,  0.9 ]], dtype=float32)
    """
    return np.divide(vdist, ndist)


@_register_nn_normalized_weight_func
def dist_fn(vdist, ndist):
    """ just use straight up distance """
    return vdist


@_register_nn_normalized_weight_func
def logratio_fn(vdist, ndist):
    return np.log(np.divide(ndist, vdist + EPS) + 1.0)


@_register_nn_normalized_weight_func
def normonly_fn(vdist, ndist):
    return np.tile(ndist[:, 0:1], (1, vdist.shape[1]))
    #return ndist[None, 0:1]


#@_register_nn_simple_weight_func
def gravity_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_):
    raise NotImplementedError('have not finished gv weighting')
    #qfx2_nnkpts = qreq_.indexer.get_nn_kpts(qfx2_nnidx)
    #qfx2_nnori = ktool.get_oris(qfx2_nnkpts)
    #qfx2_kpts  = qreq_.ibs.get_annot_kpts(qaid)  # FIXME: Highly inefficient
    #qfx2_oris  = ktool.get_oris(qfx2_kpts)
    ## Get the orientation distance
    #qfx2_oridist = vt.rowwise_oridist(qfx2_nnori, qfx2_oris)
    ## Normalize into a weight (close orientations are 1, far are 0)
    #qfx2_gvweight = (TAU - qfx2_oridist) / TAU
    ## Apply gravity vector weight to the score
    #qfx2_score *= qfx2_gvweight


def test_all_normalized_weights():
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> test_all_normalized_weights()
    """
    from ibeis.model.hots import nn_weights
    import six
    tup = nn_weights.testdata_nn_weights()
    ibs, qreq_, qaid2_nns, qaid2_nnvalid0 = tup
    qaid = qreq_.get_external_qaids()[0]

    def test_weight_fn(nn_weight, qaid2_nns, qreq_, qaid):
        from ibeis.model.hots import nn_weights
        #----
        normweight_fn = nn_weights.__dict__[nn_weight + '_fn']
        qaid2_weight1 = nn_weights.nn_normalized_weight(normweight_fn, qaid2_nns, qaid2_nnvalid0, qreq_)
        weights1 = qaid2_weight1[qaid]
        #---
        # test NN_WEIGHT_FUNC_DICT
        #---
        nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT[nn_weight]
        qaid2_weight2 = nn_normonly_weight(qaid2_nns, qaid2_nnvalid0, qreq_)
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
    python -m ibeis.model.hots.nn_weights --allexamples
    python -m ibeis.model.hots.nn_weights
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
