from __future__ import absolute_import, division, print_function
import utool
import utool as ut
#import six
import numpy as np
import vtool as vt
import functools
from ibeis.model.hots import scoring
#from ibeis.model.hots import name_scoring
from ibeis.model.hots import hstypes
from ibeis.model.hots import _pipeline_helpers as plh
from six.moves import zip
print, print_,  printDBG, rrr, profile = utool.inject(__name__, '[nnweight]')


NN_WEIGHT_FUNC_DICT = {}
MISC_WEIGHT_FUNC_DICT = {}
EPS = 1E-8


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


def componentwise_uint8_dot(qfx2_qvec, qfx2_dvec):
    """ a dot product is a componentwise multiplication of
    two vector and then a sum. Do that for arbitary vectors.
    Remember to cast uint8 to float32 and then divide by 255**2.
    BUT THESE ARE SIFT DESCRIPTORS WHICH USE THE SMALL UINT8 TRICK
    DIVIDE BY 512**2 instead
    """
    arr1 = qfx2_qvec.astype(hstypes.FS_DTYPE)
    arr2 = qfx2_dvec.astype(hstypes.FS_DTYPE)
    cosangle = vt.componentwise_dot(arr1, arr2) / hstypes.PSEUDO_UINT8_MAX_SQRD
    return cosangle


@_register_nn_simple_weight_func
def cos_match_weighter(nns_list, nnvalid0_list, qreq_):
    r"""

    CommandLine:
        python -m ibeis.model.hots.nn_weights --test-cos_match_weighter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = plh.testdata_pre_weight_neighbors('PZ_MTEST', cfgdict=dict(cos_on=True, K=10, Knorm=10))
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> assert qreq_.qparams.cos_on, 'bug setting custom params cos_weight'
        >>> cos_weight_list = nn_weights.cos_match_weighter(nns_list, nnvalid0_list, qreq_)

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
    Knorm = qreq_.qparams.Knorm
    cos_weight_list = []
    qaid_list = qreq_.get_internal_qaids()
    # Database feature index to chip index
    for qaid, nns in zip(qaid_list, nns_list):
        (qfx2_idx, qfx2_dist) = nns
        qfx2_qvec = qreq_.ibs.get_annot_vecs(qaid, config2_=qreq_.get_internal_query_config2())[np.newaxis, :, :]
        # database forground weights
        # avoid using K due to its more dynamic nature by using -Knorm
        qfx2_dvec = qreq_.indexer.get_nn_vecs(qfx2_idx.T[:-Knorm])
        # Component-wise dot product + selectivity function
        alpha = 3.0
        qfx2_cosweight = scoring.sift_selectivity_score(qfx2_qvec, qfx2_dvec, alpha)
        cos_weight_list.append(qfx2_cosweight)
    return cos_weight_list


@_register_nn_simple_weight_func
def fg_match_weighter(nns_list, nnvalid0_list, qreq_):
    r"""
    foreground feature match weighting

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> print(ut.dict_str(qreq_.qparams.__dict__, sorted_=True))
        >>> assert qreq_.qparams.fg_on == True, 'bug setting custom params fg_on'
        >>> fgvotes_list = fg_match_weighter(nns_list, nnvalid0_list, qreq_)
    """
    # Prealloc output
    Knorm = qreq_.qparams.Knorm
    fgvotes_list = []
    qaid_list = qreq_.get_internal_qaids()
    # Database feature index to chip index
    for qaid, nns in zip(qaid_list, nns_list):
        (qfx2_idx, qfx2_dist) = nns
        # database forground weights
        qfx2_dfgw = qreq_.indexer.get_nn_fgws(qfx2_idx.T[0:-Knorm].T)
        # query forground weights
        qfx2_qfgw = qreq_.ibs.get_annot_fgweights([qaid], ensure=False, config2_=qreq_.get_internal_query_config2())[0]
        # feature match forground weight
        qfx2_fgvote_weight = np.sqrt(qfx2_qfgw[:, None] * qfx2_dfgw)
        fgvotes_list.append(qfx2_fgvote_weight)
    return fgvotes_list


@_register_misc_weight_func
def distinctiveness_match_weighter(qreq_):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = plh.testdata_pre_weight_neighbors('PZ_MTEST', codename='vsone_dist_extern_distinctiveness')
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup

    TODO: finish intergration
    """
    dstcnvs_normer = qreq_.dstcnvs_normer
    assert dstcnvs_normer is not None
    qaid_list = qreq_.get_external_qaids()
    vecs_list = qreq_.ibs.get_annot_vecs(qaid_list, config2_=qreq_.get_internal_query_config2())
    dstcvs_list = []
    for vecs in vecs_list:
        qfx2_vec = vecs
        dstcvs = dstcnvs_normer.get_distinctiveness(qfx2_vec)
        dstcvs_list.append(dstcvs)
    return dstcvs_list


def nn_normalized_weight(normweight_fn, nns_list, nnvalid0_list, qreq_):
    """
    Generic function to weight nearest neighbors

    ratio, lnbnn, and other nearest neighbor based functions use this

    Args:
        normweight_fn (func): chosen weight function e.g. lnbnn
        nns_list (dict): query descriptor nearest neighbors and distances. (qfx2_nnx, qfx2_dist)
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        dict: weights_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> normweight_fn = lnbnn_fn
        >>> weights_list1 = nn_weights.nn_normalized_weight(normweight_fn, nns_list, nnvalid0_list, qreq_)
        >>> weights1 = weights_list1[0]
        >>> nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT['lnbnn']
        >>> weights_list2 = nn_normonly_weight(nns_list, nnvalid0_list, qreq_)
        >>> weights2 = weights_list2[0]
        >>> assert np.all(weights1 == weights2)
        >>> ut.assert_inbounds(weights1.sum(), 250, 300)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> normweight_fn = ratio_fn
        >>> weights_list1 = nn_weights.nn_normalized_weight(normweight_fn, nns_list, nnvalid0_list, qreq_)
        >>> weights1 = weights_list1[0]
        >>> nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT['ratio']
        >>> weights_list2 = nn_normonly_weight(nns_list, nnvalid0_list, qreq_)
        >>> weights2 = weights_list2[0]
        >>> assert np.all(weights1 == weights2)
        >>> ut.assert_inbounds(weights1.sum(), 3000, 4000)

    Ignore:
        #from ibeis.model.hots import neighbor_index as hsnbrx
        #nnindexer = hsnbrx.request_ibeis_nnindexer(qreq_)
    """
    #utool.stash_testdata('nns_list')
    #
    #Knorm = qreq_.qparams.Knorm
    Knorm = qreq_.qparams.Knorm
    normalizer_rule  = qreq_.qparams.normalizer_rule
    #with_metadata = qreq_.qparams.with_metadata
    #normweight_upper_bound = 30  # TODO:  make this specific to each normweight func

    # Prealloc output
    weight_list = []
    #if with_metadata:
    #    metadata = qreq_.metadata
    #    metakey = ut.get_funcname(normweight_fn) + '_norm_meta'
    #    metadata[metakey] = {}
    #    metakey_metadata = metadata[metakey]
    #else:
    #    metakey_metadata = None
    # Database feature index to chip index
    qaid_list = qreq_.get_internal_qaids()
    for qaid, nns in zip(qaid_list, nns_list):
        (qfx2_idx, qfx2_dist) = nns
        # Apply normalized weights
        qfx2_normweight = apply_normweight(
            normweight_fn, qaid, qfx2_idx, qfx2_dist, normalizer_rule, Knorm, qreq_)
        #with_metadata, metakey_metadata)
        #qfx2_normweight[qfx2_normweight > normweight_upper_bound] = normweight_upper_bound
        #qfx2_normweight /= normweight_upper_bound
        # Output
        weight_list.append(qfx2_normweight)
    return weight_list


def apply_normweight(normweight_fn, qaid, qfx2_idx, qfx2_dist, normalizer_rule, Knorm,
                     qreq_):
    #, with_metadata, metakey_metadata):
    """
    helper: applies the normalized weight function to one query annotation

    Returns:
        ndarray: qfx2_normweight

    Args:
        normweight_fn (func):  chosen weight function e.g. lnbnn
        qaid (int):  query annotation id
        qfx2_idx (ndarray[int32_t, ndims=2]):  mapping from query feature index to db neighbor index
        qfx2_dist (ndarray):  mapping from query feature index to dist
        normalizer_rule (str):
        Knorm (int):
        qreq_ (QueryRequest):  query request object with hyper-parameters

    CommandLine:
        python -m ibeis.model.hots.nn_weights --test-apply_normweight

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> cfgdict = {'K':10, 'Knorm': 10, 'normalizer_rule': 'name'}
        >>> tup = plh.testdata_pre_weight_neighbors(cfgdict=cfgdict)
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> Knorm = qreq_.qparams.Knorm
        >>> normweight_fn = lnbnn_fn
        >>> normalizer_rule  = qreq_.qparams.normalizer_rule
        >>> (qfx2_idx, qfx2_dist) = nns_list[0]
        >>> qfx2_normweight = nn_weights.apply_normweight(normweight_fn, qaid, qfx2_idx,
        ...         qfx2_dist, normalizer_rule, Knorm, qreq_)
        >>> ut.assert_inbounds(qfx2_normweight.sum(), 850, 950)
    """
    K = len(qfx2_idx.T) - Knorm
    assert K > 0, 'K cannot be 0'
    qfx2_nndist = qfx2_dist.T[0:K].T
    if normalizer_rule == 'last':
        # Normalizers for 'last' normalizer_rule
        qfx2_normk = np.zeros(len(qfx2_dist), hstypes.FK_DTYPE) + (K + Knorm - 1)
    elif normalizer_rule == 'name':
        # Normalizers for 'name' normalizer_rule
        qfx2_normk = get_name_normalizers(qaid, qreq_, Knorm, qfx2_idx)
    elif normalizer_rule == 'external':
        pass
    else:
        raise NotImplementedError('[nn_weights] no normalizer_rule=%r' % normalizer_rule)
    qfx2_normdist = np.array([dists[normk]
                              for (dists, normk) in zip(qfx2_dist, qfx2_normk)])
    #qfx2_normidx  = np.array([idxs[normk]
    #                          for (idxs, normk) in zip(qfx2_idx, qfx2_normk)])
    # Ensure shapes are valid
    qfx2_normdist.shape = (len(qfx2_idx), 1)
    vdist = qfx2_nndist    # voting distance
    ndist = qfx2_normdist  # normalizer distance
    qfx2_normweight = normweight_fn(vdist, ndist)
    # build meta
    #if with_metadata:
    #    normmeta_header = ('normalizer_metadata', ['norm_aid', 'norm_fx', 'norm_k'])
    #    qfx2_normmeta = np.array(
    #        [
    #            (qreq_.indexer.get_nn_aids(idx), qreq_.indexer.get_nn_featxs(idx), normk)
    #            for (normk, idx) in zip(qfx2_normk, qfx2_normidx)
    #        ]
    #    )
    #    metakey_metadata[qaid] = (normmeta_header, qfx2_normmeta)
    return qfx2_normweight


def get_name_normalizers(qaid, qreq_, Knorm, qfx2_idx):
    """
    helper: normalizers for 'name' normalizer_rule

    Args:
        qaid (int): query annotation id
        qreq_ (QueryRequest): hyper-parameters
        Knorm (int):
        qfx2_idx (ndarray):

    Returns:
        ndarray : qfx2_normk

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> from ibeis.model.hots import nn_weights
        >>> cfgdict = {'K':10, 'Knorm': 10, 'normalizer_rule': 'name'}
        >>> tup = plh.testdata_pre_weight_neighbors(cfgdict=cfgdict)
        >>> ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> Knorm = qreq_.qparams.Knorm
        >>> (qfx2_idx, qfx2_dist) = nns_list[0]
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> qfx2_normk = get_name_normalizers(qaid, qreq_, Knorm, qfx2_idx)

    """
    assert Knorm == qreq_.qparams.Knorm, 'inconsistency in qparams'
    # Get the top names you do not want your normalizer to be from
    qnid = qreq_.ibs.get_annot_name_rowids(qaid)
    K = len(qfx2_idx.T) - Knorm
    assert K > 0, 'K cannot be 0'
    # Get the 0th - Kth matching neighbors
    qfx2_topidx = qfx2_idx.T[0:K].T
    # Get tke Kth - KNth normalizing neighbors
    qfx2_normidx = qfx2_idx.T[-Knorm:].T
    # Apply temporary uniquish name
    qfx2_topaid  = qreq_.indexer.get_nn_aids(qfx2_topidx)
    qfx2_normaid = qreq_.indexer.get_nn_aids(qfx2_normidx)
    qfx2_topnid  = qreq_.ibs.get_annot_name_rowids(qfx2_topaid)
    qfx2_normnid = qreq_.ibs.get_annot_name_rowids(qfx2_normaid)
    # Inspect the potential normalizers
    qfx2_selnorm = mark_name_valid_normalizers(qnid, qfx2_topnid, qfx2_normnid)
    qfx2_normk = qfx2_selnorm + (K + Knorm)  # convert form negative to pos indexes
    return qfx2_normk


def mark_name_valid_normalizers(qnid, qfx2_topnid, qfx2_normnid):
    """
    helper: Allows matches only to the first result of a given name

    Each query feature finds its K matches and Kn normalizing matches. These are the
    candidates from which it can choose a set of matches and a single normalizer.

    A normalizer is marked as invalid if it belongs to a name that was also in its
    feature's candidate matching set.


    Args:
        qfx2_topnid (ndarray): marks the names a feature matches
        qfx2_normnid (ndarray): marks the names of the feature normalizers
        qnid (int): query name id

    Ignore:
        print(ut.doctest_repr(qfx2_normnid, 'qfx2_normnid', verbose=False))
        print(ut.doctest_repr(qfx2_topnid, 'qfx2_topnid', verbose=False))


    Returns:
        qfx2_selnorm - index of the selected normalizer for each query feature

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.nn_weights import *  # NOQA
        >>> qnid = 1
        >>> qfx2_topnid = np.array([[1, 1, 1, 1, 1],
        ...                         [1, 2, 1, 1, 1],
        ...                         [1, 2, 2, 3, 1],
        ...                         [5, 8, 9, 8, 8],
        ...                         [5, 8, 9, 8, 8],
        ...                         [6, 6, 9, 6, 8],
        ...                         [5, 8, 6, 6, 6],
        ...                         [1, 2, 8, 6, 6]], dtype=np.int32)
        >>> qfx2_normnid = np.array([[ 1, 1, 1],
        ...                          [ 2, 3, 1],
        ...                          [ 2, 3, 1],
        ...                          [ 6, 6, 6],
        ...                          [ 6, 6, 8],
        ...                          [ 2, 6, 6],
        ...                          [ 6, 6, 1],
        ...                          [ 4, 4, 9]], dtype=np.int32)
        >>> qfx2_selnorm = mark_name_valid_normalizers(qnid, qfx2_topnid, qfx2_normnid)
        >>> K = len(qfx2_topnid.T)
        >>> Knorm = len(qfx2_normnid.T)
        >>> qfx2_normk_ = qfx2_selnorm + (Knorm)  # convert form negative to pos indexes
        >>> result = str(qfx2_normk_)
        >>> print(result)
        [2 1 2 0 0 0 2 0]
    """
    # Your normalizer should be from a name that is not in any of the top
    # matches if possible. If not possible it should be from the name with the
    # highest k value.

    #%timeit np.vstack([vt.get_uncovered_mask(normnids, topnids) for topnids, normnids in zip(qfx2_topnid, qfx2_normnid)])
    #%timeit vt.compare_matrix_columns(qfx2_normnid, qfx2_topnid)
    #""" matrix = qfx2_normnid; columns = qfx2_topnid; row_matrix = matrix.T; row_list = columns.T; """
    # Find the positions in the normalizers that could be valid (assumes Knorm > 1)
    # wow, this actually seems to work an is efficient. I hardly understand the code I write.
    # takes each column in topnid and comparses it to each column in in qfx2_normnid
    # Taking the logical or of all of these results gives you a matrix with the
    # shape of qfx2_normnid that is True where a normalizing feature's name
    # appears anywhere in the corresponding row of qfx2_topnid
    qfx2_valid = np.logical_not(vt.compare_matrix_columns(qfx2_normnid, qfx2_topnid, comp_op=np.equal, logic_op=np.logical_or))

    #if qnid is not None:
    # Mark self as invalid, if given that information
    qfx2_valid = np.logical_and(qfx2_normnid != qnid, qfx2_valid)

    # For each query feature find its best normalizer (using negative indices)
    Knorm = qfx2_normnid.shape[1]
    qfx2_validxs = [np.where(normrow)[0] for normrow in qfx2_valid]
    qfx2_selnorm = np.array([validxs[0] - Knorm if len(validxs) != 0 else -1 for
                             validxs in qfx2_validxs], hstypes.FK_DTYPE)
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
def gravity_match_weighter(nns_list, nnvalid0_list, qreq_):
    raise NotImplementedError('have not finished gv weighting')
    #qfx2_nnkpts = qreq_.indexer.get_nn_kpts(qfx2_nnidx)
    #qfx2_nnori = ktool.get_oris(qfx2_nnkpts)
    #qfx2_kpts  = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.get_internal_query_config2())  # FIXME: Highly inefficient
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
    ibs, qreq_, nns_list, nnvalid0_list = plh.testdata_pre_weight_neighbors()
    qaid = qreq_.get_external_qaids()[0]

    def test_weight_fn(nn_weight, nns_list, qreq_, qaid):
        from ibeis.model.hots import nn_weights
        #----
        normweight_fn = nn_weights.__dict__[nn_weight + '_fn']
        weight_list1 = nn_weights.nn_normalized_weight(normweight_fn, nns_list, nnvalid0_list, qreq_)
        weights1 = weight_list1[0]
        #---
        # test NN_WEIGHT_FUNC_DICT
        #---
        nn_normonly_weight = nn_weights.NN_WEIGHT_FUNC_DICT[nn_weight]
        weight_list2 = nn_normonly_weight(nns_list, nnvalid0_list, qreq_)
        weights2 = weight_list2[0]
        assert np.all(weights1 == weights2)
        print(nn_weight + ' passed')

    for nn_weight in six.iterkeys(nn_weights.NN_WEIGHT_FUNC_DICT):
        normweight_key = nn_weight + '_fn'
        if normweight_key not in nn_weights.__dict__:
            continue
        test_weight_fn(nn_weight, nns_list, qreq_, qaid)


if __name__ == '__main__':
    """
    python -m ibeis.model.hots.nn_weights --allexamples
    python -m ibeis.model.hots.nn_weights
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
