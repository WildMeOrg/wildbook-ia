from __future__ import absolute_import, division, print_function
import utool
# HotSpotter
from ibeis.dev import params
from ibeis.model.jon_recognition import QueryRequest
from ibeis.model.jon_recognition import NNIndex
from ibeis.model.jon_recognition import matching_functions as mf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[mc3]', DEBUG=False)


@utool.indent_func
def quickly_ensure_qreq(ibs, qcids=None, dcids=None):
    # This function is purely for hacking, eventually prep request or something
    # new should be good enough to where this doesnt matter
    print(' --- quick ensure qreq --- ')
    qreq = ibs.qreq
    query_cfg = ibs.prefs.query_cfg
    cids = ibs.get_recognition_database_chips()
    if qcids is None:
        qcids = cids
    if dcids is None:
        dcids = cids
    qreq = prep_query_request(qreq=qreq, query_cfg=query_cfg,
                              qcids=qcids, dcids=dcids)
    #pre_cache_checks(ibs, qreq)
    pre_exec_checks(ibs, qreq)
    return qreq


@utool.indent_func
def prep_query_request(qreq=None, query_cfg=None,
                       qcids=None, dcids=None, **kwargs):
    """  Builds or modifies a query request object """
    print(' --- Prep QueryRequest --- ')
    if qreq is None:
        qreq = QueryRequest.QueryRequest()
    if qcids is not None:
        qreq.qcids = qcids
    if dcids is not None:
        qreq.dcids = dcids
    if query_cfg is None:
        query_cfg = qreq.cfg
    if len(kwargs) > 0:
        query_cfg = query_cfg.deepcopy(**kwargs)
    qreq.set_cfg(query_cfg)
    return qreq


#----------------------
# Query and database checks
#----------------------


@utool.indent_func
@profile
def pre_exec_checks(ibs, qreq):
    """ Builds the NNIndex if not already in cache """
    print('  --- Pre Exec ---')
    # Get qreq config information
    dcids = qreq.get_internal_dcids()
    feat_uid = qreq.cfg._feat_cfg.get_uid()
    dcids_uid = utool.hashstr_arr(dcids, 'dcids')
    # Ensure the index / inverted index exist for this config
    dftup_uid = dcids_uid + feat_uid
    if not dftup_uid in qreq.dftup2_index:
        # Compute the FLANN Index
        data_index = NNIndex.NNIndex(ibs, dcids)
        qreq.dftup2_index[dftup_uid] = data_index
    qreq.data_index = qreq.dftup2_index[dftup_uid]
    return qreq


#----------------------
# Main Query Logic
#----------------------

# Query Level 2
@utool.indent_decor('[QL2]')
def process_query_request(ibs, qreq, use_cache=True, safe=True):
    '''
    The standard query interface
    '''
    print(' --- Process QueryRequest --- ')
    # HotSpotter feature checks
    #if safe:
        #qreq = pre_cache_checks(ibs, qreq)

    # Try loading as many cached results as possible
    use_cache = not params.args.nocache_query and use_cache
    if use_cache:
        qcid2_res, failed_qcids = mf.try_load_resdict(ibs, qreq)
    else:
        qcid2_res = {}
        failed_qcids = qreq.qcids

    # Execute and save queries
    if len(failed_qcids) > 0:
        if safe:
            qreq = pre_exec_checks(ibs, qreq)
        computed_qcid2_res = execute_query_and_save_L1(ibs, qreq, failed_qcids)
        qcid2_res.update(computed_qcid2_res)  # Update cached results
    return qcid2_res


# Query Level 1
@utool.indent_decor('[QL1]')
def execute_query_and_save_L1(ibs, qreq, failed_qcids=[]):
    print('[q1] execute_query_and_save_L1()')
    orig_qcids = qreq.qcids
    if len(failed_qcids) > 0:
        qreq.qcids = failed_qcids
    qcid2_res = execute_query_L0(ibs, qreq)  # Execute Queries
    for qcid, res in qcid2_res.iteritems():  # Cache Save
        res.save(ibs)
    qreq.qcids = orig_qcids
    return qcid2_res


# Query Level 0
@utool.indent_decor('[QL0]')
@profile
def execute_query_L0(ibs, qreq):
    '''
    Driver logic of query pipeline
    Input:
        ibs   - HotSpotter database object to be queried
        qreq - QueryRequest Object   # use prep_qreq to create one
    Output:
        qcid2_res - mapping from query indexes to QueryResult Objects
    '''
    # Query Chip Indexes
    # * vsone qcids/dcids swapping occurs here
    qcids = qreq.get_internal_qcids()

    # Nearest neighbors (qcid2_nns)
    # * query descriptors assigned to database descriptors
    # * FLANN used here
    qcid2_nns = mf.nearest_neighbors(
        ibs, qcids, qreq)

    # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
    # * feature matches are weighted
    filt2_weights, filt2_meta = mf.weight_neighbors(
        ibs, qcid2_nns, qreq)

    # Thresholding and weighting (qcid2_nnfilter)
    # * feature matches are pruned
    qcid2_nnfilt = mf.filter_neighbors(
        ibs, qcid2_nns, filt2_weights, qreq)

    # Nearest neighbors to chip matches (qcid2_chipmatch)
    # * Inverted index used to create cid2_fmfsfk (TODO: ccid2_fmfv)
    # * Initial scoring occurs
    # * vsone inverse swapping occurs here
    qcid2_chipmatch_FILT = mf.build_chipmatches(
        qcid2_nns, qcid2_nnfilt, qreq)

    # Spatial verification (qcid2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    qcid2_chipmatch_SVER = mf.spatial_verification(
        ibs, qcid2_chipmatch_FILT, qreq, dbginfo=False)

    # Query results format (qcid2_res) (TODO: SQL / Json Encoding)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qcid2_res = mf.chipmatch_to_resdict(
        ibs, qcid2_chipmatch_SVER, filt2_meta, qreq)

    return qcid2_res
