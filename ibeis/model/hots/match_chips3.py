from __future__ import absolute_import, division, print_function
import utool
# HotSpotter
from ibeis.dev import params
from ibeis.model.hots import QueryRequest
from ibeis.model.hots import NNIndex
from ibeis.model.hots import matching_functions as mf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[mc3]', DEBUG=False)


@utool.indent_func
def quickly_ensure_qreq(ibs, qrids=None, drids=None):
    # This function is purely for hacking, eventually prep request or something
    # new should be good enough to where this doesnt matter
    print(' --- quick ensure qreq --- ')
    ibs._init_query_requestor()
    qreq = ibs.qreq
    query_cfg = ibs.cfg.query_cfg
    rids = ibs.get_recognition_database_rois()
    if qrids is None:
        qrids = rids
    if drids is None:
        drids = rids
    qreq = prep_query_request(qreq=qreq, query_cfg=query_cfg,
                              qrids=qrids, drids=drids)
    pre_exec_checks(ibs, qreq)
    return qreq


@utool.indent_func('[prep_qreq]')
def prep_query_request(qreq=None, query_cfg=None,
                       qrids=None, drids=None, **kwargs):
    """  Builds or modifies a query request object """
    print(' --- Prep QueryRequest --- ')
    if qreq is None:
        qreq = QueryRequest.QueryRequest()
    if qrids is not None:
        assert len(qrids) > 0, 'cannot query nothing!'
        qreq.qrids = qrids
    if drids is not None:
        assert len(drids) > 0, 'cannot search nothing!'
        qreq.drids = drids
    if query_cfg is None:
        query_cfg = qreq.cfg
    if len(kwargs) > 0:
        query_cfg = query_cfg.deepcopy(**kwargs)
    qreq.set_cfg(query_cfg)
    return qreq


#----------------------
# Query and database checks
#----------------------


@utool.indent_func('[pre_exec]')
#@profile
def pre_exec_checks(ibs, qreq):
    """ Builds the NNIndex if feature of the correct configuration are not in
    the cache """
    print('  --- Pre Exec ---')
    # Get qreq config information
    drids = qreq.get_internal_drids()
    feat_uid = qreq.cfg._feat_cfg.get_uid()
    drids_uid = utool.hashstr_arr(drids, 'drids')
    # Ensure the index / inverted index exist for this config
    dftup_uid = drids_uid + feat_uid
    if not dftup_uid in qreq.dftup2_index:
        # Compute the FLANN Index
        data_index = NNIndex.NNIndex(ibs, drids)
        qreq.dftup2_index[dftup_uid] = data_index
    qreq.data_index = qreq.dftup2_index[dftup_uid]
    return qreq


#----------------------
# Main Query Logic
#----------------------

# Query Level 2
@utool.indent_func('[QL2]')
def process_query_request(ibs, qreq, use_cache=True, safe=True):
    """
    The standard query interface
    """
    print(' --- Process QueryRequest --- ')
    # Try loading as many cached results as possible
    use_cache = not params.args.nocache_query and use_cache
    if use_cache:
        qrid2_res, failed_qrids = mf.try_load_resdict(qreq)
    else:
        qrid2_res = {}
        failed_qrids = qreq.qrids

    # Execute and save queries
    if len(failed_qrids) > 0:
        if safe:
            qreq = pre_exec_checks(ibs, qreq)
        computed_qrid2_res = execute_query_and_save_L1(ibs, qreq, failed_qrids)
        qrid2_res.update(computed_qrid2_res)  # Update cached results
    return qrid2_res


# Query Level 1
@utool.indent_func('[QL1]')
def execute_query_and_save_L1(ibs, qreq, failed_qrids=[]):
    print('[q1] execute_query_and_save_L1()')
    orig_qrids = qreq.qrids
    if len(failed_qrids) > 0:
        qreq.qrids = failed_qrids
    qrid2_res = execute_query_L0(ibs, qreq)  # Execute Queries
    for qrid, res in qrid2_res.iteritems():  # Cache Save
        res.save(ibs)
    qreq.qrids = orig_qrids
    return qrid2_res


# Query Level 0
@utool.indent_func('[QL0]')
@profile
def execute_query_L0(ibs, qreq):
    """
    Driver logic of query pipeline
    Input:
        ibs   - HotSpotter database object to be queried
        qreq - QueryRequest Object   # use prep_qreq to create one
    Output:
        qrid2_res - mapping from query indexes to QueryResult Objects
    """
    # Query Chip Indexes
    # * vsone qrids/drids swapping occurs here
    qrids = qreq.get_internal_qrids()

    # Nearest neighbors (qrid2_nns)
    # * query descriptors assigned to database descriptors
    # * FLANN used here
    qrid2_nns = mf.nearest_neighbors(
        ibs, qrids, qreq)

    # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
    # * feature matches are weighted
    filt2_weights, filt2_meta = mf.weight_neighbors(
        ibs, qrid2_nns, qreq)

    # Thresholding and weighting (qrid2_nnfilter)
    # * feature matches are pruned
    qrid2_nnfilt = mf.filter_neighbors(
        ibs, qrid2_nns, filt2_weights, qreq)

    # Nearest neighbors to chip matches (qrid2_chipmatch)
    # * Inverted index used to create rid2_fmfsfk (TODO: crid2_fmfv)
    # * Initial scoring occurs
    # * vsone inverse swapping occurs here
    qrid2_chipmatch_FILT = mf.build_chipmatches(
        qrid2_nns, qrid2_nnfilt, qreq)

    # Spatial verification (qrid2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    qrid2_chipmatch_SVER = mf.spatial_verification(
        ibs, qrid2_chipmatch_FILT, qreq, dbginfo=False)

    # Query results format (qrid2_res) (TODO: SQL / Json Encoding)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qrid2_res = mf.chipmatch_to_resdict(
        ibs, qrid2_chipmatch_SVER, filt2_meta, qreq)

    return qrid2_res
