from __future__ import division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[mc3]', DEBUG=False)
# Python
from os.path import join
# Science
import numpy as np
# HotSpotter
from ibeis.dev import params
from ibeis.model.jon_recognition import QueryRequest
from ibeis.model.jon_recognition import NNIndex
from ibeis.model.jon_recognition import matching_functions as mf


#----------------------


def get_bigcache_io_kwargs(ibs, qreq):
    query_uid = qreq.get_query_uid(ibs, qreq.qcids)
    cache_dir = join(ibs.dirs.cache_dir, 'bigcache_query')
    io_kwargs = {
        'dpath': cache_dir,
        'fname': 'bigcache_query',
        'uid':  query_uid,
        'ext': '.cPkl'}
    return io_kwargs


def load_bigcache_query(ibs, qreq, verbose):
    # High level caching
    io_kwargs = get_bigcache_io_kwargs(ibs, qreq)
    qcid2_res = utool.smart_load(**io_kwargs)
    if verbose:
        print('query_uid = %r' % io_kwargs['uid'])
    if qcid2_res is None:
        raise IOError('bigcache_query ... miss')
    elif len(qcid2_res) != len(qreq.qcids):
        raise IOError('bigcache_query ... outdated')
    else:
        return qcid2_res


def save_bigcache_query(qx2_res, ibs, qreq):
    io_kwargs = get_bigcache_io_kwargs(ibs, qreq)
    utool.ensuredir(io_kwargs['dpath'])
    utool.smart_save(qx2_res, **io_kwargs)


@profile
def bigcache_query(ibs, qreq, batch_size=10, use_bigcache=True,
                   limit_memory=False, verbose=True):
    qcids = qreq.qcids
    if use_bigcache and not params.args.nocache_query:
        try:
            qcid2_res = load_bigcache_query(ibs, qreq, verbose)
            return qcid2_res
        except IOError as ex:
            print(ex)
    # Perform checks
    #pre_cache_checks(ibs, qreq)
    pre_exec_checks(ibs, qreq)
    # Execute queries in batches
    qcid2_res = {}
    nBatches = int(np.ceil(len(qcids) / batch_size))
    batch_enum = enumerate(utool.ichunks(qcids, batch_size))
    for batchx, qcids_batch in batch_enum:
        print('[mc3] batch %d / %d' % (batchx, nBatches))
        qreq.qcids = qcids_batch
        print('qcids_batch=%r. quid=%r' % (qcids_batch, qreq.get_uid()))
        try:
            qcid2_res_ = process_query_request(ibs, qreq, safe=False)
            # Append current batch results if we have the memory
            if not limit_memory:
                qcid2_res.update(qcid2_res_)
        except mf.QueryException as ex:
            print('[mc3] ERROR !!!: %r' % ex)
            if params.args.strict:
                raise
            continue
    qreq.qcids = qcids
    # Need to reload all queries
    if limit_memory:
        qcid2_res = process_query_request(ibs, qreq, safe=False)
    save_bigcache_query(qcid2_res, ibs, qreq)
    return qcid2_res


@utool.indent_decor('[quick_ensure]')
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


@utool.indent_decor('[prep_qreq]')
def prep_query_request(qreq=None, query_cfg=None,
                       qcids=None, dcids=None, **kwargs):
    """  Builds or modifies a query request object """
    print(' --- prep query request ---')
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


#@profile
#@utool.indent_decor('[pre_cache]')
#def pre_cache_checks(ibs, qreq):
    #print(' --- pre cache checks --- ')
    ## Ensure ibs object is using the right config
    ##ibs.attatch_qreq(qreq)
    #feat_uid = qreq.cfg._feat_cfg.get_uid()
    ## Load any needed features or chips into memory
    #if ibs.feats.feat_uid != feat_uid:
        #print(' !! UNLOAD DATA !!')
        #print('[mc3] feat_uid = %r' % feat_uid)
        #print('[mc3] ibs.feats.feat_uid = %r' % ibs.feats.feat_uid)
        #ibs.unload_ciddata('all')
    #return qreq


@profile
@utool.indent_decor('[pre_exec]')
def pre_exec_checks(ibs, qreq):
    """ Builds the NNIndex if not already in cache """
    print(' --- pre exec checks ---')
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
    print(' --- process query request --- ')
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
@profile
@utool.indent_decor('[QL0]')
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
    neighbs = mf.nearest_neighbors(ibs, qcids, qreq)
    # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
    # * feature matches are weighted
    weights, filt2_meta = mf.weight_neighbors(ibs, neighbs, qreq)
    # Thresholding and weighting (qcid2_nnfilter)
    # * feature matches are pruned
    nnfiltFILT = mf.filter_neighbors(ibs, neighbs, weights, qreq)
    # Nearest neighbors to chip matches (qcid2_chipmatch)
    # * Inverted index used to create cid2_fmfsfk (TODO: ccid2_fmfv)
    # * Initial scoring occurs
    # * vsone inverse swapping occurs here
    matchesFILT = mf.build_chipmatches(ibs, neighbs, nnfiltFILT, qreq)
    # Spatial verification (qcid2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    matchesSVER = mf.spatial_verification(ibs, matchesFILT, qreq)
    # Query results format (qcid2_res) (TODO: SQL / Json Encoding)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qcid2_res = mf.chipmatch_to_resdict(ibs, matchesSVER, filt2_meta, qreq)
    return qcid2_res
