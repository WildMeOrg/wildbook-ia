from __future__ import division, print_function
import utool
(print, print_,
 printDBG, rrr, profile) = utool.inject(__name__, '[mc3]', DEBUG=False)
# Python
from os.path import join
# Science
import numpy as np
# HotSpotter
from ibeis.dev import params
import QueryRequest
import NNIndex
import matching_functions as mf


#----------------------


def get_bigcache_io_kwargs(ibs, qreq):
    query_uid = qreq.get_query_uid(ibs, qreq._qcxs)
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
    qcx2_res = utool.smart_load(**io_kwargs)
    if verbose:
        print('query_uid = %r' % io_kwargs['uid'])
    if qcx2_res is None:
        raise IOError('bigcache_query ... miss')
    elif len(qcx2_res) != len(qreq._qcxs):
        raise IOError('bigcache_query ... outdated')
    else:
        return qcx2_res


def save_bigcache_query(qx2_res, ibs, qreq):
    io_kwargs = get_bigcache_io_kwargs(ibs, qreq)
    utool.ensuredir(io_kwargs['dpath'])
    utool.smart_save(qx2_res, **io_kwargs)


@profile
def bigcache_query(ibs, qreq, batch_size=10, use_bigcache=True,
                   limit_memory=False, verbose=True):
    qcxs = qreq._qcxs
    if use_bigcache and not params.args.nocache_query:
        try:
            qcx2_res = load_bigcache_query(ibs, qreq, verbose)
            return qcx2_res
        except IOError as ex:
            print(ex)
    # Perform checks
    pre_cache_checks(ibs, qreq)
    pre_exec_checks(ibs, qreq)
    # Execute queries in batches
    qcx2_res = {}
    nBatches = int(np.ceil(len(qcxs) / batch_size))
    batch_enum = enumerate(utool.ichunks(qcxs, batch_size))
    for batchx, qcxs_batch in batch_enum:
        print('[mc3] batch %d / %d' % (batchx, nBatches))
        qreq._qcxs = qcxs_batch
        print('qcxs_batch=%r. quid=%r' % (qcxs_batch, qreq.get_uid()))
        try:
            qcx2_res_ = process_query_request(ibs, qreq, safe=False)
            # Append current batch results if we have the memory
            if not limit_memory:
                qcx2_res.update(qcx2_res_)
        except mf.QueryException as ex:
            print('[mc3] ERROR !!!: %r' % ex)
            if params.args.strict:
                raise
            continue
    qreq._qcxs = qcxs
    # Need to reload all queries
    if limit_memory:
        qcx2_res = process_query_request(ibs, qreq, safe=False)
    save_bigcache_query(qcx2_res, ibs, qreq)
    return qcx2_res


@utool.indent_decor('[quick_ensure]')
def quickly_ensure_qreq(ibs, qcxs=None, dcxs=None):
    # This function is purely for hacking, eventually prep request or something
    # new should be good enough to where this doesnt matter
    print(' --- quick ensure qreq --- ')
    qreq = ibs.qreq
    query_cfg = ibs.prefs.query_cfg
    cids = ibs.get_indexed_sample()
    if qcxs is None:
        qcxs = cids
    if dcxs is None:
        dcxs = cids
    qreq = prep_query_request(qreq=qreq, query_cfg=query_cfg,
                              qcxs=qcxs, dcxs=dcxs)
    pre_cache_checks(ibs, qreq)
    pre_exec_checks(ibs, qreq)
    return qreq


@utool.indent_decor('[prep_qreq]')
def prep_query_request(qreq=None, query_cfg=None, qcxs=None, dcxs=None, **kwargs):
    print(' --- prep query request ---')
    # Builds or modifies a query request object
    def loggedif(msg, condition):
        # helper function for logging if statment results
        printDBG(msg + '... ' + ['no', 'yes'][condition])
        return condition
    if not loggedif('(1) given qreq?', qreq is not None):
        qreq = QueryRequest.QueryRequest()
    if loggedif('(2) given qcxs?', qcxs is not None):
        qreq._qcxs = qcxs
    if loggedif('(3) given dcxs?', dcxs is not None):
        qreq._dcxs = dcxs
    if not loggedif('(4) given qcfg?', query_cfg is not None):
        query_cfg = qreq.cfg
    if loggedif('(4) given kwargs?', len(kwargs) > 0):
        query_cfg = query_cfg.deepcopy(**kwargs)
    #
    qreq.set_cfg(query_cfg)
    #
    assert (qreq is not None), ('invalid qeury request')
    assert (qreq._qcxs is not None and len(qreq._qcxs) > 0), (
        'query request has invalid query chip indexes')
    assert (qreq._dcxs is not None and len(qreq._dcxs) > 0), (
        'query request has invalid database chip indexes')
    assert (qreq.cfg is not None), (
        'query request has invalid query config')
    return qreq


#----------------------
# Query and database checks
#----------------------


@profile
@utool.indent_decor('[pre_cache]')
def pre_cache_checks(ibs, qreq):
    print(' --- pre cache checks --- ')
    # Ensure hotspotter object is using the right config
    ibs.attatch_qreq(qreq)
    feat_uid = qreq.cfg._feat_cfg.get_uid()
    # Load any needed features or chips into memory
    if ibs.feats.feat_uid != feat_uid:
        print(' !! UNLOAD DATA !!')
        print('[mc3] feat_uid = %r' % feat_uid)
        print('[mc3] ibs.feats.feat_uid = %r' % ibs.feats.feat_uid)
        ibs.unload_cxdata('all')
    return qreq


@profile
@utool.indent_decor('[pre_exec]')
def pre_exec_checks(ibs, qreq):
    print(' --- pre exec checks ---')
    # Get qreq config information
    dcxs = qreq.get_internal_dcxs()
    feat_uid = qreq.cfg._feat_cfg.get_uid()
    dcxs_uid = utool.hashstr_arr(dcxs, 'dcxs')
    # Ensure the index / inverted index exist for this config
    dftup_uid = dcxs_uid + feat_uid
    if not dftup_uid in qreq._dftup2_index:
        print('qreq._dftup2_index[dcxs_uid]... nn_index cache miss')
        print('dftup_uid = %r' % (dftup_uid,))
        print('len(qreq._dftup2_index) = %r' % len(qreq._dftup2_index))
        print('type(qreq._dftup2_index) = %r' % type(qreq._dftup2_index))
        print('qreq = %r' % qreq)
        cid_list = np.unique(np.hstack((qreq._dcxs, qreq._qcxs)))
        ibs.refresh_features(cid_list)
        # Compute the FLANN Index
        data_index = NNIndex.NNIndex(ibs, dcxs)
        qreq._dftup2_index[dftup_uid] = data_index
    qreq._data_index = qreq._dftup2_index[dftup_uid]
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
    if safe:
        qreq = pre_cache_checks(ibs, qreq)

    # Try loading as many cached results as possible
    use_cache = not params.args.nocache_query and use_cache
    if use_cache:
        qcx2_res, failed_qcxs = mf.try_load_resdict(ibs, qreq)
    else:
        qcx2_res = {}
        failed_qcxs = qreq._qcxs

    # Execute and save queries
    if len(failed_qcxs) > 0:
        if safe:
            qreq = pre_exec_checks(ibs, qreq)
        computed_qcx2_res = execute_query_and_save_L1(ibs, qreq, failed_qcxs)
        qcx2_res.update(computed_qcx2_res)  # Update cached results
    return qcx2_res


# Query Level 1
@utool.indent_decor('[QL1]')
def execute_query_and_save_L1(ibs, qreq, failed_qcxs=[]):
    print('[q1] execute_query_and_save_L1()')
    orig_qcxs = qreq._qcxs
    if len(failed_qcxs) > 0:
        qreq._qcxs = failed_qcxs
    qcx2_res = execute_query_L0(ibs, qreq)  # Execute Queries
    for qcx, res in qcx2_res.iteritems():  # Cache Save
        res.save(ibs)
    qreq._qcxs = orig_qcxs
    return qcx2_res


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
        qcx2_res - mapping from query indexes to QueryResult Objects
    '''
    # Query Chip Indexes
    # * vsone qcxs/dcxs swapping occurs here
    qcxs = qreq.get_internal_qcxs()
    # Nearest neighbors (qcx2_nns)
    # * query descriptors assigned to database descriptors
    # * FLANN used here
    neighbs = mf.nearest_neighbors(ibs, qcxs, qreq)
    # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
    # * feature matches are weighted
    weights, filt2_meta = mf.weight_neighbors(ibs, neighbs, qreq)
    # Thresholding and weighting (qcx2_nnfilter)
    # * feature matches are pruned
    nnfiltFILT = mf.filter_neighbors(ibs, neighbs, weights, qreq)
    # Nearest neighbors to chip matches (qcx2_chipmatch)
    # * Inverted index used to create cid2_fmfsfk (TODO: ccx2_fmfv)
    # * Initial scoring occurs
    # * vsone inverse swapping occurs here
    matchesFILT = mf.build_chipmatches(ibs, neighbs, nnfiltFILT, qreq)
    # Spatial verification (qcx2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    matchesSVER = mf.spatial_verification(ibs, matchesFILT, qreq)
    # Query results format (qcx2_res) (TODO: SQL / Json Encoding)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qcx2_res = mf.chipmatch_to_resdict(ibs, matchesSVER, filt2_meta, qreq)
    return qcx2_res
