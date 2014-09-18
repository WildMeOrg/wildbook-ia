from __future__ import absolute_import, division, print_function
import utool
import sys
import six
from ibeis.model.hots import hots_query_request
from ibeis.model.hots import hots_nn_index
from ibeis.model.hots import matching_functions as mf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[mc3]', DEBUG=False)


USE_CACHE = '--nocache-query' not in sys.argv
USE_BIGCACHE = '--nocache-big' not in sys.argv


@utool.indent_func
@profile
def quickly_ensure_qreq(ibs, qaids=None, daids=None):
    # This function is purely for hacking, eventually prep request or something
    # new should be good enough to where this doesnt matter
    if utool.NOT_QUIET:
        print(' --- quick ensure qreq --- ')
    ibs._init_query_requestor()
    qreq = ibs.qreq
    query_cfg = ibs.cfg.query_cfg
    aids = ibs.get_recognition_database_aids()
    if qaids is None:
        qaids = aids
    if daids is None:
        daids = aids
    qreq = prep_query_request(qreq=qreq, query_cfg=query_cfg,
                              qaids=qaids, daids=daids)
    pre_exec_checks(ibs, qreq)
    return qreq


#@utool.indent_func('[prep_qreq]')
@profile
def prep_query_request(qreq=None, query_cfg=None,
                       qaids=None, daids=None, **kwargs):
    """  Builds or modifies a query request object
     """
    # Ugg, what does this even do!?
    # TODO: Cleanup
    if utool.NOT_QUIET:
        print(' --- Prep QueryRequest --- ')
    if qreq is None:
        qreq = hots_query_request.QueryRequest()
    if qaids is not None:
        assert len(qaids) > 0, 'cannot query nothing!'
        qreq.qaids = qaids
    if daids is not None:
        assert len(daids) > 0, 'cannot search nothing!'
        qreq.daids = daids
    if query_cfg is None:
        # this seems broken
        query_cfg = qreq.cfg
    if len(kwargs) > 0:
        # gross
        query_cfg = query_cfg.deepcopy(**kwargs)
    qreq.set_cfg(query_cfg)
    return qreq


#----------------------
# Query and database checks
#----------------------


#@utool.indent_func('[pre_exec]')
#@profile
@profile
def pre_exec_checks(ibs, qreq):
    """
    Makes sure that the daids are indexed for nn_search
     """
    if utool.NOT_QUIET:
        print('  --- Pre Exec ---')
    feat_cfgstr = qreq.cfg._feat_cfg.get_cfgstr()
    daids_hashid = qreq.get_daids_hashid()
    # Ensure the index / inverted index exist for this config
    dftup_hashid = daids_hashid + feat_cfgstr
    if dftup_hashid not in qreq.dftup2_index:
        # Get qreq config information
        daids = qreq.get_internal_daids()
        # Compute the FLANN Index
        data_index = hots_nn_index.HOTSIndex(ibs, daids)
        # Release all memory if over the limit
        # This is a hack and not the best way to do this.
        if len(qreq.dftup2_index) > qreq.cache_limit:
            if utool.NOT_QUIET:
                print('[mc3] Clearing NNIndex Cache')
            qreq.dftup2_index = {}
        qreq.dftup2_index[dftup_hashid] = data_index
    qreq.data_index = qreq.dftup2_index[dftup_hashid]
    return qreq


#----------------------
# Main Query Logic
#----------------------

# Query Level 2
#@utool.indent_func('[Q2]')
@profile
def process_query_request(ibs, qreq,
                          safe=True,
                          use_cache=USE_CACHE,
                          use_bigcache=USE_BIGCACHE):
    """
    The standard query interface.
    INPUT:
        ibs  - ibeis control object
        qreq - query request object (should be the same as ibs.qreq)
    Checks a big cache for qaid2_qres.
    If cache miss, tries to load each qres individually.
    On an individual cache miss, it preforms the query. """
    if utool.NOT_QUIET:
        print(' --- Process QueryRequest --- ')
    if len(qreq.qaids) <= 1:
        # Do not use bigcache single queries
        use_bigcache = False
    # Try and load directly from a big cache
    if use_bigcache:
        bigcache_dpath = qreq.bigcachedir
        bigcache_fname = (ibs.get_dbname() + '_QRESMAP' +
                          qreq.get_qaids_hashid() + qreq.get_daids_hashid())
        bigcache_cfgstr = qreq.cfg.get_cfgstr()
    if use_cache and use_bigcache:
        try:
            qaid2_qres = utool.load_cache(bigcache_dpath,
                                          bigcache_fname,
                                          bigcache_cfgstr)
            print('... qaid2_qres bigcache hit')
            return qaid2_qres
        except IOError:
            print('... qaid2_qres bigcache miss')
    # Try loading as many cached results as possible
    if use_cache:
        qaid2_qres, failed_qaids = mf.try_load_resdict(qreq)
    else:
        qaid2_qres = {}
        failed_qaids = qreq.qaids

    # Execute and save queries
    if len(failed_qaids) > 0:
        if safe:
            # FIXME: Ugg, this part is dirty
            qreq = pre_exec_checks(ibs, qreq)
        computed_qaid2_qres = execute_query_and_save_L1(ibs, qreq, failed_qaids)
        qaid2_qres.update(computed_qaid2_qres)  # Update cached results
    if use_bigcache:
        utool.save_cache(bigcache_dpath,
                         bigcache_fname,
                         bigcache_cfgstr, qaid2_qres)
    return qaid2_qres


# Query Level 1
#@utool.indent_func('[Q1]')
#@profile
@profile
def execute_query_and_save_L1(ibs, qreq, failed_qaids=[]):
    #print('[q1] execute_query_and_save_L1()')
    orig_qaids = qreq.qaids
    if len(failed_qaids) > 0:
        qreq.qaids = failed_qaids
    qaid2_qres = execute_query_L0(ibs, qreq)  # Execute Queries
    for qaid, res in six.iteritems(qaid2_qres):  # Cache Save
        res.save(qreq.get_qresdir())
    qreq.qaids = orig_qaids
    return qaid2_qres


# Query Level 0
#@utool.indent_func('[Q0]')
#@profile
@profile
def execute_query_L0(ibs, qreq):
    """
    Driver logic of query pipeline
    Input:
        ibs   - HotSpotter database object to be queried
        qreq - QueryRequest Object   # use prep_qreq to create one
    Output:
        qaid2_qres - mapping from query indexes to QueryResult Objects
    """

    # Query Chip Indexes
    # * vsone qaids/daids swapping occurs here
    qaids = qreq.get_internal_qaids()

    # Nearest neighbors (qaid2_nns)
    # * query descriptors assigned to database descriptors
    # * FLANN used here
    qaid2_nns = mf.nearest_neighbors(
        ibs, qaids, qreq)

    # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
    # * feature matches are weighted
    filt2_weights, filt2_meta = mf.weight_neighbors(
        ibs, qaid2_nns, qreq)

    # Thresholding and weighting (qaid2_nnfilter)
    # * feature matches are pruned
    qaid2_nnfilt = mf.filter_neighbors(
        ibs, qaid2_nns, filt2_weights, qreq)

    # Nearest neighbors to chip matches (qaid2_chipmatch)
    # * Inverted index used to create aid2_fmfsfk (TODO: crid2_fmfv)
    # * Initial scoring occurs
    # * vsone inverse swapping occurs here
    qaid2_chipmatch_FILT = mf.build_chipmatches(
        qaid2_nns, qaid2_nnfilt, qreq)

    # Spatial verification (qaid2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    qaid2_chipmatch_SVER = mf.spatial_verification(
        ibs, qaid2_chipmatch_FILT, qreq, dbginfo=False)

    # Query results format (qaid2_qres) (TODO: SQL / Json Encoding)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qaid2_qres = mf.chipmatch_to_resdict(
        ibs, qaid2_chipmatch_SVER, filt2_meta, qreq)

    return qaid2_qres
