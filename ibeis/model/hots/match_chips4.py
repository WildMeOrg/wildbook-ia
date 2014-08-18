from __future__ import absolute_import, division, print_function
import utool
import sys
import six
from ibeis.model.hots import query_request as hsqreq
from ibeis.model.hots import neighbor_index as hsnbrx
from ibeis.model.hots import pipeline as mf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[mc3]', DEBUG=False)


USE_CACHE = '--nocache-query' not in sys.argv
USE_BIGCACHE = '--nocache-big' not in sys.argv


#@utool.indent_func('[prep_qreq]')
@profile
def prep_query_request(ibs, qaids, daids, query_cfg):
    """  Builds or modifies a query request object """
    qreq = hsqreq.get_ibeis_query_request(ibs, qaids, daids)
    return qreq


def get_ibeis_query_request(ibs, qaid_list, daid_list):
    if utool.NOT_QUIET:
        print(' --- Prep QueryRequest --- ')
    nbrx = hsnbrx.get_ibies_neighbor_index(ibs, daid_list)
    qvecs_list = ibs.get_annot_desc(qaid_list)  # Get descriptors
    cfg = ibs.cfg.query_cfg
    qparams = hsqreq.QueryParams(cfg)
    qresdir      = ibs.qresdir
    bigcache_dir = ibs.bigcachedir
    qreq = hsqreq.QueryRequest(nbrx, qaid_list, qvecs_list, nbrx, qparams, qresdir, bigcache_dir)
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
    On an individual cache miss, it preforms the query.
    """
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
        res.save(ibs)
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
