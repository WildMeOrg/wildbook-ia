'''
:vsplit neighbor_index.py
:vsplit pipeline.py
:split neighbor_index.py
'''
from __future__ import absolute_import, division, print_function
import utool
import sys
import six
from ibeis.model.hots import query_request as hsqreq
from ibeis.model.hots import neighbor_index as hsnbrx
from ibeis.model.hots import pipeline as hspipe
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
        qaid2_qres, failed_qaids = hspipe.try_load_resdict(qreq)
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
    qaid2_qres = hspipe.request_ibeis_query(ibs, qreq)  # Execute Queries
    for qaid, res in six.iteritems(qaid2_qres):  # Cache Save
        res.save(ibs)
    qreq.qaids = orig_qaids
    return qaid2_qres
