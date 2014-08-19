'''
vsplit neighbor_index.py
vsplit pipeline.py
split neighbor_index.py
'''
from __future__ import absolute_import, division, print_function
import utool
import sys
import six
from ibeis.model.hots import query_request as hsqreq
from ibeis.model.hots import pipeline as hspipe
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[mc3]', DEBUG=False)


USE_CACHE = '--nocache-query' not in sys.argv
USE_BIGCACHE = '--nocache-big' not in sys.argv


#@utool.indent_func('[prep_qreq]')
@profile
def get_ibeis_query_request(ibs, qaids, daids):
    """ Builds or modifies a query request object """
    qreq = hsqreq.get_ibeis_query_request(ibs, qaids, daids)
    return qreq


#----------------------
# Main Query Logic
#----------------------

# Query Level 2
#@utool.indent_func('[Q2]')
@profile
def submit_query_request(ibs, qaid_list, daid_list, use_cache=USE_CACHE,
                         use_bigcache=USE_BIGCACHE):
    """
    The standard query interface.
    INPUT:
        ibs  - ibeis control object
        qreq - query request object (should be the same as ibs.qreq)
    Checks a big cache for qaid2_qres.
    If cache miss, tries to load each qres individually.
    On an individual cache miss, it preforms the query.

    >>> from ibeis.model.hots.query_request import *  # NOQA
    >>> import ibeis
    >>> qaid_list = [1]
    >>> daid_list = [1, 2, 3, 4, 5]
    >>> use_bigcache = True
    >>> use_cache = True
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    """
    qreq = get_ibeis_query_request(ibs, qaid_list, daid_list)
    if utool.NOT_QUIET:
        print(' --- Process QueryRequest --- ')
    if len(qaid_list) <= 1:
        # Do not use bigcache single queries
        use_bigcache = False
    # Try and load directly from a big cache
    if use_bigcache:
        bigcache_dpath = ibs.bigcachedir
        bigcache_fname = (ibs.get_dbname() + '_QRESMAP' +
                          qreq.get_query_hashid(ibs) + qreq.get_data_hashid(ibs))
        bigcache_cfgstr = qreq.qparams.query_cfgstr
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
        qaid2_qres, failed_qaids = hspipe.try_load_resdict(qreq, ibs)
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
