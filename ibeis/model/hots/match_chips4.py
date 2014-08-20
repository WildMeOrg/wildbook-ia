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
USE_BIGCACHE = '--nocache-big' not in sys.argv and '--no-bigcache-query' not in sys.argv
MIN_BIGCACHE_BUNDLE = 20


#----------------------
# Main Query Logic
#----------------------

@profile
def submit_query_request(ibs, qaid_list, daid_list, use_cache=USE_CACHE,
                         use_bigcache=USE_BIGCACHE):
    """
    The standard query interface.
    INPUT:
        ibs  - ibeis control object
        qreq_ - query request object (should be the same as ibs.qreq_)
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
    # Create new query request object to store temporary state
    if utool.NOT_QUIET:
        print(' --- Submit QueryRequest_ --- ')
    qreq_ = hsqreq.new_ibeis_query_request(ibs, qaid_list, daid_list)
    # --- BIG CACHE ---
    # Do not use bigcache single queries
    use_bigcache_ = (use_bigcache and use_cache and
                     len(qaid_list) > MIN_BIGCACHE_BUNDLE)
    # Try and load directly from a big cache
    if use_bigcache_:
        bc_dpath = ibs.bigcachedir
        bc_fname = ''.join((ibs.get_dbname(), '_QRESMAP',
                            qreq_.get_query_hashid(ibs),
                            qreq_.get_data_hashid(ibs)))
        bc_cfgstr = qreq_.qparams.query_cfgstr
        try:
            qaid2_qres = utool.load_cache(bc_dpath, bc_fname, bc_cfgstr)
            print('... qaid2_qres bigcache hit')
            return qaid2_qres
        except IOError:
            print('... qaid2_qres bigcache miss')
    # ------------
    qaid2_qres = execute_query_and_save_L1(ibs, qreq_, use_cache)
    if use_bigcache_:
        utool.save_cache(bc_dpath, bc_fname, bc_cfgstr, qaid2_qres)
    return qaid2_qres


@profile
def execute_query_and_save_L1(ibs, qreq_, use_cache=USE_CACHE):
    #print('[q1] execute_query_and_save_L1()')
    if use_cache:
        # Try loading as many cached results as possible
        qaid2_qres_hit, cachemiss_qaids = hspipe.try_load_resdict(qreq_)
        qreq_.set_external_qaids(cachemiss_qaids)  # FIXME: changes qreq_ state
        if len(cachemiss_qaids) == 0:
            return qaid2_qres_hit
    else:
        qaid2_qres_hit = {}
    # Execute and save cachemiss queries
    qaid2_qres = hspipe.request_ibeis_query_L0(ibs, qreq_)  # execute queries
    # Cache save only misses
    qresdir = qreq_.get_qresdir()
    for qaid, res in six.iteritems(qaid2_qres):
        res.save(qresdir)
    # Merge cache hits with computed misses
    if len(qaid2_qres_hit) > 0:
        qaid2_qres.update(qaid2_qres_hit)
    del qreq_  # the query request is no longer needed
    return qaid2_qres
