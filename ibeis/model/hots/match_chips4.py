"""
DoctestCMD:
    python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.match_chips4))"
"""
from __future__ import absolute_import, division, print_function
import utool
from ibeis.model.hots import query_request
from ibeis.model.hots import pipeline as hspipe
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[mc4]')


# TODO: Move to params
USE_CACHE    = not utool.get_argflag(('--nocache-query', '--noqcache'))
USE_BIGCACHE = not utool.get_argflag(('--nocache-big', '--no-bigcache-query', '--noqcache'))
SAVE_CACHE   = not utool.get_argflag('--nocache-save')
MIN_BIGCACHE_BUNDLE = 20


#----------------------
# Main Query Logic
#----------------------

#@profile
def submit_query_request(ibs, qaid_list, daid_list, use_cache=USE_CACHE,
                         use_bigcache=USE_BIGCACHE, return_request=False,
                         custom_qparams=None):
    """
    The standard query interface.

    Checks a big cache for qaid2_qres.  If cache miss, tries to load each qres
    individually.  On an individual cache miss, it preforms the query.

    Args:
        ibs (IBEISController) : ibeis control object
        qaid_list (list): query annotation ids
        daid_list (list): database annotation ids
        use_cache (bool):
        use_bigcache (bool):

    Returns:
        qaid2_qres (dict): dict of QueryResult objects

    Examples:
        >>> from ibeis.model.hots.match_chips4 import *  # NOQA
        >>> import ibeis
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> use_bigcache = True
        >>> use_cache = True
        >>> ibs = ibeis.opendb(db='testdb1')  #doctest: +ELLIPSIS
        >>> qaid2_qres = submit_query_request(ibs, qaid_list, daid_list, use_cache, use_bigcache)

        >>> qaid2_qres, qreq_ = submit_query_request(ibs, qaid_list, daid_list, False, False, True)
    """
    # Create new query request object to store temporary state
    if utool.NOT_QUIET:
        print(' --- Submit QueryRequest_ --- ')
    # ------------
    # Build query request
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, custom_qparams)
    qreq_.qparams
    # --- BIG CACHE ---
    # Do not use bigcache single queries
    use_bigcache_ = (use_bigcache and use_cache and
                     len(qaid_list) > MIN_BIGCACHE_BUNDLE)
    if len(qaid_list) > MIN_BIGCACHE_BUNDLE:
        bc_dpath = ibs.bigcachedir
        qhashid = ibs.get_annot_uuid_hashid(qaid_list, '_QAUUIDS')
        dhashid = ibs.get_annot_uuid_hashid(daid_list, '_DAUUIDS')
        bc_fname = ''.join((ibs.get_dbname(), '_QRESMAP', qhashid, dhashid))
        bc_cfgstr = ibs.cfg.query_cfg.get_cfgstr()  # FIXME, rectify w/ qparams
        if use_bigcache_:
            # Try and load directly from a big cache
            try:
                qaid2_qres = utool.load_cache(bc_dpath, bc_fname, bc_cfgstr)
                print('... qaid2_qres bigcache hit')
                return qaid2_qres
            except IOError:
                print('... qaid2_qres bigcache miss')
    # Execute query request
    qaid2_qres = execute_query_and_save_L1(ibs, qreq_, use_cache)
    # ------------
    if len(qaid_list) > MIN_BIGCACHE_BUNDLE:
        utool.save_cache(bc_dpath, bc_fname, bc_cfgstr, qaid2_qres)
    if return_request:
        return qaid2_qres, qreq_
    return qaid2_qres


#@profile
def execute_query_and_save_L1(ibs, qreq_, use_cache=USE_CACHE):
    """
    Args:
        ibs (IBEISController):
        qreq_ (QueryRequest):
        use_cache (bool):

    Returns:
        qaid2_qres
    """
    #print('[q1] execute_query_and_save_L1()')
    if use_cache:
        if utool.DEBUG2:
            qreq_.assert_self(ibs)  # SANITY CHECK
        # Try loading as many cached results as possible
        qaid2_qres_hit, cachemiss_qaids = hspipe.try_load_resdict(qreq_)
        cachemiss_quuids = ibs.get_annot_uuids(cachemiss_qaids)
        qreq_.set_external_qaids(cachemiss_qaids, cachemiss_quuids)  # FIXME: changes qreq_ state
        #if utool.DEBUG2:
        #    qreq_.assert_self(ibs)  # SANITY CHECK
        if len(cachemiss_qaids) == 0:
            return qaid2_qres_hit
    else:
        print('[mc4] cache-query is off')
        #if __debug__:
        #    hspipe.try_load_resdict(qreq_, force_miss=True)
        qaid2_qres_hit = {}
    qreq_.assert_self(ibs)  # SANITY CHECK
    # Execute and save cachemiss queries
    qaid2_qres = hspipe.request_ibeis_query_L0(ibs, qreq_)  # execute queries
    # Cache save only misses
    if utool.DEBUG2:
        qreq_.assert_self(ibs)  # SANITY CHECK
    if SAVE_CACHE:
        hspipe.save_resdict(qreq_, qaid2_qres)
    # Merge cache hits with computed misses
    if len(qaid2_qres_hit) > 0:
        qaid2_qres.update(qaid2_qres_hit)
    del qreq_  # the query request is no longer needed
    return qaid2_qres
