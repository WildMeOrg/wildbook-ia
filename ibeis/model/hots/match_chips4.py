"""
DoctestCMD:
    python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.match_chips4))"
"""
from __future__ import absolute_import, division, print_function
import utool
import utool as ut
from ibeis.model.hots import query_request
from ibeis.model.hots import pipeline
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[mc4]')


# TODO: Move to params
USE_CACHE    = not utool.get_argflag(('--nocache-query', '--noqcache'))
USE_BIGCACHE = not utool.get_argflag(('--nocache-big', '--no-bigcache-query', '--noqcache', '--nobigcache'))
SAVE_CACHE   = not utool.get_argflag('--nocache-save')
MIN_BIGCACHE_BUNDLE = 20


#----------------------
# Main Query Logic
#----------------------

#@profile
def submit_query_request(ibs, qaid_list, daid_list, use_cache=None,
                         use_bigcache=None, return_request=False,
                         cfgdict=None, qreq_=None, verbose=pipeline.VERB_PIPELINE):
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
        >>> # SLOW_DOCTEST
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
    if use_cache is None:
        use_cache = USE_CACHE
    if use_bigcache is None:
        use_bigcache = USE_BIGCACHE
    # Create new query request object to store temporary state
    if verbose:
        print(' --- Submit QueryRequest_ --- ')
    # ------------
    # Build query request
    if qreq_ is None:
        qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list,
                                                      cfgdict, verbose=verbose)
        #qreq_.qparams
    # --- BIG CACHE ---
    # Do not use bigcache single queries
    use_bigcache_ = (use_bigcache and use_cache and
                     len(qaid_list) > MIN_BIGCACHE_BUNDLE)
    if len(qaid_list) > MIN_BIGCACHE_BUNDLE:
        bc_dpath = ibs.bigcachedir
        qhashid = ibs.get_annot_hashid_uuid(qaid_list, '_QAUUIDS')
        dhashid = ibs.get_annot_hashid_uuid(daid_list, '_DAUUIDS')
        bc_fname = ''.join((ibs.get_dbname(), '_QRESMAP', qhashid, dhashid))
        bc_cfgstr = ibs.cfg.query_cfg.get_cfgstr()  # FIXME, rectify w/ qparams
        if use_bigcache_:
            # Try and load directly from a big cache
            try:
                qaid2_qres = utool.load_cache(bc_dpath, bc_fname, bc_cfgstr)
                print('... qaid2_qres bigcache hit')
                if return_request:
                    return qaid2_qres, qreq_
                else:
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


def generate_vsone_qreqs(ibs, qreq_, qaid_list, chunksize):
    """
    helper

    Generate vsone quries one at a time, but create shallow qreqs in chunks.
    """
    #qreq_shallow_iter = ((query_request.qreq_shallow_copy(qreq_, qx), qaid)
    #                     for qx, qaid in enumerate(qaid_list))
    qreq_shallow_iter = ((qreq_.shallowcopy(qx), qaid)
                         for qx, qaid in enumerate(qaid_list))
    qreq_chunk_iter = ut.ichunks(qreq_shallow_iter, chunksize)
    for qreq_chunk in qreq_chunk_iter:
        for __qreq, qaid in qreq_chunk:
            print('Generating vsone for qaid=%d' % (qaid,))
            qres = pipeline.request_ibeis_query_L0(ibs, __qreq)[qaid]
            yield (qaid, qres)


#@profile
def execute_query_and_save_L1(ibs, qreq_, use_cache=USE_CACHE, save_cache=SAVE_CACHE, chunksize=4):
    """
    Args:
        ibs (IBEISController):
        qreq_ (QueryRequest):
        use_cache (bool):

    Returns:
        qaid2_qres

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.hots.match_chips4 import *  # NOQA
        >>> import utool as ut
        >>> from ibeis.model.hots import pipeline
        >>> cfgdict1 = dict(codename='vsone', sv_on=True)
        >>> chunksize = 2
        >>> ibs, qreq_ = pipeline.get_pipeline_testdata(cfgdict=cfgdict1, qaid_list=[1, 2, 3, 4])
        >>> use_cache = False
        >>> save_cache = False
        >>> qaid2_qres_hit = execute_query_and_save_L1(ibs, qreq_, use_cache, save_cache, chunksize)
        >>> print(qaid2_qres_hit)
    """
    #print('[q1] execute_query_and_save_L1()')
    if use_cache:
        if utool.DEBUG2:
            qreq_.assert_self(ibs)  # SANITY CHECK
        # Try loading as many cached results as possible
        qaid2_qres_hit, cachemiss_qaids = pipeline.try_load_resdict(qreq_)
        cachemiss_quuids = ibs.get_annot_uuids(cachemiss_qaids)
        qreq_.set_external_qaids(cachemiss_qaids, cachemiss_quuids)  # FIXME: changes qreq_ state
        #if utool.DEBUG2:
        #    qreq_.assert_self(ibs)  # SANITY CHECK
        if len(cachemiss_qaids) == 0:
            return qaid2_qres_hit
    else:
        print('[mc4] cache-query is off')
        #if __debug__:
        #    pipeline.try_load_resdict(qreq_, force_miss=True)
        qaid2_qres_hit = {}
    #qreq_.assert_self(ibs)  # SANITY CHECK
    # Execute and save cachemiss queries
    if qreq_.qparams.pipeline_root == 'vsone':
        # Make sure that only one external query is requested per pipeline call
        # when doing vsone
        qaid_list = qreq_.get_external_qaids()
        qaid2_qres = {}

        qres_gen = generate_vsone_qreqs(ibs, qreq_, qaid_list, chunksize)
        qres_iter = ut.progiter(qres_gen, nTotal=len(qaid_list), freq=1,
                                backspace=False, lbl='vsone query: ',
                                use_rate=True)
        qres_chunk_iter = ut.ichunks(qres_iter, chunksize)

        for qres_chunk in qres_chunk_iter:
            qaid2_qres_ = {qaid: qres for qaid, qres in qres_chunk}
            # Save chunk of vsone queries
            if save_cache:
                print('[mc4] saving vsone chunk')
                pipeline.save_resdict(qreq_, qaid2_qres_)
            # Add current chunk to results
            qaid2_qres.update(qaid2_qres_)
    else:
        qaid2_qres = pipeline.request_ibeis_query_L0(ibs, qreq_)  # execute queries
        if save_cache:
            pipeline.save_resdict(qreq_, qaid2_qres)
    # Cache save only misses
    if utool.DEBUG2:
        qreq_.assert_self(ibs)  # SANITY CHECK
    # Merge cache hits with computed misses
    if len(qaid2_qres_hit) > 0:
        qaid2_qres.update(qaid2_qres_hit)
    #del qreq_  # is the query request is no longer needed?
    return qaid2_qres


if __name__ == '__main__':
    """
    python -m ibeis.model.hots.match_chips4
    python -m ibeis.model.hots.match_chips4 --allexamples
    python -m ibeis.model.hots.match_chips4 --test-execute_query_and_save_L1
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
