"""
Runs functions in pipeline to get query reuslts and does some caching.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
#from ibeis.model.hots import query_request
from ibeis.model.hots import pipeline
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[mc4]')


# TODO: Move to params
USE_CACHE    = not ut.get_argflag(('--nocache-query', '--noqcache'))
USE_BIGCACHE = not ut.get_argflag(('--nocache-big', '--no-bigcache-query', '--noqcache', '--nobigcache'))
SAVE_CACHE   = not ut.get_argflag('--nocache-save')
#MIN_BIGCACHE_BUNDLE = 20
MIN_BIGCACHE_BUNDLE = 150


#----------------------
# Main Query Logic
#----------------------


def empty_query(ibs, qaids):
    r"""
    Hack to give an empty query a query result object

    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (?):

    Returns:
        tuple: (qaid2_qres, qreq_)

    CommandLine:
        python -m ibeis.model.hots.match_chips4 --test-empty_query
        python -m ibeis.model.hots.match_chips4 --test-empty_query --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.match_chips4 import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> # execute function
        >>> (qaid2_qres, qreq_) = empty_query(ibs, qaids)
        >>> # verify results
        >>> result = str((qaid2_qres, qreq_))
        >>> print(result)
        >>> qres = qaid2_qres[qaids[0]]
        >>> ut.assert_eq(len(qres.get_top_aids()), 0)
        >>> ut.quit_if_noshow()
        >>> qres.ishow_top(ibs, update=True, make_figtitle=True, show_query=True, sidebyside=False)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
    """
    daids = []
    qreq_ = ibs.new_query_request(qaids, daids)
    qres_list = qreq_.make_empty_query_results()
    qaid2_qres = dict(zip(qaids, qres_list))
    return qaid2_qres, qreq_


def submit_query_request_nocache(ibs, qreq_, verbose=pipeline.VERB_PIPELINE):
    assert len(qreq_.get_external_qaids()) > 0, ' no current query aids'
    if len(qreq_.get_external_daids()) == 0:
        print('[mc4] WARNING no daids... returning empty query')
        qaid2_qres, qreq_ = empty_query(ibs, qreq_.get_external_qaids())
        return qaid2_qres
    save_qcache = False
    qaid2_qres = execute_query2(ibs, qreq_, verbose, save_qcache)
    return qaid2_qres


#@profile
def submit_query_request(ibs, qaid_list, daid_list, use_cache=None,
                         use_bigcache=None, return_request=False,
                         cfgdict=None, qreq_=None,
                         verbose=pipeline.VERB_PIPELINE, save_qcache=None):
    """
    The standard query interface.

    TODO: rename use_cache to use_qcache

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

    CommandLine:
        python -m ibeis.model.hots.match_chips4 --test-submit_query_request

    Examples:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.hots.match_chips4 import *  # NOQA
        >>> import ibeis
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> use_bigcache = True
        >>> use_cache = True
        >>> ibs = ibeis.opendb(db='testdb1')
        >>> qaid2_qres = submit_query_request(ibs, qaid_list, daid_list, use_cache, use_bigcache)
        >>> qaid2_qres, qreq_ = submit_query_request(ibs, qaid_list, daid_list, False, False, True)
    """
    # Get flag defaults if necessary
    if use_cache is None:
        use_cache = USE_CACHE
    if save_qcache is None:
        save_qcache = SAVE_CACHE
    if use_bigcache is None:
        use_bigcache = USE_BIGCACHE
    # Create new query request object to store temporary state
    if verbose:
        print(' --- Submit QueryRequest_ --- ')
    # ------------
    # Build query request
    if qreq_ is None:
        qreq_ = ibs.new_query_request(qaid_list, daid_list,
                                      cfgdict=cfgdict, verbose=verbose)
    # --- BIG CACHE ---
    # Do not use bigcache single queries
    use_bigcache_ = (use_bigcache and use_cache and
                     len(qaid_list) > MIN_BIGCACHE_BUNDLE)
    if (use_bigcache_ or save_qcache) and len(qaid_list) > MIN_BIGCACHE_BUNDLE:
        bc_dpath = ibs.bigcachedir
        # TODO: SYSTEM : semantic should only be used if name scoring is on
        qhashid = ibs.get_annot_hashid_semantic_uuid(qaid_list, prefix='Q')
        dhashid = ibs.get_annot_hashid_semantic_uuid(daid_list, prefix='D')
        query_cfgstr = ut.hashstr(qreq_.get_query_cfgstr())
        bc_fname = ''.join((ibs.get_dbname(), '_QRESMAP', qhashid, dhashid, query_cfgstr))
        bc_cfgstr = ibs.cfg.query_cfg.get_cfgstr()  # FIXME, rectify w/ qparams
        if use_bigcache_:
            # Try and load directly from a big cache
            try:
                qaid2_qres = ut.load_cache(bc_dpath, bc_fname, bc_cfgstr)
            except IOError:
                if verbose:
                    print('... qaid2_qres bigcache miss: %r' % (bc_fname,))
            else:
                if verbose:
                    print('... qaid2_qres bigcache hit: %r' % (bc_fname,))
                if return_request:
                    return qaid2_qres, qreq_
                else:
                    return qaid2_qres
    # ------------
    # Execute query request
    qaid2_qres = execute_query_and_save_L1(ibs, qreq_, use_cache, save_qcache, verbose=verbose)
    # ------------
    if save_qcache and len(qaid_list) > MIN_BIGCACHE_BUNDLE:
        ut.save_cache(bc_dpath, bc_fname, bc_cfgstr, qaid2_qres)
    if return_request:
        return qaid2_qres, qreq_
    else:
        return qaid2_qres


@profile
def execute_query_and_save_L1(ibs, qreq_, use_cache, save_qcache, verbose=True):
    """
    Args:
        ibs (IBEISController):
        qreq_ (QueryRequest):
        use_cache (bool):

    Returns:
        qaid2_qres

    CommandLine:
        python -m ibeis.model.hots.match_chips4 --test-execute_query_and_save_L1

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.hots.match_chips4 import *  # NOQA
        >>> cfgdict1 = dict(codename='vsmany', sv_on=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict1, qaid_list=[1, 2, 3, 4])
        >>> use_cache, save_qcache, verbose = False, False, True
        >>> qaid2_qres_hit = execute_query_and_save_L1(ibs, qreq_, use_cache, save_qcache, verbose)
        >>> print(qaid2_qres_hit)

    Example2:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.hots.match_chips4 import *  # NOQA
        >>> cfgdict1 = dict(codename='vsone', sv_on=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict1, qaid_list=[1, 2, 3, 4])
        >>> use_cache, save_qcache, verbose = False, False, True
        >>> qaid2_qres_hit = execute_query_and_save_L1(ibs, qreq_, use_cache, save_qcache, verbose)
        >>> print(qaid2_qres_hit)
    """
    #print('[q1] execute_query_and_save_L1()')
    if use_cache:
        if ut.VERBOSE:
            print('[mc4] cache-query is on')
        if ut.DEBUG2:
            # sanity check
            qreq_.assert_self(ibs)
        # Try loading as many cached results as possible
        qaid2_qres_hit = pipeline.try_load_resdict(qreq_, verbose=verbose)
        if len(qaid2_qres_hit) == len(qreq_.get_external_qaids()):
            return qaid2_qres_hit
        else:
            if len(qaid2_qres_hit) > 0 and not ut.QUIET:
                print('... partial qres cache hit %d/%d' % (len(qaid2_qres_hit), len(qreq_.get_external_qaids())))
        cachehit_qaids = list(six.iterkeys(qaid2_qres_hit))
        # mask queries that have already been executed
        qreq_.set_external_qaid_mask(cachehit_qaids)
    else:
        if ut.VERBOSE:
            print('[mc4] cache-query is off')
        qaid2_qres_hit = {}
    qaid2_qres = execute_query2(ibs, qreq_, verbose, save_qcache)
    if ut.DEBUG2:
        # sanity check
        qreq_.assert_self(ibs)
    # Merge cache hits with computed misses
    if len(qaid2_qres_hit) > 0:
        qaid2_qres.update(qaid2_qres_hit)
    qreq_.set_external_qaid_mask(None)  # undo state changes
    return qaid2_qres


def execute_query2(ibs, qreq_, verbose, save_qcache):
    qreq_.lazy_preload(verbose=verbose)
    all_qaids = qreq_.get_external_qaids()
    qaid2_qres = {}
    # vsone must have a chunksize of 1
    chunksize = 1 if qreq_.qparams.vsone else ibs.cfg.other_cfg.hots_batch_size
    # Iterate over vsone queries in chunks. This ensures that we dont lose
    # too much time if a qreq_ crashes after the 2000th nn index.
    nTotalChunks    = ut.get_nTotalChunks(len(all_qaids), chunksize)
    qaid_chunk_iter = ut.ichunks(all_qaids, chunksize)
    _qreq_iter = (qreq_.shallowcopy(qaids=qaids) for qaids in qaid_chunk_iter)
    qreq_iter = ut.ProgressIter(_qreq_iter, nTotal=nTotalChunks, freq=1,
                                lbl='[mc4] query chunk: ', backspace=False)
    for __qreq in qreq_iter:
        if ut.VERBOSE:
            print('Generating vsmany chunk')
        __qaid2_qres = pipeline.request_ibeis_query_L0(ibs, __qreq, verbose=verbose)
        if save_qcache:
            pipeline.save_resdict(qreq_, __qaid2_qres, verbose=verbose)
        else:
            if ut.VERBOSE:
                print('[mc4] not saving vsmany chunk')
        qaid2_qres.update(__qaid2_qres)
    return qaid2_qres


if __name__ == '__main__':
    """
    python -m ibeis.model.hots.match_chips4
    python -m ibeis.model.hots.match_chips4 --allexamples --testslow
    python -m ibeis.model.hots.match_chips4 --test-execute_query_and_save_L1
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
