# -*- coding: utf-8 -*-
"""
Runs functions in pipeline to get query reuslts and does some caching.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import utool as ut
from os.path import exists
from wbia.algo.hots import chip_match
from wbia.algo.hots import pipeline

(print, rrr, profile) = ut.inject2(__name__)


# TODO: Move to params
USE_HOTSPOTTER_CACHE = pipeline.USE_HOTSPOTTER_CACHE
USE_CACHE = not ut.get_argflag(('--nocache-query', '--noqcache')) and USE_HOTSPOTTER_CACHE
USE_BIGCACHE = (
    not ut.get_argflag(
        ('--nocache-big', '--no-bigcache-query', '--noqcache', '--nobigcache')
    )
    and ut.USE_CACHE
)
USE_SUPERCACHE = (
    not ut.get_argflag(
        ('--nocache-super', '--no-supercache-query', '--noqcache', '--nosupercache')
    )
    and ut.USE_CACHE
)
SAVE_CACHE = not ut.get_argflag('--nocache-save')
# MIN_BIGCACHE_BUNDLE = 20
# MIN_BIGCACHE_BUNDLE = 150
MIN_BIGCACHE_BUNDLE = 64
HOTS_BATCH_SIZE = ut.get_argval('--hots-batch-size', type_=int, default=None)


# ----------------------
# Main Query Logic
# ----------------------


@profile
def submit_query_request(
    qreq_,
    use_cache=None,
    use_bigcache=None,
    verbose=None,
    save_qcache=None,
    use_supercache=None,
    invalidate_supercache=None,
):
    """
    Called from qreq_.execute

    Checks a big cache for qaid2_cm.  If cache miss, tries to load each cm
    individually.  On an individual cache miss, it preforms the query.

    CommandLine:
        python -m wbia.algo.hots.match_chips4 --test-submit_query_request

    Examples:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.algo.hots.match_chips4 import *  # NOQA
        >>> import wbia
        >>> qaid_list = [1]
        >>> daid_list = [1, 2, 3, 4, 5]
        >>> use_bigcache = True
        >>> use_cache = True
        >>> ibs = wbia.opendb(db='testdb1')
        >>> qreq_ = ibs.new_query_request(qaid_list, daid_list, verbose=True)
        >>> cm_list = submit_query_request(qreq_=qreq_)
    """
    # Get flag defaults if necessary
    if verbose is None:
        verbose = pipeline.VERB_PIPELINE
    if use_cache is None:
        use_cache = USE_CACHE
    if save_qcache is None:
        save_qcache = SAVE_CACHE
    if use_bigcache is None:
        use_bigcache = USE_BIGCACHE
    if use_supercache is None:
        use_supercache = USE_SUPERCACHE
    # Create new query request object to store temporary state
    if verbose:
        # print('[mc4] --- Submit QueryRequest_ --- ')
        print(ub.color_text('[mc4] --- Submit QueryRequest_ --- ', 'yellow'))
    assert qreq_ is not None, 'query request must be prebuilt'

    # Check fo empty queries
    try:
        assert len(qreq_.daids) > 0, 'there are no database chips'
        assert len(qreq_.qaids) > 0, 'there are no query chips'
    except AssertionError as ex:
        ut.printex(
            ex,
            'Impossible query request',
            iswarning=True,
            keys=['qreq_.qaids', 'qreq_.daids'],
        )
        if ut.SUPER_STRICT:
            raise
        cm_list = [None for qaid in qreq_.qaids]
    else:
        # --- BIG CACHE ---
        # Do not use bigcache single queries
        is_big = len(qreq_.qaids) > MIN_BIGCACHE_BUNDLE
        use_bigcache_ = use_bigcache and use_cache and is_big
        if use_bigcache_ or save_qcache:
            cacher = qreq_.get_big_cacher()
            if use_bigcache_:
                try:
                    qaid2_cm = cacher.load()
                    cm_list = [qaid2_cm[qaid] for qaid in qreq_.qaids]
                except (IOError, AttributeError):
                    pass
                else:
                    return cm_list
        # ------------
        # Execute query request
        qaid2_cm = execute_query_and_save_L1(
            qreq_,
            use_cache,
            save_qcache,
            verbose=verbose,
            use_supercache=use_supercache,
            invalidate_supercache=invalidate_supercache,
        )
        # ------------
        if save_qcache and is_big:
            cacher.save(qaid2_cm)

        cm_list = [qaid2_cm[qaid] for qaid in qreq_.qaids]
    return cm_list


@profile
def execute_query_and_save_L1(
    qreq_,
    use_cache,
    save_qcache,
    verbose=True,
    batch_size=None,
    use_supercache=False,
    invalidate_supercache=False,
):
    """
    Args:
        qreq_ (wbia.QueryRequest):
        use_cache (bool):

    Returns:
        qaid2_cm

    CommandLine:
        python -m wbia.algo.hots.match_chips4 execute_query_and_save_L1:0
        python -m wbia.algo.hots.match_chips4 execute_query_and_save_L1:1
        python -m wbia.algo.hots.match_chips4 execute_query_and_save_L1:2
        python -m wbia.algo.hots.match_chips4 execute_query_and_save_L1:3


    Example0:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.algo.hots.match_chips4 import *  # NOQA
        >>> cfgdict1 = dict(codename='vsmany', sv_on=True)
        >>> p = 'default' + ut.get_cfg_lbl(cfgdict1)
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(p=p, qaid_override=[1, 2, 3, 4)
        >>> use_cache, save_qcache, verbose = False, False, True
        >>> qaid2_cm = execute_query_and_save_L1(qreq_, use_cache, save_qcache, verbose)
        >>> print(qaid2_cm)

    Example1:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.algo.hots.match_chips4 import *  # NOQA
        >>> cfgdict1 = dict(codename='vsone', sv_on=True)
        >>> p = 'default' + ut.get_cfg_lbl(cfgdict1)
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(p=p, qaid_override=[1, 2, 3, 4)
        >>> use_cache, save_qcache, verbose = False, False, True
        >>> qaid2_cm = execute_query_and_save_L1(qreq_, use_cache, save_qcache, verbose)
        >>> print(qaid2_cm)

    Example1:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> # TEST SAVE
        >>> from wbia.algo.hots.match_chips4 import *  # NOQA
        >>> import wbia
        >>> cfgdict1 = dict(codename='vsmany', sv_on=True)
        >>> p = 'default' + ut.get_cfg_lbl(cfgdict1)
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(p=p, qaid_override=[1, 2, 3, 4)
        >>> use_cache, save_qcache, verbose = False, True, True
        >>> qaid2_cm = execute_query_and_save_L1(qreq_, use_cache, save_qcache, verbose)
        >>> print(qaid2_cm)

    Example2:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> # TEST LOAD
        >>> from wbia.algo.hots.match_chips4 import *  # NOQA
        >>> import wbia
        >>> cfgdict1 = dict(codename='vsmany', sv_on=True)
        >>> p = 'default' + ut.get_cfg_lbl(cfgdict1)
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(p=p, qaid_override=[1, 2, 3, 4)
        >>> use_cache, save_qcache, verbose = True, True, True
        >>> qaid2_cm = execute_query_and_save_L1(qreq_, use_cache, save_qcache, verbose)
        >>> print(qaid2_cm)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> # TEST PARTIAL HIT
        >>> from wbia.algo.hots.match_chips4 import *  # NOQA
        >>> import wbia
        >>> cfgdict1 = dict(codename='vsmany', sv_on=False, prescore_method='csum')
        >>> p = 'default' + ut.get_cfg_lbl(cfgdict1)
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(p=p, qaid_override=[1, 2, 3,
        >>>                                                               4, 5, 6,
        >>>                                                               7, 8, 9])
        >>> use_cache, save_qcache, verbose = False, True, False
        >>> qaid2_cm = execute_query_and_save_L1(qreq_, use_cache,
        >>>                                      save_qcache, verbose,
        >>>                                      batch_size=3)
        >>> cm = qaid2_cm[1]
        >>> ut.delete(cm.get_fpath(qreq_))
        >>> cm = qaid2_cm[4]
        >>> ut.delete(cm.get_fpath(qreq_))
        >>> cm = qaid2_cm[5]
        >>> ut.delete(cm.get_fpath(qreq_))
        >>> cm = qaid2_cm[6]
        >>> ut.delete(cm.get_fpath(qreq_))
        >>> print('Re-execute')
        >>> qaid2_cm_ = execute_query_and_save_L1(qreq_, use_cache,
        >>>                                       save_qcache, verbose,
        >>>                                       batch_size=3)
        >>> assert all([qaid2_cm_[qaid] == qaid2_cm[qaid] for qaid in qreq_.qaids])
        >>> [ut.delete(fpath) for fpath in qreq_.get_chipmatch_fpaths(qreq_.qaids)]

    Ignore:
        other = cm_ = qaid2_cm_[qaid]
        cm = qaid2_cm[qaid]
    """
    if invalidate_supercache:
        dpath = qreq_.get_qresdir()
        fpath_list = ut.glob('%s/*_cm_supercache_*' % (dpath,))
        for fpath in fpath_list:
            ut.delete(fpath)

    if use_cache:
        if verbose:
            print('[mc4] cache-query is on')
        if use_supercache:
            print('[mc4] supercache-query is on')
        # Try loading as many cached results as possible
        qaid2_cm_hit = {}
        external_qaids = qreq_.qaids
        fpath_list = list(
            qreq_.get_chipmatch_fpaths(external_qaids, super_qres_cache=use_supercache)
        )
        exists_flags = [exists(fpath) for fpath in fpath_list]
        qaids_hit = ut.compress(external_qaids, exists_flags)
        fpaths_hit = ut.compress(fpath_list, exists_flags)
        fpath_iter = ut.ProgIter(
            fpaths_hit,
            length=len(fpaths_hit),
            enabled=len(fpaths_hit) > 1,
            label='loading cache hits',
            adjust=True,
            freq=1,
        )
        try:
            cm_hit_list = [
                chip_match.ChipMatch.load_from_fpath(fpath, verbose=False)
                for fpath in fpath_iter
            ]
            assert all(
                [qaid == cm.qaid for qaid, cm in zip(qaids_hit, cm_hit_list)]
            ), 'inconsistent qaid and cm.qaid'
            qaid2_cm_hit = {cm.qaid: cm for cm in cm_hit_list}
        except chip_match.NeedRecomputeError:
            print('NeedRecomputeError: Some cached chips need to recompute')
            fpath_iter = ut.ProgIter(
                fpaths_hit,
                length=len(fpaths_hit),
                enabled=len(fpaths_hit) > 1,
                label='checking chipmatch cache',
                adjust=True,
                freq=1,
            )
            # Recompute those that fail loading
            qaid2_cm_hit = {}
            for fpath in fpath_iter:
                try:
                    cm = chip_match.ChipMatch.load_from_fpath(fpath, verbose=False)
                except chip_match.NeedRecomputeError:
                    pass
                else:
                    qaid2_cm_hit[cm.qaid] = cm
            print(
                '%d / %d cached matches need to be recomputed'
                % (len(qaids_hit) - len(qaid2_cm_hit), len(qaids_hit))
            )
        if len(qaid2_cm_hit) == len(external_qaids):
            return qaid2_cm_hit
        else:
            if len(qaid2_cm_hit) > 0 and not ut.QUIET:
                print(
                    '... partial cm cache hit %d/%d'
                    % (len(qaid2_cm_hit), len(external_qaids))
                )
        cachehit_qaids = list(qaid2_cm_hit.keys())
        # mask queries that have already been executed
        qreq_.set_external_qaid_mask(cachehit_qaids)
    else:
        if ut.VERBOSE:
            print('[mc4] cache-query is off')
        qaid2_cm_hit = {}
    qaid2_cm = execute_query2(qreq_, verbose, save_qcache, batch_size, use_supercache)
    # Merge cache hits with computed misses
    if len(qaid2_cm_hit) > 0:
        qaid2_cm.update(qaid2_cm_hit)
    qreq_.set_external_qaid_mask(None)  # undo state changes
    return qaid2_cm


@profile
def execute_query2(qreq_, verbose, save_qcache, batch_size=None, use_supercache=False):
    """
    Breaks up query request into several subrequests
    to process "more efficiently" and safer as well.
    """
    if qreq_.prog_hook is not None:
        preload_hook, query_hook = qreq_.prog_hook.subdivide(spacing=[0, 0.15, 0.8])
        preload_hook(0, lbl='preloading')
        qreq_.prog_hook = query_hook
    else:
        preload_hook = None
    # Load features / weights for all annotations
    qreq_.lazy_preload(prog_hook=preload_hook, verbose=verbose and ut.NOT_QUIET)

    all_qaids = qreq_.qaids
    print('len(missed_qaids) = %r' % (len(all_qaids),))
    qaid2_cm = {}
    # vsone must have a chunksize of 1
    if batch_size is None:
        if HOTS_BATCH_SIZE is None:
            hots_batch_size = qreq_.ibs.cfg.other_cfg.hots_batch_size
            # hots_batch_size = 256
        else:
            hots_batch_size = HOTS_BATCH_SIZE
    else:
        hots_batch_size = batch_size
    chunksize = 1 if qreq_.qparams.vsone else hots_batch_size

    # Iterate over vsone queries in chunks.
    n_total_chunks = ut.get_num_chunks(len(all_qaids), chunksize)
    qaid_chunk_iter = ut.ichunks(all_qaids, chunksize)
    _qreq_iter = (qreq_.shallowcopy(qaids=qaids) for qaids in qaid_chunk_iter)
    sub_qreq_iter = ut.ProgIter(
        _qreq_iter,
        length=n_total_chunks,
        freq=1,
        label='[mc4] query chunk: ',
        prog_hook=qreq_.prog_hook,
    )
    for sub_qreq_ in sub_qreq_iter:
        if ut.VERBOSE:
            print('Generating vsmany chunk')
        sub_cm_list = pipeline.request_wbia_query_L0(
            qreq_.ibs, sub_qreq_, verbose=verbose
        )
        assert len(sub_qreq_.qaids) == len(sub_cm_list), 'not aligned'
        assert all(
            [qaid == cm.qaid for qaid, cm in zip(sub_qreq_.qaids, sub_cm_list)]
        ), 'not corresonding'
        if save_qcache:
            fpath_list = list(
                qreq_.get_chipmatch_fpaths(
                    sub_qreq_.qaids, super_qres_cache=use_supercache
                )
            )
            _iter = zip(sub_cm_list, fpath_list)
            _iter = ut.ProgIter(
                _iter,
                length=len(sub_cm_list),
                label='saving chip matches',
                adjust=True,
                freq=1,
            )
            for cm, fpath in _iter:
                cm.save_to_fpath(fpath, verbose=False)
        else:
            if ut.VERBOSE:
                print('[mc4] not saving vsmany chunk')
        qaid2_cm.update({cm.qaid: cm for cm in sub_cm_list})
    return qaid2_cm


if __name__ == '__main__':
    """
    python -m wbia.algo.hots.match_chips4
    python -m wbia.algo.hots.match_chips4 --allexamples --testslow
    """
    import multiprocessing

    multiprocessing.freeze_support()
    ut.doctest_funcs()
