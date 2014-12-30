from __future__ import absolute_import, division, print_function
import six
import numpy as np
import utool as ut
import pyflann
from os.path import join
from os.path import basename, exists  # NOQA
from six.moves import range
import vtool.nearest_neighbors as nntool
from ibeis.model.hots import hstypes
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[neighbor_index]', DEBUG=False)

NOCACHE_FLANN = ut.get_argflag('--nocache-flann')

# LRU cache for nn_indexers. Ensures that only a few are ever in memory
MAX_NEIGHBOR_CACHE_SIZE = 8
NEIGHBOR_CACHE = ut.get_lru_cache(MAX_NEIGHBOR_CACHE_SIZE)
CURRENT_THREAD = None


@profile
def request_ibeis_nnindexer(qreq_, verbose=True, use_memcache=True):
    """
    CALLED BY QUERYREQUST::LOAD_INDEXER

    FIXME: and use params from qparams instead of ibs.cfg
    IBEIS interface into neighbor_index

    Args:
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        NeighborIndexer: nnindexer

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-request_ibeis_nnindexer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer(None)
        >>> nnindexer = request_ibeis_nnindexer(qreq_)
    """
    daid_list = qreq_.get_internal_daids()
    nnindexer = request_memcached_ibeis_nnindexer(qreq_, daid_list,
                                                  verbose=verbose,
                                                  use_memcache=use_memcache)
    return nnindexer


@profile
def request_augmented_ibeis_nnindexer(qreq_, daid_list, verbose=True,
                                      use_memcache=True):
    """
    DO NOT USE. THIS FUNCTION CAN CURRENTLY CAUSE A SEGFAULT

    tries to give you an indexer for the requested daids using the least amount
    of computation possible. By loading and adding to a partially build nnindex
    if possible and if that fails fallbs back to request_memcache.

    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        daid_list (list):

    Returns:
        str: nnindex_cfgstr

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-request_augmented_ibeis_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ZEB_PLAIN = ibeis.const.Species.ZEB_PLAIN
        >>> ibs = ibeis.opendb('testdb1')
        >>> use_memcache = True
        >>> verbose = True
        >>> max_covers = None
        >>> daid_list = ibs.get_valid_aids(species=ZEB_PLAIN)[0:6]
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> qreq_.qparams.min_reindex_thresh = 1
        >>> min_reindex_thresh = qreq_.qparams.min_reindex_thresh
        >>> # CLEAR CACHE for clean test
        >>> clear_uuid_cache(qreq_)
        >>> # LOAD 3 AIDS INTO CACHE
        >>> aid_list = ibs.get_valid_aids(species=ZEB_PLAIN)[0:3]
        >>> # Should fallback
        >>> nnindexer = request_augmented_ibeis_nnindexer(qreq_, aid_list)
        >>> # assert the fallback
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result2 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result2, ([4, 5, 6], [[1, 2, 3]]), 'pre augment')
        >>> # Should augment
        >>> nnindexer = request_augmented_ibeis_nnindexer(qreq_, daid_list)
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result3 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result3, ([], [[1, 2, 3, 4, 5, 6]]), 'post augment')
        >>> # Should fallback
        >>> nnindexer2 = request_augmented_ibeis_nnindexer(qreq_, daid_list)
        >>> assert nnindexer is nnindexer2
    """
    global NEIGHBOR_CACHE
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    new_aid_list, covered_aids_list = group_daids_by_cached_nnindexer(
        qreq_, daid_list, min_reindex_thresh, max_covers=1)
    can_augment = (
        len(covered_aids_list) > 0 and
        not ut.list_set_equal(covered_aids_list[0], daid_list))
    print('[aug] Requesting augmented nnindexer')
    if can_augment:
        covered_aids = covered_aids_list[0]
        #with ut.PrintStartEndContext('AUGMENTING NNINDEX', verbose=verbose):
        #    with ut.Indenter('|  '):
        print('[aug] Augmenting index %r old daids with %d new daids' %
              (len(covered_aids), len(new_aid_list)))
        # Load the base covered indexer
        # THIS SHOULD LOAD NOT REBUILD IF THE UUIDS ARE COVERED
        base_nnindexer = request_memcached_ibeis_nnindexer(qreq_, covered_aids,
                                                           verbose=verbose,
                                                           use_memcache=use_memcache)
        # Remove this indexer from the memcache because we are going to change it
        if NEIGHBOR_CACHE.has_key(base_nnindexer.cfgstr):  # NOQA
            print('Removing key from memcache')
            NEIGHBOR_CACHE[base_nnindexer.cfgstr] = None
            del NEIGHBOR_CACHE[base_nnindexer.cfgstr]
        new_vecs_list = qreq_.ibs.get_annot_vecs(new_aid_list)
        new_fgws_list = get_fgweights_hack(qreq_, new_aid_list)
        base_nnindexer.add_support(new_aid_list, new_vecs_list, new_fgws_list, verbose=True)
        # FIXME: pointer issues
        nnindexer = base_nnindexer
        # Change to the new cfgstr
        nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        nnindexer.cfgstr = nnindex_cfgstr
        cachedir = qreq_.ibs.get_flann_cachedir()
        nnindexer.save(cachedir)
        # Write to inverse uuid
        if len(daid_list) > min_reindex_thresh:
            uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
            daids_hashid   = get_data_cfgstr(qreq_.ibs, daid_list)
            visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
            write_to_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
        # Write to memcache
        if ut.VERBOSE:
            print('[aug] Wrote to memcache=%r' % (nnindex_cfgstr,))
        NEIGHBOR_CACHE[nnindex_cfgstr] = nnindexer
        return nnindexer
    else:
        print('[aug] Fallback to memcache')
        # Fallback
        nnindexer = request_memcached_ibeis_nnindexer(
            qreq_, daid_list, verbose=verbose, use_memcache=use_memcache)
        return nnindexer


@profile
def request_memcached_ibeis_nnindexer(qreq_, daid_list, use_memcache=True, verbose=True, veryverbose=False, ):
    """
    FOR INTERNAL USE ONLY
    takes custom daid list. might not be the same as what is in qreq_

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-request_memcached_ibeis_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qreq_.qparams.min_reindex_thresh = 3
        >>> ZEB_PLAIN = ibeis.const.Species.ZEB_PLAIN
        >>> daid_list = ibs.get_valid_aids(species=ZEB_PLAIN)[0:3]
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> verbose = True
        >>> use_memcache = True
        >>> # execute function
        >>> nnindexer = request_memcached_ibeis_nnindexer(qreq_, daid_list, use_memcache)
        >>> # verify results
        >>> result = str(nnindexer)
        >>> print(result)
    """
    global NEIGHBOR_CACHE
    nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
    # neighbor memory cache
    if use_memcache and NEIGHBOR_CACHE.has_key(nnindex_cfgstr):  # NOQA (has_key is for a lru cache)
        if veryverbose:
            print('... nnindex memcache hit: cfgstr=%s' % (nnindex_cfgstr,))
        nnindexer = NEIGHBOR_CACHE[nnindex_cfgstr]
    else:
        if veryverbose:
            print('... nnindex memcache miss: cfgstr=%s' % (nnindex_cfgstr,))
        # Write to inverse uuid
        nnindexer = request_diskcached_ibeis_nnindexer(qreq_, daid_list, nnindex_cfgstr, verbose)
        # Write to memcache
        if ut.VERBOSE:
            print('[disk] Wrote to memcache=%r' % (nnindex_cfgstr,))
        NEIGHBOR_CACHE[nnindex_cfgstr] = nnindexer
    return nnindexer


@profile
def request_diskcached_ibeis_nnindexer(qreq_, daid_list, nnindex_cfgstr=None, verbose=True):
    """
    builds new NeighborIndexer which will try to use a disk cached flann if
    available

    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        daid_list (list):
        nnindex_cfgstr (?):
        verbose (bool):

    Returns:
        NeighborIndexer: nnindexer

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-request_diskcached_ibeis_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> daid_list = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> verbose = True
        >>> # execute function
        >>> nnindexer = request_diskcached_ibeis_nnindexer(qreq_, daid_list, nnindex_cfgstr, verbose)
        >>> # verify results
        >>> result = str(nnindexer)
        >>> print(result)
    """
    if nnindex_cfgstr is None:
        nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
    cfgstr = nnindex_cfgstr
    cachedir     = qreq_.ibs.get_flann_cachedir()
    flann_params = qreq_.qparams.flann_params
    # Get annot descriptors to index
    aid_list = daid_list
    vecs_list = qreq_.ibs.get_annot_vecs(aid_list)
    fgws_list = get_fgweights_hack(qreq_, aid_list)
    try:
        nnindexer = new_neighbor_index(
            aid_list, vecs_list, fgws_list, flann_params, cachedir,
            cfgstr=cfgstr, verbose=verbose)
    except Exception as ex:
        ut.printex(ex, True, msg_='cannot build inverted index',
                        key_list=['ibs.get_infostr()'])
        raise
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    if len(daid_list) > min_reindex_thresh:
        uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
        daids_hashid   = get_data_cfgstr(qreq_.ibs, daid_list)
        visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
        write_to_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
    return nnindexer


@profile
def group_daids_by_cached_nnindexer(qreq_, aid_list, min_reindex_thresh,
                                    max_covers=None):
    r"""
    Args:
        ibs       (IBEISController):
        daid_list (list):

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-group_daids_by_cached_nnindexer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> ZEB_PLAIN = ibeis.const.Species.ZEB_PLAIN
        >>> daid_list = ibs.get_valid_aids(species=ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> # Set the params a bit lower
        >>> max_covers = None
        >>> qreq_.qparams.min_reindex_thresh = 1
        >>> min_reindex_thresh = qreq_.qparams.min_reindex_thresh
        >>> # STEP 0: CLEAR THE CACHE
        >>> clear_uuid_cache(qreq_)
        >>> # STEP 1: ASSERT EMPTY INDEX
        >>> aid_list = ibs.get_valid_aids(species=ZEB_PLAIN)[0:3]
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, aid_list, min_reindex_thresh, max_covers)
        >>> result1 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result1, ([1, 2, 3], []), 'pre request')
        >>> # TEST 2: SHOULD MAKE 123 COVERED
        >>> nnindexer = request_memcached_ibeis_nnindexer(qreq_, aid_list)
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, aid_list, min_reindex_thresh, max_covers)
        >>> result2 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result2, ([], [[1, 2, 3]]), 'post request')
    """
    ibs = qreq_.ibs
    # read which annotations have prebuilt caches
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        candidate_uuids = {
            key: val for key, val in six.iteritems(uuid_map)
            if len(val) >= min_reindex_thresh
        }
        #candidate_uuids = {
        #    key: set(val) for key, val in six.iteritems(uuid_map)
        #    if len(val) >= min_reindex_thresh
        #}
        #for key in list(six.iterkeys(candidate_uuids)):
        #    # remove any sets less than the threshold
        #    if len(candidate_uuids[key]) < min_reindex_thresh:
        #        del candidate_uuids[key]
    # find a maximum independent set cover of the requested annotations
    annot_vuuid_list = ibs.get_annot_visual_uuids(aid_list)
    covertup = ut.greedy_max_inden_setcover(
        candidate_uuids, annot_vuuid_list, max_covers)
    uncovered_vuuids, covered_vuuids_list, accepted_keys = covertup
    # return the grouped covered items (so they can be loaded) and
    # the remaining uuids which need to have an index computed.
    uncovered_aids_ = ibs.get_annot_aids_from_visual_uuid(uncovered_vuuids)
    covered_aids_list_ = ibs.unflat_map(
        ibs.get_annot_aids_from_visual_uuid, covered_vuuids_list)
    # FIXME:
    uncovered_aids = sorted(uncovered_aids_)
    #covered_aids_list = list(map(sorted, covered_aids_list_))
    covered_aids_list = covered_aids_list_
    return uncovered_aids, covered_aids_list


@profile
def write_to_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid):
    """
    let the multi-indexer know about any big caches we've made multi-indexer.
    Also lets nnindexer know about other prebuilt indexers so it can attempt to
    just add points to them as to avoid a rebuild.
    """
    print('Writing %d visual uuids to uuid map' % (len(visual_uuid_list)))
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        uuid_map[daids_hashid] = visual_uuid_list


def get_data_cfgstr(ibs, daid_list):
    """ part 2 data hash id """
    daids_hashid = ibs.get_annot_hashid_visual_uuid(daid_list)
    return daids_hashid


@profile
def build_nnindex_cfgstr(qreq_, daid_list):
    """
    builds a string that  uniquely identified an indexer built with parameters
    from the input query requested and indexing descriptor from the input
    annotation ids

    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        daid_list (list):

    Returns:
        str: nnindex_cfgstr

    CommandLine:
        python -c 'import utool; print(utool.auto_docstr("ibeis.model.hots.neighbor_index", "build_nnindex_cfgstr"))'
        python -m ibeis.model.hots.neighbor_index --test-build_nnindex_cfgstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> # build test data
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='testdb1')
        >>> daid_list = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> # execute function
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> # verify results
        >>> result = str(nnindex_cfgstr)
        >>> print(result)
        _VUUIDS((6)fvpdb9cud49@ll@+)_FLANN(4_kdtrees)_FEATWEIGHT(ON,uselabel,rf)_FEAT(hesaff+sift_)_CHIP(sz450)
    """
    flann_cfgstr      = qreq_.qparams.flann_cfgstr
    featweight_cfgstr = qreq_.qparams.featweight_cfgstr
    data_hashid   = get_data_cfgstr(qreq_.ibs, daid_list)
    nnindex_cfgstr = ''.join((data_hashid, flann_cfgstr, featweight_cfgstr))
    return nnindex_cfgstr


@profile
def get_nnindexer_uuid_map_fpath(qreq_):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> # build test data
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='testdb1')
        >>> daid_list = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
        >>> result = str(ut.path_ndir_split(uuid_map_fpath, 3))
        >>> print(result)
        _ibeis_cache/flann/uuid_map_FLANN(4_kdtrees)_FEAT(hesaff+sift_)_CHIP(sz450).shelf
    """
    flann_cachedir = qreq_.ibs.get_flann_cachedir()
    # Have uuid shelf conditioned on the baseline flann and feature parameters
    flann_cfgstr    = qreq_.qparams.flann_cfgstr
    feat_cfgstr     = qreq_.qparams.feat_cfgstr
    uuid_map_cfgstr = ''.join((flann_cfgstr, feat_cfgstr))
    uuid_map_ext    = '.shelf'
    uuid_map_prefix = 'uuid_map'
    uuid_map_fname  = ut.consensed_cfgstr(uuid_map_prefix, uuid_map_cfgstr) + uuid_map_ext
    uuid_map_fpath  = join(flann_cachedir, uuid_map_fname)
    return uuid_map_fpath


def clear_memcache():
    global NEIGHBOR_CACHE
    NEIGHBOR_CACHE.clear()


@profile
def clear_uuid_cache(qreq_):
    """

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-clear_uuid_cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> daids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qaids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(qaids, daids)
        >>> # execute function
        >>> fgws_list = clear_uuid_cache(qreq_)
        >>> # verify results
        >>> result = str(fgws_list)
        >>> print(result)
    """
    print('[nnindex] clearing uuid cache')
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    ut.delete(uuid_map_fpath)
    #with ut.shelf_open(uuid_map_fpath) as uuid_map:
    #    uuid_map.clear()
    print('[nnindex] finished uuid cache clear')


def print_uuid_cache(qreq_):
    """

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-print_uuid_cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> daids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qaids = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(qaids, daids)
        >>> # execute function
        >>> nnindexer = print_uuid_cache(qreq_)
        >>> # verify results
        >>> result = str(nnindexer)
        >>> print(result)
    """
    print('[nnindex] clearing uuid cache')
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        print(uuid_map)


@profile
def get_fgweights_hack(qreq_, daid_list):
    """
    hack to get  feature weights. returns None if feature weights are turned off
    in config settings
    """
    # <HACK:featweight>
    if qreq_.qparams.fg_weight != 0:
        fgws_list = qreq_.ibs.get_annot_fgweights(
            daid_list, qreq_=qreq_, ensure=True)
    else:
        fgws_list = None
    return fgws_list
    # </HACK:featweight>


@profile
def new_neighbor_index(aid_list, vecs_list, fgws_list, flann_params, cachedir,
                       cfgstr, verbose=True):
    """
    constructs neighbor index independent of ibeis

    Args:
        aid_list (list):
        vecs_list (list):
        fgws_list (list):
        flann_params (dict):
        flann_cachedir (None):
        nnindex_cfgstr (str):
        use_memcache (bool):

    Returns:
        nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> daid_list = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> verbose = True
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> cfgstr = nnindex_cfgstr
        >>> cachedir     = qreq_.ibs.get_flann_cachedir()
        >>> flann_params = qreq_.qparams.flann_params
        >>> # Get annot descriptors to index
        >>> aid_list = daid_list
        >>> vecs_list = qreq_.ibs.get_annot_vecs(aid_list)
        >>> fgws_list = get_fgweights_hack(qreq_, aid_list)
        >>> # execute function
        >>> nnindexer = new_neighbor_index(aid_list, vecs_list, fgws_list, flann_params, cachedir, cfgstr, verbose=True)

    """
    if verbose:
        print('[nnindex] nnindexer = new NeighborIndex')
    nnindexer = NeighborIndex(flann_params, cfgstr)
    if verbose:
        print('[nnindex] nnindexer.init_support()')
    # Initialize neighbor with unindexed data
    nnindexer.init_support(aid_list, vecs_list, fgws_list, verbose=verbose)
    if verbose:
        print('[nnindex] nnindexer.load_or_build()')
    # Load or build the indexing structure
    nnindexer.load_or_build(cachedir, verbose=verbose)
    if verbose:
        print('[nnindex] ...')
    return nnindexer


@profile
def prepare_index_data(aid_list, vecs_list, fgws_list, verbose=True):
    """
    flattens vecs_list and builds a reverse index from the flattened indicies
    (idx) to the original aids and fxs
    """
    # Check input
    assert len(aid_list) == len(vecs_list), 'invalid input. bad len'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')
    # Create indexes into the input aids
    ax_list = np.arange(len(aid_list))
    idx2_vec, idx2_ax, idx2_fx = invert_index(vecs_list, ax_list, verbose=verbose)
    # <HACK:fgweights>
    if fgws_list is not None:
        idx2_fgw = np.hstack(fgws_list)
        try:
            assert len(idx2_fgw) == len(idx2_vec), 'error. weights and vecs do not correspond'
        except Exception as ex:
            ut.printex(ex, keys=[(len, 'idx2_fgw'), (len, 'idx2_vec')])
            raise
    else:
        idx2_fgw = None
    # </HACK:fgweights>
    ax2_aid = np.array(aid_list)
    preptup = (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx)
    return preptup


@six.add_metaclass(ut.ReloadingMetaclass)
class NeighborIndex(object):
    """
    wrapper class around flann
    stores flann index and data it needs to index into

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()
    """
    ext     = '.flann'
    prefix1 = 'flann'

    def __init__(nnindexer, flann_params, cfgstr):
        """ initialize an empty neighbor indexer """
        nnindexer.flann    = None  # Approximate search structure
        nnindexer.ax2_aid  = None  # (A x 1) Mapping to original annot ids
        nnindexer.idx2_vec = None  # (M x D) Descriptors to index
        nnindexer.idx2_fgw = None  # (M x 1) Descriptor forground weight
        nnindexer.idx2_ax  = None  # (M x 1) Index into the aid_list
        nnindexer.idx2_fx  = None  # (M x 1) Index into the annot's features
        nnindexer.cfgstr   = cfgstr  # configuration id
        nnindexer.flann_params = flann_params
        nnindexer.cores = flann_params.get('cores', 0)
        nnindexer.num_indexed = None

    @profile
    def init_support(nnindexer, aid_list, vecs_list, fgws_list, verbose=True):
        preptup = prepare_index_data(aid_list, vecs_list, fgws_list, verbose=verbose)
        (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx) = preptup
        nnindexer.flann    = pyflann.FLANN()  # Approximate search structure
        nnindexer.ax2_aid  = ax2_aid   # (A x 1) Mapping to original annot ids
        nnindexer.idx2_vec = idx2_vec  # (M x D) Descriptors to index
        nnindexer.idx2_fgw = idx2_fgw  # (M x 1) Descriptor forground weight
        nnindexer.idx2_ax  = idx2_ax   # (M x 1) Index into the aid_list
        nnindexer.idx2_fx  = idx2_fx   # (M x 1) Index into the annot's features
        nnindexer.num_indexed = len(nnindexer.idx2_vec)
        if nnindexer.idx2_vec.dtype == hstypes.VEC_TYPE:
            nnindexer.max_distance = hstypes.VEC_PSEUDO_MAX_DISTANCE
        else:
            assert False, 'NNindexer should get uint8s right now unless the algorithm has changed'

    @profile
    def add_support(nnindexer, new_aid_list, new_vecs_list, new_fgws_list,
                    verbose=True):
        """
        adds support data (aka data to be indexed)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> new_aid_list = [2, 3, 4]
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> new_vecs_list = ibs.get_annot_vecs(new_aid_list)
            >>> new_fgws_list = ibs.get_annot_fgweights(new_aid_list)
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K, checks=checks)
            >>> nnindexer.add_support(new_aid_list, new_vecs_list, new_fgws_list)
            >>> (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K, checks=checks)
            >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        nAnnots = nnindexer.num_indexed_annots()
        nVecs = nnindexer.num_indexed_vecs()
        nNewAnnots = len(new_aid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNewAnnots)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list)
        nNewVecs = len(new_idx2_vec)
        if verbose:
            print('[nnindex] Adding %d vecs from %d annots to nnindex with %d vecs and %d annots' %
                  (nNewVecs, nNewAnnots, nVecs, nAnnots))
        new_idx2_fgw = np.hstack(new_fgws_list)
        print('STACKING')
        # Stack inverted information
        ##---
        if not hasattr(nnindexer, 'old_vecs'):
            nnindexer.old_vecs = []
        # Try to hack in a way to keep the old memory
        old_idx2_vec = nnindexer.idx2_vec
        nnindexer.old_vecs.append(old_idx2_vec)
        nnindexer.old_vecs.append(new_idx2_fgw)
        ##---
        _ax2_aid = np.hstack((nnindexer.ax2_aid, new_aid_list))
        _idx2_ax = np.hstack((nnindexer.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((nnindexer.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((old_idx2_vec, new_idx2_vec))
        _idx2_fgw = np.hstack((nnindexer.idx2_fgw, new_idx2_fgw))
        print('REPLACING')
        nnindexer.ax2_aid  = _ax2_aid
        nnindexer.idx2_ax  = _idx2_ax
        nnindexer.idx2_vec = _idx2_vec
        nnindexer.idx2_fx  = _idx2_fx
        nnindexer.idx2_fgw = _idx2_fgw
        #nnindexer.idx2_kpts   = None
        #nnindexer.idx2_oris   = None
        # Add new points to flann structure
        print('ADD POINTS (FIXME: SOMETIMES SEGFAULT OCCURS)')
        nnindexer.flann.add_points(new_idx2_vec)
        print('DONE ADD POINTS')

    def load_or_build(nnindexer, cachedir, verbose=True):
        #with ut.PrintStartEndContext(msg='CACHED NNINDEX', verbose=verbose):
        if NOCACHE_FLANN:
            print('...nnindex flann cache is forced off')
            load_success = False
        else:
            load_success = nnindexer.load(cachedir, verbose=verbose)
        if load_success:
            if not ut.QUIET:
                nVecs   = nnindexer.num_indexed_vecs()
                nAnnots = nnindexer.num_indexed_annots()
                print('...nnindex flann cache hit: %d vectors, %d annots' %
                      (nVecs, nAnnots))
        else:
            if not ut.QUIET:
                nVecs   = nnindexer.num_indexed_vecs()
                nAnnots = nnindexer.num_indexed_annots()
                print('...nnindex flann cache miss: %d vectors, %d annots' %
                      (nVecs, nAnnots))
            nnindexer.build_and_save(cachedir, verbose=verbose)

    def build_and_save(nnindexer, cachedir, verbose=True):
        nnindexer.reindex()
        nnindexer.save(cachedir, verbose=verbose)

    def reindex(nnindexer, verbose=True):
        """ indexes all vectors with FLANN. """
        num_vecs = nnindexer.num_indexed
        notify_num = 1E6
        if verbose or (not ut.QUIET and num_vecs > notify_num):
            print('...building kdtree over %d points (this may take a sec).' % num_vecs)
        idx2_vec = nnindexer.idx2_vec
        flann_params = nnindexer.flann_params
        nnindexer.flann.build_index(idx2_vec, **flann_params)

    # ---- <cachable_interface> ---

    def save(nnindexer, cachedir, verbose=True):
        flann_fpath = nnindexer.get_fpath(cachedir)
        if verbose:
            print('flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=5))
        nnindexer.flann.save_index(flann_fpath)

    def load(nnindexer, cachedir, verbose=True):
        load_success = False
        flann_fpath = nnindexer.get_fpath(cachedir)
        if ut.checkpath(flann_fpath, verbose=ut.VERBOSE):
            try:
                idx2_vec = nnindexer.idx2_vec
                nnindexer.flann.load_index(flann_fpath, idx2_vec)
                load_success = True
            except Exception as ex:
                ut.printex(ex, '... cannot load nnindex flann', iswarning=True)
        return load_success
        if ut.VERBOSE:
            print('[nnindex] load_success = %r' % (load_success,))
        #flann = nntool.flann_cache(idx2_vec, verbose=verbose, **flannkw)

    def get_prefix(nnindexer):
        return nnindexer.prefix1

    @profile
    def get_cfgstr(nnindexer):
        """ returns string which uniquely identified configuration and support data """
        flann_cfgstr_list = []
        use_params_hash = False
        if use_params_hash:
            flann_params = nnindexer.flann_params
            flann_valsig_ = str(list(flann_params.values()))
            flann_valsig = ut.remove_chars(flann_valsig_, ', \'[]')
            flann_cfgstr_list.append('_FLANN(' + flann_valsig + ')')
        use_data_hash = True
        if use_data_hash:
            idx2_vec = nnindexer.idx2_vec
            vecs_hashstr = ut.hashstr_arr(idx2_vec, '_VECS')
            flann_cfgstr_list.append(vecs_hashstr)
        flann_cfgstr = ''.join(flann_cfgstr_list)
        return flann_cfgstr

    def get_fname(nnindexer):
        return basename(nnindexer.get_fpath(''))

    def get_fpath(nnindexer, cachedir, cfgstr=None):
        _args2_fpath = ut.util_cache._args2_fpath
        dpath  = cachedir
        prefix = nnindexer.get_prefix()
        cfgstr = nnindexer.get_cfgstr()
        ext    = nnindexer.ext
        fpath  = _args2_fpath(dpath, prefix, cfgstr, ext, write_hashtbl=False)
        return fpath

    # ---- </cachable_interface> ---

    def get_dtype(nnindexer):
        return nnindexer.idx2_vec.dtype

    #@profile
    def knn(nnindexer, qfx2_vec, K, checks=1028):
        """
        Args:
            qfx2_vec : (N x D) an array of N, D-dimensional query vectors

            K: number of approximate nearest neighbors to find

        Returns: tuple of (qfx2_idx, qfx2_dist)
            qfx2_idx : (N x K) qfx2_idx[n][k] is the index of the kth
                        approximate nearest data vector w.r.t qfx2_vec[n]

            qfx2_dist : (N x K) qfx2_dist[n][k] is the distance to the kth
                        approximate nearest data vector w.r.t. qfx2_vec[n]
                        distance is normalized squared euclidean distance.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
            >>> result = str(qfx2_idx.shape) + ' ' + str(qfx2_dist.shape)
            >>> assert np.all(qfx2_dist < 1.0), 'distance should be less than 1'
            >>> print(result)
            (1257, 2) (1257, 2)

        Example2:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> qfx2_vec = np.empty((0, 128), dtype=nnindexer.get_dtype())
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
            >>> result = str(qfx2_idx.shape) + ' ' + str(qfx2_dist.shape)
            >>> print(result)
            (0, 2) (0, 2)

        """
        if K == 0:
            (qfx2_idx, qfx2_dist) = nnindexer.empty_neighbors(len(qfx2_vec), 0)
        if K > nnindexer.num_indexed or K == 0:
            # If we want more points than there are in the database
            # FLANN will raise an exception. This corner case
            # will hopefully only be hit if using the multi-indexer
            # so try this workaround which should seemlessly integrate
            # when the multi-indexer stacks the subindxer results.
            # There is a very strong possibility that this will cause errors
            # If this corner case is used in non-multi-indexer code
            K = nnindexer.num_indexed
            (qfx2_idx, qfx2_dist) = nnindexer.empty_neighbors(len(qfx2_vec), 0)
        elif len(qfx2_vec) == 0:
            (qfx2_idx, qfx2_dist) = nnindexer.empty_neighbors(0, K)
        else:
            # perform nearest neighbors
            (qfx2_idx, qfx2_dist) = nnindexer.flann.nn_index(
                qfx2_vec, K, checks=checks, cores=nnindexer.cores)
            # Ensure that distance returned are between 0 and 1
            qfx2_dist = qfx2_dist / (nnindexer.max_distance ** 2)
            #qfx2_dist = np.sqrt(qfx2_dist) / nnindexer.max_distance
        return (qfx2_idx, qfx2_dist)

    def empty_neighbors(nnindexer, nQfx, K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    def num_indexed_vecs(nnindexer):
        return len(nnindexer.idx2_vec)

    def num_indexed_annots(nnindexer):
        return len(nnindexer.ax2_aid)

    def get_nn_vecs(nnindexer, qfx2_nnidx):
        """ gets matching vectors """
        return nnindexer.idx2_vec.take(qfx2_nnidx, axis=0)

    def get_nn_axs(nnindexer, qfx2_nnidx):
        """ gets matching internal annotation indicies """
        return nnindexer.idx2_ax.take(qfx2_nnidx)

    @profile
    def get_nn_aids(nnindexer, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_aid : (N x K) qfx2_fx[n][k] is the annotation id index of the
                                kth approximate nearest data vector

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots import pipeline
            >>> cfgdict = dict()
            >>> dbname = 'testdb1'
            >>> ibs, qreq_ = pipeline.get_pipeline_testdata(dbname=dbname, cfgdict=cfgdict)
            >>> nnindexer = qreq_.indexer
            >>> qfx2_vec = qreq_.ibs.get_annot_vecs(qreq_.get_internal_qaids()[0])
            >>> num_neighbors = 4
            >>> checks = 1024
            >>> (qfx2_nnidx, qfx2_dist) = nnindexer.knn(qfx2_vec, num_neighbors, checks)
            >>> qfx2_aid = nnindexer.get_nn_aids(qfx2_nnidx)
            >>> result = qfx2_aid.shape
            >>> print(result)
            (1257, 4)
        """
        #qfx2_ax = nnindexer.idx2_ax[qfx2_nnidx]
        #qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        qfx2_ax = nnindexer.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = nnindexer.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_fx : (N x K) qfx2_fx[n][k] is the feature index (w.r.t the
                               source annotation) of the kth approximate
                               nearest data vector
        """
        #return nnindexer.idx2_fx[qfx2_nnidx]
        qfx2_fx = nnindexer.idx2_fx.take(qfx2_nnidx)
        return qfx2_fx

    def get_nn_fgws(nnindexer, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx : (N x K) qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Returns:
            qfx2_fgw : (N x K) qfx2_fgw[n][k] is the annotation id index of the
                                kth forground weight
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer(dbname='testdb1')
            >>> qfx2_nnidx = np.array([[0, 1, 2], [3, 4, 5]])
            >>> qfx2_fgw = nnindexer.get_nn_fgws(qfx2_nnidx)
        """
        #qfx2_ax = nnindexer.idx2_ax[qfx2_nnidx]
        #qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        if nnindexer.idx2_fgw is None:
            qfx2_fgw = np.ones(qfx2_nnidx.shape)
        else:
            qfx2_fgw = nnindexer.idx2_fgw.take(qfx2_nnidx)
        return qfx2_fgw


@profile
def invert_index(vecs_list, ax_list, verbose=ut.NOT_QUIET):
    """
    Aggregates descriptors of input annotations and returns inverted information
    """
    if ut.VERYVERBOSE:
        print('[nnindex] stacking descriptors from %d annotations' % len(ax_list))
    try:
        idx2_vec, idx2_ax, idx2_fx = nntool.invertable_stack(vecs_list, ax_list)
        assert idx2_vec.shape[0] == idx2_ax.shape[0]
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
    except MemoryError as ex:
        ut.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if verbose:
        print('[nnindex] stacked nVecs={nVecs} from nAnnots={nAnnots}'.format(
            nVecs=len(idx2_vec), nAnnots=len(ax_list)))
    return idx2_vec, idx2_ax, idx2_fx


def test_nnindexer(dbname='testdb1', with_indexer=True):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()
    """
    import ibeis
    daid_list = [7, 8, 9, 10, 11]
    ibs = ibeis.opendb(db=dbname)
    qreq_ = ibs.new_query_request(daid_list, daid_list)
    if with_indexer:
        nnindexer = request_ibeis_nnindexer(qreq_)
    else:
        nnindexer = None
    return nnindexer, qreq_, ibs


def test_incremental_add(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-test_incremental_add

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = test_incremental_add(ibs)
        >>> print(result)
    """
    sample_aids = ibs.get_annot_rowid_sample()
    aids1 = sample_aids[::2]
    aids2 = sample_aids[0:5]
    aids3 = sample_aids[:-1]  # NOQA
    daid_list = aids1  # NOQA
    qreq_ = ibs.new_query_request(aids1, aids1)
    nnindexer1 = request_ibeis_nnindexer(ibs.new_query_request(aids1, aids1))  # NOQA
    nnindexer2 = request_ibeis_nnindexer(ibs.new_query_request(aids2, aids2))  # NOQA

    # TODO: SYSTEM use visual uuids
    #daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)  # get_internal_data_hashid()
    items = ibs.get_annot_visual_uuids(aids3)
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        candidate_uuids = {key: set(val) for key, val in six.iteritems(uuid_map)}
    candidate_sets = candidate_uuids
    covertup = ut.greedy_max_inden_setcover(candidate_sets, items)
    uncovered_items, covered_items_list, accepted_keys = covertup
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    nnindexer3 = request_ibeis_nnindexer(ibs.new_query_request(uncovered_aids, uncovered_aids))  # NOQA

    # TODO: SYSTEM use visual uuids
    #daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)  # get_internal_data_hashid()
    items = ibs.get_annot_visual_uuids(sample_aids)
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    #contextlib.closing(shelve.open(uuid_map_fpath)) as uuid_map:
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        candidate_uuids = {key: set(val) for key, val in six.iteritems(uuid_map)}
    candidate_sets = candidate_uuids
    covertup = ut.greedy_max_inden_setcover(candidate_sets, items)
    uncovered_items, covered_items_list, accepted_keys = covertup
    covered_items = ut.flatten(covered_items_list)

    covered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(covered_items))  # NOQA
    uncovered_aids = sorted(ibs.get_annot_aids_from_visual_uuid(uncovered_items))

    uuid_map
    #uuid_map_fpath = join(flann_cachedir, 'uuid_map.shelf')
    #uuid_map = shelve.open(uuid_map_fpath)
    #uuid_map[daids_hashid] = visual_uuid_list
    #visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
    #visual_uuid_list
    #%timeit request_ibeis_nnindexer(qreq_, use_memcache=False)
    #%timeit request_ibeis_nnindexer(qreq_, use_memcache=True)

    #for uuids in uuid_set
    #    if


def subindexer_time_experiment():
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment
    """
    import ibeis
    import utool as ut
    import pyflann
    import numpy as np
    import plottool as pt
    ibs = ibeis.opendb(db='PZ_Master0')
    daid_list = ibs.get_valid_aids()
    count_list = []
    time_list = []
    flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()
    for count in ut.ProgressIter(range(1, 301)):
        daids_ = daid_list[:]
        np.random.shuffle(daids_)
        daids = daids_[0:count]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = pyflann.FLANN()
            flann.build_index(vecs, **flann_params)
        count_list.append(count)
        time_list.append(t.ellapsed)
    count_arr = np.array(count_list)
    time_arr = np.array(time_list)
    pt.plot2(count_arr, time_arr, marker='-', equal_aspect=False,
             x_label='num_annotations', y_label='FLANN build time')
    pt.update()

# ------------
# NEW


def flann_add_time_experiment(update=False):
    """
    builds plot of number of annotations vs indexer build time.

    TODO: time experiment

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-flann_add_time_experiment
        profiler.py -m ibeis.model.hots.neighbor_index --test-flann_add_time_experiment

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> import ibeis
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> update = True
        >>> result = flann_add_time_experiment(update)
        >>> # verify results
        >>> print(result)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
        #>>> ibeis.main_loop({'ibs': ibs, 'back': None})

    """
    import ibeis
    import utool as ut
    import numpy as np
    import plottool as pt

    def make_flann_index(vecs, flann_params):
        flann = pyflann.FLANN()
        flann.build_index(vecs, **flann_params)
        return flann

    def get_reindex_time(ibs, daids, flann_params):
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann = make_flann_index(vecs, flann_params)  # NOQA
        return t.ellapsed

    def get_addition_time(ibs, daids, flann, flann_params):
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        with ut.Timer(verbose=False) as t:
            flann.add_points(vecs)
        return t.ellapsed

    # Input
    #ibs = ibeis.opendb(db='PZ_MTEST')
    #ibs = ibeis.opendb(db='GZ_ALL')
    ibs = ibeis.opendb(db='PZ_Master0')
    #max_ceiling = 32
    initial = 32
    reindex_stride = 32
    addition_stride = 16
    max_ceiling = 300001
    all_daids = ibs.get_valid_aids()
    max_num = min(max_ceiling, len(all_daids))
    flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()

    # Output
    count_list,  time_list_reindex  = [], []
    count_list2, time_list_addition = [], []

    # Setup
    #all_randomize_daids_ = ut.deterministic_shuffle(all_daids[:])
    all_randomize_daids_ = all_daids
    # ensure all features are computed
    ibs.get_annot_vecs(all_randomize_daids_)

    def reindex_step(count, count_list, time_list_reindex):
        daids    = all_randomize_daids_[0:count]
        ellapsed = get_reindex_time(ibs, daids, flann_params)
        count_list.append(count)
        time_list_reindex.append(ellapsed)

    def addition_step(count, flann, count_list2, time_list_addition):
        daids = all_randomize_daids_[count:count + 1]
        ellapsed = get_addition_time(ibs, daids, flann, flann_params)
        count_list2.append(count)
        time_list_addition.append(ellapsed)

    def make_initial_index(initial):
        daids = all_randomize_daids_[0:initial + 1]
        vecs = np.vstack(ibs.get_annot_vecs(daids))
        flann = make_flann_index(vecs, flann_params)
        return flann

    # Reindex Part
    reindex_lbl = 'Reindexing'
    _reindex_iter = range(1, max_num, reindex_stride)
    reindex_iter = ut.ProgressIter(_reindex_iter, lbl=reindex_lbl)
    for count in reindex_iter:
        reindex_step(count, count_list, time_list_reindex)

    # Add Part
    flann = make_initial_index(initial)
    addition_lbl = 'Addition'
    _addition_iter = range(initial + 1, max_num, addition_stride)
    addition_iter = ut.ProgressIter(_addition_iter, lbl=addition_lbl)
    for count in addition_iter:
        addition_step(count, flann, count_list2, time_list_addition)

    print('---')
    print('Reindex took time_list_reindex %.2s seconds' % sum(time_list_reindex))
    print('Addition took time_list_reindex  %.2s seconds' % sum(time_list_addition))
    print('---')
    statskw = dict(precision=2, newlines=True)
    print('Reindex stats ' + ut.get_stats_str(time_list_reindex, **statskw))
    print('Addition stats ' + ut.get_stats_str(time_list_addition, **statskw))

    print('Plotting')

    #with pt.FigureContext:

    next_fnum = iter(range(0, 2)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    pt.plot2(count_list, time_list_reindex, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=reindex_lbl + ' Time')

    #pt.figure(fnum=next_fnum())
    pt.plot2(count_list2, time_list_addition, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=addition_lbl + ' Time')

    pt
    pt.legend()
    if update:
        pt.update()


def augment_nnindexer_experiment(update=True):
    """

    python -c "import utool; print(utool.auto_docstr('ibeis.model.hots.neighbor_index', 'augment_nnindexer_experiment'))"

    CommandLine:
        profiler.py -m ibeis.model.hots.neighbor_index --test-augment_nnindexer_experiment
        python -m ibeis.model.hots.neighbor_index --test-augment_nnindexer_experiment

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> # build test data
        >>> show = ut.get_argflag('--show')
        >>> update = show
        >>> # execute function
        >>> augment_nnindexer_experiment(update)
        >>> # verify results
        >>> if show:
        ...     from matplotlib import pyplot as plt
        ...     plt.show()

    """
    import ibeis
    import plottool as pt
    # build test data
    ZEB_PLAIN = ibeis.const.Species.ZEB_PLAIN
    #ibs = ibeis.opendb('PZ_MTEST')
    ibs = ibeis.opendb('PZ_Master0')
    all_daids = ibs.get_valid_aids(species=ZEB_PLAIN)
    qreq_ = ibs.new_query_request(all_daids, all_daids)
    initial = 128
    addition_stride = 64
    max_ceiling = 10000
    max_num = min(max_ceiling, len(all_daids))

    # Clear Caches
    ibs.delete_flann_cachedir()
    clear_memcache()
    clear_uuid_cache(qreq_)

    # Setup
    all_randomize_daids_ = ut.deterministic_shuffle(all_daids[:])
    # ensure all features are computed
    #ibs.get_annot_vecs(all_randomize_daids_, ensure=True)
    #ibs.get_annot_fgweights(all_randomize_daids_, ensure=True)

    nnindexer_list = []
    addition_lbl = 'Addition'
    _addition_iter = range(initial + 1, max_num, addition_stride)
    addition_iter = ut.ProgressIter(_addition_iter, lbl=addition_lbl)
    time_list_addition = []
    #time_list_reindex = []
    count_list = []
    for count in addition_iter:
        aid_list_ = all_randomize_daids_[0:count]
        with ut.Timer(verbose=False) as t:
            nnindexer_ = request_augmented_ibeis_nnindexer(qreq_, aid_list_)
        nnindexer_list.append(nnindexer_)
        count_list.append(count)
        time_list_addition.append(t.ellapsed)
        print('===============\n\n')
    print(ut.list_str(time_list_addition))
    print(ut.list_str(list(map(id, nnindexer_list))))
    print(ut.list_str(list([nnindxer.cfgstr for nnindxer in nnindexer_list])))

    next_fnum = iter(range(0, 1)).next  # python3 PY3
    pt.figure(fnum=next_fnum())
    pt.plot2(count_list, time_list_addition, marker='-o', equal_aspect=False,
             x_label='num_annotations', label=addition_lbl + ' Time')
    pt.legend()
    if update:
        pt.update()


def request_background_nnindexer(qreq_, daid_list):
    """ FIXME: Duplicate code """
    global CURRENT_THREAD
    if CURRENT_THREAD is not None and not CURRENT_THREAD.is_alive():
        # Make sure this function doesn't run if it is already running
        return False
    print('Requesting background reindex')
    daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)
    nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
    flann_cachedir = qreq_.ibs.get_flann_cachedir()
    # Save inverted cache uuid mappings for
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    # Grab the keypoints names and image ids before query time?
    flann_params =  qreq_.qparams.flann_params
    # Get annot descriptors to index
    vecs_list = qreq_.ibs.get_annot_vecs(daid_list)
    fgws_list = get_fgweights_hack(qreq_, daid_list)
    preptup = prepare_index_data(daid_list, vecs_list, fgws_list, verbose=True)
    (ax2_aid, idx2_vec, idx2_fgw, idx2_ax, idx2_fx) = preptup
    use_memcache = False
    # Dont hash rowids when given enough info in nnindex_cfgstr
    flann_params['cores'] = 2  # Only ues a few cores in the background
    flannkw = dict(cache_dir=flann_cachedir, cfgstr=nnindex_cfgstr,
                   flann_params=flann_params, use_memcache=use_memcache,
                   use_params_hash=False)
    #cores = flann_params.get('cores', 0)
    # Build/Load the flann index
    #flann = nntool.flann_cache(idx2_vec, verbose=verbose, **flannkw)
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)

    threadobj = ut.spawn_background_process(
        background_flann_func, idx2_vec, flannkw, uuid_map_fpath, daids_hashid,
        visual_uuid_list, min_reindex_thresh)
    CURRENT_THREAD = threadobj


def background_flann_func(idx2_vec, flannkw, uuid_map_fpath, daids_hashid,
                          visual_uuid_list, min_reindex_thresh):
    """ FIXME: Duplicate code """
    print('Starting Background FLANN')
    # FIXME. dont use flann cache
    nntool.flann_cache(idx2_vec, **flannkw)
    if len(visual_uuid_list) > min_reindex_thresh:
        write_to_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
    print('Finished Background FLANN')


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.neighbor_index
        python -m ibeis.model.hots.neighbor_index --allexamples
        python -m ibeis.model.hots.neighbor_index --allexamples --noface --nosrc

        profiler.sh ibeis/model/hots/neighbor_index.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
