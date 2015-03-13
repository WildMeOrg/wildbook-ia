"""
TODO:
    Remove Bloat

module which handles the building and caching of individual flann indexes


CommandLine:
    # Runs the incremental query test
    # {0:testdb1, 1:PZ_MTEST, 2:GZ_ALL, 3:PZ_Master0}
    python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0
    python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:1
    python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:2
    python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:3

    utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0
    utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:1
    utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:3

"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import utool as ut
import pyflann
#import lockfile
from os.path import join
from os.path import basename, exists  # NOQA
from six.moves import range, zip, map  # NOQA
import vtool.nearest_neighbors as nntool
from ibeis.model.hots import hstypes
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[neighbor_index]', DEBUG=False)

NOCACHE_FLANN = ut.get_argflag('--nocache-flann')

# LRU cache for nn_indexers. Ensures that only a few are ever in memory
MAX_NEIGHBOR_CACHE_SIZE = 8
NEIGHBOR_CACHE = ut.get_lru_cache(MAX_NEIGHBOR_CACHE_SIZE)
# Background process for building indexes
CURRENT_THREAD = None
# Global map to keep track of UUID lists with prebuild indexers.
UUID_MAP = ut.ddict(dict)


class UUIDMapHyrbridCache(object):
    """
    Class that lets multiple ways of writing to the uuid_map
    be swapped in and out interchangably

    TODO: the global read / write should periodically sync itself to disk and it
    should be loaded from disk initially
    """
    def __init__(self):
        self.uuid_maps = ut.ddict(dict)
        #self.uuid_map_fpath = uuid_map_fpath
        #self.init(uuid_map_fpath, min_reindex_thresh)

    def init(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        #self.read_func  = self.read_uuid_map_cpkl
        #self.write_func = self.write_uuid_map_cpkl
        self.read_func  = self.read_uuid_map_dict
        self.write_func = self.write_uuid_map_dict

    #def __call__(self):
    #    return  self.read_func(*self.args, **self.kwargs)

    def dump(self, cachedir):
        # TODO: DUMP AND LOAD THIS HYBRID CACHE TO DISK
        #write_uuid_map_cpkl
        fname = 'uuid_maps_hybrid_cache.cPkl'
        cpkl_fpath = join(cachedir, fname)
        ut.lock_and_save_cPkl(cpkl_fpath, self.uuid_maps)

    def load(self, cachedir):
        fname = 'uuid_maps_hybrid_cache.cPkl'
        cpkl_fpath = join(cachedir, fname)
        self.uuid_maps = ut.lock_and_load_cPkl(cpkl_fpath)

    #def __setitem__(self, daids_hashid, visual_uuid_list):
    #    uuid_map_fpath = self.uuid_map_fpath
    #    self.write_func(uuid_map_fpath, visual_uuid_list, daids_hashid)

    @profile
    def read_uuid_map_dict(self, uuid_map_fpath, min_reindex_thresh):
        """ uses in memory dictionary instead of disk """
        uuid_map = self.uuid_maps[uuid_map_fpath]
        candidate_uuids = {
            key: val for key, val in six.iteritems(uuid_map)
            if len(val) >= min_reindex_thresh
        }
        return candidate_uuids

    @profile
    def write_uuid_map_dict(self, uuid_map_fpath, visual_uuid_list, daids_hashid):
        """ uses in memory dictionary instead of disk """
        #with ut.EmbedOnException():
        uuid_map = self.uuid_maps[uuid_map_fpath]
        uuid_map[daids_hashid] = visual_uuid_list

    #@profile
    #def read_uuid_map_shelf(self, uuid_map_fpath, min_reindex_thresh):
    #    #with ut.EmbedOnException():
    #    with lockfile.LockFile(uuid_map_fpath + '.lock'):
    #        with ut.shelf_open(uuid_map_fpath) as uuid_map:
    #            candidate_uuids = {
    #                key: val for key, val in six.iteritems(uuid_map)
    #                if len(val) >= min_reindex_thresh
    #            }
    #    return candidate_uuids

    #@profile
    #def write_uuid_map_shelf(self, uuid_map_fpath, visual_uuid_list, daids_hashid):
    #    print('Writing %d visual uuids to uuid map' % (len(visual_uuid_list)))
    #    with lockfile.LockFile(uuid_map_fpath + '.lock'):
    #        with ut.shelf_open(uuid_map_fpath) as uuid_map:
    #            uuid_map[daids_hashid] = visual_uuid_list

    #@profile
    #def read_uuid_map_cpkl(self, uuid_map_fpath, min_reindex_thresh):
    #    with lockfile.LockFile(uuid_map_fpath + '.lock'):
    #        #with ut.shelf_open(uuid_map_fpath) as uuid_map:
    #        try:
    #            uuid_map = ut.load_cPkl(uuid_map_fpath)
    #            candidate_uuids = {
    #                key: val for key, val in six.iteritems(uuid_map)
    #                if len(val) >= min_reindex_thresh
    #            }
    #        except IOError:
    #            return {}
    #    return candidate_uuids

    #@profile
    #def write_uuid_map_cpkl(self, uuid_map_fpath, visual_uuid_list, daids_hashid):
    #    """
    #    let the multi-indexer know about any big caches we've made multi-indexer.
    #    Also lets nnindexer know about other prebuilt indexers so it can attempt to
    #    just add points to them as to avoid a rebuild.
    #    """
    #    print('Writing %d visual uuids to uuid map' % (len(visual_uuid_list)))
    #    with lockfile.LockFile(uuid_map_fpath + '.lock'):
    #        try:
    #            uuid_map = ut.load_cPkl(uuid_map_fpath)
    #        except IOError:
    #            uuid_map = {}
    #        uuid_map[daids_hashid] = visual_uuid_list
    #        ut.save_cPkl(uuid_map_fpath, uuid_map)


UUID_MAP_CACHE = UUIDMapHyrbridCache()


@profile
def write_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid):
    """
    let the multi-indexer know about any big caches we've made multi-indexer.
    Also lets nnindexer know about other prebuilt indexers so it can attempt to
    just add points to them as to avoid a rebuild.
    """
    #UUID_MAP_CACHE[daids_hashid] = visual_uuid_list
    UUID_MAP_CACHE.write_uuid_map_dict(uuid_map_fpath, visual_uuid_list, daids_hashid)
    #with ContextUUIDMap(uuid_map_fpath, None) as uuid_map:
    #    uuid_map[daids_hashid] = visual_uuid_list


@profile
def read_uuid_map(uuid_map_fpath, min_reindex_thresh):

    #with ContextUUIDMap(uuid_map_fpath, min_reindex_thresh) as uuid_map:
    candidate_uuids = UUID_MAP_CACHE.read_uuid_map_dict(uuid_map_fpath, min_reindex_thresh)
    return candidate_uuids


@profile
def get_nnindexer_uuid_map_fpath(qreq_):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> # build test data
        >>> ibs, qreq_ = plh.get_pipeline_testdata(defaultdb='testdb1', preload=False)
        >>> uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
        >>> result = str(ut.path_ndir_split(uuid_map_fpath, 3))
        >>> print(result)
        _ibeis_cache/flann/uuid_map_FLANN(8_kdtrees)_FEAT(hesaff+sift_)_CHIP(sz450).cPkl
    """
    flann_cachedir = qreq_.ibs.get_flann_cachedir()
    # Have uuid shelf conditioned on the baseline flann and feature parameters
    flann_cfgstr    = qreq_.qparams.flann_cfgstr
    feat_cfgstr     = qreq_.qparams.feat_cfgstr
    uuid_map_cfgstr = ''.join((flann_cfgstr, feat_cfgstr))
    #uuid_map_ext    = '.shelf'
    uuid_map_ext    = '.cPkl'
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
        >>> ibs, qreq_ = plh.get_pipeline_testdata(defaultdb='testdb1', preload=False)
        >>> # execute function
        >>> fgws_list = clear_uuid_cache(qreq_)
        >>> # verify results
        >>> result = str(fgws_list)
        >>> print(result)
    """
    print('[nnindex] clearing uuid cache')
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    ut.delete(uuid_map_fpath)
    ut.delete(uuid_map_fpath + '.lock')
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
        >>> ibs, qreq_ = plh.get_pipeline_testdata(defaultdb='PZ_Master0', preload=False)
        >>> # execute function
        >>> print_uuid_cache(qreq_)
        >>> # verify results
        >>> result = str(nnindexer)
        >>> print(result)
    """
    print('[nnindex] clearing uuid cache')
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    candidate_uuids = read_uuid_map(uuid_map_fpath, 0)
    print(candidate_uuids)


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
        >>> use_memcache, max_covers, verbose = True, None, True
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
    new_daid_list, covered_aids_list = group_daids_by_cached_nnindexer(
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
              (len(covered_aids), len(new_daid_list)))
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

        new_vecs_list, new_fgws_list = get_support_data(qreq_, new_daid_list)
        base_nnindexer.add_support(new_daid_list, new_vecs_list, new_fgws_list, verbose=True)
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
            write_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
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
        if veryverbose or ut.VERYVERBOSE:
            print('... nnindex memcache hit: cfgstr=%s' % (nnindex_cfgstr,))
        nnindexer = NEIGHBOR_CACHE[nnindex_cfgstr]
    else:
        if veryverbose or ut.VERYVERBOSE:
            print('... nnindex memcache miss: cfgstr=%s' % (nnindex_cfgstr,))
        # Write to inverse uuid
        nnindexer = request_diskcached_ibeis_nnindexer(qreq_, daid_list, nnindex_cfgstr, verbose)
        # Write to memcache
        if ut.VERBOSE or ut.VERYVERBOSE:
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
    flann_params['checks'] = qreq_.qparams.checks
    # Get annot descriptors to index
    vecs_list, fgws_list = get_support_data(qreq_, daid_list)
    try:
        nnindexer = new_neighbor_index(
            daid_list, vecs_list, fgws_list, flann_params, cachedir,
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
        write_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
    return nnindexer


@profile
def group_daids_by_cached_nnindexer(qreq_, daid_list, min_reindex_thresh,
                                    max_covers=None):
    r"""
    FIXME: This function is slow due to ibs.get_annot_aids_from_visual_uuid
    282.253 seconds for 600 queries

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
        >>> daid_list = ibs.get_valid_aids(species=ZEB_PLAIN)[0:3]
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result1 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result1, ([1, 2, 3], []), 'pre request')
        >>> # TEST 2: SHOULD MAKE 123 COVERED
        >>> nnindexer = request_memcached_ibeis_nnindexer(qreq_, daid_list)
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result2 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result2, ([], [[1, 2, 3]]), 'post request')
    """
    ibs = qreq_.ibs
    # read which annotations have prebuilt caches
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    candidate_uuids = read_uuid_map(uuid_map_fpath, min_reindex_thresh)
    # find a maximum independent set cover of the requested annotations
    annot_vuuid_list = ibs.get_annot_visual_uuids(daid_list)  # 3.2 %
    covertup = ut.greedy_max_inden_setcover(
        candidate_uuids, annot_vuuid_list, max_covers)  # 0.2 %
    uncovered_vuuids, covered_vuuids_list, accepted_keys = covertup
    # return the grouped covered items (so they can be loaded) and
    # the remaining uuids which need to have an index computed.
    #
    uncovered_aids_ = ibs.get_annot_aids_from_visual_uuid(uncovered_vuuids)  # 28.0%
    covered_aids_list_ = ibs.unflat_map(
        ibs.get_annot_aids_from_visual_uuid, covered_vuuids_list)  # 68%
    # FIXME:
    uncovered_aids = sorted(uncovered_aids_)
    #covered_aids_list = list(map(sorted, covered_aids_list_))
    covered_aids_list = covered_aids_list_
    return uncovered_aids, covered_aids_list


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
        _VUUIDS((6)fvpdb9cud49@ll@+)_FLANN(8_kdtrees)_FEATWEIGHT(ON,uselabel,rf)_FEAT(hesaff+sift_)_CHIP(sz450)
    """
    flann_cfgstr      = qreq_.qparams.flann_cfgstr
    featweight_cfgstr = qreq_.qparams.featweight_cfgstr
    data_hashid   = get_data_cfgstr(qreq_.ibs, daid_list)
    nnindex_cfgstr = ''.join((data_hashid, flann_cfgstr, featweight_cfgstr))
    return nnindex_cfgstr


@profile
def get_fgweights_hack(qreq_, daid_list):
    """
    hack to get  feature weights. returns None if feature weights are turned off
    in config settings
    """
    # <HACK:featweight>
    if qreq_.qparams.fg_on:
        fgws_list = qreq_.ibs.get_annot_fgweights(
            daid_list, config2_=qreq_.get_internal_data_config2(), ensure=True)
    else:
        fgws_list = None
    return fgws_list
    # </HACK:featweight>


def get_support_data(qreq_, daid_list):
    vecs_list = qreq_.ibs.get_annot_vecs(daid_list, config2_=qreq_.get_internal_data_config2())
    fgws_list = get_fgweights_hack(qreq_, daid_list)
    return vecs_list, fgws_list


@profile
def new_neighbor_index(daid_list, vecs_list, fgws_list, flann_params, cachedir,
                       cfgstr, verbose=True):
    """
    constructs neighbor index independent of ibeis

    Args:
        daid_list (list):
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
        >>> vecs_list, fgws_list = get_support_data(qreq_, daid_list)
        >>> # execute function
        >>> nnindexer = new_neighbor_index(daid_list, vecs_list, fgws_list, flann_params, cachedir, cfgstr, verbose=True)

    """
    nnindexer = NeighborIndex(flann_params, cfgstr)
    # Initialize neighbor with unindexed data
    nnindexer.init_support(daid_list, vecs_list, fgws_list, verbose=verbose)
    # Load or build the indexing structure
    nnindexer.load_or_build(cachedir, verbose=verbose)
    return nnindexer


@profile
def prepare_index_data(aid_list, vecs_list, fgws_list, verbose=True):
    """
    flattens vecs_list and builds a reverse index from the flattened indices
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
        nnindexer.max_distance = None  # max possible distance for normalization
        nnindexer.cfgstr   = cfgstr  # configuration id
        nnindexer.flann_params = flann_params
        nnindexer.cores  = flann_params.get('cores', 0)
        nnindexer.checks = flann_params.get('checks', 1028)
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
            raise AssertionError('NNindexer should get uint8s right now unless the algorithm has changed')
        nnindexer.max_distance_sqrd = nnindexer.max_distance ** 2

    @profile
    def add_support(nnindexer, new_daid_list, new_vecs_list, new_fgws_list,
                    verbose=True):
        """
        adds support data (aka data to be indexed)

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> nnindexer, qreq_, ibs = test_nnindexer()
            >>> new_daid_list = [2, 3, 4]
            >>> K = 2
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K)
            >>> new_vecs_list, new_fgws_list = get_support_data(qreq_, new_daid_list)
            >>> nnindexer.add_support(new_daid_list, new_vecs_list, new_fgws_list)
            >>> (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K)
            >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        nAnnots = nnindexer.num_indexed_annots()
        nVecs = nnindexer.num_indexed_vecs()
        nNewAnnots = len(new_daid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNewAnnots)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list)
        nNewVecs = len(new_idx2_vec)
        if verbose or ut.VERYVERBOSE:
            print('[nnindex] Adding %d vecs from %d annots to nnindex with %d vecs and %d annots' %
                  (nNewVecs, nNewAnnots, nVecs, nAnnots))
        print('STACKING')
        # Stack inverted information
        ##---
        if not hasattr(nnindexer, 'old_vecs'):
            nnindexer.old_vecs = []
        # Try to hack in a way to keep the old memory
        old_idx2_vec = nnindexer.idx2_vec
        nnindexer.old_vecs.append(old_idx2_vec)
        if nnindexer.idx2_fgw is not None:
            new_idx2_fgw = np.hstack(new_fgws_list)
            nnindexer.old_vecs.append(new_idx2_fgw)
        ##---
        _ax2_aid = np.hstack((nnindexer.ax2_aid, new_daid_list))
        _idx2_ax = np.hstack((nnindexer.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((nnindexer.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((old_idx2_vec, new_idx2_vec))
        if nnindexer.idx2_fgw is not None:
            _idx2_fgw = np.hstack((nnindexer.idx2_fgw, new_idx2_fgw))
        print('REPLACING')
        nnindexer.ax2_aid  = _ax2_aid
        nnindexer.idx2_ax  = _idx2_ax
        nnindexer.idx2_vec = _idx2_vec
        nnindexer.idx2_fx  = _idx2_fx
        if nnindexer.idx2_fgw is not None:
            nnindexer.idx2_fgw = _idx2_fgw
        #nnindexer.idx2_kpts   = None
        #nnindexer.idx2_oris   = None
        # Add new points to flann structure
        print('ADD POINTS (FIXME: SOMETIMES SEGFAULT OCCURS)')
        print('new_idx2_vec.dtype = %r' % new_idx2_vec.dtype)
        print('new_idx2_vec.shape = %r' % (new_idx2_vec.shape,))
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
        if ut.VERYVERBOSE or verbose or (not ut.QUIET and num_vecs > notify_num):
            print('[nnindex] ...building kdtree over %d points (this may take a sec).' % num_vecs)
        idx2_vec = nnindexer.idx2_vec
        flann_params = nnindexer.flann_params
        nnindexer.flann.build_index(idx2_vec, **flann_params)

    # ---- <cachable_interface> ---

    def save(nnindexer, cachedir, verbose=True):
        flann_fpath = nnindexer.get_fpath(cachedir)
        if ut.VERYVERBOSE or verbose:
            print('[nnindex] flann.save_index(%r)' % ut.path_ndir_split(flann_fpath, n=5))
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
        if ut.VERYVERBOSE or ut.VERBOSE:
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
    def knn(nnindexer, qfx2_vec, K):
        """
        Returns the indices and squared distance to the nearest K neighbors.
        The distance is noramlized between zero and one using
        VEC_PSEUDO_MAX_DISTANCE = (np.sqrt(2) * VEC_PSEUDO_MAX)

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
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> K = 2
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K)
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
            >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K)
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
                qfx2_vec, K, checks=nnindexer.checks, cores=nnindexer.cores)
            # Ensure that distance returned are between 0 and 1
            qfx2_dist = np.divide(qfx2_dist, nnindexer.max_distance_sqrd)
            #qfx2_dist = np.sqrt(qfx2_dist) / nnindexer.max_distance_sqrd
        return (qfx2_idx, qfx2_dist)

    def empty_neighbors(nnindexer, nQfx, K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    def num_indexed_vecs(nnindexer):
        return len(nnindexer.idx2_vec)

    def num_indexed_annots(nnindexer):
        return len(nnindexer.ax2_aid)

    def get_indexed_aids(nnindexer):
        return nnindexer.ax2_aid

    def get_nn_vecs(nnindexer, qfx2_nnidx):
        """ gets matching vectors """
        return nnindexer.idx2_vec.take(qfx2_nnidx, axis=0)

    def get_nn_axs(nnindexer, qfx2_nnidx):
        """ gets matching internal annotation indices """
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
            >>> cfgdict = dict()
            >>> ibs, qreq_ = plh.get_pipeline_testdata(defaultdb='testdb1', cfgdict=cfgdict, preload=True)
            >>> nnindexer = qreq_.indexer
            >>> qfx2_vec = qreq_.ibs.get_annot_vecs(qreq_.get_internal_qaids()[0], config2_=qreq_.get_internal_query_config2())
            >>> num_neighbors = 4
            >>> (qfx2_nnidx, qfx2_dist) = nnindexer.knn(qfx2_vec, num_neighbors)
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
        r"""
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
        idx2_vec, idx2_ax, idx2_fx = nntool.invertible_stack(vecs_list, ax_list)
        assert idx2_vec.shape[0] == idx2_ax.shape[0]
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
    except MemoryError as ex:
        ut.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if ut.VERYVERBOSE or verbose:
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


# ------------
# NEW


@profile
def check_background_process():
    """
    checks to see if the process has finished and then
    writes the uuid map to disk
    """
    global CURRENT_THREAD
    if CURRENT_THREAD is None or CURRENT_THREAD.is_alive():
        print('[FG] background thread is not ready yet')
        return False
    # Get info set in background process
    finishtup = CURRENT_THREAD.finishtup
    (uuid_map_fpath, daids_hashid, visual_uuid_list, min_reindex_thresh) = finishtup
    # Clean up background process
    CURRENT_THREAD.join()
    CURRENT_THREAD = None
    # Write data to current uuidcache
    if len(visual_uuid_list) > min_reindex_thresh:
        write_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
    return True


def can_request_background_nnindexer():
    return CURRENT_THREAD is None or not CURRENT_THREAD.is_alive()


@profile
def request_background_nnindexer(qreq_, daid_list):
    """ FIXME: Duplicate code

    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        daid_list (list):

    Returns:
        bool:

    CommandLine:
        python -m ibeis.model.hots.neighbor_index --test-request_background_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> from ibeis.model.hots import neighbor_index  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> daid_list = ibs.get_valid_aids(species=ibeis.const.Species.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> # execute function
        >>> neighbor_index.request_background_nnindexer(qreq_, daid_list)
        >>> # verify results
        >>> result = str(False)
        >>> print(result)
    """
    global CURRENT_THREAD
    print('Requesting background reindex')
    if not can_request_background_nnindexer():
        # Make sure this function doesn't run if it is already running
        print('REQUEST DENIED')
        return False
    print('REQUEST ACCPETED')
    daids_hashid = qreq_.ibs.get_annot_hashid_visual_uuid(daid_list)
    cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
    cachedir = qreq_.ibs.get_flann_cachedir()
    # Save inverted cache uuid mappings for
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    # Grab the keypoints names and image ids before query time?
    flann_params =  qreq_.qparams.flann_params
    # Get annot descriptors to index
    vecs_list, fgws_list = get_support_data(qreq_, daid_list)
    # Dont hash rowids when given enough info in nnindex_cfgstr
    flann_params['cores'] = 2  # Only ues a few cores in the background
    # Build/Load the flann index
    #flann = nntool.flann_cache(idx2_vec, verbose=verbose, **flannkw)
    uuid_map_fpath   = get_nnindexer_uuid_map_fpath(qreq_)
    visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)

    # set temporary attribute for when the thread finishes
    finishtup = (uuid_map_fpath, daids_hashid, visual_uuid_list, min_reindex_thresh)
    CURRENT_THREAD = ut.spawn_background_process(
        background_flann_func, cachedir, daid_list, vecs_list, fgws_list,
        flann_params, cfgstr)

    CURRENT_THREAD.finishtup = finishtup


def background_flann_func(cachedir, daid_list, vecs_list, fgws_list, flann_params, cfgstr,
                          uuid_map_fpath, daids_hashid,
                          visual_uuid_list, min_reindex_thresh):
    """ FIXME: Duplicate code """
    print('[BG] Starting Background FLANN')
    # FIXME. dont use flann cache
    #nntool.flann_cache(idx2_vec, **flannkw)
    nnindexer = NeighborIndex(flann_params, cfgstr)
    # Initialize neighbor with unindexed data
    nnindexer.init_support(daid_list, vecs_list, fgws_list, verbose=True)
    # Load or build the indexing structure
    nnindexer.load_or_build(cachedir, verbose=True)
    if len(visual_uuid_list) > min_reindex_thresh:
        write_uuid_map(uuid_map_fpath, visual_uuid_list, daids_hashid)
    print('[BG] Finished Background FLANN')


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.neighbor_index
        python -m ibeis.model.hots.neighbor_index --allexamples
        python -m ibeis.model.hots.neighbor_index --allexamples --noface --nosrc

        utprof.sh ibeis/model/hots/neighbor_index.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
