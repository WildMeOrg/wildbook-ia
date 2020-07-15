# -*- coding: utf-8 -*-
"""
NEEDS CLEANUP
"""
from __future__ import absolute_import, division, print_function
from os.path import join
import six
import utool as ut
from six.moves import range, zip, map  # NOQA
from wbia.algo.hots import _pipeline_helpers as plh  # NOQA
from wbia.algo.hots.neighbor_index import NeighborIndex, get_support_data

(print, rrr, profile) = ut.inject2(__name__)


USE_HOTSPOTTER_CACHE = not ut.get_argflag('--nocache-hs')
NOCACHE_UUIDS = ut.get_argflag('--nocache-uuids') and USE_HOTSPOTTER_CACHE

# LRU cache for nn_indexers. Ensures that only a few are ever in memory
# MAX_NEIGHBOR_CACHE_SIZE = ut.get_argval('--max-neighbor-cachesize', type_=int, default=2)
MAX_NEIGHBOR_CACHE_SIZE = ut.get_argval('--max-neighbor-cachesize', type_=int, default=1)
# Background process for building indexes
CURRENT_THREAD = None
# Global map to keep track of UUID lists with prebuild indexers.
UUID_MAP = ut.ddict(dict)
NEIGHBOR_CACHE = ut.get_lru_cache(MAX_NEIGHBOR_CACHE_SIZE)


class UUIDMapHyrbridCache(object):
    """
    Class that lets multiple ways of writing to the uuid_map
    be swapped in and out interchangably

    TODO: the global read / write should periodically sync itself to disk and it
    should be loaded from disk initially
    """

    def __init__(self):
        self.uuid_maps = ut.ddict(dict)
        # self.uuid_map_fpath = uuid_map_fpath
        # self.init(uuid_map_fpath, min_reindex_thresh)

    def init(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # self.read_func  = self.read_uuid_map_cpkl
        # self.write_func = self.write_uuid_map_cpkl
        self.read_func = self.read_uuid_map_dict
        self.write_func = self.write_uuid_map_dict

    def dump(self, cachedir):
        # TODO: DUMP AND LOAD THIS HYBRID CACHE TO DISK
        # write_uuid_map_cpkl
        fname = 'uuid_maps_hybrid_cache.cPkl'
        cpkl_fpath = join(cachedir, fname)
        ut.lock_and_save_cPkl(cpkl_fpath, self.uuid_maps)

    def load(self, cachedir):
        """
        Returns a cache UUIDMap
        """
        fname = 'uuid_maps_hybrid_cache.cPkl'
        cpkl_fpath = join(cachedir, fname)
        self.uuid_maps = ut.lock_and_load_cPkl(cpkl_fpath)

    # def __call__(self):
    #    return  self.read_func(*self.args, **self.kwargs)

    # def __setitem__(self, daids_hashid, visual_uuid_list):
    #    uuid_map_fpath = self.uuid_map_fpath
    #    self.write_func(uuid_map_fpath, visual_uuid_list, daids_hashid)

    # @profile
    # def read_uuid_map_shelf(self, uuid_map_fpath, min_reindex_thresh):
    #    #with ut.EmbedOnException():
    #    with lockfile.LockFile(uuid_map_fpath + '.lock'):
    #        with ut.shelf_open(uuid_map_fpath) as uuid_map:
    #            candidate_uuids = {
    #                key: val for key, val in six.iteritems(uuid_map)
    #                if len(val) >= min_reindex_thresh
    #            }
    #    return candidate_uuids

    # @profile
    # def write_uuid_map_shelf(self, uuid_map_fpath, visual_uuid_list, daids_hashid):
    #    print('Writing %d visual uuids to uuid map' % (len(visual_uuid_list)))
    #    with lockfile.LockFile(uuid_map_fpath + '.lock'):
    #        with ut.shelf_open(uuid_map_fpath) as uuid_map:
    #            uuid_map[daids_hashid] = visual_uuid_list

    # @profile
    # def read_uuid_map_cpkl(self, uuid_map_fpath, min_reindex_thresh):
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

    # @profile
    # def write_uuid_map_cpkl(self, uuid_map_fpath, visual_uuid_list, daids_hashid):
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

    @profile
    def read_uuid_map_dict(self, uuid_map_fpath, min_reindex_thresh):
        """ uses in memory dictionary instead of disk """
        uuid_map = self.uuid_maps[uuid_map_fpath]
        candidate_uuids = {
            key: val
            for key, val in six.iteritems(uuid_map)
            if len(val) >= min_reindex_thresh
        }
        return candidate_uuids

    @profile
    def write_uuid_map_dict(self, uuid_map_fpath, visual_uuid_list, daids_hashid):
        """
        uses in memory dictionary instead of disk

        let the multi-indexer know about any big caches we've made multi-indexer.
        Also lets nnindexer know about other prebuilt indexers so it can attempt to
        just add points to them as to avoid a rebuild.
        """
        if NOCACHE_UUIDS:
            print('uuid cache is off')
            return
        # with ut.EmbedOnException():
        uuid_map = self.uuid_maps[uuid_map_fpath]
        uuid_map[daids_hashid] = visual_uuid_list


UUID_MAP_CACHE = UUIDMapHyrbridCache()


# @profile
def get_nnindexer_uuid_map_fpath(qreq_):
    """
    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache get_nnindexer_uuid_map_fpath

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='testdb1', p='default:fgw_thresh=.3')
        >>> uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
        >>> result = str(ut.path_ndir_split(uuid_map_fpath, 3))
        >>> print(result)

        .../_wbia_cache/flann/uuid_map_mzwwsbjisbkdxorl.cPkl
        .../_wbia_cache/flann/uuid_map_FLANN(8_kdtrees_fgwthrsh=0.3)_Feat(hesaff+sift)_Chip(sz700,width).cPkl
        .../_wbia_cache/flann/uuid_map_FLANN(8_kdtrees)_Feat(hesaff+sift)_Chip(sz700,width).cPkl
        .../_wbia_cache/flann/uuid_map_FLANN(8_kdtrees)_FEAT(hesaff+sift_)_CHIP(sz450).cPkl
    """
    flann_cachedir = qreq_.ibs.get_flann_cachedir()
    # Have uuid shelf conditioned on the baseline flann and feature parameters
    flann_cfgstr = qreq_.qparams.flann_cfgstr
    feat_cfgstr = qreq_.qparams.feat_cfgstr
    chip_cfgstr = qreq_.qparams.chip_cfgstr
    featweight_cfgstr = qreq_.qparams.featweight_cfgstr
    if qreq_.qparams.fgw_thresh is None or qreq_.qparams.fgw_thresh == 0:
        uuid_map_cfgstr = ''.join((flann_cfgstr, feat_cfgstr, chip_cfgstr))
    else:
        uuid_map_cfgstr = ''.join(
            (flann_cfgstr, featweight_cfgstr, feat_cfgstr, chip_cfgstr)
        )
    # uuid_map_ext    = '.shelf'
    uuid_map_ext = '.cPkl'
    uuid_map_prefix = 'uuid_map'
    uuid_map_fname = ut.consensed_cfgstr(uuid_map_prefix, uuid_map_cfgstr) + uuid_map_ext
    uuid_map_fpath = join(flann_cachedir, uuid_map_fname)
    return uuid_map_fpath


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
        python -m wbia.algo.hots.neighbor_index_cache --test-build_nnindex_cfgstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(db='testdb1')
        >>> daid_list = ibs.get_valid_aids(species=wbia.const.TEST_SPECIES.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=dict(fg_on=False))
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> result = str(nnindex_cfgstr)
        >>> print(result)

        _VUUIDS((6)ylydksaqdigdecdd)_FLANN(8_kdtrees)_FeatureWeight(detector=cnn,sz256,thresh=20,ksz=20,enabled=False)_FeatureWeight(detector=cnn,sz256,thresh=20,ksz=20,enabled=False)

        _VUUIDS((6)ylydksaqdigdecdd)_FLANN(8_kdtrees)_FEATWEIGHT(OFF)_FEAT(hesaff+sift_)_CHIP(sz450)
    """
    flann_cfgstr = qreq_.qparams.flann_cfgstr
    featweight_cfgstr = qreq_.qparams.featweight_cfgstr
    feat_cfgstr = qreq_.qparams.feat_cfgstr
    chip_cfgstr = qreq_.qparams.chip_cfgstr
    # FIXME; need to include probchip (or better yet just use depcache)
    # probchip_cfgstr = qreq_.qparams.chip_cfgstr
    data_hashid = get_data_cfgstr(qreq_.ibs, daid_list)
    nnindex_cfgstr = ''.join(
        (data_hashid, flann_cfgstr, featweight_cfgstr, feat_cfgstr, chip_cfgstr)
    )
    return nnindex_cfgstr


def clear_memcache():
    global NEIGHBOR_CACHE
    NEIGHBOR_CACHE.clear()


def clear_uuid_cache(qreq_):
    """
    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache --test-clear_uuid_cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='testdb1', p='default:fg_on=True')
        >>> fgws_list = clear_uuid_cache(qreq_)
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
        python -m wbia.algo.hots.neighbor_index_cache --test-print_uuid_cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='PZ_Master0', p='default:fg_on=False')
        >>> print_uuid_cache(qreq_)
        >>> result = str(nnindexer)
        >>> print(result)
    """
    print('[nnindex] clearing uuid cache')
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    candidate_uuids = UUID_MAP_CACHE.read_uuid_map_dict(uuid_map_fpath, 0)
    print(candidate_uuids)


def request_wbia_nnindexer(qreq_, verbose=True, **kwargs):
    """
    CALLED BY QUERYREQUST::LOAD_INDEXER
    IBEIS interface into neighbor_index_cache

    Args:
        qreq_ (QueryRequest): hyper-parameters

    Returns:
        NeighborIndexer: nnindexer

    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache request_wbia_nnindexer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> nnindexer, qreq_, ibs = testdata_nnindexer(None)
        >>> nnindexer = request_wbia_nnindexer(qreq_)
    """
    daid_list = qreq_.get_internal_daids()
    if not hasattr(qreq_.qparams, 'use_augmented_indexer'):
        qreq_.qparams.use_augmented_indexer = True
    if False and qreq_.qparams.use_augmented_indexer:
        nnindexer = request_augmented_wbia_nnindexer(qreq_, daid_list, **kwargs)
    else:
        nnindexer = request_memcached_wbia_nnindexer(qreq_, daid_list, **kwargs)
    return nnindexer


def request_augmented_wbia_nnindexer(
    qreq_, daid_list, verbose=True, use_memcache=True, force_rebuild=False, memtrack=None,
):
    r"""
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
        python -m wbia.algo.hots.neighbor_index_cache --test-request_augmented_wbia_nnindexer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ZEB_PLAIN = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> ibs = wbia.opendb('testdb1')
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
        >>> nnindexer = request_augmented_wbia_nnindexer(qreq_, aid_list)
        >>> # assert the fallback
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result2 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result2, ([4, 5, 6], [[1, 2, 3]]), 'pre augment')
        >>> # Should augment
        >>> nnindexer = request_augmented_wbia_nnindexer(qreq_, daid_list)
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result3 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result3, ([], [[1, 2, 3, 4, 5, 6]]), 'post augment')
        >>> # Should fallback
        >>> nnindexer2 = request_augmented_wbia_nnindexer(qreq_, daid_list)
        >>> assert nnindexer is nnindexer2
    """
    global NEIGHBOR_CACHE
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    if not force_rebuild:
        new_daid_list, covered_aids_list = group_daids_by_cached_nnindexer(
            qreq_, daid_list, min_reindex_thresh, max_covers=1
        )
        can_augment = len(covered_aids_list) > 0 and not ut.list_set_equal(
            covered_aids_list[0], daid_list
        )
    else:
        can_augment = False
    if verbose:
        print('[aug] Requesting augmented nnindexer')
    if can_augment:
        covered_aids = covered_aids_list[0]
        if verbose:
            print(
                '[aug] Augmenting index %r old daids with %d new daids'
                % (len(covered_aids), len(new_daid_list))
            )
        # Load the base covered indexer
        # THIS SHOULD LOAD NOT REBUILD IF THE UUIDS ARE COVERED
        base_nnindexer = request_memcached_wbia_nnindexer(
            qreq_, covered_aids, verbose=verbose, use_memcache=use_memcache
        )
        # Remove this indexer from the memcache because we are going to change it
        if NEIGHBOR_CACHE.has_key(  # NOQA (has_key is for a lru cache)
            base_nnindexer.cfgstr
        ):
            print('Removing key from memcache')
            NEIGHBOR_CACHE[base_nnindexer.cfgstr] = None
            del NEIGHBOR_CACHE[base_nnindexer.cfgstr]

        support_data = get_support_data(qreq_, new_daid_list)
        (new_vecs_list, new_fgws_list, new_fxs_list) = support_data
        base_nnindexer.add_support(
            new_daid_list, new_vecs_list, new_fgws_list, new_fxs_list, verbose=True
        )
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
            daids_hashid = get_data_cfgstr(qreq_.ibs, daid_list)
            visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
            UUID_MAP_CACHE.write_uuid_map_dict(
                uuid_map_fpath, visual_uuid_list, daids_hashid
            )
        # Write to memcache
        if ut.VERBOSE:
            print('[aug] Wrote to memcache=%r' % (nnindex_cfgstr,))
        NEIGHBOR_CACHE[nnindex_cfgstr] = nnindexer
        return nnindexer
    else:
        # if ut.VERBOSE:
        if verbose:
            print('[aug] Nothing to augment, fallback to memcache')
        # Fallback
        nnindexer = request_memcached_wbia_nnindexer(
            qreq_,
            daid_list,
            verbose=verbose,
            use_memcache=use_memcache,
            force_rebuild=force_rebuild,
            memtrack=memtrack,
        )
        return nnindexer


def request_memcached_wbia_nnindexer(
    qreq_,
    daid_list,
    use_memcache=True,
    verbose=ut.NOT_QUIET,
    veryverbose=False,
    force_rebuild=False,
    memtrack=None,
    prog_hook=None,
):
    r"""
    FOR INTERNAL USE ONLY
    takes custom daid list. might not be the same as what is in qreq_

    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache --test-request_memcached_wbia_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> qreq_.qparams.min_reindex_thresh = 3
        >>> ZEB_PLAIN = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> daid_list = ibs.get_valid_aids(species=ZEB_PLAIN)[0:3]
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> verbose = True
        >>> use_memcache = True
        >>> # execute function
        >>> nnindexer = request_memcached_wbia_nnindexer(qreq_, daid_list, use_memcache)
        >>> # verify results
        >>> result = str(nnindexer)
        >>> print(result)
    """
    global NEIGHBOR_CACHE
    # try:
    if veryverbose:
        print('[nnindex.MEMCACHE] len(NEIGHBOR_CACHE) = %r' % (len(NEIGHBOR_CACHE),))
        # the lru cache wont be recognized by get_object_size_str, cast to pure python objects
        print(
            '[nnindex.MEMCACHE] size(NEIGHBOR_CACHE) = %s'
            % (ut.get_object_size_str(NEIGHBOR_CACHE.items()),)
        )
    # if memtrack is not None:
    #    memtrack.report('IN REQUEST MEMCACHE')
    nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
    # neighbor memory cache
    if (
        not force_rebuild
        and use_memcache
        and NEIGHBOR_CACHE.has_key(nnindex_cfgstr)  # NOQA (has_key is for a lru cache)
    ):
        if veryverbose or ut.VERYVERBOSE or ut.VERBOSE:
            print('... nnindex memcache hit: cfgstr=%s' % (nnindex_cfgstr,))
        nnindexer = NEIGHBOR_CACHE[nnindex_cfgstr]
    else:
        if veryverbose or ut.VERYVERBOSE or ut.VERBOSE:
            print('... nnindex memcache miss: cfgstr=%s' % (nnindex_cfgstr,))
        # Write to inverse uuid
        nnindexer = request_diskcached_wbia_nnindexer(
            qreq_,
            daid_list,
            nnindex_cfgstr,
            verbose,
            force_rebuild=force_rebuild,
            memtrack=memtrack,
            prog_hook=prog_hook,
        )
        NEIGHBOR_CACHE_WRITE = True
        if NEIGHBOR_CACHE_WRITE:
            # Write to memcache
            if ut.VERBOSE or ut.VERYVERBOSE:
                print('[disk] Write to memcache=%r' % (nnindex_cfgstr,))
            NEIGHBOR_CACHE[nnindex_cfgstr] = nnindexer
        else:
            if ut.VERBOSE or ut.VERYVERBOSE:
                print('[disk] Did not write to memcache=%r' % (nnindex_cfgstr,))
    return nnindexer


def request_diskcached_wbia_nnindexer(
    qreq_,
    daid_list,
    nnindex_cfgstr=None,
    verbose=True,
    force_rebuild=False,
    memtrack=None,
    prog_hook=None,
):
    r"""
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
        python -m wbia.algo.hots.neighbor_index_cache --test-request_diskcached_wbia_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> daid_list = ibs.get_valid_aids(species=wbia.const.TEST_SPECIES.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> verbose = True
        >>> # execute function
        >>> nnindexer = request_diskcached_wbia_nnindexer(qreq_, daid_list, nnindex_cfgstr, verbose)
        >>> # verify results
        >>> result = str(nnindexer)
        >>> print(result)
    """
    if nnindex_cfgstr is None:
        nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
    cfgstr = nnindex_cfgstr
    cachedir = qreq_.ibs.get_flann_cachedir()
    flann_params = qreq_.qparams.flann_params
    flann_params['checks'] = qreq_.qparams.checks
    # if memtrack is not None:
    #    memtrack.report('[PRE SUPPORT]')
    # Get annot descriptors to index
    if prog_hook is not None:
        prog_hook.set_progress(1, 3, 'Loading support data for indexer')
    print('[nnindex] Loading support data for indexer')
    vecs_list, fgws_list, fxs_list = get_support_data(qreq_, daid_list)
    if memtrack is not None:
        memtrack.report('[AFTER GET SUPPORT DATA]')
    try:
        nnindexer = new_neighbor_index(
            daid_list,
            vecs_list,
            fgws_list,
            fxs_list,
            flann_params,
            cachedir,
            cfgstr=cfgstr,
            verbose=verbose,
            force_rebuild=force_rebuild,
            memtrack=memtrack,
            prog_hook=prog_hook,
        )
    except Exception as ex:
        ut.printex(
            ex, True, msg_='cannot build inverted index', key_list=['ibs.get_infostr()']
        )
        raise
    # Record these uuids in the disk based uuid map so they can be augmented if
    # needed
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    if len(daid_list) > min_reindex_thresh:
        uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
        daids_hashid = get_data_cfgstr(qreq_.ibs, daid_list)
        visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)
        UUID_MAP_CACHE.write_uuid_map_dict(uuid_map_fpath, visual_uuid_list, daids_hashid)
        if memtrack is not None:
            memtrack.report('[AFTER WRITE_UUID_MAP]')
    return nnindexer


def group_daids_by_cached_nnindexer(
    qreq_, daid_list, min_reindex_thresh, max_covers=None
):
    r"""
    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache --test-group_daids_by_cached_nnindexer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ZEB_PLAIN = wbia.const.TEST_SPECIES.ZEB_PLAIN
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
        >>> nnindexer = request_memcached_wbia_nnindexer(qreq_, daid_list)
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(
        ...     qreq_, daid_list, min_reindex_thresh, max_covers)
        >>> result2 = uncovered_aids, covered_aids_list
        >>> ut.assert_eq(result2, ([], [[1, 2, 3]]), 'post request')
    """
    ibs = qreq_.ibs
    # read which annotations have prebuilt caches
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    candidate_uuids = UUID_MAP_CACHE.read_uuid_map_dict(
        uuid_map_fpath, min_reindex_thresh
    )
    # find a maximum independent set cover of the requested annotations
    annot_vuuid_list = ibs.get_annot_visual_uuids(daid_list)  # 3.2 %
    covertup = ut.greedy_max_inden_setcover(
        candidate_uuids, annot_vuuid_list, max_covers
    )  # 0.2 %
    uncovered_vuuids, covered_vuuids_list, accepted_keys = covertup
    # return the grouped covered items (so they can be loaded) and
    # the remaining uuids which need to have an index computed.
    #
    uncovered_aids_ = ibs.get_annot_aids_from_visual_uuid(uncovered_vuuids)  # 28.0%
    covered_aids_list_ = ibs.unflat_map(
        ibs.get_annot_aids_from_visual_uuid, covered_vuuids_list
    )  # 68%
    # FIXME:
    uncovered_aids = sorted(uncovered_aids_)
    # covered_aids_list = list(map(sorted, covered_aids_list_))
    covered_aids_list = covered_aids_list_
    return uncovered_aids, covered_aids_list


def get_data_cfgstr(ibs, daid_list):
    """ part 2 data hash id """
    daids_hashid = ibs.get_annot_hashid_visual_uuid(daid_list)
    return daids_hashid


def new_neighbor_index(
    daid_list,
    vecs_list,
    fgws_list,
    fxs_list,
    flann_params,
    cachedir,
    cfgstr,
    force_rebuild=False,
    verbose=True,
    memtrack=None,
    prog_hook=None,
):
    r"""
    constructs neighbor index independent of wbia

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

    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache --test-new_neighbor_index

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='testdb1', a='default:species=zebra_plains', p='default:fgw_thresh=.999')
        >>> daid_list = qreq_.daids
        >>> nnindex_cfgstr = build_nnindex_cfgstr(qreq_, daid_list)
        >>> ut.exec_funckw(new_neighbor_index, globals())
        >>> cfgstr = nnindex_cfgstr
        >>> cachedir     = qreq_.ibs.get_flann_cachedir()
        >>> flann_params = qreq_.qparams.flann_params
        >>> # Get annot descriptors to index
        >>> vecs_list, fgws_list, fxs_list = get_support_data(qreq_, daid_list)
        >>> nnindexer = new_neighbor_index(daid_list, vecs_list, fgws_list, fxs_list, flann_params, cachedir, cfgstr, verbose=True)
        >>> result = ('nnindexer.ax2_aid = %s' % (str(nnindexer.ax2_aid),))
        >>> print(result)
        nnindexer.ax2_aid = [1 2 3 4 5 6]
    """
    nnindexer = NeighborIndex(flann_params, cfgstr)
    # if memtrack is not None:
    #    memtrack.report('CREATEED NEIGHTOB INDEX')
    # Initialize neighbor with unindexed data
    nnindexer.init_support(daid_list, vecs_list, fgws_list, fxs_list, verbose=verbose)
    if memtrack is not None:
        memtrack.report('AFTER INIT SUPPORT')
    # Load or build the indexing structure
    nnindexer.ensure_indexer(
        cachedir,
        verbose=verbose,
        force_rebuild=force_rebuild,
        memtrack=memtrack,
        prog_hook=prog_hook,
    )
    if memtrack is not None:
        memtrack.report('AFTER LOAD OR BUILD')
    return nnindexer


def testdata_nnindexer(dbname='testdb1', with_indexer=True, use_memcache=True):
    r"""

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> nnindexer, qreq_, ibs = testdata_nnindexer('PZ_Master1')
        >>> S = np.cov(nnindexer.idx2_vec.T)
        >>> import wbia.plottool as pt
        >>> pt.ensureqt()
        >>> pt.plt.imshow(S)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> nnindexer, qreq_, ibs = testdata_nnindexer()
    """
    import wbia

    daid_list = [7, 8, 9, 10, 11]
    ibs = wbia.opendb(db=dbname)
    # use_memcache isn't use here because we aren't lazy loading the indexer
    cfgdict = dict(fg_on=False)
    qreq_ = ibs.new_query_request(
        daid_list, daid_list, use_memcache=use_memcache, cfgdict=cfgdict
    )
    if with_indexer:
        # we do an explicit creation of an indexer for these tests
        nnindexer = request_wbia_nnindexer(qreq_, use_memcache=use_memcache)
    else:
        nnindexer = None
    return nnindexer, qreq_, ibs


# ------------
# NEW


def check_background_process():
    r"""
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
        UUID_MAP_CACHE.write_uuid_map_dict(uuid_map_fpath, visual_uuid_list, daids_hashid)
    return True


def can_request_background_nnindexer():
    return CURRENT_THREAD is None or not CURRENT_THREAD.is_alive()


def request_background_nnindexer(qreq_, daid_list):
    r""" FIXME: Duplicate code

    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        daid_list (list):

    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache --test-request_background_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_cache import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> daid_list = ibs.get_valid_aids(species=wbia.const.TEST_SPECIES.ZEB_PLAIN)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> # execute function
        >>> request_background_nnindexer(qreq_, daid_list)
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
    flann_params = qreq_.qparams.flann_params
    # Get annot descriptors to index
    vecs_list, fgws_list, fxs_list = get_support_data(qreq_, daid_list)
    # Dont hash rowids when given enough info in nnindex_cfgstr
    flann_params['cores'] = 2  # Only ues a few cores in the background
    # Build/Load the flann index
    uuid_map_fpath = get_nnindexer_uuid_map_fpath(qreq_)
    visual_uuid_list = qreq_.ibs.get_annot_visual_uuids(daid_list)

    # set temporary attribute for when the thread finishes
    finishtup = (uuid_map_fpath, daids_hashid, visual_uuid_list, min_reindex_thresh)
    CURRENT_THREAD = ut.spawn_background_process(
        background_flann_func,
        cachedir,
        daid_list,
        vecs_list,
        fgws_list,
        fxs_list,
        flann_params,
        cfgstr,
    )

    CURRENT_THREAD.finishtup = finishtup


def background_flann_func(
    cachedir,
    daid_list,
    vecs_list,
    fgws_list,
    fxs_list,
    flann_params,
    cfgstr,
    uuid_map_fpath,
    daids_hashid,
    visual_uuid_list,
    min_reindex_thresh,
):
    r""" FIXME: Duplicate code """
    print('[BG] Starting Background FLANN')
    # FIXME. dont use flann cache
    nnindexer = NeighborIndex(flann_params, cfgstr)
    # Initialize neighbor with unindexed data
    nnindexer.init_support(daid_list, vecs_list, fgws_list, fxs_list, verbose=True)
    # Load or build the indexing structure
    nnindexer.ensure_indexer(cachedir, verbose=True)
    if len(visual_uuid_list) > min_reindex_thresh:
        UUID_MAP_CACHE.write_uuid_map_dict(uuid_map_fpath, visual_uuid_list, daids_hashid)
    print('[BG] Finished Background FLANN')


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.neighbor_index_cache
        python -m wbia.algo.hots.neighbor_index_cache --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
