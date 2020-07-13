# -*- coding: utf-8 -*-
"""
DEPRICATE

module which uses multiple flann indexes as a way of working around adding
points to a single flann structure which seems to cause crashes.
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import zip, map, range
import numpy as np
import utool as ut
import vtool as vt
from wbia.algo.hots import neighbor_index_cache
from wbia.algo.hots import hstypes

(print, rrr, profile) = ut.inject2(__name__)


USE_FORGROUND_REINDEX = ut.get_argflag(('--use-foreground-reindex', '--fg-reindex'))


def testdata_mindexer():
    import wbia

    ibs = wbia.opendb(db='PZ_MTEST')
    daid_list = ibs.get_valid_aids()[1:60]
    cfgdict = dict(fg_on=False)
    qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=cfgdict)
    index_method = 'name'
    mxer = request_wbia_mindexer(qreq_, index_method)
    return mxer, qreq_, ibs


@profile
def group_daids_for_indexing_by_name(ibs, daid_list, num_indexers=8, verbose=True):
    """
    returns groups with only one annotation per name in each group
    """
    tup = ibs.group_annots_by_known_names(daid_list)
    aidgroup_list, invalid_aids = tup
    largest_groupsize = max(map(len, aidgroup_list))
    num_bins = min(largest_groupsize, num_indexers)
    if verbose or ut.VERYVERBOSE:
        print('[mindex] num_indexers = %d ' % (num_indexers,))
        print('[mindex] largest_groupsize = %d ' % (largest_groupsize,))
        print('[mindex] num_bins = %d ' % (num_bins,))
    # Group annotations for indexing according to the split criteria
    aids_list, overflow_aids = ut.sample_zip(
        aidgroup_list, num_bins, allow_overflow=True, per_bin=1
    )
    if __debug__:
        # All groups have the same name
        nidgroup_list = ibs.unflat_map(ibs.get_annot_name_rowids, aidgroup_list)
        for nidgroup in nidgroup_list:
            assert ut.allsame(nidgroup), 'bad name grouping'
    if __debug__:
        # All subsiquent indexer are subsets (in name/identity space)
        # of the previous
        nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
        prev_ = None
        for nids in nids_list:
            if prev_ is None:
                prev_ = set(nids)
            else:
                assert prev_.issuperset(nids), 'bad indexer grouping'
    return aids_list, overflow_aids, num_bins


@profile
def request_wbia_mindexer(qreq_, index_method='multi', verbose=True):
    """

    CommandLine:
        python -m wbia.algo.hots.multi_index --test-request_wbia_mindexer:2

    Example0:
        >>> # SLOW_DOCTEST
        >>> from wbia.algo.hots.multi_index import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(db='PZ_MTEST')
        >>> valid_aids = ibs.get_valid_aids()
        >>> daid_list = valid_aids[1:60]
        >>> cfgdict = dict(fg_on=False)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=cfgdict)
        >>> index_method = 'multi'
        >>> mxer = request_wbia_mindexer(qreq_, index_method)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.multi_index import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(db='PZ_Master0')
        >>> valid_aids = ibs.get_valid_aids()
        >>> daid_list = valid_aids[1:60]
        >>> cfgdict = dict(fg_on=False)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=cfgdict)
        >>> index_method = 'multi'
        >>> mxer = request_wbia_mindexer(qreq_, index_method)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> # Test background reindex
        >>> from wbia.algo.hots.multi_index import *  # NOQA
        >>> import wbia
        >>> import time
        >>> ibs = wbia.opendb(db='PZ_MTEST')
        >>> valid_aids = ibs.get_valid_aids()
        >>> # Remove all cached nnindexers
        >>> ibs.delete_flann_cachedir()
        >>> # This request should build a new nnindexer
        >>> daid_list = valid_aids[1:30]
        >>> cfgdict = dict(fg_on=False)
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=cfgdict)
        >>> index_method = 'multi'
        >>> mxer = request_wbia_mindexer(qreq_, index_method)
        >>> ut.assert_eq(len(mxer.nn_indexer_list), 1, 'one subindexer')
        >>> # The next request should trigger a background process
        >>> # and build two subindexer
        >>> daid_list = valid_aids[1:60]
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=cfgdict)
        >>> index_method = 'multi'
        >>> mxer = request_wbia_mindexer(qreq_, index_method)
        >>> # Do some work in the foreground to ensure that it doesnt block
        >>> # the background job
        >>> print('[FG] sleeping or doing bit compute')
        >>> # Takes about 15 seconds
        >>> with ut.Timer():
        ...     ut.enumerate_primes(int(9E4))
        >>> #time.sleep(10)
        >>> print('[FG] done sleeping')
        >>> ut.assert_eq(len(mxer.nn_indexer_list), 2, 'two subindexer')
        >>> # And this shoud build just one subindexer
        >>> daid_list = valid_aids[1:60]
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list, cfgdict=cfgdict)
        >>> index_method = 'multi'
        >>> mxer = request_wbia_mindexer(qreq_, index_method)
        >>> ut.assert_eq(len(mxer.nn_indexer_list), 1, 'one big subindexer')

    """
    min_reindex_thresh = qreq_.qparams.min_reindex_thresh
    max_subindexers = qreq_.qparams.max_subindexers

    daid_list = qreq_.get_internal_daids()
    print('[mindex] make MultiNeighborIndex over %d annots' % (len(daid_list),))
    print('[mindex] index_method=%r' % index_method)

    # Split annotations into groups accorindg to index_method
    ibs = qreq_.ibs
    if index_method == 'name':
        # each group are annotations of the same name
        num_indexers = 8
        aids_list, overflow_aids, num_bins = group_daids_for_indexing_by_name(
            ibs, daid_list, num_indexers, verbose
        )
    elif index_method == 'multi':
        neighbor_index_cache.check_background_process()
        # Use greedy set cover to get a list of nnindxers that are already built
        tup = neighbor_index_cache.group_daids_by_cached_nnindexer(
            qreq_, daid_list, min_reindex_thresh
        )
        uncovered_aids, covered_aids_list = tup
        # If the number of bins gets too big do a reindex
        # in the background
        num_subindexers = len(covered_aids_list) + (len(uncovered_aids) > 1)
        if num_subindexers > max_subindexers:
            print('need to reindex something')
            if USE_FORGROUND_REINDEX:
                aids_list = [sorted(ut.flatten(covered_aids_list))]
                # ut.embed()
            else:
                neighbor_index_cache.request_background_nnindexer(qreq_, daid_list)
                aids_list = covered_aids_list
        else:
            aids_list = covered_aids_list
        if len(uncovered_aids) > 0:
            aids_list.append(uncovered_aids)
        num_bins = len(aids_list)
    else:
        raise AssertionError('unknown index_method=%r' % (index_method,))

    # Build a neighbor indexer for each
    nn_indexer_list = []
    # extra_indexes = []
    for tx, aids in enumerate(aids_list):
        print(
            '[mindex] building forest %d/%d with %d aids' % (tx + 1, num_bins, len(aids))
        )
        if len(aids) > 0:
            # Dont bother shallow copying qreq_ here.
            # just passing aids is enough
            nnindexer = neighbor_index_cache.request_memcached_wbia_nnindexer(qreq_, aids)
            nn_indexer_list.append(nnindexer)
    # if len(unknown_aids) > 0:
    #    print('[mindex] building unknown forest')
    #    unknown_vecs_list = ibs.get_annot_vecs(overflow_aids, config2_=qreq_.get_internal_data_config2())
    #    unknown_index = NeighborIndex(overflow_aids, unknown_vecs_list)
    #    extra_indexes.append(unknown_index)
    # # print('[mindex] building normalizer forest')  # TODO
    # mxer.nn_indexer_list = nn_indexer_list
    # mxer.extra_indexes = extra_indexes
    # mxer.overflow_index = overflow_index
    # mxer.unknown_index = unknown_index
    mxer = MultiNeighborIndex(nn_indexer_list, min_reindex_thresh, max_subindexers)
    return mxer


# @profile
def sort_along_rows(qfx2_xxx, qfx2_sortx):
    """
    sorts each row in qfx2_xxx with the corresponding row in qfx2_sortx
    """
    if qfx2_xxx.size == 0:
        return qfx2_xxx
    # return np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])
    return np.vstack([row.take(sortx) for sortx, row in zip(qfx2_sortx, qfx2_xxx)])


@six.add_metaclass(ut.ReloadingMetaclass)
class MultiNeighborIndex(object):
    """
    TODO: rename to DistributedNeighborIndex

    Generalization of a NeighborIndex
    More abstract wrapper around flann

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.algo.hots.multi_index import *  # NOQA
        >>> mxer, qreq_, ibs = testdata_mindexer()
    """

    def __init__(mxer, nn_indexer_list, min_reindex_thresh=10, max_subindexers=2):
        mxer.nn_indexer_list = nn_indexer_list  # List of single indexes
        # Parameters for adding support to multi_indexer
        mxer.min_reindex_thresh = min_reindex_thresh
        mxer.max_subindexers = max_subindexers

    def get_dtype(mxer):
        return mxer.nn_indexer_list[0].get_dtype()

    # @profile
    def multi_knn(mxer, qfx2_vec, K):
        """
        Does a query on each of the subindexer kdtrees
        returns list of the results

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_idx_list, qfx2_dist_list) = mxer.multi_knn(qfx2_vec, K)
            >>> shape_list = list(map(np.shape, qfx2_idx_list))
            >>> d1_list = ut.get_list_column(shape_list, 0)
            >>> d2_list = ut.get_list_column(shape_list, 1)
            >>> ut.assert_eq(d2_list, [3] * 6)
            >>> ut.assert_eq(d1_list, [len(qfx2_vec)] * 6)
        """
        qfx2_idx_list = []
        qfx2_dist_list = []
        for nnindexer in mxer.nn_indexer_list:
            # Returns distances in ascending order for each query descriptor
            (_qfx2_idx, _qfx2_dist) = nnindexer.knn(qfx2_vec, K=K)
            qfx2_idx_list.append(_qfx2_idx)
            qfx2_dist_list.append(_qfx2_dist)
        return qfx2_idx_list, qfx2_dist_list

    @profile
    def knn(mxer, qfx2_vec, K):
        """
        Polymorphic interface to knn, but uses the multindex backend

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-knn:0

        Example1:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
            >>> print(qfx2_imx.shape)
            >>> assert qfx2_imx.shape[1] == 18
            >>> ut.assert_inbounds(qfx2_imx.shape[0], 1073, 1079)

        Example2:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qfx2_vec = np.empty((0, 128), dtype=mxer.get_dtype())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
            >>> result = str(np.shape(qfx2_imx))
            >>> print(result)
            (0, 18)
        """
        (qfx2_idx_list, qfx2_dist_list) = mxer.multi_knn(qfx2_vec, K)
        qfx2_imx_list = []
        offset_list = mxer.get_offsets()
        prev = 0
        for _qfx2_idx, offset in zip(qfx2_idx_list, offset_list):
            # Returns distances in ascending order for each query descriptor
            qfx2_imx_list.append(_qfx2_idx + prev)
            prev = offset
        # Combine results from each tree
        qfx2_imx_ = np.hstack(qfx2_imx_list)
        qfx2_dist_ = np.hstack(qfx2_dist_list)
        # Sort over all tree result distances
        qfx2_sortx = qfx2_dist_.argsort(axis=1)
        # Apply sorting to concatenated results
        qfx2_dist = sort_along_rows(qfx2_dist_, qfx2_sortx)
        qfx2_imx = sort_along_rows(qfx2_imx_, qfx2_sortx)
        return (qfx2_imx, qfx2_dist)

    def get_offsets(mxer):
        r"""
        Returns:
            list:

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-get_offsets

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> offset_list = mxer.get_offsets()
            >>> #target = np.array([15257, 12769,  4819,  3542,  2694])
            >>> target = np.array([21384, 36627, 49435, 54244, 57786, 60482])
            >>> error = ut.assert_almost_eq(offset_list, target, 100)
            >>> print('error.max() = %r' % (error.max(),))
        """
        nIndexed_list = mxer.get_nIndexed_list()
        offset_list = np.cumsum(nIndexed_list)
        return offset_list

    def get_nIndexed_list(mxer):
        """
        returns a list of the number of indexed vectors in each subindexer

        Args:

        Returns:
            list : nIndexed_list

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-get_nIndexed_list

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> nIndexed_list = mxer.get_nIndexed_list()
            >>> target = np.array([21384, 15243, 12808, 4809, 3542, 2696])
            >>> error = ut.assert_almost_eq(nIndexed_list, target, 100)
            >>> print('error.max() = %r' % (error.max(),))
            >>> #np.all(ut.inbounds(nIndexed_list, low, high))
        """
        nIndexed_list = [
            nnindexer.num_indexed_vecs() for nnindexer in mxer.nn_indexer_list
        ]
        return nIndexed_list

    def get_multi_indexed_aids(mxer):
        index_aids_list = np.array(
            [nnindexer.get_indexed_aids() for nnindexer in mxer.nn_indexer_list]
        )
        return index_aids_list

    def get_indexed_aids(mxer):
        return ut.flatten(mxer.get_multi_indexed_aids())

    def get_multi_num_indexed_annots(mxer):
        num_indexed_list = np.array(
            [nnindexer.num_indexed_annots() for nnindexer in mxer.nn_indexer_list]
        )
        return num_indexed_list

    def assert_can_add_aids(mxer, new_aid_list):
        """
        Aids that are already indexed should never be added.

        Ignore:
            qreq_vsmany_ = ut.search_stack_for_localvar('qreq_vsmany_')
            qreq_vsmany_.daids
            qreq_vsmany_.qaids
            qreq_vsmany_.ibs.get_annot_exemplar_flags(new_aid_list)
        """
        indexed_aids_list = mxer.get_multi_indexed_aids()
        indexed_aids = np.hstack(indexed_aids_list)
        uncovered_mask = vt.get_uncovered_mask(indexed_aids, new_aid_list)
        if not np.all(uncovered_mask):
            msg_list = [
                'new aids must be disjoint from current aids',
                'new_aid_list = %r' % (new_aid_list,),
            ]
            msg = '\n'.join(msg_list)
            raise AssertionError(msg)

    def add_support(mxer, new_aid_list, new_vecs_list, new_fgws_list, verbose=True):
        """
        Chooses indexer with smallest number of annotations and reindexes it.
        """
        print('adding multi-indexer support')
        raise NotImplementedError()

    def add_wbia_support(mxer, qreq_, new_aid_list):
        """
        Chooses indexer with smallest number of annotations and reindexes it.

        Args:
            qreq_ (QueryRequest):  query request object with hyper-parameters
            new_aid_list (list):

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-add_wbia_support

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> new_aid_list = ibs.get_valid_aids()[70:80]
            >>> # execute function
            >>> result = mxer.add_wbia_support(qreq_, new_aid_list)
            >>> # verify results
            >>> print(result)
        """
        print('adding multi-indexer support')
        # Assert that the aids are indeed new
        mxer.assert_can_add_aids(new_aid_list)
        # Find the indexer to add to
        num_indexed_list = mxer.get_multi_num_indexed_annots()
        min_argx = num_indexed_list.argmin()
        nnindexer_old = mxer.nn_indexer_list[min_argx]
        # Combine old and new aids
        prev_aids = nnindexer_old.get_indexed_aids()
        new_aid_list_ = np.append(prev_aids, new_aid_list)
        # Reindexed combined aids
        nnindexer_new = neighbor_index_cache.request_memcached_wbia_nnindexer(
            qreq_, new_aid_list_
        )
        # Replace the old nnindexer with the new nnindexer
        mxer.nn_indexer_list[min_argx] = nnindexer_new
        mxer.min_reindex_thresh = qreq_.qparams.min_reindex_thresh

        if neighbor_index_cache.can_request_background_nnindexer():
            # Check if background process needs to be spawned
            # FIXME: this does not belong in method code
            num_indexed_list_new = mxer.get_multi_num_indexed_annots()
            new_smalled_size = min(num_indexed_list_new)
            need_reindex = (
                new_smalled_size > mxer.min_reindex_thresh
                or len(num_indexed_list_new) > mxer.max_subindexers
            )
            if need_reindex:
                if USE_FORGROUND_REINDEX:
                    raise NotImplementedError('no foreground reindex in stateful query')
                else:
                    # Reindex the multi-indexed trees in the background
                    aid_list = mxer.get_indexed_aids()
                    neighbor_index_cache.request_background_nnindexer(qreq_, aid_list)

    def num_indexed_vecs(mxer):
        """
        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> num_indexed = mxer.num_indexed_vecs()
            >>> ut.assert_inbounds(num_indexed, 60300, 60500)
        """
        return np.sum(
            [nnindexer.num_indexed_vecs() for nnindexer in mxer.nn_indexer_list]
        )

    def num_indexed_annots(mxer):
        """
        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> result = mxer.num_indexed_annots()
            >>> print(result)
            59
        """
        return np.sum(
            [nnindexer.num_indexed_annots() for nnindexer in mxer.nn_indexer_list]
        )

    def iter_subindexers(mxer, qfx2_imx):
        """
        generates subindexers, indices, and maskss within them
        that partially correspond to indices in qfx2_imx that belong
        to that subindexer

        Args:
            qfx2_imx (ndarray):

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-iter_subindexers

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3, 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
            >>> genlist_ = list(mxer.iter_subindexers(qfx2_imx))
            >>> covered = np.zeros(qfx2_imx.shape)
            >>> for nnindexer, idxs, mask in genlist_:
            ...     print(covered.sum())
            ...     assert idxs.size == mask.sum()
            ...     assert covered[mask].sum() == 0
            ...     covered[mask] = True
            >>> print(covered.sum())
            >>> assert covered.sum() == covered.size
        """
        nn_indexer_list = mxer.nn_indexer_list
        offset_list = mxer.get_offsets()
        prev = 0
        for nnindexer, offset in zip(nn_indexer_list, offset_list):
            mask = np.logical_and(qfx2_imx >= prev, qfx2_imx < offset)
            idxs = qfx2_imx[mask] - prev
            yield nnindexer, idxs, mask
            prev = offset

    # def get_nn_featxs(mxer, qfx2_imx):
    #    qfx2_fx = np.empty(qfx2_imx.shape, dtype=np.int32)
    #    nn_indexer_list = mxer.nn_indexer_list
    #    offset_list = mxer.get_offsets()
    #    prev = 0
    #    for nnindexer, offset in zip(nn_indexer_list, offset_list):
    #        mask = np.logical_and(qfx2_imx >= prev, qfx2_imx < offset)
    #        idxs = qfx2_imx[mask] - prev
    #        fxs = nnindexer.get_nn_featxs(idxs)
    #        qfx2_fx[mask] = fxs
    #        prev = offset
    #    return qfx2_fx

    def get_nn_aids(mxer, qfx2_imx):
        r"""
        Args:
            qfx2_imx (ndarray):

        Returns:
            ndarray: qfx2_aid

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-get_nn_aids

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qaid = 1
            >>> qfx2_vec = ibs.get_annot_vecs(qaid, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
            >>> qfx2_aid = mxer.get_nn_aids(qfx2_imx)
            >>> gt_aids = ibs.get_annot_groundtruth(qaid)
            >>> result = np.array_str(qfx2_aid[0:2])
            >>> # Make sure there are lots (like 5%) of correct matches
            >>> mask_cover = vt.get_covered_mask(qfx2_aid, gt_aids)
            >>> num_correct   = mask_cover.sum()
            >>> num_incorrect = (~mask_cover).sum()
            >>> print('fraction correct = %r' % (num_correct / float(num_incorrect),))
            >>> ut.assert_inbounds(num_correct, 900, 1100,
            ...                    'not enough matches to groundtruth')
        """
        # qfx2_aid = -np.ones(qfx2_imx.shape, dtype=np.int32)
        qfx2_aid = np.empty(qfx2_imx.shape, dtype=np.int32)
        for nnindexer, idxs, mask in mxer.iter_subindexers(qfx2_imx):
            qfx2_aid[mask] = nnindexer.get_nn_aids(idxs)
        return qfx2_aid

    def get_nn_featxs(mxer, qfx2_imx):
        r"""
        Args:
            qfx2_imx (ndarray):

        Returns:
            ndarray: qfx2_fx

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-get_nn_featxs

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
            >>> qfx2_fgw = mxer.get_nn_featxs(qfx2_imx)
            >>> result = np.array_str(qfx2_fgw)
            >>> print(result)
        """
        # qfx2_fx = -np.ones(qfx2_imx.shape, dtype=np.int32)
        qfx2_fx = np.empty(qfx2_imx.shape, dtype=np.int32)
        for nnindexer, idxs, mask in mxer.iter_subindexers(qfx2_imx):
            qfx2_fx[mask] = nnindexer.get_nn_featxs(idxs)
        return qfx2_fx

    def get_nn_fgws(mxer, qfx2_imx):
        r"""
        Args:
            qfx2_imx (ndarray):

        Returns:
            ndarray: qfx2_fgw

        CommandLine:
            python -m wbia.algo.hots.multi_index --test-get_nn_fgws

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
            >>> qfx2_fgw = mxer.get_nn_fgws(qfx2_imx)
            >>> result = np.array_str(qfx2_fgw)
            >>> print(result)
        """
        # qfx2_fgw = -np.ones(qfx2_imx.shape, dtype=np.float32)
        qfx2_fgw = np.empty(qfx2_imx.shape, dtype=hstypes.FS_DTYPE)
        for nnindexer, idxs, mask in mxer.iter_subindexers(qfx2_imx):
            qfx2_fgw[mask] = nnindexer.get_nn_fgws(idxs)
        return qfx2_fgw

    def knn2(mxer, qfx2_vec, K):
        """
        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.algo.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = testdata_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1, config2_=qreq_.get_internal_query_config2())
            >>> (qfx2_dist_, qfx2_idx_,  qfx2_fx_, qfx2_ax_, qfx2_rankx_, qfx2_treex_,) = mxer.knn2(qfx2_vec, K)
        """
        qfx2_idx_list = []
        qfx2_dist_list = []
        qfx2_ax_list = []
        qfx2_fx_list = []
        qfx2_rankx_list = []  # ranks index
        qfx2_treex_list = []  # tree index
        for tx, nnindexer in enumerate(mxer.nn_indexer_list):
            # Returns distances in ascending order for each query descriptor
            (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K)
            qfx2_fx = nnindexer.get_nn_featxs(qfx2_idx)
            qfx2_ax = nnindexer.get_nn_axs(qfx2_idx)
            qfx2_idx_list.append(qfx2_idx)
            qfx2_dist_list.append(qfx2_dist)
            qfx2_fx_list.append(qfx2_fx)
            qfx2_ax_list.append(qfx2_ax)
            qfx2_rankx_list.append(
                np.array([[rankx for rankx in range(qfx2_idx.shape[1])]] * len(qfx2_idx))
            )
            qfx2_treex_list.append(
                np.array([[tx for rankx in range(qfx2_idx.shape[1])]] * len(qfx2_idx))
            )
        # Combine results from each tree
        qfx2_idx = np.hstack(qfx2_idx_list)
        qfx2_dist = np.hstack(qfx2_dist_list)
        qfx2_rankx = np.hstack(qfx2_rankx_list)
        qfx2_treex = np.hstack(qfx2_treex_list)
        qfx2_ax = np.hstack(qfx2_ax_list)
        qfx2_fx = np.hstack(qfx2_fx_list)

        # Sort over all tree result distances
        qfx2_sortx = qfx2_dist.argsort(axis=1)
        # Apply sorting to concatenated results

        def foreach_row_sort_cols(qfx2_xxx):
            return np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])

        qfx2_dist_ = foreach_row_sort_cols(qfx2_dist)
        qfx2_idx_ = foreach_row_sort_cols(qfx2_idx)
        qfx2_ax_ = foreach_row_sort_cols(qfx2_ax)
        qfx2_fx_ = foreach_row_sort_cols(qfx2_fx)
        qfx2_rankx_ = foreach_row_sort_cols(qfx2_rankx)
        qfx2_treex_ = foreach_row_sort_cols(qfx2_treex)
        return (
            qfx2_dist_,
            qfx2_idx_,
            qfx2_fx_,
            qfx2_ax_,
            qfx2_rankx_,
            qfx2_treex_,
        )


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.multi_index
        python -m wbia.algo.hots.multi_index --allexamples
        python -m wbia.algo.hots.multi_index --allexamples --noface --nosrc

        utprof.sh wbia/algo/hots/multi_index.py --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
