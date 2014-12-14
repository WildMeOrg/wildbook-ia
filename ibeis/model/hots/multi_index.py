from __future__ import absolute_import, division, print_function
import six
from six.moves import zip, map, range
from ibeis import ibsfuncs
import numpy as np
import utool as ut
from ibeis.model.hots import neighbor_index
(print, print_, printDBG, rrr_, profile) = ut.inject(__name__, '[multi_index]', DEBUG=False)


@profile
def group_daids_by_cached_nnindexer(ibs, aid_list):
    r"""
    Args:
        ibs       (IBEISController):
        daid_list (list):

    CommandLine:
        python -m ibeis.model.hots.multi_index --test-group_daids_by_cached_nnindexer

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.multi_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(ibs, aid_list)
        >>> result = uncovered_aids, covered_aids_list
        >>> print(result)
    """
    annot_vuuid_list = ibs.get_annot_visual_uuids(aid_list)
    uuid_map_fpath = neighbor_index.get_nnindexer_uuid_map_fpath(ibs)
    # read which annotations have prebuilt caches
    with ut.shelf_open(uuid_map_fpath) as uuid_map:
        candidate_uuids = {key: set(val) for key, val in six.iteritems(uuid_map)}
    # find a maximum independent set cover
    tup = ut.greedy_max_inden_setcover(candidate_uuids, annot_vuuid_list)
    uncovered_vuuids, covered_vuuids_list = tup[0:2]
    # return the grouped covered items (so they can be loaded) and
    # the remaining uuids which need to have an index computed.
    uncovered_aids_ = ibs.get_annot_aids_from_visual_uuid(uncovered_vuuids)  # NOQA
    covered_aids_list_ = ibsfuncs.unflat_map(ibs.get_annot_aids_from_visual_uuid, covered_vuuids_list)
    uncovered_aids = sorted(uncovered_aids_)
    covered_aids_list = list(map(sorted, covered_aids_list_))
    return uncovered_aids, covered_aids_list


@profile
def group_daids_for_indexing_by_name(ibs, daid_list, num_indexers=8,
                                     verbose=True):
    """
    returns groups with only one annotation per name in each group
    """
    tup = ibs.group_annots_by_known_names(daid_list)
    aidgroup_list, invalid_aids = tup
    largest_groupsize = max(map(len, aidgroup_list))
    num_bins = min(largest_groupsize, num_indexers)
    if verbose:
        print('[mindex] num_indexers = %d ' % (num_indexers,))
        print('[mindex] largest_groupsize = %d ' % (largest_groupsize,))
        print('[mindex] num_bins = %d ' % (num_bins,))
    # Group annotations for indexing according to the split criteria
    aids_list, overflow_aids = ut.sample_zip(
        aidgroup_list, num_bins, allow_overflow=True, per_bin=1)
    if __debug__:
        # All groups have the same name
        nidgroup_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, aidgroup_list)
        for nidgroup in nidgroup_list:
            assert ut.list_allsame(nidgroup), 'bad name grouping'
    if __debug__:
        # All subsiquent indexer are subsets (in name/identity space)
        # of the previous
        nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, aids_list)
        prev_ = None
        for nids in nids_list:
            if prev_ is None:
                prev_ = set(nids)
            else:
                assert prev_.issuperset(nids), 'bad indexer grouping'
    return aids_list, overflow_aids, num_bins


@profile
def request_ibeis_mindexer(qreq_, index_method='multi', verbose=True):
    """

    Examples:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.multi_index import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_MTEST')
        >>> valid_aids = ibs.get_valid_aids()
        >>> daid_list = valid_aids[1:60]
        >>> qreq_ = ibs.new_query_request(daid_list, daid_list)
        >>> index_method = 'multi'
        >>> mxer = request_ibeis_mindexer(qreq_, index_method)
    """

    daid_list = qreq_.get_internal_daids()
    print('[mindex] make MultiNeighborIndex over %d annots' % (len(daid_list),))

    # Split annotations into groups accorindg to index_method
    ibs = qreq_.ibs
    if index_method == 'name':
        # each group are annotations of the same name
        num_indexers = 8
        aids_list, overflow_aids, num_bins = group_daids_for_indexing_by_name(ibs, daid_list, num_indexers, verbose)
    elif index_method == 'multi':
        uncovered_aids, covered_aids_list = group_daids_by_cached_nnindexer(ibs, daid_list)
        aids_list = covered_aids_list
        if len(uncovered_aids) > 0:
            aids_list.append(uncovered_aids)
        num_bins = len(aids_list)
    else:
        raise AssertionError('unknown index_method=%r' % (index_method,))

    # Build a neighbor indexer for each
    nn_indexer_list = []
    #extra_indexes = []
    for tx, aids in enumerate(aids_list):
        print('[mindex] building forest %d/%d with %d aids' %
                (tx + 1, num_bins, len(aids)))
        if len(aids) > 0:
            # Dont bother shallow copying qreq_ here.
            # just passing aids is enough
            nnindexer = neighbor_index.internal_request_ibeis_nnindexer(qreq_, aids)
            nn_indexer_list.append(nnindexer)
    #if len(unknown_aids) > 0:
    #    print('[mindex] building unknown forest')
    #    unknown_vecs_list = ibs.get_annot_vecs(overflow_aids)
    #    unknown_index = NeighborIndex(overflow_aids, unknown_vecs_list)
    #    extra_indexes.append(unknown_index)
    ##print('[mindex] building normalizer forest')  # TODO
    #mxer.nn_indexer_list = nn_indexer_list
    #mxer.extra_indexes = extra_indexes
    #mxer.overflow_index = overflow_index
    #mxer.unknown_index = unknown_index
    mxer = MultiNeighborIndex(nn_indexer_list)
    return mxer


def test_mindexer():
    import ibeis
    ibs = ibeis.opendb(db='PZ_MTEST')
    daid_list = ibs.get_valid_aids()[1:60]
    qreq_ = ibs.new_query_request(daid_list, daid_list)
    index_method = 'name'
    mxer = request_ibeis_mindexer(qreq_, index_method)
    return mxer, qreq_, ibs


#@profile
def sort_along_rows(qfx2_xxx, qfx2_sortx):
    """
    sorts each row in qfx2_xxx with the corresponding row in qfx2_sortx
    """
    if qfx2_xxx.size == 0:
        return qfx2_xxx
    #return np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])
    return np.vstack([row.take(sortx) for sortx, row in zip(qfx2_sortx, qfx2_xxx)])


@six.add_metaclass(ut.ReloadingMetaclass)
class MultiNeighborIndex(object):
    """
    Generalization of a NeighborIndex
    More abstract wrapper around flann

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.multi_index import *  # NOQA
        >>> mxer, qreq_, ibs = test_mindexer()
    """

    def __init__(mxer, nn_indexer_list):
        mxer.nn_indexer_list = nn_indexer_list  # List of single indexes

    def get_dtype(mxer):
        return mxer.nn_indexer_list[0].get_dtype()

    #@profile
    def multi_knn(mxer, qfx2_vec, K, checks):
        """
        Does a query on each of the subindexer kdtrees
        returns list of the results

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1024
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_idx_list, qfx2_dist_list) = mxer.multi_knn(qfx2_vec, K, checks)
            >>> result = str(list(map(np.shape, qfx2_idx_list)))
            >>> print(result)
            [(1074, 3), (1074, 3), (1074, 3), (1074, 3), (1074, 3), (1074, 3)]
        """
        qfx2_idx_list  = []
        qfx2_dist_list = []
        for nnindexer in mxer.nn_indexer_list:
            # Returns distances in ascending order for each query descriptor
            (_qfx2_idx, _qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
            qfx2_idx_list.append(_qfx2_idx)
            qfx2_dist_list.append(_qfx2_dist)
        return qfx2_idx_list, qfx2_dist_list

    @profile
    def knn(mxer, qfx2_vec, K, checks):
        """
        Polymorphic interface to knn, but uses the multindex backend

        CommandLine:
            python -m ibeis.model.hots.multi_index --test-knn:0

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
            >>> print(qfx2_imx.shape)
            >>> assert qfx2_imx.shape[1] == 18
            >>> ut.assert_inbounds(qfx2_imx.shape[0], 1073, 1079)

        Example2:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1028
            >>> qfx2_vec = np.empty((0, 128), dtype=mxer.get_dtype())
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
            >>> result = str(np.shape(qfx2_imx))
            >>> print(result)
            (0, 18)
        """
        (qfx2_idx_list, qfx2_dist_list) = mxer.multi_knn(qfx2_vec, K, checks)
        qfx2_imx_list = []
        offset_list = mxer.get_offsets()
        prev = 0
        for _qfx2_idx, offset in zip(qfx2_idx_list, offset_list):
            # Returns distances in ascending order for each query descriptor
            qfx2_imx_list.append(_qfx2_idx + prev)
            prev = offset
        # Combine results from each tree
        qfx2_imx_   = np.hstack(qfx2_imx_list)
        qfx2_dist_  = np.hstack(qfx2_dist_list)
        # Sort over all tree result distances
        qfx2_sortx = qfx2_dist_.argsort(axis=1)
        # Apply sorting to concatenated results
        qfx2_dist  = sort_along_rows(qfx2_dist_, qfx2_sortx)
        qfx2_imx   = sort_along_rows(qfx2_imx_, qfx2_sortx)
        return (qfx2_imx, qfx2_dist)

    def get_offsets(mxer):
        r"""
        Returns:
            list:

        CommandLine:
            python -m ibeis.model.hots.multi_index --test-get_offsets

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> result = mxer.get_offsets()
            >>> print(result)
            [21384 36627 49435 54244 57786 60482]
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
            python -m ibeis.model.hots.multi_index --test-get_nIndexed_list

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> result = mxer.get_nIndexed_list()
            >>> print(result)
            [21384, 15243, 12808, 4809, 3542, 2696]
        """
        nIndexed_list = [nnindexer.num_indexed_vecs()
                         for nnindexer in mxer.nn_indexer_list]
        return nIndexed_list

    def add_points(mxer, new_aid_list, new_vecs_list, new_fgws_list):
        raise NotImplementedError()

    def num_indexed_vecs(mxer):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> result = mxer.num_indexed_vecs()
            >>> print(result)
            60482

        54244
        54200 on win32
        """
        return np.sum([nnindexer.num_indexed_vecs()
                       for nnindexer in mxer.nn_indexer_list])

    def num_indexed_annots(mxer):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> result = mxer.num_indexed_annots()
            >>> print(result)
            59
        """
        return np.sum([nnindexer.num_indexed_annots()
                       for nnindexer in mxer.nn_indexer_list])

    def iter_subindexers(mxer, qfx2_imx):
        """
        generates subindexers, indicies, and maskss within them
        that partially correspond to indicies in qfx2_imx that belong
        to that subindexer

        Args:
            qfx2_imx (ndarray):

        CommandLine:
            python -m ibeis.model.hots.multi_index --test-iter_subindexers

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
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

    #def get_nn_featxs(mxer, qfx2_imx):
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
            python -m ibeis.model.hots.multi_index --test-get_nn_aids

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
            >>> qfx2_aid = mxer.get_nn_aids(qfx2_imx)
            >>> result = np.array_str(qfx2_aid[0:2])
            >>> print(result)
            [[ 8 16  8 55 53  3  3 28 38 34 48 38 34 25 21 21 33 34]
             [34 45 30 27 18 35 28 52  8  4 60 25 14 25 21 26 15 15]]
        """
        #qfx2_aid = -np.ones(qfx2_imx.shape, dtype=np.int32)
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
            python -m ibeis.model.hots.multi_index --test-get_nn_featxs

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
            >>> qfx2_fgw = mxer.get_nn_featxs(qfx2_imx)
            >>> result = np.array_str(qfx2_fgw)
            >>> print(result)
        """
        #qfx2_fx = -np.ones(qfx2_imx.shape, dtype=np.int32)
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
            python -m ibeis.model.hots.multi_index --test-get_nn_fgws

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> import numpy as np
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K, checks = 3, 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
            >>> qfx2_fgw = mxer.get_nn_fgws(qfx2_imx)
            >>> result = np.array_str(qfx2_fgw)
            >>> print(result)
        """
        #qfx2_fgw = -np.ones(qfx2_imx.shape, dtype=np.float32)
        qfx2_fgw = np.empty(qfx2_imx.shape, dtype=np.float32)
        for nnindexer, idxs, mask in mxer.iter_subindexers(qfx2_imx):
            qfx2_fgw[mask] = nnindexer.get_nn_fgws(idxs)
        return qfx2_fgw

    def knn2(mxer, qfx2_vec, K):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.multi_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_dist_, qfx2_idx_,  qfx2_fx_, qfx2_ax_, qfx2_rankx_, qfx2_treex_,) = mxer.knn2(qfx2_vec, K)
        """
        qfx2_idx_list   = []
        qfx2_dist_list = []
        qfx2_ax_list  = []
        qfx2_fx_list   = []
        qfx2_rankx_list = []  # ranks index
        qfx2_treex_list = []  # tree index
        for tx, nnindexer in enumerate(mxer.nn_indexer_list):
            # Returns distances in ascending order for each query descriptor
            (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=1024)
            qfx2_fx = nnindexer.get_nn_featxs(qfx2_idx)
            qfx2_ax = nnindexer.get_nn_axs(qfx2_idx)
            qfx2_idx_list.append(qfx2_idx)
            qfx2_dist_list.append(qfx2_dist)
            qfx2_fx_list.append(qfx2_fx)
            qfx2_ax_list.append(qfx2_ax)
            qfx2_rankx_list.append(np.array([[rankx for rankx in range(qfx2_idx.shape[1])]] * len(qfx2_idx)))
            qfx2_treex_list.append(np.array([[tx for rankx in range(qfx2_idx.shape[1])]] * len(qfx2_idx)))
        # Combine results from each tree
        qfx2_idx   = np.hstack(qfx2_idx_list)
        qfx2_dist  = np.hstack(qfx2_dist_list)
        qfx2_rankx = np.hstack(qfx2_rankx_list)
        qfx2_treex = np.hstack(qfx2_treex_list)
        qfx2_ax    = np.hstack(qfx2_ax_list)
        qfx2_fx    = np.hstack(qfx2_fx_list)

        # Sort over all tree result distances
        qfx2_sortx = qfx2_dist.argsort(axis=1)
        # Apply sorting to concatenated results
        def foreach_row_sort_cols(qfx2_xxx):
            return  np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])
        qfx2_dist_  = foreach_row_sort_cols(qfx2_dist)
        qfx2_idx_   = foreach_row_sort_cols(qfx2_idx)
        qfx2_ax_    = foreach_row_sort_cols(qfx2_ax)
        qfx2_fx_    = foreach_row_sort_cols(qfx2_fx)
        qfx2_rankx_ = foreach_row_sort_cols(qfx2_rankx)
        qfx2_treex_ = foreach_row_sort_cols(qfx2_treex)
        return (qfx2_dist_, qfx2_idx_,  qfx2_fx_, qfx2_ax_, qfx2_rankx_, qfx2_treex_,)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.multi_index
        python -m ibeis.model.hots.multi_index --allexamples
        python -m ibeis.model.hots.multi_index --allexamples --noface --nosrc

        profiler.sh ibeis/model/hots/multi_index.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
