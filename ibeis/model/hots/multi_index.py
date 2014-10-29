"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.multi_index))"
python -m doctest -v ibeis/model/hots/multi_index.py
python -m doctest ibeis/model/hots/multi_index.py
"""
from __future__ import absolute_import, division, print_function
# Standard
import six
from six.moves import zip, map, range
#from itertools import chain
from ibeis import ibsfuncs
# Science
import numpy as np
# UTool
import utool
import ibeis.model.hots.neighbor_index as nbrx
# VTool
(print, print_, printDBG, rrr_, profile) = utool.inject(__name__, '[multi_index]', DEBUG=False)


def new_ibeis_mindexer(ibs, daid_list,
                       num_indexers=8,
                       split_method='name'):
    """
    Examples:
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mxer, qreq_, ibs = test_mindexer()
    """
    print('[mindex] make MultiNeighborIndex over %d annots' % (len(daid_list),))

    # Split annotations into groups accorindg to split_method
    if split_method == 'name':
        split_func = ibsfuncs.group_annots_by_known_names
        # each group are annotations of the same name
        aidgroup_list, invalid_aids = split_func(ibs, daid_list)
    else:
        raise AssertionError('unknown split_method=%r' % (split_method,))
    largest_groupsize = max(map(len, aidgroup_list))
    print('[mindex] num_indexers = %d ' % (num_indexers,))
    print('[mindex] largest_groupsize = %d ' % (largest_groupsize,))
    num_bins = min(largest_groupsize, num_indexers)
    print('[mindex] num_bins = %d ' % (num_bins,))
    #

    # Group annotations for indexing according to the split criteria
    aids_list, overflow_aids = utool.sample_zip(
        aidgroup_list, num_bins, allow_overflow=True, per_bin=1)

    if __debug__:
        # All groups have the same name
        nidgroup_list = ibsfuncs.unflat_map(ibs.get_annot_nids, aidgroup_list)
        for nidgroup in nidgroup_list:
            assert utool.list_allsame(nidgroup), 'bad name grouping'
    if __debug__:
        # All subsiquent indexer are subsets (in name/identity space)
        # of the previous
        nids_list = ibsfuncs.unflat_map(ibs.get_annot_nids, aids_list)
        prev_ = None
        for nids in nids_list:
            if prev_ is None:
                prev_ = set(nids)
            else:
                assert prev_.issuperset(nids), 'bad indexer grouping'

    # Build a neighbor indexer for each
    nn_indexer_list = []
    #extra_indexes = []
    for tx, aids in enumerate(aids_list):
        print('[mindex] building forest %d/%d with %d aids' %
                (tx + 1, num_bins, len(aids)))
        if len(aids) > 0:
            nnindexer = nbrx.new_ibeis_nnindexer(ibs, aids)
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
    from ibeis.model.hots.query_request import new_ibeis_query_request
    import ibeis
    ibs = ibeis.opendb(db='PZ_Mothers')
    daid_list = ibs.get_valid_aids()[1:60]
    qreq_ = new_ibeis_query_request(ibs, daid_list, daid_list)
    num_indexers = 4
    split_method = 'name'
    mxer = new_ibeis_mindexer(ibs, qreq_.get_internal_daids(),
                                  num_indexers, split_method)
    return mxer, qreq_, ibs


@six.add_metaclass(utool.ReloadingMetaclass)
class MultiNeighborIndex(object):
    """
    Generalization of a NeighborIndex
    More abstract wrapper around flann

    Example:
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mxer, qreq_, ibs = test_mindexer()
    """

    def __init__(mxer, nn_indexer_list):
        mxer.nn_indexer_list = nn_indexer_list  # List of single indexes

    def multi_knn(mxer, qfx2_vec, K, checks):
        """
        Example:
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K = 3
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_idx_list, qfx2_dist_list) = mxer.multi_knn(qfx2_vec, K)
        """
        qfx2_idx_list   = []
        qfx2_dist_list = []
        for nnindexer in mxer.nn_indexer_list:
            # Returns distances in ascending order for each query descriptor
            (_qfx2_idx, _qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
            qfx2_idx_list.append(_qfx2_idx)
            qfx2_dist_list.append(_qfx2_dist)
        return qfx2_idx_list, qfx2_dist_list

    def get_nIndexed_list(mxer):
        nIndexed_list = [nnindexer.num_indexed_vecs() for nnindexer in mxer.nn_indexer_list]
        return nIndexed_list

    def get_offsets(mxer):
        nIndexed_list = mxer.get_nIndexed_list()
        offset_list = np.cumsum(nIndexed_list)
        return offset_list

    def knn(mxer, qfx2_vec, K, checks):
        """
        Polymorphic interface to knn, but uses the multindex backend

        Example:
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K = 3
            >>> checks = 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K)
        """
        (qfx2_idx_list, qfx2_dist_list) = mxer.multi_knn(qfx2_vec, K, checks)
        qfx2_imx_list = []
        offset_list = mxer.get_offsets()
        for _qfx2_idx, offset in zip(qfx2_idx_list, offset_list):
            # Returns distances in ascending order for each query descriptor
            qfx2_imx_list.append(_qfx2_idx + offset)
        # Combine results from each tree
        qfx2_imx_   = np.hstack(qfx2_imx_list)
        qfx2_dist_  = np.hstack(qfx2_dist_list)
        # Sort over all tree result distances
        qfx2_sortx = qfx2_dist_.argsort(axis=1)
        # Apply sorting to concatenated results
        def sortaxis1(qfx2_xxx):
            return  np.vstack([row[sortx] for sortx, row
                               in zip(qfx2_sortx, qfx2_xxx)])
        qfx2_dist  = sortaxis1(qfx2_dist_)
        qfx2_imx   = sortaxis1(qfx2_imx_)
        return (qfx2_imx, qfx2_dist)

    def add_points(mxer, new_aid_list, new_vecs_list):
        raise NotImplementedError()

    def num_indexed_vecs(mxer):
        """
        Example:
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> out = mxer.num_indexed_vecs()
            >>> print(out)
            54141
        """
        return np.sum([nnindexer.num_indexed_vecs()
                       for nnindexer in mxer.nn_indexer_list])

    def num_indexed_annots(mxer):
        """
        Example:
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> out = mxer.num_indexed_annots()
            >>> print(out)
            53
        """
        return np.sum([nnindexer.num_indexed_annots()
                       for nnindexer in mxer.nn_indexer_list])

    def get_nn_aids(mxer, qfx2_imx):
        qfx2_aid = np.empty(qfx2_imx.shape, dtype=np.int32)
        nn_indexer_list = mxer.nn_indexer_list
        offset_list = mxer.get_offsets()
        prev = 0
        for nnindexer, offset in zip(nn_indexer_list, offset_list):
            mask = np.logical_and(qfx2_imx >= prev, qfx2_imx < offset)
            idxs = qfx2_imx[mask] - prev
            aids = nnindexer.get_nn_aids(idxs)
            qfx2_aid[mask] = aids
            prev = offset
        return qfx2_aid

    def get_nn_featxs(mxer, qfx2_imx):
        qfx2_fx = np.empty(qfx2_imx.shape, dtype=np.int32)
        nn_indexer_list = mxer.nn_indexer_list
        offset_list = mxer.get_offsets()
        prev = 0
        for nnindexer, offset in zip(nn_indexer_list, offset_list):
            mask = np.logical_and(qfx2_imx >= prev, qfx2_imx < offset)
            idxs = qfx2_imx[mask] - prev
            fxs = nnindexer.get_nn_featxs(idxs)
            qfx2_fx[mask] = fxs
            prev = offset
        return qfx2_fx

    def split_imxs_gen(mxer, qfx2_imx):
        """
        Example:
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
            >>> mxer, qreq_, ibs = test_mindexer()
            >>> K = 3
            >>> checks = 1028
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> (qfx2_imx, qfx2_dist) = mxer.knn(qfx2_vec, K, checks)
        """
        offset_list = mxer.get_offsets()
        prev = 0
        for offset in offset_list:
            mask = np.logical_and(qfx2_imx >= prev, qfx2_imx < offset)
            yield mask, prev
            prev = offset

    def knn2(mxer, qfx2_vec, K):
        """
        Example:
            >>> from ibeis.model.hots.neighbor_index import *  # NOQA
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
