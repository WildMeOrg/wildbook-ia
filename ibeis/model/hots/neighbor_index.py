"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.neighbor_index))"
python -m doctest -v ibeis/model/hots/neighbor_index.py
python -m doctest ibeis/model/hots/neighbor_index.py
"""
from __future__ import absolute_import, division, print_function
# Standard
from six.moves import zip, map, range
#from itertools import chain
import sys
# Science
import numpy as np
# UTool
import utool
# VTool
from ibeis import ibsfuncs
import atexit
import vtool.nearest_neighbors as nntool
(print, print_, printDBG, rrr_, profile) = utool.inject(__name__, '[neighbor_index]', DEBUG=False)

NOCACHE_FLANN = '--nocache-flann' in sys.argv


# cache for heavyweight nn structures.
# ensures that only one is in memory
NEIGHBOR_CACHE = {}
MAX_NEIGHBOR_CACHE_SIZE = 32


def rrr():
    global NEIGHBOR_CACHE
    NEIGHBOR_CACHE.clear()
    rrr_()


@atexit.register
def __cleanup():
    """ prevents flann errors (not for cleaning up individual objects) """
    global NEIGHBOR_CACHE
    NEIGHBOR_CACHE.clear()
    try:
        del NEIGHBOR_CACHE
    except NameError:
        pass


def new_neighbor_indexer(aid_list=[], vecs_list=[], flann_params={},
                         flann_cachedir=None, indexer_cfgstr='',
                         hash_rowids=True, use_cache=not NOCACHE_FLANN,
                         use_params_hash=True):
    print('[nnindexer] building NeighborIndex object')
    _check_input(aid_list, vecs_list)
    # Create indexes into the input aids
    ax_list = np.arange(len(aid_list))
    idx2_vec, idx2_ax, idx2_fx = invert_index(vecs_list, ax_list)
    ax2_aid   = np.array(aid_list)
    if hash_rowids:
        # Fingerprint
        aids_hashstr = utool.hashstr_arr(aid_list, '_AIDS')
        cfgstr = aids_hashstr + indexer_cfgstr
    else:
        # Dont hash rowids when given enough info in indexer_cfgstr
        cfgstr = indexer_cfgstr
    # Build/Load the flann index
    flann = nntool.flann_cache(idx2_vec, **{
        'cache_dir': flann_cachedir,
        'cfgstr': cfgstr,
        'flann_params': flann_params,
        'use_cache': use_cache,
        'use_params_hash': use_params_hash})
    nnindexer = NeighborIndex(ax2_aid, idx2_vec, idx2_ax, idx2_fx, flann)
    return nnindexer


def new_ibeis_nnindexer(ibs, daid_list):
    """
    IBEIS interface into neighbor_index

    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> nnindexer, qreq_, ibs = test_nnindexer() # doctest: +ELLIPSIS

    """
    global NEIGHBOR_CACHE
    daids_hashid = ibs.get_annot_uuid_hashid(daid_list, '_DUUIDS')
    flann_cfgstr = ibs.cfg.query_cfg.flann_cfg.get_cfgstr()
    feat_cfgstr  = ibs.cfg.query_cfg._feat_cfg.get_cfgstr()
    indexer_cfgstr = daids_hashid + flann_cfgstr + feat_cfgstr
    try:
        # neighbor cache
        if indexer_cfgstr in NEIGHBOR_CACHE:
            nnindexer = NEIGHBOR_CACHE[indexer_cfgstr]
            return nnindexer
        else:
            # Grab the keypoints names and image ids before query time
            #rx2_kpts = ibs.get_annot_kpts(daid_list)
            #rx2_gid  = ibs.get_annot_gids(daid_list)
            #rx2_nid  = ibs.get_annot_nids(daid_list)
            flann_params = ibs.cfg.query_cfg.flann_cfg.get_dict_args()
            # Get annotation descriptors that will be searched
            vecs_list = ibs.get_annot_desc(daid_list)
            flann_cachedir = ibs.get_flann_cachedir()
            nnindexer = new_neighbor_indexer(
                daid_list, vecs_list, flann_params, flann_cachedir,
                indexer_cfgstr, hash_rowids=False, use_params_hash=False)
            if len(NEIGHBOR_CACHE) > MAX_NEIGHBOR_CACHE_SIZE:
                NEIGHBOR_CACHE.clear()
            NEIGHBOR_CACHE[indexer_cfgstr] = nnindexer
            return nnindexer
    except Exception as ex:
        utool.printex(ex, True, msg_='cannot build inverted index',
                        key_list=['ibs.get_infostr()'])
        raise


def new_ibeis_mindexer(ibs, daid_list,
                       num_indexers=8,
                       split_method='name'):
    """
    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> mindexer, qreq_, ibs = test_mindexer()
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
            nnindexer = new_ibeis_nnindexer(ibs, aids)
            nn_indexer_list.append(nnindexer)
    #if len(unknown_aids) > 0:
    #    print('[mindex] building unknown forest')
    #    unknown_vecs_list = ibs.get_annot_desc(overflow_aids)
    #    unknown_index = NeighborIndex(overflow_aids, unknown_vecs_list)
    #    extra_indexes.append(unknown_index)
    ##print('[mindex] building normalizer forest')  # TODO
    #mindexer.nn_indexer_list = nn_indexer_list
    #mindexer.extra_indexes = extra_indexes
    #mindexer.overflow_index = overflow_index
    #mindexer.unknown_index = unknown_index
    mindexer = MultiNeighborIndex(nn_indexer_list)
    return mindexer


def _check_input(aid_list, vecs_list):
    assert len(aid_list) == len(vecs_list), 'invalid input'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')


def test_nnindexer():
    from ibeis.model.hots.query_request import new_ibeis_query_request
    import ibeis
    daid_list = [7, 8, 9, 10, 11]
    ibs = ibeis.opendb(db='testdb1')
    qreq_ = new_ibeis_query_request(ibs, daid_list, daid_list)
    nnindexer = new_ibeis_nnindexer(ibs, qreq_.get_internal_daids())
    return nnindexer, qreq_, ibs


def test_mindexer():
    from ibeis.model.hots.query_request import new_ibeis_query_request
    import ibeis
    ibs = ibeis.opendb(db='PZ_Mothers')
    daid_list = ibs.get_valid_aids()[1:60]
    qreq_ = new_ibeis_query_request(ibs, daid_list, daid_list)
    num_indexers = 4
    split_method = 'name'
    mindexer = new_ibeis_mindexer(ibs, qreq_.get_internal_daids(),
                                  num_indexers, split_method)
    return mindexer, qreq_, ibs


class NeighborIndex(object):
    """
    More abstract wrapper around flann
    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> nnindexer, qreq_, ibs = test_nnindexer()  #doctest: +ELLIPSIS
    """

    def __init__(nnindexer, ax2_aid, idx2_vec, idx2_ax, idx2_fx, flann):
        nnindexer.ax2_aid   = ax2_aid
        nnindexer.idx2_vec  = idx2_vec
        nnindexer.idx2_ax   = idx2_ax  # Index into the aid_list
        nnindexer.idx2_fx   = idx2_fx  # Index into the annot's features
        nnindexer.flann     = flann

    def rrr(nnindexer):
        from ibeis.model.hots import neighbor_index as nnindex
        nnindex.rrr()
        print('reloading NeighborIndex')
        utool.reload_class_methods(nnindexer, nnindex.NeighborIndex)

    def knn(nnindexer, qfx2_vec, K, checks=1028):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()  #doctest: +ELLIPSIS
        >>> new_aid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> new_vecs_list = ibs.get_annot_desc(new_aid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (qfx2_idx, qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=checks)
        """
        (qfx2_idx, qfx2_dist) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        return (qfx2_idx, qfx2_dist)

    def add_points(nnindexer, new_aid_list, new_vecs_list):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, qreq_, ibs = test_nnindexer()  #doctest: +ELLIPSIS
        >>> new_aid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> new_vecs_list = ibs.get_annot_desc(new_aid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (qfx2_idx1, qfx2_dist1) = nnindexer.knn(qfx2_vec, K, checks=checks)
        >>> nnindexer.add_points(new_aid_list, new_vecs_list)
        >>> (qfx2_idx2, qfx2_dist2) = nnindexer.knn(qfx2_vec, K, checks=checks)
        >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        nAnnots = nnindexer.num_indexed_annots()
        nNew    = len(new_aid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNew)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list)
        # Stack inverted information
        _ax2_aid = np.hstack((nnindexer.ax2_aid, new_aid_list))
        _idx2_ax = np.hstack((nnindexer.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((nnindexer.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((nnindexer.idx2_vec, new_idx2_vec))
        nnindexer.ax2_aid  = _ax2_aid
        nnindexer.idx2_ax  = _idx2_ax
        nnindexer.idx2_vec = _idx2_vec
        nnindexer.idx2_fx  = _idx2_fx
        #nnindexer.idx2_kpts   = None
        #nnindexer.idx2_oris   = None
        # Add new points to flann structure
        nnindexer.flann.add_points(new_idx2_vec)

    def num_indexed_vecs(nnindexer):
        return len(nnindexer.idx2_vec)

    def num_indexed_annots(nnindexer):
        return len(nnindexer.ax2_aid)

    def get_indexed_axs(nnindexer):
        return nnindexer.idx2_ax

    def get_nn_axs(nnindexer, qfx2_nndx):
        #return nnindexer.idx2_ax[qfx2_nndx]
        return nnindexer.idx2_ax.take(qfx2_nndx)

    def get_nn_aids(nnindexer, qfx2_nndx):
        #qfx2_ax = nnindexer.idx2_ax[qfx2_nndx]
        #qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        qfx2_ax = nnindexer.idx2_ax.take(qfx2_nndx)
        qfx2_aid = nnindexer.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nndx):
        #return nnindexer.idx2_fx[qfx2_nndx]
        return nnindexer.idx2_fx.take(qfx2_nndx)


class MultiNeighborIndex(object):
    """
    Generalization of a HOTSNNIndex
    More abstract wrapper around flann
    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> mindexer, qreq_, ibs = test_mindexer()
    """

    def __init__(mindexer, nn_indexer_list):
        mindexer.nn_indexer_list = nn_indexer_list

    def knn(mindexer, qfx2_vec, K):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mindexer, qreq_, ibs = test_mindexer()
        >>> K = 3
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> (qfx2_imx_, qfx2_dist_) = mindexer.knn(qfx2_vec, K)
        """
        qfx2_imx_list   = []
        qfx2_dist_list = []
        offset = 0
        for tx, nnindexer in enumerate(mindexer.nn_indexer_list):
            # Returns distances in ascending order for each query descriptor
            (_qfx2_idx, _qfx2_dist) = nnindexer.knn(qfx2_vec, K, checks=1024)
            _qfx2_imx = _qfx2_idx + offset
            qfx2_imx_list.append(_qfx2_imx)
            qfx2_dist_list.append(_qfx2_dist)
            offset += nnindexer.num_indexed_vecs()
        # Combine results from each tree
        qfx2_imx_   = np.hstack(qfx2_imx_list)
        qfx2_dist_  = np.hstack(qfx2_dist_list)
        # Sort over all tree result distances
        qfx2_sortx = qfx2_dist_.argsort(axis=1)
        # Apply sorting to concatenated results
        def sortaxis1(qfx2_xxx):
            return  np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])
        qfx2_dist  = sortaxis1(qfx2_dist_)
        qfx2_imx   = sortaxis1(qfx2_imx_)
        return (qfx2_imx, qfx2_dist)

    def knn2(mindexer, qfx2_vec, K):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mindexer, qreq_, ibs = test_mindexer()
        >>> K = 3
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> (qfx2_dist_, qfx2_idx_,  qfx2_fx_, qfx2_ax_, qfx2_rankx_, qfx2_treex_,) = mindexer.knn2(qfx2_vec, K)
        """
        qfx2_idx_list   = []
        qfx2_dist_list = []
        qfx2_ax_list  = []
        qfx2_fx_list   = []
        qfx2_rankx_list = []  # ranks index
        qfx2_treex_list = []  # tree index
        for tx, nnindexer in enumerate(mindexer.nn_indexer_list):
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

    def add_points(mindexer, new_aid_list, new_vecs_list):
        raise NotImplementedError()

    def num_indexed_vecs(mindexer):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mindexer, qreq_, ibs = test_mindexer()
        >>> out = mindexer.num_indexed_vecs()
        >>> print(out)
        54141
        """
        return np.sum([nnindexer.num_indexed_vecs()
                          for tx, nnindexer in enumerate(mindexer.nn_indexer_list)])

    def num_indexed_annots(mindexer):
        return np.sum([nnindexer.num_indexed_annots()
                          for tx, nnindexer in enumerate(mindexer.nn_indexer_list)])

    def get_indexed_axs(mindexer):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mindexer, qreq_, ibs = test_mindexer()
        >>> out = mindexer.get_indexed_axs()
        >>> print(out)
        54141
        """
        stack = [nnindexer.get_indexed_axs() for tx, nnindexer in enumerate(mindexer.nn_indexer_list)]
        return np.vstack(stack)

    def get_nn_axs(mindexer, qfx2_nndx):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> mindexer, qreq_, ibs = test_mindexer()
        """
        raise NotImplementedError()
        return np.vstack([nnindexer.get_nn_axs(qfx2_nndx)
                          for tx, nnindexer in enumerate(mindexer.nn_indexer_list)])

    def get_nn_aids(mindexer, qfx2_nndx):
        raise NotImplementedError()
        return np.vstack([nnindexer.get_nn_aids(qfx2_nndx)
                          for tx, nnindexer in enumerate(mindexer.nn_indexer_list)])

    def get_nn_featxs(mindexer, qfx2_nndx):
        raise NotImplementedError()
        return np.vstack([nnindexer.get_nn_featxs(qfx2_nndx)
                          for tx, nnindexer in enumerate(mindexer.nn_indexer_list)])


def invert_index(vecs_list, ax_list):
    """
    Aggregates descriptors of input annotations and returns inverted information
    """
    if utool.NOT_QUIET:
        print('[hsnbrx] stacking descriptors from %d annotations'
                % len(ax_list))
    try:
        idx2_vec, idx2_ax, idx2_fx = nntool.invertable_stack(vecs_list, ax_list)
        assert idx2_vec.shape[0] == idx2_ax.shape[0]
        assert idx2_vec.shape[0] == idx2_fx.shape[0]
    except MemoryError as ex:
        utool.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if utool.NOT_QUIET:
        print('stacked nVecs={nVecs} from nAnnots={nAnnots}'.format(
            nVecs=len(idx2_vec), nAnnots=len(ax_list)))
    return idx2_vec, idx2_ax, idx2_fx


#if __name__ == '__main__':
#    #python -m doctest -v ibeis/model/hots/neighbor_index.py
#    import doctest
#    doctest.testmod()
