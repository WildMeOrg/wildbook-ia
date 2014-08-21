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
import vtool.nearest_neighbors as nntool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[neighbor_index]', DEBUG=False)

NOCACHE_FLANN = '--nocache-flann' in sys.argv


def make_test_index():
    import ibeis
    ibs = ibeis.test_main(db='testdb1')
    aid_list = [7, 8, 9, 10, 11]
    vecs_list = ibs.get_annot_desc(aid_list)
    nnindexer = NeighborIndex(aid_list, vecs_list)
    return nnindexer, ibs


def _check_input(aid_list, vecs_list):
    assert len(aid_list) == len(vecs_list), 'invalid input'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')


class NeighborIndex(object):
    """
    More abstract wrapper around flann
    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> nnindexer, ibs = make_test_index()  #doctest: +ELLIPSIS
    """

    def rrr(nnindexer):
        from ibeis.model.hots import neighbor_index as nnindex
        nnindex.rrr()
        print('reloading NeighborIndex')
        utool.reload_class_methods(nnindexer, nnindex.NeighborIndex)

    def __init__(nnindexer, aid_list=[], vecs_list=[], flann_params={},
                 flann_cachedir='.', indexer_cfgstr='', hash_rowids=True,
                 use_cache=not NOCACHE_FLANN, use_params_hash=True):
        _check_input(aid_list, vecs_list)
        #print('[nnindexer] building NeighborIndex object')
        # Create indexes into the input aids
        ax_list = np.arange(len(aid_list))
        idx2_vec, idx2_ax, idx2_fx = invert_index(vecs_list, ax_list)
        nnindexer.ax2_aid   = np.array(aid_list)
        nnindexer.idx2_vec  = idx2_vec
        nnindexer.idx2_ax   = idx2_ax  # Index into the aid_list
        nnindexer.idx2_fx   = idx2_fx  # Index into the annot's features
        if hash_rowids:
            # Fingerprint
            aids_hashstr = utool.hashstr_arr(aid_list, '_AIDS')
            cfgstr = aids_hashstr + indexer_cfgstr
        else:
            # Dont hash rowids when given enough info in indexer_cfgstr
            cfgstr = indexer_cfgstr
        nnindexer.cfgstr = cfgstr
        # Build/Load the flann index
        nnindexer.flann = nntool.flann_cache(idx2_vec, **{
            'cache_dir': flann_cachedir,
            'cfgstr': cfgstr,
            'flann_params': flann_params,
            'use_cache': use_cache,
            'use_params_hash': use_params_hash})

    def add_points(nnindexer, new_aid_list, new_vecs_list):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, ibs = make_test_index()  #doctest: +ELLIPSIS
        >>> new_aid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> new_vecs_list = ibs.get_annot_desc(new_aid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (qfx2_dx1, qfx2_dist1) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        >>> nnindexer.add_points(new_aid_list, new_vecs_list)
        >>> (qfx2_dx2, qfx2_dist2) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        >>> assert qfx2_dx2.max() > qfx2_dx1.max()
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
        nnindexer.ax2_aid = _ax2_aid
        nnindexer.idx2_ax  = _idx2_ax
        nnindexer.idx2_vec = _idx2_vec
        nnindexer.idx2_fx  = _idx2_fx
        #nnindexer.idx2_kpts   = None
        #nnindexer.idx2_oris   = None
        # Add new points to flann structure
        nnindexer.flann.add_points(new_idx2_vec)

    def knn(nnindexer, qfx2_vec, K, checks=1028):
        """
        >>> K = 2
        >>> checks = 1028
        """
        (qfx2_dx, qfx2_dist) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        return (qfx2_dx, qfx2_dist)

    def num_indexed_vecs(nnindexer):
        return len(nnindexer.idx2_vec)

    def num_indexed_annots(nnindexer):
        return len(nnindexer.ax2_aid)

    def get_indexed_axs(nnindexer):
        return nnindexer.idx2_ax

    def get_nn_axs(nnindexer, qfx2_nndx):
        return nnindexer.idx2_ax[qfx2_nndx]

    def get_nn_aids(nnindexer, qfx2_nndx):
        qfx2_ax = nnindexer.idx2_ax[qfx2_nndx]
        qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nndx):
        return nnindexer.idx2_fx[qfx2_nndx]


class MultiNeighborIndex(object):
    """
    Generalization of a HOTSNNIndex

    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> import ibeis
    >>> daid_list = [1, 2, 3, 4]
    >>> num_forests = 8
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    <BLANKLINE>
    ...
    >>> mindexer = MultiNeighborIndex(ibs, daid_list, num_forests)  #doctest: +ELLIPSIS
    [nnsindex...
    >>> print(mindexer) #doctest: +ELLIPSIS
    <ibeis.model.hots.neighbor_index.HOTSMultiIndex object at ...>
    """

    def __init__(mindexer, ibs, daid_list, num_forests=8):
        print('[nnsindex] make HOTSMultiIndex over %d annots' % (len(daid_list),))
        # Remove unknown names
        aid_list = daid_list
        known_aids_list, unknown_aids = ibsfuncs.group_annots_by_known_names(ibs, aid_list)

        num_bins = min(max(map(len, known_aids_list)), num_forests)

        # Put one name per forest
        forest_aids, overflow_aids = utool.sample_zip(
            known_aids_list, num_bins, allow_overflow=True, per_bin=1)

        # Build a neighbor indexer for each
        forest_indexes = []
        extra_indexes = []
        for tx, aid_list in enumerate(forest_aids):
            print('[nnsindex] building forest %d/%d with %d aids' %
                  (tx + 1, num_bins, len(aid_list)))
            if len(aid_list) > 0:
                vecs_list = ibs.get_annot_desc(aid_list)
                nnindexer = NeighborIndex(aid_list, vecs_list)
                forest_indexes.append(nnindexer)

        if len(overflow_aids) > 0:
            print('[nnsindex] building overflow forest')
            overflow_index = NeighborIndex(ibs, overflow_aids)
            extra_indexes.append(overflow_index)
        if len(unknown_aids) > 0:
            print('[nnsindex] building unknown forest')
            unknown_index = NeighborIndex(ibs, unknown_aids)
            extra_indexes.append(unknown_index)
        #print('[nnsindex] building normalizer forest')  # TODO

        mindexer.forest_indexes = forest_indexes
        mindexer.extra_indexes = extra_indexes
        #mindexer.overflow_index = overflow_index
        #mindexer.unknown_index = unknown_index

    def knn(mindexer, qfx2_desc, num_neighbors):
        qfx2_dx_list   = []
        qfx2_dist_list = []
        qfx2_aid_list  = []
        qfx2_fx_list   = []
        qfx2_rankx_list = []  # ranks index
        qfx2_treex_list = []  # tree index
        for tx, mindexer in enumerate(mindexer.forest_indexes):
            flann = mindexer.flann
            # Returns distances in ascending order for each query descriptor
            (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, num_neighbors, checks=1024)
            qfx2_dx_list.append(qfx2_dx)
            qfx2_dist_list.append(qfx2_dist)
            qfx2_fx = mindexer.idx2_fx[qfx2_dx]
            qfx2_aid = mindexer.idx2_aid[qfx2_dx]
            qfx2_fx_list.append(qfx2_fx)
            qfx2_aid_list.append(qfx2_aid)
            qfx2_rankx_list.append(np.array([[rankx for rankx in range(qfx2_dx.shape[1])]] * len(qfx2_dx)))
            qfx2_treex_list.append(np.array([[tx for rankx in range(qfx2_dx.shape[1])]] * len(qfx2_dx)))
        # Combine results from each tree
        (qfx2_dist_, qfx2_aid_,  qfx2_fx_, qfx2_dx_, qfx2_rankx_, qfx2_treex_,) = \
            join_split_nn(qfx2_dist_list, qfx2_dist_list, qfx2_rankx_list, qfx2_treex_list)


def join_split_nn(qfx2_dx_list, qfx2_dist_list, qfx2_aid_list, qfx2_fx_list, qfx2_rankx_list, qfx2_treex_list):
    qfx2_dx    = np.hstack(qfx2_dx_list)
    qfx2_dist  = np.hstack(qfx2_dist_list)
    qfx2_rankx = np.hstack(qfx2_rankx_list)
    qfx2_treex = np.hstack(qfx2_treex_list)
    qfx2_aid   = np.hstack(qfx2_aid_list)
    qfx2_fx    = np.hstack(qfx2_fx_list)

    # Sort over all tree result distances
    qfx2_sortx = qfx2_dist.argsort(axis=1)
    # Apply sorting to concatenated results
    qfx2_dist_  = [row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_dist)]
    qfx2_aid_   = [row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_dx)]
    qfx2_fx_    = [row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_aid)]
    qfx2_dx_    = [row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_fx)]
    qfx2_rankx_ = [row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_rankx)]
    qfx2_treex_ = [row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_treex)]
    return (qfx2_dist_, qfx2_aid_,  qfx2_fx_, qfx2_dx_, qfx2_rankx_, qfx2_treex_,)


def split_index_daids(mindexer):
    for mindexer in mindexer.forest_indexes:
        pass


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
