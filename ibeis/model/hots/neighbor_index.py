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
    rowid_list = [7, 8, 9, 10, 11]
    vecs_list = ibs.get_annot_desc(rowid_list)
    nnindexer = NeighborIndex(rowid_list, vecs_list)
    return nnindexer, ibs


def _check_input(rowid_list, vecs_list):
    assert len(rowid_list) == len(vecs_list), 'invalid input'
    assert len(rowid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')


class NeighborIndex(object):
    """
    More abstract wrapper around flann
    """

    def rrr(nnindexer):
        from ibeis.model.hots import neighbor_index as nnindex
        nnindex.rrr()
        print('reloading NeighborIndex')
        utool.reload_class_methods(nnindexer, nnindex.NeighborIndex)

    def __init__(nnindexer, rowid_list=[], vecs_list=[], flann_params={},
                 flann_cachedir='.', indexer_cfgstr='', hash_rowids=True,
                 use_cache=not NOCACHE_FLANN, use_params_hash=True):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, ibs = make_test_index()  #doctest: +ELLIPSIS
        """
        _check_input(rowid_list, vecs_list)
        #print('[nnindexer] building NeighborIndex object')
        dx2_vec, dx2_rowid, dx2_fx = try_invert_vecx(vecs_list, rowid_list)
        nnindexer.rowid_list = rowid_list
        nnindexer.dx2_rowid  = dx2_rowid
        nnindexer.dx2_vec    = dx2_vec
        nnindexer.dx2_fx     = dx2_fx
        if hash_rowids:
            # Fingerprint
            rowid_cfgstr = utool.hashstr_arr(rowid_list, '_ROWIDS')
            cfgstr = rowid_cfgstr + indexer_cfgstr
        else:
            # Dont hash rowids when given enough info in indexer_cfgstr
            cfgstr = indexer_cfgstr
        nnindexer.cfgstr = cfgstr
        # Build/Load the flann index
        nnindexer.flann = nntool.flann_cache(dx2_vec, **{
            'cache_dir': flann_cachedir,
            'cfgstr': cfgstr,
            'flann_params': flann_params,
            'use_cache': use_cache,
            'use_params_hash': use_params_hash})

    def add_points(nnindexer, new_rowid_list, new_vecs_list):
        """
        >>> from ibeis.model.hots.neighbor_index import *  # NOQA
        >>> nnindexer, ibs = make_test_index()  #doctest: +ELLIPSIS
        >>> new_rowid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> new_vecs_list = ibs.get_annot_desc(new_rowid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (_, qfx2_dist1) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        >>> nnindexer.add_points(new_vecs_list, new_vecs_list)
        >>> (_, qfx2_dist2) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        >>> assert qfx2_dx2.max() > qfx2_dx1.max()
        """
        new_dx2_vec, new_dx2_rowid, new_dx2_fx = \
                try_invert_vecx(new_vecs_list, new_rowid_list)
        # Stack inverted information
        _dx2_rowid = np.hstack((nnindexer.dx2_rowid, new_dx2_rowid))
        _dx2_fx = np.hstack((nnindexer.dx2_fx, new_dx2_fx))
        _dx2_vec = np.vstack((nnindexer.dx2_vec, new_dx2_vec))
        nnindexer.dx2_rowid  = _dx2_rowid
        nnindexer.dx2_vec    = _dx2_vec
        nnindexer.dx2_fx     = _dx2_fx
        # Add new points to flann structure
        nnindexer.flann.add_points(new_dx2_vec)

    def knn(nnindexer, qfx2_vec, K, checks=1028):
        """
        >>> K = 2
        >>> checks = 1028
        """
        (qfx2_dx, qfx2_dist) = nnindexer.flann.nn_index(qfx2_vec, K, checks=checks)
        return (qfx2_dx, qfx2_dist)

    def nn_index2(nnindexer, qreq, qfx2_vec):
        """ return nearest neighbors from this data_index's flann object """
        flann   = nnindexer.flann
        K       = qreq.cfg.nn_cfg.K
        Knorm   = qreq.cfg.nn_cfg.Knorm
        checks  = qreq.cfg.nn_cfg.checks
        # todo store some parameters?

        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_vec, K + Knorm, checks=checks)
        qfx2_aid = nnindexer.dx2_aid[qfx2_dx]
        qfx2_fx  = nnindexer.dx2_fx[qfx2_dx]
        return qfx2_aid, qfx2_fx, qfx2_dist, K, Knorm


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

    </CYTH>
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

        forest_indexes = []
        extra_indexes = []
        for tx, aids in enumerate(forest_aids):
            print('[nnsindex] building forest %d/%d with %d aids' %
                  (tx + 1, num_bins, len(aids)))
            if len(aids) > 0:
                mindexer = NeighborIndex(ibs, aids)
                forest_indexes.append(mindexer)

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


#@utool.classmember(HOTSMultiIndex)
def knn(mindexer, qfx2_desc, num_neighbors):
    """ </CYTH> """
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
        qfx2_fx = mindexer.dx2_fx[qfx2_dx]
        qfx2_aid = mindexer.dx2_aid[qfx2_dx]
        qfx2_fx_list.append(qfx2_fx)
        qfx2_aid_list.append(qfx2_aid)
        qfx2_rankx_list.append(np.array([[rankx for rankx in range(qfx2_dx.shape[1])]] * len(qfx2_dx)))
        qfx2_treex_list.append(np.array([[tx for rankx in range(qfx2_dx.shape[1])]] * len(qfx2_dx)))
    # Combine results from each tree
    (qfx2_dist_, qfx2_aid_,  qfx2_fx_, qfx2_dx_, qfx2_rankx_, qfx2_treex_,) = \
            join_split_nn(qfx2_dist_list, qfx2_dist_list, qfx2_rankx_list, qfx2_treex_list)


def join_split_nn(qfx2_dx_list, qfx2_dist_list, qfx2_aid_list, qfx2_fx_list, qfx2_rankx_list, qfx2_treex_list):
    """ </CYTH> """
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


#@utool.classmember(HOTSMultiIndex)
def split_index_daids(mindexer):
    """ </CYTH> """
    for mindexer in mindexer.forest_indexes:
        pass


def try_invert_vecx(vecs_list, rowid_list):
    """
    Aggregates descriptors of input annotations and returns inverted information
    """
    if utool.NOT_QUIET:
        print('[agg_desc] stacking descriptors from %d annotations'
                % len(rowid_list))
    try:
        dx2_vec, dx2_rowid, dx2_fx = nntool.map_vecx_to_rowids(vecs_list, rowid_list)
        assert dx2_vec.shape[0] == dx2_rowid.shape[0]
        assert dx2_vec.shape[0] == dx2_fx.shape[0]
    except MemoryError as ex:
        utool.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if utool.NOT_QUIET:
        print('stacked nVecs={nVecs} from nAnnots={nAnnots}'.format(
            nVecs=len(dx2_vec), nAnnots=len(dx2_rowid)))
    return dx2_vec, dx2_rowid, dx2_fx


#if __name__ == '__main__':
#    #python -m doctest -v ibeis/model/hots/neighbor_index.py
#    import doctest
#    doctest.testmod()
