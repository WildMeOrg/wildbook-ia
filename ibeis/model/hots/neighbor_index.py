"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.hots_nn_index))"
python -m doctest -v ibeis/model/hots/hots_nn_index.py
python -m doctest ibeis/model/hots/hots_nn_index.py
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


def get_ibies_neighbor_index(ibs, daid_list, NEIGHBOR_CACHE={}):
    """
    >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
    >>> import ibeis
    >>> daid_list = [1, 2, 3, 4]
    >>> num_forests = 8
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    """
    try:
        # Grab the keypoints names and image ids before query time
        #rx2_kpts = ibs.get_annot_kpts(daid_list)
        #rx2_gid  = ibs.get_annot_gids(daid_list)
        #rx2_nid  = ibs.get_annot_nids(daid_list)
        duuid_list = ibs.get_annot_uuids(daid_list)
        dauuid_cfgstr = utool.hashstr_arr(duuid_list, 'duuids')  # todo change to uuids
        feat_cfgstr = ibs.cfg.feat_cfg.get_cfgstr()
        flann_cachedir = ibs.get_flann_cachedir()
        indexed_cfgstr = dauuid_cfgstr + feat_cfgstr
        # TODO: neighbor cache goes here:
        if indexed_cfgstr in NEIGHBOR_CACHE:
            neighbor_index = NEIGHBOR_CACHE[indexed_cfgstr]
        else:
            flann_params = {
                'algorithm': 'kdtree',
                'trees': 4
            }
            rowid_list = daid_list
            vecs_list = ibs.get_annot_desc(daid_list)
            _tup = (rowid_list, vecs_list, flann_cachedir, flann_params,
                    indexed_cfgstr)
            neighbor_index = NeighborIndex(*_tup)
            return neighbor_index
    except Exception as ex:
        utool.printex(ex, True, msg_='cannot build inverted index', key_list=['ibs.get_infostr()'])
        raise


def try_map_vecx_to_rowids(vecs_list, rowid_list):
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


def make_test_index():
    import ibeis
    ibs = ibeis.test_main(db='testdb1')
    rowid_list = [1, 2, 3, 4]
    vecs_list = ibs.get_annot_desc(rowid_list)
    nbrx = NeighborIndex(rowid_list, vecs_list)
    return nbrx, ibs


class NeighborIndex(object):
    def __init__(nbrx, rowid_list=[], vecs_list=[], flann_params={},
                 flann_cachedir='.', indexed_cfgstr='',
                 use_cache=not NOCACHE_FLANN, *args):
        """
        >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
        >>> nbrx, ibs = make_test_index()  #doctest: +ELLIPSIS
        """
        print('[nnindex] building NeighborIndex object')
        assert len(rowid_list) == len(vecs_list), 'invalid input'
        assert len(rowid_list) > 0, (
            'len(aid_list) == 0. Cannot invert index without features!')
        # Fingerprint
        rowid_cfgstr = utool.hashstr_arr(rowid_list, 'rowids')
        indexed_cfgstr_ = rowid_cfgstr + indexed_cfgstr
        # Agg Data, Build the nn index
        dx2_vec, dx2_rowid, dx2_fx = try_map_vecx_to_rowids(vecs_list, rowid_list)
        # indexed_cfgstr ~ _rowids((6)qbm6uaegu7gv!ut!)_FEAT(params)
        flannkw = {
            'cache_dir': flann_cachedir,
            'cfgstr': indexed_cfgstr_,
            'flann_params': flann_params,
            'use_cache': use_cache,
        }
        # Build/Load the flann index
        #nbrx = NeighborIndex(dx2_rowid, dx2_vec, dx2_fx, flannkw)
        nbrx._initialize(dx2_rowid, dx2_vec, dx2_fx, flannkw)

    def _initialize(nbrx, dx2_rowid, dx2_vec, dx2_fx, flannkw):
        nbrx.flann = nntool.flann_cache(dx2_vec, **flannkw)
        nbrx.dx2_rowid  = dx2_rowid
        nbrx.dx2_vec    = dx2_vec
        nbrx.dx2_fx     = dx2_fx

    def __getstate__(nbrx):
        """ This class it not pickleable """
        #printDBG('get state HOTSIndex')
        return None

    def add_points(nbrx, new_vecs_list, new_rowid_list):
        """
        >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
        >>> nbrx, ibs = make_test_index()  #doctest: +ELLIPSIS
        >>> new_rowid_list = [5, 6, 7]
        >>> qrowid_list = [1, 2, 3]
        >>> new_vecs_list = ibs.get_annot_desc(new_rowid_list)
        >>> nbrx.add_points(new_dx2_rowid, new_vecs_list)
        >>> qvecs_list = ibs.get_annot_desc(daid_list)
        """
        new_dx2_vec, new_dx2_rowid, new_dx2_fx = try_map_vecx_to_rowids(
            new_vecs_list, new_rowid_list)
        _dx2_rowid = (nbrx.dx2_rowid, new_dx2_rowid)
        _dx2_vec = (nbrx.dx2_vec, new_dx2_vec)
        _dx2_fx = (nbrx.dx2_fx, new_dx2_fx)
        nbrx.dx2_rowid  = _dx2_rowid
        nbrx.dx2_vec    = _dx2_vec
        nbrx.dx2_fx     = _dx2_fx
        nbrx.flann.add_points(new_dx2_vec)

    def nn_index(nbrx, qfx2_desc, K, checks):
        (qfx2_dx, qfx2_dist) = nbrx.flann.nn_index(qfx2_desc, K, checks=checks)
        return (qfx2_dx, qfx2_dist)

    def nn_index2(nbrx, qreq, qfx2_desc):
        """ return nearest neighbors from this data_index's flann object """
        flann   = nbrx.flann
        K       = qreq.cfg.nn_cfg.K
        Knorm   = qreq.cfg.nn_cfg.Knorm
        checks  = qreq.cfg.nn_cfg.checks
        # todo store some parameters?

        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        qfx2_aid = nbrx.dx2_aid[qfx2_dx]
        qfx2_fx  = nbrx.dx2_fx[qfx2_dx]
        return qfx2_aid, qfx2_fx, qfx2_dist, K, Knorm


class MultiNeighborIndex(object):
    """
    Generalization of a HOTSNNIndex

    >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
    >>> import ibeis
    >>> daid_list = [1, 2, 3, 4]
    >>> num_forests = 8
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    <BLANKLINE>
    ...
    >>> nbrx = MultiNeighborIndex(ibs, daid_list, num_forests)  #doctest: +ELLIPSIS
    [nnsindex...
    >>> print(nbrx) #doctest: +ELLIPSIS
    <ibeis.model.hots.hots_nn_index.HOTSMultiIndex object at ...>

    </CYTH>
    """

    def __init__(nbrx, ibs, daid_list, num_forests=8):
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
                nbrx = NeighborIndex(ibs, aids)
                forest_indexes.append(nbrx)

        if len(overflow_aids) > 0:
            print('[nnsindex] building overflow forest')
            overflow_index = NeighborIndex(ibs, overflow_aids)
            extra_indexes.append(overflow_index)
        if len(unknown_aids) > 0:
            print('[nnsindex] building unknown forest')
            unknown_index = NeighborIndex(ibs, unknown_aids)
            extra_indexes.append(unknown_index)
        #print('[nnsindex] building normalizer forest')  # TODO

        nbrx.forest_indexes = forest_indexes
        nbrx.extra_indexes = extra_indexes
        #nbrx.overflow_index = overflow_index
        #nbrx.unknown_index = unknown_index


#@utool.classmember(HOTSMultiIndex)
def nn_index(nbrx, qfx2_desc, num_neighbors):
    """ </CYTH> """
    qfx2_dx_list   = []
    qfx2_dist_list = []
    qfx2_aid_list  = []
    qfx2_fx_list   = []
    qfx2_rankx_list = []  # ranks index
    qfx2_treex_list = []  # tree index
    for tx, nbrx in enumerate(nbrx.forest_indexes):
        flann = nbrx.flann
        # Returns distances in ascending order for each query descriptor
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, num_neighbors, checks=1024)
        qfx2_dx_list.append(qfx2_dx)
        qfx2_dist_list.append(qfx2_dist)
        qfx2_fx = nbrx.dx2_fx[qfx2_dx]
        qfx2_aid = nbrx.dx2_aid[qfx2_dx]
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
def split_index_daids(nbrx):
    """ </CYTH> """
    for nbrx in nbrx.forest_indexes:
        pass


#if __name__ == '__main__':
#    #python -m doctest -v ibeis/model/hots/hots_nn_index.py
#    import doctest
#    doctest.testmod()
