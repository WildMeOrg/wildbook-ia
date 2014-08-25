"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.neighbor_index))"
python -m doctest -v ibeis/model/hots/neighbor_index.py
python -m doctest ibeis/model/hots/neighbor_index.py
"""
from __future__ import absolute_import, division, print_function
# Standard
import six
#from itertools import chain
import sys
# Science
import numpy as np
# UTool
import utool
# VTool
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
    try:
        NEIGHBOR_CACHE.clear()
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
    nnindexer = NeighborIndex(aid_list, idx2_vec, idx2_ax, idx2_fx, flann)
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


@six.add_metaclass(utool.ReloadingMetaclass)
class NeighborIndex(object):
    """
    Abstract wrapper around flann
    >>> from ibeis.model.hots.neighbor_index import *  # NOQA
    >>> nnindexer, qreq_, ibs = test_nnindexer()  #doctest: +ELLIPSIS
    """

    def __init__(nnindexer, aid_list, idx2_vec, idx2_ax, idx2_fx, flann):
        nnindexer.ax2_aid  = np.array(aid_list)  # Mapping to original annot ids
        nnindexer.idx2_vec = idx2_vec  # (M x D) Descriptors to index
        nnindexer.idx2_ax  = idx2_ax   # (M x 1) Index into the aid_list
        nnindexer.idx2_fx  = idx2_fx   # (M x 1) Index into the annot's features
        nnindexer.flann    = flann     # Approximate search structure

    def knn(nnindexer, qfx2_vec, K, checks=1028):
        """
        Input:
            qfx2_vec - (N x D): an array of N, D-dimensional query vectors

            K: number of approximate nearest neighbors to find

        Output: tuple of (qfx2_idx, qfx2_dist)
            qfx2_idx - (N x K): qfx2_idx[n][k] is the index of the kth
                        approximate nearest data vector w.r.t qfx2_vec[n]

            qfx2_dist - (N x K): qfx2_dist[n][k] is the distance to the kth
                        approximate nearest data vector w.r.t. qfx2_vec[n]

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

    def empty_neighbors(K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
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

    def get_nn_axs(nnindexer, qfx2_nnidx):
        #return nnindexer.idx2_ax[qfx2_nnidx]
        return nnindexer.idx2_ax.take(qfx2_nnidx)

    def get_nn_aids(nnindexer, qfx2_nnidx):
        """
        Input:
            qfx2_nnidx - (N x K): qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Output:
            qfx2_aid - (N x K): qfx2_fx[n][k] is the annotation id index of the
                                kth approximate nearest data vector
        """
        #qfx2_ax = nnindexer.idx2_ax[qfx2_nnidx]
        #qfx2_aid = nnindexer.ax2_aid[qfx2_ax]
        qfx2_ax = nnindexer.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = nnindexer.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nnidx):
        """
        Input:
            qfx2_nnidx - (N x K): qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Output:
            qfx2_fx - (N x K): qfx2_fx[n][k] is the feature index (w.r.t the
                               source annotation) of the kth approximate
                               nearest data vector
        """
        #return nnindexer.idx2_fx[qfx2_nnidx]
        return nnindexer.idx2_fx.take(qfx2_nnidx)


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
