"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.centroid_index))"
python -m doctest -v ibeis/model/hots/centroid_index.py
python -m doctest ibeis/model/hots/centroid_index.py
"""
from __future__ import absolute_import, division, print_function
# Standard
import six
#from itertools import chain
# Science
import numpy as np
# UTool
import vtool
import utool
# VTool
import vtool.nearest_centroids as nntool
(print, print_, printDBG, rrr_, profile) = utool.inject(__name__, '[entroid_index]')


NOCACHE_CENTROID = utool.get_flag('--nocache-centroid')


# TODO:
class NeighborAssignment():
    def __init__(asgn):
        pass


def test_cindex():
    from ibeis.model.hots.query_request import new_ibeis_query_request
    import ibeis
    daid_list = [7, 8, 9, 10, 11]
    ibs = ibeis.opendb(db='testdb1')
    qreq_ = new_ibeis_query_request(ibs, daid_list, daid_list)
    cindex = new_ibeis_cindex(ibs, qreq_.get_internal_daids())
    return cindex, qreq_, ibs


def new_centroid_index(aid_list=[], vecs_list=[], flann_params={},
                       flann_cachedir=None, indexer_cfgstr='', hash_rowids=True,
                       use_cache=not NOCACHE_CENTROID, use_params_hash=True):
    print('[cindex] building CentroidIndex object')
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
    ax2_aid = np.array(aid_list)
    cindex = CentroidIndex(ax2_aid, idx2_vec, idx2_ax, idx2_fx, flann)
    return cindex


def new_ibeis_cindex(ibs, daid_list):
    """
    IBEIS interface into centroid_index

    >>> from ibeis.model.hots.centroid_index import *  # NOQA
    >>> cindex, qreq_, ibs = test_cindex() # doctest: +ELLIPSIS

    """
    daids_hashid = ibs.get_annot_uuid_hashid(daid_list, '_DUUIDS')
    flann_cfgstr = ibs.cfg.query_cfg.flann_cfg.get_cfgstr()
    feat_cfgstr  = ibs.cfg.query_cfg._feat_cfg.get_cfgstr()
    indexer_cfgstr = daids_hashid + flann_cfgstr + feat_cfgstr
    try:
        # Grab the keypoints names and image ids before query time
        flann_params = ibs.cfg.query_cfg.flann_cfg.get_dict_args()
        # Get annotation descriptors that will be searched
        vecs_list = ibs.get_annot_desc(daid_list)
        flann_cachedir = ibs.get_flann_cachedir()
        cindex = new_centroid_index(
            daid_list, vecs_list, flann_params, flann_cachedir,
            indexer_cfgstr, hash_rowids=False, use_params_hash=False)
        return cindex
    except Exception as ex:
        utool.printex(ex, True, msg_='cannot build inverted index', key_list=['ibs.get_infostr()'])
        raise


def _check_input(aid_list, vecs_list):
    assert len(aid_list) == len(vecs_list), 'invalid input'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')


@six.add_metaclass(utool.ReloadingMetaclass)
class CentroidIndex(object):
    """
    Abstract wrapper around flann
    >>> from ibeis.model.hots.centroid_index import *  # NOQA
    >>> cindex, qreq_, ibs = test_cindex()  #doctest: +ELLIPSIS
    """

    def __init__(cindex, ax2_aid, idx2_vec, idx2_ax, idx2_fx, flann):
        cindex.ax2_aid  = ax2_aid   # (A x 1) Mapping to original annot ids
        cindex.idx2_vec = idx2_vec  # (M x D) Descriptors to index
        cindex.idx2_ax  = idx2_ax   # (M x 1) Index into the aid_list
        cindex.idx2_fx  = idx2_fx   # (M x 1) Index into the annot's features
        cindex.flann    = flann     # Approximate search structure

    def knn(cindex, qfx2_vec, K, checks=1028):
        """
        Input:
            qfx2_vec - (N x D): an array of N, D-dimensional query vectors

            K: number of approximate nearest centroids to find

        Output: tuple of (qfx2_idx, qfx2_dist)
            qfx2_idx - (N x K): qfx2_idx[n][k] is the index of the kth
                        approximate nearest data vector w.r.t qfx2_vec[n]

            qfx2_dist - (N x K): qfx2_dist[n][k] is the distance to the kth
                        approximate nearest data vector w.r.t. qfx2_vec[n]

        >>> from ibeis.model.hots.centroid_index import *  # NOQA
        >>> cindex, qreq_, ibs = test_cindex()  #doctest: +ELLIPSIS
        >>> new_aid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> new_vecs_list = ibs.get_annot_desc(new_aid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (qfx2_idx, qfx2_dist) = cindex.knn(qfx2_vec, K, checks=checks)
        """
        (qfx2_idx, qfx2_dist) = cindex.flann.nn_index(qfx2_vec, K, checks=checks)
        return (qfx2_idx, qfx2_dist)

    def empty_centroids(K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    def add_points(cindex, new_aid_list, new_vecs_list):
        """
        >>> from ibeis.model.hots.centroid_index import *  # NOQA
        >>> cindex, qreq_, ibs = test_cindex()  #doctest: +ELLIPSIS
        >>> new_aid_list = [2, 3, 4]
        >>> qfx2_vec = ibs.get_annot_desc(1)
        >>> new_vecs_list = ibs.get_annot_desc(new_aid_list)
        >>> K = 2
        >>> checks = 1028
        >>> (qfx2_idx1, qfx2_dist1) = cindex.knn(qfx2_vec, K, checks=checks)
        >>> cindex.add_points(new_aid_list, new_vecs_list)
        >>> (qfx2_idx2, qfx2_dist2) = cindex.knn(qfx2_vec, K, checks=checks)
        >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        nAnnots = cindex.num_indexed_annots()
        nNew    = len(new_aid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNew)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list)
        # Stack inverted information
        _ax2_aid = np.hstack((cindex.ax2_aid, new_aid_list))
        _idx2_ax = np.hstack((cindex.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((cindex.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((cindex.idx2_vec, new_idx2_vec))
        cindex.ax2_aid  = _ax2_aid
        cindex.idx2_ax  = _idx2_ax
        cindex.idx2_vec = _idx2_vec
        cindex.idx2_fx  = _idx2_fx
        #cindex.idx2_kpts   = None
        #cindex.idx2_oris   = None
        # Add new points to flann structure
        cindex.flann.add_points(new_idx2_vec)

    def num_indexed_vecs(cindex):
        return len(cindex.idx2_vec)

    def num_indexed_annots(cindex):
        return len(cindex.ax2_aid)

    def get_nn_axs(cindex, qfx2_nnidx):
        #return cindex.idx2_ax[qfx2_nnidx]
        return cindex.idx2_ax.take(qfx2_nnidx)

    def get_nn_aids(cindex, qfx2_nnidx):
        """
        Input:
            qfx2_nnidx - (N x K): qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Output:
            qfx2_aid - (N x K): qfx2_fx[n][k] is the annotation id index of the
                                kth approximate nearest data vector
        """
        #qfx2_ax = cindex.idx2_ax[qfx2_nnidx]
        #qfx2_aid = cindex.ax2_aid[qfx2_ax]
        qfx2_ax = cindex.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = cindex.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(cindex, qfx2_nnidx):
        """
        Input:
            qfx2_nnidx - (N x K): qfx2_idx[n][k] is the index of the kth
                                  approximate nearest data vector
        Output:
            qfx2_fx - (N x K): qfx2_fx[n][k] is the feature index (w.r.t the
                               source annotation) of the kth approximate
                               nearest data vector
        """
        #return cindex.idx2_fx[qfx2_nnidx]
        return cindex.idx2_fx.take(qfx2_nnidx)


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


def vlad(qfx2_vec, qfx2_cvec):
    qfx2_rvec = qfx2_cvec - qfx2_vec
    aggvlad = qfx2_rvec.sum(axis=0)
    aggvlad_norm = vtool.l2normalize(aggvlad)
    return aggvlad_norm



#if __name__ == '__main__':
#    #python -m doctest -v ibeis/model/hots/centroid_index.py
#    import doctest
#    doctest.testmod()
