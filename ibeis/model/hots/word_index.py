"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.hots.word_index))"
python -m doctest -v ibeis/model/hots/word_index.py
python -m doctest ibeis/model/hots/word_index.py
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
import vtool.nearest_neighbors as nntool
(print, print_, printDBG, rrr_, profile) = utool.inject(__name__, '[entroid_index]')


NOCACHE_WORD = utool.get_argflag('--nocache-word')


# TODO:
class NeighborAssignment():
    def __init__(asgn):
        pass


def test_windex():
    from ibeis.model.hots.query_request import new_ibeis_query_request
    import ibeis
    daid_list = [7, 8, 9, 10, 11]
    ibs = ibeis.opendb(db='testdb1')
    qreq_ = new_ibeis_query_request(ibs, daid_list, daid_list)
    windex = new_ibeis_windex(ibs, qreq_.get_internal_daids())
    return windex, qreq_, ibs


def new_word_index(aid_list=[], vecs_list=[], flann_params={},
                       flann_cachedir=None, indexer_cfgstr='', hash_rowids=True,
                       use_cache=not NOCACHE_WORD, use_params_hash=True):
    print('[windex] building WordIndex object')
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
    windex = WordIndex(ax2_aid, idx2_vec, idx2_ax, idx2_fx, flann)
    return windex


def new_ibeis_windex(ibs, daid_list):
    """
    IBEIS interface into word_index

    >>> from ibeis.model.hots.word_index import *  # NOQA
    >>> windex, qreq_, ibs = test_windex() # doctest: +ELLIPSIS

    """
    daids_hashid = ibs.get_annot_hashid_visual_uuid(daid_list, 'D')
    flann_cfgstr = ibs.cfg.query_cfg.flann_cfg.get_cfgstr()
    feat_cfgstr  = ibs.cfg.query_cfg._feat_cfg.get_cfgstr()
    indexer_cfgstr = daids_hashid + flann_cfgstr + feat_cfgstr
    try:
        # Grab the keypoints names and image ids before query time
        flann_params = ibs.cfg.query_cfg.flann_cfg.get_flann_params()
        # Get annotation descriptors that will be searched
        vecs_list = ibs.get_annot_vecs(daid_list)
        flann_cachedir = ibs.get_flann_cachedir()
        windex = new_word_index(
            daid_list, vecs_list, flann_params, flann_cachedir,
            indexer_cfgstr, hash_rowids=False, use_params_hash=False)
        return windex
    except Exception as ex:
        utool.printex(ex, True, msg_='cannot build inverted index', key_list=['ibs.get_infostr()'])
        raise


def _check_input(aid_list, vecs_list):
    assert len(aid_list) == len(vecs_list), 'invalid input'
    assert len(aid_list) > 0, ('len(aid_list) == 0.'
                                    'Cannot invert index without features!')


@six.add_metaclass(utool.ReloadingMetaclass)
class WordIndex(object):
    """
    Abstract wrapper around flann

    Example:
        >>> from ibeis.model.hots.word_index import *  # NOQA
        >>> windex, qreq_, ibs = test_windex()  #doctest: +ELLIPSIS
    """

    def __init__(windex, ax2_aid, idx2_vec, idx2_ax, idx2_fx, flann):
        windex.ax2_aid  = ax2_aid   # (A x 1) Mapping to original annot ids
        windex.idx2_vec = idx2_vec  # (M x D) Descriptors to index
        windex.idx2_ax  = idx2_ax   # (M x 1) Index into the aid_list
        windex.idx2_fx  = idx2_fx   # (M x 1) Index into the annot's features
        windex.flann    = flann     # Approximate search structure

    def knn(windex, qfx2_vec, K, checks=1028):
        """
        Args:
            qfx2_vec (ndarray): (N x D) array of N, D-dimensional query vectors

            K (int): number of approximate nearest words to find

        Returns:
            tuple of (qfx2_idx, qfx2_dist)

            qfx2_idx (ndarray):  (N x K) qfx2_idx[n][k] is the index of the kth
                        approximate nearest data vector w.r.t qfx2_vec[n]

            qfx2_dist (ndarray): (N x K) qfx2_dist[n][k] is the distance to the kth
                        approximate nearest data vector w.r.t. qfx2_vec[n]

        Example:
            >>> from ibeis.model.hots.word_index import *  # NOQA
            >>> windex, qreq_, ibs = test_windex()  #doctest: +ELLIPSIS
            >>> new_aid_list = [2, 3, 4]
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> new_vecs_list = ibs.get_annot_vecs(new_aid_list)
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx, qfx2_dist) = windex.knn(qfx2_vec, K, checks=checks)
        """
        (qfx2_idx, qfx2_dist) = windex.flann.nn_index(qfx2_vec, K, checks=checks)
        return (qfx2_idx, qfx2_dist)

    def empty_words(K):
        qfx2_idx  = np.empty((0, K), dtype=np.int32)
        qfx2_dist = np.empty((0, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    def add_points(windex, new_aid_list, new_vecs_list):
        """
        Example:
            >>> from ibeis.model.hots.word_index import *  # NOQA
            >>> windex, qreq_, ibs = test_windex()  #doctest: +ELLIPSIS
            >>> new_aid_list = [2, 3, 4]
            >>> qfx2_vec = ibs.get_annot_vecs(1)
            >>> new_vecs_list = ibs.get_annot_vecs(new_aid_list)
            >>> K = 2
            >>> checks = 1028
            >>> (qfx2_idx1, qfx2_dist1) = windex.knn(qfx2_vec, K, checks=checks)
            >>> windex.add_points(new_aid_list, new_vecs_list)
            >>> (qfx2_idx2, qfx2_dist2) = windex.knn(qfx2_vec, K, checks=checks)
            >>> assert qfx2_idx2.max() > qfx2_idx1.max()
        """
        nAnnots = windex.num_indexed_annots()
        nNew    = len(new_aid_list)
        new_ax_list = np.arange(nAnnots, nAnnots + nNew)
        new_idx2_vec, new_idx2_ax, new_idx2_fx = \
                invert_index(new_vecs_list, new_ax_list)
        # Stack inverted information
        _ax2_aid = np.hstack((windex.ax2_aid, new_aid_list))
        _idx2_ax = np.hstack((windex.idx2_ax, new_idx2_ax))
        _idx2_fx = np.hstack((windex.idx2_fx, new_idx2_fx))
        _idx2_vec = np.vstack((windex.idx2_vec, new_idx2_vec))
        windex.ax2_aid  = _ax2_aid
        windex.idx2_ax  = _idx2_ax
        windex.idx2_vec = _idx2_vec
        windex.idx2_fx  = _idx2_fx
        #windex.idx2_kpts   = None
        #windex.idx2_oris   = None
        # Add new points to flann structure
        windex.flann.add_points(new_idx2_vec)

    def num_indexed_vecs(windex):
        return len(windex.idx2_vec)

    def num_indexed_annots(windex):
        return len(windex.ax2_aid)

    def get_nn_axs(windex, qfx2_nnidx):
        #return windex.idx2_ax[qfx2_nnidx]
        return windex.idx2_ax.take(qfx2_nnidx)

    def get_nn_aids(windex, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx (ndarray): (N x K) qfx2_idx[n][k] is the index of the kth
                approximate nearest data vector

        Returns:
            ndarray: qfx2_aid - (N x K) qfx2_fx[n][k] is the annotation id index
                of the kth approximate nearest data vector
        """
        #qfx2_ax = windex.idx2_ax[qfx2_nnidx]
        #qfx2_aid = windex.ax2_aid[qfx2_ax]
        qfx2_ax = windex.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = windex.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(windex, qfx2_nnidx):
        """
        Args:
            qfx2_nnidx (ndarray): (N x K) qfx2_idx[n][k] is the index of the kth
                approximate nearest data vector

        Returns:
            ndarray: qfx2_fx - (N x K) qfx2_fx[n][k] is the feature index (w.r.t
                the source annotation) of the kth approximate nearest data vector
        """
        #return windex.idx2_fx[qfx2_nnidx]
        return windex.idx2_fx.take(qfx2_nnidx)


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
#    #python -m doctest -v ibeis/model/hots/word_index.py
#    import doctest
#    doctest.testmod()
