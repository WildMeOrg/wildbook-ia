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
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[nnindex]', DEBUG=False)

NOCACHE_FLANN = '--nocache-flann' in sys.argv


def get_indexed_cfgstr(ibs, aid_list):
    """
    Creates a config string for the input into the nearest neighbors index
    It is based off of the features which were computed for it and the indexes
    of the input annotations.

    TODO: We should probably use the Annotation UUIDS rather than the ROWIDs
    to compute this configstr

    """
    feat_cfgstr = ibs.cfg.feat_cfg.get_cfgstr()
    # returns something like: _daids((6)qbm6uaegu7gv!ut!)_FEAT(params)
    daid_cfgstr = utool.hashstr_arr(aid_list, 'daids')  # todo change to uuids
    new_cfgstr = '_' + daid_cfgstr + feat_cfgstr
    return new_cfgstr


def build_ibs_inverted_descriptor_index(ibs, aid_list):
    """
    Aggregates descriptors of input annotations and returns inverted information
    """
    try:
        if len(aid_list) == 0:
            msg = ('len(aid_list) == 0\n'
                    'Cannot build inverted index without features!')
            raise AssertionError(msg)
        desc_list = ibs.get_annot_desc(aid_list)
        dx2_desc, dx2_aid, dx2_fx = _try_build_inverted_descriptor_index(aid_list, desc_list)
        return dx2_desc, dx2_aid, dx2_fx
    except Exception as ex:
        intostr = ibs.get_infostr()
        print(intostr)
        utool.printex(ex, 'cannot build inverted index', key_list=list(locals().keys()))
        raise


def _try_build_inverted_descriptor_index(aid_list, desc_list):
    """
    Wrapper which performs logging and error checking
    """
    if utool.NOT_QUIET:
        print('[agg_desc] stacking descriptors from %d annotations' % len(aid_list))
    try:
        dx2_desc, dx2_aid, dx2_fx = _build_inverted_descriptor_index(aid_list, desc_list)
    except MemoryError as ex:
        utool.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    if utool.NOT_QUIET:
        print('[agg_desc] stacked %d descriptors from %d annotations'
                % (len(dx2_desc), len(aid_list)))
    return dx2_desc, dx2_aid, dx2_fx


def _build_inverted_descriptor_index(aid_list, desc_list):
    """
    Stacks descriptors into a flat structure and returns inverse mapping from
    flat database descriptor indexes (dx) to annotation ids (aid) and feature
    indexes (fx). Feature indexes are w.r.t. annotation indexes.

    Output:
        dx2_desc - flat descriptor stack
        dx2_aid  - inverted index into annotations
        dx2_fx   - inverted index into features

    # Example with 2D Descriptors
    >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
    >>> from ibeis.model.hots.hots_nn_index import _build_inverted_descriptor_index
    >>> DESC_TYPE = np.uint8
    >>> aid_list  = [1, 2, 3, 4, 5]
    >>> desc_list = [
    ...     np.array([[0, 0], [0, 1]], dtype=DESC_TYPE),
    ...     np.array([[5, 3], [2, 30], [1, 1]], dtype=DESC_TYPE),
    ...     np.empty((0, 2), dtype=DESC_TYPE),
    ...     np.array([[5, 3], [2, 30], [1, 1]], dtype=DESC_TYPE),
    ...     np.array([[3, 3], [42, 42], [2, 6]], dtype=DESC_TYPE),
    ...     ]
    >>> dx2_desc, dx2_aid, dx2_fx = _build_inverted_descriptor_index(aid_list, desc_list)
    >>> print(repr(dx2_desc.T))
    array([[ 0,  0,  5,  2,  1,  5,  2,  1,  3, 42,  2],
           [ 0,  1,  3, 30,  1,  3, 30,  1,  3, 42,  6]], dtype=uint8)
    >>> print(repr(dx2_aid))
    array([1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 5])
    >>> print(repr(dx2_fx))
    array([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    cdef:
        list aid_list, desc_list
        long nFeat, aid
        iter aid_nFeat_iter, nFeat_iter, _ax2_aid, _ax2_fx
        np.ndarray dx2_aid, dx2_fx, dx2_desc

    """
    # Build inverted index of (aid, fx) pairs
    aid_nFeat_iter = zip(aid_list, map(len, desc_list))
    nFeat_iter = map(len, desc_list)
    # generate aid inverted index for each feature in each annotation
    _ax2_aid = ([aid] * nFeat for (aid, nFeat) in aid_nFeat_iter)
    # Avi: please test the timing of the lines neighboring this statement.
    #_ax2_aid = ([aid] * nFeat for (aid, nFeat) in aid_nFeat_iter)
    # generate featx inverted index for each feature in each annotation
    _ax2_fx  = (range(nFeat) for nFeat in nFeat_iter)
    # Flatten generators into the inverted index
    #dx2_aid = np.array(list(chain.from_iterable(_ax2_aid)))
    #dx2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    dx2_aid = np.array(utool.flatten(_ax2_aid))
    dx2_fx  = np.array(utool.flatten(_ax2_fx))
    # Stack descriptors into numpy array corresponding to inverted inexed
    # This might throw a MemoryError
    dx2_desc = np.vstack(desc_list)
    return dx2_desc, dx2_aid, dx2_fx


#@utool.indent_func('[build_invx]')
def build_flann_inverted_index(ibs, aid_list, **kwargs):
    """
    Build a inverted index (using FLANN)
    """
    # Aggregate descriptors
    dx2_desc, dx2_aid, dx2_fx = build_ibs_inverted_descriptor_index(ibs, aid_list)
    # hash which annotations are input
    indexed_cfgstr = get_indexed_cfgstr(ibs, aid_list)
    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    flann_cachedir = ibs.get_flann_cachedir()
    precomp_kwargs = {'cache_dir': flann_cachedir,
                      'cfgstr': indexed_cfgstr,
                      'flann_params': flann_params,
                      'use_cache': kwargs.get('use_cache', not NOCACHE_FLANN)}
    # Build/Load the flann index
    flann = nntool.flann_cache(dx2_desc, **precomp_kwargs)
    return dx2_desc, dx2_aid, dx2_fx, flann


class HOTSIndex(object):
    """ HotSpotter Nearest Neighbor (FLANN) Index Class
    >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
    >>> import ibeis
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    <BLANKLINE>
    ...
    >>> daid_list = [1, 2, 3, 4]
    >>> hsindex = HOTSIndex(ibs, daid_list)  #doctest: +ELLIPSIS
    [nnindex...
    >>> print(hsindex) #doctest: +ELLIPSIS
    <ibeis.model.hots.hots_nn_index.HOTSIndex object at ...>

    """
    def __init__(hsindex, ibs, daid_list, **kwargs):
        print('[nnindex] building HOTSIndex object')
        dx2_desc, dx2_aid, dx2_fx, flann = build_flann_inverted_index(
            ibs, daid_list, **kwargs)
        # Agg Data
        hsindex.dx2_aid  = dx2_aid
        hsindex.dx2_fx   = dx2_fx
        hsindex.dx2_data = dx2_desc
        # Grab the keypoints names and image ids before query time
        #hsindex.rx2_kpts = ibs.get_annot_kpts(daid_list)
        #hsindex.rx2_gid  = ibs.get_annot_gids(daid_list)
        #hsindex.rx2_nid  = ibs.get_annot_nids(daid_list)
        hsindex.flann = flann

    def __getstate__(hsindex):
        """ This class it not pickleable """
        #printDBG('get state HOTSIndex')
        return None

    #def __del__(hsindex):
    #    """ Ensure flann is propertly removed """
    #    printDBG('deleting HOTSIndex')
    #    if getattr(hsindex, 'flann', None) is not None:
    #        nn_selfindex.flann.delete_index()
    #        #del hsindex.flann
    #    hsindex.flann = None

    def nn_index(hsindex, qfx2_desc, K, checks):
        (qfx2_dx, qfx2_dist) = hsindex.flann.nn_index(qfx2_desc, K, checks=checks)
        return (qfx2_dx, qfx2_dist)

    def nn_index2(hsindex, qreq, qfx2_desc):
        """ return nearest neighbors from this data_index's flann object """
        flann   = hsindex.flann
        K       = qreq.cfg.nn_cfg.K
        Knorm   = qreq.cfg.nn_cfg.Knorm
        checks  = qreq.cfg.nn_cfg.checks

        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        qfx2_aid = hsindex.dx2_aid[qfx2_dx]
        qfx2_fx  = hsindex.dx2_fx[qfx2_dx]
        return qfx2_aid, qfx2_fx, qfx2_dist, K, Knorm


class HOTSMultiIndex(object):
    """
    Generalization of a HOTSNNIndex

    >>> from ibeis.model.hots.hots_nn_index import *  # NOQA
    >>> import ibeis
    >>> daid_list = [1, 2, 3, 4]
    >>> num_forests = 8
    >>> ibs = ibeis.test_main(db='testdb1')  #doctest: +ELLIPSIS
    <BLANKLINE>
    ...
    >>> split_index = HOTSMultiIndex(ibs, daid_list, num_forests)  #doctest: +ELLIPSIS
    [nnsindex...
    >>> print(split_index) #doctest: +ELLIPSIS
    <ibeis.model.hots.hots_nn_index.HOTSMultiIndex object at ...>

    """

    def __init__(split_index, ibs, daid_list, num_forests=8):
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
                hsindex = HOTSIndex(ibs, aids)
                forest_indexes.append(hsindex)

        if len(overflow_aids) > 0:
            print('[nnsindex] building overflow forest')
            overflow_index = HOTSIndex(ibs, overflow_aids)
            extra_indexes.append(overflow_index)
        if len(unknown_aids) > 0:
            print('[nnsindex] building unknown forest')
            unknown_index = HOTSIndex(ibs, unknown_aids)
            extra_indexes.append(unknown_index)
        #print('[nnsindex] building normalizer forest')  # TODO

        split_index.forest_indexes = forest_indexes
        split_index.extra_indexes = extra_indexes
        #split_index.overflow_index = overflow_index
        #split_index.unknown_index = unknown_index


#@utool.classmember(HOTSMultiIndex)
def nn_index(split_index, qfx2_desc, num_neighbors):
    qfx2_dx_list   = []
    qfx2_dist_list = []
    qfx2_aid_list  = []
    qfx2_fx_list   = []
    qfx2_rankx_list = []  # ranks index
    qfx2_treex_list = []  # tree index
    for tx, hsindex in enumerate(split_index.forest_indexes):
        flann = hsindex.flann
        # Returns distances in ascending order for each query descriptor
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, num_neighbors, checks=1024)
        qfx2_dx_list.append(qfx2_dx)
        qfx2_dist_list.append(qfx2_dist)
        qfx2_fx = hsindex.dx2_fx[qfx2_dx]
        qfx2_aid = hsindex.dx2_aid[qfx2_dx]
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


#@utool.classmember(HOTSMultiIndex)
def split_index_daids(split_index):
    for hsindex in split_index.forest_indexes:
        pass


#if __name__ == '__main__':
#    #python -m doctest -v ibeis/model/hots/hots_nn_index.py
#    import doctest
#    doctest.testmod()
