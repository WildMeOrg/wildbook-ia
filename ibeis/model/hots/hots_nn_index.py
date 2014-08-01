from __future__ import absolute_import, division, print_function
# Standard
from six.moves import zip, map, range
from itertools import chain
import sys
# Science
import numpy as np
# UTool
import utool
# VTool
import vtool.nearest_neighbors as nntool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[nnindex]', DEBUG=False)

NOCACHE_FLANN = '--nocache-flann' in sys.argv


#@utool.indent_func('[get_flann_cfgstr]')
def get_flann_cfgstr(ibs, aid_list):
    """ </CYTHE> """
    feat_cfgstr   = ibs.cfg.feat_cfg.get_cfgstr()
    sample_cfgstr = utool.hashstr_arr(aid_list, 'daids')
    cfgstr = '_' + sample_cfgstr + feat_cfgstr
    return cfgstr


#@utool.indent_func('[agg_desc]')
def aggregate_descriptors(ibs, aid_list):
    """ Aggregates descriptors with inverted information
     Return agg_index to(2) -> desc (descriptor)
                               aid (annotation rowid)
                               fx (feature index w.r.t. aid)
    </CYTH> """
    if not utool.QUIET:
        print('[agg_desc] stacking descriptors from %d annotations' % len(aid_list))
    desc_list = ibs.get_annot_desc(aid_list)
    # Build inverted index of (aid, fx) pairs
    aid_nFeat_iter = zip(aid_list, map(len, desc_list))
    nFeat_iter = map(len, desc_list)
    # generate aid inverted index for each feature in each annotation
    _ax2_aid = ([aid] * nFeat for (aid, nFeat) in aid_nFeat_iter)
    # generate featx inverted index for each feature in each annotation
    _ax2_fx  = (range(nFeat) for nFeat in nFeat_iter)
    # Flatten generators into the inverted index
    dx2_aid = np.array(list(chain.from_iterable(_ax2_aid)))
    dx2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    try:
        # Stack descriptors into numpy array corresponding to inverted inexed
        dx2_desc = np.vstack(desc_list)
        print('[agg_desc] stacked %d descriptors from %d annotations' % (len(dx2_desc), len(aid_list)))
    except MemoryError as ex:
        utool.printex(ex, 'cannot build inverted index', '[!memerror]')
        raise
    return dx2_desc, dx2_aid, dx2_fx


#@utool.indent_func('[build_invx]')
def build_flann_inverted_index(ibs, aid_list):
    """
    Build a inverted index (using FLANN)
    </CYTH> """
    try:
        if len(aid_list) == 0:
            msg = ('len(aid_list) == 0\n'
                    'Cannot build inverted index without features!')
            raise AssertionError(msg)
        dx2_desc, dx2_aid, dx2_fx = aggregate_descriptors(ibs, aid_list)
    except Exception as ex:
        intostr = ibs.get_infostr()  # NOQA
        dbname = ibs.get_dbname()  # NOQA
        num_images = ibs.get_num_images()  # NOQA
        num_annotations = ibs.get_num_annotations()      # NOQA
        num_names = ibs.get_num_names()    # NOQA
        utool.printex(ex, '', 'cannot build inverted index', locals().keys())
        raise
    # Build/Load the flann index
    flann_cfgstr = get_flann_cfgstr(ibs, aid_list)
    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    precomp_kwargs = {'cache_dir': ibs.get_flann_cachedir(),
                      'cfgstr': flann_cfgstr,
                      'flann_params': flann_params,
                      'force_recompute': NOCACHE_FLANN}
    flann = nntool.flann_cache(dx2_desc, **precomp_kwargs)
    return dx2_desc, dx2_aid, dx2_fx, flann


class NNIndex(object):
    """ Nearest Neighbor (FLANN) Index Class </CYTH> """
    def __init__(nn_index, ibs, daid_list):
        print('[nnindex] building NNIndex object')
        dx2_desc, dx2_aid, dx2_fx, flann = build_flann_inverted_index(ibs, daid_list)
        # Agg Data
        nn_index.dx2_aid  = dx2_aid
        nn_index.dx2_fx   = dx2_fx
        nn_index.dx2_data = dx2_desc
        # Grab the keypoints names and image ids before query time
        #nn_index.rx2_kpts = ibs.get_annot_kpts(daid_list)
        #nn_index.rx2_gid  = ibs.get_annot_gids(daid_list)
        #nn_index.rx2_nid  = ibs.get_annot_nids(daid_list)
        nn_index.flann = flann

    def __getstate__(nn_index):
        """ This class it not pickleable """
        #printDBG('get state NNIndex')
        return None

    #def __del__(nn_index):
    #    """ Ensure flann is propertly removed """
    #    printDBG('deleting NNIndex')
    #    if getattr(nn_index, 'flann', None) is not None:
    #        nn_index.flann.delete_index()
    #        #del nn_index.flann
    #    nn_index.flann = None

    def nn_index2(nn_index, qreq, qfx2_desc):
        """ return nearest neighbors from this data_index's flann object """
        flann   = nn_index.flann
        K       = qreq.cfg.nn_cfg.K
        Knorm   = qreq.cfg.nn_cfg.Knorm
        checks  = qreq.cfg.nn_cfg.checks

        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        qfx2_aid = nn_index.dx2_aid[qfx2_dx]
        qfx2_fx  = nn_index.dx2_fx[qfx2_dx]
        return qfx2_aid, qfx2_fx, qfx2_dist, K, Knorm


class NNSplitIndex(object):
    """ Nearest Neighbor (FLANN) Index Class </CYTH> """
    def __init__(split_index, ibs, daid_list, num_forests=8):
        print('[nnsindex] make NNSplitIndex over %d annots' % (len(daid_list),))
        aid_list = daid_list
        nid_list = ibs.get_annot_nids(aid_list)
        #flag_list = ibs.get_annot_exemplar_flag(aid_list)
        nid2_aids = utool.group_items(aid_list, nid_list)
        key_list = nid2_aids.keys()
        aids_list = nid2_aids.values()
        isunknown_list = ibs.is_nid_unknown(key_list)

        known_aids  = utool.filterfalse_items(aids_list, isunknown_list)
        uknown_aids = utool.flatten(utool.filter_items(aids_list, isunknown_list))

        num_forests_ = min(max(map(len, aids_list)), num_forests)

        # Put one name per forest
        forest_aids, overflow_aids = utool.sample_zip(known_aids, num_forests_,
                                                      allow_overflow=True,
                                                      per_bin=1)

        forest_indexes = []
        extra_indexes = []
        for tx, aids in enumerate(forest_aids):
            print('[nnsindex] building forest %d/%d with %d aids' % (tx + 1, num_forests_, len(aids)))
            if len(aids) > 0:
                nn_index = NNIndex(ibs, aids)
                forest_indexes.append(nn_index)

        if len(overflow_aids) > 0:
            print('[nnsindex] building overflow forest')
            overflow_index = NNIndex(ibs, overflow_aids)
            extra_indexes.append(overflow_index)
        if len(uknown_aids) > 0:
            print('[nnsindex] building unknown forest')
            unknown_index = NNIndex(ibs, uknown_aids)
            extra_indexes.append(unknown_index)
        #print('[nnsindex] building normalizer forest')  # TODO

        split_index.forest_indexes = forest_indexes
        split_index.extra_indexes = extra_indexes
        #split_index.overflow_index = overflow_index
        #split_index.unknown_index = unknown_index


@utool.classmember(NNSplitIndex)
def nn_index(split_index, qfx2_desc, num_neighbors):
    """ </CYTH> """
    qfx2_dx_list   = []
    qfx2_dist_list = []
    qfx2_aid_list  = []
    qfx2_fx_list   = []
    qfx2_rankx_list = []  # ranks index
    qfx2_treex_list = []  # tree index
    for tx, nn_index in enumerate(split_index.forest_indexes):
        flann = nn_index.flann
        # Returns distances in ascending order for each query descriptor
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, num_neighbors, checks=1024)
        qfx2_dx_list.append(qfx2_dx)
        qfx2_dist_list.append(qfx2_dist)
        qfx2_fx = nn_index.dx2_fx[qfx2_dx]
        qfx2_aid = nn_index.dx2_aid[qfx2_dx]
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


@utool.classmember(NNSplitIndex)
def split_index_daids(split_index):
    """ </CYTH> """
    for nn_index in split_index.forest_indexes:
        pass
