from __future__ import absolute_import, division, print_function
# Standard
from itertools import izip, chain, imap
# Science
import numpy as np
# UTool
import utool
# VTool
import vtool.nearest_neighbors as nntool
# IBEIS
from ibeis.dev import params
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[nnindex]', DEBUG=False)


@utool.indent_func
def get_flann_uid(ibs, cid_list):
    feat_uid   = ibs.cfg.feat_cfg.get_uid()
    sample_uid = utool.hashstr_arr(cid_list, 'dcids')
    uid = '_' + sample_uid + feat_uid
    return uid


@utool.indent_func
def aggregate_descriptors(ibs, cid_list):
    """ Aggregates descriptors with inverted information
     Return agg_index to(2) -> desc (descriptor)
                               cid (chip uid)
                               fx (feature index w.r.t. cid)
    """
    rid_list  = ibs.get_chip_rids(cid_list)  # TODO: RIDS are first order
    desc_list = ibs.get_roi_desc(rid_list)
    # Build inverted index of (cid, fx) pairs
    cid_nFeat_iter = izip(cid_list, imap(len, desc_list))
    nFeat_iter = imap(len, desc_list)
    # generate cid inverted index for each feature in each chip
    _ax2_cid = ([cid] * nFeat for (cid, nFeat) in cid_nFeat_iter)
    # generate featx inverted index for each feature in each chip
    _ax2_fx  = (xrange(nFeat) for nFeat in nFeat_iter)
    # Flatten generators into the inverted index
    ax2_cid = np.array(list(chain.from_iterable(_ax2_cid)))
    ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    try:
        # Stack descriptors into numpy array corresponding to inverted inexed
        ax2_desc = np.vstack(desc_list)
    except MemoryError as ex:
        utool.print_exception(ex, 'cannot build inverted index', '[!memerror]')
        raise
    return ax2_desc, ax2_cid, ax2_fx


@utool.indent_func
def build_flann_inverted_index(ibs, cid_list):
    try:
        ax2_desc, ax2_cid, ax2_fx = aggregate_descriptors(ibs, cid_list)
    except Exception as ex:
        intostr = ibs.get_infostr()  # NOQA
        utool.print_exception(ex, 'cannot build inverted index', '[!build_invx]',
                                  ['infostr'])
        raise
    # Build/Load the flann index
    flann_uid = get_flann_uid(ibs, cid_list)
    flann_params = {'algorithm': 'kdtree', 'trees': 4}
    precomp_kwargs = {'cache_dir': ibs.cachedir,
                      'uid': flann_uid,
                      'flann_params': flann_params,
                      'force_recompute': params.args.nocache_flann}
    flann = nntool.cached_flann(ax2_desc, **precomp_kwargs)
    return ax2_desc, ax2_cid, ax2_fx, flann


class NNIndex(object):
    """ Nearest Neighbor (FLANN) Index Class """
    def __init__(nn_index, ibs, dcid_list):
        print('[nnindex] building NNIndex object')
        drid_list = ibs.get_chip_rids(dcid_list)  # TODO: FIRST CLASS ROIs
        try:
            if len(dcid_list) == 0:
                raise AssertionError('Cannot build inverted index without features!')
            ax2_desc, ax2_cid, ax2_fx, flann = build_flann_inverted_index(ibs, dcid_list)
        except Exception as ex:
            dbname = ibs.get_dbname()  # NOQA
            num_images = ibs.get_num_images()  # NOQA
            num_rois = ibs.get_num_rois()      # NOQA
            num_names = ibs.get_num_names()    # NOQA
            utool.print_exception(ex, '', '[nn]', locals().keys())
            raise
        # Agg Data
        nn_index.ax2_cid  = ax2_cid
        nn_index.ax2_fx   = ax2_fx
        nn_index.ax2_data = ax2_desc

        # Grab the keypoints names and image ids before query time
        #nn_index.rx2_kpts = ibs.get_roi_kpts(drid_list)
        #nn_index.rx2_gid  = ibs.get_roi_gids(drid_list)
        #nn_index.rx2_nid  = ibs.get_roi_nids(drid_list)
        nn_index.flann = flann

    def __getstate__(nn_index):
        """ This class it not pickleable """
        printDBG('get state NNIndex')
        return None

    def __del__(nn_index):
        """ Ensure flann is propertly removed """
        printDBG('deleting NNIndex')
        if getattr(nn_index, 'flann', None) is not None:
            nn_index.flann.delete_index()
            nn_index.flann = None
