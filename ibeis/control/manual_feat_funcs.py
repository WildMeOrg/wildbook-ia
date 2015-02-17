from __future__ import absolute_import, division, print_function
import six  # NOQA
import functools
from ibeis import constants as const
from ibeis.control import accessor_decors
from ibeis.control.accessor_decors import (adder, ider, default_decorator,
                                           getter_1to1, getter_1toM, deleter)
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_feats]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


ANNOT_ROWID   = 'annot_rowid'
CHIP_ROWID    = 'chip_rowid'
FEAT_VECS     = 'feature_vecs'
FEAT_KPTS     = 'feature_keypoints'
FEAT_NUM_FEAT = 'feature_num_feats'


# ----------------
# ROOT LEAF FUNCTIONS
# ----------------


@register_ibs_method
@getter_1to1
def get_annot_feat_rowids(ibs, aid_list, ensure=False, eager=True, nInput=None, qreq_=None):
    cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=ensure, eager=eager, nInput=nInput, qreq_=qreq_)
    fid_list = ibs.get_chip_fids(cid_list, ensure=ensure, eager=eager, nInput=nInput)
    return fid_list


@register_ibs_method
@ut.accepts_numpy
@getter_1toM
#@cache_getter(const.ANNOTATION_TABLE, 'kpts')
def get_annot_kpts(ibs, aid_list, ensure=True, eager=True, nInput=None,
                   qreq_=None):
    """
    Args:
        aid_list (list):

    Returns:
        kpts_list (list): annotation descriptor keypoints
    """
    fid_list  = ibs.get_annot_feat_rowids(aid_list, ensure=ensure, eager=eager, nInput=nInput, qreq_=qreq_)
    kpts_list = ibs.get_feat_kpts(fid_list, eager=eager, nInput=nInput)
    return kpts_list


@register_ibs_method
@getter_1toM
def get_annot_vecs(ibs, aid_list, ensure=True, eager=True, nInput=None,
                   qreq_=None):
    """
    Returns:
        vecs_list (list): annotation descriptor vectors
    """
    fid_list  = ibs.get_annot_feat_rowids(aid_list, ensure=ensure, eager=eager, nInput=nInput, qreq_=qreq_)
    vecs_list = ibs.get_feat_vecs(fid_list, eager=eager, nInput=nInput)
    return vecs_list


@register_ibs_method
@getter_1to1
def get_annot_num_feats(ibs, aid_list, ensure=False, eager=True, nInput=None,
                        qreq_=None):
    """
    Args:
        aid_list (list):

    Returns:
        nFeats_list (list): num descriptors per annotation

    CommandLine:
        python -m ibeis.control.manual_feat_funcs --test-get_annot_num_feats

    Example:
        >>> # ENABLE_DOCTEST
        >>> # this test might fail on different machines due to
        >>> # determenism bugs in hesaff maybe? or maybe jpeg...
        >>> # in which case its hopeless
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:3]
        >>> nFeats_list = get_annot_num_feats(ibs, aid_list, ensure=True)
        >>> assert len(nFeats_list) == 3
        >>> ut.assert_inbounds(nFeats_list[0], 1256, 1258)
        >>> ut.assert_inbounds(nFeats_list[1],  910,  921)
        >>> ut.assert_inbounds(nFeats_list[2], 1340, 1343)

    [1257, 920, 1342]
    """
    fid_list = ibs.get_annot_feat_rowids(aid_list, ensure=ensure, nInput=nInput, qreq_=qreq_)
    nFeats_list = ibs.get_num_feats(fid_list)
    return nFeats_list


# ----------------
# PARENT LEAF FUNCTIONS
# ----------------

@register_ibs_method
@adder
def add_chip_feats(ibs, cid_list, force=False, qreq_=None):
    """ Computes the features for every chip without them """
    from ibeis.model.preproc import preproc_feat
    fid_list = ibs.get_chip_fids(cid_list, ensure=False, qreq_=qreq_)
    dirty_cids = ut.get_dirty_items(cid_list, fid_list)
    if len(dirty_cids) > 0:
        if ut.VERBOSE:
            print('[ibs] adding %d / %d features' % (len(dirty_cids), len(cid_list)))
        params_iter = preproc_feat.add_feat_params_gen(ibs, dirty_cids, qreq_=qreq_)
        colnames = (CHIP_ROWID, 'config_rowid', FEAT_NUM_FEAT, FEAT_KPTS, FEAT_VECS)
        get_rowid_from_superkey = functools.partial(ibs.get_chip_fids, ensure=False, qreq_=qreq_)
        fid_list = ibs.dbcache.add_cleanly(const.FEATURE_TABLE, colnames, params_iter, get_rowid_from_superkey)

    return fid_list


@register_ibs_method
def get_chip_feat_rowids(ibs, cid_list, ensure=True, eager=True, nInput=None, qreq_=None):
    # alias for get_chip_fids
    return get_chip_fids(ibs, cid_list, ensure=ensure, eager=eager, nInput=nInput, qreq_=qreq_)


@register_ibs_method
@getter_1to1
@accessor_decors.dev_cache_getter(const.CHIP_TABLE, 'feature_rowid')
def get_chip_fids(ibs, cid_list, ensure=True, eager=True, nInput=None, qreq_=None):
    if ensure:
        ibs.add_chip_feats(cid_list, qreq_=qreq_)
    feat_config_rowid = ibs.get_feat_config_rowid(qreq_=qreq_)
    colnames = ('feature_rowid',)
    where_clause = CHIP_ROWID + '=? AND config_rowid=?'
    params_iter = ((cid, feat_config_rowid) for cid in cid_list)
    fid_list = ibs.dbcache.get_where(const.FEATURE_TABLE, colnames, params_iter,
                                     where_clause, eager=eager,
                                     nInput=nInput)
    return fid_list


# ----------------
# NATIVE FUNCTIONS
# ----------------


@register_ibs_method
@ider
def _get_all_fids(ibs):
    """ alias """
    return _get_all_feat_rowids(ibs)


@register_ibs_method
def _get_all_feat_rowids(ibs):
    """
    Returns:
        list_ (list): unfiltered fids (computed feature rowids) for every
    configuration (YOU PROBABLY SHOULD NOT USE THIS)"""
    all_fids = ibs.dbcache.get_all_rowids(const.FEATURE_TABLE)
    return all_fids


@register_ibs_method
@ider
def get_valid_fids(ibs, qreq_=None):
    """ Valid feature rowids of the current configuration """
    # FIXME: configids need reworking
    feat_config_rowid = ibs.get_feat_config_rowid(qreq_=qreq_)
    fid_list = ibs.dbcache.get_all_rowids_where(const.FEATURE_TABLE, 'config_rowid=?', (feat_config_rowid,))
    return fid_list


@register_ibs_method
@deleter
@accessor_decors.cache_invalidator(const.FEATURE_TABLE)
def delete_features(ibs, fid_list):
    """ deletes images from the database that belong to fids"""
    if ut.VERBOSE:
        print('[ibs] deleting %d features' % len(fid_list))
    ibs.dbcache.delete_rowids(const.FEATURE_TABLE, fid_list)


@register_ibs_method
@default_decorator
def get_feat_config_rowid(ibs, qreq_=None):
    """
    Returns the feature configuration id based on the cfgstr
    defined by ibs.cfg.feat_cfg.get_cfgstr()

    # FIXME: Configs are still handled poorly
    used in ibeis.model.preproc.preproc_feats in the param
    generator. (that should probably be moved into the controller)
    """
    if qreq_ is not None:
        # TODO store config_rowid in qparams
        # Or find better way to do this in general
        feat_cfg_suffix = qreq_.qparams.feat_cfgstr
    else:
        feat_cfg_suffix = ibs.cfg.feat_cfg.get_cfgstr()
    feat_cfg_rowid = ibs.add_config(feat_cfg_suffix)
    return feat_cfg_rowid


@register_ibs_method
@getter_1toM
@accessor_decors.cache_getter(const.FEATURE_TABLE, FEAT_KPTS)
def get_feat_kpts(ibs, fid_list, eager=True, nInput=None):
    """
    Returns:
        kpts_list (list): chip keypoints in [x, y, iv11, iv21, iv22, ori] format
    """
    kpts_list = ibs.dbcache.get(const.FEATURE_TABLE, (FEAT_KPTS,), fid_list, eager=eager, nInput=nInput)
    return kpts_list


@register_ibs_method
@getter_1toM
@accessor_decors.cache_getter(const.FEATURE_TABLE, FEAT_VECS)
def get_feat_vecs(ibs, fid_list, eager=True, nInput=None):
    """
    Returns:
        vecs_list (list): chip SIFT descriptors
    """
    vecs_list = ibs.dbcache.get(const.FEATURE_TABLE, (FEAT_VECS,), fid_list, eager=eager, nInput=nInput)
    return vecs_list


@register_ibs_method
@getter_1to1
@accessor_decors.cache_getter(const.FEATURE_TABLE, FEAT_NUM_FEAT)
def get_num_feats(ibs, fid_list, eager=True, nInput=None):
    """
    Returns:
        nFeats_list (list): the number of keypoint / descriptor pairs
    """
    nFeats_list = ibs.dbcache.get(const.FEATURE_TABLE, (FEAT_NUM_FEAT,), fid_list, eager=True, nInput=None)
    nFeats_list = [(-1 if nFeats is None else nFeats) for nFeats in nFeats_list]
    return nFeats_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_chip_funcs
        python -m ibeis.control.manual_chip_funcs --allexamples
        python -m ibeis.control.manual_chip_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
