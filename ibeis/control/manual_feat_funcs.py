"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.control.template_generator')"
sh Tgen.sh --key feat --Tcfg with_setters=False with_getters=True  with_adders=True --modfname manual_feat_funcs
sh Tgen.sh --key feat --Tcfg with_deleters=True --autogen_modname manual_feat_funcs
"""
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

CONFIG_ROWID      = 'config_rowid'
FEAT_ROWID        = 'feature_rowid'


# ----------------
# ROOT LEAF FUNCTIONS
# ----------------


@register_ibs_method
@deleter
#@accessor_decors.dev_cache_invalidator(const.FEAT, 'feature_rowid')
def delete_annot_feats(ibs, aid_list, config2_=None):
    """ annot.feat.delete(aid_list)

    Args:
        aid_list

    TemplateInfo:
        Tdeleter_rl_depenant
        root = annot
        leaf = feat

    CommandLine:
        python -m ibeis.control.manual_feat_funcs --test-delete_annot_feats
        python -m ibeis.control.manual_feat_funcs --test-delete_annot_feats --verb-control

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[:1]
        >>> fids_list = ibs.get_annot_feat_rowids(aid_list, config2_=config2_, ensure=True)
        >>> num_deleted1 = ibs.delete_annot_feats(aid_list, config2_=config2_)
        >>> ut.assert_eq(num_deleted1, len(fids_list))
        >>> num_deleted2 = ibs.delete_annot_feats(aid_list, config2_=config2_)
        >>> ut.assert_eq(num_deleted2, 0)
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d annots leaf nodes' % len(aid_list))
    # Delete any dependants
    _feat_rowid_list = ibs.get_annot_feat_rowids(
        aid_list, config2_=config2_, ensure=False)
    feat_rowid_list = ut.filter_Nones(_feat_rowid_list)
    num_deleted = ibs.delete_feats(feat_rowid_list)
    return num_deleted


@register_ibs_method
@getter_1to1
def get_annot_feat_rowids(ibs, aid_list, ensure=True, eager=True, nInput=None, config2_=None):
    cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=ensure, eager=eager, nInput=nInput, config2_=config2_)
    fid_list = ibs.get_chip_feat_rowids(cid_list, ensure=ensure, eager=eager, nInput=nInput, config2_=config2_)
    return fid_list


@register_ibs_method
@ut.accepts_numpy
@getter_1toM
#@cache_getter(const.ANNOTATION_TABLE, 'kpts')
def get_annot_kpts(ibs, aid_list, ensure=True, eager=True, nInput=None,
                   config2_=None):
    """
    Args:
        aid_list (int):  list of annotation ids
        ensure (bool):  eager evaluation if True
        eager (bool):
        nInput (None):
        config2_ (QueryRequest):  query request object with hyper-parameters

    Returns:
        kpts_list (list): annotation descriptor keypoints

    CommandLine:
        python -m ibeis.control.manual_feat_funcs --test-get_annot_kpts --show
        python -m ibeis.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9
        python -m ibeis.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9 --verbose
        python -m ibeis.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9 --verbose --no-affine-invariance
        python -m ibeis.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9 --verbose --no-affine-invariance --scale_max=20

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
        >>> import vtool as vt
        >>> import numpy as np
        >>> import ibeis
        >>> # build test data
        >>> ibs, qreq1_ = plh.get_pipeline_testdata(defaultdb='testdb1', preload=False, cfgdict=dict(rotation_invariance=True))
        >>> ibs, qreq2_ = plh.get_pipeline_testdata(defaultdb='testdb1', preload=False, cfgdict=dict(rotation_invariance=False))
        >>> aid_list = qreq1_.get_external_qaids()
        >>> print('qreq1 params: ' + qreq1_.qparams.feat_cfgstr)
        >>> print('qreq2 params: ' + qreq2_.qparams.feat_cfgstr)
        >>> print('id(qreq1): ' + str(id(qreq1_)))
        >>> print('id(qreq2): ' + str(id(qreq2_)))
        >>> print('feat_config_rowid1 = %r' % (ibs.get_feat_config_rowid(config2_=qreq1_),))
        >>> print('feat_config_rowid2 = %r' % (ibs.get_feat_config_rowid(config2_=qreq2_),))
        >>> # Force recomputation of features
        >>> with ut.Indenter('[DELETE1]'):
        ...     ibs.delete_annot_feats(aid_list, config2_=qreq1_)
        >>> with ut.Indenter('[DELETE2]'):
        ...     ibs.delete_annot_feats(aid_list, config2_=qreq2_)
        >>> eager, ensure, nInput = True, True, None
        >>> # execute function
        >>> with ut.Indenter('[GET1]'):
        ...     kpts1_list = get_annot_kpts(ibs, aid_list, ensure, eager, nInput, qreq1_)
        >>> with ut.Indenter('[GET2]'):
        ...     kpts2_list = get_annot_kpts(ibs, aid_list, ensure, eager, nInput, qreq2_)
        >>> # verify results
        >>> assert not np.all(vt.get_oris(kpts1_list[0]) == 0)
        >>> assert np.all(vt.get_oris(kpts2_list[0]) == 0)
        >>> ut.quit_if_noshow()
        >>> #ibeis.viz.viz_chip.show_chip(ibs, aid_list[0], config2_=qreq1_, ori=True)
        >>> ibeis.viz.interact.interact_chip.ishow_chip(ibs, aid_list[0], config2_=qreq1_.qparams, ori=True, fnum=1)
        >>> ibeis.viz.interact.interact_chip.ishow_chip(ibs, aid_list[0], config2_=qreq2_.qparams, ori=True, fnum=2)
        >>> ut.show_if_requested()
    """
    fid_list  = ibs.get_annot_feat_rowids(aid_list, ensure=ensure, eager=eager, nInput=nInput, config2_=config2_)
    kpts_list = ibs.get_feat_kpts(fid_list, eager=eager, nInput=nInput)
    return kpts_list


@register_ibs_method
@getter_1toM
def get_annot_vecs(ibs, aid_list, ensure=True, eager=True, nInput=None,
                   config2_=None):
    """
    Returns:
        vecs_list (list): annotation descriptor vectors
    """
    fid_list  = ibs.get_annot_feat_rowids(aid_list, ensure=ensure, eager=eager, nInput=nInput, config2_=config2_)
    vecs_list = ibs.get_feat_vecs(fid_list, eager=eager, nInput=nInput)
    return vecs_list


@register_ibs_method
@getter_1to1
def get_annot_num_feats(ibs, aid_list, ensure=True, eager=True, nInput=None,
                        config2_=None):
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
    fid_list = ibs.get_annot_feat_rowids(aid_list, ensure=ensure, nInput=nInput, config2_=config2_)
    nFeats_list = ibs.get_num_feats(fid_list)
    return nFeats_list


# ----------------
# PARENT LEAF FUNCTIONS
# ----------------

#@register_ibs_method
#@adder
#def add_chip_feats(ibs, cid_list, force=False, config2_=None):
#    """ Computes the features for every chip without them """
#    from ibeis.model.preproc import preproc_feat
#    fid_list = ibs.get_chip_fids(cid_list, ensure=False, config2_=config2_)
#    dirty_cids = ut.get_dirty_items(cid_list, fid_list)
#    if len(dirty_cids) > 0:
#        #if ut.VERBOSE:
#        print('[ibs] adding %d / %d features' % (len(dirty_cids), len(cid_list)))
#        params_iter = preproc_feat.add_feat_params_gen(ibs, dirty_cids, config2_=config2_)
#        colnames = (CHIP_ROWID, 'config_rowid', FEAT_NUM_FEAT, FEAT_KPTS, FEAT_VECS)
#        get_rowid_from_superkey = functools.partial(ibs.get_chip_fids, ensure=False, config2_=config2_)
#        fid_list = ibs.dbcache.add_cleanly(const.FEATURE_TABLE, colnames, params_iter, get_rowid_from_superkey)

#    return fid_list


@register_ibs_method
@deleter
def delete_chip_feats(ibs, chip_rowid_list, config2_=None):
    """ chip.feat.delete(chip_rowid_list)

    Args:
        chip_rowid_list

    TemplateInfo:
        Tdeleter_rl_depenant
        parent = chip
        leaf = feat

    Example:
        >>> # ENABLE_DOCTEST
        >>> ibs, config2_ = testdata_ibs()
        >>> chip_rowid_list = ibs._get_all_chip_rowids()[::3]
        >>> ibs.delete_chip_feats(chip_rowid_list, config2_=config2_)
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d chips leaf nodes' % len(chip_rowid_list))
    # Delete any dependants
    _feat_rowid_list = ibs.get_chip_feat_rowids(
        chip_rowid_list, config2_=config2_, ensure=False)
    feat_rowid_list = ut.filter_Nones(_feat_rowid_list)
    num_deleted = ibs.delete_feats(feat_rowid_list)
    return num_deleted


@register_ibs_method
@adder
def add_chip_feats(ibs, chip_rowid_list, config2_=None, verbose=not ut.QUIET, return_num_dirty=False):
    """ chip.feat.add(chip_rowid_list)

    CRITICAL FUNCTION MUST EXIST FOR ALL DEPENDANTS
    Adds / ensures / computes a dependant property

    Args:
         chip_rowid_list

    Returns:
        returns feat_rowid_list of added (or already existing feats)

    TemplateInfo:
        Tadder_pl_dependant
        parent = chip
        leaf = feat

    CommandLine:
        python -m ibeis.control.manual_feat_funcs --test-add_chip_feats

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> chip_rowid_list = ibs._get_all_chip_rowids()[::3]
        >>> feat_rowid_list = ibs.add_chip_feats(chip_rowid_list, config2_=config2_)
        >>> assert len(feat_rowid_list) == len(chip_rowid_list)
        >>> ut.assert_all_not_None(feat_rowid_list)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> chip_rowid_list = ibs._get_all_chip_rowids()[0:10]
        >>> sub_chip_rowid_list1 = chip_rowid_list[0:6]
        >>> sub_chip_rowid_list2 = chip_rowid_list[5:7]
        >>> sub_chip_rowid_list3 = chip_rowid_list[0:7]
        >>> sub_feat_rowid_list1 = ibs.get_chip_feat_rowids(sub_chip_rowid_list1, config2_=config2_, ensure=True)
        >>> ibs.get_chip_feat_rowids(sub_chip_rowid_list1, config2_=config2_, ensure=True)
        >>> sub_feat_rowid_list1, num_dirty0 = ibs.add_chip_feats(sub_chip_rowid_list1, config2_=config2_, return_num_dirty=True)
        >>> assert num_dirty0 == 0
        >>> ut.assert_all_not_None(sub_feat_rowid_list1)
        >>> ibs.delete_chip_feats(sub_chip_rowid_list2)
        >>> #ibs.delete_chip_feat(sub_chip_rowid_list2)?
        >>> sub_feat_rowid_list3 = ibs.get_chip_feat_rowids(sub_chip_rowid_list3, config2_=config2_, ensure=False)
        >>> # Only the last two should be None
        >>> ut.assert_all_not_None(sub_feat_rowid_list3[0:5], 'sub_feat_rowid_list3[0:5])')
        >>> assert sub_feat_rowid_list3[5:7] == [None, None]
        >>> sub_feat_rowid_list3_ensured, num_dirty1 = ibs.add_chip_feats(sub_chip_rowid_list3, config2_=config2_,  return_num_dirty=True)
        >>> assert num_dirty1 == 2, 'Only two params should have been computed here'
        >>> ut.assert_all_not_None(sub_feat_rowid_list3_ensured)
    """
    from ibeis.model.preproc import preproc_feat
    ut.assert_all_not_None(chip_rowid_list, ' chip_rowid_list')
    # Get requested configuration id
    config_rowid = ibs.get_feat_config_rowid(config2_=config2_)
    # Find leaf rowids that need to be computed
    initial_feat_rowid_list = get_chip_feat_rowids_(
        ibs, chip_rowid_list, config2_=config2_)
    # Get corresponding "dirty" parent rowids
    isdirty_list = ut.flag_None_items(initial_feat_rowid_list)
    dirty_chip_rowid_list = ut.filter_items(chip_rowid_list, isdirty_list)
    num_dirty = len(dirty_chip_rowid_list)
    num_total = len(chip_rowid_list)
    if num_dirty > 0:
        if verbose:
            fmtstr = '[add_chip_feats] adding %d / %d new feat for config_rowid=%r'
            print(fmtstr % (num_dirty, num_total, config_rowid))
        # Dependant columns do not need true from_superkey getters.
        # We can use the Tgetter_pl_dependant_rowids_ instead
        get_rowid_from_superkey = functools.partial(
            ibs.get_chip_feat_rowids_, config2_=config2_)
        proptup_gen = preproc_feat.generate_feat_properties(
            ibs, dirty_chip_rowid_list, config2_=config2_)
        dirty_params_iter = (
            (chip_rowid, config_rowid, feature_nFeat,
             feature_kpt_arr, feature_vec_arr)
            for chip_rowid, (feature_nFeat, feature_kpt_arr, feature_vec_arr,) in
            zip(dirty_chip_rowid_list, proptup_gen)
        )
        colnames = ['chip_rowid', 'config_rowid',
                    'feature_num_feats', 'feature_keypoints', 'feature_vecs']
        #feat_rowid_list = ibs.dbcache.add_cleanly(const.FEATURE_TABLE, colnames, dirty_params_iter, get_rowid_from_superkey)
        ibs.dbcache._add(const.FEATURE_TABLE, colnames, dirty_params_iter)
        # Now that the dirty params are added get the correct order of rowids
        feat_rowid_list = get_rowid_from_superkey(chip_rowid_list)
    else:
        feat_rowid_list = initial_feat_rowid_list
    if return_num_dirty:
        return feat_rowid_list, num_dirty
    return feat_rowid_list


@register_ibs_method
@getter_1to1
#@accessor_decors.dev_cache_getter(const.CHIP_TABLE, 'feature_rowid')
def get_chip_feat_rowids(ibs, chip_rowid_list, config2_=None, ensure=False, eager=True, nInput=None):
    """ feat_rowid_list <- chip.feat.rowids[chip_rowid_list]

    get feat rowids of chip under the current state configuration
    if ensure is True, this function is equivalent to add_chip_feats

    Args:
        chip_rowid_list (list):
        ensure (bool): default false

    Returns:
        list: feat_rowid_list

    TemplateInfo:
        Tgetter_pl_dependant_rowids
        parent = chip
        leaf = feat

    Timeit:
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> # Test to see if there is any overhead to injected vs native functions
        >>> %timeit get_chip_feat_rowids(ibs, chip_rowid_list)
        >>> %timeit ibs.get_chip_feat_rowids(chip_rowid_list)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_feat_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> chip_rowid_list = ibs._get_all_chip_rowids()
        >>> ensure = False
        >>> feat_rowid_list = ibs.get_chip_feat_rowids(chip_rowid_list, config2_, ensure)
        >>> assert len(feat_rowid_list) == len(chip_rowid_list)
    """
    if ensure:
        feat_rowid_list = add_chip_feats(ibs, chip_rowid_list, config2_=config2_)
    else:
        feat_rowid_list = get_chip_feat_rowids_(
            ibs, chip_rowid_list, config2_=config2_, eager=eager, nInput=nInput)
    return feat_rowid_list


@register_ibs_method
@getter_1to1
def get_chip_feat_rowids_(ibs, chip_rowid_list, config2_=None, eager=True, nInput=None):
    """
    equivalent to get_chip_feat_rowids_ except ensure is constrained
    to be False.

    Also you save a stack frame because get_chip_feat_rowids just
    calls this function if ensure is False

    TemplateInfo:
        Tgetter_pl_dependant_rowids_
    """
    colnames = (FEAT_ROWID,)
    config_rowid = ibs.get_feat_config_rowid(config2_=config2_)
    andwhere_colnames = (CHIP_ROWID, CONFIG_ROWID,)
    params_iter = ((chip_rowid, config_rowid,)
                   for chip_rowid in chip_rowid_list)
    feat_rowid_list = ibs.dbcache.get_where2(
        const.FEATURE_TABLE, colnames, params_iter, andwhere_colnames, eager=eager, nInput=nInput)
    return feat_rowid_list


#@register_ibs_method
#def get_chip_feat_rowids(ibs, cid_list, ensure=True, eager=True, nInput=None, config2_=None):
#    # alias for get_chip_fids
#    return get_chip_fids(ibs, cid_list, ensure=ensure, eager=eager, nInput=nInput, config2_=config2_)


#@register_ibs_method
#@getter_1to1
#@accessor_decors.dev_cache_getter(const.CHIP_TABLE, 'feature_rowid')
#def get_chip_fids(ibs, cid_list, ensure=True, eager=True, nInput=None, config2_=None):
#    if ensure:
#        ibs.add_chip_feats(cid_list, config2_=config2_)
#    feat_config_rowid = ibs.get_feat_config_rowid(config2_=config2_)
#    colnames = ('feature_rowid',)
#    where_clause = CHIP_ROWID + '=? AND config_rowid=?'
#    params_iter = ((cid, feat_config_rowid) for cid in cid_list)
#    fid_list = ibs.dbcache.get_where(const.FEATURE_TABLE, colnames, params_iter,
#                                     where_clause, eager=eager,
#                                     nInput=nInput)
#    return fid_list


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
def get_valid_fids(ibs, config2_=None):
    """ Valid feature rowids of the current configuration """
    # FIXME: configids need reworking
    feat_config_rowid = ibs.get_feat_config_rowid(config2_=config2_)
    fid_list = ibs.dbcache.get_all_rowids_where(const.FEATURE_TABLE, 'config_rowid=?', (feat_config_rowid,))
    return fid_list


@register_ibs_method
@deleter
@accessor_decors.cache_invalidator(const.FEATURE_TABLE)
def delete_features(ibs, feat_rowid_list, config2_=None):
    """ deletes images from the database that belong to fids"""
    from ibeis.model.preproc import preproc_feat
    if ut.VERBOSE:
        print('[ibs] deleting %d features' % len(feat_rowid_list))
    # remove non-sql external dependeinces of these rowids
    preproc_feat.on_delete(ibs, feat_rowid_list)
    # remove dependants of these rowids
    featweight_rowid_list = ut.filter_Nones(ibs.get_feat_featweight_rowids(feat_rowid_list, config2_=config2_, ensure=False))
    ibs.delete_featweight(featweight_rowid_list)
    # remove these rowids
    ibs.dbcache.delete_rowids(const.FEATURE_TABLE, feat_rowid_list)
    num_deleted = len(ut.filter_Nones(feat_rowid_list))
    return num_deleted


@register_ibs_method
@deleter
@accessor_decors.cache_invalidator(const.FEATURE_TABLE)
def delete_feats(ibs, feat_rowid_list, config2_=None):
    """ alias """
    num_deleted = delete_features(ibs, feat_rowid_list, config2_=config2_)
    return num_deleted


@register_ibs_method
@default_decorator
def get_feat_config_rowid(ibs, config2_=None):
    """
    Returns the feature configuration id based on the cfgstr
    defined by ibs.cfg.feat_cfg.get_cfgstr()

    # FIXME: Configs are still handled poorly
    used in ibeis.model.preproc.preproc_feats in the param
    generator. (that should probably be moved into the controller)
    """
    if config2_ is not None:
        # TODO store config_rowid in qparams
        # Or find better way to do this in general
        #feat_cfg_suffix = config2_.qparams.feat_cfgstr
        #feat_cfg_suffix = config2_.qparams.feat_cfgstr
        feat_cfg_suffix = config2_.get('feat_cfgstr')
        assert feat_cfg_suffix is not None
    else:
        feat_cfg_suffix = ibs.cfg.feat_cfg.get_cfgstr()
    #print(feat_cfg_suffix)
    #print(config2_)
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


def testdata_ibs():
    import ibeis
    ibs = ibeis.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_feat_funcs
        python -m ibeis.control.manual_feat_funcs --allexamples
        python -m ibeis.control.manual_feat_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
