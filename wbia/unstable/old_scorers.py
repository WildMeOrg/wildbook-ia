# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import utool as ut
import numpy as np
from wbia.algo.hots.nn_weights import (
    _register_nn_simple_weight_func,
    _register_misc_weight_func,
)

(print, rrr, profile) = ut.inject2(__name__)


def get_annot_kpts_baseline_weights(ibs, aid_list, config2_=None, config={}):
    r"""
    Returns weights based on distinctiveness and/or features score / or ones.  Customized based on config.

    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        aid_list (int):  list of annotation ids
        config (dict):

    Returns:
        list: weights_list

    CommandLine:
        python -m wbia.algo.hots.scoring --test-get_annot_kpts_baseline_weights

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring('testdb1')
        >>> aid_list = cm.daid_list
        >>> config = qreq_.qparams
        >>> config2_ = qreq_.qparams
        >>> kpts_list = qreq_.ibs.get_annot_kpts(aid_list, config2_=config2_)
        >>> weights_list = get_annot_kpts_baseline_weights(qreq_.ibs, aid_list, config2_, config)
        >>> depth1 = ut.get_list_column(ut.depth_profile(kpts_list), 0)
        >>> depth2 = ut.depth_profile(weights_list)
        >>> assert depth1 == depth2
        >>> print(depth1)
        >>> result = str(depth2)
        >>> print(result)
    """
    import scipy as sp

    # TODO: clip the fgweights? (dilation?)
    # TODO; normalize and paramatarize and clean
    # dcvs_on = config.get('dcvs_on')
    fg_on = config.get('fg_on')
    weight_lists = []
    # if dcvs_on:
    #     raise NotImplementedError('dcvs')
    #     # qdstncvs_list = get_kpts_distinctiveness(ibs, aid_list, config2_, config)
    #     # weight_lists.append(qdstncvs_list)
    if fg_on:
        qfgweight_list = ibs.get_annot_fgweights(aid_list, ensure=True, config2_=config2_)
        weight_lists.append(qfgweight_list)
    if len(weight_lists) == 0:
        baseline_weights_list = [
            np.ones(num, np.float)
            for num in ibs.get_annot_num_feats(aid_list, config2_=config2_)
        ]
        # baseline_weights_list = [None] * len(aid_list)
    else:
        # geometric mean of the selected weights
        baseline_weights_list = [
            sp.stats.gmean(weight_tup) for weight_tup in zip(*weight_lists)
        ]
    return baseline_weights_list


def get_mask_func(config):
    from vtool import coverage_kpts, coverage_grid

    # DEPRICATE

    # maskscore_mode = config.get('maskscore_mode', 'grid')
    maskscore_mode = 'grid'
    # print(maskscore_mode)
    FUNC_ARGS_DICT = {
        'grid': (coverage_grid.make_grid_coverage_mask, coverage_grid.COVGRID_DEFAULT),
        'kpts': (coverage_kpts.make_kpts_coverage_mask, coverage_kpts.COVKPTS_DEFAULT),
    }
    make_mask_func, func_defaults = FUNC_ARGS_DICT[maskscore_mode]
    # cov_cfg = func_defaults.updated_cfgdict(config)
    # Hack to make kwargs happy
    cov_cfg = ut.filter_valid_kwargs(make_mask_func, config)
    return make_mask_func, cov_cfg


# ### MASK COVERAGE SCORING ####


def compute_annot_coverage_score(qreq_, cm, config={}):
    """
    CommandLine:
        python -m wbia.algo.hots.scoring --test-compute_annot_coverage_score:0

    Example0:
        >>> # SLOW_DOCTEST
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring()
        >>> config = qreq_.qparams
        >>> daid_list, score_list = compute_annot_coverage_score(qreq_, cm, config)
        >>> ut.assert_inbounds(np.array(score_list), 0, 1, eq=True)
        >>> result = ut.repr2(score_list, precision=3)
        >>> print(result)
    """
    # DEPRICATE
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_annot_coverage_mask_generator(
        make_mask_func, qreq_, cm, config, cov_cfg
    )
    daid_list, score_list = score_masks(masks_iter)
    return daid_list, score_list


def compute_name_coverage_score(qreq_, cm, config={}):
    """
    CommandLine:
        python -m wbia.algo.hots.scoring --test-compute_name_coverage_score:0

    Example0:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring()
        >>> cm.evaluate_dnids(qreq_)
        >>> config = qreq_.qparams
        >>> dnid_list, score_list = compute_name_coverage_score(qreq_, cm, config)
        >>> ut.assert_inbounds(np.array(score_list), 0, 1, eq=True)
        >>> result = ut.repr2(score_list, precision=3)
        >>> print(result)
    """
    # DEPRICATE
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_name_coverage_mask_generator(
        make_mask_func, qreq_, cm, config, cov_cfg
    )
    dnid_list, score_list = score_masks(masks_iter)
    return dnid_list, score_list


def score_masks(masks_iter):
    # DEPRICATE
    id_score_list = [
        (id_, score_matching_mask(weight_mask_m, weight_mask))
        for id_, weight_mask_m, weight_mask in masks_iter
    ]
    id_list = np.array(ut.get_list_column(id_score_list, 0))
    coverage_score = np.array(ut.get_list_column(id_score_list, 1))
    return id_list, coverage_score


def score_matching_mask(weight_mask_m, weight_mask):
    # DEPRICATE
    coverage_score = weight_mask_m.sum() / weight_mask.sum()
    return coverage_score


def general_annot_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg):
    """
    DEPRICATE

    Yeilds:
        daid, weight_mask_m, weight_mask

    CommandLine:
        python -m wbia.algo.hots.scoring --test-general_annot_coverage_mask_generator --show
        python -m wbia.algo.hots.scoring --test-general_annot_coverage_mask_generator --show --qaid 18

    Note:
        Evaluate output one at a time or it will get clobbered

    Example0:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring('PZ_MTEST', qaid_list=[18])
        >>> config = qreq_.qparams
        >>> make_mask_func, cov_cfg = get_mask_func(config)
        >>> masks_iter = general_annot_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg)
        >>> daid_list, score_list, masks_list = evaluate_masks_iter(masks_iter)
        >>> #assert daid_list[idx] ==
        >>> ut.quit_if_noshow()
        >>> idx = score_list.argmax()
        >>> daids = [daid_list[idx]]
        >>> daid, weight_mask_m, weight_mask = masks_list[idx]
        >>> show_single_coverage_mask(qreq_, cm, weight_mask_m, weight_mask, daids)
        >>> ut.show_if_requested()
    """
    if ut.VERYVERBOSE:
        print('[acov] make_mask_func = %r' % (make_mask_func,))
        print('[acov] cov_cfg = %s' % (ut.repr2(cov_cfg),))
    return general_coverage_mask_generator(
        make_mask_func,
        qreq_,
        cm.qaid,
        cm.daid_list,
        cm.fm_list,
        cm.fs_list,
        config,
        cov_cfg,
    )


def general_name_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg):
    """
    DEPRICATE

    Yeilds:
        nid, weight_mask_m, weight_mask

    CommandLine:
        python -m wbia.algo.hots.scoring --test-general_name_coverage_mask_generator --show
        python -m wbia.algo.hots.scoring --test-general_name_coverage_mask_generator --show --qaid 18

    Note:
        Evaluate output one at a time or it will get clobbered

    Example0:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring('PZ_MTEST', qaid_list=[18])
        >>> config = qreq_.qparams
        >>> make_mask_func, cov_cfg = get_mask_func(config)
        >>> masks_iter = general_name_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg)
        >>> dnid_list, score_list, masks_list = evaluate_masks_iter(masks_iter)
        >>> ut.quit_if_noshow()
        >>> nidx = np.where(dnid_list == cm.qnid)[0][0]
        >>> daids = cm.get_groundtruth_daids()
        >>> dnid, weight_mask_m, weight_mask = masks_list[nidx]
        >>> show_single_coverage_mask(qreq_, cm, weight_mask_m, weight_mask, daids)
        >>> ut.show_if_requested()
    """
    import vtool as vt

    if ut.VERYVERBOSE:
        print('[ncov] make_mask_func = %r' % (make_mask_func,))
        print('[ncov] cov_cfg = %s' % (ut.repr2(cov_cfg),))
    assert cm.dnid_list is not None, 'eval nids'
    unique_dnids, groupxs = vt.group_indices(cm.dnid_list)
    fm_groups = vt.apply_grouping_(cm.fm_list, groupxs)
    fs_groups = vt.apply_grouping_(cm.fs_list, groupxs)
    fs_name_list = [np.hstack(fs_group) for fs_group in fs_groups]
    fm_name_list = [np.vstack(fm_group) for fm_group in fm_groups]
    return general_coverage_mask_generator(
        make_mask_func,
        qreq_,
        cm.qaid,
        unique_dnids,
        fm_name_list,
        fs_name_list,
        config,
        cov_cfg,
    )


def general_coverage_mask_generator(
    make_mask_func, qreq_, qaid, id_list, fm_list, fs_list, config, cov_cfg
):
    """ agnostic to whether or not the id/fm/fs lists are name or annotation groups

    DEPRICATE
    """
    if ut.VERYVERBOSE:
        print('[acov] make_mask_func = %r' % (make_mask_func,))
        print('[acov] cov_cfg = %s' % (ut.repr2(cov_cfg),))
    # Distinctivness and foreground weight
    qweights = get_annot_kpts_baseline_weights(
        qreq_.ibs, [qaid], config2_=qreq_.extern_query_config2, config=config
    )[0]
    # Denominator weight mask
    chipsize = qreq_.ibs.get_annot_chip_sizes(qaid, config2_=qreq_.extern_query_config2)
    qkpts = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.extern_query_config2)
    weight_mask = make_mask_func(qkpts, chipsize, qweights, resize=False, **cov_cfg)
    # Prealloc data for loop
    weight_mask_m = weight_mask.copy()
    # Apply weighted scoring to matches
    for daid, fm, fs in zip(id_list, fm_list, fs_list):
        # CAREFUL weight_mask_m is overriden on every iteration
        weight_mask_m = compute_general_matching_coverage_mask(
            make_mask_func, chipsize, fm, fs, qkpts, qweights, cov_cfg, out=weight_mask_m,
        )
        yield daid, weight_mask_m, weight_mask


def compute_general_matching_coverage_mask(
    make_mask_func, chipsize, fm, fs, qkpts, qweights, cov_cfg, out=None
):
    """
    DEPRICATE
    """
    # Get matching query keypoints
    # SYMMETRIC = False
    # if SYMMETRIC:
    #    get_annot_kpts_baseline_weights()
    qkpts_m = qkpts.take(fm.T[0], axis=0)
    weights_m = fs * qweights.take(fm.T[0], axis=0)
    # hacky buisness
    # weights_m  = qweights.take(fm.T[0], axis=0)
    # weights_m = fs
    weight_mask_m = make_mask_func(
        qkpts_m, chipsize, weights_m, out=out, resize=False, **cov_cfg
    )
    return weight_mask_m


def get_masks(qreq_, cm, config={}):
    r"""
    testing function
    DEPRICATE

    CommandLine:
        # SHOW THE BASELINE AND MATCHING MASKS
        python -m wbia.algo.hots.scoring --test-get_masks
        python -m wbia.algo.hots.scoring --test-get_masks \
            --maskscore_mode=kpts --show --prior_coeff=.5 --unconstrained_coeff=.3 --constrained_coeff=.2
        python -m wbia.algo.hots.scoring --test-get_masks \
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5
        python -m wbia.algo.hots.scoring --test-get_masks --qaid 4\
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5
        python -m wbia.algo.hots.scoring --test-get_masks --qaid 86\
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5 --grid_scale_factor=.5

        python -m wbia.algo.hots.scoring --test-get_masks --show --db PZ_MTEST --qaid 18
        python -m wbia.algo.hots.scoring --test-get_masks --show --db PZ_MTEST --qaid 1

    Example:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> import wbia
        >>> qreq_, cm = plh.testdata_scoring('PZ_MTEST', qaid_list=[18])
        >>> config = qreq_.qparams
        >>> id_list, score_list, masks_list = get_masks(qreq_, cm, config)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> show_coverage_mask(qreq_, cm, masks_list, index=score_list.argmax())
        >>> pt.show_if_requested()
    """
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_annot_coverage_mask_generator(
        make_mask_func, qreq_, cm, config, cov_cfg
    )
    # copy weight mask as it comes back if you want to see them
    id_list, score_list, masks_list = evaluate_masks_iter(masks_iter)

    return id_list, score_list, masks_list


def evaluate_masks_iter(masks_iter):
    """ save evaluation of a masks iter
    DEPRICATE
    """
    masks_list = [
        (id_, weight_mask_m.copy(), weight_mask)
        for id_, weight_mask_m, weight_mask in masks_iter
    ]
    id_list, score_list = score_masks(masks_list)
    return id_list, score_list, masks_list


def show_coverage_mask(qreq_, cm, masks_list, index=0, fnum=None):
    """
    DEPRICATE
    """
    daid, weight_mask_m, weight_mask = masks_list[index]
    daids = [daid]
    show_single_coverage_mask(qreq_, cm, weight_mask_m, weight_mask, daids, fnum=None)


def show_single_coverage_mask(qreq_, cm, weight_mask_m, weight_mask, daids, fnum=None):
    """
    DEPRICATE
    """
    import wbia.plottool as pt
    import vtool as vt
    from wbia import viz

    fnum = pt.ensure_fnum(fnum)
    idx_list = ut.dict_take(cm.daid2_idx, daids)
    nPlots = len(idx_list) + 1
    nRows, nCols = pt.get_square_row_cols(nPlots)
    pnum_ = pt.make_pnum_nextgen(nRows, nCols)
    pt.figure(fnum=fnum, pnum=(1, 2, 1))
    # Draw coverage masks with bbox
    # <FlipHack>
    # weight_mask_m = np.fliplr(np.flipud(weight_mask_m))
    # weight_mask = np.fliplr(np.flipud(weight_mask))
    # </FlipHack>
    stacked_weights, offset_tup, sf_tup = vt.stack_images(
        weight_mask_m, weight_mask, return_sf=True
    )
    (woff, hoff) = offset_tup[1]
    wh1 = weight_mask_m.shape[0:2][::-1]
    wh2 = weight_mask.shape[0:2][::-1]
    pt.imshow(
        255 * (stacked_weights),
        fnum=fnum,
        pnum=pnum_(0),
        title='(query image) What did match vs what should match',
    )
    pt.draw_bbox((0, 0) + wh1, bbox_color=(0, 0, 1))
    pt.draw_bbox((woff, hoff) + wh2, bbox_color=(0, 0, 1))
    # Get contributing matches
    qaid = cm.qaid
    daid_list = daids
    fm_list = ut.take(cm.fm_list, idx_list)
    fs_list = ut.take(cm.fs_list, idx_list)
    # Draw matches
    for px, (daid, fm, fs) in enumerate(zip(daid_list, fm_list, fs_list), start=1):
        viz.viz_matches.show_matches2(
            qreq_.ibs,
            qaid,
            daid,
            fm,
            fs,
            draw_pts=False,
            draw_lines=True,
            draw_ell=False,
            fnum=fnum,
            pnum=pnum_(px),
            darken=0.5,
        )
    coverage_score = score_matching_mask(weight_mask_m, weight_mask)
    pt.set_figtitle('score=%.4f' % (coverage_score,))


def show_annot_weights(qreq_, aid, config={}):
    r"""
    DEMO FUNC
    DEPRICATE

    CommandLine:
        python -m wbia.algo.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --maskscore_mode='grid'
        python -m wbia.algo.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --maskscore_mode='kpts'
        python -m wbia.algo.hots.scoring --test-show_annot_weights --show --db PZ_Master0 --aid 1
        python -m wbia.algo.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_()
        >>> ibs = qreq_.ibs
        >>> aid = qreq_.qaids[0]
        >>> config = qreq_.qparams
        >>> show_annot_weights(qreq_, aid, config)
        >>> pt.show_if_requested()
    """
    from vtool import coverage_kpts

    # import wbia.plottool as pt
    fnum = 1
    chipsize = qreq_.ibs.get_annot_chip_sizes(aid, config2_=qreq_.extern_query_config2)
    chip = qreq_.ibs.get_annot_chips(aid, config2_=qreq_.extern_query_config2)
    qkpts = qreq_.ibs.get_annot_kpts(aid, config2_=qreq_.extern_query_config2)
    weights = get_annot_kpts_baseline_weights(
        qreq_.ibs, [aid], config2_=qreq_.extern_query_config2, config=config
    )[0]
    make_mask_func, cov_cfg = get_mask_func(config)
    mask = make_mask_func(qkpts, chipsize, weights, resize=True, **cov_cfg)
    coverage_kpts.show_coverage_map(
        chip, mask, None, qkpts, fnum, ell_alpha=0.2, show_mask_kpts=False
    )
    # pt.set_figtitle(mode)


# ### FEATURE WEIGHTS ####
# TODO: qreq_


def sift_selectivity_score(vecs1_m, vecs2_m, cos_power=3.0, dtype=np.float):
    import vtool as vt
    from wbia.algo.hots import hstypes

    """
    applies selectivity score from SMK paper
    Take componentwise dot produt and divide by 512**2 because of the
    sift descriptor uint8 trick
    """
    # Compute dot product (cosine of angle between sift descriptors)
    cosangle = vt.componentwise_dot(vecs1_m.astype(dtype), vecs2_m.astype(dtype))
    # Adjust for uint8 trick
    cosangle /= hstypes.PSEUDO_UINT8_MAX_SQRD
    # apply selectivity functiodictin
    selectivity_score = np.power(cosangle, cos_power)
    # If cosine can be less than 0 replace previous line with next line
    # or just write an rvec_selectivity_score function
    # selectivity_score = np.multiply(np.sign(cosangle), np.power(cosangle, cos_power))
    return selectivity_score


@_register_nn_simple_weight_func
def cos_match_weighter(nns_list, nnvalid0_list, qreq_):
    r"""
    Uses smk-like selectivity function. Need to gridsearch for a good alpha.

    CommandLine:
        python -m wbia.algo.hots.nn_weights --test-cos_match_weighter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> from wbia.algo.hots import nn_weights
        >>> #tup = plh.testdata_pre_weight_neighbors('PZ_MTEST', cfgdict=dict(cos_on=True, K=5, Knorm=5))
        >>> #ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='PZ_MTEST', p=['default:cos_on=True,K=5,Knorm=5'])
        >>> nns_list, nnvalid0_list = args
        >>> assert qreq_.qparams.cos_on, 'bug setting custom params cos_weight'
        >>> cos_weight_list = nn_weights.cos_match_weighter(nns_list, nnvalid0_list, qreq_)
    """
    from wbia.algo.hots import scoring

    Knorm = qreq_.qparams.Knorm
    cos_weight_list = []
    qconfig2_ = qreq_.get_internal_query_config2()
    # Database feature index to chip index
    for nns in nns_list:
        qfx2_qvec = qreq_.ibs.get_annot_vecs(nns.qaid, config2_=qconfig2_)
        qvecs = qfx2_qvec.take(nns.qfx_list, axis=0)
        qvecs = qvecs[np.newaxis, :, :]
        # database forground weights
        # avoid using K due to its more dynamic nature by using -Knorm
        dvecs = qreq_.indexer.get_nn_vecs(nns.neighb_idxs.T[:-Knorm])
        # Component-wise dot product + selectivity function
        alpha = 3.0
        neighb_cosweight = scoring.sift_selectivity_score(qvecs, dvecs, alpha)
        cos_weight_list.append(neighb_cosweight)
    return cos_weight_list


@_register_misc_weight_func
def distinctiveness_match_weighter(qreq_):
    r"""
    TODO: finish intergration

    # Example:
    #     >>> # SLOW_DOCTEST
    #     >>> from wbia.algo.hots.nn_weights import *  # NOQA
    #     >>> from wbia.algo.hots import nn_weights
    #     >>> tup = plh.testdata_pre_weight_neighbors('PZ_MTEST', codename='vsone_dist_extern_distinctiveness')
    #     >>> ibs, qreq_, nns_list, nnvalid0_list = tup
    """
    raise NotImplementedError('Not finished')
    dstcnvs_normer = qreq_.dstcnvs_normer
    assert dstcnvs_normer is not None
    qaid_list = qreq_.qaids
    vecs_list = qreq_.ibs.get_annot_vecs(
        qaid_list, config2_=qreq_.get_internal_query_config2()
    )
    # TODO: need to filter on nn.fxs_list if features were threshed away
    dstcvs_list = []
    for vecs in vecs_list:
        qfx2_vec = vecs
        dstcvs = dstcnvs_normer.get_distinctiveness(qfx2_vec)
        dstcvs_list.append(dstcvs)
    return dstcvs_list


@_register_nn_simple_weight_func
def borda_match_weighter(nns_list, nnvalid0_list, qreq_):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.nn_weights import *  # NOQA
        >>> #tup = plh.testdata_pre_weight_neighbors('PZ_MTEST')
        >>> #ibs, qreq_, nns_list, nnvalid0_list = tup
        >>> qreq_, args = plh.testdata_pre('weight_neighbors', defaultdb='PZ_MTEST')
        >>> nns_list, nnvalid0_list = args
        >>> bordavote_weight_list = borda_match_weighter(nns_list, nnvalid0_list, qreq_)
        >>> result = ('bordavote_weight_list = %s' % (str(bordavote_weight_list),))
        >>> print(result)
    """
    bordavote_weight_list = []
    # FIXME: K
    K = qreq_.qparams.K
    _branks = np.arange(1, K + 1, dtype=np.float)[::-1]
    bordavote_weight_list = [
        np.tile(_branks, (len(neighb_idx), 1)) for (neighb_idx, neighb_dist) in nns_list
    ]
    return bordavote_weight_list
