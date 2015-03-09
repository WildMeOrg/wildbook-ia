"""
TODO:
optional symetric and asymmetric search

"""

from __future__ import absolute_import, division, print_function
import six  # NOQA
import numpy as np
import vtool as vt
import utool as ut
from vtool import coverage_kpts
from vtool import coverage_grid
from ibeis.model.hots import hstypes
#from ibeis.model.hots import name_scoring
from ibeis.model.hots import distinctiveness_normalizer
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
import scipy.stats.mstats as spmstat
from six.moves import zip, range, map  # NOQA
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[scoring]', DEBUG=False)


@profile
def score_chipmatch_list(qreq_, cm_list, score_method):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-score_chipmatch_list
        python -m ibeis.model.hots.scoring --test-score_chipmatch_list:0 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver()
        >>> score_method = qreq_.qparams.prescore_method
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> cm = cm_list[0]
        >>> assert cm.score_list.argmax() == 0
        >>> cm.testshow_single(qreq_)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> cm = cm_list[0]
        >>> score_method = qreq_.qparams.score_method
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> assert cm.score_list.argmax() == 0
        >>> cm.testshow_single(qreq_)
    """
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        for cm in cm_list:
            cm.score_csum(qreq_)
    elif score_method == 'nsum':
        for cm in cm_list:
            cm.score_nsum(qreq_)
    else:
        raise Exception('[hs] unknown scoring method:' + score_method)


def compute_csum_score(cm, qreq_=None):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-compute_csum_score

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_.ibs)
        >>> cm.qnid = 1   # Hack for testdb1 names
        >>> gt_flags = cm.get_groundtruth_flags()
        >>> annot_score_list = compute_csum_score(cm)
        >>> assert annot_score_list[gt_flags].max() > annot_score_list[~gt_flags].max()
        >>> assert annot_score_list[gt_flags].max() > 10.0
    """
    fs_list = cm.get_fsv_prod_list()
    csum_score_list = np.array([np.sum(fs) for fs in fs_list])
    return csum_score_list


def get_name_shortlist_aids(daid_list, dnid_list, annot_score_list,
                            name_score_list, nid2_nidx,
                            nNameShortList, nAnnotPerName):
    r"""
    CommandLine:
        python -m ibeis.model.hots.scoring --test-get_name_shortlist_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> # build test data
        >>> daid_list        = np.array([11, 12, 13, 14, 15, 16, 17])
        >>> dnid_list        = np.array([21, 21, 21, 22, 22, 23, 24])
        >>> annot_score_list = np.array([ 6,  2,  3,  5,  6,  3,  2])
        >>> name_score_list  = np.array([ 8,  9,  5,  4])
        >>> nid2_nidx        = {21:0, 22:1, 23:2, 24:3}
        >>> nNameShortList, nAnnotPerName = 3, 2
        >>> # execute function
        >>> args = (daid_list, dnid_list, annot_score_list, name_score_list,
        ...         nid2_nidx, nNameShortList, nAnnotPerName)
        >>> top_daids = get_name_shortlist_aids(*args)
        >>> # verify results
        >>> result = str(top_daids)
        >>> print(result)
        [15, 14, 11, 13, 16]
    """
    unique_nids, groupxs    = vt.group_indices(np.array(dnid_list))
    grouped_annot_scores    = vt.apply_grouping(annot_score_list, groupxs)
    grouped_daids           = vt.apply_grouping(np.array(daid_list), groupxs)
    # Ensure name score list is aligned with the unique_nids
    aligned_name_score_list = name_score_list.take(ut.dict_take(nid2_nidx, unique_nids))
    # Sort each group by the name score
    group_sortx             = aligned_name_score_list.argsort()[::-1]
    _top_daid_groups        = ut.list_take(grouped_daids, group_sortx)
    _top_annot_score_groups = ut.list_take(grouped_annot_scores, group_sortx)
    top_daid_groups         = ut.listclip(_top_daid_groups, nNameShortList)
    top_annot_score_groups  = ut.listclip(_top_annot_score_groups, nNameShortList)
    # Sort within each group by the annotation score
    top_daid_sortx_groups   = [annot_score_group.argsort()[::-1]
                               for annot_score_group in top_annot_score_groups]
    top_sorted_daid_groups  = [daids_group.take(sortx) for daids_group, sortx in
                               zip(top_daid_groups, top_daid_sortx_groups)]
    top_clipped_daids = [ut.listclip(sorted_daid_group, nAnnotPerName)
                         for sorted_daid_group in top_sorted_daid_groups]
    top_daids = ut.flatten(top_clipped_daids)
    return top_daids


@profile
def make_chipmatch_shortlists(qreq_, cm_list, nNameShortList, nAnnotPerName):
    """
    Makes shortlists for reranking

    CommandLine:
        python -m ibeis.model.hots.scoring --test-make_chipmatch_shortlists --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
        >>> score_method    = 'nsum'
        >>> nNameShortList  = 5
        >>> nAnnotPerName   = 6
        >>> # apply scores
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> cm_input = cm_list[0]
        >>> #assert cm_input.dnid_list.take(cm_input.argsort())[0] == cm_input.qnid
        >>> # execute function
        >>> cm_shortlist = make_chipmatch_shortlists(qreq_, cm_list, nNameShortList, nAnnotPerName)
        >>> cm_input.print_rawinfostr()
        >>> cm = cm_shortlist[0]
        >>> cm.print_rawinfostr()
        >>> # should be sorted already from the shortlist take
        >>> top_nid_list = cm.dnid_list
        >>> top_aid_list = cm.daid_list
        >>> qnid = cm.qnid
        >>> print('top_aid_list = %r' % (top_aid_list,))
        >>> print('top_nid_list = %r' % (top_nid_list,))
        >>> print('qnid = %r' % (qnid,))
        >>> rankx = top_nid_list.tolist().index(qnid)
        >>> assert rankx == 0, 'qnid=%r should be first rank, not rankx=%r' % (qnid, rankx)
        >>> max_num_rerank = nNameShortList * nAnnotPerName
        >>> min_num_rerank = nNameShortList
        >>> ut.assert_inbounds(len(top_nid_list), min_num_rerank, max_num_rerank, 'incorrect number in shortlist', eq=True)
        >>> cm.testshow_single(qreq_, daid=top_aid_list[0])
    """
    print('[scoring] Making shortlist nNameShortList=%r, nAnnotPerName=%r' % (nNameShortList, nAnnotPerName))
    cm_shortlist = []
    for cm in cm_list:
        assert cm.score_list is not None, 'score list must be computed'
        assert cm.annot_score_list is not None, 'annot_score_list must be computed'
        #nscore_tup = name_scoring.group_scores_by_name(qreq_.ibs, cm.daid_list, cm.score_list)
        #(sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscore_tup
        # Clip number of names
        #_top_aids_list  = ut.listclip(sorted_aids, nNameShortList)
        # Clip number of annots per name
        #_top_clipped_aids_list = [ut.listclip(aids, nAnnotPerName) for aids in _top_aids_list]
        #top_aids = ut.flatten(_top_clipped_aids_list)
        top_aids = cm.get_name_shortlist_aids(nNameShortList, nAnnotPerName)
        cm_subset = cm.shortlist_subset(top_aids)
        cm_shortlist.append(cm_subset)
    return cm_shortlist


#### FEATURE WEIGHTS ####
# TODO: qreq_


def sift_selectivity_score(vecs1_m, vecs2_m, cos_power=3.0, dtype=np.float64):
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
    #selectivity_score = np.multiply(np.sign(cosangle), np.power(cosangle, cos_power))
    return selectivity_score


@profile
def get_kpts_distinctiveness(qreq_, aid_list, config={}):
    """
    per-species disinctivness wrapper around ibeis cached function
    """
    dcvs_kw = distinctiveness_normalizer.DCVS_DEFAULT.updated_cfgdict(config)
    dstncvs_list = qreq_.ibs.get_annot_kpts_distinctiveness(aid_list, qreq_=qreq_, **dcvs_kw)
    return dstncvs_list


def get_annot_kpts_baseline_weights(qreq_, aid_list, config={}):
    r"""
    Args:
        qreq_ (QueryRequest):  query request object with hyper-parameters
        aid_list (int):  list of annotation ids
        config (dict):

    Returns:
        list: weights_list

    CommandLine:
        python -m ibeis.model.hots.scoring --test-get_annot_kpts_baseline_weights

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring()
        >>> aid_list = cm.daid_list
        >>> config = qreq_.qparams
        >>> # execute function
        >>> weights_list = get_annot_kpts_baseline_weights(qreq_, aid_list, config)
        >>> # verify results
        >>> result = str(weights_list)
        >>> print(result)
    """
    # TODO: clip the fgweights? (dilation?)
    # TODO; normalize and paramatarize and clean
    dcvs_on = config.get('dcvs_on')
    fg_on = config.get('fg_on')
    weight_lists = []
    if dcvs_on:
        qdstncvs_list = get_kpts_distinctiveness(qreq_, aid_list, config)
        weight_lists.append(qdstncvs_list)
    if fg_on:
        qfgweight_list = qreq_.ibs.get_annot_fgweights(aid_list, ensure=True, qreq_=qreq_)
        weight_lists.append(qfgweight_list)
    # geometric mean of the selected weights
    baseline_weights_list = [spmstat.gmean(weight_tup) for weight_tup in zip(*weight_lists)]
    return baseline_weights_list


def get_mask_func(config):
    maskscore_mode = config.get('maskscore_mode', 'grid')
    #print(maskscore_mode)
    FUNC_ARGS_DICT = {
        'grid': (coverage_grid.make_grid_coverage_mask, coverage_grid.COVGRID_DEFAULT),
        'kpts': (coverage_kpts.make_kpts_coverage_mask, coverage_kpts.COVKPTS_DEFAULT),
    }
    make_mask_func, func_defaults = FUNC_ARGS_DICT[maskscore_mode]
    cov_cfg = func_defaults.updated_cfgdict(config)
    # Hack to make kwargs happy
    #cov_cfg = ut.filter_valid_kwargs(make_mask_func, config)
    return make_mask_func, cov_cfg


#### MASK COVERAGE SCORING ####


def compute_annot_coverage_score(qreq_, cm, config={}):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-compute_annot_coverage_score:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring()
        >>> config = qreq_.qparams
        >>> daid_list, score_list = compute_annot_coverage_score(qreq_, cm, config)
        >>> ut.assert_inbounds(np.array(score_list), 0, 1, eq=True)
        >>> result = ut.list_str(score_list, precision=3)
        >>> print(result)
    """
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_annot_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg)
    daid_list, score_list = score_masks(masks_iter)
    return daid_list, score_list


def compute_name_coverage_score(qreq_, cm, config={}):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-compute_name_coverage_score:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> qreq_, cm = plh.testdata_scoring()
        >>> cm.evaluate_dnids(qreq_.ibs)
        >>> config = qreq_.qparams
        >>> dnid_list, score_list = compute_name_coverage_score(qreq_, cm, config)
        >>> ut.assert_inbounds(np.array(score_list), 0, 1, eq=True)
        >>> result = ut.list_str(score_list, precision=3)
        >>> print(result)
    """
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_name_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg)
    dnid_list, score_list = score_masks(masks_iter)
    return dnid_list, score_list


def score_masks(masks_iter):
    id_score_list = [(id_, score_matching_mask(weight_mask_m, weight_mask))
                     for id_, weight_mask_m, weight_mask in masks_iter]
    id_list        = np.array(ut.get_list_column(id_score_list, 0))
    coverage_score = np.array(ut.get_list_column(id_score_list, 1))
    return id_list, coverage_score


def score_matching_mask(weight_mask_m, weight_mask):
    coverage_score = weight_mask_m.sum() / weight_mask.sum()
    return coverage_score


def general_annot_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg):
    """
    Yeilds:
        daid, weight_mask_m, weight_mask

    CommandLine:
        python -m ibeis.model.hots.scoring --test-general_annot_coverage_mask_generator --show
        python -m ibeis.model.hots.scoring --test-general_annot_coverage_mask_generator --show --qaid 18

    Note:
        Evaluate output one at a time or it will get clobbered

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
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
        print('[acov] cov_cfg = %s' % (ut.dict_str(cov_cfg),))
    return general_coverage_mask_generator(make_mask_func, qreq_, cm.qaid, cm.daid_list, cm.fm_list, cm.fs_list, config, cov_cfg)


def general_name_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg):
    """
    Yeilds:
        nid, weight_mask_m, weight_mask

    CommandLine:
        python -m ibeis.model.hots.scoring --test-general_name_coverage_mask_generator --show
        python -m ibeis.model.hots.scoring --test-general_name_coverage_mask_generator --show --qaid 18

    Note:
        Evaluate output one at a time or it will get clobbered

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
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
    if ut.VERYVERBOSE:
        print('[ncov] make_mask_func = %r' % (make_mask_func,))
        print('[ncov] cov_cfg = %s' % (ut.dict_str(cov_cfg),))
    assert cm.dnid_list is not None, 'eval nids'
    unique_dnids, groupxs = vt.group_indices(cm.dnid_list)
    fm_groups = vt.apply_grouping_(cm.fm_list, groupxs)
    fs_groups = vt.apply_grouping_(cm.fs_list, groupxs)
    fs_name_list = [np.hstack(fs_group) for fs_group in fs_groups]
    fm_name_list = [np.vstack(fm_group) for fm_group in fm_groups]
    return general_coverage_mask_generator(make_mask_func, qreq_, cm.qaid, unique_dnids, fm_name_list, fs_name_list, config, cov_cfg)


def general_coverage_mask_generator(make_mask_func, qreq_, qaid, id_list, fm_list, fs_list, config, cov_cfg):
    """ agnostic to whether or not the id/fm/fs lists are name or annotation groups """
    if ut.VERYVERBOSE:
        print('[acov] make_mask_func = %r' % (make_mask_func,))
        print('[acov] cov_cfg = %s' % (ut.dict_str(cov_cfg),))
    # Distinctivness and foreground weight
    qweights = get_annot_kpts_baseline_weights(qreq_, [qaid], config)[0]
    # Denominator weight mask
    chipsize    = qreq_.ibs.get_annot_chip_sizes(qaid, qreq_=qreq_)
    qkpts       = qreq_.ibs.get_annot_kpts(qaid, qreq_=qreq_)
    weight_mask = make_mask_func(qkpts, chipsize, qweights, resize=False, **cov_cfg)
    # Prealloc data for loop
    weight_mask_m = weight_mask.copy()
    # Apply weighted scoring to matches
    for daid, fm, fs in zip(id_list, fm_list, fs_list):
        # CAREFUL weight_mask_m is overriden on every iteration
        weight_mask_m = compute_general_matching_coverage_mask(
            make_mask_func, chipsize, fm, fs, qkpts, qweights, cov_cfg, out=weight_mask_m)
        yield daid, weight_mask_m, weight_mask


def compute_general_matching_coverage_mask(make_mask_func, chipsize, fm, fs,
                                           qkpts, qweights, cov_cfg, out=None):
    # Get matching query keypoints
    #SYMMETRIC = False
    #if SYMMETRIC:
    #    get_annot_kpts_baseline_weights()
    qkpts_m    = qkpts.take(fm.T[0], axis=0)
    weights_m  = fs * qweights.take(fm.T[0], axis=0)
    # hacky buisness
    #weights_m  = qweights.take(fm.T[0], axis=0)
    #weights_m = fs
    weight_mask_m = make_mask_func(qkpts_m, chipsize, weights_m,
                                   out=out, resize=False,
                                   **cov_cfg)
    return weight_mask_m


def get_masks(qreq_, cm, config={}):
    r"""
    testing function

    CommandLine:
        # SHOW THE BASELINE AND MATCHING MASKS
        python -m ibeis.model.hots.scoring --test-get_masks
        python -m ibeis.model.hots.scoring --test-get_masks \
            --maskscore_mode=kpts --show --prior_coeff=.5 --unconstrained_coeff=.3 --constrained_coeff=.2
        python -m ibeis.model.hots.scoring --test-get_masks \
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5
        python -m ibeis.model.hots.scoring --test-get_masks --qaid 4\
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5
        python -m ibeis.model.hots.scoring --test-get_masks --qaid 86\
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5 --grid_scale_factor=.5

        python -m ibeis.model.hots.scoring --test-get_masks --show --db PZ_MTEST --qaid 18
        python -m ibeis.model.hots.scoring --test-get_masks --show --db PZ_MTEST --qaid 1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> qreq_, cm = plh.testdata_scoring('PZ_MTEST', qaid_list=[18])
        >>> config = qreq_.qparams
        >>> # execute function
        >>> id_list, score_list, masks_list = get_masks(qreq_, cm, config)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> show_coverage_mask(qreq_, cm, masks_list, index=score_list.argmax())
        >>> pt.show_if_requested()
    """
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_annot_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg)
    # copy weight mask as it comes back if you want to see them
    id_list, score_list, masks_list = evaluate_masks_iter(masks_iter)

    return id_list, score_list, masks_list


def evaluate_masks_iter(masks_iter):
    """ save evaluation of a masks iter """
    masks_list = [(id_, weight_mask_m.copy(), weight_mask)
                   for id_, weight_mask_m, weight_mask  in masks_iter]
    id_list, score_list = score_masks(masks_list)
    return id_list, score_list, masks_list


def show_coverage_mask(qreq_, cm, masks_list, index=0, fnum=None):
    daid, weight_mask_m, weight_mask = masks_list[index]
    daids = [daid]
    show_single_coverage_mask(qreq_, cm, weight_mask_m, weight_mask, daids, fnum=None)


def show_single_coverage_mask(qreq_, cm, weight_mask_m, weight_mask, daids, fnum=None):
    import plottool as pt
    from ibeis import viz
    fnum = pt.ensure_fnum(fnum)
    idx_list = ut.dict_take(cm.daid2_idx, daids)
    nPlots = len(idx_list) + 1
    nRows, nCols = pt.get_square_row_cols(nPlots)
    pnum_ = pt.make_pnum_nextgen(nRows, nCols)
    pt.figure(fnum=fnum, pnum=(1, 2, 1))
    # Draw coverage masks with bbox
    # <FlipHack>
    #weight_mask_m = np.fliplr(np.flipud(weight_mask_m))
    #weight_mask = np.fliplr(np.flipud(weight_mask))
    # </FlipHack>
    stacked_weights, woff, hoff = pt.stack_images(weight_mask_m, weight_mask)
    wh1 = weight_mask_m.shape[0:2][::-1]
    wh2 = weight_mask.shape[0:2][::-1]
    pt.imshow(255 * (stacked_weights), fnum=fnum, pnum=pnum_(0))
    pt.draw_bbox((   0,    0) + wh1, bbox_color=(0, 0, 1))
    pt.draw_bbox((woff, hoff) + wh2, bbox_color=(0, 0, 1))
    # Get contributing matches
    qaid = cm.qaid
    daid_list = daids
    fm_list = ut.list_take(cm.fm_list, idx_list)
    fs_list = ut.list_take(cm.fs_list, idx_list)
    # Draw matches
    for px, (daid, fm, fs) in enumerate(zip(daid_list, fm_list, fs_list), start=1):
        viz.viz_matches.show_matches2(qreq_.ibs, qaid, daid, fm, fs,
                                      draw_pts=False, draw_lines=True,
                                      draw_ell=False, fnum=fnum, pnum=pnum_(px),
                                      darken=.5)
    coverage_score = score_matching_mask(weight_mask_m, weight_mask)
    pt.set_figtitle('score=%.4f' % (coverage_score,))


def show_annot_weights(qreq_, aid, config={}):
    r"""
    DEMO FUNC

    CommandLine:
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --maskscore_mode='grid'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --maskscore_mode='kpts'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_Master0 --aid 1
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh
        >>> import plottool as pt
        >>> ibs, qreq_ = plh.get_pipeline_testdata(qaid_list=[1], defaultdb='testdb1', cmdline_ok=True)
        >>> aid = qreq_.get_external_qaids()[0]
        >>> config = qreq_.qparams
        >>> show_annot_weights(qreq_, aid, config)
        >>> pt.show_if_requested()
    """
    #import plottool as pt
    fnum = 1
    chipsize = qreq_.ibs.get_annot_chip_sizes(aid, qreq_=qreq_)
    chip  = qreq_.ibs.get_annot_chips(aid, qreq_=qreq_)
    qkpts = qreq_.ibs.get_annot_kpts(aid, qreq_=qreq_)
    weights = get_annot_kpts_baseline_weights(qreq_, [aid], config)[0]
    make_mask_func, cov_cfg = get_mask_func(config)
    mask = make_mask_func(qkpts, chipsize, weights, resize=True, **cov_cfg)
    coverage_kpts.show_coverage_map(chip, mask, None, qkpts, fnum, ell_alpha=.2, show_mask_kpts=False)
    #pt.set_figtitle(mode)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.scoring
        python -m ibeis.model.hots.scoring --allexamples
        python -m ibeis.model.hots.scoring --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
