"""
special pipeline for vsone specific functions

Current Issues:
    * getting feature distinctiveness is too slow, we can either try a different
      model, or precompute feature distinctiveness.

      - we can reduce the size of the vsone shortlist

TODOLIST:
    * Precompute distinctivness
    * keep feature matches from vsmany (allow fm_B)
    * Each keypoint gets
      - foregroundness
      - global distinctivness (databasewide) LNBNN
      - local distinctivness (imagewide) RATIO
      - regional match quality (descriptor based) COS
    * Circular hesaff keypoints
    * Asymetric weight scoring

    * FIX BUGS IN score_chipmatch_nsum FIRST THING TOMORROW.
     dict keys / vals are being messed up. very inoccuous


TestFuncs:
    >>> # VsMany Only
    python -m ibeis.model.hots.vsone_pipeline --test-show_post_vsmany_vser --show
    >>> # VsOne Only
    python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show --no-vsmany_coeff
    >>> # VsOne + VsMany
    python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show

"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import vtool as vt
#from ibeis.model.hots import neighbor_index
from ibeis.model.hots import voting_rules2 as vr2
from ibeis.model.hots import name_scoring
from ibeis.model.hots import hstypes
from ibeis.model.hots import scoring
#import pyflann
import functools
#from ibeis.model.hots import coverage_kpts
from vtool import matching
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
import utool as ut
from six.moves import zip, range, reduce  # NOQA
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[vsonepipe]', DEBUG=False)

#from collections import namedtuple


def show_post_vsmany_vser():
    """ TESTFUNC just show the input data

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-show_post_vsmany_vser --show --homog
        python -m ibeis.model.hots.vsone_pipeline --test-show_post_vsmany_vser --show --csum --homog

    Example:
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> show_post_vsmany_vser()
    """
    import plottool as pt
    ibs, qreq_, qaid2_vsm_chipmatch, qaid_list, rrvsone_cfgdict = testdata_post_vsmany_sver()
    # HACK TO PRESCORE
    ibs = qreq_.ibs
    vsm_cm_list = prepare_vsmany_chipmatch(qreq_, qaid2_vsm_chipmatch)
    show_all_top_chipmatches(ibs, vsm_cm_list, figtitle='vsmany post sver')
    pt.show_if_requested()


def get_normalized_score_column(fsv, colx, min_, max_, power):
    fs = fsv.T[colx].T.copy()
    fs = vt.clipnorm(fs, min_, max_, out=fs)
    fs = np.power(fs, power, out=fs)
    return fs


def prepare_vsmany_chipmatch(qreq_, qaid2_vsm_chipmatch):
    """ gets normalized vsmany priors

        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, qaid2_vsm_chipmatch, qaid_list, rrvsone_cfgdict = testdata_post_vsmany_sver()
        >>> vsm_cm_list = prepare_vsmany_chipmatch(qreq_, qaid2_vsm_chipmatch)

    """
    # Hack: populate aid2 score field in chipmatch using prescore
    for qaid, chipmatch_VSMANY in six.iteritems(qaid2_vsm_chipmatch):
        daid_list, prescore_list = vr2.score_chipmatch_nsum(qaid, chipmatch_VSMANY, qreq_)
        ut.dict_assign(chipmatch_VSMANY.aid2_score, daid_list, prescore_list)
    vsmany_filtkey_list = qreq_.qparams.get_postsver_filtkey_list()
    # convert to chipmatch2
    vsm_cm_list = [
        hstypes.ChipMatch2.from_chipmatch_old(chipmatch_VSMANY, qaid=qaid, fsv_col_lbls=vsmany_filtkey_list)
        for qaid, chipmatch_VSMANY in six.iteritems(qaid2_vsm_chipmatch)
    ]
    # grab normalized lnbnn scores
    lnbnn_index    = vsmany_filtkey_list.index('lnbnn')
    # TODO: paramatraize
    fs_lnbnn_min   = .0001
    fs_lnbnn_max   = .05
    fs_lnbnn_power = 1.0
    _args = (lnbnn_index, fs_lnbnn_min, fs_lnbnn_max, fs_lnbnn_power)
    for vsmany_cm in vsm_cm_list:
        vsmany_fs_list = [get_normalized_score_column(vsmany_fsv, *_args) for vsmany_fsv in vsmany_cm.fsv_list]
        vsmany_fsv_list = matching.ensure_fsv_list(vsmany_fs_list)
        vsmany_cm.fsv_list = vsmany_fsv_list
    return vsm_cm_list


@profile
def make_chipmatch_shortlist(qreq_, cm_list):
    """
    Makes shortlists for vsone reranking

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-make_chipmatch_shortlist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, qaid2_vsm_chipmatch, qaid_list, rrvsone_cfgdict = testdata_post_vsmany_sver()
        >>> vsm_cm_list = prepare_vsmany_chipmatch(qreq_, qaid2_vsm_chipmatch)
        >>> cm_list = vsm_cm_list
        >>> cm_shortlist = make_chipmatch_shortlist(qreq_, cm_list)
        >>> top_nid_list = ibs.get_annot_name_rowids(cm_shortlist[0].daid_list)
        >>> qnid = ibs.get_annot_name_rowids(cm_shortlist[0].qaid)
        >>> print('top_aid_list = %r' % (cm_shortlist[0].daid_list,))
        >>> print('top_nid_list = %r' % (top_nid_list,))
        >>> print('qnid = %r' % (qnid,))
        >>> assert top_nid_list.index(qnid) == 0, 'qnid=%r should be first rank' % (qnid,)
        >>> max_num_rerank = qreq_.qparams.nNameShortlistVsone * qreq_.qparams.nAnnotPerName
        >>> min_num_rerank = qreq_.qparams.nNameShortlistVsone
        >>> ut.assert_inbounds(len(top_nid_list), min_num_rerank, max_num_rerank, 'incorrect number in shortlist')
    """
    ibs = qreq_.ibs
    score_method = qreq_.qparams.score_method
    assert score_method == 'nsum'
    # TODO: paramaterize
    # Params: the max number of top names to get and the max number of
    # annotations per name to verify against
    nNameShortlistVsone = qreq_.qparams.nNameShortlistVsone
    nAnnotPerName       = qreq_.qparams.nAnnotPerName
    cm_shortlist = []
    for cm in cm_list:
        nscore_tup    = name_scoring.group_scores_by_name(ibs, cm.daid_list, cm.score_list)
        (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscore_tup
        # Clip number of names
        _top_aids_list  = ut.listclip(sorted_aids, nNameShortlistVsone)
        # Clip number of annots per name
        _top_clipped_aids_list = [ut.listclip(aids, nAnnotPerName) for aids in _top_aids_list]
        top_aids = ut.flatten(_top_clipped_aids_list)
        cm_subset = cm.shortlist_subset(top_aids)
        cm_shortlist.append(cm_subset)
    return cm_shortlist


#@profile
def vsone_reranking(qreq_, qaid2_vsm_chipmatch, verbose=False):
    """
    Driver function

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show

        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show

        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking
        utprof.py -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, qaid2_vsm_chipmatch, qaid_list, rrvsone_cfgdict = testdata_post_vsmany_sver()
        >>> # qaid2_vsm_chipmatch = ut.dict_subset(qaid2_vsm_chipmatch, [6])
        >>> qaid2_chipmatch_VSONE = vsone_reranking(qreq_, qaid2_vsm_chipmatch)
        >>> #qaid2_chipmatch = qaid2_chipmatch_VSONE
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     figtitle = ut.dict_str(rrvsone_cfgdict, newlines=False)
        ...     show_all_top_chipmatches(qreq_.ibs, qaid2_chipmatch_VSONE, figtitle=figtitle)
        ...     pt.show_if_requested()
    """
    ibs = qreq_.ibs
    config = qreq_.qparams
    # Get vsmany chipmatches into input format
    vsm_cm_list = prepare_vsmany_chipmatch(qreq_, qaid2_vsm_chipmatch)
    # Filter down to a shortlist
    cm_shortlist = make_chipmatch_shortlist(qreq_, vsm_cm_list)
    # Execute vsone reranking
    _prog = functools.partial(ut.ProgressIter, nTotal=len(cm_shortlist),
                              lbl='VSONE RERANKING', freq=1)
    reranked_list = [
        single_vsone_rerank(ibs, prior_cm, config)
        for prior_cm in _prog(cm_shortlist)
    ]
    # Format the output into oldstyle chipmatches
    qaid2_chipmatch_VSONE = {
        cm.qaid: cm.to_oldstyle_chipmatch()
        for cm in reranked_list
    }
    return qaid2_chipmatch_VSONE


def marge_matches_lists(fmfs_A, fmfs_B):
    if fmfs_A is None:
        return fmfs_B
    fm_A_list, fsv_A_list = fmfs_A
    fm_B_list, fsv_B_list = fmfs_B
    fm_merge_list = []
    fsv_merge_list = []
    fsv_A_list = matching.ensure_fsv_list(fsv_A_list)
    fsv_B_list = matching.ensure_fsv_list(fsv_B_list)
    for fm_A, fm_B, fsv_A, fsv_B in zip(fm_A_list, fm_B_list, fsv_A_list, fsv_B_list):
        fm_merged, fs_merged = matching.marge_matches(fm_A, fm_B, fsv_A, fsv_B)
        fm_merge_list.append(fm_merged)
        fsv_merge_list.append(fs_merged)
    fmfs_merge = fm_merge_list, fsv_merge_list
    return fmfs_merge


@profile
def sver_fmfs_merge(ibs, qaid, daid_list, fmfs_merge, config={}):
    from vtool import spatial_verification as sver
    # params
    xy_thresh    = config.get('xy_thresh') * 2
    scale_thresh = config.get('scale_thresh') * 2
    ori_thresh   = config.get('ori_thresh') * 2
    min_nInliers = config.get('min_nInliers')
    # input data
    fm_list, fsv_list = fmfs_merge
    kpts1 = ibs.get_annot_kpts(qaid)
    kpts2_list = ibs.get_annot_kpts(daid_list)
    chip2_dlen_sqrd_list = ibs.get_annot_chip_dlen_sqrd(daid_list)  # chip diagonal length
    res_list = []
    #
    # homog_inliers
    for kpts2, chip2_dlen_sqrd, fm, fsv in zip(kpts2_list, chip2_dlen_sqrd_list, fm_list, fsv_list):
        sv_tup = sver.spatially_verify_kpts(
            kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh,
            chip2_dlen_sqrd, min_nInliers,
            returnAff=False)
        (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) = sv_tup
        fm_SV = fm.take(homog_inliers, axis=0)
        fsv_SV = fsv.take(homog_inliers, axis=0)
        res_list.append((fm_SV, fsv_SV, H))
    fm_list_SV = ut.get_list_column(res_list, 0)
    fsv_list_SV = ut.get_list_column(res_list, 1)
    H_list = ut.get_list_column(res_list, 2)
    fmfs_merge_SV = (fm_list_SV, fsv_list_SV)
    return fmfs_merge_SV, H_list


def get_selectivity_score_list(ibs, qaid, daid_list, fm_list, cos_power):
    vecs1 = ibs.get_annot_vecs(qaid)
    vecs2_list = ibs.get_annot_vecs(daid_list)
    vecs1_m_iter = (vecs1.take(fm.T[0], axis=0) for fm in fm_list)
    vecs2_m_iter = (vecs2.take(fm.T[1], axis=0) for fm, vecs2 in zip(fm_list, vecs2_list))
    # Rescore constrained using selectivity function
    fs_list = [scoring.sift_selectivity_score(vecs1_m, vecs2_m, cos_power)
                   for vecs1_m, vecs2_m in zip(vecs1_m_iter, vecs2_m_iter)]
    return fs_list


@profile
def refine_matches(ibs, prior_cm, config={}):
    """
    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog --sver_unconstrained
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog --sver_constrained&
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog --sver_constrained --sver_unconstrained&

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = testdata_matching()
        >>> config = qreq_.qparams
        >>> fm_list, fs_list, H_list = refine_matches(ibs, prior_cm, config)
        >>> cm = hstypes.ChipMatch2.from_prior(prior_cm, fm_list, fs_list, H_list)
        >>> cm.print_csv(ibs=ibs)
        >>> prior_cm.print_csv(ibs=ibs)
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     show_single_chipmatch(ibs, cm)
        ...     pt.set_figtitle(qreq_.qparams.query_cfgstr)
        ...     pt.show_if_requested()
    """
    prior_coeff         = config.get('prior_coeff')
    unconstrained_coeff = config.get('unconstrained_coeff')
    constrained_coeff   = config.get('constrained_coeff')
    sver_unconstrained  = config.get('sver_unconstrained')
    sver_constrained    = config.get('sver_constrained')
    # TODO: Param
    scr_cos_power      = 3.0
    #
    qaid           = prior_cm.qaid
    daid_list      = prior_cm.daid_list
    fm_prior_list  = prior_cm.fm_list
    fsv_prior_list = prior_cm.fsv_list
    H_prior_list   = prior_cm.H_list
    H_list         = H_prior_list

    assert unconstrained_coeff is not None, '%r' % (unconstrained_coeff,)

    col_coeff_list = []
    fmfs_merge = None

    if prior_coeff != 0:
        # Merge into result
        col_coeff_list.append(prior_coeff)
        fmfs_prior = (fm_prior_list, fsv_prior_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_prior)

    if unconstrained_coeff != 0:
        col_coeff_list.append(unconstrained_coeff)
        unc_match_results = compute_query_unconstrained_matches(ibs, qaid, daid_list, config)
        fm_unc_list, fs_unc_list, fm_norm_unc_list = unc_match_results
        # Merge into result
        fmfs_unc = (fm_unc_list, fs_unc_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_unc)

    # We have the option of spatially verifying the merged results from the
    # prior and the new unconstrained matches.
    if sver_unconstrained:
        fmfs_merge, H_list = sver_fmfs_merge(ibs, qaid, daid_list, fmfs_merge, config)

    if constrained_coeff != 0:
        scr_match_results = compute_query_constrained_matches(ibs, qaid, daid_list, H_list, config)
        fm_scr_list, fs_scr_list, fm_norm_scr_list = scr_match_results
        fs_scr_list = get_selectivity_score_list(ibs, qaid, daid_list, fm_scr_list, scr_cos_power)
        # Merge into result
        fmfs_scr = (fm_scr_list, fs_scr_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_scr)
        col_coeff_list.append(constrained_coeff)

    # Another optional round of spatial verification
    if sver_constrained:
        fmfs_merge, H_list = sver_fmfs_merge(ibs, qaid, daid_list, fmfs_merge, config)

    coeffs = np.array(col_coeff_list)
    # merge different match types
    fm_list, fsv_list = fmfs_merge
    # apply linear combination
    fs_list = [(np.nan_to_num(fsv) * coeffs[None, :]).sum(axis=1) for fsv in fsv_list]
    return fm_list, fs_list, H_list


@profile
def single_vsone_rerank(ibs, prior_cm, config={}):
    r"""
    Runs a single vsone-pair (query, daid_list)

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_rerank:0
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_rerank:0 --show

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = testdata_matching()
        >>> config = qreq_.qparams
        >>> rerank_cm = single_vsone_rerank(ibs, prior_cm, config)
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     show_single_chipmatch(ibs, rerank_cm)
        ...     pt.show_if_requested()
        >>> print(rerank_cm.score_list)
    """

    use_gridcov_scoring = config.get('use_gridcov_scoring')
    use_kptscov_scoring = config.get('use_kptscov_scoring')

    #print('==================')
    fm_list, fs_list, H_list = refine_matches(ibs, prior_cm, config)
    qaid      = prior_cm.qaid
    daid_list = prior_cm.daid_list

    if use_gridcov_scoring:
        cov_score_list = scoring.compute_grid_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config=config)
    elif use_kptscov_scoring:
        cov_score_list = scoring.compute_kpts_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config=config)
    else:
        raise AssertionError('must choose either grid or kpts coverage')

    #NAME_SCORING = True
    NAME_SCORING = False
    if NAME_SCORING:
        # Keep only the best annotation per name
        # FIXME: There may be a problem here
        nscore_tup = name_scoring.group_scores_by_name(ibs, daid_list, cov_score_list)
        score_list = ut.flatten([scores[0:1].tolist() + ([0] * (len(scores) - 1))
                                 for scores in nscore_tup.sorted_scores])
    else:
        score_list = cov_score_list
    # Convert our one score to a score vector here
    fsv_list = matching.ensure_fsv_list(fs_list)
    rerank_cm = hstypes.ChipMatch2(qaid, daid_list, fm_list, fsv_list, None, score_list, H_list)
    return rerank_cm


def quick_vsone_flann(ibs, qvecs):
    flann_cachedir = ibs.get_flann_cachedir()
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    use_cache = save = False and ut.is_developer()
    flann = vt.flann_cache(qvecs, flann_cachedir, flann_params=flann_params,
                           quiet=True, verbose=False, use_cache=use_cache, save=save)
    return flann


@profile
def compute_query_constrained_matches(ibs, qaid, daid_list, H_list, config):
    """

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show --shownorm
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show --shownorm --homog

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = testdata_matching()
        >>> config = qreq_.qparams
        >>> qaid, daid_list, H_list = ut.dict_take(prior_cm, ['qaid', 'daid_list', 'H_list'])
        >>> match_results = compute_query_constrained_matches(ibs, qaid, daid_list, H_list, config)
        >>> fm_SCR_list, fs_SCR_list, fm_norm_SCR_list = match_results
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     idx = ut.listfind(ibs.get_annot_nids(daid_list), ibs.get_annot_nids(qaid))
        ...     args = (ibs, qaid, daid_list, fm_SCR_list, fs_SCR_list, fm_norm_SCR_list, H_list)
        ...     show_single_match(*args, index=idx)
        ...     pt.set_title('unconstrained')
        ...     pt.show_if_requested()
    """
    scr_ratio_thresh     = config.get('scr_ratio_thresh', .7)
    scr_K                = config.get('scr_K', 7)
    scr_match_xy_thresh  = config.get('scr_match_xy_thresh', .05)
    scr_norm_xy_min      = config.get('scr_norm_xy_min', 0.1)
    scr_norm_xy_max      = config.get('scr_norm_xy_max', 1.0)
    scr_norm_xy_bounds = (scr_norm_xy_min, scr_norm_xy_max)
    # query info
    vecs1 = ibs.get_annot_vecs(qaid)
    kpts1 = ibs.get_annot_kpts(qaid)
    # database info
    vecs2_list = ibs.get_annot_vecs(daid_list)
    kpts2_list = ibs.get_annot_kpts(daid_list)
    chip2_dlen_sqrd_list = ibs.get_annot_chip_dlen_sqrd(daid_list)  # chip diagonal length
    # build flann for query vectors
    flann = quick_vsone_flann(ibs, vecs1)
    # match database chips to query chip
    scr_kwargs = {
        'scr_K'            : scr_K,
        'match_xy_thresh'  : scr_match_xy_thresh,
        'norm_xy_bounds'   : scr_norm_xy_bounds,
        'scr_ratio_thresh' : scr_ratio_thresh,
        'fm_dtype'         : hstypes.FM_DTYPE,
        'fs_dtype'         : hstypes.FS_DTYPE,
    }
    #print('scr_kwargs = ' + ut.dict_str(scr_kwargs))
    # Homographys in H_list map image1 space into image2 space
    scrtup_list = [
        matching.spatially_constrained_ratio_match(
            flann, vecs2, kpts1, kpts2, H, chip2_dlen_sqrd, **scr_kwargs)
        for vecs2, kpts2, chip2_dlen_sqrd, H in
        zip(vecs2_list, kpts2_list, chip2_dlen_sqrd_list, H_list)]
    # return matches and scores
    fm_SCR_list = ut.get_list_column(scrtup_list, 0)
    fs_SCR_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_SCR_list = ut.get_list_column(scrtup_list, 2)
    match_results = fm_SCR_list, fs_SCR_list, fm_norm_SCR_list
    return match_results


@profile
def compute_query_unconstrained_matches(ibs, qaid, daid_list, config):
    """

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show --shownorm
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show --shownorm --homog

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = testdata_matching()
        >>> qaid, daid_list, H_list = ut.dict_take(prior_cm, ['qaid', 'daid_list', 'H_list'])
        >>> match_results = compute_query_unconstrained_matches(ibs, qaid, daid_list, config)
        >>> fm_RAT_list, fs_RAT_list, fm_norm_RAT_list = match_results
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     idx = ut.listfind(ibs.get_annot_nids(daid_list), ibs.get_annot_nids(qaid))
        ...     args = (ibs, qaid, daid_list, fm_RAT_list, fs_RAT_list, fm_norm_RAT_list, H_list)
        ...     show_single_match(*args, index=idx)
        ...     pt.set_title('unconstrained')
        ...     pt.show_if_requested()
    """
    unc_ratio_thresh = config.get('unc_ratio_thresh', .625)
    # query info
    qvecs = ibs.get_annot_vecs(qaid)
    # database info
    dvecs_list = ibs.get_annot_vecs(daid_list)
    # build flann for query vectors
    flann = quick_vsone_flann(ibs, qvecs)
    # match database chips to query chip
    rat_kwargs = {
        'unc_ratio_thresh' : unc_ratio_thresh,
        'fm_dtype'     : hstypes.FM_DTYPE,
        'fs_dtype'     : hstypes.FS_DTYPE,
    }
    #print('rat_kwargs = ' + ut.dict_str(rat_kwargs))
    scrtup_list = [
        matching.unconstrained_ratio_match(
            flann, dvecs, **rat_kwargs)
        for dvecs in dvecs_list]
    # return matches and scores
    fm_RAT_list = ut.get_list_column(scrtup_list, 0)
    fs_RAT_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_RAT_list = ut.get_list_column(scrtup_list, 2)
    match_results = fm_RAT_list, fs_RAT_list, fm_norm_RAT_list
    return match_results


# -----------------------------
# TESTDATA
# -----------------------------


def testdata_post_vsmany_sver():
    """
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
    """
    from ibeis.model import Config
    dbname = ut.get_argval('--db', str, 'PZ_MTEST')
    cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
    rrvsone_cfgdict = dict(Config.RerankVsOneConfig().parse_items())
    rrvsone_cfgdict = ut.util_arg.argparse_dict(rrvsone_cfgdict)
    cfgdict.update(rrvsone_cfgdict)
    cfgdict['rrvsone_on'] = True
    qaid = ut.get_argval('--qaid', int, 1)
    daid_list = ut.get_argval('--daid_list', list, None)
    #ut.embed()
    daid_list = 'all' if daid_list is None else daid_list
    qaid_list = [qaid]
    # VSMANY TO GET HOMOG
    ibs, qreq_ = plh.get_pipeline_testdata(dbname, cfgdict=cfgdict, qaid_list=qaid_list, daid_list=daid_list)
    if len(ibs.get_annot_groundtruth(qaid)) == 0:
        print('WARNING: qaid=%r has no groundtruth' % (qaid,))
    locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
    qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
    return ibs, qreq_, qaid2_chipmatch, qaid_list, rrvsone_cfgdict


def testdata_matching():
    """
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
    """
    ibs, qreq_, qaid2_chipmatch, qaid_list, rrvsone_cfgdict = testdata_post_vsmany_sver()
    vsm_cm_list   = prepare_vsmany_chipmatch(qreq_, qaid2_chipmatch)
    prior_cm_list = make_chipmatch_shortlist(qreq_, vsm_cm_list)
    prior_cm      = prior_cm_list[0]
    return ibs, qreq_, prior_cm


# -----------------------------
# GRIDSEARCH
# -----------------------------


def gridsearch_constrained_matches():
    r"""

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_constrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_constrained_matches --show --testindex 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> gridsearch_constrained_matches()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    fnum = 1

    varied_dict = {
        'scr_ratio_thresh': [.625, .3, .9, 0.0, 1.0],
        'scr_K': [7, 2],
        'scr_xy_thresh': [.05, 1.0, .1],
        'scr_norm_xy_min': [0, .1, .2],
        'scr_norm_xy_max': [1, .3],
    }
    slice_dict = {
        'scr_ratio_thresh': slice(0, 1),
        'scr_K': slice(0, 2),
        'scr_xy_thresh': slice(0, 2),
        'scr_norm_xy_min': slice(0, 2),
        'scr_norm_xy_max': slice(0, 2),
    }

    def constrain_func(cfg):
        if cfg['scr_norm_xy_min'] >= cfg['scr_norm_xy_max']:
            return False

    # Make configuration for every parameter setting
    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(
        varied_dict, constrain_func, slice_dict=slice_dict, defaultslice=slice(0, 10))
    #fname = None  # 'easy1.png'
    ibs, qreq_, prior_cm = testdata_matching()
    qaid      = prior_cm.qaid
    daid_list = prior_cm.daid_list
    H_list    = prior_cm.H_list
    #config = qreq_.qparams
    cfgresult_list = [
        compute_query_constrained_matches(ibs, qaid, daid_list, H_list, cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='scr match')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list, H_list=H_list, index=index)
    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore')

    #if use_separate_norm:
    #    ut.interact_gridsearch_result_images(
    #        functools.partial(show_single_match, use_separate_norm=True), cfgdict_list, cfglbl_list,
    #        cfgresult_list, fnum=fnum + 1, figtitle='constrained ratio match', unpack=True,
    #        max_plots=25, scorelbl='sumscore')
    pt.iup()

# -----------------------------
# VISUALIZATIONS
# -----------------------------


def show_single_match(ibs, qaid, daid_list, fm_list, fs_list, fm_norm_list=None, H_list=None, index=None, **kwargs):
    use_sameaxis_norm = ut.get_argflag('--shownorm')
    fs = fs_list[index]
    fm = fm_list[index]
    if use_sameaxis_norm:
        fm_norm = fm_norm_list[index]
    else:
        fm_norm = None
    kwargs['darken'] = .7
    daid = daid_list[index]
    H1 = H_list[index]
    #H1 = None  # uncomment to see warping
    show_constrained_chipmatch(ibs, qaid, daid, fm, fs=fs, H1=H1, fm_norm=fm_norm, **kwargs)


def show_constrained_chipmatch(ibs, qaid, daid, fm, fs=None, fm_norm=None,
                               H1=None, fnum=None, pnum=None, **kwargs):
    from ibeis.viz import viz_matches
    if not ut.get_argflag('--homog'):
        H1 = None
    viz_matches.show_matches2(ibs, qaid, daid, fm=fm, fs=fs, fm_norm=fm_norm,
                              H1=H1, fnum=fnum, pnum=pnum, show_name=False, **kwargs)
    #pt.set_title('score = %.3f' % (score,))


def show_single_chipmatch(ibs, chipmatch, qaid=None, fnum=None):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    if qaid is None:
        qaid = chipmatch.qaid
    CLIP_TOP = 6
    daid_list     = list(six.iterkeys(chipmatch.aid2_fm))
    score_list    = ut.dict_take(chipmatch.aid2_score, daid_list)
    top_daid_list = ut.listclip(ut.sortedby(daid_list, score_list, reverse=True), CLIP_TOP)
    nRows, nCols  = pt.get_square_row_cols(len(top_daid_list), fix=True)
    next_pnum     = pt.make_pnum_nextgen(nRows, nCols)
    for daid in top_daid_list:
        fm    = chipmatch.aid2_fm[daid]
        fsv   = chipmatch.aid2_fsv[daid]
        fs    = fsv.prod(axis=1)
        H1 = chipmatch.aid2_H[daid]
        pnum = next_pnum()
        #with ut.EmbedOnException():
        show_constrained_chipmatch(ibs, qaid, daid, fm=fm, fs=fs, H1=H1, fnum=fnum, pnum=pnum)
        score = chipmatch.aid2_score[daid]
        pt.set_title('score = %.3f' % (score,))


def show_all_top_chipmatches(ibs, qaid2_chipmatch, fnum_offset=0, figtitle=''):
    """ helper """
    import plottool as pt

    if isinstance(qaid2_chipmatch, list):
        # hack newstyle back to oldstyle
        qaid2_chipmatch = {cm.qaid: cm.to_oldstyle_chipmatch() for cm in qaid2_chipmatch}

    for fnum_, (qaid, chipmatch) in enumerate(six.iteritems(qaid2_chipmatch)):
        #cm.foo()
        fnum = fnum_ + fnum_offset
        show_single_chipmatch(ibs, chipmatch, qaid, fnum)
        #pt.figure(fnum=fnum, doclf=True, docla=True)
        pt.set_figtitle('qaid=%r %s' % (qaid, figtitle))


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline
        python -m ibeis.model.hots.vsone_pipeline --allexamples
        python -m ibeis.model.hots.vsone_pipeline --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
