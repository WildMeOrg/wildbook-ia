"""
special pipeline for vsone specific functions

Current Issues:
    * getting feature distinctiveness is too slow, we can either try a different
      model, or precompute feature distinctiveness.

      - we can reduce the size of the vsone shortlist

TODOLIST:
    * Unconstrained is a terrible name. It is constrianed by the ratio
    * Precompute distinctivness
    #* keep feature matches from vsmany (allow fm_B)
    #* Each keypoint gets
    #  - foregroundness
    #  - global distinctivness (databasewide) LNBNN
    #  - local distinctivness (imagewide) RATIO
    #  - regional match quality (descriptor based) COS
    * Asymetric weight scoring

    * FIX BUGS IN score_chipmatch_nsum FIRST THING TOMORROW.
     dict keys / vals are being messed up. very inoccuous

    Visualization to "prove" that vsone works


TestFuncs:
    >>> # VsMany Only
    python -m ibeis.model.hots.vsone_pipeline --test-show_post_vsmany_vser --show
    >>> # VsOne Only
    python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show --no-vsmany_coeff
    >>> # VsOne + VsMany
    python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show


    >>> # Rerank Vsone Test Harness
    python -c "import utool as ut; ut.write_modscript_alias('Tvs1RR.sh', 'dev.py', '--allgt  --db PZ_MTEST --index 1:40:2')"
    sh Tvs1RR.sh -t custom:rrvsone_on=True custom custom:rrvsone_on=True
    sh Tvs1RR.sh -t custom custom:rrvsone_on=True --print-scorediff-mat-stats
    sh Tvs1RR.sh -t custom:rrvsone_on=True custom:rrvsone_on=True, --print-confusion-stats --print-scorediff-mat-stats

    --print-scorediff-mat-stats --print-confusion-stats

"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
import numpy as np
import vtool as vt
#from ibeis.model.hots import neighbor_index
#from ibeis.model.hots import name_scoring
from ibeis.model.hots import hstypes
from ibeis.model.hots import chip_match
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
    ibs, qreq_, cm_list_SVER, qaid_list  = plh.testdata_pre_vsonerr()
    # HACK TO PRESCORE
    ibs = qreq_.ibs
    prepare_vsmany_chipmatch(qreq_, cm_list_SVER)
    show_all_ranked_matches(ibs, cm_list_SVER, figtitle='vsmany post sver')
    pt.show_if_requested()


def get_normalized_score_column(fsv, colx, min_, max_, power):
    fs = fsv.T[colx].T.copy()
    fs = fs if min_ == 0 and max_ == 1 else vt.clipnorm(fs, min_, max_, out=fs)
    fs = fs if power == 1 else np.power(fs, power, out=fs)
    return fs


def prepare_vsmany_chipmatch(qreq_, cm_list_SVER):
    """ gets normalized vsmany priors

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, cm_list_SVER, qaid_list  = plh.testdata_pre_vsonerr()
        >>> prepare_vsmany_chipmatch(qreq_, cm_list_SVER)
    """
    # Hack: populate aid2 score field in cmtup_old using prescore
    # grab normalized lnbnn scores
    fs_lnbnn_min   = qreq_.qparams.fs_lnbnn_min
    fs_lnbnn_max   = qreq_.qparams.fs_lnbnn_max
    fs_lnbnn_power = qreq_.qparams.fs_lnbnn_power
    _args = (fs_lnbnn_min, fs_lnbnn_max, fs_lnbnn_power)
    for cm in cm_list_SVER:
        lnbnn_index    = cm.fsv_col_lbls.index('lnbnn')
        vsmany_fs_list = [get_normalized_score_column(vsmany_fsv, lnbnn_index, *_args) for vsmany_fsv in cm.fsv_list]
        cm.fsv_list = matching.ensure_fsv_list(vsmany_fs_list)
        cm.fs_list = vsmany_fs_list


#@profile
def vsone_reranking(qreq_, cm_list_SVER, verbose=False):
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
        >>> ibs, qreq_, cm_list_SVER, qaid_list  = plh.testdata_pre_vsonerr()
        >>> # cm_list_SVER = ut.dict_subset(cm_list_SVER, [6])
        >>> cm_list_VSONE = vsone_reranking(qreq_, cm_list_SVER)
        >>> #cm_list = cm_list_VSONE
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     figtitle = 'FIXME USE SUBSET OF CFGDICT'  # ut.dict_str(rrvsone_cfgdict, newlines=False)
        ...     show_all_ranked_matches(qreq_.ibs, cm_list_VSONE, figtitle=figtitle)
        ...     pt.show_if_requested()
    """
    config = qreq_.qparams
    # Filter down to a shortlist
    nNameShortlist = qreq_.qparams.nNameShortlistVsone
    nAnnotPerName  = qreq_.qparams.nAnnotPerNameVsOne
    scoring.score_chipmatch_list(qreq_, cm_list_SVER, 'nsum')
    prepare_vsmany_chipmatch(qreq_, cm_list_SVER)
    cm_shortlist = scoring.make_chipmatch_shortlists(qreq_, cm_list_SVER, nNameShortlist, nAnnotPerName)
    #return {cm.qaid: cm.to_cmtup_old() for cm in cm_shortlist }
    # Execute vsone reranking
    _prog = functools.partial(ut.ProgressIter, nTotal=len(cm_shortlist),
                              lbl='VSONE RERANKING', freq=1)
    cm_list_VSONE = [
        single_vsone_rerank(qreq_, prior_cm, config)
        for prior_cm in _prog(cm_shortlist)
    ]
    return cm_list_VSONE


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


def get_selectivity_score_list(qreq_, qaid, daid_list, fm_list, cos_power):
    vecs1 = qreq_.ibs.get_annot_vecs(qaid, qreq_=qreq_)
    vecs2_list = qreq_.ibs.get_annot_vecs(daid_list, qreq_=qreq_)
    vecs1_m_iter = (vecs1.take(fm.T[0], axis=0) for fm in fm_list)
    vecs2_m_iter = (vecs2.take(fm.T[1], axis=0) for fm, vecs2 in zip(fm_list, vecs2_list))
    # Rescore constrained using selectivity function
    fs_list = [scoring.sift_selectivity_score(vecs1_m, vecs2_m, cos_power)
               for vecs1_m, vecs2_m in zip(vecs1_m_iter, vecs2_m_iter)]
    return fs_list


@profile
def sver_fmfs_merge(qreq_, qaid, daid_list, fmfs_merge, config={}):
    from vtool import spatial_verification as sver
    # params
    # TODO paramaterize better
    xy_thresh    = config.get('xy_thresh') * 1.5
    scale_thresh = config.get('scale_thresh') * 2
    ori_thresh   = config.get('ori_thresh') * 2
    min_nInliers = config.get('min_nInliers')
    # input data
    fm_list, fs_list = fmfs_merge
    fsv_list   = matching.ensure_fsv_list(fs_list)
    kpts1      = qreq_.ibs.get_annot_kpts(qaid, qreq_=qreq_)
    kpts2_list = qreq_.ibs.get_annot_kpts(daid_list, qreq_=qreq_)
    chip2_dlen_sqrd_list = qreq_.ibs.get_annot_chip_dlensqrd(daid_list, qreq_=qreq_)  # chip diagonal length
    res_list = []
    # homog_inliers
    for kpts2, chip2_dlen_sqrd, fm, fsv in zip(kpts2_list, chip2_dlen_sqrd_list, fm_list, fsv_list):
        sv_tup = sver.spatially_verify_kpts(
            kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh,
            chip2_dlen_sqrd, min_nInliers,
            returnAff=False)
        if sv_tup is not None:
            (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) = sv_tup
            fm_SV = fm.take(homog_inliers, axis=0)
            fsv_SV = fsv.take(homog_inliers, axis=0)
        else:
            fm_SV = np.empty((0, 2), dtype=hstypes.FM_DTYPE)
            fsv_SV = np.empty((0, fsv.shape[1]))
            H = np.eye(3)
        res_list.append((fm_SV, fsv_SV, H))

    fm_list_SV  = ut.get_list_column(res_list, 0)
    fsv_list_SV = ut.get_list_column(res_list, 1)
    H_list      = ut.get_list_column(res_list, 2)
    fmfs_merge_SV = (fm_list_SV, fsv_list_SV)
    return fmfs_merge_SV, H_list


@profile
def refine_matches(qreq_, prior_cm, config={}):
    """
    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog --sver_unconstrained
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog --sver_constrained&
        python -m ibeis.model.hots.vsone_pipeline --test-refine_matches --show --homog --sver_constrained --sver_unconstrained&

        python dev.py -t custom:rrvsone_on=True --allgt --index 0:40 --db PZ_MTEST --print-confusion-stats --print-scorediff-mat-stats
        python dev.py -t custom:rrvsone_on=True custom --allgt --index 0:40 --db PZ_MTEST --print-confusion-stats --print-scorediff-mat-stats

        python dev.py -t custom:rrvsone_on=True,constrained_coeff=0 custom --qaid 12 --db PZ_MTEST \
            --print-confusion-stats --print-scorediff-mat-stats --show --va

        python dev.py -t custom:rrvsone_on=True,constrained_coeff=0,maskscore_mode=kpts --qaid 12 --db PZ_MTEST  \
            --print-confusion-stats --print-scorediff-mat-stats --show --va

        python dev.py -t custom:rrvsone_on=True,maskscore_mode=kpts --qaid 12 --db PZ_MTEST \
                --print-confusion-stats --print-scorediff-mat-stats --show --va

        use_kptscov_scoring

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching('PZ_MTEST')
        >>> config = qreq_.qparams
        >>> unscored_cm = refine_matches(qreq_, prior_cm, config)
        >>> unscored_cm.print_csv(ibs=ibs)
        >>> prior_cm.print_csv(ibs=ibs)
        >>> prior_cm.testshow_ranked(qreq_, figtitle=qreq_.qparams.query_cfgstr)
    """
    # THIS CAUSES THE ISSUE
    #prior_cm.fs_list = prior_cm.fsv_list
    #return prior_cm
    if qreq_.ibs.get_annot_num_feats(prior_cm.qaid, qreq_=qreq_) == 0:
        num_daids = len(prior_cm.daid_list)
        empty_unscored_cm = chip_match.ChipMatch2.from_unscored(
            prior_cm, ut.alloc_lists(num_daids), ut.alloc_lists(num_daids), ut.alloc_lists(num_daids))
        return empty_unscored_cm

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
        unc_match_results = compute_query_unconstrained_matches(qreq_, qaid, daid_list, config)
        fm_unc_list, fs_unc_list, fm_norm_unc_list = unc_match_results
        # Merge into result
        fmfs_unc = (fm_unc_list, fs_unc_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_unc)

        # We have the option of spatially verifying the merged results from the
        # prior and the new unconstrained matches.
        if sver_unconstrained:
            fmfs_merge, H_list = sver_fmfs_merge(qreq_, qaid, daid_list, fmfs_merge, config)

    if constrained_coeff != 0:
        scr_match_results = compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, config)
        fm_scr_list, fs_scr_list, fm_norm_scr_list = scr_match_results
        fs_scr_list = get_selectivity_score_list(qreq_, qaid, daid_list, fm_scr_list, scr_cos_power)
        # Merge into result
        fmfs_scr = (fm_scr_list, fs_scr_list)
        fmfs_merge = marge_matches_lists(fmfs_merge, fmfs_scr)
        col_coeff_list.append(constrained_coeff)

        # Another optional round of spatial verification
        if sver_constrained:
            fmfs_merge, H_list = sver_fmfs_merge(qreq_, qaid, daid_list, fmfs_merge, config)

    coeffs = np.array(col_coeff_list)
    assert np.isclose(coeffs.sum(), 1.0), 'must sum to 1 coeffs = %r' % (coeffs)
    # merge different match types
    fm_list, fsv_list = fmfs_merge
    # apply linear combination
    fs_list = [(np.nan_to_num(fsv) * coeffs[None, :]).sum(axis=1) for fsv in fsv_list]

    unscored_cm = chip_match.ChipMatch2.from_unscored(prior_cm, fm_list, fs_list, H_list)
    return unscored_cm


@profile
def single_vsone_rerank(qreq_, prior_cm, config={}):
    r"""
    Runs a single vsone-pair (query, daid_list)

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_rerank
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_rerank --show
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_rerank --show --qaid 18

    Example1:
        >>> # ENABLE_DOCTEST
        >>> import plottool as pt
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching('PZ_MTEST')
        >>> config = qreq_.qparams
        >>> rerank_cm = single_vsone_rerank(qreq_, prior_cm, config)
        >>> rerank_cm.print_rawinfostr()
        >>> rerank_cm.print_csv()
        >>> print(rerank_cm.score_list)
        >>> ut.quit_if_noshow()
        >>> prior_cm.score_nsum(qreq_)
        >>> prior_cm.show_ranked_matches(qreq_, fnum=1, figtitle='prior')
        >>> rerank_cm.show_ranked_matches(qreq_, fnum=2, figtitle='rerank')
        >>> pt.show_if_requested()
    """
    #print('==================')
    unscored_cm = refine_matches(qreq_, prior_cm, config)

    if qreq_.qparams.covscore_on:
        unscored_cm.score_coverage(qreq_)
    else:
        # Apply score weights
        data_baseline_weight_list = scoring.get_annot_kpts_baseline_weights(qreq_, unscored_cm.daid_list, config)
        query_baseline_weight = scoring.get_annot_kpts_baseline_weights(qreq_, [unscored_cm.qaid], config)[0]
        qfx_list = [fm.T[0] for fm in unscored_cm.fm_list]
        dfx_list = [fm.T[1] for fm in unscored_cm.fm_list]

        qfweight_list = [query_baseline_weight.take(qfx) for qfx in qfx_list]
        dfweight_list = [data_baseline_weight.take(dfx)
                         for dfx, data_baseline_weight in zip(dfx_list, data_baseline_weight_list)]
        fweight_list = [np.sqrt(qfweight * dfweight) for qfweight, dfweight in
                        zip(qfweight_list, dfweight_list)]
        # hack in the distinctivness and fgweights
        unscored_cm.fs_list = [fs * fweight for fs, fweight in zip(unscored_cm.fs_list, fweight_list)]
        unscored_cm.fsv_list = matching.ensure_fsv_list(unscored_cm.fs_list)

        #queryside_weights =
        #dfweights_list =
        # hack
        unscored_cm.score_nsum(qreq_)

    # Convert our one score to a score vector here
    rerank_cm = unscored_cm
    rerank_cm.fsv_list = matching.ensure_fsv_list(rerank_cm.fs_list)
    return rerank_cm


def quick_vsone_flann(flann_cachedir, qvecs):
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    use_cache = save = False and ut.is_developer()
    flann = vt.flann_cache(qvecs, flann_cachedir, flann_params=flann_params,
                           quiet=True, verbose=False, use_cache=use_cache, save=save)
    return flann


@profile
def compute_query_unconstrained_matches(qreq_, qaid, daid_list, config):
    """

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show --shownorm
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_unconstrained_matches --show --shownorm --homog

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching()
        >>> config = qreq_.qparams
        >>> qaid, daid_list, H_list = ut.dict_take(prior_cm, ['qaid', 'daid_list', 'H_list'])
        >>> match_results = compute_query_unconstrained_matches(qreq_, qaid, daid_list, config)
        >>> fm_RAT_list, fs_RAT_list, fm_norm_RAT_list = match_results
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     idx = ut.listfind(ibs.get_annot_nids(daid_list), ibs.get_annot_nids(qaid))
        ...     args = (ibs, qaid, daid_list, fm_RAT_list, fs_RAT_list, fm_norm_RAT_list, H_list)
        ...     show_single_match(*args, index=idx)
        ...     pt.set_title('unconstrained')
        ...     pt.show_if_requested()
    """
    unc_ratio_thresh = config['unc_ratio_thresh']
    #, .625)
    qvecs = qreq_.ibs.get_annot_vecs(qaid, qreq_=qreq_)
    dvecs_list = qreq_.ibs.get_annot_vecs(daid_list, qreq_=qreq_)
    print(len(qvecs))
    flann = quick_vsone_flann(qreq_.ibs.get_flann_cachedir(), qvecs)
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
    fm_RAT_list = ut.get_list_column(scrtup_list, 0)
    fs_RAT_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_RAT_list = ut.get_list_column(scrtup_list, 2)
    match_results = fm_RAT_list, fs_RAT_list, fm_norm_RAT_list
    return match_results


@profile
def compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, config):
    """

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show --shownorm
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show --shownorm --homog
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show --homog
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show --homog --index 2

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, prior_cm = plh.testdata_matching()
        >>> config = qreq_.qparams
        >>> print(config.query_cfgstr)
        >>> qaid, daid_list, H_list = ut.dict_take(prior_cm, ['qaid', 'daid_list', 'H_list'])
        >>> match_results = compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, config)
        >>> fm_SCR_list, fs_SCR_list, fm_norm_SCR_list = match_results
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     idx = ut.listfind(ibs.get_annot_nids(daid_list), ibs.get_annot_nids(qaid))
        ...     index = ut.get_argval('--index', int, idx)
        ...     args = (ibs, qaid, daid_list, fm_SCR_list, fs_SCR_list, fm_norm_SCR_list, H_list)
        ...     show_single_match(*args, index=index)
        ...     pt.set_title('constrained')
        ...     pt.show_if_requested()
    """
    scr_ratio_thresh     = config.get('scr_ratio_thresh', .1)
    scr_K                = config.get('scr_K', 7)
    scr_match_xy_thresh  = config.get('scr_match_xy_thresh', .05)
    scr_norm_xy_min      = config.get('scr_norm_xy_min', 0.1)
    scr_norm_xy_max      = config.get('scr_norm_xy_max', 1.0)
    scr_norm_xy_bounds = (scr_norm_xy_min, scr_norm_xy_max)
    vecs1 = qreq_.ibs.get_annot_vecs(qaid, qreq_=qreq_)
    kpts1 = qreq_.ibs.get_annot_kpts(qaid)
    vecs2_list = qreq_.ibs.get_annot_vecs(daid_list, qreq_=qreq_)
    kpts2_list = qreq_.ibs.get_annot_kpts(daid_list, qreq_=qreq_)
    chip2_dlen_sqrd_list = qreq_.ibs.get_annot_chip_dlensqrd(daid_list, qreq_=qreq_)  # chip diagonal length
    # build flann for query vectors
    flann = quick_vsone_flann(qreq_.ibs.get_flann_cachedir(), vecs1)
    # match database chips to query chip
    scr_kwargs = {
        'scr_K'            : scr_K,
        'match_xy_thresh'  : scr_match_xy_thresh,
        'norm_xy_bounds'   : scr_norm_xy_bounds,
        'scr_ratio_thresh' : scr_ratio_thresh,
        'fm_dtype'         : hstypes.FM_DTYPE,
        'fs_dtype'         : hstypes.FS_DTYPE,
    }
    print('scr_kwargs = ' + ut.dict_str(scr_kwargs))
    # Homographys in H_list map image1 space into image2 space
    scrtup_list = [
        matching.spatially_constrained_ratio_match(
            flann, vecs2, kpts1, kpts2, H, chip2_dlen_sqrd, **scr_kwargs)
        for vecs2, kpts2, chip2_dlen_sqrd, H in
        zip(vecs2_list, kpts2_list, chip2_dlen_sqrd_list, H_list)]
    fm_SCR_list = ut.get_list_column(scrtup_list, 0)
    fs_SCR_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_SCR_list = ut.get_list_column(scrtup_list, 2)
    match_results = fm_SCR_list, fs_SCR_list, fm_norm_SCR_list
    return match_results


# -----------------------------
# GRIDSEARCH
# -----------------------------


COVKPTS_DEFAULT = vt.coverage_kpts.COVKPTS_DEFAULT
COVGRID_DEFAULT = vt.coverage_grid.COVGRID_DEFAULT

OTHER_RRVSONE_PARAMS = ut.ParamInfoList('OTHERRRVSONE', [
    #ut.ParamInfo('fs_lnbnn_min', .0001),
    #ut.ParamInfo('fs_lnbnn_max', .05),
    #ut.ParamInfo('fs_lnbnn_power', 1.0),
    ut.ParamInfo('fs_lnbnn_min', 0.0, hideif=0.0),
    ut.ParamInfo('fs_lnbnn_max', 1.0, hideif=1.0),
    ut.ParamInfo('fs_lnbnn_power', 1.0, hideif=1.0),
    ut.ParamInfoBool('covscore_on', False, hideif=True),
    ut.ParamInfo('dcvs_on', False),
])


SHORTLIST_DEFAULTS = ut.ParamInfoList('SLIST', [
    ut.ParamInfo('nNameShortlistVsone', 20, 'nNm='),
    ut.ParamInfo('nAnnotPerNameVsOne', 3, 'nApN='),
])

# matching types
COEFF_DEFAULTS = ut.ParamInfoList('COEFF', [
    ut.ParamInfo('prior_coeff', .6, 'prior_coeff='),
    ut.ParamInfo('unconstrained_coeff',    .4, 'unc_coeff='),
    ut.ParamInfo('constrained_coeff',     0.0, 'scr_coeff=', hideif=0.0),
    ut.ParamInfo('sver_unconstrained',   True, 'sver_unc=', hideif=lambda cfg: cfg['unconstrained_coeff'] <= 0),
    ut.ParamInfo('sver_constrained',    False, 'sver_scr=', hideif=lambda cfg: cfg['constrained_coeff'] <= 0),
    ut.ParamInfo('maskscore_mode', 'grid', 'cov='),
]
)

UNC_DEFAULTS = ut.ParamInfoList('UNC', [
    ut.ParamInfo('unc_ratio_thresh', .8, 'uncRat=', varyvals=[.625, .82, .9, 1.0, .8]),
])


def scr_constraint_func(cfg):
    if cfg['scr_norm_xy_min'] >= cfg['scr_norm_xy_max']:
        return False

SCR_DEFAULTS = ut.ParamInfoList('SCR', [
    ut.ParamInfo('scr_match_xy_thresh', .15, 'xy=',
                 varyvals=[.05, 1.0, .1], varyslice=slice(0, 2)),
    ut.ParamInfo('scr_norm_xy_min', 0.1, '',
                 varyvals=[0, .1, .2], varyslice=slice(0, 2)),
    ut.ParamInfo('scr_norm_xy_max', 1.0, '',
                 varyvals=[1, .3], varyslice=slice(0, 2)),
    ut.ParamInfo('scr_ratio_thresh', .95, 'scrRat=',
                 varyvals=[.625, .3, .9, 0.0, 1.0], varyslice=slice(0, 1)),
    ut.ParamInfo('scr_K', 7, 'scK',
                 varyvals=[7, 2], varyslice=slice(0, 2)),
],
    scr_constraint_func,
    hideif=lambda cfg: cfg['constrained_coeff'] <= 0)


def gridsearch_single_vsone_rerank():
    r"""

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_single_vsone_rerank --show
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_single_vsone_rerank --show --testindex 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_unconstrained_matches()
        >>> pt.show_if_requested()
    """
    ibs, qreq_, prior_cm = plh.testdata_matching()
    config = qreq_.qparams
    import plottool as pt
    fnum = pt.ensure_fnum(None)
    # Make configuration for every parameter setting
    cfgdict_ = dict(prescore_method='nsum', score_method='nsum', sver_weighting=True)
    cfgdict_['rrvsone_on'] = True
    # HACK TO GET THE DATA WE WANT WITHOUT UNNCESSARY COMPUTATION
    # Get pipeline testdata for this configuration
    ibs, qreq_ = plh.get_pipeline_testdata(
        cfgdict=cfgdict_, qaid_list=[1], daid_list='all', defaultdb='PZ_MTEST',
        cmdline_ok=True, preload=False)
    qaid_list = qreq_.get_external_qaids().tolist()
    qaid = qaid_list[0]
    daid_list = qreq_.ibs.get_annot_groundtruth(qaid)[0:1]
    #
    cfgdict_list, cfglbl_list = COEFF_DEFAULTS.get_gridsearch_input(defaultslice=slice(0, 10))
    assert len(cfgdict_list) > 0
    #config = qreq_.qparams
    cfgresult_list = [
        single_vsone_rerank(qreq_, prior_cm, config)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='rerank')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list, index=index)

    def onclick_func(fm_list, fs_list, fm_norm_list):
        from ibeis.viz.interact import interact_matches
        aid2 = daid_list[index]
        cm = chip_match.ChipMatch2(qaid=qaid, daid_list=daid_list, fm_list=fm_list, fsv_list=fs_list)
        cm.fs_list = fs_list
        interact_matches.MatchInteraction(ibs, cm, aid2=aid2, fnum=None)

    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore', onclick_func=onclick_func)
    pt.iup()


def gridsearch_constrained_matches():
    r"""
    Search spatially constrained matches

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_constrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_constrained_matches --show --qaid 41
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_constrained_matches --show --testindex 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> gridsearch_constrained_matches()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    fnum = pt.ensure_fnum(None)
    # Make configuration for every parameter setting
    cfgdict_list, cfglbl_list = SCR_DEFAULTS.get_gridsearch_input(defaultslice=slice(0, 10))
    #fname = None  # 'easy1.png'
    ibs, qreq_, prior_cm = plh.testdata_matching()
    qaid      = prior_cm.qaid
    daid_list = prior_cm.daid_list
    H_list    = prior_cm.H_list
    #config = qreq_.qparams
    cfgresult_list = [
        compute_query_constrained_matches(qreq_, qaid, daid_list, H_list, cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='scr match')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list, H_list=H_list, index=index)

    def onclick_func(fm_list, fs_list, fm_norm_list):
        from ibeis.viz.interact import interact_matches
        aid2 = daid_list[index]
        cm = chip_match.ChipMatch2(qaid=qaid, daid_list=daid_list, fm_list=fm_list, fsv_list=fs_list)
        cm.fs_list = fs_list
        interact_matches.MatchInteraction(ibs, cm, aid2=aid2, fnum=None)

    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore', onclick_func=onclick_func)

    #if use_separate_norm:
    #    ut.interact_gridsearch_result_images(
    #        functools.partial(show_single_match, use_separate_norm=True), cfgdict_list, cfglbl_list,
    #        cfgresult_list, fnum=fnum + 1, figtitle='constrained ratio match', unpack=True,
    #        max_plots=25, scorelbl='sumscore')
    pt.iup()


def gridsearch_unconstrained_matches():
    r"""
    Search unconstrained ratio test vsone match

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 27
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 41 --daid_list 39
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 40 --daid_list 39
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --testindex 2


        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 117 --daid_list 118 --db PZ_Master0
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_unconstrained_matches --show --qaid 117 --daid_list 118 --db PZ_Master0 --rotation_invariance

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_unconstrained_matches()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    fnum = pt.ensure_fnum(None)
    # Make configuration for every parameter setting
    cfgdict_ = dict(prescore_method='nsum', score_method='nsum', sver_weighting=True)
    cfgdict_['rrvsone_on'] = True
    # HACK TO GET THE DATA WE WANT WITHOUT UNNCESSARY COMPUTATION
    # Get pipeline testdata for this configuration
    ibs, qreq_ = plh.get_pipeline_testdata(
        cfgdict=cfgdict_, qaid_list=[1], daid_list='gt', defaultdb='PZ_MTEST',
        cmdline_ok=True, preload=False)
    qaid_list = qreq_.get_external_qaids().tolist()
    qaid = qaid_list[0]
    daid_list = qreq_.get_external_query_groundtruth(qaid)[0:1]
    #
    cfgdict_list, cfglbl_list = UNC_DEFAULTS.get_gridsearch_input(defaultslice=slice(0, 10))
    assert len(cfgdict_list) > 0
    #config = qreq_.qparams
    cfgresult_list = [
        compute_query_unconstrained_matches(qreq_, qaid, daid_list, cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='scr match')
    ]
    # which result to look at
    index = ut.get_argval('--testindex', int, 0)
    score_list = [scrtup[1][index].sum() for scrtup in cfgresult_list]
    #score_list = [scrtup[1][0].sum() / len(scrtup[1][0]) for scrtup in cfgresult_list]
    showfunc = functools.partial(show_single_match, ibs, qaid, daid_list, index=index)

    def onclick_func(fm_list, fs_list, fm_norm_list):
        from ibeis.viz.interact import interact_matches
        aid2 = daid_list[index]
        cm = chip_match.ChipMatch2(qaid=qaid, daid_list=daid_list, fm_list=fm_list, fsv_list=fs_list)
        cm.fs_list = fs_list
        interact_matches.MatchInteraction(ibs, cm, aid2=aid2, fnum=None)

    ut.interact_gridsearch_result_images(
        showfunc, cfgdict_list, cfglbl_list,
        cfgresult_list, score_list=score_list, fnum=fnum,
        figtitle='constrained ratio match', unpack=True,
        max_plots=25, scorelbl='sumscore', onclick_func=onclick_func)
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
    if H_list is None:
        H1 = None
    else:
        H1 = H_list[index]
    #H1 = None  # uncomment to see warping
    show_matches(ibs, qaid, daid, fm, fs=fs, H1=H1, fm_norm=fm_norm, **kwargs)


def show_matches(ibs, qaid, daid, fm, fs=None, fm_norm=None,
                               H1=None, fnum=None, pnum=None, **kwargs):
    from ibeis.viz import viz_matches
    if not ut.get_argflag('--homog'):
        H1 = None

    viz_matches.show_matches2(ibs, qaid, daid, fm=fm, fs=fs, fm_norm=fm_norm, ori=True,
                              H1=H1, fnum=fnum, pnum=pnum, show_name=False, **kwargs)
    #else:
    #    from ibeis.viz.interact import interact_matches
    #    cm = chip_match.ChipMatch2(qaid, [daid], [fm], [fs])
    #    interact_matches.MatchInteraction(ibs, cm, fnum=None, aid2=daid)

    #pt.set_title('score = %.3f' % (score,))


def show_ranked_matches(ibs, cm, fnum=None):
    import plottool as pt
    qaid = cm.qaid
    if fnum is None:
        fnum = pt.next_fnum()
    CLIP_TOP = 6
    top_idx_list  = ut.listclip(cm.argsort(), CLIP_TOP)
    nRows, nCols  = pt.get_square_row_cols(len(top_idx_list), fix=True)
    next_pnum     = pt.make_pnum_nextgen(nRows, nCols)
    for idx in top_idx_list:
        daid  = cm.daid_list[idx]
        fm    = cm.fm_list[idx]
        fsv   = cm.fsv_list[idx]
        fs    = fsv.prod(axis=1)
        H1 = cm.H_list[idx]
        pnum = next_pnum()
        #with ut.EmbedOnException():
        show_matches(ibs, qaid, daid, fm=fm, fs=fs, H1=H1, fnum=fnum, pnum=pnum)
        score = None if cm.score_list is None else cm.score_list[idx]
        if score is not None:
            pt.set_title('score = %.3f' % (score,))
        else:
            pt.set_title('score = %r' % (score,))


def show_all_ranked_matches(ibs, cm_list, fnum_offset=0, figtitle=''):
    """ helper """
    import plottool as pt
    for fnum_, cm in enumerate(cm_list):
        #cm.foo()
        fnum = fnum_ + fnum_offset
        show_ranked_matches(ibs, cm, fnum)
        #pt.figure(fnum=fnum, doclf=True, docla=True)
        pt.set_figtitle('qaid=%r %s' % (cm.qaid, figtitle))


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
