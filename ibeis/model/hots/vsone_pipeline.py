"""
special pipeline for vsone specific functions

Current Issues:
    * getting feature distinctiveness is too slow, we can either try a different
      model, or precompute feature distinctiveness.

      - we can reduce the size of the vsone shortlist

TODOLIST:
    * Precompute distinctivness
    * keep feature matches from vsmany (allow prior_fm)
    * Each keypoint gets
      - foregroundness
      - global distinctivness (databasewide) LNBNN
      - local distinctivness (imagewide) RATIO
      - regional match quality (descriptor based) COS
    * Circular hesaff keypoints
    * Asymetric weight scoring

    * FIX BUGS IN score_chipmatch_nsum FIRST THING TOMORROW.
     dict keys / vals are being messed up. very inoccuous

"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import vtool as vt
#from ibeis.model.hots import neighbor_index
from ibeis.model.hots import name_scoring
from ibeis.model.hots import hstypes
from ibeis.model.hots import scoring
#import pyflann
#from ibeis.model.hots import coverage_kpts
from vtool import constrained_matching
from vtool import matching  # NOQA
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
import utool as ut
from six.moves import zip, range  # NOQA
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[vsonepipe]', DEBUG=False)


#@profile
def vsone_reranking(qreq_, qaid2_chipmatch, verbose=False):
    """
    Driver function for vsone reranking

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show

        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking
        utprof.py -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking

    Example:
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
        >>> qaid_list = [1, 4, 6]
        >>> #qaid_list = qaid_list[2:3]
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict, qaid_list=qaid_list)
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> # qaid2_chipmatch = ut.dict_subset(qaid2_chipmatch, [6])
        >>> qaid2_chipmatch_VSONE = vsone_reranking(qreq_, qaid2_chipmatch)
        >>> #qaid2_chipmatch = qaid2_chipmatch_VSONE
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     show_top_chipmatches(qreq_.ibs, qaid2_chipmatch_VSONE)
        >>>     pt.show_if_requested()
    """
    ibs = qreq_.ibs
    config = qreq_.qparams
    # First find a shortlist to execute vsone reranking on
    shortlist_tup = make_rerank_pair_shortlist(qreq_, qaid2_chipmatch)
    qaid_list, daids_list, Hs_list, prior_chipmatch_list = shortlist_tup
    # Then execute vsone reranking
    prior_filtkey_list = qreq_.qparams.get_postsver_filtkey_list()
    qaid2_chipmatch_VSONE = execute_vsone_reranking(
        ibs, config, qaid_list, daids_list, Hs_list, prior_chipmatch_list, prior_filtkey_list)
    return qaid2_chipmatch_VSONE


@profile
def make_rerank_pair_shortlist(qreq_, qaid2_chipmatch):
    """
    Makes shortlists for vsone reranking

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-make_rerank_pair_shortlist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
        >>> # execute test function
        >>> qaid_list, top_aids_list, top_Hs_list, prior_chipmatch_list = make_rerank_pair_shortlist(qreq_, qaid2_chipmatch)
        >>> # verify results
        >>> top_aid_list = top_aids_list[0]
        >>> top_nid_list = ibs.get_annot_name_rowids(top_aid_list)
        >>> print('top_aid_list = %r' % (top_aid_list,))
        >>> print('top_nid_list = %r' % (top_nid_list,))
        >>> assert top_nid_list.index(1) == 0, 'name 1 should be rank 1'
        >>> max_num_rerank = qreq_.qparams.nNameShortlistVsone * qreq_.qparams.nAnnotPerName
        >>> min_num_rerank = qreq_.qparams.nNameShortlistVsone
        >>> ut.assert_inbounds(len(top_nid_list), min_num_rerank, max_num_rerank, 'incorrect number in shortlist')

    Ignore:
        #vsone_query_pairs = make_rerank_pair_shortlist(qreq_, qaid2_chipmatch)
        #ibs = qreq_.ibs
        #for fnum, vsone_pair_tup in enumerate(vsone_query_pairs):
        #    (qaid, daid_list, H_list) = vsone_pair_tup
        #    nRows, nCols = pt.get_square_row_cols(len(daid_list))
        #    next_pnum = pt.make_pnum_nextgen(*daid_list)
        #    for daid in daid_list:
        #        fm = qaid2_chipmatch[qaid].aid2_fm[daid]
        #        H = qaid2_chipmatch[qaid].aid2_H[daid]
        #        viz_sver.show_constrained_match(ibs, qaid, daid, H, fm, pnum=next_pnum())
    """
    from ibeis.model.hots import pipeline
    ibs = qreq_.ibs
    score_method = qreq_.qparams.score_method
    assert score_method == 'nsum'
    # TODO: paramaterize
    # Params: the max number of top names to get and the max number of
    # annotations per name to verify against
    nNameShortlistVsone = qreq_.qparams.nNameShortlistVsone
    nAnnotPerName       = qreq_.qparams.nAnnotPerName
    print('vsone reranking. ')
    qaid_list = list(six.iterkeys(qaid2_chipmatch))
    chipmatch_list = ut.dict_take(qaid2_chipmatch, qaid_list)
    daids_list = []
    Hs_list = []
    prior_chipmatch_list = []
    for qaid, chipmatch in zip(qaid_list, chipmatch_list):
        daid2_prescore = pipeline.score_chipmatch(qreq_, qaid, chipmatch, score_method)
        daid_list      = np.array(list(daid2_prescore.keys()))
        prescore_arr   = np.array(list(daid2_prescore.values()))
        # HACK POPULATE AID2_SCORE FIELD IN CHIPMATCH TUPLE
        ut.dict_assign(chipmatch.aid2_score, daid_list, prescore_arr)
        #
        nscore_tup = name_scoring.group_scores_by_name(ibs, daid_list, prescore_arr)
        (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscore_tup
        # Clip number of names
        top_aids_list_  = ut.listclip(sorted_aids, nNameShortlistVsone)
        # Clip number of annots per name
        top_aids_list = [ut.listclip(aids, nAnnotPerName) for aids in top_aids_list_]
        top_aids = ut.flatten(top_aids_list)
        prior_chipmatch = hstypes.chipmatch_subset(chipmatch, top_aids)
        top_H_list   = ut.dict_take(chipmatch.aid2_H, top_aids)
        # append shortlist results for this query aid
        prior_chipmatch_list.append(prior_chipmatch)
        daids_list.append(top_aids)
        Hs_list.append(top_H_list)
    shortlist_tup = qaid_list, daids_list, Hs_list, prior_chipmatch_list
    return shortlist_tup


#@profile
def execute_vsone_reranking(ibs, config, qaid_list, daids_list_, Hs_list,
                            prior_chipmatch_list, prior_filtkey_list):
    r"""
    runs several pairs of (qaid, daids) vsone matches
    For each qaid, daids pair in the lists, execute a query
    """
    progkw = dict(lbl='VSONE RERANKING', freq=1)
    daid_score_fm_fsv_tup_list = [
        single_vsone_query(ibs, qaid, daid_list, H_list, prior_chipmatch, config, prior_filtkey_list)
        for (qaid, daid_list, H_list, prior_chipmatch) in
        ut.ProgressIter(
            zip(qaid_list, daids_list_, Hs_list, prior_chipmatch_list),
            nTotal=len(qaid_list), **progkw)
    ]
    # Unpack results into their respective types
    daids_list   = ut.get_list_column(daid_score_fm_fsv_tup_list, 0)
    scores_list  = ut.get_list_column(daid_score_fm_fsv_tup_list, 1)
    fms_list     = ut.get_list_column(daid_score_fm_fsv_tup_list, 2)
    fsvs_list    = ut.get_list_column(daid_score_fm_fsv_tup_list, 3)
    #vsone_res_tup = (daids_list, scores_list, fms_list, fsvs_list)
    #(daid_list, scores_list, fms_list, fsvs_list) = vsone_res_tup
    # Format the output into chipmatches
    chipmatch_VSONE_list = []
    for daids, scores, fms, fsvs, Hs in zip(daids_list, scores_list, fms_list, fsvs_list, Hs_list):
        fks = [np.ones(len(fm), dtype=hstypes.FK_DTYPE) for fm in fms]
        aid2_fm    = dict(zip(daids, fms))
        aid2_fsv   = dict(zip(daids, fsvs))
        aid2_fk    = dict(zip(daids, fks))
        aid2_score = dict(zip(daids, scores))
        aid2_H     = dict(zip(daids, Hs))
        chipmatch_VSONE = hstypes.ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
        #cm = hstypes.ChipMatch2(chipmatch_VSONE)
        #fm.foo()
        chipmatch_VSONE_list.append(chipmatch_VSONE)
    qaid2_chipmatch_VSONE = dict(zip(qaid_list, chipmatch_VSONE_list))
    #qaid2_scores = dict(zip(qaid_list, scores_list))
    return qaid2_chipmatch_VSONE


@profile
def single_vsone_query(ibs, qaid, daid_list, H_list, prior_chipmatch=None,
                        config={}, prior_filtkey_list=None):
    r"""
    Runs a single vsone-pair query

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid (int):  query annotation id
        daid_list (list):
        H_list (list):

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_query:0

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid = 1
        >>> daid_list = [3, 2]
        >>> H_list = [
        ...  np.array([[ -4.68815126e-01,   7.80306795e-02,  -2.23674587e+01],
        ...            [  4.54394231e-02,  -7.67438835e-01,   5.92158624e+01],
        ...            [  2.12918867e-04,  -8.64851418e-05,  -6.21472492e-01]]),
        ...  np.array([[  5.11319128e-01,  -2.69211436e-04,  -3.18079183e+01],
        ...            [ -5.97449121e-02,   4.67044573e-01,   5.27655556e+01],
        ...            [  1.06650025e-04,   8.70310639e-05,   5.28664052e-01]])
        ... ]
        >>> daid_fm_fs_score_tup = single_vsone_query(ibs, qaid, daid_list, H_list)
        >>> daid_list, score_list, fm_list, fsv_list = daid_fm_fs_score_tup
        >>> print(score_list)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, qaid, daid_list, H_list, prior_chipmatch, prior_filtkey_list = testdata_matching()
        >>> config = qreq_.qparams
        >>> daid_fm_fs_score_tup = single_vsone_query(ibs, qaid, daid_list, H_list)
        >>> daid_list, score_list, fm_list, fsv_list = daid_fm_fs_score_tup
        >>> print(score_list)

    Ignore:
        from ibeis.viz import viz_sver
        import plottool as pt

        next_pnum = pt.make_pnum_nextgen(*pt.get_square_row_cols(len(daid_list)))
        for fm, daid, H in zip(fm_SCR_list, daid_list, H_list):
            viz_sver.show_constrained_match(ibs, qaid, daid, H, fm, pnum=next_pnum())
        pt.update()
    """
    from ibeis.model.hots import name_scoring
    #print('==================')
    vsone_fm_list, vsone_fs_list = compute_query_matches(ibs, qaid, daid_list, H_list, config=config)

    if prior_chipmatch is not None:
        # COMBINE VSONE WITH VSMANY MATCHES
        prior_fm_list = ut.dict_take(prior_chipmatch.aid2_fm, daid_list)
        prior_fsv_list = ut.dict_take(prior_chipmatch.aid2_fsv, daid_list)
        new_fm_list = []
        new_fs_list = []
        vecs1 = ibs.get_annot_vecs(qaid)
        _iter = zip(daid_list, vsone_fm_list, vsone_fs_list, prior_fm_list, prior_fsv_list)
        for daid, vsone_fm, vsone_fs, prior_fm, prior_fsv in _iter:
            vecs2 = ibs.get_annot_vecs(daid)
            fm_both, fs_both = merge_vsone_with_prior(
                vecs1, vecs2,  vsone_fm, vsone_fs, prior_fm, prior_fsv,
                prior_filtkey_list)
            new_fm_list.append(fm_both)
            new_fs_list.append(fs_both)
        fm_list = new_fm_list
        fs_list = new_fs_list
    else:
        fm_list = vsone_fm_list
        fs_list = vsone_fs_list

    #if qaid == 6:
    #    print(list(map(len, fm_list)))
    #    print(list(map(len, fs_list)))
    #    #print(list(map(len, fsv_list)))
    #    print(daid_list)
    #    ut.embed()

    cov_score_list = scoring.compute_grid_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config=config)
    # TODO: paramatarize
    #cov_score_list = scoring.compute_kpts_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config=config)
    NAME_SCORING = True
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
    num_filts = 1  # currently only using one vector here.
    num_matches_iter = map(len, fm_list)
    fsv_list = [fs.reshape((num_matches, num_filts))
                for fs, num_matches in zip(fs_list, num_matches_iter)]
    daid_score_fm_fsv_tup = (daid_list, score_list, fm_list, fsv_list)
    return daid_score_fm_fsv_tup


def merge_vsone_with_prior(vecs1, vecs2, vsone_fm, vsone_fs, prior_fm, prior_fsv, prior_filtkey_list):
    # Find relevant prior keys
    # TODO: normalized lnbnn scores are very very low
    # these need to be adjusted as well.
    lnbnn_index = prior_filtkey_list.index('lnbnn')
    prior_fs = prior_fsv.T[lnbnn_index].T

    # These indicies were found in both vsone and prior
    vsone_flags, prior_flags = vt.intersect2d_flags(vsone_fm, prior_fm)
    fm_both  = vsone_fm.compress(vsone_flags, axis=0)
    fm_vsone = vsone_fm.compress(~vsone_flags, axis=0)
    fm_prior = prior_fm.compress(~prior_flags, axis=0)
    #
    fs_both_vsone = vsone_fs.compress(vsone_flags)
    fs_both_prior = prior_fs.compress(prior_flags)
    fs_only_vsone = vsone_fs.compress(~vsone_flags)
    fs_only_prior = prior_fs.compress(~prior_flags)

    # Normalize prior features scores
    # TODO: normalize external to this function
    # TODO: parametarize
    fs_prior_min = .0001
    fs_prior_max = .05
    fs_prior_power = 1.0
    fs_only_prior_norm = vt.clipnorm(fs_only_prior, fs_prior_min, fs_prior_max)
    fs_only_prior_norm = np.power(fs_only_prior_norm, fs_prior_power, out=fs_only_prior_norm)

    # Merge feature matches
    fm_both = np.vstack([fm_both, fm_vsone, fm_prior])
    # Merge feature scores
    offset1 = len(fs_both_vsone)
    offset2 = offset1 + len(fm_vsone)
    offset3 = offset2 + len(fm_prior)
    fsv_both_cols = ['local', 'global', 'regional']
    fsv_both = np.full((len(fm_both), len(fsv_both_cols)), np.nan)
    fsv_both[0:offset1, 0]       = fs_both_vsone
    fsv_both[0:offset1, 1]       = fs_both_prior
    fsv_both[offset1:offset2, 0] = fs_only_vsone
    fsv_both[offset2:offset3, 1] = fs_only_prior_norm

    # find cosine angle between matching vectors
    vecs1_m = vecs1.take(fm_both.T[0], axis=0)
    vecs2_m = vecs2.take(fm_both.T[1], axis=0)
    # TODO: Param
    cos_power = 3.0
    fs_region = scoring.sift_selectivity_score(vecs1_m, vecs2_m, cos_power)
    fsv_both[:, 2] = fs_region

    # NEED TO MERGE THESE INTO A SINGLE SCORE
    # A USING LINEAR COMBINATION?
    #nan_weight = ~np.isnan(fsv_both)
    # for now simply take the maximum of the 3 scores
    fs_both = np.nan_to_num(fsv_both).max(axis=1)
    return fm_both, fs_both


@profile
def compute_query_matches(ibs, qaid, daid_list, H_list, config={}):
    r""" calls specified vsone matching routine for single (qaid, daids) pair """
    # TODO: implement unconstrained regular vsone
    fm_list, fs_list, fm_norm_SCR_list = compute_query_constrained_matches(ibs, qaid, daid_list, H_list, config)
    return fm_list, fs_list


@profile
def compute_query_constrained_matches(ibs, qaid, daid_list, H_list, config):
    """

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-compute_query_constrained_matches --show

    Example1:
        >>> # ENABLE_DOCTEST
        >>> import numpy.linalg as npl
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> ibs, qreq_, qaid, daid_list, H_list, prior_chipmatch, prior_filtkey_list = testdata_matching()
        >>> config = qreq_.qparams
        >>> fm_SCR_list, fs_SCR_list, fm_norm_SCR_list = compute_query_constrained_matches(ibs, qaid, daid_list, H_list, config)
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     idxs = np.where(np.array(ibs.get_annot_nids(daid_list)) == ibs.get_annot_nids(qaid))[0]
        >>>     idx = idxs[0]
        >>>     daid = daid_list[idx]
        >>>     H = H_list[idx]
        >>>     ut.depth_profile(fs_SCR_list)
        >>>     fm = fm_SCR_list[idx]
        >>>     fs = fs_SCR_list[idx]
        >>>     show_constrained_chipmatch(ibs, qaid, daid, fm, fs, H, fnum=1, pnum=None)
        >>>     pt.show_if_requested()
    """
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    match_xy_thresh  = config.get('scr_xy_thresh', .05)
    scr_ratio_thresh = config.get('scr_ratio_thresh', .7)
    normalizer_mode  = config.get('scr_normalizer_mode', 'far')
    K                = config.get('scr_K', 7)
    # QUERY INFO
    qvecs = ibs.get_annot_vecs(qaid)
    qkpts = ibs.get_annot_kpts(qaid)
    #query_dlen_sqrd = ibs.get_annot_chip_dlen_sqrd([qaid])[0]
    # BUILD FLANN FOR QUERY VECTORS
    flann_cachedir = ibs.get_flann_cachedir()
    use_cache = save = ut.is_developer()
    flann = vt.flann_cache(qvecs, flann_cachedir, flann_params=flann_params,
                           quiet=True, verbose=False, use_cache=use_cache, save=save)
    fm_SCR_list = []
    fs_SCR_list = []
    scrtup_list = []
    # DATABASE INFO
    dvecs_list = ibs.get_annot_vecs(daid_list)
    dkpts_list = ibs.get_annot_kpts(daid_list)
    diaglen_sqrd_list = ibs.get_annot_chip_dlen_sqrd(daid_list)
    dinfo_list = zip(dvecs_list, dkpts_list, diaglen_sqrd_list, H_list)
    for daid, dinfo in zip(daid_list, dinfo_list):
        # THIS CAN BE SWAPED WITH PURE RATIO TEST
        # ALSO, SVER CAN BE ADDED ON THE END
        dvecs, dkpts, dlen_sqrd2, H = dinfo
        scr_tup = spatially_constrained_ratio_match(
            flann, dvecs, qkpts, dkpts, H, dlen_sqrd2, match_xy_thresh,
            scr_ratio_thresh, K, normalizer_mode)
        scrtup_list.append(scr_tup)
    fm_SCR_list = ut.get_list_column(scrtup_list, 0)
    fs_SCR_list = ut.get_list_column(scrtup_list, 1)
    fm_norm_SCR_list = ut.get_list_column(scrtup_list, 2)
    return fm_SCR_list, fs_SCR_list, fm_norm_SCR_list


@profile
def spatially_constrained_ratio_match(flann, dvecs, qkpts, dkpts, H, dlen_sqrd2,
                                      match_xy_thresh, scr_ratio_thresh,  K, normalizer_mode):
    dfx2_qfx, _dfx2_dist = flann.nn_index(dvecs, num_neighbors=K, checks=800)
    dfx2_dist = np.divide(_dfx2_dist.astype(np.float64), hstypes.VEC_PSEUDO_MAX_DISTANCE_SQRD)
    # Remove infeasible matches
    kpts1, kpts2 = qkpts, dkpts
    fx2_to_fx1  = dfx2_qfx
    fx2_to_dist = dfx2_dist
    # COMPUTE CONSTRAINED ASSIGNMENTS
    # H should map from query to database chip (1 to 2)
    assigntup = constrained_matching.assign_spatially_constrained_matches(
        dlen_sqrd2, kpts1, kpts2, H, fx2_to_fx1, fx2_to_dist,
        match_xy_thresh, normalizer_mode=normalizer_mode)
    fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
    # FILTER ASSIGNMENTS VIA THE RATIO TEST
    ratio_tup = matching.ratio_test(fx2_match, fx1_match, fx1_norm, match_dist, norm_dist, scr_ratio_thresh)
    (fm_SCR, fs_SCR, fm_norm_SCR) = ratio_tup
    return fm_SCR, fs_SCR, fm_norm_SCR


def gridsearch_constrained_matches():
    r"""

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_constrained_matches --show

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
        'scr_ratio_thresh': [.625, 0.0, 1.0],
        'scr_normalizer_mode': ['far', 'nearby', 'plus']
    }
    slice_dict = {
        'scr_ratio_thresh': slice(0, 1),
        'scr_normalizer_mode': slice(0, 5),

    }
    # Make configuration for every parameter setting
    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict, slice_dict=slice_dict)
    #fname = None  # 'easy1.png'
    ibs, qreq_, qaid, daid_list, H_list, prior_chipmatch, prior_filtkey_list = testdata_matching()
    #config = qreq_.qparams
    testindex = 0
    daid_list = daid_list[testindex:testindex + 1]
    H_list = H_list[testindex:testindex + 1]
    cfgresult_list = [
        compute_query_constrained_matches(ibs, qaid, daid_list, H_list, cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='constrained ratio match')
    ]

    def showfunc(fm_list, fs_list, fm_norm_list=None, **kwargs):
        fs = fs_list[0]
        if kwargs.pop('use_norm', False):
            fm_norm = None
            fm = fm_norm_list[0]
            kwargs['cmap'] = 'cool'
        else:
            fm = fm_list[0]
            fm_norm = None
        kwargs['ell'] = False
        kwargs['darken'] = .3
        daid = daid_list[0]
        H1 = H_list[0]
        H1 = None  # uncomment to see warping
        show_constrained_chipmatch(ibs, qaid, daid, fm, fs=fs, H1=H1, fm_norm=fm_norm, **kwargs)

    import functools

    ut.interact_gridsearch_result_images(
        functools.partial(showfunc, use_norm=False), cfgdict_list, cfglbl_list,
        cfgresult_list, fnum=fnum, figtitle='constrained ratio match', unpack=True,
        max_plots=25)

    ut.interact_gridsearch_result_images(
        functools.partial(showfunc, use_norm=True), cfgdict_list, cfglbl_list,
        cfgresult_list, fnum=fnum + 1, figtitle='constrained ratio match', unpack=True,
        max_plots=25)
    pt.iup()


def testdata_matching():
    """
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
    qaid = 1
    qaid_list = [qaid]
    # VSMANY TO GET HOMOG
    ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict, qaid_list=qaid_list)
    locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
    qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
    qaid_list, top_aids_list, top_Hs_list, prior_chipmatch_list = make_rerank_pair_shortlist(qreq_, qaid2_chipmatch)
    qaid = qaid_list[0]
    daid_list = top_aids_list[0]
    H_list = top_Hs_list[0]
    prior_chipmatch = prior_chipmatch_list[0]
    prior_filtkey_list = qreq_.qparams.get_postsver_filtkey_list()
    return ibs, qreq_, qaid, daid_list, H_list, prior_chipmatch, prior_filtkey_list


def show_constrained_chipmatch(ibs, qaid, daid, fm, fs=None, fm_norm=None,
                               H1=None, fnum=None, pnum=None, **kwargs):
    # viz function for compute_query_constrained_matches
    from ibeis.viz import viz_sver
    import plottool as pt
    viz_sver.show_constrained_match(ibs, qaid, daid, H1=H1, fm=fm,  fs=fs, fm_norm=fm_norm, fnum=fnum, pnum=pnum, **kwargs)
    truth = ibs.get_match_truth(qaid, daid)
    color = {1: pt.TRUE_GREEN, 2: pt.UNKNOWN_PURP, 0: pt.FALSE_RED}[truth]
    pt.draw_border(pt.gca(), color, 4)
    #pt.set_title('score = %.3f' % (score,))


def show_top_chipmatches(ibs, qaid2_chipmatch, fnum_offset=0, figtitle=''):
    """ helper """
    #from ibeis.viz import viz_sver
    import plottool as pt
    CLIP_TOP = 6
    for fnum_, (qaid, chipmatch) in enumerate(six.iteritems(qaid2_chipmatch)):
        #cm = hstypes.ChipMatch2(chipmatch)
        #cm.foo()
        fnum = fnum_ + fnum_offset
        #pt.figure(fnum=fnum, doclf=True, docla=True)
        daid_list     = list(six.iterkeys(chipmatch.aid2_fm))
        score_list    = ut.dict_take(chipmatch.aid2_score, daid_list)
        top_daid_list = ut.listclip(ut.sortedby(daid_list, score_list, reverse=True), CLIP_TOP)
        nRows, nCols  = pt.get_square_row_cols(len(top_daid_list), fix=True)
        next_pnum     = pt.make_pnum_nextgen(nRows, nCols)
        for daid in top_daid_list:
            fm    = chipmatch.aid2_fm[daid]
            fsv   = chipmatch.aid2_fs[daid]
            fs    = fsv.prod(axis=1)
            H1     = chipmatch.aid2_H[daid]
            #H1 = None
            pnum = next_pnum()
            show_constrained_chipmatch(ibs, qaid, daid, fm=fm, fs=fs, H1=H1, fnum=fnum, pnum=pnum)
            score = chipmatch.aid2_score[daid]
            #viz_sver.show_constrained_match(ibs, qaid, daid, H, fm, fnum=fnum, pnum=next_pnum())
            #truth = ibs.get_match_truth(qaid, daid)
            #color = {1: pt.TRUE_GREEN, 2: pt.UNKNOWN_PURP, 0: pt.FALSE_RED}[truth]
            #pt.draw_border(pt.gca(), color, 4)
            pt.set_title('score = %.3f' % (score,))
            #top_score_list = ut.dict_take(chipmatch.aid2_score, top_daid_list)
            #top_fm_list    = ut.dict_take(chipmatch.aid2_fm, top_daid_list)
            #top_fsv_list   = ut.dict_take(chipmatch.aid2_fsv, top_daid_list)
            #top_H_list     = ut.dict_take(chipmatch.aid2_H, top_daid_list)
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
