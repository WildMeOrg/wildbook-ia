"""
special pipeline for vsone specific functions


Current Issues:
    * getting feature distinctiveness is too slow, we can either try a different
      model, or precompute feature distinctiveness.

      - we can reduce the size of the vsone shortlist


"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
import vtool as vt
#from ibeis.model.hots import neighbor_index
from ibeis.model.hots import name_scoring
from ibeis.model.hots import hstypes
#import pyflann
#from ibeis.model.hots import coverage_image
from vtool import coverage_image
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
from ibeis.model.hots import distinctiveness_normalizer
import utool as ut
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[vsonepipe]', DEBUG=False)


def show_top_chipmatches(ibs, qaid2_chipmatch, fnum_offset=0, figtitle=''):
    """ helper """
    from ibeis.viz import viz_sver
    import plottool as pt
    CLIP_TOP = 6
    for fnum_, (qaid, chipmatch) in enumerate(six.iteritems(qaid2_chipmatch)):
        fnum = fnum_ + fnum_offset
        #pt.figure(fnum=fnum, doclf=True, docla=True)
        daid_list = list(six.iterkeys(chipmatch.aid2_fm))
        score_list = ut.dict_take(chipmatch.aid2_score, daid_list)
        top_daid_list = ut.listclip(ut.sortedby(daid_list, score_list, reverse=True), CLIP_TOP)
        nRows, nCols = pt.get_square_row_cols(len(top_daid_list), fix=True)
        next_pnum = pt.make_pnum_nextgen(nRows, nCols)
        for daid in top_daid_list:
            fm = chipmatch.aid2_fm[daid]
            H = chipmatch.aid2_H[daid]
            score = chipmatch.aid2_score[daid]
            viz_sver.show_constrained_match(ibs, qaid, daid, H, fm, fnum=fnum, pnum=next_pnum())
            if ibs.get_match_truth(qaid, daid):
                pt.draw_border(pt.gca(), pt.TRUE_GREEN, 4)
            pt.set_title('score = %.3f' % (score,))
            #top_score_list = ut.dict_take(chipmatch.aid2_score, top_daid_list)
            #top_fm_list    = ut.dict_take(chipmatch.aid2_fm, top_daid_list)
            #top_fsv_list   = ut.dict_take(chipmatch.aid2_fsv, top_daid_list)
            #top_H_list     = ut.dict_take(chipmatch.aid2_H, top_daid_list)
        pt.set_figtitle('qaid=%r %s' % (qaid, figtitle))

    #vsone_query_pairs = make_vsone_rerank_pairs(qreq_, qaid2_chipmatch)
    #ibs = qreq_.ibs
    #for fnum, vsone_pair_tup in enumerate(vsone_query_pairs):
    #    (qaid, daid_list, H_list) = vsone_pair_tup
    #    nRows, nCols = pt.get_square_row_cols(len(daid_list))
    #    next_pnum = pt.make_pnum_nextgen(*daid_list)
    #    for daid in daid_list:
    #        fm = qaid2_chipmatch[qaid].aid2_fm[daid]
    #        H = qaid2_chipmatch[qaid].aid2_H[daid]
    #        viz_sver.show_constrained_match(ibs, qaid, daid, H, fm, pnum=next_pnum())


def show_annot_weights(ibs, aid, mode='dstncvs'):
    r"""
    DEMO FUNC

    Args:
        ibs (IBEISController):  ibeis controller object
        aid (int):  annotation id
        mode (str):

    CommandLine:
        alias show_annot_weights='python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show'
        show_annot_weights
        show_annot_weights --db PZ_MTEST --aid 1 --mode 'dstncvs'
        show_annot_weights --db PZ_MTEST --aid 1 --mode 'fgweight'&
        show_annot_weights --db GZ_ALL --aid 1 --mode 'dstncvs'
        show_annot_weights --db GZ_ALL --aid 1 --mode 'fgweight'&


        python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show --db GZ_ALL --aid 1 --mode 'dstncvs'
        python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'dstncvs'
        python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show --db GZ_ALL --aid 1 --mode 'fgweight'
        python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'fgweight'

        python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show --db GZ_ALL --aid 1 --mode 'dstncvs*fgweight'
        python -m ibeis.model.hots.vsone_pipeline --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'dstncvs*fgweight'

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> import plottool as pt
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(ut.get_argval('--db', type_=str, default='testdb1'))
        >>> aid = ut.get_argval('--aid', type_=int, default=1)
        >>> mode = ut.get_argval('--mode', type_=str, default='dstncvs')
        >>> # execute function
        >>> show_annot_weights(ibs, aid, mode)
        >>> pt.show_if_requested()
    """
    import functools
    import plottool as pt
    fnum = 1
    chipsize = ibs.get_annot_chipsizes(aid)[::-1]
    chip = ibs.get_annot_chips(aid)
    kpts = ibs.get_annot_kpts(aid)
    mode = mode.strip('\'')  # win32 hack
    fx2_score = 1.0
    weight_fn_dict = {
        'dstncvs': functools.partial(get_kpts_distinctiveness, ibs),
        'fgweight': ibs.get_annot_fgweights,
    }
    key_list = mode.split('*')
    for key in key_list:
        #print(key)
        get_weight = weight_fn_dict[key]
        fx2_weight = get_weight([aid])[0]
        #print(fx2_weight)
        fx2_score = fx2_score * fx2_weight
    fx2_score **= 1 / len(key_list)  # geometric average
    #ut.print_resource_usage()
    mask, patch = coverage_image.make_coverage_mask(
        kpts, chipsize, fx2_score=fx2_score, mode='max')
    #ut.print_resource_usage()
    coverage_image.show_coverage_map(chip, mask, patch, kpts, fnum, ell_alpha=.2, show_mask_kpts=False)
    pt.set_figtitle(mode)


#@profile
def vsone_reranking(qreq_, qaid2_chipmatch, verbose=False):
    """
    Driver function for vsone reranking

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking
        utprof.py -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking
        python -m ibeis.model.hots.vsone_pipeline --test-vsone_reranking --show

    Example:
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict, qaid_list=[1, 4, 6])
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid2_chipmatch_VSONE = vsone_reranking(qreq_, qaid2_chipmatch)
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     show_top_chipmatches(qreq_.ibs, qaid2_chipmatch_VSONE)
        >>>     pt.show_if_requested()

    Ignore:
        max_depth = None
        max_depth2 = 1
        print(ut.depth_profile(Hs_list, 0))
        print(ut.depth_profile(scores_list, max_depth))
        print(ut.depth_profile(daid_list, max_depth))
        print(ut.depth_profile(fms_list, max_depth2))
        print(ut.depth_profile(fsvs_list, max_depth))
    """
    # First find a shortlist to execute vsone reranking on
    qaid_list, daids_list, Hs_list = make_vsone_rerank_pairs(qreq_, qaid2_chipmatch)  # NOQA
    # Then execute vsone reranking
    vsone_res_tup = execute_vsone_reranking(qreq_, qaid_list, daids_list, Hs_list)
    # Format the output into chipmatches
    (daid_list, scores_list, fms_list, fsvs_list) = vsone_res_tup
    chipmatch_VSONE_list = []
    for daids, scores, fms, fsvs, Hs in zip(daid_list, scores_list, fms_list, fsvs_list, Hs_list):
        fks = [np.ones(len(fm), dtype=hstypes.FK_DTYPE) for fm in fms]
        aid2_fm    = dict(zip(daids, fms))
        aid2_fsv   = dict(zip(daids, fsvs))
        aid2_fk    = dict(zip(daids, fks))
        aid2_score = dict(zip(daids, scores))
        aid2_H     = dict(zip(daids, Hs))
        chipmatch_VSONE = hstypes.ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
        chipmatch_VSONE_list.append(chipmatch_VSONE)
    qaid2_chipmatch_VSONE = dict(zip(qaid_list, chipmatch_VSONE_list))
    #qaid2_scores = dict(zip(qaid_list, scores_list))
    return qaid2_chipmatch_VSONE


@profile
def make_vsone_rerank_pairs(qreq_, qaid2_chipmatch):
    """
    Makes shortlists for vsone reranking

    Example:
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> cfgdict = dict(dupvote_weight=1.0, prescore_method='nsum', score_method='nsum', sver_weighting=True)
        >>> ibs, qreq_ = plh.get_pipeline_testdata('PZ_MTEST', cfgdict=cfgdict)
        >>> locals_ = plh.testrun_pipeline_upto(qreq_, 'chipmatch_to_resdict')
        >>> qaid2_chipmatch = locals_['qaid2_chipmatch_SVER']
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> chipmatch = qaid2_chipmatch[qaid]
        >>> qaid_list, top_aids_list, top_Hs_list = make_vsone_rerank_pairs(qreq_, qaid2_chipmatch)
        >>> top_aid_list = top_aids_list[0]
        >>> top_nid_list = ibs.get_annot_name_rowids(top_aid_list)
        >>> assert top_nid_list.index(1) == 0, 'name 1 should be rank 1'
        >>> assert len(top_nid_list) == 5, 'should have 3 names and up to 2 image per name'
    """
    from ibeis.model.hots import pipeline
    score_method = qreq_.qparams.score_method
    # TODO: paramaterize
    nNameShortlistVsone = 5
    nAnnotPerName = 4
    assert score_method == 'nsum'
    print('vsone reranking. ')
    qaid_list = list(six.iterkeys(qaid2_chipmatch))
    chipmatch_list = ut.dict_take(qaid2_chipmatch, qaid_list)
    daids_list = []
    Hs_list = []
    for qaid, chipmatch in zip(qaid_list, chipmatch_list):
        daid2_prescore = pipeline.score_chipmatch(qreq_, qaid, chipmatch, score_method)
        daid_list      = np.array(daid2_prescore.keys())
        prescore_arr   = np.array(daid2_prescore.values())
        # HACK POPULATE AID2_SCORE FIELD IN CHIPMATCH TUPLE
        ut.dict_assign(chipmatch.aid2_score, daid_list, prescore_arr)
        #
        nscore_tup = name_scoring.group_scores_by_name(qreq_.ibs, daid_list, prescore_arr)
        (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscore_tup
        top_aids_list  = ut.listclip(sorted_aids, nNameShortlistVsone)
        top_aids_list_ = [ut.listclip(aids, nAnnotPerName) for aids in top_aids_list]
        top_aid_list = ut.flatten(top_aids_list_)
        top_H_list   = ut.dict_take(chipmatch.aid2_H, top_aid_list)
        daids_list.append(top_aid_list)
        Hs_list.append(top_H_list)
    return qaid_list, daids_list, Hs_list


#@profile
def execute_vsone_reranking(qreq_, qaid_list, daids_list_, Hs_list):
    r""" runs several pairs of (qaid, daids) vsone matches """
    ibs = qreq_.ibs
    # For each qaid, daids pair in the lists, execute a query
    vsone_iter = zip(qaid_list, daids_list_, Hs_list)
    progkw = dict(lbl='VSONE RERANKING', freq=1)
    vsone_prog_iter = ut.ProgressIter(vsone_iter, nTotal=len(qaid_list), **progkw)
    daid_score_fm_fsv_tup_list = [
        single_vsone_query(ibs, qaid, daid_list, H_list)
        for (qaid, daid_list, H_list) in vsone_prog_iter
    ]
    # Unpack results into their respective types
    daids_list   = ut.get_list_column(daid_score_fm_fsv_tup_list, 0)
    scores_list  = ut.get_list_column(daid_score_fm_fsv_tup_list, 1)
    fms_list     = ut.get_list_column(daid_score_fm_fsv_tup_list, 2)
    fsvs_list    = ut.get_list_column(daid_score_fm_fsv_tup_list, 3)
    vsone_res_tup = (daids_list, scores_list, fms_list, fsvs_list)
    return vsone_res_tup


@profile
def single_vsone_query(ibs, qaid, daid_list, H_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaid (int):  query annotation id
        daid_list (list):
        H_list (list):

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-single_vsone_query

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid = 1
        >>> daid_list = [3, 2, 5, 4]
        >>> H_list = [
        ...  np.array([[ -4.68815126e-01,   7.80306795e-02,  -2.23674587e+01],
        ...            [  4.54394231e-02,  -7.67438835e-01,   5.92158624e+01],
        ...            [  2.12918867e-04,  -8.64851418e-05,  -6.21472492e-01]]),
        ...  np.array([[  5.11319128e-01,  -2.69211436e-04,  -3.18079183e+01],
        ...            [ -5.97449121e-02,   4.67044573e-01,   5.27655556e+01],
        ...            [  1.06650025e-04,   8.70310639e-05,   5.28664052e-01]]),
        ...  np.array([[  4.47902439e-01,  -1.79874835e-01,  -1.88314836e-01],
        ...            [ -2.61825221e-02,   3.59390616e-01,   6.47754036e+01],
        ...            [ -1.02783595e-04,  -5.74416869e-04,   6.88664085e-01]]),
        ...  np.array([[  4.94544421e-01,   2.05268712e-01,  -5.35167763e+01],
        ...            [ -1.99183336e-01,   7.97940559e-01,  -2.45807386e+01],
        ...            [ -4.60593287e-04,   1.36874405e-03,   3.83659263e-01]])]
        >>> #species = ibeis.const.Species.ZEB_PLAIN
        >>> #dstncvs_normer = distinctiveness_normalizer.request_species_distinctiveness_normalizer(species)
        >>> # execute function
        >>> daid_fm_fs_score_tup = single_vsone_query(ibs, qaid, daid_list, H_list)
        >>> daid_list, fm_list, fs_list, score_list = daid_fm_fs_score_tup
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
    print('==================')
    #ut.print_resource_usage()
    fm_list, fs_list = compute_query_matches(ibs, qaid, daid_list, H_list)  # 35.8
    #ut.print_resource_usage()
    # BIG MEMORY JUMP HERE
    cov_score_list = compute_query_coverage(ibs, qaid, daid_list, fm_list, fs_list)  # 64.2
    #ut.print_resource_usage()
    NAME_SCORING = True
    if NAME_SCORING:
        # Keep only the best annotation per name
        nscore_tup = name_scoring.group_scores_by_name(ibs, daid_list, cov_score_list)
        score_list = ut.flatten([scores[0:1].tolist() + ([0] * (len(scores) - 1))
                                 for scores in nscore_tup.sorted_scores])
    else:
        score_list = cov_score_list
    # Convert our one score to a score vector here
    num_matches_iter = map(len, fm_list)
    num_filts = 1  # currently only using one vector here.
    fsv_list = [fs.reshape((num_matches, num_filts))
                for fs, num_matches in zip(fs_list, num_matches_iter)]
    daid_score_fm_fsv_tup = (daid_list, score_list, fm_list, fsv_list)
    return daid_score_fm_fsv_tup


@profile
def compute_query_matches(ibs, qaid, daid_list, H_list):
    r""" calls specified vsone matching routine for single (qaid, daids) pair """
    fm_list, fs_list = compute_query_constrained_matches(ibs, qaid, daid_list, H_list)
    return fm_list, fs_list


@profile
def compute_query_constrained_matches(ibs, qaid, daid_list, H_list):
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    match_xy_thresh = .05
    ratio_thresh2 = .7
    K = 7
    qvecs = ibs.get_annot_vecs(qaid)
    qkpts = ibs.get_annot_kpts(qaid)
    flann_cachedir = ibs.get_flann_cachedir()
    flann = vt.flann_cache(qvecs, flann_cachedir, flann_params=flann_params,
                           verbose=False, use_cache=False, save=False)
    fm_SCR_list = []
    fs_SCR_list = []
    dvecs_list = ibs.get_annot_vecs(daid_list)
    dkpts_list = ibs.get_annot_kpts(daid_list)
    dfgws_list = ibs.get_annot_fgweights(daid_list)
    dlen_sqrd_list = ibs.get_annot_chip_dlen_sqrd(daid_list)
    dinfo_list = zip(dvecs_list, dkpts_list, dfgws_list, dlen_sqrd_list, H_list)
    for daid, dinfo in zip(daid_list, dinfo_list):
        # THIS CAN BE SWAPED WITH PURE RATIO TEST
        # ALSO, SVER CAN BE ADDED ON THE END
        dvecs, dkpts, dfgws, dlen_sqrd, H = dinfo
        fm_SCR, fs_SCR = spatially_constrained_match(
            flann, dvecs, qkpts, dkpts, H, dlen_sqrd, match_xy_thresh,
            ratio_thresh2, K)
        fm_SCR_list.append(fm_SCR)
        fs_SCR_list.append(fs_SCR)
    del flann
    print('---------------- ;)(')
    return fm_SCR_list, fs_SCR_list


@profile
def compute_query_coverage(ibs, qaid, daid_list, fm_list, fs_list):
    """
    Returns a grayscale chip match which represents which pixels
    should be matches in order for a candidate to be considered a match.

    Ignore:
        import plottool as pt
        pt.imshow(weight_mask * 255, update=True, fnum=2)
        pt.imshow(weight_mask_m * 255, update=True, fnum=3)
        pt.imshow(weight_color * 255, update=True, fnum=3)
    """
    # Distinctivness Weight
    #  Hits     Time     Per Hit    %Time
    #     3      7213349 2404449.7     28.6
    qdstncvs  = get_kpts_distinctiveness(ibs, [qaid])[0]
    #  Hits     Time     Per Hit    %Time
    #     3     14567165 4855721.7     57.8
    ddstncvs_list  = get_kpts_distinctiveness(ibs, daid_list)
    # Foreground weight
    qfgweight = ibs.get_annot_fgweights([qaid], ensure=True)[0]
    dfgweight_list = ibs.get_annot_fgweights(daid_list, ensure=True)
    # Make weight mask
    qchipsize = ibs.get_annot_chipsizes(qaid)[::-1]
    qkpts     = ibs.get_annot_kpts(qaid)
    mode = 'max'
    # Foregroundness*Distinctiveness weight mask
    weights = (qfgweight * qdstncvs) ** .5
    #  Hits     Time     Per Hit    %Time
    #    3      2298873 766291.0      9.1
    #print('==--==--==--==--==--==')
    #ut.print_resource_usage()
    weight_mask, patch = coverage_image.make_coverage_mask(
        qkpts, qchipsize, fx2_score=weights, mode=mode)  # 9% of the time
    #ut.print_resource_usage()
    # Apply weighted scoring to matches
    score_list = []
    for fm, fs, ddstncvs, dfgweight in zip(fm_list, fs_list, ddstncvs_list, dfgweight_list):
        # Get matching query keypoints
        qkpts_m     = qkpts.take(fm.T[0], axis=0)
        ddstncvs_m  = ddstncvs.take(fm.T[1], axis=0)
        dfgweight_m = dfgweight.take(fm.T[1], axis=0)
        qdstncvs_m  = qdstncvs.take(fm.T[0], axis=0)
        qfgweight_m = qfgweight.take(fm.T[0], axis=0)
        weights_m = fs * np.sqrt(qdstncvs_m * ddstncvs_m) * np.sqrt(qfgweight_m * dfgweight_m)
        # Hits     Time     Per Hit    %Time
        #  46      1000214  21743.8      4.0
        weight_mask_m, patch = coverage_image.make_coverage_mask(
            qkpts_m, qchipsize, fx2_score=weights_m, mode=mode)  # 4% of the time
        #if True:
        #    stacktup = (weight_mask, np.zeros(weight_mask.shape), weight_mask_m)
        #    weight_color = np.dstack(stacktup)
        coverage_score = weight_mask_m.sum() / weight_mask.sum()
        score_list.append(coverage_score)
    #ut.print_resource_usage()
    return score_list


@profile
def spatially_constrained_match(flann, dvecs, qkpts, dkpts, H, dlen_sqrd,
                                match_xy_thresh, ratio_thresh2,  K):
    from vtool import constrained_matching
    # Find candidate matches matches
    #  Hits     Time     Per Hit    %Time
    #    46     13082250 284396.7     94.6
    dfx2_qfx, _dfx2_dist = flann.nn_index(dvecs, num_neighbors=K, checks=800)
    dfx2_dist = np.divide(_dfx2_dist, hstypes.VEC_PSEUDO_MAX_DISTANCE_SQRD)
    # Remove infeasible matches
    constraintup = constrained_matching.spatially_constrain_matches(
        dlen_sqrd, qkpts, dkpts, H, dfx2_qfx, dfx2_dist, match_xy_thresh,
        normalizer_mode='far')
    (fm_SC, fm_norm_SC, match_dist_list, norm_dist_list) = constraintup
    fs_SC = 1 - np.divide(match_dist_list, norm_dist_list)   # NOQA
    # Given matching distance and normalizing distance, filter by ratio scores
    fm_SCR, fs_SCR, fm_norm_SCR = constrained_matching.ratio_test2(
        match_dist_list, norm_dist_list, fm_SC, fm_norm_SC, ratio_thresh2)
    return fm_SCR, fs_SCR


@profile
def get_kpts_distinctiveness(ibs, aid_list):
    """
    per-species disinctivness wrapper around ibeis cached function
    """
    aid_list = np.array(aid_list)
    sid_list = np.array(ibs.get_annot_species_rowids(aid_list))
    # Compute distinctivness separately for each species
    unique_sids, groupxs = vt.group_indicies(sid_list)
    aids_groups = vt.apply_grouping(aid_list, groupxs)
    species_text_list = ibs.get_species_texts(unique_sids)
    # Map distinctivness computation
    normer_list = [distinctiveness_normalizer.request_species_distinctiveness_normalizer(species)
                   for species in species_text_list]
    # Reduce to get results
    dstncvs_groups = [
        # uses ibeis non-persistant cache
        # code lives in manual_ibeiscontrol_funcs
        ibs.get_annot_kpts_distinctiveness(aids, dstncvs_normer=dstncvs_normer)
        for dstncvs_normer, aids in zip(normer_list, aids_groups)
    ]
    dstncvs_list = vt.invert_apply_grouping(dstncvs_groups, groupxs)
    return dstncvs_list


def coverage_grid(kpts, chipsize, weights, grid_scale_factor=.3, grid_steps=1):
    r"""
    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        chipsize (tuple):
        weights (ndarray):

    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-coverage_grid --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> kpts, chipsize, weights = testdata_coveragegrid()
        >>> grid_scale_factor = .3
        >>> grid_steps = 1
        >>> coverage_gridtup = coverage_grid(kpts, chipsize, weights)
        >>> num_cols, num_rows, subbin_xy_arr, neighbor_subbin_center_arr, neighbor_subbin_weights = coverage_gridtup
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     visualize_coverage_grid(num_cols, num_rows, subbin_xy_arr,
        >>>                    neighbor_subbin_center_arr, neighbor_subbin_weights)
        >>>     pt.show_if_requested()
    """

    def get_subbin_xy_neighbors(subbin_index00, grid_steps, num_cols, num_rows):
        """ Generate all neighbor of a bin
        subbin_index00 = left and up subbin index
        """
        subbin_index00 = np.floor(subbin_index00).astype(np.int32)
        subbin_x0, subbin_y0 = subbin_index00
        step_list = np.arange(1 - grid_steps, grid_steps + 1)
        offset_list = [
            # broadcast to the shape we will add too
            np.array([xoff, yoff])[:, None]
            for xoff, yoff in list(ut.iprod(step_list, step_list))]
        neighbor_subbin_index_list = [
            np.add(subbin_index00, offset)
            for offset in offset_list
        ]
        # Concatenate all subbin indexes into one array for faster vectorized ops
        neighbor_subbin_index_arr = np.dstack(neighbor_subbin_index_list).T

        # Clip with no wrapparound
        min_val = np.array([0, 0])
        max_val = np.array([num_cols, num_rows])

        np.clip(neighbor_subbin_index_arr,
                min_val[None, None, :],
                max_val[None, None, :],
                out=neighbor_subbin_index_arr)
        return neighbor_subbin_index_arr

    def compute_subbin_to_bins_dist(neighbor_subbin_center_arr, subbin_xy_arr):
        _tmp = np.subtract(neighbor_subbin_center_arr, subbin_xy_arr.T[None, :])
        neighbor_subbin_sqrddist_arr = np.power(_tmp, 2, out=_tmp).sum(axis=2)
        return neighbor_subbin_sqrddist_arr

    def weighted_gaussian_falloff(neighbor_subbin_sqrddist_arr, weights):
        _gaussweights = vt.gauss_func1d_unnormalized(neighbor_subbin_sqrddist_arr)
        # Each column sums to 1
        np.divide(_gaussweights, _gaussweights.sum(axis=0)[None, :], out=_gaussweights)
        # Scale initial weights by the gaussian falloff
        neighbor_subbin_weights = np.multiply(_gaussweights, weights[None, :])
        return neighbor_subbin_weights

    # Compute grid size and stride
    chip_w, chip_h = chipsize
    num_rows = vt.iround(grid_scale_factor * chip_h)
    num_cols = vt.iround(grid_scale_factor * chip_w)
    chipstride = np.array((chip_w / num_cols, chip_h / num_rows))
    # Find keypoint subbin locations relative to edges
    xy_arr = vt.get_xys(kpts)
    subbin_xy_arr = np.divide(xy_arr, chipstride[:, None])
    # Find subbin locations relative to centers
    frac_subbin_index = np.subtract(subbin_xy_arr, .5)
    neighbor_subbin_index_arr = get_subbin_xy_neighbors(frac_subbin_index, grid_steps, num_cols, num_rows)
    # Find centers
    neighbor_subbin_center_arr = np.add(neighbor_subbin_index_arr, .5)
    # compute distance to neighbors
    neighbor_subbin_sqrddist_arr = compute_subbin_to_bins_dist(neighbor_subbin_center_arr, subbin_xy_arr)
    # scale weights using guassia falloff
    neighbor_subbin_weights = weighted_gaussian_falloff(neighbor_subbin_sqrddist_arr, weights)

    coverage_gridtup = num_cols, num_rows, subbin_xy_arr, neighbor_subbin_center_arr, neighbor_subbin_weights
    return coverage_gridtup


def testdata_coveragegrid():
    import vtool as vt
    # build test data
    kpts = vt.dummy.get_dummy_kpts()
    kpts = np.vstack((kpts, [0, 0, 1, 1, 1, 0]))
    kpts = np.vstack((kpts, [0.01, 10, 1, 1, 1, 0]))
    kpts = np.vstack((kpts, [0.94, 11.5, 1, 1, 1, 0]))
    chipsize = tuple(vt.iceil(vt.get_kpts_image_extent(kpts)).tolist())
    weights = np.ones(len(kpts))
    return kpts, chipsize, weights


def gridsearch_coverage_grid():
    """
    CommandLine:
        python -m ibeis.model.hots.vsone_pipeline --test-gridsearch_coverage_grid --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.vsone_pipeline import *  # NOQA
        >>> import plottool as pt
        >>> gridsearch_coverage_grid()
        >>> pt.show_if_requested()
    """
    kpts, chipsize, weights = testdata_coveragegrid()
    search_basis = {
        'grid_scale_factor': [.1, .25, .5, 1.0],
        'grid_steps': [1, 2, 3, 10],
    }
    param_slice_dict = {
        'grid_scale_factor' : slice(0, 4),
        'grid_steps'        : slice(0, 4),
    }
    varied_dict = {
        key: val[param_slice_dict.get(key, slice(0, 1))]
        for key, val in six.iteritems(search_basis)
    }
    # Make configuration for every parameter setting
    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict)
    coverage_gridtup_list = [
        coverage_grid(kpts, chipsize, weights, **cfgdict)
        for cfgdict in ut.ProgressIter(cfgdict_list, lbl='coverage grid')
    ]

    fnum = 1
    ut.interact_gridsearch_result_images(
        visualize_coverage_grid, cfgdict_list, cfglbl_list,
        coverage_gridtup_list, fnum=fnum, figtitle='coverage grid', unpack=True,
        max_plots=25)

    import plottool as pt
    pt.iup()
    #pt.show_if_requested()


def visualize_coverage_grid(num_cols, num_rows, subbin_xy_arr,
                            neighbor_subbin_center_arr, neighbor_subbin_weights,
                            fnum=None, pnum=None):
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    fig = pt.figure(fnum, pnum=pnum)
    ax = fig.gca()
    x_edge_indices = np.arange(num_cols)
    y_edge_indices = np.arange(num_rows)
    x_center_indices = vt.hist_edges_to_centers(x_edge_indices)
    y_center_indices = vt.hist_edges_to_centers(y_edge_indices)
    x_center_grid, y_center_grid = np.meshgrid(x_center_indices, y_center_indices)
    ax.set_xticks(x_edge_indices)
    ax.set_yticks(y_edge_indices)
    # Plot keypoint locs
    ax.scatter(subbin_xy_arr[0], subbin_xy_arr[1], marker='o')
    # Plot Weighted Lines to Subbins
    pt_colors = pt.distinct_colors(len(subbin_xy_arr.T))
    for subbin_centers, subbin_weights in zip(neighbor_subbin_center_arr,
                                              neighbor_subbin_weights):
        for pt_xys, center_xys, weight, color in zip(subbin_xy_arr.T, subbin_centers,
                                                     subbin_weights, pt_colors):
            # Adjsut weight to alpha for easier visualization
            alpha = weight
            #alpha **= .5
            #min_viz_alpha = .1
            #alpha = alpha * (1 - min_viz_alpha) + min_viz_alpha
            ax.plot(*np.vstack((pt_xys, center_xys)).T, color=color, alpha=alpha, lw=3)
    # Plot Grid Centers
    num_cells = num_cols * num_rows
    grid_alpha = min(.4, max(1 - (num_cells / 500), .1))
    grid_color = [.6, .6, .6, grid_alpha]
    #print(grid_color)
    # Plot grid cetners
    ax.scatter(x_center_grid, y_center_grid, marker='.', color=grid_color,
               s=grid_alpha)

    ax.set_xlim(0, num_cols - 1)
    ax.set_ylim(0, num_rows - 1)
    #-----
    pt.dark_background()
    ax.grid(True, color=[.3, .3, .3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


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
