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
    daid_score_fm_fsv_tup_list = [
        single_vsone_query(ibs, qaid, daid_list, H_list)
        for (qaid, daid_list, H_list) in
        zip(qaid_list, daids_list_, Hs_list)]
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
    fm_list, fsv_list = compute_query_matches(ibs, qaid, daid_list, H_list)  # 35.8
    cov_score_list = compute_query_coverage(ibs, qaid, daid_list, fm_list, fsv_list)  # 64.2
    # Keep only the best annotation per name
    NAME_SCORING = True
    if NAME_SCORING:
        nscore_tup = name_scoring.group_scores_by_name(ibs, daid_list, cov_score_list)
        score_list = ut.flatten([scores[0:1].tolist() + ([0] * (len(scores) - 1))
                                 for scores in nscore_tup.sorted_scores])
    else:
        score_list = cov_score_list
    daid_score_fm_fsv_tup = (daid_list, score_list, fm_list, fsv_list)
    return daid_score_fm_fsv_tup


@profile
def compute_query_matches(ibs, qaid, daid_list, H_list):
    r""" calls specified vsone matching routine for single (qaid, daids) pair """
    fm_list, fsv_list = compute_query_constrained_matches(ibs, qaid, daid_list, H_list)
    return fm_list, fsv_list


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
    weight_mask, patch = coverage_image.make_coverage_mask(
        qkpts, qchipsize, fx2_score=weights, mode=mode)  # 9% of the time
    # Apply weighted scoring to matches
    score_list = []
    for fm, fs, ddstncvs, dfgweight in zip(fm_list, fs_list, ddstncvs_list, dfgweight_list):
        # Get matching query keypoints
        qkpts_m = qkpts.take(fm.T[0], axis=0)
        ddstncvs_m = ddstncvs.take(fm.T[1], axis=0)
        dfgweight_m = dfgweight.take(fm.T[1], axis=0)
        qdstncvs_m = qdstncvs.take(fm.T[0], axis=0)
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
    # Filter by ratio scores
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
