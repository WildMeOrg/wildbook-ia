from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import six  # NOQA
import numpy as np
#from vtool import keypoint as ktool
from vtool import coverage_kpts
from vtool import spatial_verification as sver
from vtool import matching
#import numpy.linalg as npl
#import scipy.sparse as sps
#import scipy.sparse.linalg as spsl
#from numpy.core.umath_tests import matrix_multiply
#import vtool.keypoint as ktool
#import vtool.linalg as ltool
#profile = ut.profile


def assign_nearest_neighbors(vecs1, vecs2, K=2):
    import vtool as vt
    import pyflann
    checks = 800
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    #pseudo_max_dist_sqrd = (np.sqrt(2) * 512) ** 2
    #pseudo_max_dist_sqrd = 2 * (512 ** 2)
    flann = vt.flann_cache(vecs1, flann_params=flann_params)
    try:
        fx2_to_fx1, fx2_to_dist = matching.normalized_nearest_neighbors(flann, vecs2, K, checks)
        #fx2_to_fx1, _fx2_to_dist = flann.nn_index(vecs2, num_neighbors=K, checks=checks)
    except pyflann.FLANNException:
        print('vecs1.shape = %r' % (vecs1.shape,))
        print('vecs2.shape = %r' % (vecs2.shape,))
        print('vecs1.dtype = %r' % (vecs1.dtype,))
        print('vecs2.dtype = %r' % (vecs2.dtype,))
        raise
    #fx2_to_dist = np.divide(_fx2_to_dist, pseudo_max_dist_sqrd)
    return fx2_to_fx1, fx2_to_dist


def baseline_vsone_ratio_matcher(testtup, cfgdict={}):
    r"""
    spatially constrained ratio matching

    CommandLine:
        python -m vtool.constrained_matching --test-spatially_constrianed_matcher

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.constrained_matching import *  # NOQA
        >>> import vtool as vt
        >>> testtup = testdata_matcher()
        >>> # execute function
        >>> basetup, base_meta = baseline_vsone_ratio_matcher(testtup)
        >>> # verify results
        >>> print(basetup)
    """
    rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2 = testtup
    return baseline_vsone_ratio_matcher_(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict={})


def spatially_constrianed_matcher(testtup, basetup, cfgdict={}):
    r"""
    spatially constrained ratio matching

    CommandLine:
        python -m vtool.constrained_matching --test-spatially_constrianed_matcher

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.constrained_matching import *  # NOQA
        >>> import vtool as vt
        >>> testtup = testdata_matcher()
        >>> basetup, base_meta = baseline_vsone_ratio_matcher(testtup)
        >>> # execute function
        >>> nexttup, next_meta = spatially_constrianed_matcher(testtup, basetup)
        >>> # verify results
        >>> print(nexttup)
    """
    (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testtup
    (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT) = basetup
    return spatially_constrianed_matcher_(kpts1, vecs1, kpts2, vecs2,
                                          dlen_sqrd2, H_RAT, cfgdict={})


def baseline_vsone_ratio_matcher_(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict={}):
    r"""
    Args:
        vecs1 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        vecs2 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints

    Ignore:
        %pylab qt4
        import plottool as pt
        pt.imshow(rchip1)
        pt.draw_kpts2(kpts1)

        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
    """
    #import vtool as vt
    sver_xy_thresh = cfgdict.get('sver_xy_thresh', .01)
    ratio_thresh =  cfgdict.get('ratio_thresh', .625)
    #ratio_thresh =  .99
    # GET NEAREST NEIGHBORS
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=2)
    assigntup = matching.assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist)
    fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
    fm_ORIG = np.vstack((fx1_match, fx2_match)).T
    fs_ORIG = 1 - np.divide(match_dist, norm_dist)
    # APPLY RATIO TEST
    fm_RAT, fs_RAT, fm_norm_RAT = matching.ratio_test(fx2_match, fx1_match, fx1_norm, match_dist, norm_dist, ratio_thresh)
    # SPATIAL VERIFICATION FILTER
    #with ut.EmbedOnException():
    match_weights = np.ones(len(fm_RAT))
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_RAT, sver_xy_thresh, dlen_sqrd2, match_weights=match_weights)
    if svtup is not None:
        (homog_inliers, homog_errors, H_RAT) = svtup[0:3]
    else:
        H_RAT = np.eye(3)
        homog_inliers = []
    fm_SV = fm_RAT[homog_inliers]
    fs_SV = fs_RAT[homog_inliers]
    fm_norm_SV = fm_norm_RAT[homog_inliers]

    base_tup = (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT)
    base_meta = (fm_norm_RAT, fm_norm_SV)
    return base_tup, base_meta


def spatially_constrianed_matcher_(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2,
                                   H_RAT, cfgdict={}):
    #import vtool as vt

    #match_xy_thresh = .1
    #sver_xy_thresh = .01
    #ratio_thresh2 = .8
    # Observation, scores don't change above K=7
    # on easy test case
    #search_K = 7  # 3
    search_K = cfgdict.get('search_K', 7)
    ratio_thresh2   = cfgdict.get('ratio_thresh2', .8)
    sver_xy_thresh2 = cfgdict.get('sver_xy_thresh2', .01)
    normalizer_mode = cfgdict.get('normalizer_mode', 'far')
    match_xy_thresh = cfgdict.get('match_xy_thresh', .1)

    # ASSIGN CANDIDATES
    # Get candidate nearest neighbors
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=search_K)

    # COMPUTE CONSTRAINTS
    #normalizer_mode = 'far'
    constrain_tup = spatially_constrain_matches(dlen_sqrd2, kpts1, kpts2, H_RAT,
                                                fx2_to_fx1, fx2_to_dist,
                                                match_xy_thresh,
                                                normalizer_mode=normalizer_mode)
    (fm_SC, fm_norm_SC, match_dist, norm_dist) = constrain_tup
    fx2_match = fm_SC.T[1]
    fx1_match = fm_SC.T[1]
    fx1_norm  = fm_norm_SC.T[1]

    fm_SCR, fs_SCR, fm_norm_SCR = matching.ratio_test(fx2_match, fx1_match,
                                                      fx1_norm, match_dist,
                                                      norm_dist,  ratio_thresh2)
    fs_SC = 1 - np.divide(match_dist, norm_dist)   # NOQA
    #fm_SCR, fs_SCR, fm_norm_SCR = ratio_test2(match_dist, norm_dist, fm_SC,
    #                                                fm_norm_SC, ratio_thresh2)

    # Another round of verification
    match_weights = np.ones(len(fm_SCR))
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_SCR, sver_xy_thresh2, dlen_sqrd2, match_weights=match_weights)
    if svtup is not None:
        (homog_inliers, homog_errors, H_SCR) = svtup[0:3]
    else:
        H_SCR = np.eye(3)
        homog_inliers = []
    fm_SCRSV = fm_SCR[homog_inliers]
    fs_SCRSV = fs_SCR[homog_inliers]

    fm_norm_SVSCR = fm_norm_SCR[homog_inliers]

    nexttup = (fm_SC, fs_SC, fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR)
    next_meta = (fm_norm_SC, fm_norm_SCR, fm_norm_SVSCR)
    return nexttup, next_meta


#def ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh):
#    fx2_to_ratio = np.divide(fx2_to_dist.T[0], fx2_to_dist.T[1])
#    fx2_to_isvalid = fx2_to_ratio < ratio_thresh
#    fx2_m = np.where(fx2_to_isvalid)[0]
#    fx1_m = fx2_to_fx1.T[0].take(fx2_m)
#    fs_RAT = np.subtract(1.0, fx2_to_ratio.take(fx2_m))
#    fm_RAT = np.vstack((fx1_m, fx2_m)).T
#    # return normalizer info as well
#    fx1_m_normalizer = fx2_to_fx1.T[1].take(fx2_m)
#    fm_norm_RAT = np.vstack((fx1_m_normalizer, fx2_m)).T
#    return fm_RAT, fs_RAT, fm_norm_RAT


#def ratio_test2(match_dist_list, norm_dist_list, fm_SC, fm_norm_SC, ratio_thresh2=.8):
#    ratio_list = np.divide(match_dist_list, norm_dist_list)
#    #ratio_thresh = .625
#    #ratio_thresh = .725
#    isvalid_list = np.less(ratio_list, ratio_thresh2)
#    valid_ratios = ratio_list[isvalid_list]
#    fm_SCR = fm_SC[isvalid_list]
#    fs_SCR = np.subtract(1.0, valid_ratios)  # NOQA
#    fm_norm_SCR = fm_norm_SC[isvalid_list]
#    #fm_SCR = np.vstack((fx1_m, fx2_m)).T  # NOQA
#    return fm_SCR, fs_SCR, fm_norm_SCR


def spatially_constrain_matches(dlen_sqrd2, kpts1, kpts2, H_RAT,
                                fx2_to_fx1, fx2_to_dist,
                                match_xy_thresh, normalizer_mode='far'):
    r"""
    helper for spatially_constrianed_matcher
    OLD FUNCTION

    Args:
        dlen_sqrd2 (?):
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H_RAT (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        fx2_to_fx1 (ndarray):
        fx2_to_dist (ndarray):
        match_xy_thresh (?): threshold is specified as a fraction of the diagonal chip length
        normalizer_mode (str):
    """
    # Find the normalized spatial error of all candidate matches
    #####

    # Filter out matches that could not be constrained

    if normalizer_mode == 'plus':
        norm_xy_bounds = (0, np.inf)
    elif normalizer_mode == 'far':
        norm_xy_bounds = (match_xy_thresh, np.inf)
    elif normalizer_mode == 'nearby':
        norm_xy_bounds = (0, match_xy_thresh)
    else:
        raise AssertionError('normalizer_mode=%r' % (normalizer_mode,))

    assigntup = matching.assign_spatially_constrained_matches(
        dlen_sqrd2, kpts1, kpts2, H_RAT, fx2_to_fx1, fx2_to_dist,
        match_xy_thresh, norm_xy_bounds=norm_xy_bounds)

    fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup

    fm_constrained = np.vstack((fx1_match, fx2_match)).T
    # return noramlizers as well
    fm_norm_constrained = np.vstack((fx1_norm, fx2_match)).T

    constraintup = (fm_constrained, fm_norm_constrained, match_dist, norm_dist)
    return constraintup


def compute_forgroundness(fpath1, kpts1, species='zebra_plains'):
    """
    hack in foregroundness
    """
    import pyrf
    import vtool as vt
    from os.path import exists
    # hack for getting a model (not entirely ibeis independent)
    trees_path = ut.get_app_resource_dir('ibeis', 'detectmodels', 'rf', species)
    tree_fpath_list = ut.glob(trees_path, '*.txt')
    detector = pyrf.Random_Forest_Detector()
    # TODO; might need to downsample
    forest = detector.forest(tree_fpath_list, verbose=False)
    gpath_list = [fpath1]
    output_gpath_list = [gpath + '.' + species + '.probchip.png' for gpath in gpath_list]
    detectkw = {
        'scale_list': [1.15, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1],
        'output_gpath_list': output_gpath_list,
        'mode': 1,  # mode one outputs probimage
    }
    results_iter = detector.detect(forest, gpath_list, **detectkw)
    results_list = list(results_iter)  # NOQA
    probchip_list = [vt.imread(gpath, grayscale=True) if exists(gpath) else None for gpath in output_gpath_list]
    #vtpatch.get_warped_patches()
    fgweights_list = []
    kpts_list = [kpts1]
    for probchip, kpts in zip(probchip_list,  kpts_list):
        patch_list  = [vt.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0 for kp in kpts]
        weight_list = [vt.gaussian_average_patch(patch) for patch in patch_list]
        #weight_list = [patch.sum() / (patch.size) for patch in patch_list]
        weights = np.array(weight_list, dtype=np.float32)
        fgweights_list.append(weights)
    fgweights = fgweights_list[0]
    return fgweights


def compute_distinctivness(vecs_list, species='zebra_plains'):
    """
    hack in distinctivness
    """
    from ibeis.model.hots import distinctiveness_normalizer
    cachedir = ut.get_app_resource_dir('ibeis', 'distinctiveness_model')
    dstcnvs_normer = distinctiveness_normalizer.DistinctivnessNormalizer(species, cachedir=cachedir)
    dstcnvs_normer.load(cachedir)
    dstncvs_list = [dstcnvs_normer.get_distinctiveness(vecs) for vecs in vecs_list]
    return dstncvs_list


@six.add_metaclass(ut.ReloadingMetaclass)
class Annot(object):
    """
    fpath1 = ut.grab_test_imgpath(fname1)
    fpath2 = ut.grab_test_imgpath(fname2)
    annot1 = Annot(fpath1)
    annot2 = Annot(fpath2)
    annot = annot1

    """
    def __init__(annot, fpath, species='zebra_plains'):
        annot.fpath = fpath
        annot.species = species
        annot.kpts      = None
        annot.vecs      = None
        annot.rchip     = None
        annot.dstncvs   = None
        annot.fgweights = None
        annot.dstncvs_mask = None
        annot.fgweight_mask = None
        annot.load()

    def show(annot):
        import plottool as pt
        pt.imshow(annot.rchip)
        pt.draw_kpts2(annot.kpts)

    def show_dstncvs_mask(annot, title='wd', update=True, **kwargs):
        import plottool as pt
        pt.imshow(annot.dstncvs_mask * 255.0, update=update, title=title, **kwargs)

    def show_fgweight_mask(annot, title='fg', update=True, **kwargs):
        import plottool as pt
        pt.imshow(annot.fgweight_mask * 255.0, update=update, title=title, **kwargs)

    def load(annot):
        from vtool import image as gtool
        from vtool import features as feattool
        kpts, vecs = feattool.extract_features(annot.fpath)
        annot.kpts      = kpts
        annot.vecs      = vecs
        annot.rchip     = gtool.imread(annot.fpath)
        annot.dstncvs   = compute_distinctivness([annot.vecs], annot.species)[0]
        annot.fgweights = compute_forgroundness(annot.fpath, annot.kpts, annot.species)
        annot.chipshape = annot.rchip.shape
        annot.dlen_sqrd = annot.chipshape[0] ** 2 + annot.chipshape[1] ** 2

    def lazy_compute(annot):
        if annot.dstncvs_mask is None:
            annot.compute_dstncvs_mask()
        if annot.fgweight_mask is None:
            annot.compute_fgweight_mask()

    def compute_fgweight_mask(annot):
        keys = ['kpts', 'chipshape', 'fgweights']
        kpts, chipshape, fgweights = ut.dict_take(annot.__dict__, keys)
        chipsize = chipshape[0:2][::-1]
        fgweight_mask = coverage_kpts.make_kpts_coverage_mask(
            kpts, chipsize, fgweights, mode='max', resize=True, return_patch=False)
        annot.fgweight_mask = fgweight_mask

    def compute_dstncvs_mask(annot):
        keys = ['kpts', 'chipshape', 'dstncvs']
        kpts, chipshape, dstncvs = ut.dict_take(annot.__dict__, keys)
        chipsize = chipshape[0:2][::-1]
        dstncvs_mask = coverage_kpts.make_kpts_coverage_mask(
            kpts, chipsize, dstncvs, mode='max', resize=True, return_patch=False)
        annot.dstncvs_mask = dstncvs_mask

    def baseline_match(annot, annot2):
        cfgdict = {}
        annot1 = annot
        keys = ['kpts', 'vecs']
        kpts1, vecs1 = ut.dict_take(annot1.__dict__, keys)
        kpts2, vecs2 = ut.dict_take(annot2.__dict__, keys)
        dlen_sqrd2 = annot2.dlen_sqrd
        basetup, base_meta = baseline_vsone_ratio_matcher_(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict)
        (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT) = basetup
        (fm_norm_RAT, fm_norm_SV) = base_meta
        match_ORIG = AnnotMatch(annot1, annot2, fm_ORIG, fs_ORIG, 'ORIG')  # NOQA
        match_RAT  = AnnotMatch(annot1, annot2, fm_RAT,  fs_RAT,  'RAT', fm_norm_RAT)  # NOQA
        match_SV   = AnnotMatch(annot1, annot2, fm_SV,   fs_SV,   'SV', fm_norm_SV)
        match_SV.H = H_RAT
        return match_ORIG, match_RAT, match_SV

    def constrained_match(annot, match_SV):
        cfgdict = {}
        annot1 = match_SV.annot1
        assert annot1 is annot
        annot2 = match_SV.annot2
        keys = ['kpts', 'vecs']
        kpts1, vecs1 = ut.dict_take(annot1.__dict__, keys)
        kpts2, vecs2 = ut.dict_take(annot2.__dict__, keys)
        dlen_sqrd2 = annot2.dlen_sqrd
        H_RAT = match_SV.H
        nexttup, next_meta = spatially_constrianed_matcher_(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, H_RAT, cfgdict)
        (fm_SC, fs_SC, fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR) = nexttup
        (fm_norm_SC, fm_norm_SCR, fm_norm_SCRSV) = next_meta
        match_SC    = AnnotMatch(annot1, annot2, fm_SC, fs_SC, 'SC', fm_norm_SC)  # NOQA
        match_SCR   = AnnotMatch(annot1, annot2, fm_SCR,  fs_SCR,  'SCR', fm_norm_SCR)  # NOQA
        match_SCRSV = AnnotMatch(annot1, annot2, fm_SCRSV,   fs_SCRSV, 'SCRSV', fm_norm_SCRSV)
        match_SCRSV.H = H_SCR
        return match_SC, match_SCR, match_SCRSV


@six.add_metaclass(ut.ReloadingMetaclass)
class AnnotMatch(object):
    r"""

    Example1:
        >>> from vtool.constrained_matching import *  # NOQA
        >>> fname1, fname2 = 'easy1.png', 'easy2.png'
        >>> fpath1 = ut.grab_test_imgpath(fname1)
        >>> fpath2 = ut.grab_test_imgpath(fname2)
        >>> annot1, annot2 = Annot(fpath1), Annot(fpath2)
        >>> match_ORIG, match_RAT, match_SV = annot1.baseline_match(annot2)
        >>> match = match_SV
        >>> match_SC, match_SCR, match_SCRSV = annot1.constrained_match(match_SV)
        >>> match = match_SCR
        >>> # ___
        >>> match_list = [match_ORIG, match_RAT, match_SV, match_SC, match_SCR, match_SCRSV]
        >>> # false match
        >>> fname3 = 'hard3.png'
        >>> fpath3 = ut.grab_test_imgpath(fname3)
        >>> annot3 = Annot(fpath3)
        >>> match_tn_list = []
        >>> match_tn_list.extend(annot1.baseline_match(annot3))
        >>> match_SV_fn = match_tn_list[-1]
        >>> match_tn_list.extend(annot1.constrained_match(match_SV_fn))
        >>> # ___
        >>> print('___________')
        >>> for match in match_list:
        >>>    match.print_scores()
        >>> print('___________')
        >>> for match_tn in match_tn_list:
        >>>    match_tn.print_scores()
        >>> print('___________')
        >>> for match, match_tn in zip(match_list, match_tn_list):
        >>>    match.print_score_diffs(match_tn)

    Ignore::
        match.show_matches(fnum=1, update=True)

        match.show_normalizers(fnum=2, update=True)
    """
    def __init__(match, annot1, annot2, fm, fs, key=None, fm_norm=None):
        match.key = key
        match.annot1 = annot1
        match.annot2 = annot2
        match.fm = fm
        match.fs = fs
        match.fm_norm = fm_norm

        # Matching coverage of annot2
        match.coverage_mask2 = None

        # Scalar scores of theis match
        match.num_matches = None
        match.sum_score = None
        match.ave_score = None
        match.weight_ave_score = None
        match.coverage_score = None
        match.weighted_coverage_score = None

    def compute_scores(match):
        match.num_matches = len(match.fm)
        match.sum_score = match.fs.sum()
        match.ave_score = match.fs.sum() / match.fs.shape[0]
        match.weight_ave_score = match.compute_weighte_average_score()
        match.coverage_score = match.coverage_mask2.sum() / np.prod(match.coverage_mask2.shape)
        match.weighted_coverage_score = match.compute_weighted_coverage_score()

    def compute_weighte_average_score(match):
        """ old scoring measure """
        import vtool as vt
        # Get distinctivness and forground of matching points
        fx1_list, fx2_list = match.fm.T
        annot1 = match.annot1
        annot2 = match.annot2
        dstncvs1  = annot1.dstncvs.take(fx1_list)
        dstncvs2  = annot2.dstncvs.take(fx2_list)
        fgweight1 = annot1.fgweights.take(fx1_list)
        fgweight2 = annot2.fgweights.take(fx2_list)
        dstncvs = np.sqrt(dstncvs1 * dstncvs2)
        fgweight = np.sqrt(fgweight1 * fgweight2)
        fsv = np.vstack((match.fs, dstncvs, fgweight)).T
        fs_new = vt.weighted_average_scoring(fsv, [0], [1, 2])
        weight_ave_score = fs_new.sum()
        return weight_ave_score

    def lazy_compute(match):
        match.annot2.lazy_compute()
        if match.coverage_mask2 is None:
            match.compute_coverage_mask()
        match.compute_scores()

    def compute_weighted_coverage_score(match):
        weight_mask = np.sqrt(match.annot2.dstncvs_mask * match.annot2.fgweight_mask)
        conerage_score = (match.coverage_mask2.sum() / weight_mask.sum())
        return conerage_score

    def compute_coverage_mask(match):
        """ compute matching coverage of annot """
        fm = match.fm
        fs = match.fs
        kpts2       = match.annot2.kpts
        chipshape2 = match.annot2.chipshape
        chipsize2 = chipshape2[0:2][::-1]
        kpts2_m = kpts2.take(fm.T[1], axis=0)
        coverage_mask2 = coverage_kpts.make_kpts_coverage_mask(
            kpts2_m, chipsize2, fs, mode='max', resize=True, return_patch=False)
        match.coverage_mask2 = coverage_mask2

    # --- INFO ---

    def print_scores(match):
        match.lazy_compute()
        score_keys = ['num_matches', 'sum_score', 'ave_score',
                      'weight_ave_score', 'coverage_score',
                      'weighted_coverage_score']
        msglist = []
        for key in score_keys:
            msglist.append(' * %s = %6.2f' % (key, match.__dict__[key]))
        msglist_aligned = ut.align_lines(msglist, '=')
        msg = '\n'.join(msglist_aligned)
        print('key = %r' % (match.key,))
        print(msg)

    def print_score_diffs(match, match_tn):
        score_keys = ['num_matches', 'sum_score', 'ave_score',
                      'weight_ave_score', 'coverage_score',
                      'weighted_coverage_score']
        msglist = [' * <key> =   <tp>,   <tn>, <diff>, <factor>']
        for key in score_keys:
            score = match.__dict__[key]
            score_tn = match_tn.__dict__[key]
            score_diff = score - score_tn
            score_factor = score / score_tn
            msglist.append(' * %s = %6.2f, %6.2f, %6.2f, %6.2f' % (key, score, score_tn, score_diff, score_factor))
        msglist_aligned = ut.align_lines(msglist, '=')
        msg = '\n'.join(msglist_aligned)
        print('key = %r' % (match.key,))
        print(msg)

    def show_matches(match, fnum=None, pnum=None, update=True):
        import plottool as pt
        from plottool import plot_helpers as ph
        # hack keys out of namespace
        keys = ['rchip', 'kpts']
        rchip1, kpts1 = ut.dict_take(match.annot1.__dict__, keys)
        rchip2, kpts2 = ut.dict_take(match.annot2.__dict__, keys)
        fs, fm = match.fs, match.fm
        cmap = 'hot'
        draw_lines = True
        if fnum is None:
            fnum = pt.next_fnum()
        pt.figure(fnum=fnum, pnum=pnum)
        #doclf=True, docla=True)
        ax, xywh1, xywh2 = pt.show_chipmatch2(
            rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs, fnum=fnum, cmap=cmap,
            draw_lines=draw_lines)
        ph.set_plotdat(ax, 'viztype', 'matches')
        ph.set_plotdat(ax, 'key', match.key)
        title = match.key + '\n num=%d, sum=%.2f' % (len(fm), sum(fs))
        pt.set_title(title)
        if update:
            pt.update()
        return ax, xywh1, xywh2

    def show_normalizers(match, fnum=None, pnum=None, update=True):
        import plottool as pt
        from plottool import plot_helpers as ph
        # hack keys out of namespace
        keys = ['rchip', 'kpts']
        rchip1, kpts1 = ut.dict_take(match.annot1.__dict__, keys)
        rchip2, kpts2 = ut.dict_take(match.annot2.__dict__, keys)
        fs, fm = match.fs, match.fm_norm
        cmap = 'cool'
        draw_lines = True
        if fnum is None:
            fnum = pt.next_fnum()
        pt.figure(fnum=fnum, pnum=pnum)
        #doclf=True, docla=True)
        ax, xywh1, xywh2 = pt.show_chipmatch2(
            rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs, fnum=fnum, cmap=cmap,
            draw_lines=draw_lines)
        ph.set_plotdat(ax, 'viztype', 'matches')
        ph.set_plotdat(ax, 'key', match.key)
        title = match.key + '\n num=%d, sum=%.2f' % (len(fm), sum(fs))
        pt.set_title(title)
        if update:
            pt.update()
        return ax, xywh1, xywh2


def testdata_matcher(fname1='easy1.png', fname2='easy2.png'):
    """"
    fname1 = 'easy1.png'
    fname2 = 'hard3.png'

    annot1 = Annot(fpath1)
    annot2 = Annot(fpath2)
    """
    import utool as ut
    from vtool import image as gtool
    from vtool import features as feattool
    fpath1 = ut.grab_test_imgpath(fname1)
    fpath2 = ut.grab_test_imgpath(fname2)
    kpts1, vecs1 = feattool.extract_features(fpath1)
    kpts2, vecs2 = feattool.extract_features(fpath2)
    rchip1 = gtool.imread(fpath1)
    rchip2 = gtool.imread(fpath2)
    #chip1_shape = vt.gtool.open_image_size(fpath1)
    chip2_shape = gtool.open_image_size(fpath2)
    dlen_sqrd2 = chip2_shape[0] ** 2 + chip2_shape[1] ** 2
    testtup = (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2)

    return testtup


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.constrained_matching
        python -m vtool.constrained_matching --allexamples
        python -m vtool.constrained_matching --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
