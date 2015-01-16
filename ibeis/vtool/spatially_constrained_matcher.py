from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import numpy as np
#import numpy.linalg as npl
#import scipy.sparse as sps
#import scipy.sparse.linalg as spsl
#from numpy.core.umath_tests import matrix_multiply
#import vtool.keypoint as ktool
#import vtool.linalg as ltool
profile = ut.profile
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[constr]', DEBUG=False)


TAU = np.pi * 2


def testdata_matcher():
    import utool as ut
    import vtool as vt
    fpath1 = ut.grab_test_imgpath('easy1.png')
    fpath2 = ut.grab_test_imgpath('easy2.png')
    kpts1, vecs1 = vt.extract_features(fpath1)
    kpts2, vecs2 = vt.extract_features(fpath2)
    rchip1 = vt.imread(fpath1)
    rchip2 = vt.imread(fpath2)
    #chip1_shape = vt.gtool.open_image_size(fpath1)
    chip2_shape = vt.gtool.open_image_size(fpath2)
    dlen_sqrd2 = chip2_shape[0] ** 2 + chip2_shape[1]
    return (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2)


class SimpleMatcher(object):
    def __init__(self):
        self.fm = None
        self.fs = None
        self.H = None
        self.fs_V = None
        self.fm_V = None
        self.basetup = None
        self.testtup = None

    def setstate_testdata(self):
        r"""
        CommandLine:
            python -m vtool.spatially_constrained_matcher --test-setstate_svscr

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.spatially_constrained_matcher import *  # NOQA
            >>> # build test data
            >>> self = SimpleMatcher()
            >>> # execute function
            >>> result = self.setstate_testdata()
            >>> # verify results
            >>> print(result)
        """
        testtup = testdata_matcher()
        rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2 = testtup
        basetup = baseline_vsone_ratio_matcher(rchip1, rchip2, vecs1, vecs2, kpts1, kpts2, dlen_sqrd2)
        nexttup = match_scr(testtup, basetup)
        self.nexttup = nexttup
        self.basetup = basetup
        self.testtup = testtup

    def visualize(self):
        """

        CommandLine:
            python -m vtool.spatially_constrained_matcher --test-visualize --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.spatially_constrained_matcher import *  # NOQA
            >>> # build test data
            >>> self = SimpleMatcher()
            >>> self.setstate_testdata()
            >>> # execute function
            >>> result = self.visualize()
        """
        import plottool as pt
        if getattr(self, 'baseline', None) is None:
            result = self.setstate_testdata()

        rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2 = self.testtup
        fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H      = self.basetup
        fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR              = self.nexttup

        locals_ = locals()

        try:
            del locals_['title']
        except KeyError:
            pass

        fnum = 1
        pt.figure(fnum=fnum, doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nRows=2, nCols=3)

        show_matches(fm_ORIG, fs_ORIG, title='matches: initial neighbors', pnum=next_pnum(), **locals_)

        show_matches(fm_RAT, fs_RAT, title='matches: ratio filtered', pnum=next_pnum(), **locals_)

        show_matches(fm_SV, fs_SV, title='matches: ratio filtered + SV', pnum=next_pnum(), **locals_)

        next_pnum()

        show_matches(fm_SCR, fs_SCR, title='matches: spatially constrained', pnum=next_pnum(), **locals_)

        show_matches(fm_SCRSV, fs_SCRSV, title='matches: spatially constrained + SV', pnum=next_pnum(), **locals_)

        pt.show_if_requested()


def match_scr(testtup, basetup):
    r"""
    spatially constrained ratio matching

    CommandLine:
        python -m vtool.spatially_constrained_matcher --test-match_scr

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.spatially_constrained_matcher import *  # NOQA
        >>> import vtool as vt
        >>> testtup = testdata_matcher()
        >>> (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testtup
        >>> basetup = baseline_vsone_ratio_matcher(rchip1, rchip2, vecs1, vecs2, kpts1, kpts2, dlen_sqrd2)
        >>> self = SimpleMatcher()
        >>> # execute function
        >>> nexttup = match_scr(testtup, basetup)
        >>> # verify results
        >>> print(nexttup)
    """
    #import vtool as vt
    (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testtup
    (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H) = basetup

    match_xy_thresh = .1
    sver_xy_thresh = .01
    # Observation, scores don't change above K=7
    # on easy test case
    search_K = 7  # 3

    # ASSIGN CANDIDATES
    # Get candidate nearest neighbors
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=search_K)

    # COMPUTE CONSTRAINTS
    mode = 'nearby'
    #mode = 'far'
    fx1_list, fx2_list, match_dist_list, norm_dist_list = constrain_matches(
        dlen_sqrd2, kpts1, kpts2, H, fx2_to_fx1, fx2_to_dist, match_xy_thresh, mode=mode)

    def ratio_test2(match_dist_list, norm_dist_list, fx1_list, fx2_list):
        ratio_list = np.divide(match_dist_list, norm_dist_list)
        #ratio_thresh = .625
        #ratio_thresh = .725
        ratio_thresh = .8
        validratio_list = ratio_list < ratio_thresh
        fx2_m = fx2_list[validratio_list]
        fx1_m = fx1_list[validratio_list]
        fs_SCR = np.subtract(1.0, ratio_list[validratio_list])  # NOQA
        fm_SCR = np.vstack((fx1_m, fx2_m)).T  # NOQA
        return fm_SCR, fs_SCR

    fm_SCR, fs_SCR = ratio_test2(match_dist_list, norm_dist_list, fx1_list, fx2_list)

    # Another round of verification
    fm_SVSCR, fs_SVSCR, H_SCR = spatial_verification_(kpts1, kpts2, fm_SCR, fs_SCR, dlen_sqrd2, sver_xy_thresh)

    nexttup = (fm_SCR, fs_SCR, fm_SVSCR, fs_SVSCR, H_SCR)
    return nexttup


def baseline_vsone_ratio_matcher(rchip1, rchip2, vecs1, vecs2, kpts1, kpts2, dlen_sqrd2):
    r"""
    Args:
        vecs1 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        vecs2 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        python -m vtool.spatially_constrained_matcher --test-baseline_vsone_ratio_matcher

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.spatially_constrained_matcher import *  # NOQA
        >>> # build test data
        >>> (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testdata_matcher()
        >>> # execute function
        >>> result = baseline_vsone_ratio_matcher(vecs1, vecs2, kpts1, kpts2)
        >>> # verify results
        >>> print(result)

    Ignore:
        %pylab qt4
        import plottool as pt
        pt.imshow(rchip1)
        pt.draw_kpts2(kpts1)

        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
    """
    #import vtool as vt
    xy_thresh = .01
    ratio_thresh = .625
    # GET NEAREST NEIGHBORS
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=2)
    fx2_m = np.arange(len(fx2_to_fx1))
    fx1_m = fx2_to_fx1.T[0]
    fm_ORIG = np.vstack((fx1_m, fx2_m)).T
    #fs_ORIG = fx2_to_dist.T[0]
    fs_ORIG = 1 - np.divide(fx2_to_dist.T[0], fx2_to_dist.T[1])
    #np.ones(len(fm_ORIG))
    # APPLY RATIO TEST
    fm_RAT, fs_RAT = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
    # SPATIAL VERIFICATION FILTER
    fm_SV, fs_SV, H = spatial_verification_(kpts1, kpts2, fm_RAT, fs_RAT, dlen_sqrd2, xy_thresh)
    base_tup = (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H)
    return base_tup


def assign_nearest_neighbors(vecs1, vecs2, K=2):
    import vtool as vt
    checks = 800
    flann_params = {
        'algorithm': 'kdtree',
        'trees': 8
    }
    #pseudo_max_dist_sqrd = (np.sqrt(2) * 512) ** 2
    pseudo_max_dist_sqrd = 2 * (512 ** 2)
    flann = vt.flann_cache(vecs1, flann_params=flann_params)
    fx2_to_fx1, _fx2_to_dist = flann.nn_index(vecs2, num_neighbors=K, checks=checks)
    fx2_to_dist = np.divide(_fx2_to_dist, pseudo_max_dist_sqrd)
    return fx2_to_fx1, fx2_to_dist


def ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh):
    fx2_to_ratio = np.divide(fx2_to_dist.T[0], fx2_to_dist.T[1])
    fx2_to_isvalid = fx2_to_ratio < ratio_thresh
    fx2_m = np.where(fx2_to_isvalid)[0]
    fx1_m = fx2_to_fx1.T[0].take(fx2_m)
    fs = np.subtract(1.0, fx2_to_ratio.take(fx2_m))
    fm = np.vstack((fx1_m, fx2_m)).T
    return fm, fs


def spatial_verification_(kpts1, kpts2, fm, fs, dlen_sqrd2, xy_thresh):
    import vtool as vt
    scale_thresh = 2
    ori_thresh = TAU / 4.0
    svtup = vt.spatially_verify_kpts(
        kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh, dlen_sqrd2)
    (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) = svtup
    fm_SV = fm.take(homog_inliers, axis=0)
    fs_SV = fs.take(homog_inliers, axis=0)
    return fm_SV, fs_SV, H


def show_matches(fm, fs, fnum=1, pnum=None, title='', **locals_):
    #locals_ = locals()
    import plottool as pt
    # hack keys out of namespace
    keys = 'rchip1, rchip2, kpts1, kpts2'.split(', ')
    rchip1, rchip2, kpts1, kpts2 = ut.dict_take(locals_, keys)
    pt.figure(fnum=fnum, pnum=pnum)
    #doclf=True, docla=True)
    pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs, fnum=fnum)
    title += ' %d, %.2f' % (len(fm), sum(fs))
    pt.set_title(title)
    #pt.set_figtitle(title)
    # if update:
    #pt.iup()


def constrain_matches(dlen_sqrd2, kpts1, kpts2, H, fx2_to_fx1, fx2_to_dist, xy_thresh, mode='far'):
    import vtool as vt
    def get_candidate_spatial_error():
        # Transform img1 keypoints into img2 space
        xy2    = vt.get_xys(kpts2)
        xy1_t = vt.transform_kpts_xys(kpts1, H)
        # get spatial keypoint distance to all neighbor candidates
        bcast_xy2   = xy2[:, None, :].T
        bcast_xy1_t = xy1_t.T[fx2_to_fx1]
        fx2_to_xyerr_sqrd = vt.L2_sqrd(bcast_xy2, bcast_xy1_t)
        return fx2_to_xyerr_sqrd

    # Get the error of each match.
    # This has shape = (nFeats, K)
    fx2_to_xyerr_sqrd = get_candidate_spatial_error()
    fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
    fx2_to_xyerr_norm = fx2_to_xyerr / np.sqrt(dlen_sqrd2)
    # Find candidates which are valid (or nearby)
    fx2_to_isnearby = fx2_to_xyerr_norm < xy_thresh
    #print(ut.get_stats_str(fx2_to_isnearby, newlines=True))
    #(fx2_to_xyerr_sqrd / dlen_sqrd2 < xy_thresh) == fx2_to_isnearby

    #ut.get_stats_str(fx2_to_xyerr)
    print(ut.get_stats_str(fx2_to_xyerr_sqrd, newlines=True))
    print(ut.get_stats_str(fx2_to_xyerr, newlines=True))
    print(ut.get_stats_str(fx2_to_xyerr_norm, newlines=True))

    # FIND FEASIBLE MATCHES
    # isclose is a numpy keyword ...
    def tryget_nearby_first(isnearby):
        """
        get first position that is nearby

        isnearby = np.array([False, False, False,  True,  True, False, False, True, False, False])
        """
        poslist_nearby = np.where(isnearby)[0]
        truepos = None if len(poslist_nearby) == 0 else poslist_nearby[0]
        return truepos

    def tryget_far_second(isnearby, truepos):
        """ get second position that is far away """
        if truepos is None:
            return None
        offset = truepos + 1
        rel_isnearby = isnearby[offset:]
        rel_poslist_far = np.where(~rel_isnearby)[0]
        falsepos = None if len(rel_poslist_far) == 0 else rel_poslist_far[0] + offset
        return falsepos

    def tryget_nearby_second(isnearby, truepos):
        """ get second position that is nearby """
        if truepos is None:
            return None
        offset = truepos + 1
        rel_isnearby = isnearby[offset:]
        rel_poslist_nearby = np.where(rel_isnearby)[0]
        falsepos = None if len(rel_poslist_nearby) == 0 else rel_poslist_nearby[0] + offset
        return falsepos

    tryget_second = tryget_far_second if mode == 'far' else tryget_nearby_second

    # Get the nearbyst decriptor match that is within the constraints
    fx2_to_fx1_first  = [tryget_nearby_first(isnearby)
                         for isnearby in fx2_to_isnearby]

    fx2_to_fx1_second = [tryget_second(isnearby, truepos)
                         for isnearby, truepos in
                         zip(fx2_to_isnearby, fx2_to_fx1_first)]

    assert fx2_to_fx1_first != fx2_to_fx1_second
    fx2_to_hasmatch = [pos is not None for pos in fx2_to_fx1_second]

    # We now have 2d coordinates into fx2_to_fx1
    _shape2d = fx2_to_isnearby.shape
    fx2_list = np.where(fx2_to_hasmatch)[0]
    k_match_list = np.array(ut.list_take(fx2_to_fx1_first, fx2_list))
    k_norm_list = np.array(ut.list_take(fx2_to_fx1_second, fx2_list))
    _match_index_2d = np.vstack((fx2_list, k_match_list))
    _norm_index_2d  = np.vstack((fx2_list, k_norm_list))
    # Covnert into 1d coordinates for flat indexing into fx2_to_fx1
    match_index_1d = np.ravel_multi_index(_match_index_2d, _shape2d)
    norm_index_1d  = np.ravel_multi_index(_norm_index_2d, _shape2d)

    norm_wasnearby = fx2_to_isnearby.take(norm_index_1d)
    match_wasnearby = fx2_to_isnearby.take(match_index_1d)

    # Find initial matches
    fx1_list = fx2_to_fx1.take(match_index_1d)
    # compute constrained ratio score
    match_dist_list = fx2_to_dist.take(match_index_1d)
    norm_dist_list = fx2_to_dist.take(norm_index_1d)

    assert np.all(match_wasnearby), 'index mapping is invalid'
    if mode == 'far':
        assert not np.any(norm_wasnearby), 'index mapping is invalid'
    elif mode == 'nearby':
        assert np.all(norm_wasnearby), 'index mapping is invalid'
    else:
        raise AssertionError('invalid mode=%r' % (mode,))

    assert not np.any(match_index_1d == norm_index_1d), 'index is same'
    assert not np.any(match_dist_list == norm_dist_list), 'dist is same'

    #ut.embed()
    return fx1_list, fx2_list, match_dist_list, norm_dist_list

#def interactive_code():
#    import plottool as pt
#    INTERACT_RATIO = False
#    if INTERACT_RATIO:
#        #ratio_thresh_list = [.625] + np.linspace(.5, .7, 10).tolist()
#        #for ratio_thresh in ut.InteractiveIter(ratio_thresh_list):
#        print('ratio_thresh = %r' % (ratio_thresh,))
#        #fm, fs = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
#        import plottool as pt
#        pt.figure(fnum=1, doclf=True, docla=True)
#        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm_SCR, fs=fs_SCR, fnum=1)
#        pt.set_figtitle('inspect spatially constrained ratio')
#        pt.update()

#    INTERACT_RATIO = False
#    if INTERACT_RATIO:
#        ratio_thresh_list = [.625] + np.linspace(.5, .7, 10).tolist()
#        for ratio_thresh in ut.InteractiveIter(ratio_thresh_list):
#            print('ratio_thresh = %r' % (ratio_thresh,))
#            fm, fs = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
#            pt.figure(fnum=1, doclf=True, docla=True)
#            pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs, fnum=1)
#            pt.set_figtitle('inspect ratio')
#            pt.update()

#    INTERACT_SVER = False
#    if INTERACT_SVER:
#        xy_thresh_list = [xy_thresh] + np.linspace(.01, .1, 10).tolist()
#        for xy_thresh in ut.InteractiveIter(xy_thresh_list):
#            print('xy_thresh = %r' % (xy_thresh,))
#            fm_SV, fs_SV, H = spatial_verification_(kpts1, kpts2, fm, fs, dlen_sqrd2, xy_thresh)
#            pt.figure(fnum=1, doclf=True, docla=True)
#            pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm_SV, fs=fs_SV, fnum=1)
#            pt.set_figtitle('inspect sver')
#            pt.update()

#def spatially_constrained_flann_matcher(flann1, vecs2, kpts1, kpts2, H,
#                                        xy_thresh):
#    pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.spatially_constrained_matcher
        python -m vtool.spatially_constrained_matcher --allexamples
        python -m vtool.spatially_constrained_matcher --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
