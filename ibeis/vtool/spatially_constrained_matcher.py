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


def simple_vsone_ratio_matcher(rchip1, rchip2, vecs1, vecs2, kpts1, kpts2, dlen_sqrd2):
    r"""
    Args:
        vecs1 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        vecs2 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        python -m vtool.spatially_constrained_matcher --test-simple_vsone_ratio_matcher

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.spatially_constrained_matcher import *  # NOQA
        >>> # build test data
        >>> (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testdata_matcher()
        >>> # execute function
        >>> result = simple_vsone_ratio_matcher(vecs1, vecs2, kpts1, kpts2)
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
    import vtool as vt
    import plottool as pt
    xy_thresh = .01
    ratio_thresh = .625
    # GET NEAREST NEIGHBORS
    def assign_nearest_neighbors(vecs1, vecs2):
        checks = 800
        flann_params = {
            'algorithm': 'kdtree',
            'trees': 8
        }
        #pseudo_max_dist_sqrd = (np.sqrt(2) * 512) ** 2
        pseudo_max_dist_sqrd = 2 * (512 ** 2)
        flann = vt.flann_cache(vecs1, flann_params=flann_params)
        fx2_to_fx1, _fx2_to_dist = flann.nn_index(vecs2, num_neighbors=2, checks=checks)
        fx2_to_dist = np.divide(_fx2_to_dist, pseudo_max_dist_sqrd)
        return fx2_to_fx1, fx2_to_dist
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2)

    # APPLY RATIO TEST
    def ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh):
        fx2_to_ratio = np.divide(fx2_to_dist.T[0], fx2_to_dist.T[1])
        fx2_to_isvalid = fx2_to_ratio < ratio_thresh
        fx2_m = np.where(fx2_to_isvalid)[0]
        fx1_m = fx2_to_fx1.T[0].take(fx2_m)
        fs = np.subtract(1.0, fx2_to_ratio.take(fx2_m))
        fm = np.vstack((fx1_m, fx2_m)).T
        return fm, fs
    fm, fs = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)

    # SPATIAL VERIFICATION FILTER
    def spatial_verification(kpts1, kpts2, fm, fs, dlen_sqrd2, xy_thresh):
        scale_thresh = 2
        ori_thresh = TAU / 4.0
        svtup = vt.spatially_verify_kpts(
            kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh, dlen_sqrd2)
        (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) = svtup
        fm_SV = fm.take(homog_inliers, axis=0)
        fs_SV = fs.take(homog_inliers, axis=0)
        return fm_SV, fs_SV, H
    fm_SV, fs_SV, H = spatial_verification(kpts1, kpts2, fm, fs, dlen_sqrd2, xy_thresh)

    INTERACT_RATIO = False
    if INTERACT_RATIO:
        ratio_thresh_list = [.625] + np.linspace(.5, .7, 10).tolist()
        for ratio_thresh in ut.InteractiveIter(ratio_thresh_list):
            print('ratio_thresh = %r' % (ratio_thresh,))
            fm, fs = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
            pt.figure(fnum=1, doclf=True, docla=True)
            pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs, fnum=1)
            pt.set_figtitle('inspect ratio')
            pt.update()

    INTERACT_SVER = False
    if INTERACT_SVER:
        xy_thresh_list = [xy_thresh] + np.linspace(.01, .1, 10).tolist()
        for xy_thresh in ut.InteractiveIter(xy_thresh_list):
            print('xy_thresh = %r' % (xy_thresh,))
            fm_SV, fs_SV, H = spatial_verification(kpts1, kpts2, fm, fs, dlen_sqrd2, xy_thresh)
            pt.figure(fnum=1, doclf=True, docla=True)
            pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm_SV, fs=fs_SV, fnum=1)
            pt.set_figtitle('inspect sver')
            pt.update()


def spatially_constrained_matcher(rchip1, rchip2, vecs1, vecs2, kpts1, kpts2, dlen_sqrd2, fm_SV, H, xy_thresh):
    r"""
    Args:
        vecs1 (?):
        vecs2 (?):
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H (?):
        xy_thresh (?):

    CommandLine:
        python -m vtool.spatially_constrained_matcher --test-spatially_constrained_matcher

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.spatially_constrained_matcher import *  # NOQA
        >>> import vtool as vt
        >>> (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testdata_matcher()
        >>> # build test data
        >>> H = '?'
        >>> xy_thresh = '?'
        >>> # execute function
        >>> result = spatially_constrained_matcher(vecs1, vecs2, kpts1, kpts2, H, xy_thresh)
        >>> # verify results
        >>> print(result)
    """

    xy_thresh = .2

    def assign_nearest_neighbors(vecs1, vecs2):
        K = 10
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

    import vtool as vt
    # ASSIGN CANDIDATES
    # Get candidate nearest neighbors
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2)

    # COMPUTE CONSTRAINTS
    # Transform img1 keypoints into img2 space
    def get_img2space_xys(kpts1, H):
        xyz1   = vt.get_homog_xyzs(kpts1)
        xyz1_t = vt.matrix_multiply(H, xyz1)
        xy1_t  = vt.homogonize(xyz1_t)
        return xy1_t
    xy2    = vt.get_xys(kpts2)
    xy1_t = get_img2space_xys(kpts1, H)
    fx2_to_fx1
    # get spatial keypoint distance to all neighbor candidates
    bcast_xy2   = xy2[:, None, :].T
    bcast_xy1_t = xy1_t.T[fx2_to_fx1]
    fx2_to_xyerr_sqrd = vt.L2_sqrd(bcast_xy2, bcast_xy1_t)
    fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
    fx2_to_xyerr_norm = fx2_to_xyerr / np.sqrt(dlen_sqrd2)
    fx2_to_validcand = fx2_to_xyerr_norm < xy_thresh
    #print(ut.get_stats_str(fx2_to_validcand, newlines=True))
    #(fx2_to_xyerr_sqrd / dlen_sqrd2 < xy_thresh) == fx2_to_validcand

    #ut.get_stats_str(fx2_to_xyerr)
    print(ut.get_stats_str(fx2_to_xyerr_sqrd, newlines=True))
    print(ut.get_stats_str(fx2_to_xyerr, newlines=True))
    print(ut.get_stats_str(fx2_to_xyerr_norm, newlines=True))

    # FIND FEASIBLE MATCHES
    def tryget_close_first(validcand):
        """
        get first position that is close

        validcand = np.array([False, False, False,  True,  True, False, False, True, False, False])
        """
        poslist_true = np.where(validcand)[0]
        truepos = None if len(poslist_true) == 0 else poslist_true[0]
        return truepos

    def tryget_far_second(validcand, truepos):
        if truepos is None:
            return None
        """ get second position that is far away """
        rel_validcand = validcand[truepos:]
        rel_poslist_false = np.where(~rel_validcand)[0]
        falsepos = None if len(rel_poslist_false) == 0 else rel_poslist_false[0] + truepos
        return falsepos

    def tryget_close_second(validcand, truepos):
        """ get second position that is close """
        if truepos is None:
            return None
        rel_validcand = validcand[truepos:]
        rel_poslist_true = np.where(rel_validcand)[0]
        falsepos = None if len(rel_poslist_true) == 0 else rel_poslist_true[0] + truepos
        return falsepos

    # Get the closest decriptor match that is within the constraints
    fx2_to_fx1_first  = [tryget_close_first(validcand)
                         for validcand in fx2_to_validcand]
    fx2_to_fx1_second = [tryget_far_second(validcand, truepos)
                         for validcand, truepos in
                         zip(fx2_to_validcand, fx2_to_fx1_first)]
    fx2_to_hasmatch = [pos is not None for pos in fx2_to_fx1_second]

    # We now have 2d coordinates into fx2_to_fx1
    fx2_list = np.where(fx2_to_hasmatch)[0]
    k_match_list = np.array(ut.list_take(fx2_to_fx1_first, fx2_list))
    k_norm_list = np.array(ut.list_take(fx2_to_fx1_second, fx2_list))
    match_index_2d = np.vstack((fx2_list, k_match_list))
    norm_index_2d  = np.vstack((fx2_list, k_norm_list))
    # Covnert into 1d coordinates for flat indexing into fx2_to_fx1
    match_index_1d = np.ravel_multi_index(match_index_2d, fx2_to_validcand.shape)
    norm_index_1d = np.ravel_multi_index(norm_index_2d, fx2_to_validcand.shape)

    assert np.all(fx2_to_validcand.take(match_index_1d)), 'index mapping is invalid'
    assert not np.any(fx2_to_validcand.take(norm_index_1d)), 'index mapping is invalid'

    # Find initial matches
    fx1_list = fx2_to_fx1.take(match_index_1d)
    # compute constrained ratio score
    match_dist_list = fx2_to_dist.take(match_index_1d)
    norm_dist_list = fx2_to_dist.take(norm_index_1d)

    ratio_list = np.divide(match_dist_list, norm_dist_list)

    #ratio_thresh = .625
    #ratio_thresh = .725
    ratio_thresh = .8
    validratio_list = ratio_list < ratio_thresh
    fx2_m = fx2_list[validratio_list]
    fx1_m = fx1_list[validratio_list]
    fs_SCR = np.subtract(1.0, ratio_list[validratio_list])
    fm_SCR = np.vstack((fx1_m, fx2_m)).T

    INTERACT_RATIO = False
    if INTERACT_RATIO:
        #ratio_thresh_list = [.625] + np.linspace(.5, .7, 10).tolist()
        #for ratio_thresh in ut.InteractiveIter(ratio_thresh_list):
        print('ratio_thresh = %r' % (ratio_thresh,))
        #fm, fs = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
        import plottool as pt
        pt.figure(fnum=1, doclf=True, docla=True)
        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm_SCR, fs=fs_SCR, fnum=1)
        pt.set_figtitle('inspect spatially constrained ratio')
        pt.update()


def spatially_constrained_flann_matcher(flann1, vecs2, kpts1, kpts2, H,
                                        xy_thresh):
    pass


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
