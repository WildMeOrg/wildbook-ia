# -*- coding: utf-8 -*-
"""
Spatial verification of keypoint matches

Notation::
    1_m = img1_matches; 2_m = img2_matches
    x and y are locations, invV is the elliptical shapes.
    fx are the original feature indexes (used for making sure 1 keypoint isn't assigned to 2)

Look Into::
    Standard
    skimage.transform
    http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
    skimage.transform.fast_homography(im, H)

FIXME:
    is it scaled_thresh or scaled_thresh_sqrd

References:
    http://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws11-12/3DCV_WS11-12_lec04.pdf
    http://www.imgfsr.com/CVPR2011/Tutorial6/RANSAC_CVPR2011.pdf
    http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf Page 317

Notes:
    Invariants of affine transforms - parallel lines, ratios of parallel lengths, ratios of areas
    Invariants of homographies - cross‐ratio of four points on a line (ratio of ratio)
"""
from __future__ import absolute_import, division, print_function
from six.moves import range
import warnings  # NOQA
import six  # NOQA
import utool as ut
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from numpy.core.umath_tests import matrix_multiply
import vtool.keypoint as ktool
import vtool.linalg as ltool
import vtool.distance
import cv2
from .util_math import TAU

try:
    from vtool import sver_c_wrapper
    HAVE_SVER_C_WRAPPER = not ut.get_argflag('--no-c')
except Exception as ex:
    HAVE_SVER_C_WRAPPER = False
    if ut.VERBOSE:
        ut.printex(ex, 'please build the sver c wrapper (run with --rebuild-sver')
    if False:
        raise


VERBOSE_SVER = ut.get_argflag('--verb-sver')

SV_DTYPE = np.float64
INDEX_DTYPE = np.int32


def build_lstsqrs_Mx9(xy1_mn, xy2_mn):
    """ Builds the M x 9 least squares matrix

    CommandLine:
        python -m vtool.spatial_verification --test-build_lstsqrs_Mx9

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair()
        >>> xy1_mn = ktool.get_xys(kpts1).astype(np.float64)
        >>> xy2_mn = ktool.get_xys(kpts2).astype(np.float64)
        >>> Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
        >>> result = (ub.repr2(Mx9[0:2], suppress_small=True, precision=2, with_dtype=True))
        >>> print(result)
        np.array([[  0.00e+00,   0.00e+00,   0.00e+00,  -3.20e+01,  -2.72e+01,
                    -1.00e+00,   8.82e+02,   7.49e+02,   2.76e+01],
                  [  3.20e+01,   2.72e+01,   1.00e+00,   0.00e+00,   0.00e+00,
                     0.00e+00,  -1.09e+03,  -9.28e+02,  -3.42e+01]], dtype=np.float64)

    References:
        http://dip.sun.ac.za/~stefan/TW793/attach/notes/homography_estimation.pdf
        http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf Page 317
        http://vision.ece.ucsb.edu/~zuliani/Research/RANSAC/docs/RANSAC4Dummies.pdf page 53
    """
    x1_mn = xy1_mn[0]
    y1_mn = xy1_mn[1]
    x2_mn = xy2_mn[0]
    y2_mn = xy2_mn[1]
    num_pts = x1_mn.shape[0]
    #Mx9 = np.zeros((2 * num_pts, 9), dtype=SV_DTYPE)
    Mx9 = np.empty((2 * num_pts, 9), dtype=SV_DTYPE)
    for ix in range(num_pts):  # Loop over inliers
        # Concatenate all 2x9 matrices into an Mx9 matrix
        u2        = x2_mn[ix]
        v2        = y2_mn[ix]
        x1        = x1_mn[ix]
        y1        = y1_mn[ix]
        (d, e, f) = (     -x1,      -y1,  -1)
        (g, h, i) = ( v2 * x1,  v2 * y1,  v2)
        (j, k, l) = (      x1,       y1,   1)
        (p, q, r) = (-u2 * x1, -u2 * y1, -u2)
        Mx9[ix * 2]     = (0, 0, 0, d, e, f, g, h, i)
        Mx9[ix * 2 + 1] = (j, k, l, 0, 0, 0, p, q, r)
    return Mx9


def try_svd(M):
    """
    CommandLine:
        python -m vtool.spatial_verification try_svd

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> rng = np.random.RandomState(42)
        >>> num = 1000
        >>> xy1_mn = rng.randn(2, num)
        >>> xy2_mn = rng.randn(2, num)
        >>> M = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
        >>> print('M.shape = %r' % (M.shape,))
        >>> USV = npl.svd(M, full_matrices=True, compute_uv=True)
        >>> USV = try_svd(M)

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> num = np.ceil(np.sqrt(2000))
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair(wh_num=(num, num))
        >>> xy1_mn = ktool.get_xys(kpts1).astype(np.float64)
        >>> xy2_mn = ktool.get_xys(kpts2).astype(np.float64)
        >>> M = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
        >>> print('M.shape = %r' % (M.shape,))
        >>> USV = npl.svd(M, full_matrices=True, compute_uv=True)
        >>> USV = try_svd(M)
    """
    #if M.shape[0] > 2500:
    #    # hack to prevent bug in lapack
    #    M = M[:2500]
    try:
        USV = npl.svd(M, full_matrices=True, compute_uv=True)
    except MemoryError as ex:
        ut.printex(ex, '[sver] Caught MemErr during full SVD. Trying sparse SVD.')
        M_sparse = sps.lil_matrix(M)
        USV = spsl.svds(M_sparse)
    except npl.LinAlgError as ex:
        ut.printex(ex, '[sver] svd did not converge')
        raise
    except Exception as ex:
        ut.printex(ex, '[sver] svd error')
        raise
    return USV


def build_affine_lstsqrs_Mx6(xy1_man, xy2_man):
    """
    CURRENTLY NOT WORKING

    CommandLine:
        python -m vtool.spatial_verification --test-build_affine_lstsqrs_Mx6

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair()
        >>> xy1_man = ktool.get_xys(kpts1).astype(np.float64)
        >>> xy2_man = ktool.get_xys(kpts2).astype(np.float64)
        >>> Mx6 = build_affine_lstsqrs_Mx6(xy1_man, xy2_man)
        >>> print(ub.repr2(Mx6))
        >>> result = ut.hashstr(Mx6)
        >>> print(result)

    Sympy:
        import sympy as sym
        x1, y1, x2, y2 = sym.symbols('x1, y1, x2, y2')
        A = sym.Matrix([
            [x1, y1,  0,  0, 1, 0],
            [ 0,  0, x1, y1, 0, 1],
        ])
        b = sym.Matrix([[x2], [y2]])
        x = (A.T.multiply(A)).inv().multiply(A.T.multiply(b))
        x = (A.T.multiply(A)).pinv().multiply(A.T.multiply(b))

    References:
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf page 22
    """
    x1_mn = xy1_man[0]
    y1_mn = xy1_man[1]
    x2_mn = xy2_man[0]
    y2_mn = xy2_man[1]
    num_pts = x1_mn.shape[0]
    Mx6 = np.empty((2 * num_pts, 6), dtype=SV_DTYPE)
    b = np.empty((2 * num_pts, 1), dtype=SV_DTYPE)
    for ix in range(num_pts):  # Loop over inliers
        # Concatenate all 2x9 matrices into an Mx6 matrix
        x1 = x1_mn[ix]
        x2 = x2_mn[ix]
        y1 = y1_mn[ix]
        y2 = y2_mn[ix]
        Mx6[ix * 2]     = (x1, y1,  0,  0,  1,  0)
        Mx6[ix * 2 + 1] = ( 0,  0, x1, y1,  0,  1)
        b[ix * 2] = x2
        b[ix * 2 + 1] = y2

    #npl.solve(b, Mx6)
    U, s, Vt = try_svd(Mx6)

    if False:
        S_ = np.zeros((len(U), len(Vt)))
        S_[np.diag_indices(len(s))] = s
        U.dot(S_).dot(Vt)
        assert np.allclose(U.dot(S_).dot(Vt), Mx6)
        assert not np.allclose(U.dot(S_).dot(Vt.T), Mx6)

    # Inefficient, but I think the math works
    # We want to solve Ax=b (where A is the Mx6 in this case)
    # Ax = b
    # (U S V.T) x = b
    # x = (U.T inv(S) V) b
    Sinv = np.zeros((len(Vt), len(U)))
    Sinv[np.diag_indices(len(s))] = 1 / s
    a = Vt.T.dot(Sinv).dot(U.T).dot(b).T[0]
    A = np.array([
        [a[0], a[1], a[4]],
        [a[2], a[3], a[5]],
        [   0,    0,    1],
    ])
    return A
    # TODO FIXME
    #return Mx6


def compute_affine(xy1_man, xy2_man):
    """
    Args:
        xy1_mn (ndarray[ndim=2]): xy points in image1
        xy2_mn (ndarray[ndim=2]): corresponding xy points in image 2

    Returns:
        ndarray[shape=(3,3)]: A - affine matrix

    CommandLine:
        python -m vtool.spatial_verification --test-compute_affine:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> import vtool.keypoint as ktool
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair()
        >>> xy1_mn = ktool.get_xys(kpts1)
        >>> xy2_mn = ktool.get_xys(kpts2)
        >>> A = compute_affine(xy1_mn, xy1_mn)
        >>> result =str(A)
        >>> result = np.array_str(A, precision=2)
        >>> print(result)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> import vtool.keypoint as ktool
        >>> import plottool as pt
        >>> xy1_man, xy2_man, rchip1, rchip2, T1, T2 = testdata_matching_affine_inliers_normalized()
        >>> A_prime = compute_affine(xy1_man, xy2_man)
        >>> A = npl.solve(T2, A_prime).dot(T1)
        >>> A /= A[2, 2]
        >>> result = np.array_str(A, precision=2)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> rchip2_blendA = pt.draw_sv.get_blended_chip(rchip1, rchip2, A)
        >>> pt.imshow(rchip2_blendA)
        >>> ut.show_if_requested()
        [[  1.19e+00  -1.06e-02  -4.49e+01]
         [ -2.22e-01   1.12e+00  -2.78e+01]
         [  0.00e+00   0.00e+00   1.00e+00]]
    """
    # Solve for the nullspace of the Mx6 matrix (solves least squares)
    #Mx6 = build_affine_lstsqrs_Mx6(xy1_man, xy2_man)
    A = build_affine_lstsqrs_Mx6(xy1_man, xy2_man)
    #U, S, V = try_svd(Mx6)
    #a = V[5]  # Hack for Cython.wraparound(False)
    #a = a / a[-1]
    #A = np.array([
    #    [a[0], a[1], a[4]],
    #    [a[2], a[3], a[5]],
    #    [   0,    0,    1],
    #])
    return A


def compute_homog(xy1_mn, xy2_mn):
    """
    Generate 6 degrees of freedom homography transformation
    Computes homography from normalized (0 to 1) point correspondences
    from 2 --> 1
    (database->query)

    Args:
        xy1_mn (ndarray[ndim=2]): xy points in image1
        xy2_mn (ndarray[ndim=2]): corresponding xy points in image 2

    Returns:
        ndarray[shape=(3,3)]: H - homography matrix

    CommandLine:
        python -m vtool.spatial_verification --test-compute_homog:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.keypoint as ktool
        >>> import vtool.demodata as demodata
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair()
        >>> xy1_mn = ktool.get_xys(kpts1)
        >>> xy2_mn = ktool.get_xys(kpts2)
        >>> H = compute_homog(xy1_mn, xy2_mn)
        >>> #result = ut.hashstr(H)
        >>> result = np.array_str(H, precision=2)
        >>> print(result)
        [[  1.83e-03   2.85e-03  -7.11e-01]
         [  2.82e-03   1.80e-03  -7.03e-01]
         [  1.67e-05   1.68e-05  -5.53e-03]]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.keypoint as ktool
        >>> import plottool as pt
        >>> xy1_man, xy2_man, rchip1, rchip2, T1, T2 = testdata_matching_affine_inliers_normalized()
        >>> H_prime = compute_homog(xy1_man, xy2_man)
        >>> H = npl.solve(T2, H_prime).dot(T1)
        >>> H /= H[2, 2]
        >>> result = np.array_str(H, precision=2)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> rchip2_blendH = pt.draw_sv.get_blended_chip(rchip1, rchip2, H)
        >>> pt.imshow(rchip2_blendH)
        >>> ut.show_if_requested()
        [[  9.22e-01  -2.50e-01   2.75e+01]
         [ -2.04e-01   8.79e-01  -7.94e+00]
         [ -1.82e-04  -5.99e-04   1.00e+00]]
    """
    # Solve for the nullspace of the Mx9 matrix (solves least squares)
    Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
    U, S, V = try_svd(Mx9)
    # Rearange the nullspace into a homography
    #h = V[-1]  # v = V.H
    h = V[8]  # Hack for Cython.wraparound(False)
    # FIXME: THERE IS A BUG HERE, sometimes len(V) = 6. why???
    H = np.vstack((h[0:3], h[3:6], h[6:9]))
    return H


def testdata_matching_affine_inliers():
    import vtool.demodata as demodata
    import vtool as vt
    scale_thresh = 2.0
    xy_thresh = ut.get_argval('--xy-thresh', type_=float, default=.01)
    dlen_sqrd2 = 447271.015
    ori_thresh = 1.57
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    featkw = ut.argparse_dict(vt.get_extract_features_default_params())
    fname1 = ut.get_argval('--fname1', type_=str, default='easy1.png')
    fname2 = ut.get_argval('--fname2', type_=str, default='easy2.png')
    (kpts1, kpts2, fm, fs, rchip1, rchip2) = demodata.testdata_ratio_matches(fname1, fname2, **featkw)
    aff_inliers, aff_errors, Aff = get_best_affine_inliers_(
        kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh)
    return kpts1, kpts2, fm, aff_inliers, rchip1, rchip2, xy_thresh_sqrd


def testdata_matching_affine_inliers_normalized():
    tup = testdata_matching_affine_inliers()
    kpts1, kpts2, fm, aff_inliers, rchip1, rchip2, xy_thresh_sqrd = tup
    kpts1_ma = kpts1.take(fm.T[0].take(aff_inliers), axis=0)
    kpts2_ma = kpts2.take(fm.T[1].take(aff_inliers), axis=0)
    #kpts1_ma, kpts2_ma, rchip1, rchip2, xy_thresh_sqrd = testdata_matching_affine_inliers()
    # Matching affine inliers
    xy1_ma = ktool.get_xys(kpts1_ma)
    xy2_ma = ktool.get_xys(kpts2_ma)
    # Matching affine inliers normalized
    xy1_man, T1 = ltool.whiten_xy_points(xy1_ma)
    xy2_man, T2 = ltool.whiten_xy_points(xy2_ma)
    return xy1_man, xy2_man, rchip1, rchip2, T1, T2


def _test_hypothesis_inliers(Aff, invVR1s_m, xy2_m, det2_m, ori2_m,
                             xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    """
    Critical section code. Inner loop of _test_hypothesis_inliers

    Returns:
        tuple: hypo_inliers, hypo_errors

    CommandLine:
        python -m vtool.spatial_verification --test-_test_hypothesis_inliers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> from vtool.spatial_verification import _test_hypothesis_inliers  # NOQA
        >>> import vtool.demodata as demodata
        >>> import vtool.keypoint as ktool
        >>> _kw1 = dict(seed=12, damping=1.2, wh_stride=(30, 30))
        >>> _kw2 = dict(seed=24, damping=1.6, wh_stride=(30, 30))
        >>> kpts1 = demodata.perterbed_grid_kpts(**_kw1).astype(np.float64)
        >>> kpts2 = demodata.perterbed_grid_kpts(**_kw2).astype(np.float64)
        >>> fm = demodata.make_dummy_fm(len(kpts1)).astype(np.int32)
        >>> kpts1_m = kpts1[fm.T[0]]
        >>> kpts2_m = kpts2[fm.T[1]]
        >>> xy_thresh_sqrd = np.float64(.009) ** 2
        >>> scale_thresh_sqrd = np.float64(2)
        >>> ori_thresh = np.float64(TAU / 4)
        >>> # Get keypoints to project in matrix form
        >>> #invVR1s_m = ktool.get_invV_mats(kpts1_m, with_trans=True, with_ori=True)
        >>> #print(invVR1s_m[0])
        >>> invVR1s_m = ktool.get_invVR_mats3x3(kpts1_m)
        >>> RV1s_m = ktool.get_RV_mats_3x3(kpts1_m)
        >>> invVR2s_m = ktool.get_invVR_mats3x3(kpts2_m)
        >>> # The transform from kp1 to kp2 is given as:
        >>> Aff_mats = matrix_multiply(invVR2s_m, RV1s_m)
        >>> Aff = Aff_mats[0]
        >>> # Get components to test projects against
        >>> xy2_m  = ktool.get_invVR_mats_xys(invVR2s_m)
        >>> det2_m = ktool.get_sqrd_scales(kpts2_m)
        >>> ori2_m = ktool.get_invVR_mats_oris(invVR2s_m)
        >>> output = _test_hypothesis_inliers(Aff, invVR1s_m, xy2_m, det2_m, ori2_m, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)
        >>> output_str = ub.repr2(output, precision=2, suppress_small=True)
        >>> print('output_str = %s' % (output_str,))
        >>> hypo_inliers, hypo_errors = output
        >>> # Inverting matrices is different in python2
        >>> result = 'nInliers=%r hash=%s' % (len(hypo_inliers), ut.hash_data(output))
        >>> print(result)
        nInliers=1 hash=mvbcrvpbtylihsytiowbtcsmozlmoaun

    Ignore:
        kpts = kpts1_m
        ut.hash_data(kpts)
        ut.hash_data(invVR2s_m)
        ut.hash_data(kpts1_m)
        ut.hash_data(RV1s_m)
        hist1, hist2 = xy2_m.T, _xy1_mt.T
        dtype = SV_DTYPE

    Timeit:
        %timeit xy_err < xy_thresh_sqrd
        %timeit np.less(xy_err, xy_thresh_sqrd)
    """
    # Map keypoints from image 1 onto image 2
    invVR1s_mt = matrix_multiply(Aff, invVR1s_m)

    # Get projection components
    _xy1_mt   = ktool.get_invVR_mats_xys(invVR1s_mt)
    _det1_mt  = ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
    _ori1_mt  = ktool.get_invVR_mats_oris(invVR1s_mt)
    ## Check for projection errors
    xy_err    = vtool.distance.L2_sqrd(xy2_m.T, _xy1_mt.T, dtype=SV_DTYPE)
    scale_err = vtool.distance.det_distance(_det1_mt, det2_m)
    ori_err   = vtool.distance.ori_distance(_ori1_mt, ori2_m)

    # Mark keypoints which are inliers to this hypothosis
    xy_inliers_flag    = np.less(xy_err, xy_thresh_sqrd)
    scale_inliers_flag = np.less(scale_err, scale_thresh_sqrd)
    ori_inliers_flag   = np.less(ori_err, ori_thresh)
    #np.logical_and(xy_inliers_flag, scale_inliers_flag)
    # TODO Add uniqueness of matches constraint
    #hypo_inliers_flag = np.empty(xy_inliers_flag.size, dtype=np.bool)
    hypo_inliers_flag = xy_inliers_flag  # Try to re-use memory
    np.logical_and(hypo_inliers_flag, ori_inliers_flag, out=hypo_inliers_flag)
    np.logical_and(hypo_inliers_flag, scale_inliers_flag, out=hypo_inliers_flag)
    #hypo_inliers_flag = np.logical_and.reduce(
    #    [xy_inliers_flag, ori_inliers_flag, scale_inliers_flag])
    # this is also slower
    hypo_inliers = np.where(hypo_inliers_flag)[0]
    hypo_errors = (xy_err, ori_err, scale_err)
    return hypo_inliers, hypo_errors


def get_affine_inliers(kpts1, kpts2, fm, fs,
                        xy_thresh_sqrd,
                        scale_thresh_sqrd,
                        ori_thresh):
    """
    Estimates inliers deterministically using elliptical shapes

    Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    We transform from chip1 -> chip2
    The determinants are squared keypoint scales

    Returns:
        tuple: aff_inliers_list, aff_errors_list, Aff_mats

    Notes:
        FROM PERDOCH 2009::
            H = inv(Aj).dot(Rj.T).dot(Ri).dot(Ai)
            H = inv(Aj).dot(Ai)
            The input invVs = perdoch.invA's

    CommandLine:
        python2 -m vtool.spatial_verification --test-get_affine_inliers
        python3 -m vtool.spatial_verification --test-get_affine_inliers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> import vtool.keypoint as ktool
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair((100, 100))
        >>> fm = demodata.make_dummy_fm(len(kpts1)).astype(np.int32)
        >>> fs = np.ones(len(fm), dtype=np.float64)
        >>> xy_thresh_sqrd = ktool.KPTS_DTYPE(.009) ** 2
        >>> scale_thresh_sqrd = ktool.KPTS_DTYPE(2)
        >>> ori_thresh = ktool.KPTS_DTYPE(TAU / 4)
        >>> output = get_affine_inliers(kpts1, kpts2, fm, fs, xy_thresh_sqrd,
        >>>                             scale_thresh_sqrd, ori_thresh)
        >>> output_str = ut.repr3(output, precision=2, suppress_small=True)
        >>> print('output_str = %s' % (output_str,))
        >>> aff_inliers_list, aff_errors_list, Aff_mats = output
        >>> result = 'nInliers=%r hash=%s' % (len(aff_inliers_list), ut.hash_data(output_str))
        >>> print(result)
        nInliers=9 hash=lwmgwuyvameoegkgmfrrvkjkykqxlapd

    Ignore::
        from vtool.spatial_verification import *  # NOQA
        import vtool.demodata as demodata
        import vtool.keypoint as ktool
        kpts1, kpts2 = demodata.get_dummy_kpts_pair((100, 100))
        a = kpts1[fm.T[0]]
        b = kpts1.take(fm.T[0])

        align = fm.dtype.itemsize * fm.shape[1]
        align2 = [fm.dtype.itemsize, fm.dtype.itemsize]
        viewtype1 = np.dtype(np.void, align)
        viewtype2 = np.dtype(np.int32, align2)
        c = np.ascontiguousarray(fm).view(viewtype1)
        fm_view = np.ascontiguousarray(fm).view(viewtype1)
        qfx = fm.view(np.dtype(np.int32 np.int32.itemsize))
        dfx = fm.view(np.dtype(np.int32, np.int32.itemsize))
        d = np.ascontiguousarray(c).view(viewtype2)

        fm.view(np.dtype(np.void, align))
        np.ascontiguousarray(fm).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
    """
    #http://ipython-books.github.io/featured-01/
    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)

    # Get keypoints to project in matrix form
    #invVR2s_m = ktool.get_invV_mats(kpts2_m, with_trans=True, with_ori=True)
    #invVR1s_m = ktool.get_invV_mats(kpts1_m, with_trans=True, with_ori=True)
    invVR2s_m = ktool.get_invVR_mats3x3(kpts2_m)
    invVR1s_m = ktool.get_invVR_mats3x3(kpts1_m)
    RV1s_m    = ktool.invert_invV_mats(invVR1s_m)  # 539 us
    # BUILD ALL HYPOTHESIS TRANSFORMS: The transform from kp1 to kp2 is:
    Aff_mats = matrix_multiply(invVR2s_m, RV1s_m)
    # Get components to test projects against
    xy2_m  = ktool.get_xys(kpts2_m)
    det2_m = ktool.get_sqrd_scales(kpts2_m)
    ori2_m = ktool.get_oris(kpts2_m)
    # SLOWER EQUIVALENT
    # RV1s_m    = ktool.get_V_mats(kpts1_m, with_trans=True, with_ori=True)  # 5.2 ms
    # xy2_m  = ktool.get_invVR_mats_xys(invVR2s_m)
    # ori2_m = ktool.get_invVR_mats_oris(invVR2s_m)
    # assert np.all(ktool.get_oris(kpts2_m) == ktool.get_invVR_mats_oris(invVR2s_m))
    # assert np.all(ktool.get_xys(kpts2_m) == ktool.get_invVR_mats_xys(invVR2s_m))

    # The previous versions of this function were all roughly comparable.
    # The for loop one was the slowest. I'm deciding to go with the one
    # where there is no internal function definition. It was moderately faster,
    # and it gives us access to profile that function
    inliers_and_errors_list = [_test_hypothesis_inliers(Aff, invVR1s_m, xy2_m,
                                                        det2_m, ori2_m,
                                                        xy_thresh_sqrd,
                                                        scale_thresh_sqrd,
                                                        ori_thresh)
                               for Aff in Aff_mats]
    aff_inliers_list = [tup[0] for tup in inliers_and_errors_list]
    aff_errors_list  = [tup[1] for tup in inliers_and_errors_list]
    return aff_inliers_list, aff_errors_list, Aff_mats


def get_best_affine_inliers(kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh,
                            ori_thresh, forcepy=False):
    """ Tests each hypothesis and returns only the best transformation and inliers
    """
    # Test each affine hypothesis
    # get list if inliers, errors, the affine matrix for each hypothesis
    if HAVE_SVER_C_WRAPPER and not forcepy:
        aff_inliers_list, aff_errors_list, Aff_mats = sver_c_wrapper.get_affine_inliers_cpp(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh)
    else:
        aff_inliers_list, aff_errors_list, Aff_mats = get_affine_inliers(kpts1, kpts2,
                                                                         fm, fs,
                                                                         xy_thresh_sqrd,
                                                                         scale_thresh,
                                                                         ori_thresh)

    #ut.embed()
    # Determine the best hypothesis using the number of inliers
    # TODO: other measures in the error lists could be used as well
    #nInliers_list = np.array([len(inliers) for inliers in aff_inliers_list])

    weight_list = np.array([fs.take(inliers).sum() for inliers in aff_inliers_list])
    #ut.embed()
    #sortx = weight_list.argsort()[::-1]  # sort by non-inliers
    #best_index = sortx[0]  # chose best
    best_index = weight_list.argmax()
    aff_inliers = aff_inliers_list[best_index]
    aff_errors = aff_errors_list[best_index]
    Aff = Aff_mats[best_index]
    return aff_inliers, aff_errors, Aff


def get_normalized_affine_inliers(kpts1, kpts2, fm, aff_inliers):
    """
    returns xy-inliers that are normalized to have a mean of 0 and std of 1 as
    well as the transformations so the inverse can be taken
    """
    fm_affine = fm.take(aff_inliers, axis=0)
    # Get corresponding points and shapes
    kpts1_ma = kpts1.take(fm_affine.T[0], axis=0)
    kpts2_ma = kpts2.take(fm_affine.T[1], axis=0)
    #kpts1_ma = kpts1.take(fm_affine.T[0], axis=0)
    #kpts2_ma = kpts2.take(fm_affine.T[1], axis=0)
    # Normalize affine inliers xy locations
    xy1_ma = ktool.get_xys(kpts1_ma)
    xy2_ma = ktool.get_xys(kpts2_ma)
    xy1_man, T1 = ltool.whiten_xy_points(xy1_ma)
    xy2_man, T2 = ltool.whiten_xy_points(xy2_ma)
    return xy1_man, xy2_man, T1, T2


def unnormalize_transform(M_prime, T1, T2):
    # Then compute ax = b  [aka: x = npl.solve(a, b)]
    M = npl.solve(T2, M_prime).dot(T1)  # Unnormalize
    # homographies that only differ by a scale factor are equivalent
    M /= M[2, 2]
    return M


def estimate_refined_transform(kpts1, kpts2, fm, aff_inliers, refine_method='homog'):
    """ estimates final transformation using normalized affine inliers

    References:
        http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    """
    xy1_man, xy2_man, T1, T2 = get_normalized_affine_inliers(kpts1, kpts2, fm, aff_inliers)
    # Compute homgraphy transform from chip1 -> chip2 using affine inliers
    # homographys assume the two images are planar, or the camera is
    # rotating around the subject

    if refine_method == 'homog':
        H_prime = compute_homog(xy1_man, xy2_man)
    elif refine_method == 'affine':
        H_prime = compute_affine(xy1_man, xy2_man)
    elif refine_method == 'cv2-homog':
        H_prime, mask = cv2.findHomography(xy1_man.T, xy2_man.T, method=0)
    elif refine_method == 'cv2-ransac-homog':
        H_prime, mask = cv2.findHomography(
            xy1_man.T, xy2_man.T, method=cv2.RANSAC,
            ransacReprojThreshold=3)
    elif refine_method == 'cv2-lmeds-homog':
        H_prime, mask = cv2.findHomography(xy1_man.T, xy2_man.T, method=cv2.LMEDS)
    #elif refine_method == 'fund':
    #    # the fundamental matrix only implies: x'.T.dot(F).dot(x) == 0
    #    # it maps a point from one image onto a line in the second image.
    #    H_prime = cv2.findFundamentalMat(xy1_man.T, xy2_man.T, method=cv2.FM_LMEDS)[0]
    #    H_prime = cv2.findFundamentalMat(xy1_man.T, xy2_man.T, method=cv2.FM_8POINT)[0]
    else:
        raise NotImplementedError('[vtool] Unknown refine_method=%r' % (refine_method,))

    #H_prime /= H_prime[2, 2]
    # Different methods?
    #H_prime = compute_affine(xy1_man, xy2_man)
    #import cv2
    #H_prime = cv2.findHomography(xy1_man.T, xy2_man.T)[0]
    #H = compute_affine(xy1_ma, xy2_ma)
    #print(H)
    H = unnormalize_transform(H_prime, T1, T2)
    rank = npl.matrix_rank(H)
    #print(rank)
    if rank != 3:
        raise npl.LinAlgError('Rank defficient homography ')
    return H


def test_homog_errors(H, kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh,
                      ori_thresh, full_homog_checks=True):
    r"""
    Test to see which keypoints the homography correctly maps

    Args:
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        fm (list):  list of feature matches as tuples (qfx, dfx)
        xy_thresh_sqrd (float):
        scale_thresh (float):
        ori_thresh (float):  angle in radians
        full_homog_checks (bool):

    Returns:
        tuple: homog_tup1

    CommandLine:
        python -m vtool.spatial_verification --test-test_homog_errors:0 --show
        python -m vtool.spatial_verification --test-test_homog_errors:0 --show --rotation_invariance
        python -m vtool.spatial_verification --test-test_homog_errors:0 --show --rotation_invariance --no-affine-invariance --xy-thresh=.001
        python -m vtool.spatial_verification --test-test_homog_errors:0 --show --rotation_invariance --no-affine-invariance --xy-thresh=.001 --no-full-homog-checks
        python -m vtool.spatial_verification --test-test_homog_errors:0 --show --no-full-homog-checks
        # --------------
        # Shows (sorta) how inliers are computed
        python -m vtool.spatial_verification --test-test_homog_errors:1 --show
        python -m vtool.spatial_verification --test-test_homog_errors:1 --show --rotation_invariance
        python -m vtool.spatial_verification --test-test_homog_errors:1 --show --rotation_invariance --no-affine-invariance --xy-thresh=.001
        python -m vtool.spatial_verification --test-test_homog_errors:1 --show --rotation_invariance --xy-thresh=.001
        python -m vtool.spatial_verification --test-test_homog_errors:0 --show --rotation_invariance --xy-thresh=.001

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import plottool as pt
        >>> kpts1, kpts2, fm, aff_inliers, rchip1, rchip2, xy_thresh_sqrd = testdata_matching_affine_inliers()
        >>> H = estimate_refined_transform(kpts1, kpts2, fm, aff_inliers)
        >>> scale_thresh, ori_thresh = 2.0, 1.57
        >>> full_homog_checks = not ut.get_argflag('--no-full-homog-checks')
        >>> homog_tup1 = test_homog_errors(H, kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh, ori_thresh, full_homog_checks)
        >>> homog_tup = (homog_tup1[0], homog_tup1[2])
        >>> ut.quit_if_noshow()
        >>> pt.draw_sv.show_sv(rchip1, rchip2, kpts1, kpts2, fm, homog_tup=homog_tup)
        >>> ut.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import plottool as pt
        >>> kpts1, kpts2, fm_, aff_inliers, rchip1, rchip2, xy_thresh_sqrd = testdata_matching_affine_inliers()
        >>> H = estimate_refined_transform(kpts1, kpts2, fm_, aff_inliers)
        >>> scale_thresh, ori_thresh = 2.0, 1.57
        >>> full_homog_checks = not ut.get_argflag('--no-full-homog-checks')
        >>> # ----------------
        >>> # Take subset of feature matches
        >>> fm = fm_
        >>> scale_err, xy_err, ori_err = \
        ...     ut.exec_func_src(test_homog_errors, globals(), locals(),
        ...     'scale_err, xy_err, ori_err'.split(', '))
        >>> # we only care about checking out scale and orientation here. ignore bad xy points
        >>> xy_inliers_flag = np.less(xy_err, xy_thresh_sqrd)
        >>> scale_err[~xy_inliers_flag] = 0
        >>> # filter
        >>> fm = fm_[np.array(scale_err).argsort()[::-1][:10]]
        >>> fm = fm_[np.array(scale_err).argsort()[::-1][:10]]
        >>> # Exec sourcecode
        >>> kpts1_m, kpts2_m, off_xy1_m, off_xy1_mt, dxy1_m, dxy1_mt, xy2_m, xy1_m, xy1_mt, scale_err, xy_err, ori_err = \
        ...     ut.exec_func_src(test_homog_errors, globals(), locals(),
        ...     'kpts1_m, kpts2_m, off_xy1_m, off_xy1_mt, dxy1_m, dxy1_mt, xy2_m, xy1_m, xy1_mt, scale_err, xy_err, ori_err'.split(', '))
        >>> #---------------
        >>> ut.quit_if_noshow()
        >>> pt.figure(fnum=1, pnum=(1, 2, 1), title='orig points and offset point')
        >>> segments_list1 = np.array(list(zip(xy1_m.T.tolist(), off_xy1_m.T.tolist())))
        >>> pt.draw_line_segments(segments_list1, color=pt.LIGHT_BLUE)
        >>> pt.dark_background()
        >>> #---------------
        >>> pt.figure(fnum=1, pnum=(1, 2, 2), title='transformed points and matching points')
        >>> #---------------
        >>> # first have to make corresponding offset points
        >>> # Use reference point for scale and orientation tests
        >>> oris2_m   = ktool.get_oris(kpts2_m)
        >>> scales2_m = ktool.get_scales(kpts2_m)
        >>> dxy2_m    = np.vstack((np.sin(oris2_m), -np.cos(oris2_m)))
        >>> scaled_dxy2_m = dxy2_m * scales2_m[None, :]
        >>> off_xy2_m = xy2_m + scaled_dxy2_m
        >>> # Draw transformed semgents
        >>> segments_list2 = np.array(list(zip(xy2_m.T.tolist(), off_xy2_m.T.tolist())))
        >>> pt.draw_line_segments(segments_list2, color=pt.GREEN)
        >>> # Draw corresponding matches semgents
        >>> segments_list3 = np.array(list(zip(xy1_mt.T.tolist(), off_xy1_mt.T.tolist())))
        >>> pt.draw_line_segments(segments_list3, color=pt.RED)
        >>> # Draw matches between correspondences
        >>> segments_list4 = np.array(list(zip(xy1_mt.T.tolist(), xy2_m.T.tolist())))
        >>> pt.draw_line_segments(segments_list4, color=pt.ORANGE)
        >>> pt.dark_background()
        >>> #---------------
        >>> #vt.get _xy_axis_extents(kpts1_m)
        >>> #pt.draw_sv.show_sv(rchip1, rchip2, kpts1, kpts2, fm, homog_tup=homog_tup)
        >>> ut.show_if_requested()
    """
    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)
    # Transform all xy1 matches to xy2 space
    xy1_m   = ktool.get_xys(kpts1_m)
    #with ut.embed_on_exception_context:
    xy1_mt  = ltool.transform_points_with_homography(H, xy1_m)
    #xy1_mt  = ktool.transform_kpts_xys(H, kpts1_m)
    xy2_m   = ktool.get_xys(kpts2_m)
    # --- Find (Squared) Homography Distance Error ---
    # You cannot test for scale or orientation easily here because
    # you no longer have an ellipse? (maybe, probably have a conic) when using a
    # projective transformation
    xy_err = vtool.distance.L2_sqrd(xy1_mt.T, xy2_m.T)
    # Estimate final inliers
    #ut.embed()
    if full_homog_checks:
        # TODO: may need to use more than one reference point
        # Use reference point for scale and orientation tests
        oris1_m   = ktool.get_oris(kpts1_m)
        scales1_m = ktool.get_scales(kpts1_m)
        # Get point offsets with unit length
        dxy1_m    = np.vstack((np.sin(oris1_m), -np.cos(oris1_m)))
        scaled_dxy1_m = dxy1_m * scales1_m[None, :]
        off_xy1_m = xy1_m + scaled_dxy1_m
        # transform reference point
        off_xy1_mt = ltool.transform_points_with_homography(H, off_xy1_m)
        scaled_dxy1_mt = xy1_mt - off_xy1_mt
        scales1_mt = npl.norm(scaled_dxy1_mt, axis=0)
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        dxy1_mt = scaled_dxy1_mt / scales1_mt
        # adjust for gravity vector being 0
        oris1_mt = np.arctan2(dxy1_mt[1], dxy1_mt[0]) - ktool.GRAVITY_THETA
        _det1_mt = scales1_mt ** 2
        det2_m = ktool.get_sqrd_scales(kpts2_m)
        ori2_m = ktool.get_oris(kpts2_m)
        #xy_err    = vtool.distance.L2_sqrd(xy2_m.T, _xy1_mt.T)
        scale_err = vtool.distance.det_distance(_det1_mt, det2_m)
        ori_err   = vtool.distance.ori_distance(oris1_mt, ori2_m)
        ###
        xy_inliers_flag = np.less(xy_err, xy_thresh_sqrd)
        scale_inliers_flag = np.less(scale_err, scale_thresh)
        ori_inliers_flag   = np.less(ori_err, ori_thresh)
        hypo_inliers_flag = xy_inliers_flag  # Try to re-use memory
        np.logical_and(hypo_inliers_flag, ori_inliers_flag, out=hypo_inliers_flag)
        np.logical_and(hypo_inliers_flag, scale_inliers_flag, out=hypo_inliers_flag)
        # Seems slower due to memory
        #hypo_inliers_flag = np.logical_and.reduce(
        #    [xy_inliers_flag, ori_inliers_flag, scale_inliers_flag])
        # this is also slower
        #hypo_inliers_flag = np.logical_and.reduce((xy_inliers_flag,
        #ori_inliers_flag, scale_inliers_flag), out=xy_inliers_flag)
        refined_inliers = np.where(hypo_inliers_flag)[0].astype(INDEX_DTYPE)
        refined_errors = (xy_err, ori_err, scale_err)
    else:
        refined_inliers = np.where(xy_err < xy_thresh_sqrd)[0].astype(INDEX_DTYPE)
        refined_errors = (xy_err, None, None)
    homog_tup1 = (refined_inliers, refined_errors, H)
    return homog_tup1


def test_affine_errors(H, kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd,
                       ori_thresh):
    """
    used for refinement as opposed to initial estimation
    """
    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)
    invVR1s_m = ktool.get_invVR_mats3x3(kpts1_m)
    xy2_m  = ktool.get_xys(kpts2_m)
    det2_m = ktool.get_sqrd_scales(kpts2_m)
    ori2_m = ktool.get_oris(kpts2_m)
    refined_inliers, refined_errors = _test_hypothesis_inliers(
        H, invVR1s_m, xy2_m, det2_m, ori2_m, xy_thresh_sqrd, scale_thresh_sqrd,
        ori_thresh)
    refined_tup1 = (refined_inliers, refined_errors, H)
    return refined_tup1


def refine_inliers(kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd,
                   scale_thresh=2.0, ori_thresh=1.57, full_homog_checks=True,
                   refine_method='homog'):
    """
    Given a set of hypothesis inliers, computes a homography and refines inliers
    returned homography maps image1 space into image2 space

    CommandLine:
        python -m vtool.spatial_verification --test-refine_inliers
        python -m vtool.spatial_verification --test-refine_inliers:0
        python -m vtool.spatial_verification --test-refine_inliers:1 --show

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.demodata as demodata
        >>> import vtool.keypoint as ktool
        >>> kpts1, kpts2 = demodata.get_dummy_kpts_pair((100, 100))
        >>> fm = demodata.make_dummy_fm(len(kpts1)).astype(np.int32)
        >>> aff_inliers = np.arange(len(fm))
        >>> xy_thresh_sqrd = .01 * ktool.get_kpts_dlen_sqrd(kpts2)
        >>> homogtup = refine_inliers(kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd)
        >>> refined_inliers, refined_errors, H = homogtup
        >>> result = ub.repr2(homogtup, precision=2, nl=True, suppress_small=True, nobr=True)
        >>> print(result)
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (
            np.array([   4.36,    5.28,    3.29,   13.05,  114.46,   48.97,   17.66,
                        25.83,    3.82]),
            np.array([ 0.1 ,  0.02,  0.44,  0.36,  0.18,  0.25,  0.04,  0.33,  0.47]),
            np.array([ 1.13,  1.11,  1.68,  1.89,  1.13,  1.42,  1.08,  1.01,  1.43]),
        ),
        np.array([[  9.18e-01,  -4.82e-02,   7.21e+00],
                  [ -6.86e-03,   9.09e-01,   4.13e+00],
                  [ -1.21e-04,  -3.45e-04,   1.00e+00]]),

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.keypoint as ktool
        >>> import plottool as pt
        >>> kpts1, kpts2, fm, aff_inliers, rchip1, rchip2, xy_thresh_sqrd = testdata_matching_affine_inliers()
        >>> homog_tup1 = refine_inliers(kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd)
        >>> homog_tup = (homog_tup1[0], homog_tup1[2])
        >>> ut.quit_if_noshow()
        >>> pt.draw_sv.show_sv(rchip1, rchip2, kpts1, kpts2, fm, homog_tup=homog_tup)
        >>> ut.show_if_requested()

    """
    H = estimate_refined_transform(kpts1, kpts2, fm, aff_inliers,
                                   refine_method=refine_method)
    if refine_method.endswith('homog'):
        homog_tup1 = test_homog_errors(H, kpts1, kpts2, fm, xy_thresh_sqrd,
                                       scale_thresh, ori_thresh, full_homog_checks)
    #elif refine_method == 'cv2-homog':
    #    homog_tup1 = test_homog_errors(H, kpts1, kpts2, fm, xy_thresh_sqrd,
    #                                   scale_thresh, ori_thresh, full_homog_checks)
    elif refine_method == 'affine':
        homog_tup1 = test_affine_errors(H, kpts1, kpts2, fm, xy_thresh_sqrd,
                                        scale_thresh, ori_thresh)
    return homog_tup1


def get_best_affine_inliers_(kpts1, kpts2, fm, fs, xy_thresh_sqrd,
                             scale_thresh, ori_thresh):
    if HAVE_SVER_C_WRAPPER:
        aff_inliers, aff_errors, Aff = sver_c_wrapper.get_best_affine_inliers_cpp(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh)
    else:
        if ut.NOT_QUIET:
            print('WARNING: sver has not been compiled')
        aff_inliers, aff_errors, Aff = get_best_affine_inliers(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh)
    return aff_inliers, aff_errors, Aff


def spatially_verify_kpts(kpts1, kpts2, fm,
                          xy_thresh=.01,
                          scale_thresh=2.0,
                          ori_thresh=TAU / 4.0,
                          dlen_sqrd2=None,
                          min_nInliers=4,
                          match_weights=None,
                          returnAff=False,
                          full_homog_checks=True,
                          refine_method='homog',
                          max_nInliers=5000,
                          ):
    """
    Driver function
    Spatially validates feature matches

    FIXME: there is a non-determenism here

    Returned homography maps image1 space into image2 space.

    Args:
        kpts1 (ndarray[ndim=2]): all keypoints in image 1
        kpts2 (ndarray[ndim=2]): all keypoints in image 2
        fm (ndarray[ndim=2]): matching keypoint indexes [..., (kp1x, kp2x), ...]
        xy_thresh (float): spatial distance threshold under affine transform to
                           be considered a match
        scale_thresh (float):
        ori_thresh (float):
        dlen_sqrd2 (float): diagonal length squared of image/chip 2
        min_nInliers (int): default=4
        returnAff (bool): returns best affine hypothesis as well
        max_nInliers (int): homog is not considered after this threshold

    Returns:
        tuple : (refined_inliers, refined_errors, H, aff_inliers, aff_errors, Aff) if success else None

    CommandLine:
        python -m vtool.spatial_verification --test-spatially_verify_kpts --show
        python -m vtool.spatial_verification --test-spatially_verify_kpts --show --refine-method='affine'
        python -m vtool.spatial_verification --test-spatially_verify_kpts --dpath figures --show --save ~/latex/crall-candidacy-2015/figures/sver_kpts.jpg  # NOQA
        python -m vtool.spatial_verification --test-spatially_verify_kpts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *
        >>> import vtool.demodata as demodata
        >>> import vtool as vt
        >>> fname1 = ut.get_argval('--fname1', type_=str, default='easy1.png')
        >>> fname2 = ut.get_argval('--fname2', type_=str, default='easy2.png')
        >>> default_dict = vt.get_extract_features_default_params()
        >>> default_dict['ratio_thresh'] = .625
        >>> kwargs = ut.argparse_dict(default_dict)
        >>> (kpts1, kpts2, fm, fs, rchip1, rchip2) = demodata.testdata_ratio_matches(fname1, fname2, **kwargs)
        >>> xy_thresh = .01
        >>> dlen_sqrd2 = 447271.015
        >>> ori_thresh = 1.57
        >>> min_nInliers = 4
        >>> returnAff = True
        >>> scale_thresh = 2.0
        >>> match_weights = np.ones(len(fm), dtype=np.float64)
        >>> refine_method = ut.get_argval('--refine-method', default='homog')
        >>> svtup = spatially_verify_kpts(kpts1, kpts2, fm, xy_thresh,
        >>>                               scale_thresh, ori_thresh, dlen_sqrd2,
        >>>                               min_nInliers, match_weights, returnAff,
        >>>                               refine_method=refine_method)
        >>> assert svtup is not None and len(svtup) == 6, 'sver failed'
        >>> refined_inliers, refined_errors, H = svtup[0:3]
        >>> aff_inliers, aff_errors, Aff = svtup[3:6]
        >>> #print('aff_errors = %r' % (aff_errors,))
        >>> print('aff_inliers = %r' % (aff_inliers,))
        >>> print('refined_inliers = %r' % (refined_inliers,))
        >>> #print('refined_errors = %r' % (refined_errors,))
        >>> result = ut.list_type_profile(svtup, with_dtype=False)
        >>> #result = ub.repr2(svtup, precision=3)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> homog_tup = (refined_inliers, H)
        >>> aff_tup = (aff_inliers, Aff)
        >>> pt.draw_sv.show_sv(rchip1, rchip2, kpts1, kpts2, fm, aff_tup=aff_tup, homog_tup=homog_tup, refine_method=refine_method)
        >>> pt.show_if_requested()
        tuple(numpy.ndarray, tuple(numpy.ndarray*3), numpy.ndarray, numpy.ndarray, tuple(numpy.ndarray*3), numpy.ndarray)

    """
    if len(fm) == 0:
        if VERBOSE_SVER:
            print('[sver] Cannot verify with no matches')
        svtup = None
        return svtup
    # Cast keypoints to float64 to avoid numerical issues
    kpts1 = kpts1.astype(np.float64, casting='same_kind', copy=False)
    kpts2 = kpts2.astype(np.float64, casting='same_kind', copy=False)
    #kpts1 = kpts1.astype(np.float64)
    #kpts2 = kpts2.astype(np.float64)
    assert match_weights is not None, 'provide at least ones please for match_weights'
    fs = match_weights
    # Get diagonal length if not provided
    if dlen_sqrd2 is None:
        kpts2_m = kpts2.take(fm.T[1], axis=0)
        dlen_sqrd2 = ktool.get_kpts_dlen_sqrd(kpts2_m)
    # Determine the best hypothesis transformation and get its inliers
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    aff_inliers, aff_errors, Aff = get_best_affine_inliers_(
        kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh)
    #print(aff_inliers)

    # Return if there are not enough inliers to compute homography
    if len(aff_inliers) < min_nInliers:
        # Test user defined param
        if VERBOSE_SVER:
            print('[sver] Failed spatial verification len(aff_inliers) = %r' %
                  (len(aff_inliers),))
        svtup = None
        return svtup
    if ((refine_method.endswith('homog') and len(aff_inliers) < 7) or
         len(aff_inliers) < 4):
        # Test fundamental param
        # need to have 4 or more inliers to comopute an affine
        # and need at least 7 to compute a homography
        if VERBOSE_SVER:
            print('[sver] Failed spatial verification len(aff_inliers) = %r' %
                  (len(aff_inliers),))
        svtup = None
        return svtup

    if len(aff_inliers) >= max_nInliers:
        # If there are a very large number of affine inliers, then the affine
        # matrix is probably good enough.
        svtup = (aff_inliers, aff_errors, Aff, aff_inliers, aff_errors, Aff)
        return svtup

    # Refine inliers using a projective transformation (homography)
    try:
        refined_inliers, refined_errors, H = refine_inliers(
            kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd, scale_thresh,
            ori_thresh, full_homog_checks, refine_method=refine_method)
        #print(refined_inliers)
    except npl.LinAlgError as ex:
        if ut.VERYVERBOSE and ut.SUPER_STRICT:
            ut.printex(ex, 'numeric error in homog estimation.', iswarning=True)
        return None
    except ValueError as ex:
        if ut.VERYVERBOSE and ut.SUPER_STRICT:
            ut.printex(ex, 'error cv2 in homog estimation.', iswarning=True)
        return None
    except IndexError:
        raise
    except Exception as ex:
        # There is a weird error that starts with MemoryError and ends up
        # makeing len(h) = 6.
        ut.printex(ex, 'Unknown error in homog estimation.',
                      keys=['kpts1', 'kpts2',  'fm', 'fm.shape', 'kpts1.shape',
                            (len, 'aff_inliers'),
                            'kpts2.shape', 'xy_thresh', 'scale_thresh',
                            'dlen_sqrd2', 'min_nInliers'])
        if ut.SUPER_STRICT:
            print('SUPER_STRICT is on. Reraising')
            raise
        return None
    if VERBOSE_SVER:
        print('[sver] Succesfully finished spatial verification.')
    if returnAff:
        svtup = (refined_inliers, refined_errors, H, aff_inliers, aff_errors, Aff)
        return svtup
    else:
        svtup = (refined_inliers, refined_errors, H, None, None, None)
        return svtup


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.spatial_verification
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
