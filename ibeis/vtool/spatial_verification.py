"""
python -c "import vtool, doctest; print(doctest.testmod(vtool.spatial_verification))"

Spatial verification of keypoint matches

Notation:
    1_m = img1_matches; 2_m = img2_matches
    x and y are locations, invV is the elliptical shapes.
    fx are the original feature indexes (used for making sure 1 keypoint isn't assigned to 2)

Look Into:
    Standard
    skimage.transform
    http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
    skimage.transform.fast_homography(im, H)
"""
from __future__ import absolute_import, division, print_function
from six.moves import range
import utool
# Science
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
# VTool
import vtool.keypoint as ktool
import vtool.linalg as ltool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[sver]', DEBUG=False)

np.tau = 2 * np.pi  # tauday.org

SV_DTYPE = np.float64

"""
<CYTH>
#ctypedef np.float64_t SV_DTYPE
SV_DTYPE = np.float64
</CYTH>
"""


@profile
def build_lstsqrs_Mx9(xy1_mn, xy2_mn):
    """ Builds the M x 9 least squares matrix

    >>> from vtool.spatial_verification import *  # NOQA
    >>> import vtool.tests.dummy as dummy
    >>> import vtool.keypoint as ktool
    >>> kpts1 = dummy.perterbed_grid_kpts(seed=12, damping=1.2, wh_stride=(30, 30))
    >>> kpts2 = dummy.perterbed_grid_kpts(seed=24, damping=1.6, wh_stride=(30, 30))
    >>> xy1_mn = ktool.get_xys(kpts1).astype(np.float64)
    >>> xy2_mn = ktool.get_xys(kpts2).astype(np.float64)
    >>> Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
    >>> print(utool.hashstr(Mx9))
    cbuegpb+fma6uk9l

    <CYTH returns="np.ndarray[np.float64_t, ndim=2]">
    cdef:
        np.ndarray[np.float64_t, ndim=2] xy1_mn
        np.ndarray[np.float64_t, ndim=2] xy2_mn
        np.ndarray[np.float64_t, ndim=2] Mx9
        np.float64_t u2, v2, d, e, f, g, h, i, j, k, l, p, q, r
        long num_pts
        long ix
    </CYTH>
    """
    x1_mn, y1_mn = xy1_mn
    x2_mn, y2_mn = xy2_mn
    num_pts = len(x1_mn)
    Mx9 = np.zeros((2 * num_pts, 9), dtype=SV_DTYPE)
    for ix in range(num_pts):  # Loop over inliers
        # Concatenate all 2x9 matrices into an Mx9 matrix
        u2        = x2_mn[ix]
        v2        = y2_mn[ix]
        (d, e, f) = (     -x1_mn[ix],      -y1_mn[ix],  -1)
        (g, h, i) = ( v2 * x1_mn[ix],  v2 * y1_mn[ix],  v2)
        (j, k, l) = (      x1_mn[ix],       y1_mn[ix],   1)
        (p, q, r) = (-u2 * x1_mn[ix], -u2 * y1_mn[ix], -u2)
        Mx9[ix * 2]     = (0, 0, 0, d, e, f, g, h, i)
        Mx9[ix * 2 + 1] = (j, k, l, 0, 0, 0, p, q, r)
    return Mx9


_build_lstsqrs_Mx9_cyth = build_lstsqrs_Mx9  # HACK HACK HACK


@profile
def compute_homog(xy1_mn, xy2_mn):
    """
    Generate 6 degrees of freedom homography transformation
    Computes homography from normalized (0 to 1) point correspondences
    from 2 --> 1

    >>> from vtool.spatial_verification import *  # NOQA
    >>> import vtool.tests.dummy as dummy
    >>> import vtool.keypoint as ktool
    >>> kpts1 = dummy.perterbed_grid_kpts(seed=12, damping=1.2, wh_stride=(30, 30))
    >>> kpts2 = dummy.perterbed_grid_kpts(seed=24, damping=1.6, wh_stride=(30, 30))
    >>> xy1_mn = ktool.get_xys(kpts1).astype(np.float64)
    >>> xy2_mn = ktool.get_xys(kpts2).astype(np.float64)
    >>> compute_homog(xy1_mn, xy2_mn)
    array([[  1.83339765e-03,   2.84967769e-03,  -7.11014174e-01],
           [  2.82477716e-03,   1.80000317e-03,  -7.03139797e-01],
           [  1.66839210e-05,   1.67525379e-05,  -5.52890916e-03]])

    <CYTH returns="np.ndarray[np.float64_t, ndim=2]">
    cdef:
        np.ndarray[np.float64_t, ndim=2] xy1_mn
        np.ndarray[np.float64_t, ndim=2] xy2_mn
        np.ndarray[np.float64_t, ndim=2] Mx9
    </CYTH>
    """
    # Solve for the nullspace of the Mx9 matrix (solves least squares)
    Mx9 = _build_lstsqrs_Mx9_cyth(xy1_mn, xy2_mn)
    try:
        (U, S, V) = npl.svd(Mx9, full_matrices=False)
    except MemoryError as ex:
        print('[sver] Caught MemErr %r during full SVD. Trying sparse SVD.' % (ex))
        Mx9Sparse = sps.lil_matrix(Mx9)
        (U, S, V) = spsl.svds(Mx9Sparse)
    except npl.LinAlgError as ex:
        print('[sver] svd did not converge: %r' % ex)
        raise
    except Exception as ex:
        print('[sver] svd error: %r' % ex)
        raise
    # Rearange the nullspace into a homography
    #h = V[-1]  # v = V.H
    h = V[8]  # Hack for cython.wraparound(False)
    H = np.vstack((h[0:3], h[3:6], h[6:9]))
    return H

# Total time: 6.39448 s
#
# Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
# 80                                           @profile
# 81                                           def _test_hypothosis_inliers(Aff, invVR1s_m, xy2_m, det2_m, ori2_m,
# 82                                                                        xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
# 83                                               # Map keypoints from image 1 onto image 2
# 84     18581      1380010     74.3      6.6      invVR1s_mt = ktool.matrix_multiply(Aff, invVR1s_m)
# 85                                               # Get projection components
# 86     18581       899671     48.4      4.3      _xy1_mt   = ktool.get_invVR_mats_xys(invVR1s_mt)
# 87     18581      7374378    396.9     35.2      _det1_mt  = npl.det(invVR1s_mt[:, 0:2, 0:2])  # ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
# 88     18581      3133362    168.6     15.0      _ori1_mt  = ktool.get_invVR_mats_oris(invVR1s_mt)
# 89                                               # Check for projection errors
# 90                                               #xy_err    = ltool.L2_sqrd(xy2_m.T, _xy1_mt.T)
# 91     18581      2434911    131.0     11.6      xy_err    = ltool.L2_sqrd(xy2_m.T, _xy1_mt.T)
# 92     18581      1999379    107.6      9.5      scale_err = ltool.det_distance(_det1_mt, det2_m)
# 93     18581      1786259     96.1      8.5      ori_err   = ltool.ori_distance(_ori1_mt, ori2_m)
# 94                                               # Mark keypoints which are inliers to this hypothosis
# 95     18581       231433     12.5      1.1      xy_inliers_flag    = xy_err    < xy_thresh_sqrd
# 96     18581       278045     15.0      1.3      scale_inliers_flag = scale_err < scale_thresh_sqrd
# 97     18581       217338     11.7      1.0      ori_inliers_flag   = ori_err   < ori_thresh
# 98                                               # TODO Add uniqueness of matches constraint
# 99     18581       840893     45.3      4.0      hypo_inliers_flag = ltool.and_lists(xy_inliers_flag, ori_inliers_flag, scale_inliers_flag)
#100     18581        55803      3.0      0.3      hypo_errors = (xy_err, ori_err, scale_err)
#101     18581       249742     13.4      1.2      hypo_inliers = np.where(hypo_inliers_flag)[0]
#102     18581        58484      3.1      0.3      return hypo_inliers, hypo_errors


@profile
def _test_hypothesis_inliers(Aff, invVR1s_m, xy2_m, det2_m, ori2_m,
                             xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    """
    <CYTH returns="tuple">
    cdef:
        np.ndarray[np.float64_t, ndim=2] Aff
        np.ndarray[np.float64_t, ndim=3] invVR1s_m
        np.ndarray[np.float64_t, ndim=2] xy2_m
        np.ndarray[np.float64_t, ndim=1] det2_m
        np.ndarray[np.float64_t, ndim=1] ori2_m
        np.float64_t xy_thresh_sqrd
        np.float64_t scale_thresh_sqrd
        np.float64_t ori_thresh
        np.ndarray[np.float64_t, ndim=3] invVR1s_mt
        np.ndarray[np.float64_t, ndim=2] _xy1_mt
        np.ndarray[np.float64_t, ndim=1] _det1_mt
        np.ndarray[np.float64_t, ndim=1] _ori1_mt
        np.ndarray[np.float64_t, ndim=1] xy_err
        np.ndarray[np.float64_t, ndim=1] scale_err
        np.ndarray[np.float64_t, ndim=1] scale_err
        np.ndarray[np.uint8_t, ndim=1] xy_inliers_flag
        np.ndarray[np.uint8_t, ndim=1] scale_inliers_flag
        np.ndarray[np.uint8_t, ndim=1] ori_inliers_flag
        tuple hypo_errors
        np.ndarray[np.int_t, ndim=1] hypo_inliers
    </CYTH>
    """
    # Map keypoints from image 1 onto image 2
    invVR1s_mt = ktool.matrix_multiply(Aff, invVR1s_m)
    # Get projection components
    _xy1_mt   = ktool.get_invVR_mats_xys_cyth(invVR1s_mt)
    #_det1_mt  = npl.det(invVR1s_mt[:, 0:2, 0:2])  # ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
    _det1_mt  = ktool.get_invVR_mats_sqrd_scale_cyth(invVR1s_mt)  # Seedup: 396.9/19.4 = 20x
    _ori1_mt  = ktool.get_invVR_mats_oris_cyth(invVR1s_mt)
    # Check for projection errors
    #xy_err    = ltool.L2_sqrd(xy2_m.T, _xy1_mt.T)
    xy_err    = ltool.L2_sqrd(xy2_m.T, _xy1_mt.T)  # Speedup: 131.0/36.4 = 3.5x
    #scale_err = ltool.det_distance(_det1_mt, det2_m)
    scale_err = ltool.det_distance(_det1_mt, det2_m)  # Speedup: 107.6/38 = 2.8
    ori_err   = ltool.ori_distance(_ori1_mt, ori2_m)
    # Mark keypoints which are inliers to this hypothosis
    xy_inliers_flag    = xy_err    < xy_thresh_sqrd
    scale_inliers_flag = scale_err < scale_thresh_sqrd
    ori_inliers_flag   = ori_err   < ori_thresh
    # TODO Add uniqueness of matches constraint
    hypo_inliers_flag = ltool.and_lists(xy_inliers_flag, ori_inliers_flag, scale_inliers_flag)
    hypo_errors = (xy_err, ori_err, scale_err)
    hypo_inliers = np.where(hypo_inliers_flag)[0]
    return hypo_inliers, hypo_errors


@profile
def get_affine_inliers(kpts1, kpts2, fm,
                        xy_thresh_sqrd,
                        scale_thresh_sqrd,
                        ori_thresh):
    """ Estimates inliers deterministically using elliptical shapes
    Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    We transform from chip1 -> chip2
    The determinants are squared keypoint scales

    >>> from vtool.spatial_verification import *  # NOQA
    >>> import vtool.tests.dummy as dummy
    >>> import vtool.keypoint as ktool
    >>> kpts1 = dummy.perterbed_grid_kpts(seed=12, damping=1.2, wh_stride=(30, 30)).astype(np.float64)
    >>> kpts2 = dummy.perterbed_grid_kpts(seed=24, damping=1.6, wh_stride=(30, 30)).astype(np.float64)
    >>> fm = dummy.make_dummy_fm(len(kpts1)).astype(np.int64)
    >>> xy_thresh_sqrd = ktool.KPTS_DTYPE(.009) ** 2
    >>> scale_thresh_sqrd = ktool.KPTS_DTYPE(2)
    >>> ori_thresh = ktool.KPTS_DTYPE(np.tau / 4)
    >>> output = get_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)
    >>> print(utool.hashstr(repr(output[0])))
    edb8bw55zdw!xal@


    FROM PERDOCH 2009:
        H = inv(Aj).dot(Rj.T).dot(Ri).dot(Ai)
        H = inv(Aj).dot(Ai)
        The input invVs = perdoch.invA's

    <CYTH returns="tuple">
    cdef:
        np.ndarray[np.float64_t, ndim=2] kpts1
        np.ndarray[np.float64_t, ndim=2] kpts2
        np.ndarray[np.int64_t, ndim=2] fm
        np.float64_t xy_thresh_sqrd
        np.float64_t scale_thresh_sqrd
        np.float64_t ori_thresh
        np.ndarray[np.float64_t, ndim=3] invVR1s_m
        np.ndarray[np.float64_t, ndim=3] V1s_m
        np.ndarray[np.float64_t, ndim=3] invVR2s_m
        np.ndarray[np.float64_t, ndim=3] Aff_mats
        np.ndarray[np.float64_t, ndim=2] xy2_m
        np.ndarray[np.float64_t, ndim=1] det2_m
        np.ndarray[np.float64_t, ndim=1] ori2_m
        list inliers_and_errors_list
        list errors_list
        list errors_list
    </CYTH>
    """
    kpts1_m = kpts1[fm.T[0]]
    kpts2_m = kpts2[fm.T[1]]

    # Get keypoints to project in matrix form
    invVR1s_m = ktool.get_invV_mats(kpts1_m, with_trans=True, with_ori=True)
    V1s_m = ktool.get_V_mats(kpts1_m, with_trans=True, with_ori=True)
    invVR2s_m = ktool.get_invV_mats(kpts2_m, with_trans=True, with_ori=True)
    # The transform from kp1 to kp2 is given as:
    Aff_mats = ktool.matrix_multiply(invVR2s_m, V1s_m)
    # Get components to test projects against
    xy2_m  = ktool.get_invVR_mats_xys(invVR2s_m)
    det2_m = ktool.get_sqrd_scales(kpts2_m)  # PYX FLOAT_1D
    ori2_m = ktool.get_invVR_mats_oris(invVR2s_m)

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
    inliers_list = [tup[0] for tup in inliers_and_errors_list]
    errors_list  = [tup[1] for tup in inliers_and_errors_list]
    return inliers_list, errors_list, Aff_mats


@profile
def get_best_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh,
                            ori_thresh):
    """ Tests each hypothesis and returns only the best transformation and inliers
    <CYTH returns="tuple">
    cdef:
        np.ndarray[np.float64_t, ndim=2] Aff
        list inliers_list
        list aff_inliers
        list Aff_mats
        list errors_list
    </CYTH>
    """
    # Test each affine hypothesis
    inliers_list, errors_list, Aff_mats = get_affine_inliers(kpts1, kpts2, fm,
                                                             xy_thresh_sqrd,
                                                             scale_thresh,
                                                             ori_thresh)
    aff_inliers, Aff = determine_best_inliers(inliers_list, errors_list, Aff_mats)
    return aff_inliers, Aff


@profile
def determine_best_inliers(inliers_list, errors_list, Aff_mats):
    """ Currently this function just uses the number of inliers as a metric
    <CYTH returns="tuple">
    cdef:
    </CYTH>
    """
    # Determine the best hypothesis using the number of inliers
    nInliers_list = np.array([len(inliers) for inliers in inliers_list])
    best_mxs = nInliers_list.argsort()[::-1]
    # Return inliers and transformation
    best_mx = best_mxs[0]
    aff_inliers = inliers_list[best_mx]
    Aff = Aff_mats[best_mx]
    return aff_inliers, Aff


@profile
def get_homography_inliers(kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd):
    """
    Given a set of hypothesis inliers, computes a homography and refines inliers
    <CYTH returns="tuple">
    cdef:
    </CYTH>
    """
    fm_affine = fm[aff_inliers]
    kpts1_m = kpts1[fm.T[0]]
    kpts2_m = kpts2[fm.T[1]]
    # Get corresponding points and shapes
    kpts1_ma = kpts1[fm_affine.T[0]]
    kpts2_ma = kpts2[fm_affine.T[1]]
    # Normalize affine inliers xy locations
    xy1_ma = ktool.get_xys(kpts1_ma)
    xy2_ma = ktool.get_xys(kpts2_ma)
    xy1_mn, T1 = ltool.whiten_xy_points(xy1_ma)
    xy2_mn, T2 = ltool.whiten_xy_points(xy2_ma)
    # Compute homgraphy transform from chip1 -> chip2 using affine inliers
    H_prime = compute_homog(xy1_mn, xy2_mn)
    # Then compute ax = b  [aka: x = npl.solve(a, b)]
    H = npl.solve(T2, H_prime).dot(T1)  # Unnormalize
    # Transform all xy1 matches to xy2 space
    xyz1_m  = ktool.get_homog_xyzs(kpts1_m)
    xyz1_mt = ltool.matrix_multiply(H, xyz1_m)
    xy1_mt  = ltool.homogonize(xyz1_mt)
    xy2_m   = ktool.get_xys(kpts2_m)

    # --- Find (Squared) Homography Distance Error ---
    # You cannot test for scale or orientation easilly here because
    # you no longer have an ellipse when using a projective transformation
    xy_err = ltool.L2_sqrd(xy1_mt.T, xy2_m.T)  # FIXME: cython version seems to crash here, why?
    # Estimate final inliers
    homog_inliers = np.where(xy_err < xy_thresh_sqrd)[0]
    return homog_inliers, H


@profile
def spatial_verification(kpts1, kpts2, fm,
                         xy_thresh,
                         scale_thresh,
                         ori_thresh,
                         dlen_sqrd2=None,
                         min_num_inliers=4):
    """
    Driver function
    Spatially validates feature matches

    <CYTH returns="tuple">
    cdef:
    </CYTH>
    """
    kpts1 = kpts1.astype(np.float64, casting='same_kind', copy=False)
    kpts2 = kpts2.astype(np.float64, casting='same_kind', copy=False)
    # Get diagonal length if not provided
    if dlen_sqrd2 is None:
        kpts2_m = kpts2[fm.T[1]]
        dlen_sqrd2 = ktool.get_diag_extent_sqrd(kpts2_m)
    # Determine the best hypothesis transformation and get its inliers
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    aff_inliers, Aff = get_best_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd,
                                               scale_thresh, ori_thresh)
    # Return if there are not enough inliers to compute homography
    if len(aff_inliers) < min_num_inliers:
        return None
    # Refine inliers using a projective transformation (homography)
    try:
        homog_inliers, H = get_homography_inliers(kpts1, kpts2, fm, aff_inliers,
                                                  xy_thresh_sqrd)
    except npl.LinAlgError as ex:
        print('[sver] Warning 285 %r' % ex)
        return None
    return homog_inliers, H, aff_inliers, Aff


import cyth
if cyth.DYNAMIC:
    exec(cyth.import_cyth_execstr(__name__))
else:
    # <AUTOGEN_CYTH>
    # Regen command: python -c "import vtool.spatial_verification" --cyth-write
    try:
        if not cyth.WITH_CYTH:
            raise ImportError('no cyth')
        import vtool._spatial_verification_cyth
        __test_hypothesis_inliers_cyth = vtool._spatial_verification_cyth.__test_hypothesis_inliers_cyth
        _build_lstsqrs_Mx9_cyth        = vtool._spatial_verification_cyth._build_lstsqrs_Mx9_cyth
        _compute_homog_cyth            = vtool._spatial_verification_cyth._compute_homog_cyth
        _determine_best_inliers_cyth   = vtool._spatial_verification_cyth._determine_best_inliers_cyth
        _get_affine_inliers_cyth       = vtool._spatial_verification_cyth._get_affine_inliers_cyth
        _get_best_affine_inliers_cyth  = vtool._spatial_verification_cyth._get_best_affine_inliers_cyth
        _get_homography_inliers_cyth   = vtool._spatial_verification_cyth._get_homography_inliers_cyth
        _spatial_verification_cyth     = vtool._spatial_verification_cyth._spatial_verification_cyth
        _test_hypothesis_inliers_cyth  = vtool._spatial_verification_cyth.__test_hypothesis_inliers_cyth
        build_lstsqrs_Mx9_cyth         = vtool._spatial_verification_cyth._build_lstsqrs_Mx9_cyth
        compute_homog_cyth             = vtool._spatial_verification_cyth._compute_homog_cyth
        determine_best_inliers_cyth    = vtool._spatial_verification_cyth._determine_best_inliers_cyth
        get_affine_inliers_cyth        = vtool._spatial_verification_cyth._get_affine_inliers_cyth
        get_best_affine_inliers_cyth   = vtool._spatial_verification_cyth._get_best_affine_inliers_cyth
        get_homography_inliers_cyth    = vtool._spatial_verification_cyth._get_homography_inliers_cyth
        spatial_verification_cyth      = vtool._spatial_verification_cyth._spatial_verification_cyth
        CYTHONIZED = True
    except ImportError:
        _test_hypothesis_inliers_cyth = _test_hypothesis_inliers
        build_lstsqrs_Mx9_cyth        = build_lstsqrs_Mx9
        compute_homog_cyth            = compute_homog
        determine_best_inliers_cyth   = determine_best_inliers
        get_affine_inliers_cyth       = get_affine_inliers
        get_best_affine_inliers_cyth  = get_best_affine_inliers
        get_homography_inliers_cyth   = get_homography_inliers
        spatial_verification_cyth     = spatial_verification
        CYTHONIZED = False
    # </AUTOGEN_CYTH>
