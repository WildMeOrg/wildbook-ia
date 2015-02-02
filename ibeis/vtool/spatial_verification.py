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
"""
from __future__ import absolute_import, division, print_function
from six.moves import range
import utool as ut
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from numpy.core.umath_tests import matrix_multiply
import vtool.keypoint as ktool
import vtool.linalg as ltool

try:
    from vtool import sver_c_wrapper
except Exception as ex:
    ut.printex(ex, 'please build the sver c wrapper (run with --rebuild-sver')
    raise

profile = ut.profile
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[sver]', DEBUG=False)

"""
#if CYTH
#ctypedef np.float64_t SV_DTYPE
SV_DTYPE = np.float64
cdef np.float64_t TAU
#endif
"""

SV_DTYPE = np.float64
TAU = 2 * np.pi  # tauday.org


@profile
def build_lstsqrs_Mx9(xy1_mn, xy2_mn):
    """ Builds the M x 9 least squares matrix

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.tests.dummy as dummy
        >>> kpts1, kpts2 = dummy.get_dummy_kpts_pair()
        >>> xy1_mn = ktool.get_xys(kpts1).astype(np.float64)
        >>> xy2_mn = ktool.get_xys(kpts2).astype(np.float64)
        >>> Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
        >>> result = ut.hashstr(Mx9)
        >>> print(result)
        f@2l62+2!ppow8yw

    Cyth:
        #CYTH_RETURNS np.ndarray[np.float64_t, ndim=2]
        #if CYTH
        #CYTH_PARAM_TYPES:
            np.ndarray[np.float64_t, ndim=2] xy1_mn
            np.ndarray[np.float64_t, ndim=2] xy2_mn
            np.ndarray[np.float64_t, ndim=1] x1_mn, y1_mn,  x2_mn, y2_mn
            np.ndarray[np.float64_t, ndim=2] Mx9
        cdef:
            Py_ssize_t num_pts
            Py_ssize_t ix
        #endif
    """
    x1_mn = xy1_mn[0]
    y1_mn = xy1_mn[1]
    x2_mn = xy2_mn[0]
    y2_mn = xy2_mn[1]
    num_pts = x1_mn.shape[0]
    #Mx9 = np.zeros((2 * num_pts, 9), dtype=SV_DTYPE)
    Mx9 = np.empty((2 * num_pts, 9), dtype=SV_DTYPE)
    """
    #if CYTH
    for ix in range(num_pts):  # Loop over inliers
        # Concatenate all 2x9 matrices into an Mx9 matrix
        Mx9[ix * 2, 3]  = -x1_mn[ix]
        Mx9[ix * 2, 4]  = -y1_mn[ix]
        Mx9[ix * 2, 5]  = -1.0
        Mx9[ix * 2, 6]  = y2_mn[ix] * x1_mn[ix]
        Mx9[ix * 2, 7]  = y2_mn[ix] * y1_mn[ix]
        Mx9[ix * 2, 8]  = y2_mn[ix]

        Mx9[ix * 2 + 1, 0] = x1_mn[ix]
        Mx9[ix * 2 + 1, 1] = y1_mn[ix]
        Mx9[ix * 2 + 1, 2] = -1.0
        Mx9[ix * 2 + 1, 6] = -x2_mn[ix] * x1_mn[ix]
        Mx9[ix * 2 + 1, 7] = -x2_mn[ix] * y1_mn[ix]
        Mx9[ix * 2 + 1, 8] = -x2_mn[ix]
    #else
    """
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
    "#endif"
    return Mx9


@profile
def compute_homog(xy1_mn, xy2_mn):
    """
    Generate 6 degrees of freedom homography transformation
    Computes homography from normalized (0 to 1) point correspondences
    from 2 --> 1

    Args:
        xy1_mn (ndarray[ndim=2]): xy points in image1
        xy2_mn (ndarray[ndim=2]): corresponding xy points in image 2

    Returns:
        ndarray[shape=(3,3)]: H - homography matrix

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.tests.dummy as dummy
        >>> import vtool.keypoint as ktool
        >>> kpts1, kpts2 = dummy.get_dummy_kpts_pair()
        >>> xy1_mn = ktool.get_xys(kpts1)
        >>> xy2_mn = ktool.get_xys(kpts2)
        >>> H = compute_homog(xy1_mn, xy2_mn)
        >>> #result = ut.hashstr(H)
        >>> result =str(H)
        >>> result = np.array_str(H, precision=2)
        >>> print(result)
        [[  1.83e-03   2.85e-03  -7.11e-01]
         [  2.82e-03   1.80e-03  -7.03e-01]
         [  1.67e-05   1.68e-05  -5.53e-03]]

    Cyth:
        #CYTH_RETURNS np.ndarray[np.float64_t, ndim=2]
        #CYTH_PARAM_TYPES:
            np.ndarray[np.float64_t, ndim=2] xy1_mn
            np.ndarray[np.float64_t, ndim=2] xy2_mn
        #if CYTH
        cdef:
            np.ndarray[np.float64_t, ndim=2] Mx9
        #endif
    """
    # Solve for the nullspace of the Mx9 matrix (solves least squares)
    #Mx9 = build_lstsqrs_Mx9_cyth(xy1_mn, xy2_mn)  # NOQA  # TODO: re-enable
    Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
    try:
        (U, S, V) = npl.svd(Mx9, full_matrices=True, compute_uv=True)
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
    h = V[8]  # Hack for Cython.wraparound(False)
    # FIXME: THERE IS A BUG HERE, sometimes len(V) = 6. why???
    H = np.vstack((h[0:3], h[3:6], h[6:9]))
    return H


@profile
def _test_hypothesis_inliers(Aff, invVR1s_m, xy2_m, det2_m, ori2_m,
                             xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh):
    """
    Critical section code. Inner loop of _test_hypothesis_inliers

    CommandLine:
        python -m vtool.spatial_verification --test-_test_hypothesis_inliers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> from vtool.spatial_verification import _test_hypothesis_inliers  # NOQA
        >>> import vtool.tests.dummy as dummy
        >>> import vtool.keypoint as ktool
        >>> _kw1 = dict(seed=12, damping=1.2, wh_stride=(30, 30))
        >>> _kw2 = dict(seed=24, damping=1.6, wh_stride=(30, 30))
        >>> kpts1 = dummy.perterbed_grid_kpts(**_kw1).astype(np.float64)
        >>> kpts2 = dummy.perterbed_grid_kpts(**_kw2).astype(np.float64)
        >>> fm = dummy.make_dummy_fm(len(kpts1)).astype(np.int32)
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
        >>> result = ut.hashstr(output)
        >>> print(result)
        +%q&%je52nlyli5&

    Cyth:
        #if CYTH
        #CYTH_PARAM_TYPES:
            np.ndarray[np.float64_t, ndim=2] Aff
            np.ndarray[np.float64_t, ndim=3] invVR1s_m
            np.ndarray[np.float64_t, ndim=2] xy2_m
            np.ndarray[np.float64_t, ndim=1] det2_m
            np.ndarray[np.float64_t, ndim=1] ori2_m
            np.float64_t xy_thresh_sqrd
            np.float64_t scale_thresh_sqrd
            np.float64_t ori_thresh
        cdef:
            np.ndarray[np.float64_t, ndim=3] invVR1s_mt
            np.ndarray[np.float64_t, ndim=2] _xy1_mt
            np.ndarray[np.float64_t, ndim=1] _det1_mt
            np.ndarray[np.float64_t, ndim=1] _ori1_mt
            np.ndarray[np.float64_t, ndim=1] xy_err
            np.ndarray[np.float64_t, ndim=1] scale_err
            np.ndarray[np.uint8_t, ndim=1, cast=True] xy_inliers_flag
            np.ndarray[np.uint8_t, ndim=1, cast=True] scale_inliers_flag
            np.ndarray[np.uint8_t, ndim=1, cast=True] ori_inliers_flag
            np.ndarray[np.uint8_t, ndim=1, cast=True] hypo_inliers_flag
            tuple hypo_errors
            np.ndarray[np.int_t, ndim=1] hypo_inliers
        #endif

    Timeit:
        %timeit xy_err < xy_thresh_sqrd
        %timeit np.less(xy_err, xy_thresh_sqrd)
    """
    # Map keypoints from image 1 onto image 2
    invVR1s_mt = matrix_multiply(Aff, invVR1s_m)

    # Get projection components
    #_xy1_mt   = ktool.get_invVR_mats_xys(invVR1s_mt)
    #_det1_mt  = ktool.get_invVR_mats_sqrd_scale_cyth(invVR1s_mt)  # Seedup: 396.9/19.4 = 20x
    #_ori1_mt  = ktool.get_invVR_mats_oris(invVR1s_mt)
    # Check for projection errors
    #xy_err    = ltool.L2_sqrd_cyth(xy2_m.T, _xy1_mt.T)  # Speedup: 131.0/36.4 = 3.5x
    #scale_err = ltool.det_distance_cyth(_det1_mt, det2_m)  # Speedup: 107.6/38 = 2.8
    #ori_err   = ltool.ori_distance_cyth(_ori1_mt, ori2_m)

    # Get projection components
    _xy1_mt   = ktool.get_invVR_mats_xys(invVR1s_mt)
    _det1_mt  = ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
    _ori1_mt  = ktool.get_invVR_mats_oris(invVR1s_mt)
    ## Check for projection errors
    xy_err    = ltool.L2_sqrd(xy2_m.T, _xy1_mt.T)
    scale_err = ltool.det_distance(_det1_mt, det2_m)
    ori_err   = ltool.ori_distance(_ori1_mt, ori2_m)

    # Mark keypoints which are inliers to this hypothosis
    xy_inliers_flag    = np.less(xy_err, xy_thresh_sqrd)
    scale_inliers_flag = np.less(scale_err, scale_thresh_sqrd)
    np.logical_and(xy_inliers_flag, scale_inliers_flag)
    ori_inliers_flag   = np.less(ori_err, ori_thresh)
    # TODO Add uniqueness of matches constraint
    #hypo_inliers_flag = np.empty(xy_inliers_flag.size, dtype=np.bool)
    hypo_inliers_flag = xy_inliers_flag  # Try to re-use memory
    np.logical_and(hypo_inliers_flag, ori_inliers_flag, out=hypo_inliers_flag)
    np.logical_and(hypo_inliers_flag, scale_inliers_flag, out=hypo_inliers_flag)
    #hypo_inliers_flag = ltool.and_3lists(xy_inliers_flag, ori_inliers_flag, scale_inliers_flag)
    hypo_errors = (xy_err, ori_err, scale_err)
    hypo_inliers = np.where(hypo_inliers_flag)[0]
    return hypo_inliers, hypo_errors


@profile
def get_affine_inliers(kpts1, kpts2, fm,
                        xy_thresh_sqrd,
                        scale_thresh_sqrd,
                        ori_thresh):
    """
    Estimates inliers deterministically using elliptical shapes

    Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    We transform from chip1 -> chip2
    The determinants are squared keypoint scales

    FROM PERDOCH 2009::
        H = inv(Aj).dot(Rj.T).dot(Ri).dot(Ai)
        H = inv(Aj).dot(Ai)
        The input invVs = perdoch.invA's

    CommandLine:
        python -m vtool.spatial_verification --test-get_affine_inliers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.tests.dummy as dummy
        >>> import vtool.keypoint as ktool
        >>> kpts1, kpts2 = dummy.get_dummy_kpts_pair((100, 100))
        >>> fm = dummy.make_dummy_fm(len(kpts1)).astype(np.int32)
        >>> xy_thresh_sqrd = ktool.KPTS_DTYPE(.009) ** 2
        >>> scale_thresh_sqrd = ktool.KPTS_DTYPE(2)
        >>> ori_thresh = ktool.KPTS_DTYPE(TAU / 4)
        >>> output = get_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh)
        >>> result = ut.hashstr(output)
        >>> print(result)
        89kz8nh6p+66t!+u


    Cyth:
        #if CYTH
        #CYTH_PARAM_TYPES:
            np.ndarray[np.float64_t, ndim=2] kpts1
            np.ndarray[np.float64_t, ndim=2] kpts2
            np.ndarray[np.int32_t, ndim=2] fm
            np.float64_t xy_thresh_sqrd
            np.float64_t scale_thresh_sqrd
            np.float64_t ori_thresh
        cdef:
            np.ndarray[np.float64_t, ndim=2] kpts1_m
            np.ndarray[np.float64_t, ndim=2] kpts2_m
            np.ndarray[np.float64_t, ndim=3] invVR1s_m
            np.ndarray[np.float64_t, ndim=3] V1s_m
            np.ndarray[np.float64_t, ndim=3] invVR2s_m
            np.ndarray[np.float64_t, ndim=3] Aff_mats
            np.ndarray[np.float64_t, ndim=2] xy2_m
            np.ndarray[np.float64_t, ndim=1] det2_m
            np.ndarray[np.float64_t, ndim=1] ori2_m
            np.ndarray[np.float64_t, ndim=2] Aff
            tuple tup
            list inliers_and_errors_list
            list errors_list
        #endif


    Ignore::
        from vtool.spatial_verification import *  # NOQA
        import vtool.tests.dummy as dummy
        import vtool.keypoint as ktool
        kpts1, kpts2 = dummy.get_dummy_kpts_pair((100, 100))
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


@profile
def get_best_affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh,
                            ori_thresh):
    """ Tests each hypothesis and returns only the best transformation and inliers

    #if CYTH
    #CYTH_PARAM_TYPES:
        np.ndarray[np.float64_t, ndim=2] kpts1
        np.ndarray[np.float64_t, ndim=2] kpts2
        np.ndarray[np.int32_t, ndim=2] fm
        np.float64_t xy_thresh_sqrd
        np.float64_t scale_thresh_sqrd
        np.float64_t ori_thresh
    cdef:
        list aff_inliers_list
        list aff_inliers
        list Aff_mats
        list aff_errors_list
    #endif
    """
    # Test each affine hypothesis
    # get list if inliers, errors, the affine matrix for each hypothesis
    USE_CPP_WRAPPER = True
    if USE_CPP_WRAPPER:
        aff_inliers_list, aff_errors_list, Aff_mats = sver_c_wrapper.get_affine_inliers_cpp(
            kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh, ori_thresh)
    else:
        aff_inliers_list, aff_errors_list, Aff_mats = get_affine_inliers(kpts1, kpts2, fm,
                                                                         xy_thresh_sqrd,
                                                                         scale_thresh,
                                                                         ori_thresh)

    # Determine the best hypothesis using the number of inliers
    # TODO: other measures in the error lists could be used as well
    nInliers_list = np.array([len(inliers) for inliers in aff_inliers_list])
    sortx = nInliers_list.argsort()[::-1]  # sort by non-inliers
    best_index = sortx[0]  # chose best
    aff_inliers = aff_inliers_list[best_index]
    aff_errors = aff_errors_list[best_index]
    Aff = Aff_mats[best_index]
    return aff_inliers, aff_errors, Aff


@profile
def get_homography_inliers(kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd):
    """
    Given a set of hypothesis inliers, computes a homography and refines inliers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *  # NOQA
        >>> import vtool.tests.dummy as dummy
        >>> import vtool.keypoint as ktool
        >>> kpts1, kpts2 = dummy.get_dummy_kpts_pair((100, 100))
        >>> fm = dummy.make_dummy_fm(len(kpts1)).astype(np.int32)

    Timeit::
        %timeit kpts1.take(fm.T[0].astype(np.int32), axis=0)
        %timeit kpts1[fm.T[0]]

        %timeit kpts1[fm.T[0]]
        %timeit kpts2[fm.T[1]]
        4.23 us per loop

        %timeit kpts1[fm.take(0, axis=1)]
        %timeit kpts2[fm.take(1, axis=1)]
        5.32 us per loop

        INDEX_TYPE = np.int32
        %timeit kpts1.take(fm.take(0, axis=1), axis=0)
        %timeit kpts2.take(fm.take(1, axis=1), axis=0)
        2.77 us per loop

        %timeit kpts1.take(fm.T[0], axis=0)
        %timeit kpts2.take(fm.T[1], axis=0)
        1.48 us per loop
    """
    fm_affine = fm[aff_inliers]

    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)

    # Get corresponding points and shapes
    kpts1_ma = kpts1[fm_affine.T[0]]
    kpts2_ma = kpts2[fm_affine.T[1]]
    #kpts1_ma = kpts1.take(fm_affine.T[0], axis=0)
    #kpts2_ma = kpts2.take(fm_affine.T[1], axis=0)
    # Normalize affine inliers xy locations
    xy1_ma = ktool.get_xys(kpts1_ma)
    xy2_ma = ktool.get_xys(kpts2_ma)
    xy1_man, T1 = ltool.whiten_xy_points(xy1_ma)
    xy2_man, T2 = ltool.whiten_xy_points(xy2_ma)
    # Compute homgraphy transform from chip1 -> chip2 using affine inliers
    H_prime = compute_homog(xy1_man, xy2_man)

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
    xy_err = ltool.L2_sqrd(xy1_mt.T, xy2_m.T)
    homog_errors = (xy_err, None, None)
    # Estimate final inliers
    homog_inliers = np.where(xy_err < xy_thresh_sqrd)[0]
    return homog_inliers, homog_errors, H


@profile
def spatially_verify_kpts(kpts1, kpts2, fm,
                          xy_thresh=.01,
                          scale_thresh=2.0,
                          ori_thresh=TAU / 4.0,
                          dlen_sqrd2=None,
                          min_nInliers=4,
                          returnAff=False):
    """
    Driver function
    Spatially validates feature matches

    Args:
        kpts1 (ndarray[ndim=2]): all keypoints in image 1
        kpts2 (ndarray[ndim=2]): all keypoints in image 2
        fm (ndarray[ndim=2]): matching keypoint indexes [..., (kp1x, kp2x), ...]
        xy_thresh (float): spatial distance threshold under affine transform to be considered a match
        scale_thresh (float):
        ori_thresh (float):
        dlen_sqrd2 (float): diagonal length squared of image/chip 2
        min_nInliers (int): default=4
        returnAff (bool): returns best affine hypothesis as well

    Returns:
        tuple : (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff) if success else None

    CommandLine:
        python -m vtool.spatial_verification --test-spatially_verify_kpts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.spatial_verification import *
        >>> import vtool.tests.dummy as dummy
        >>> (kpts1, kpts2, fm, fs, rchip1, rchip2) = dummy.testdata_ratio_matches()
        >>> xy_thresh = .01
        >>> dlen_sqrd2 = 447271.015
        >>> min_nInliers = 4
        >>> returnAff = True
        >>> ori_thresh = 1.57
        >>> scale_thresh = 2.0
        >>> svtup = spatially_verify_kpts(kpts1, kpts2, fm, xy_thresh, scale_thresh, ori_thresh, dlen_sqrd2, min_nInliers, returnAff)
        >>> assert svtup is not None and len(svtup) == 6, 'sver failed'
        >>> homog_inliers, homog_errors, H = svtup[0:3]
        >>> aff_inliers, aff_errors, Aff = svtup[3:6]
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     homog_tup = (homog_inliers, H)
        >>>     aff_tup = (aff_inliers, Aff)
        >>>     pt.draw_sv.show_sv(rchip1, rchip2, kpts1, kpts2, fm, aff_tup=aff_tup, homog_tup=homog_tup)
        >>>     pt.show_if_requested()
    """
    if ut.VERYVERBOSE:
        print('[sver] Starting spatial verification')
    if len(fm) == 0:
        if ut.VERYVERBOSE:
            print('[sver] Cannot verify with no matches')
        svtup = None
        return svtup
    # Cast keypoints to float64 to avoid numerical issues
    kpts1 = kpts1.astype(np.float64, casting='same_kind', copy=False)
    kpts2 = kpts2.astype(np.float64, casting='same_kind', copy=False)
    # Get diagonal length if not provided
    if dlen_sqrd2 is None:
        kpts2_m = kpts2.take(fm.T[1], axis=0)
        dlen_sqrd2 = ktool.get_kpts_dlen_sqrd(kpts2_m)
    # Determine the best hypothesis transformation and get its inliers
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    aff_inliers, aff_errors, Aff = get_best_affine_inliers(
        kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh, ori_thresh)
    # Return if there are not enough inliers to compute homography
    if len(aff_inliers) < min_nInliers:
        if ut.VERYVERBOSE:
            print('[sver] Failed spatial verification len(aff_inliers) = %r' %
                  (len(aff_inliers),))
        svtup = None
        return svtup
    # Refine inliers using a projective transformation (homography)
    try:
        homog_inliers, homog_errors, H = get_homography_inliers(
            kpts1, kpts2, fm, aff_inliers, xy_thresh_sqrd)
    except npl.LinAlgError as ex:
        ut.printex(ex, 'numeric error in homog estimation.', iswarning=True)
        return None
    except Exception as ex:
        # There is a weird error that starts with MemoryError and ends up
        # makeing len(h) = 6.
        ut.printex(ex, 'Unknown error in homog estimation.',
                      keys=['kpts1', 'kpts2',  'fm', 'xy_thresh',
                            'scale_thresh', 'dlen_sqrd2', 'min_nInliers'])
        if ut.SUPER_STRICT:
            print('SUPER_STRICT is on. Reraising')
            raise
        return None
    if ut.VERYVERBOSE:
        print('[sver] Finished spatial verification.')
    if returnAff:
        svtup = (homog_inliers, homog_errors, H, aff_inliers, aff_errors, Aff)
        return svtup
    else:
        svtup = (homog_inliers, homog_errors, H, None, None, None)
        return svtup

#try:
#    import cyth
#    if cyth.DYNAMIC:
#        exec(cyth.import_cyth_execstr(__name__))
#    else:
#        # <AUTOGEN_CYTH>
#        # Regen command: python -c "import vtool.linalg" --cyth-write
#        # </AUTOGEN_CYTH>
#        pass
#except Exception as ex:
#    pass

if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.spatial_verification
        python -m vtool.spatial_verification --allexamples

    SeeAlso:
        python -m vtool.tests.test_spatial_verification
        utprof.py -m vtool.tests.test_spatial_verification
    """
    import utool as ut  # NOQA
    ut.doctest_funcs()
