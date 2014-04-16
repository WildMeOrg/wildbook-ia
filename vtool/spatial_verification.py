from __future__ import absolute_import, division, print_function
import utool
# Science
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
# Standard
# skimage.transform
# http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
# skimage.transform.fast_homography(im, H)

# vtool
import vtool.keypoint as ktool
import vtool.linalg as ltool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[sver]', DEBUG=False)

np.tau = 2 * np.pi  # tauday.org

#PYX START
"""
// These are cython style comments for maintaining python compatibility
cimport numpy as np
ctypedef np.float64_t FLOAT64
"""
#PYX MAP FLOAT_2D np.ndarray[FLOAT64, ndim=2]
#PYX MAP FLOAT_1D np.ndarray[FLOAT64, ndim=1]
#PYX END
SV_DTYPE = np.float64


def build_lstsqrs_Mx9(xy1_mn, xy2_mn):
    # Builds the M x 9 least squares matrix
    x1_mn, y1_mn = xy1_mn
    x2_mn, y2_mn = xy2_mn
    num_pts = len(x1_mn)
    Mx9 = np.zeros((2 * num_pts, 9), dtype=SV_DTYPE)
    for ix in xrange(num_pts):  # Loop over inliers
        # Concatinate all 2x9 matrices into an Mx9 matrix
        u2        = x2_mn[ix]
        v2        = y2_mn[ix]
        (d, e, f) = (     -x1_mn[ix],      -y1_mn[ix],  -1)
        (g, h, i) = ( v2 * x1_mn[ix],  v2 * y1_mn[ix],  v2)
        (j, k, l) = (      x1_mn[ix],       y1_mn[ix],   1)
        (p, q, r) = (-u2 * x1_mn[ix], -u2 * y1_mn[ix], -u2)
        Mx9[ix * 2]     = (0, 0, 0, d, e, f, g, h, i)
        Mx9[ix * 2 + 1] = (j, k, l, 0, 0, 0, p, q, r)
    return Mx9


@profile
def compute_homog(xy1_mn, xy2_mn):
    '''Generate 6 degrees of freedom homography transformation
    Computes homography from normalized (0 to 1) point correspondences
    from 2 --> 1 '''
    #printDBG('[sver] compute_homog')
    # Solve for the nullspace of the Mx9 matrix (solves least squares)
    Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
    try:
        (U, S, V) = npl.svd(Mx9, full_matrices=False)
    except MemoryError as ex:
        print('[sver] Caught MemErr %r during full SVD. Trying sparse SVD.' % (ex))
        Mx9Sparse = sps.lil_matrix(Mx9)
        (U, S, V) = spsl.svds(Mx9Sparse)
    except npl.LinAlgError as ex:
        print('[sver] svd did not converge: %r' % ex)
        raise  # return np.eye(3)
    except Exception as ex:
        print('[sver] svd error: %r' % ex)
        print('[sver] Mx9.shape = %r' % (Mx9.shape,))
        raise
    # Rearange the nullspace into a homography
    h = V[-1]  # v = V.H
    H = np.vstack((h[0:3], h[3:6], h[6:9]))
    return H


#---
# --------------------------------
# TODO: This is one of the slowest functions we have right now
# This needs to be sped up
#import numba
#@numba.autojit
#PYX DEFINE
#def affine_inliers(FLOAT_2D x1_m, FLOAT_2D y1_m, FLOAT_2D invV1_m,  FLOAT_2D fx1_m,
#                   FLOAT_2D x2_m, FLOAT_2D y2_m, FLOAT_2D invV2_m,
#                   float xy_thresh_sqrd,
#                   float max_scale, float min_scale):
@profile
def affine_inliers2(kpts1, kpts2, fm,
                    xy_thresh_sqrd,
                    scale_thresh_sqrd,
                    ori_thresh):
    """ Estimates inliers deterministically using elliptical shapes
    1_m = img1_matches; 2_m = img2_matches
    x and y are locations, invV is the elliptical shapes.
    fx are the original feature indexes (used for making sure 1 keypoint isn't assigned to 2)

    FROM PERDOCH 2009:
        H = inv(Aj).dot(Rj.T).dot(Ri).dot(Ai)
        H = inv(Aj).dot(Ai)
        The input invVs = perdoch.invA's

    We transform from 1 - >2
    Get keypoint scales (determinant)
    Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    """
    kpts1_m = kpts1[fm.T[0]]
    kpts2_m = kpts2[fm.T[1]]

    # Get keypoints to project
    invVR1s_m = ktool.get_invV_mats(kpts1_m, with_trans=True, with_ori=True)
    V1s_m     = ktool.get_V_mats(kpts1_m, with_trans=True, with_ori=True)
    invVR2s_m = ktool.get_invV_mats(kpts2_m, with_trans=True, with_ori=True)
    # The transform from kp1 to kp2 is given as:
    # Aff = inv(invV2).dot(V1)
    Aff_mats = ktool.matrix_multiply(invVR2s_m, V1s_m)
    # Get components to test projects against
    det2_m = ktool.get_sqrd_scales(kpts2_m)  # PYX FLOAT_1D
    _xy2_m   = invVR2s_m[:, 0, 0:2]
    _ori2_m  = ktool.get_invVR_mats_oris(invVR2s_m)
    # Test all hypothesis
    def test_hypothosis_inliers(Aff):
        # Map keypoints from image 1 onto image 2
        invVR1s_mt = ktool.matrix_multiply(Aff, invVR1s_m)
        # Get projection components
        _xy1_mt   = ktool.get_invVR_mats_xys(invVR1s_mt)
        _ori1_mt  = ktool.get_invVR_mats_oris(invVR1s_mt)
        _det1_mt  = ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
        # Check for projection errors
        ori_err   = ltool.ori_distance(_ori1_mt, _ori2_m)
        xy_err    = ltool.L2_sqrd(_xy2_m, _xy1_mt)
        scale_err = ltool.det_distance(_det1_mt, det2_m)
        # Mark keypoints which are inliers to this hypothosis
        xy_inliers_flag = xy_err < xy_thresh_sqrd
        ori_inliers_flag = ori_err < ori_thresh
        scale_inliers_flag = scale_err < scale_thresh_sqrd
        hypo_inliers_flag = ltool.logical_and_many(xy_inliers_flag, ori_inliers_flag, scale_inliers_flag)
        hypo_inliers = np.where(hypo_inliers_flag)[0]
        # TODO Add uniqueness of matches constraint
        return hypo_inliers

    # Enumerate all hypothesis
    inliers_list = [test_hypothosis_inliers(Aff) for Aff in Aff_mats]
    # Determine best hypothesis
    nInliers_list = np.array(map(len, inliers_list))
    best_mxs = nInliers_list.argsort()[::-1]
    best_mx = best_mxs[0]
    # Return best hypothesis and inliers
    # TODO: In the future maybe average very good hypothesis?
    best_inliers = inliers_list[best_mx]
    best_Aff = Aff_mats[best_mx, :, :]
    return best_Aff, best_inliers


@profile
def affine_inliers(x1_m, y1_m, invV1_m, fx1_m,
                   x2_m, y2_m, invV2_m,
                   xy_thresh_sqrd,
                   max_scale, min_scale):
    '''Estimates inliers deterministically using elliptical shapes
    1_m = img1_matches; 2_m = img2_matches
    x and y are locations, invV is the elliptical shapes.
    fx are the original feature indexes (used for making sure 1 keypoint isn't assigned to 2)

    FROM PERDOCH 2009:
        H = inv(Aj).dot(Rj.T).dot(Ri).dot(Ai)
        H = inv(Aj).dot(Ai)
        The input invVs = perdoch.invA's

    We transform from 1 - >2
    '''
    #printDBG('[sver] affine_inliers')
    #print(repr((invV1_m.T[0:10]).T))
    #print(repr((invV2_m.T[0:10]).T))
    #with utool.Timer('enume all'):
    #fx1_uq, fx1_ui = np.unique(fx1_m, return_inverse=True)
    #fx2_uq, fx2_ui = np.unique(fx2_m, return_inverse=True)
    best_inliers = []
    num_best_inliers = 0
    best_mx  = None
    # Get keypoint scales (determinant)
    det1_m = ltool.det_ltri(invV1_m)  # PYX FLOAT_1D
    det2_m = ltool.det_ltri(invV2_m)  # PYX FLOAT_1D
    # Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    #V2_m = inv_ltri(invV2_m, det2_m)
    V1_m = ltool.inv_ltri(invV1_m, det1_m)
    # The transform from kp1 to kp2 is given as:
    # Aff = inv(invV2).dot(invV1)
    Aff_list = ltool.dot_ltri(invV2_m, V1_m)
    # Compute scale change of all transformations
    detAff_list = ltool.det_ltri(Aff_list)
    # Test all hypothesis
    for mx in xrange(len(x1_m)):
        # --- Get the mth hypothesis ---
        A11, A21, A22 = Aff_list[:, mx]
        Adet = detAff_list[mx]
        x1_hypo, y1_hypo = x1_m[mx], y1_m[mx]
        x2_hypo, y2_hypo = x2_m[mx], y2_m[mx]
        # --- Transform from xy1 to xy2 ---
        x1_mt = x2_hypo + A11 * (x1_m - x1_hypo)
        y1_mt = y2_hypo + A21 * (x1_m - x1_hypo) + A22 * (y1_m - y1_hypo)
        # --- Find (Squared) Distance Error ---
        xy_err = (x1_mt - x2_m) ** 2 + (y1_mt - y2_m) ** 2
        # --- Find (Squared) Scale Error ---
        #scale_err = Adet * det2_m / det1_m
        scale_err = Adet * det1_m / det2_m
        # --- Determine Inliers ---
        xy_inliers_flag = xy_err < xy_thresh_sqrd
        scale_inliers_flag = np.logical_and(scale_err > min_scale,
                                            scale_err < max_scale)
        hypo_inliers_flag = np.logical_and(xy_inliers_flag, scale_inliers_flag)
        #---
        #---------------------------------
        # TODO: More sophisticated scoring
        # Currently I'm using the number of inliers as a transformations'
        # goodness. Also the way I'm accoutning for multiple assignment
        # does not take into account any error reporting
        #---------------------------------
        '''
        unique_assigned1 = flag_unique(fx1_ui[hypo_inliers_flag])
        unique_assigned2 = flag_unique(fx2_ui[hypo_inliers_flag])
        unique_assigned_flag = np.logical_and(unique_assigned1,
                                              unique_assigned2)
        hypo_inliers = np.where(hypo_inliers_flag)[0][unique_assigned_flag]
        '''
        hypo_inliers = np.where(hypo_inliers_flag)[0]

        #---
        # Try to not double count inlier matches that are counted twice
        # probably need something a little bit more robust here.
        unique_hypo_inliers = np.unique(fx1_m[hypo_inliers])
        num_hypo_inliers = len(unique_hypo_inliers)
        # --- Update Best Inliers ---
        if num_hypo_inliers > num_best_inliers:
            best_mx = mx
            best_inliers = hypo_inliers
            num_best_inliers = num_hypo_inliers
    if not best_mx is None:
        (A11, A21, A22) = Aff_list[:, best_mx]
        (x1, y1) = (x1_m[best_mx], y1_m[best_mx])
        (x2, y2) = (x2_m[best_mx], y2_m[best_mx])
        xt = x2 - A11 * x1
        yt = y2 - A21 * x1 - A22 * y1
        # Save the winning hypothesis transformation
        best_Aff = np.array([(A11,   0,  xt),
                             (A21, A22,  yt),
                             (  0,   0,   1)])
    else:
        best_Aff = np.eye(3)
    return best_Aff, best_inliers


@profile
def homography_inliers(kpts1, kpts2, fm,
                       xy_thresh,
                       #max_scale,
                       #min_scale,
                       scale_thresh,
                       ori_thresh,
                       dlen_sqrd2=None,
                       min_num_inliers=4,
                       just_affine=False):
    fx1_m, fx2_m = fm[:, 0], fm[:, 1]
    kpts1_m = kpts1[fx1_m, :]
    kpts2_m = kpts2[fx2_m, :]
    # Get diagonal length
    dlen_sqrd2 = ktool.get_diag_extent_sqrd(kpts2_m) if dlen_sqrd2 is None else dlen_sqrd2
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    fx1_m = fm[:, 0]
    Aff, aff_inliers = affine_inliers2(kpts1_m, kpts2_m, fm,
                                       xy_thresh_sqrd, scale_thresh, ori_thresh)
    # Cannot find good affine correspondence
    if just_affine:
        #raise Exception('No affine inliers')
        return Aff, aff_inliers
    if len(aff_inliers) < min_num_inliers:
        return None
    # Get corresponding points and shapes
    kpts1_ma = kpts1_m[aff_inliers]
    kpts2_ma = kpts2_m[aff_inliers]
    xy1_ma = ktool.get_xys(kpts1_ma)
    xy2_ma = ktool.get_xys(kpts2_ma)
    # Normalize affine inliers
    xy1_mn, T1 = ltool.whiten_xy_points(xy1_ma)
    xy2_mn, T2 = ltool.whiten_xy_points(xy2_ma)
    # Compute homgraphy transform from 1-->2 using affine inliers
    # Then compute ax = b  # x = npl.solve(a, b)
    try:
        H_prime = compute_homog(xy1_mn, xy2_mn)
        H = npl.solve(T2, H_prime).dot(T1)  # Unnormalize
    except npl.LinAlgError as ex:
        print('[sver] Warning 285 %r' % ex)
        return None

    # Transform all xy1 matches to xy2 space
    xyz1_m = ktool.get_homog_xys(kpts1_m)
    xyz1_mt = ltool.matrix_multiply(H, xyz1_m)
    xy1_mt = ltool.homogonize(xyz1_mt)
    xy1_m  = xyz1_m[0:2]

    # --- Find (Squared) Homography Distance Error ---
    #scale_err = np.abs(npl.det(H)) * det2_m / det1_m
    xy_err = ltool.L2_sqrd(xy1_mt, xy1_m)
    # Estimate final inliers
    inliers = np.where(xy_err < xy_thresh_sqrd)[0]
    return H, inliers, Aff, aff_inliers
