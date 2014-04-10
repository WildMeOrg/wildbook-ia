from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[sv2]', DEBUG=False)
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


def build_lstsqrs_Mx9(x1_mn, y1_mn, x2_mn, y2_mn):
    # Builds the M x 9 least squares matrix
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
def compute_homog(x1_mn, y1_mn, x2_mn, y2_mn):
    '''Generate 6 degrees of freedom homography transformation
    Computes homography from normalized (0 to 1) point correspondences
    from 2 --> 1 '''
    #printDBG('[sv2] compute_homog')
    # Solve for the nullspace of the Mx9 matrix (solves least squares)
    Mx9 = build_lstsqrs_Mx9(x1_mn, y1_mn, x2_mn, y2_mn)
    try:
        (U, S, V) = npl.svd(Mx9, full_matrices=False)
    except MemoryError as ex:
        print('[sv2] Caught MemErr %r during full SVD. Trying sparse SVD.' % (ex))
        Mx9Sparse = sps.lil_matrix(Mx9)
        (U, S, V) = spsl.svds(Mx9Sparse)
    except npl.LinAlgError as ex:
        print('[sv2] svd did not converge: %r' % ex)
        raise  # return np.eye(3)
    except Exception as ex:
        print('[sv2] svd error: %r' % ex)
        print('[sv2] Mx9.shape = %r' % (Mx9.shape,))
        raise
    # Rearange the nullspace into a homography
    h = V[-1]  # v = V.H
    H = np.vstack((h[0:3], h[3:6], h[6:9]))
    return H


def normalize_xy_points(x_m, y_m):
    'Returns a transformation to normalize points to mean=0, stddev=1'
    mu_x = x_m.mean()  # center of mass
    mu_y = y_m.mean()
    std_x = x_m.std()
    std_y = y_m.std()
    sx = 1.0 / std_x if std_x > 0 else 1  # average xy magnitude
    sy = 1.0 / std_y if std_x > 0 else 1
    T = np.array([(sx, 0, -mu_x * sx),
                  (0, sy, -mu_y * sy),
                  (0,  0,  1)])
    x_norm = (x_m - mu_x) * sx
    y_norm = (y_m - mu_y) * sy
    return x_norm, y_norm, T


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
    #printDBG('[sv2] affine_inliers')
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
                       max_scale,
                       min_scale,
                       dlen_sqrd2=None,
                       min_num_inliers=4,
                       just_affine=False):
    #printDBG('[sv2] homography_inliers')
    #if len(fm) < min_num_inliers:
        #return None
    # Not enough data
    # Estimate affine correspondence convert to SV_DTYPE
    # matching feature indexes
    fx1_m, fx2_m = fm[:, 0], fm[:, 1]
    kpts1_m = kpts1[fx1_m, :]
    kpts2_m = kpts2[fx2_m, :]
    # x, y, a, c, d : postion, shape
    x1_m, y1_m, invV1_m, oris1_m = ktool.cast_split(kpts1_m, SV_DTYPE)
    x2_m, y2_m, invV2_m, oris2_m = ktool.cast_split(kpts2_m, SV_DTYPE)
    # Get diagonal length
    dlen_sqrd2 = ktool.diag_extent_sqrd(kpts2_m) if dlen_sqrd2 is None else dlen_sqrd2
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    fx1_m = fm[:, 0]
    #fx2_m = fm[:, 1]
    Aff, aff_inliers = affine_inliers(x1_m, y1_m, invV1_m, fx1_m,
                                      x2_m, y2_m, invV2_m,
                                      xy_thresh_sqrd, max_scale, min_scale)
    # Cannot find good affine correspondence
    if just_affine:
        #raise Exception('No affine inliers')
        return Aff, aff_inliers
    if len(aff_inliers) < min_num_inliers:
        return None
    # Get corresponding points and shapes
    (x1_ma, y1_ma, invV1_m) = (
        x1_m[aff_inliers], y1_m[aff_inliers], invV1_m[:, aff_inliers])
    (x2_ma, y2_ma, invV2_m) = (
        x2_m[aff_inliers], y2_m[aff_inliers], invV2_m[:, aff_inliers])
    # Normalize affine inliers
    x1_mn, y1_mn, T1 = normalize_xy_points(x1_ma, y1_ma)
    x2_mn, y2_mn, T2 = normalize_xy_points(x2_ma, y2_ma)
    try:
        # Compute homgraphy transform from 1-->2 using affine inliers
        H_prime = compute_homog(x1_mn, y1_mn, x2_mn, y2_mn)
        # Computes ax = b # x = npl.solve(a, b)
        H = npl.solve(T2, H_prime).dot(T1)  # Unnormalize
    except npl.LinAlgError as ex:
        print('[sv2] Warning 285 %r' % ex)
        # raise
        return None

    ((H11, H12, H13),
     (H21, H22, H23),
     (H31, H32, H33)) = H
    # Transform all xy1 matches to xy2 space
    x1_mt = H11 * (x1_m) + H12 * (y1_m) + H13
    y1_mt = H21 * (x1_m) + H22 * (y1_m) + H23
    z1_mt = H31 * (x1_m) + H32 * (y1_m) + H33
    # --- Find (Squared) Homography Distance Error ---
    #scale_err = np.abs(npl.det(H)) * det2_m / det1_m
    z1_mt[z1_mt == 0] = 1E-14  # Avoid divide by zero
    xy_err = ((x1_mt / z1_mt) - x2_m) ** 2 + ((y1_mt / z1_mt) - y2_m) ** 2
    # Estimate final inliers
    inliers = np.where(xy_err < xy_thresh_sqrd)[0]
    return H, inliers
