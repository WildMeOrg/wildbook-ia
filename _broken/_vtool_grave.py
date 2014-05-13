#flake8:noqa
#PYX START
"""
// These are cython style comments for maintaining python compatibility
cimport numpy as np
ctypedef np.float64_t FLOAT64
"""
#PYX MAP FLOAT_2D np.ndarray[FLOAT64, ndim=2]
#PYX MAP FLOAT_1D np.ndarray[FLOAT64, ndim=1]
#PYX END

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
