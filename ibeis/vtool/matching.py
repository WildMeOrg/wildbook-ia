from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import numpy as np
from vtool import keypoint as ktool
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[matching]', DEBUG=False)


# maximum SIFT matching distance based using uint8 trick from hesaff
PSEUDO_MAX_VEC_COMPONENT = 512
PSEUDO_MAX_DIST_SQRD = 2 * (PSEUDO_MAX_VEC_COMPONENT ** 2)
PSEUDO_MAX_DIST = np.sqrt(2) * (PSEUDO_MAX_VEC_COMPONENT)


def normalized_nearest_neighbors(flann, vecs2, K, checks=800):
    """
    uses flann index to return nearest neighbors with distances normalized
    between 0 and 1 using sifts uint8 trick
    """
    fx2_to_fx1, _fx2_to_dist = flann.nn_index(vecs2, num_neighbors=K, checks=checks)
    fx2_to_dist = np.divide(np.sqrt(_fx2_to_dist.astype(np.float64)), PSEUDO_MAX_DIST)  # normalized dist
    #fx2_to_dist = np.divide(_fx2_to_dist.astype(np.float64), PSEUDO_MAX_DIST_SQRD)  # squared normalized dist
    return fx2_to_fx1, fx2_to_dist


def assign_spatially_constrained_matches(chip2_dlen_sqrd, kpts1, kpts2, H,
                                         fx2_to_fx1, fx2_to_dist, match_xy_thresh,
                                         norm_xy_bounds=(0.0, 1.0)):
    """
    Args:
        chip2_dlen_sqrd (dict):
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix that maps image1 space into image2 space
        fx2_to_fx1 (ndarray): image2s nearest feature indices in image1
        fx2_to_dist (ndarray):
        match_xy_thresh (float):
        norm_xy_bounds (tuple):

    Returns:
        tuple: assigntup(
            fx2_match, - matching feature indicies in image 2
            fx1_match, - matching feature indicies in image 1
            fx1_norm,  - normmalizing indicies in image 1
            match_dist, - descriptor distances between fx2_match and fx1_match
            norm_dist, - descriptor distances between fx2_match and fx1_norm
            )

    CommandLine:
        python -m vtool.matching --test-assign_spatially_constrained_matches

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> kpts1 = np.array([[  6.,   4.,   15.84,    4.66,    7.24,    0.  ],
        ...                   [  9.,   3.,   20.09,    5.76,    6.2 ,    0.  ],
        ...                   [  1.,   1.,   12.96,    1.73,    8.77,    0.  ],])
        >>> kpts2 = np.array([[  2.,   1.,   12.11,    0.38,    8.04,    0.  ],
        ...                   [  5.,   1.,   22.4 ,    1.31,    5.04,    0.  ],
        ...                   [  6.,   1.,   19.25,    1.74,    4.72,    0.  ],])
        >>> match_xy_thresh = .37
        >>> chip2_dlen_sqrd = 1400
        >>> norm_xy_bounds = (0.0, 1.0)
        >>> H = np.array([[ 2,  0, 0],
        >>>               [ 0,  1, 0],
        >>>               [ 0,  0, 1]])
        >>> fx2_to_fx1 = np.array([[2, 1, 0],
        >>>                        [0, 1, 2],
        >>>                        [2, 0, 1]], dtype=np.int32)
        >>> fx2_to_dist = np.array([[.40, .80, .85],
        >>>                         [.30, .50, .60],
        >>>                         [.80, .90, .91]], dtype=np.float32)
        >>> # verify results
        >>> assigntup = assign_spatially_constrained_matches(chip2_dlen_sqrd, kpts1, kpts2, H, fx2_to_fx1, fx2_to_dist, match_xy_thresh, norm_xy_bounds)
        >>> fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
        >>> result = ut.list_str(assigntup, precision=3)
        >>> print(result)
        (
            np.array([0, 1, 2]),
            np.array([2, 0, 2], dtype=np.int32),
            np.array([1, 1, 1], dtype=np.int32),
            np.array([ 0.4,  0.3,  0.8], dtype=np.float32),
            np.array([ 0.8,  0.5,  0.9], dtype=np.float32),
        )

    Example:

    assigns spatially constrained vsone match using results of nearest
    neighbors.
    """
    # Find spatial errors of keypoints under current homography (kpts1 mapped into image2 space)
    fx2_to_xyerr_sqrd = ktool.get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
    fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
    fx2_to_xyerr_norm = np.divide(fx2_to_xyerr, np.sqrt(chip2_dlen_sqrd))

    # Find matches and normalizers that satisfy spatial constraints
    fx2_to_valid_match      = ut.inbounds(fx2_to_xyerr_norm, 0.0, match_xy_thresh, eq=True)
    fx2_to_valid_normalizer = ut.inbounds(fx2_to_xyerr_norm, *norm_xy_bounds, eq=True)
    fx2_to_fx1_match_col = ut.find_first_true_indicies(fx2_to_valid_match)
    fx2_to_fx1_norm_col  = ut.find_next_true_indicies(fx2_to_valid_normalizer, fx2_to_fx1_match_col)

    assert fx2_to_fx1_match_col != fx2_to_fx1_norm_col, 'normlizers are matches!'

    fx2_to_hasmatch = [pos is not None for pos in fx2_to_fx1_norm_col]
    # IMAGE 2 Matching Features
    fx2_match = np.where(fx2_to_hasmatch)[0]
    match_col_list = np.array(ut.list_take(fx2_to_fx1_match_col, fx2_match), dtype=fx2_match.dtype)
    norm_col_list = np.array(ut.list_take(fx2_to_fx1_norm_col, fx2_match), dtype=fx2_match.dtype)

    # We now have 2d coordinates into fx2_to_fx1
    # Covnert into 1d coordinates for flat indexing into fx2_to_fx1
    _match_index_2d = np.vstack((fx2_match, match_col_list))
    _norm_index_2d  = np.vstack((fx2_match, norm_col_list))
    _shape2d        = fx2_to_fx1.shape
    match_index_1d  = np.ravel_multi_index(_match_index_2d, _shape2d)
    norm_index_1d   = np.ravel_multi_index(_norm_index_2d, _shape2d)

    # Find initial matches
    # IMAGE 1 Matching Features
    fx1_match = fx2_to_fx1.take(match_index_1d)
    fx1_norm  = fx2_to_fx1.take(norm_index_1d)
    # compute constrained ratio score
    match_dist = fx2_to_dist.take(match_index_1d)
    norm_dist  = fx2_to_dist.take(norm_index_1d)

    # package and return
    assigntup = fx2_match, fx1_match, fx1_norm, match_dist, norm_dist
    return assigntup


def assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist):
    """
    assigns vsone matches using results of nearest neighbors.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> # build test data
        >>> fx2_to_fx1 = np.array([[ 77,   971],
        >>>                        [116,   120],
        >>>                        [122,   128],
        >>>                        [1075,  692],
        >>>                        [ 530,   45],
        >>>                        [  45,  530]], dtype=np.int32)
        >>> fx2_to_dist = np.array([[ 0.05907059,  0.2389698 ],
        >>>                         [ 0.02129555,  0.24083519],
        >>>                         [ 0.03901863,  0.24756241],
        >>>                         [ 0.14974403,  0.15112305],
        >>>                         [ 0.22693443,  0.24428177],
        >>>                         [ 0.2155838 ,  0.23641014]], dtype=np.float64)
        >>> assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist)
        >>> fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
        >>> result = ut.list_str(assigntup, precision=3)
        >>> print(result)
        (
            np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            np.array([  77,  116,  122, 1075,  530,   45], dtype=np.int32),
            np.array([971, 120, 128, 692,  45, 530], dtype=np.int32),
            np.array([ 0.059,  0.021,  0.039,  0.15 ,  0.227,  0.216]),
            np.array([ 0.239,  0.241,  0.248,  0.151,  0.244,  0.236]),
        )
    """
    fx2_match = np.arange(len(fx2_to_fx1), dtype=fx2_to_fx1.dtype)
    fx1_match = fx2_to_fx1.T[0]
    fx1_norm  = fx2_to_fx1.T[1]
    match_dist = fx2_to_dist.T[0]
    norm_dist  = fx2_to_dist.T[1]
    assigntup = fx2_match, fx1_match, fx1_norm, match_dist, norm_dist
    return assigntup


def unconstrained_ratio_match(flann, vecs2, unc_ratio_thresh=.625, fm_dtype=np.int32, fs_dtype=np.float32):
    """ Lowes ratio matching

    fs_dtype = kwargs.get('fs_dtype', np.float32)
    fm_dtype = kwargs.get('fm_dtype', np.int32)

    """
    fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(flann, vecs2, K=2, checks=800)
    #ut.embed()
    assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist)
    fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
    ratio_tup = ratio_test(fx2_match, fx1_match, fx1_norm, match_dist,
                           norm_dist, unc_ratio_thresh, fm_dtype=fm_dtype,
                           fs_dtype=fs_dtype)
    return ratio_tup


@profile
def spatially_constrained_ratio_match(flann, vecs2, kpts1, kpts2, H, chip2_dlen_sqrd,
                                      match_xy_thresh=1.0, scr_ratio_thresh=.625, scr_K=7,
                                      norm_xy_bounds=(0.0, 1.0),
                                      fm_dtype=np.int32, fs_dtype=np.float32):
    """
    performs nearest neighbors, then assigns based on spatial constraints, the
    last step performs a ratio test.

    H - a homography H that maps image1 space into image2 space
    H should map from query to database chip (1 to 2)
    """
    assert H.shape == (3, 3)
    # Find several of image2's features nearest matches in image1
    fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(flann, vecs2, scr_K, checks=800)
    # Then find those which satisfify the constraints
    assigntup = assign_spatially_constrained_matches(
        chip2_dlen_sqrd, kpts1, kpts2, H, fx2_to_fx1, fx2_to_dist,
        match_xy_thresh, norm_xy_bounds=norm_xy_bounds)
    fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
    # filter assignments via the ratio test
    scr_tup = ratio_test(fx2_match, fx1_match, fx1_norm, match_dist,
                         norm_dist, scr_ratio_thresh, fm_dtype=fm_dtype,
                         fs_dtype=fs_dtype)
    return scr_tup


def ratio_test(fx2_match, fx1_match, fx1_norm, match_dist, norm_dist,
               ratio_thresh=.625, fm_dtype=np.int32, fs_dtype=np.float32):
    r"""
    Lowes ratio test for one-vs-one feature matches.

    Assumes reverse matches (image2 to image1) and returns (image1 to image2)
    matches. Generalized to accept any match or normalizer not just K=1 and K=2.

    Args:
        fx2_to_fx1 (ndarray): nearest neighbor indicies (from flann)
        fx2_to_dist (ndarray): nearest neighbor distances (from flann)
        ratio_thresh (float):
        match_col (int or ndarray): column of matching indices
        norm_col (int or ndarray): column of normalizng indicies

    Returns:
        tuple: (fm_RAT, fs_RAT, fm_norm_RAT)

    CommandLine:
        python -m vtool.matching --test-ratio_test

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> # build test data
        >>> fx2_match  = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        >>> fx1_match  = np.array([77, 116, 122, 1075, 530, 45], dtype=np.int32)
        >>> fx1_norm   = np.array([971, 120, 128, 692, 45, 530], dtype=np.int32)
        >>> match_dist = np.array([ 0.059, 0.021, 0.039, 0.15 , 0.227, 0.216])
        >>> norm_dist  = np.array([ 0.239, 0.241, 0.248, 0.151, 0.244, 0.236])
        >>> ratio_thresh = .625
        >>> # execute function
        >>> ratio_tup = ratio_test(fx2_match, fx1_match, fx1_norm, match_dist, norm_dist, ratio_thresh)
        >>> (fm_RAT, fs_RAT, fm_norm_RAT) = ratio_tup
        >>> # verify results
        >>> result = ut.list_str(ratio_tup, precision=3)
        >>> print(result)
        (
            np.array([[ 77,   0],
                      [116,   1],
                      [122,   2]], dtype=np.int32),
            np.array([ 0.753,  0.913,  0.843], dtype=np.float32),
            np.array([[971,   0],
                      [120,   1],
                      [128,   2]], dtype=np.int32),
        )
    """
    fx2_to_ratio = np.divide(match_dist, norm_dist).astype(fs_dtype)
    fx2_to_isvalid = np.less(fx2_to_ratio, ratio_thresh)
    fx2_match_RAT = fx2_match.compress(fx2_to_isvalid).astype(fm_dtype)
    fx1_match_RAT = fx1_match.compress(fx2_to_isvalid).astype(fm_dtype)
    fx1_norm_RAT = fx1_norm.compress(fx2_to_isvalid).astype(fm_dtype)
    # Turn the ratio into a score
    fs_RAT = np.subtract(1.0, fx2_to_ratio.compress(fx2_to_isvalid))
    fm_RAT = np.vstack((fx1_match_RAT, fx2_match_RAT)).T
    # return normalizer info as well
    fm_norm_RAT = np.vstack((fx1_norm_RAT, fx2_match_RAT)).T
    ratio_tup = fm_RAT, fs_RAT, fm_norm_RAT
    return ratio_tup


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.matching
        python -m vtool.matching --allexamples
        python -m vtool.matching --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
