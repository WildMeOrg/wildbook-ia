from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import numpy as np
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[constr]', DEBUG=False)


def nearest_neighbors_to_matches(fx2_to_fx1, fx2_to_dist):
    """
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
        >>> matchtup = nearest_neighbors_to_matches(fx2_to_fx1, fx2_to_dist)
        >>> fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = matchtup
        >>> result = ut.list_str(matchtup, precision=3)
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
    return fx2_match, fx1_match, fx1_norm, match_dist, norm_dist


def ratio_test(fx2_match, fx1_match, fx1_norm, match_dist, norm_dist, ratio_thresh=.625):
    r"""
    Lowes ratio test for one-vs-one feature matches.
    Assumes reverse matches (image2 to image1) and returns
    (image1 to image2) matches.

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
        >>> fx2_match  = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        >>> fx1_match  = np.array([77, 116, 122, 1075, 530, 45], dtype=np.int32)
        >>> fx1_norm   = np.array([971, 120, 128, 692, 45, 530], dtype=np.int32)
        >>> match_dist = np.array([ 0.059, 0.021, 0.039, 0.15 , 0.227, 0.216])
        >>> norm_dist  = np.array([ 0.239, 0.241, 0.248, 0.151, 0.244, 0.236])
        >>> matchtup = nearest_neighbors_to_matches(fx2_to_fx1, fx2_to_dist)
        >>> fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = matchtup
        >>> ratio_thresh = .625
        >>> # execute function
        >>> ratio_tup = ratio_test(fx2_match, fx1_match, fx1_norm, match_dist, norm_dist, ratio_thresh)
        >>> (fm_RAT, fs_RAT, fm_norm_RAT) = ratio_tup
        >>> # verify results
        >>> result = ut.list_str(rattup, precision=3)
        >>> print(result)
        (
            np.array([[ 77,   0],
                      [116,   1],
                      [122,   2]]),
            np.array([ 0.753,  0.912,  0.842]),
            np.array([[971,   0],
                      [120,   1],
                      [128,   2]]),
        )
    """
    fx2_to_ratio = np.divide(match_dist, norm_dist)
    fx2_to_isvalid = np.less(fx2_to_ratio, ratio_thresh)
    fx2_match_RAT = fx2_match.compress(fx2_to_isvalid)
    fx1_match_RAT = fx1_match.compress(fx2_to_isvalid)
    fx1_norm_RAT = fx1_norm.compress(fx2_to_isvalid)
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
