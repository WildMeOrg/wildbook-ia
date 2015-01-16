from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import numpy as np
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import spatial_verification as sver
#import numpy.linalg as npl
#import scipy.sparse as sps
#import scipy.sparse.linalg as spsl
#from numpy.core.umath_tests import matrix_multiply
#import vtool.keypoint as ktool
#import vtool.linalg as ltool
profile = ut.profile
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[constr]', DEBUG=False)

"""
Write paramater interactions

show true match and false match

"""

TAU = np.pi * 2


def show_example():
    r"""
    CommandLine:
        python -m vtool.constrained_matching --test-show_example --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.constrained_matching import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> # execute function
        >>> result = show_example()
        >>> # verify results
        >>> print(result)
        >>> pt.present()
        >>> pt.show_if_requested()
    """
    #ut.util_grabdata.get_valid_test_imgkeys()
    testtup1 = testdata_matcher('easy1.png', 'easy2.png')
    testtup2 = testdata_matcher('easy1.png', 'hard3.png')
    self1 = SimpleMatcher()
    self1.run_matching(testtup1)

    self2 = SimpleMatcher()
    self2.run_matching(testtup2)

    #self1.visualize_matches()
    #self2.visualize_matches()

    self1.visualize_normalizers()
    self2.visualize_normalizers()


def testdata_matcher(fname1='easy1.png', fname2='easy2.png'):
    """"
    CommandLine:
        python -m vtool.constrained_matching --test-testdata_matcher

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.constrained_matching import *  # NOQA
        >>> # build test data
        >>> fname1 = 'easy1.png'
        >>> fname2 = 'hard3.png'
        >>> # execute function
        >>> testtup = testdata_matcher(fname1, fname2)
        >>> # verify results
        >>> result = str(testtup)
        >>> print(result)
    """
    import utool as ut
    from vtool import image as gtool
    from vtool import features as feattool
    fpath1 = ut.grab_test_imgpath(fname1)
    fpath2 = ut.grab_test_imgpath(fname2)
    kpts1, vecs1 = feattool.extract_features(fpath1)
    kpts2, vecs2 = feattool.extract_features(fpath2)
    rchip1 = gtool.imread(fpath1)
    rchip2 = gtool.imread(fpath2)
    #chip1_shape = vt.gtool.open_image_size(fpath1)
    chip2_shape = gtool.open_image_size(fpath2)
    dlen_sqrd2 = chip2_shape[0] ** 2 + chip2_shape[1]
    testtup = (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2)
    return testtup


class SimpleMatcher(object):
    def __init__(self):
        self.fm = None
        self.fs = None
        self.H = None
        self.fs_V = None
        self.fm_V = None
        self.basetup = None
        self.testtup = None

    def visualize_matches(self):
        rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2 = self.testtup
        fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H      = self.basetup
        fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR              = self.nexttup
        fm_normalizer,                                         = self.base_meta
        fm_normalizer_constrained,                             = self.next_meta

        locals_ = ut.delete_dict_keys(locals(), ['title'])

        nRows = 3
        nCols = 2
        show_matches_ = self.start_new_viz(locals_, nRows, nCols)

        show_matches_(fm_ORIG, fs_ORIG, title='initial neighbors')

        show_matches_(fm_RAT, fs_RAT, title='ratio filtered')
        show_matches_(fm_normalizer, fs_RAT, title='ratio normalizers')

        #next_pnum()

        show_matches_(fm_SCR, fs_SCR, title='spatially constrained')
        show_matches_(fm_normalizer, fs_RAT, title='ratio normalizers')

        show_matches_(fm_SCRSV, fs_SCRSV, title='spatially constrained + SV')

    def visualize_normalizers(self):
        rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2 = self.testtup
        fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H      = self.basetup
        fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR              = self.nexttup
        fm_normalizer,                                         = self.base_meta
        fm_normalizer_constrained,                             = self.next_meta

        locals_ = ut.delete_dict_keys(locals(), ['title'])

        nRows = 2
        nCols = 2
        show_matches_ = self.start_new_viz(locals_, nRows, nCols)

        show_matches_(fm_RAT, fs_RAT, title='ratio filtered')
        show_matches_(fm_normalizer, fs_RAT, title='ratio normalizers')

        show_matches_(fm_SCR, fs_SCR, title='spatially constrained')
        show_matches_(fm_normalizer, fs_RAT, title='ratio normalizers')

    def start_new_viz(locals_, nRows, nCols):
        import plottool as pt
        next_pnum = pt.make_pnum_nextgen(nRows=nRows, nCols=nCols)
        fnum = pt.next_fnum()
        pt.figure(fnum=fnum, doclf=True, docla=True)

        def show_matches_(*args, **kwargs):
            showkw = locals_.copy()
            showkw['pnum'] = next_pnum()
            showkw['fnum'] = fnum
            showkw.update(kwargs)
            show_matches(*args, **showkw)
        return show_matches_

    def run_matching(self, testtup):
        basetup, base_meta = baseline_vsone_ratio_matcher(testtup)
        nexttup, next_meta = spatially_constrianed_matcher(testtup, basetup)
        self.nexttup = nexttup
        self.basetup = basetup
        self.testtup = testtup
        self.base_meta = base_meta
        self.next_meta = next_meta

    def setstate_testdata(self):
        testtup = testdata_matcher()
        self.run_matching(testtup)


def baseline_vsone_ratio_matcher(testtup):
    r"""
    Args:
        vecs1 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        vecs2 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints

    Ignore:
        %pylab qt4
        import plottool as pt
        pt.imshow(rchip1)
        pt.draw_kpts2(kpts1)

        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
    """
    rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2 = testtup
    #import vtool as vt
    sver_xy_thresh = .01
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
    fm_RAT, fs_RAT, fm_RAT_normalizer = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
    # SPATIAL VERIFICATION FILTER
    fm_SV, fs_SV, H = sver_inliers(kpts1, kpts2, fm_RAT, fs_RAT, dlen_sqrd2, sver_xy_thresh)
    base_tup = (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H)
    base_meta = (fm_RAT_normalizer,)
    return base_tup, base_meta


def spatially_constrianed_matcher(testtup, basetup):
    r"""
    spatially constrained ratio matching

    CommandLine:
        python -m vtool.constrained_matching --test-spatially_constrianed_matcher

    Example:
        >>> # DISABLE_DOCTEST
        >>> import plottool as pt
        >>> from vtool.constrained_matching import *  # NOQA
        >>> import vtool as vt
        >>> testtup = testdata_matcher()
        >>> basetup, base_meta = baseline_vsone_ratio_matcher(testtup)
        >>> self = SimpleMatcher()
        >>> # execute function
        >>> nexttup, next_meta = spatially_constrianed_matcher(testtup, basetup)
        >>> # verify results
        >>> print(nexttup)
    """
    #import vtool as vt
    (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testtup
    (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H) = basetup

    match_xy_thresh = .1
    sver_xy_thresh = .01
    ratio_thresh2 = .8
    # Observation, scores don't change above K=7
    # on easy test case
    search_K = 7  # 3

    # ASSIGN CANDIDATES
    # Get candidate nearest neighbors
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=search_K)

    # COMPUTE CONSTRAINTS
    normalizer_mode = 'nearby'
    #normalizer_mode = 'far'
    constrain_tup = constrain_matches(dlen_sqrd2, kpts1, kpts2, H, fx2_to_fx1,
                                      fx2_to_dist, match_xy_thresh,
                                      normalizer_mode=normalizer_mode)
    (fm_constrained, fm_normalizer_constrained, match_dist_list, norm_dist_list) = constrain_tup

    fm_SCR, fs_SCR = ratio_test2(match_dist_list, norm_dist_list, fm_constrained, ratio_thresh2)

    # Another round of verification
    fm_SVSCR, fs_SVSCR, H_SCR = sver_inliers(kpts1, kpts2, fm_SCR, fs_SCR, dlen_sqrd2, sver_xy_thresh)

    nexttup = (fm_SCR, fs_SCR, fm_SVSCR, fs_SVSCR, H_SCR)
    next_meta = (fm_normalizer_constrained,)
    return nexttup, next_meta


def get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1):
    """ transforms img2 to img2 and finds squared spatial error

    Args:
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        fx2_to_fx1 (ndarray): has shape (nMatch, K)

    Returns:
        ndarray: fx2_to_xyerr_sqrd has shape (nMatch, K)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.constrained_matching import *  # NOQA
        >>> # build test data
        >>> kpts1 = np.array([[ 129.83,   46.97,   15.84,    4.66,    7.24,    0.  ],
        ...                  [ 137.88,   49.87,   20.09,    5.76,    6.2 ,    0.  ],
        ...                  [ 115.95,   53.13,   12.96,    1.73,    8.77,    0.  ],
        ...                  [ 324.88,  172.58,  127.69,   41.29,   50.5 ,    0.  ],
        ...                  [ 285.44,  254.61,  136.06,   -4.77,   76.69,    0.  ],
        ...                  [ 367.72,  140.81,  172.13,   12.99,   96.15,    0.  ]], dtype=np.float32)
        >>> kpts2 = np.array([[ 318.93,   11.98,   12.11,    0.38,    8.04,    0.  ],
        ...                   [ 509.47,   12.53,   22.4 ,    1.31,    5.04,    0.  ],
        ...                   [ 514.03,   13.04,   19.25,    1.74,    4.72,    0.  ],
        ...                   [ 490.19,  185.49,   95.67,   -4.84,   88.23,    0.  ],
        ...                   [ 316.97,  206.07,   90.87,    0.07,   80.45,    0.  ],
        ...                   [ 366.07,  140.05,  161.27,  -47.01,   85.62,    0.  ]], dtype=np.float32)
        >>> H = np.array([[ -0.70098,   0.12273,   5.18734],
        >>>               [  0.12444,  -0.63474,  14.13995],
        >>>               [  0.00004,   0.00025,  -0.64873]])
        >>> fx2_to_fx1 = np.array([[5, 4, 1, 0],
        >>>                        [0, 1, 5, 4],
        >>>                        [0, 1, 5, 4],
        >>>                        [2, 3, 1, 5],
        >>>                        [5, 1, 0, 4],
        >>>                        [3, 1, 5, 0]], dtype=np.int32)
        >>> # execute function
        >>> fx2_to_xyerr_sqrd = get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
        >>> fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
        >>> # verify results
        >>> result = str(fx2_to_xyerr)
        >>> print(result)
        [[  82.84813777  186.23801821  183.97945482  192.63939757]
         [ 382.98822312  374.35627682  122.17899418  289.15964849]
         [ 387.56336468  378.93044982  126.38890667  292.39140223]
         [ 419.24610836  176.66835461  400.17549807  167.41056948]
         [ 174.269059    274.28862645  281.03010583   33.520562  ]
         [  54.08322366  269.64496323   94.71123543  277.70556825]]

    """
    # Transform img1 keypoints into img2 space
    xy2    = ktool.get_xys(kpts2)
    xy1_t = ktool.transform_kpts_xys(kpts1, H)
    # get spatial keypoint distance to all neighbor candidates
    bcast_xy2   = xy2[:, None, :].T
    bcast_xy1_t = xy1_t.T[fx2_to_fx1]
    fx2_to_xyerr_sqrd = ltool.L2_sqrd(bcast_xy2, bcast_xy1_t)
    return fx2_to_xyerr_sqrd


def constrain_matches(dlen_sqrd2, kpts1, kpts2, H, fx2_to_fx1, fx2_to_dist, match_xy_thresh, normalizer_mode='far'):
    r"""
    Args:
        dlen_sqrd2 (?):
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        fx2_to_fx1 (ndarray):
        fx2_to_dist (ndarray):
        match_xy_thresh (?): threshold is specified as a fraction of the diagonal chip length
        normalizer_mode (str):
    """
    # Find the normalized spatial error of all candidate matches
    fx2_to_xyerr_sqrd = get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
    fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
    fx2_to_xyerr_norm = np.divide(fx2_to_xyerr, np.sqrt(dlen_sqrd2))

    # Find matches and normalizers which are within the spatial constraints

    fx2_to_valid_match = ut.inbounds(fx2_to_xyerr_norm, 0, match_xy_thresh)
    fx2_to_fx1_match = ut.find_first_true_indicies(fx2_to_valid_match)

    if normalizer_mode == 'plus':
        fx2_to_fx1_norm = [None if fx1 is None else fx1 + 1 for fx1 in fx2_to_fx1_match]
    else:
        # Set normalizer constraints
        if normalizer_mode == 'far':
            normalizer_xy_bounds = (match_xy_thresh, None)
        if normalizer_mode == 'nearby':
            normalizer_xy_bounds = (0, match_xy_thresh)
        else:
            raise AssertionError('normalizer_mode=%r' % (normalizer_mode,))
        fx2_to_valid_normalizer = ut.inbounds(fx2_to_xyerr_norm, *normalizer_xy_bounds)
        fx2_to_fx1_norm = ut.find_next_true_indicies(fx2_to_valid_normalizer, fx2_to_fx1_match)

    # Filter out matches that could not be constrained
    assert fx2_to_fx1_match != fx2_to_fx1_norm
    fx2_to_hasmatch = [pos is not None for pos in fx2_to_fx1_norm]
    fx2_list = np.where(fx2_to_hasmatch)[0]
    k_match_list = np.array(ut.list_take(fx2_to_fx1_match, fx2_list))
    k_norm_list = np.array(ut.list_take(fx2_to_fx1_norm, fx2_list))

    # We now have 2d coordinates into fx2_to_fx1
    # Covnert into 1d coordinates for flat indexing into fx2_to_fx1
    _shape2d = fx2_to_fx1.shape
    _match_index_2d = np.vstack((fx2_list, k_match_list))
    _norm_index_2d  = np.vstack((fx2_list, k_norm_list))
    match_index_1d = np.ravel_multi_index(_match_index_2d, _shape2d)
    norm_index_1d  = np.ravel_multi_index(_norm_index_2d, _shape2d)

    # Find initial matches
    fx1_list = fx2_to_fx1.take(match_index_1d)
    fx1_norm_list = fx2_to_fx1.take(norm_index_1d)
    # compute constrained ratio score
    match_dist_list = fx2_to_dist.take(match_index_1d)
    norm_dist_list = fx2_to_dist.take(norm_index_1d)

    fm_constrained = np.vstack((fx1_list, fx2_list))
    # return noramlizers as well
    fm_normalizer_constrained = np.vstack((fx1_norm_list, fx2_list))

    assert not np.any(match_index_1d == norm_index_1d), 'index is same'
    assert not np.any(match_dist_list == norm_dist_list), 'dist is same'

    #ut.embed()
    constraintup = fm_constrained, fm_normalizer_constrained, match_dist_list, norm_dist_list
    return constraintup


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
    import pyflann
    try:
        fx2_to_fx1, _fx2_to_dist = flann.nn_index(vecs2, num_neighbors=K, checks=checks)
    except pyflann.FLANNException:
        print('vecs1.shape = %r' % (vecs1.shape,))
        print('vecs2.shape = %r' % (vecs2.shape,))
        print('vecs1.dtype = %r' % (vecs1.dtype,))
        print('vecs2.dtype = %r' % (vecs2.dtype,))
        raise
    fx2_to_dist = np.divide(_fx2_to_dist, pseudo_max_dist_sqrd)
    return fx2_to_fx1, fx2_to_dist


def ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh):

    fx2_to_ratio = np.divide(fx2_to_dist.T[0], fx2_to_dist.T[1])
    fx2_to_isvalid = fx2_to_ratio < ratio_thresh
    fx2_m = np.where(fx2_to_isvalid)[0]
    fx1_m = fx2_to_fx1.T[0].take(fx2_m)
    fs = np.subtract(1.0, fx2_to_ratio.take(fx2_m))
    fm = np.vstack((fx1_m, fx2_m)).T
    # return normalizer info as well
    fx1_m_normalizer = fx2_to_fx1.T[1].take(fx2_m)
    fm_normalizer = np.vstack((fx1_m_normalizer, fx2_m)).T
    return fm, fs, fm_normalizer


def ratio_test2(match_dist_list, norm_dist_list, fm_constrained, ratio_thresh2=.8):
    ratio_list = np.divide(match_dist_list, norm_dist_list)
    #ratio_thresh = .625
    #ratio_thresh = .725
    validratio_list = np.less(ratio_list, ratio_thresh2)
    fm_SCR = fm_constrained.T[validratio_list].T
    fs_SCR = np.subtract(1.0, ratio_list[validratio_list])  # NOQA
    #fm_SCR = np.vstack((fx1_m, fx2_m)).T  # NOQA
    return fm_SCR, fs_SCR


def sver_inliers(kpts1, kpts2, fm, fs, dlen_sqrd2, sver_xy_thresh):
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm, sver_xy_thresh, dlen_sqrd2)
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
    title = title + '\n num=%d, sum=%.2f' % (len(fm), sum(fs))
    pt.set_title(title)
    #pt.set_figtitle(title)
    # if update:
    #pt.iup()


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.constrained_matching
        python -m vtool.constrained_matching --allexamples
        python -m vtool.constrained_matching --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
