# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import numpy as np
from collections import namedtuple
from vtool import keypoint as ktool
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[matching]', DEBUG=False)

MatchTup3 = namedtuple('MatchTup3', ('fm', 'fs', 'fm_norm'))
MatchTup2 = namedtuple('MatchTup2', ('fm', 'fs'))


# maximum SIFT matching distance based using uint8 trick from hesaff
PSEUDO_MAX_VEC_COMPONENT = 512
PSEUDO_MAX_DIST_SQRD = 2 * (PSEUDO_MAX_VEC_COMPONENT ** 2)
PSEUDO_MAX_DIST = np.sqrt(2) * (PSEUDO_MAX_VEC_COMPONENT)


class SingleMatch(ut.NiceRepr):

    def __init__(self, matches, metadata):
        self.matches = matches
        self.metadata = metadata

    def show(self, *args, **kwargs):
        show_matching_dict(self.matches, self.metadata, *args, **kwargs)

    def make_interaction(self, *args, **kwargs):
        return make_match_interaction(self.matches, self.metadata, *args, **kwargs)

    def __nice__(self):
        return ' ' + ', '.join([key + '=%d' % (len(m.fm)) for key, m in self.matches.items()])
        #tup = (len(self.matches['ORIG'][0]), len(self.matches['RAT'][0]),
        #       len(self.matches['RAT+SV'][0]), )
        #return ' %d, %d, %d' % tup


def make_match_interaction(matches, metadata, type_='RAT+SV', **kwargs):
    import plottool.interact_matches
    #import plottool as pt
    fm, fs = matches[type_][0:2]
    H1 = metadata['H_' + type_.split('+')[0]]
    #fm, fs = matches['RAT'][0:2]
    rchip1 = metadata['rchip1']
    rchip2 = metadata['rchip2']
    kpts1 = metadata['kpts1']
    kpts2 = metadata['kpts2']

    vecs1 = metadata['vecs1']
    vecs2 = metadata['vecs2']

    #pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)

    vecs1, vecs2 = ut.dict_take(metadata, ['vecs1', 'vecs2'])
    fsv = fs[:, None]
    interact = plottool.interact_matches.MatchInteraction2(
        rchip1, rchip2, kpts1, kpts2, fm, fs, fsv, vecs1, vecs2, H1=H1,
        **kwargs)
    return interact


def show_matching_dict(matches, metadata, *args, **kwargs):
    interact = make_match_interaction(matches, metadata, *args, **kwargs)
    interact.show_page()
    return interact
    #MatchInteraction2


def vsone_image_fpath_matching(rchip_fpath1, rchip_fpath2, cfgdict={}, metadata_=None):
    r"""
    Args:
        rchip_fpath1 (str):
        rchip_fpath2 (str):
        cfgdict (dict): (default = {})

    CommandLine:
        python -m vtool --tf vsone_image_fpath_matching --show
        python -m vtool --tf vsone_image_fpath_matching --show --helpx
        python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+siam128
        python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+siam128 --ratio-thresh=.9
        python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+sift --ratio-thresh=.8
        python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+sift --ratio-thresh=.8

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> import vtool as vt
        >>> rchip_fpath1 = ut.grab_test_imgpath('easy1.png')
        >>> rchip_fpath2 = ut.grab_test_imgpath('easy2.png')
        >>> import pyhesaff
        >>> metadata_ = None
        >>> default_cfgdict = dict(feat_type='hesaff+sift', ratio_thresh=.625,
        >>>                        **pyhesaff.get_hesaff_default_params())
        >>> cfgdict = ut.parse_dict_from_argv(default_cfgdict)
        >>> matches, metadata = vsone_image_fpath_matching(rchip_fpath1,
        >>>                                                rchip_fpath2, cfgdict)
        >>> ut.quit_if_noshow()
        >>> show_matching_dict(matches, metadata, mode=1)
        >>> ut.show_if_requested()
    """
    metadata = ut.LazyDict()
    if metadata_ is not None:
        metadata.update(metadata_)
    metadata['rchip_fpath1'] = rchip_fpath1
    metadata['rchip_fpath2'] = rchip_fpath2
    matches, metdata =  vsone_matching(metadata, cfgdict)
    return matches, metdata


def testdata_annot_metadata(rchip_fpath, cfgdict={}):
    metadata = ut.LazyDict({'rchip_fpath': rchip_fpath})
    ensure_metadata_feats(metadata, '', cfgdict)
    return metadata


def ensure_metadata_feats(metadata, suffix='', cfgdict={}):
    r"""
    Adds feature evaluation keys to a lazy dictionary

    Args:
        metadata (utool.LazyDict):
        suffix (str): (default = '')
        cfgdict (dict): (default = {})

    CommandLine:
        python -m vtool.matching --exec-ensure_metadata_feats --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> rchip_fpath = ut.grab_test_imgpath('easy1.png')
        >>> metadata = ut.LazyDict({'rchip_fpath': rchip_fpath})
        >>> suffix = ''
        >>> cfgdict = {}
        >>> ensure_metadata_feats(metadata, suffix, cfgdict)
        >>> assert len(metadata._stored_results) == 1
        >>> metadata['kpts']
        >>> assert len(metadata._stored_results) == 4
        >>> metadata['vecs']
        >>> assert len(metadata._stored_results) == 5
    """
    import vtool as vt
    rchip_key = 'rchip' + suffix
    _feats_key = '_feats' + suffix
    kpts_key = 'kpts' + suffix
    vecs_key = 'vecs' + suffix
    rchip_fpath_key = 'rchip_fpath' + suffix

    if rchip_key not in metadata:
        def eval_rchip1():
            rchip_fpath1 = metadata[rchip_fpath_key]
            return vt.imread(rchip_fpath1)
        metadata.set_lazy_func(rchip_key, eval_rchip1)

    if kpts_key not in metadata or vecs_key not in metadata:
        def eval_feats():
            rchip = metadata[rchip_key]
            _feats = vt.extract_features(rchip, **cfgdict)
            return _feats

        def eval_kpts():
            _feats = metadata[_feats_key]
            kpts = _feats[0]
            return kpts

        def eval_vecs():
            _feats = metadata[_feats_key]
            vecs = _feats[1]
            return vecs
        metadata.set_lazy_func(_feats_key, eval_feats)
        metadata.set_lazy_func(kpts_key, eval_kpts)
        metadata.set_lazy_func(vecs_key, eval_vecs)
    return metadata


def vsone_matching2(metadata, cfgdict={}, verbose=None):
    matches, metadata = vsone_matching(metadata, cfgdict=cfgdict, verbose=verbose)
    match = SingleMatch(matches, metadata)
    return match


def vsone_matching(metadata, cfgdict={}, verbose=None):
    """
    Metadata is a dictionary that contains either computed information
    necessary for matching or the dependenceis of those computations.

    Args:
        metadata (utool.LazyDict):
        cfgdict (dict): (default = {})
        verbose (bool):  verbosity flag(default = None)

    Returns:
        tuple: (matches, metadata)
    """
    # import vtool as vt
    assert isinstance(metadata, ut.LazyDict), 'type(metadata)=%r' % (type(metadata),)

    ensure_metadata_feats(metadata, suffix='1', cfgdict=cfgdict)
    ensure_metadata_feats(metadata, suffix='2', cfgdict=cfgdict)

    if 'dlen_sqrd2' not in metadata:
        def eval_dlen_sqrd2():
            rchip2 = metadata['rchip2']
            dlen_sqrd2 = rchip2.shape[0] ** 2 + rchip2.shape[1] ** 2
            return dlen_sqrd2
        metadata.set_lazy_func('dlen_sqrd2', eval_dlen_sqrd2)

    # Exceute relevant dependencies
    kpts1 = metadata['kpts1']
    vecs1 = metadata['vecs1']
    kpts2 = metadata['kpts2']
    vecs2 = metadata['vecs2']
    dlen_sqrd2 = metadata['dlen_sqrd2']
    flann1 = metadata.get('flann1', None)
    flann2 = metadata.get('flann2', None)

    matches, output_metdata = vsone_feature_matching(
        kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict=cfgdict,
        flann1=flann1, flann2=flann2, verbose=verbose)
    metadata.update(output_metdata)
    return matches, metadata


def vsone_feature_matching(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict={},
                           flann1=None, flann2=None, verbose=None):
    r"""
    Actual logic for matching
    Args:
        vecs1 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        vecs2 (ndarray[uint8_t, ndim=2]): SIFT descriptors
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints

    Ignore:
        >>> from vtool.matching import *  # NOQA
        %pylab qt4
        import plottool as pt
        pt.imshow(rchip1)
        pt.draw_kpts2(kpts1)

        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
        pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
    """
    import vtool as vt
    import pyflann
    from vtool import spatial_verification as sver
    #import vtool as vt
    sver_xy_thresh = cfgdict.get('sver_xy_thresh', .01)
    ratio_thresh =  cfgdict.get('ratio_thresh', .625)
    refine_method =  cfgdict.get('refine_method', 'homog')
    symmetric =  cfgdict.get('symmetric', False)
    K =  cfgdict.get('K', 1)
    Knorm =  cfgdict.get('Knorm', 1)
    #ratio_thresh =  .99
    # GET NEAREST NEIGHBORS
    checks = 800
    #pseudo_max_dist_sqrd = (np.sqrt(2) * 512) ** 2
    #pseudo_max_dist_sqrd = 2 * (512 ** 2)
    if verbose is None:
        verbose = True

    flann_params = {'algorithm': 'kdtree', 'trees': 8}
    if flann1 is None:
        flann1 = vt.flann_cache(vecs1, flann_params=flann_params, verbose=verbose)

    print('symmetric = %r' % (symmetric,))
    if symmetric:
        if flann2 is None:
            flann2 = vt.flann_cache(vecs2, flann_params=flann_params, verbose=verbose)

    try:
        num_neighbors = K + Knorm
        fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(flann1, vecs2, num_neighbors, checks)
        #fx2_to_fx1, _fx2_to_dist = flann1.nn_index(vecs2, num_neighbors=K, checks=checks)
        if symmetric:
            fx1_to_fx2, fx1_to_dist = normalized_nearest_neighbors(flann2, vecs1, K, checks)

    except pyflann.FLANNException:
        print('vecs1.shape = %r' % (vecs1.shape,))
        print('vecs2.shape = %r' % (vecs2.shape,))
        print('vecs1.dtype = %r' % (vecs1.dtype,))
        print('vecs2.dtype = %r' % (vecs2.dtype,))
        raise

    if symmetric:
        is_symmetric = flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2)
        fx2_to_fx1 = fx2_to_fx1.compress(is_symmetric, axis=0)
        fx2_to_dist = fx2_to_dist.compress(is_symmetric, axis=0)

    assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist)

    fx2_match, fx1_match, fx1_norm, match_dist, norm_dist = assigntup
    fm_ORIG = np.vstack((fx1_match, fx2_match)).T
    fs_ORIG = 1 - np.divide(match_dist, norm_dist)
    # APPLY RATIO TEST
    fm_RAT, fs_RAT, fm_norm_RAT = ratio_test(fx2_match, fx1_match, fx1_norm,
                                             match_dist, norm_dist,
                                             ratio_thresh)
    # SPATIAL VERIFICATION FILTER
    #with ut.EmbedOnException():
    match_weights = np.ones(len(fm_RAT))
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_RAT, sver_xy_thresh,
                                       dlen_sqrd2, match_weights=match_weights,
                                       refine_method=refine_method)
    if svtup is not None:
        (homog_inliers, homog_errors, H_RAT) = svtup[0:3]
    else:
        H_RAT = np.eye(3)
        homog_inliers = []
    fm_RAT_SV = fm_RAT.take(homog_inliers, axis=0)
    fs_RAT_SV = fs_RAT.take(homog_inliers, axis=0)
    fm_norm_RAT_SV = fm_norm_RAT[homog_inliers]

    top_percent = .5
    top_idx = ut.take_percentile(fx2_to_dist.T[0].argsort(), top_percent)
    fm_TOP = fm_ORIG.take(top_idx, axis=0)
    fs_TOP = fx2_to_dist.T[0].take(top_idx)
    #match_weights = np.ones(len(fm_TOP))
    #match_weights = (np.exp(fs_TOP) / np.sqrt(np.pi * 2))
    match_weights = 1 - fs_TOP
    #match_weights = np.ones(len(fm_TOP))
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_TOP, sver_xy_thresh,
                                       dlen_sqrd2, match_weights=match_weights,
                                       refine_method=refine_method)
    if svtup is not None:
        (homog_inliers, homog_errors, H_TOP) = svtup[0:3]
        np.sqrt(homog_errors[0] / dlen_sqrd2)
    else:
        H_TOP = np.eye(3)
        homog_inliers = []
    fm_TOP_SV = fm_TOP.take(homog_inliers, axis=0)
    fs_TOP_SV = fs_TOP.take(homog_inliers, axis=0)

    matches = {
        'ORIG'   : MatchTup2(fm_ORIG, fs_ORIG),
        'RAT'    : MatchTup3(fm_RAT, fs_RAT, fm_norm_RAT),
        'RAT+SV' : MatchTup3(fm_RAT_SV, fs_RAT_SV, fm_norm_RAT_SV),
        'TOP'    : MatchTup2(fm_TOP, fs_TOP),
        'TOP+SV' : MatchTup2(fm_TOP_SV, fs_TOP_SV),
    }
    output_metdata = {
        'H_RAT': H_RAT,
        'H_TOP': H_TOP,
    }
    return matches, output_metdata


def normalized_nearest_neighbors(flann, vecs2, K, checks=800):
    """
    uses flann index to return nearest neighbors with distances normalized
    between 0 and 1 using sifts uint8 trick
    """
    import vtool as vt
    fx2_to_fx1, _fx2_to_dist_sqrd = flann.nn_index(vecs2, num_neighbors=K, checks=checks)
    _fx2_to_dist = np.sqrt(_fx2_to_dist_sqrd.astype(np.float64))
    fx2_to_dist = np.divide(_fx2_to_dist, PSEUDO_MAX_DIST)  # normalized dist
    fx2_to_fx1 = vt.atleast_nd(fx2_to_fx1, 2)
    fx2_to_dist = vt.atleast_nd(fx2_to_dist, 2)
    #fx2_to_dist = np.divide(_fx2_to_dist.astype(np.float64),
    #PSEUDO_MAX_DIST_SQRD)  # squared normalized dist
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
            fx2_match, - matching feature indices in image 2
            fx1_match, - matching feature indices in image 1
            fx1_norm,  - normmalizing indices in image 1
            match_dist, - descriptor distances between fx2_match and fx1_match
            norm_dist, - descriptor distances between fx2_match and fx1_norm
            )

    CommandLine:
        python -m vtool.matching --test-assign_spatially_constrained_matches

    Example:
        >>> # ENABLE_DOCTEST
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
            np.array([0, 1, 2], dtype=np.int32),
            np.array([2, 0, 2], dtype=np.int32),
            np.array([1, 1, 0], dtype=np.int32),
            np.array([ 0.4,  0.3,  0.8], dtype=np.float32),
            np.array([ 0.8,  0.5,  0.9], dtype=np.float32),
        )

    Example:

    assigns spatially constrained vsone match using results of nearest
    neighbors.
    """
    import vtool as vt
    index_dtype = fx2_to_fx1.dtype
    # Find spatial errors of keypoints under current homography (kpts1 mapped into image2 space)
    fx2_to_xyerr_sqrd = ktool.get_match_spatial_squared_error(kpts1, kpts2, H, fx2_to_fx1)
    fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
    fx2_to_xyerr_norm = np.divide(fx2_to_xyerr, np.sqrt(chip2_dlen_sqrd))

    # Find matches and normalizers that satisfy spatial constraints
    fx2_to_valid_match      = ut.inbounds(fx2_to_xyerr_norm, 0.0, match_xy_thresh, eq=True)
    fx2_to_valid_normalizer = ut.inbounds(fx2_to_xyerr_norm, *norm_xy_bounds, eq=True)
    fx2_to_fx1_match_col = vt.find_first_true_indices(fx2_to_valid_match)
    fx2_to_fx1_norm_col  = vt.find_next_true_indices(fx2_to_valid_normalizer, fx2_to_fx1_match_col)

    assert fx2_to_fx1_match_col != fx2_to_fx1_norm_col, 'normlizers are matches!'

    fx2_to_hasmatch = [pos is not None for pos in fx2_to_fx1_norm_col]
    # IMAGE 2 Matching Features
    fx2_match = np.where(fx2_to_hasmatch)[0].astype(index_dtype)
    match_col_list = np.array(ut.take(fx2_to_fx1_match_col, fx2_match), dtype=fx2_match.dtype)
    norm_col_list = np.array(ut.take(fx2_to_fx1_norm_col, fx2_match), dtype=fx2_match.dtype)

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
            np.array([ 0.059,  0.021,  0.039,  0.15 ,  0.227,  0.216], dtype=np.float64),
            np.array([ 0.239,  0.241,  0.248,  0.151,  0.244,  0.236], dtype=np.float64),
        )
    """
    fx2_match = np.arange(len(fx2_to_fx1), dtype=fx2_to_fx1.dtype)
    fx1_match = fx2_to_fx1.T[0]
    fx1_norm  = fx2_to_fx1.T[1]
    match_dist = fx2_to_dist.T[0]
    norm_dist  = fx2_to_dist.T[1]
    assigntup = fx2_match, fx1_match, fx1_norm, match_dist, norm_dist
    return assigntup


def flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> rng = np.random.RandomState(0)
        >>> fx2_to_fx1 = np.array([[ 0,  1],
        >>>                        [ 1,  4],
        >>>                        [ 3,  4],
        >>>                        [ 2,  0]], dtype=np.int32)
        >>> fx1_to_fx2 = np.array([[ 0, 1],
        >>>                        [ 2, 1],
        >>>                        [ 3, 1],
        >>>                        [ 3, 1],
        >>>                        [ 0, 1]], dtype=np.int32)
        >>> is_symmetric1 = flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2)
        >>> result = ut.array_repr2(is_symmetric1)
        >>> print(result)
        array([ True, False,  True, False], dtype=bool)
    """
    # np.arange(len(fx2_to_fx1), dtype=fx2_to_fx1.dtype)
    match_fx1_to_fx2 = fx1_to_fx2.T[0]
    match_fx2_to_fx1 = fx2_to_fx1.T[0]
    indices2 = np.arange(len(match_fx2_to_fx1))
    is_symmetric2 = match_fx1_to_fx2[match_fx2_to_fx1] == indices2
    return is_symmetric2


def unconstrained_ratio_match(flann, vecs2, unc_ratio_thresh=.625,
                              fm_dtype=np.int32, fs_dtype=np.float32):
    """ Lowes ratio matching

    from vtool.matching import *  # NOQA
    fs_dtype = rat_kwargs.get('fs_dtype', np.float32)
    fm_dtype = rat_kwargs.get('fm_dtype', np.int32)
    unc_ratio_thresh = rat_kwargs.get('unc_ratio_thresh', .625)

    """
    fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(
        flann, vecs2, K=2, checks=800)
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
        fx2_to_fx1 (ndarray): nearest neighbor indices (from flann)
        fx2_to_dist (ndarray): nearest neighbor distances (from flann)
        ratio_thresh (float):
        match_col (int or ndarray): column of matching indices
        norm_col (int or ndarray): column of normalizng indices

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
    ratio_tup = MatchTup3(fm_RAT, fs_RAT, fm_norm_RAT)
    return ratio_tup


def ensure_fsv_list(fsv_list):
    """ ensure fs is at least Nx1 """
    return [fsv[:, None] if len(fsv.shape) == 1 else fsv
            for fsv in fsv_list]


def marge_matches(fm_A, fm_B, fsv_A, fsv_B):
    """ combines feature matches from two matching algorithms

    Args:
        fm_A (ndarray[ndims=2]): type A feature matches
        fm_B (ndarray[ndims=2]): type B feature matches
        fsv_A (ndarray[ndims=2]): type A feature scores
        fsv_B (ndarray[ndims=2]): type B feature scores

    Returns:
        tuple: (fm_both, fs_both)

    CommandLine:
        python -m vtool.matching --test-marge_matches

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> # build test data
        >>> fm_A  = np.array([[ 15, 17], [ 54, 29], [ 95, 111], [ 25, 125], [ 97, 125]], dtype=np.int32)
        >>> fm_B  = np.array([[ 11, 21], [ 15, 17], [ 25, 125], [ 30,  32]], dtype=np.int32)
        >>> fsv_A = np.array([[ .1, .2], [1.0, .9], [.8,  .2],  [.1, .1], [1.0, .9]], dtype=np.float32)
        >>> fsv_B = np.array([[.12], [.3], [.5], [.7]], dtype=np.float32)
        >>> # execute function
        >>> (fm_both, fs_both) = marge_matches(fm_A, fm_B, fsv_A, fsv_B)
        >>> # verify results
        >>> result = ut.list_str((fm_both, fs_both), precision=3)
        >>> print(result)
        (
            np.array([[ 15,  17],
                      [ 25, 125],
                      [ 54,  29],
                      [ 95, 111],
                      [ 97, 125],
                      [ 11,  21],
                      [ 30,  32]], dtype=np.int32),
            np.array([[ 0.1 ,  0.2 ,  0.3 ],
                      [ 0.1 ,  0.1 ,  0.5 ],
                      [ 1.  ,  0.9 ,   nan],
                      [ 0.8 ,  0.2 ,   nan],
                      [ 1.  ,  0.9 ,   nan],
                      [  nan,   nan,  0.12],
                      [  nan,   nan,  0.7 ]], dtype=np.float64),
        )
    """
    # Flag rows found in both fmA and fmB
    # that are intersecting (both) or unique (only)
    import vtool as vt
    flags_both_A, flags_both_B = vt.intersect2d_flags(fm_A, fm_B)
    flags_only_A = np.logical_not(flags_both_A)
    flags_only_B = np.logical_not(flags_both_B)
    # independent matches
    fm_both_AB  = fm_A.compress(flags_both_A, axis=0)
    fm_only_A   = fm_A.compress(flags_only_A, axis=0)
    fm_only_B   = fm_B.compress(flags_only_B, axis=0)
    # independent scores
    fsv_both_A = fsv_A.compress(flags_both_A, axis=0)
    fsv_both_B = fsv_B.compress(flags_both_B, axis=0)
    fsv_only_A = fsv_A.compress(flags_only_A, axis=0)
    fsv_only_B = fsv_B.compress(flags_only_B, axis=0)
    # build merge offsets
    offset1 = len(fm_both_AB)
    offset2 = offset1 + len(fm_only_A)
    offset3 = offset2 + len(fm_only_B)
    # Merge feature matches
    fm_merged = np.vstack([fm_both_AB, fm_only_A, fm_only_B])
    # Merge feature scores
    num_rows = fm_merged.shape[0]
    num_cols_A = fsv_A.shape[1]
    num_cols_B = fsv_B.shape[1]
    num_cols = num_cols_A + num_cols_B
    fsv_merged = np.full((num_rows, num_cols), np.nan)
    fsv_merged[0:offset1, 0:num_cols_A] = fsv_both_A
    fsv_merged[0:offset1, num_cols_A:]  = fsv_both_B
    fsv_merged[offset1:offset2, 0:num_cols_A] = fsv_only_A
    fsv_merged[offset2:offset3, num_cols_A:]  = fsv_only_B
    return fm_merged, fsv_merged


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
