from __future__ import absolute_import, division, print_function
#from six.moves import range
import utool as ut
import numpy as np
from vtool import keypoint as ktool
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


#class SimpleMatchConfig(ut.Pref):
#    def __init__(simple_cfg, *args, **kwargs):
#        super(SimpleMatchConfig, simple_cfg).__init__(*args, **kwargs)
#        simple_cfg.normalizer_mode = 'nearby'


TAU = np.pi * 2


def param_interaction():
    r"""
    Returns:
        ?: testtup

    CommandLine:
        python -m vtool.constrained_matching --test-param_interaction

    Notes:
        python -m vtool.constrained_matching --test-param_interaction
        setparam normalizer_mode=nearby
        setparam normalizer_mode=far
        setparam ratio_thresh=.625
        setparam ratio_thresh=.5

        setparam ratio_thresh2=.625
        normalizer_mode=plus


    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.constrained_matching import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> testtup = param_interaction()
        >>> # verify results
        >>> result = str(testtup)
        >>> print(result)
    """
    import plottool as pt
    USE_IBEIS = ut.is_developer()
    if USE_IBEIS:
        from ibeis.model.hots import devcases
        index = 2
        fpath1, fpath2, fpath3 = devcases.get_dev_test_fpaths(index)
        testtup1 = testdata_matcher(fpath1, fpath2)
        testtup2 = testdata_matcher(fpath1, fpath3)
    else:
        testtup1 = testdata_matcher('easy1.png', 'easy2.png')
        testtup2 = testdata_matcher('easy1.png', 'hard3.png')
    testtup_list = [testtup1, testtup2]
    simp_list = [SimpleMatcher(testtup) for testtup in testtup_list]
    varied_dict = dict([
        ('sver_xy_thresh', .1),
        ('ratio_thresh', .625),
        ('search_K', 7),
        ('ratio_thresh2', .625),
        ('sver_xy_thresh2', .01),
        ('normalizer_mode', ['nearby', 'far', 'plus'][1]),
        ('match_xy_thresh', .1),
    ])
    cfgdict_list = ut.all_dict_combinations(varied_dict)
    tried_configs = []

    vizkey_ = ['normalizers']
    vizkey_ = ['matches']

    def toggle_vizmode(iiter, actionkey, value, vizkey_=vizkey_):
        if vizkey_[0] == 'normalizers':
            vizkey_[0] = 'matches'
        if vizkey_[0] == 'matches':
            vizkey_[0] = 'normalizers'
        print('toggling')

    def set_param(iiter, actionkey, value, vizkey_=vizkey_):
        """
        value = 'search_K=3'
        """
        paramkey, paramval = value.split('=')
        print('parsing value=%r' % (value,))
        def strip_quotes(str_):
            dq = ut.DOUBLE_QUOTE
            sq = ut.SINGLE_QUOTE
            return str_.strip(dq).strip(sq).strip(dq)
        # Sanatize
        paramkey = strip_quotes(paramkey)
        paramval = ut.smart_cast2(strip_quotes(paramval))
        print('setting cfgdict[%r]=%r' % (paramkey, paramval))
        iiter.iterable[iiter.index][paramkey] = paramval

    custom_actions = [
        ('toggle', ['t'], 'toggle', toggle_vizmode),
        ('set_param', ['setparam'], 'sets a config param using key=val format', set_param)
    ]

    for cfgdict in ut.InteractiveIter(cfgdict_list,
                                      #default_action='reload',
                                      custom_actions=custom_actions,
                                      wraparound=True):
        for simp in simp_list:
            simp.run_matching(cfgdict=cfgdict)
        vizkey = vizkey_[0]
        print('vizkey = %r' % (vizkey,))
        for fnum, simp in enumerate(simp_list):
            simp.visualize(vizkey, fnum=fnum)
        tried_configs.append(cfgdict.copy())
        print('Current Config = ')
        print(ut.dict_str(cfgdict))
        pt.present()
        pt.update()


def testdata_matcher(fname1='easy1.png', fname2='easy2.png'):
    """"
    fname1 = 'easy1.png'
    fname2 = 'hard3.png'
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
    def __init__(simp, testtup):
        simp.testtup = testtup
        simp.basetup = None
        simp.nexttup = None

    def visualize(simp, key, **kwargs):
        visualize_method = getattr(simp, 'visualize_' + key)
        return visualize_method(**kwargs)

    def visualize_matches(simp, **kwargs):
        r"""
        CommandLine:
            python -m vtool.constrained_matching --test-visualize_matches --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.constrained_matching import *  # NOQA
            >>> import plottool as pt
            >>> simp = SimpleMatcher(testdata_matcher())
            >>> simp.run_matching()
            >>> result = simp.visualize_matches()
            >>> pt.show_if_requested()
        """
        nRows = 2
        nCols = 3
        show_matches_ = simp.start_new_viz(nRows, nCols, **kwargs)

        show_matches_('ORIG')
        show_matches_('RAT')
        show_matches_('SV')
        show_matches_('SC')
        show_matches_('SCR')
        show_matches_('SCRSV')

    def visualize_normalizers(simp, **kwargs):
        """
        CommandLine:
            python -m vtool.constrained_matching --test-visualize_normalizers --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.constrained_matching import *  # NOQA
            >>> import plottool as pt
            >>> simp = SimpleMatcher(testdata_matcher())
            >>> simp.run_matching()
            >>> result = simp.visualize_normalizers()
            >>> pt.show_if_requested()
        """
        nRows = 2
        nCols = 2
        show_matches_ = simp.start_new_viz(nRows, nCols, **kwargs)

        show_matches_('RAT')
        show_matches_('SCR')

        show_matches_('RAT', norm=True)
        show_matches_('SCR', norm=True)

        #show_matches_(fm_RAT, fs_RAT, title='ratio filtered')
        #show_matches_(fm_SCR, fs_SCR, title='constrained matches')

        #show_matches_(fm_norm_RAT, fs_RAT, title='ratio normalizers', cmap='cool')
        #show_matches_(fm_norm_SCR, fs_SCR, title='constrained normalizers', cmap='cool')

    def start_new_viz(simp, nRows, nCols, fnum=None):
        import plottool as pt

        rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2  = simp.testtup
        fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT   = simp.basetup
        fm_SC, fs_SC, fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR = simp.nexttup
        fm_norm_RAT, fm_norm_SV                                 = simp.base_meta
        fm_norm_SC, fm_norm_SCR, fm_norm_SVSCR                  = simp.next_meta

        locals_ = ut.delete_dict_keys(locals(), ['title'])

        keytitle_tups = [
            ('ORIG', 'initial neighbors'),
            ('RAT', 'ratio filtered'),
            ('SV', 'ratio filtered + SV'),
            ('SC', 'spatially constrained'),
            ('SCR', 'spatially constrained + ratio'),
            ('SCRSV', 'spatially constrained + SV'),
        ]
        keytitle_dict = dict(keytitle_tups)
        key_list = ut.get_list_column(keytitle_tups, 0)
        matchtup_dict = {
            key: (locals_['fm_' + key], locals_['fs_' + key])
            for key in key_list
        }
        normtup_dict = {
            key: locals_.get('fm_norm_' + key, None)
            for key in key_list
        }

        next_pnum = pt.make_pnum_nextgen(nRows=nRows, nCols=nCols)
        if fnum is None:
            fnum = pt.next_fnum()
        pt.figure(fnum=fnum, doclf=True, docla=True)

        def show_matches_(key, **kwargs):
            assert key in key_list, 'unknown key=%r' % (key,)
            showkw = locals_.copy()
            showkw['pnum'] = next_pnum()
            showkw['fnum'] = fnum
            showkw.update(kwargs)
            _fm, _fs = matchtup_dict[key]
            title = keytitle_dict[key]
            if kwargs.get('norm', False):
                _fm = normtup_dict[key]
                assert _fm is not None, key
                showkw['cmap'] = 'cool'
                title += ' normalizers'
            show_matches(_fm, _fs, title=title, **showkw)
        # state hack
        show_matches_.next_pnum = next_pnum
        return show_matches_

    def run_matching(simp, testtup=None, cfgdict={}):
        if testtup is None:
            testtup = simp.testtup
        basetup, base_meta = baseline_vsone_ratio_matcher(testtup, cfgdict)
        nexttup, next_meta = spatially_constrianed_matcher(testtup, basetup, cfgdict)
        simp.nexttup = nexttup
        simp.basetup = basetup
        simp.testtup = testtup
        simp.base_meta = base_meta
        simp.next_meta = next_meta

    def setstate_testdata(simp):
        testtup = testdata_matcher()
        simp.run_matching(testtup)


def score_matches():
    pass


def baseline_vsone_ratio_matcher(testtup, cfgdict={}):
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
    sver_xy_thresh = cfgdict.get('sver_xy_thresh', .01)
    ratio_thresh =  cfgdict.get('ratio_thresh', .625)
    # GET NEAREST NEIGHBORS
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=2)
    fx2_m = np.arange(len(fx2_to_fx1))
    fx1_m = fx2_to_fx1.T[0]
    fm_ORIG = np.vstack((fx1_m, fx2_m)).T
    #fs_ORIG = fx2_to_dist.T[0]
    fs_ORIG = 1 - np.divide(fx2_to_dist.T[0], fx2_to_dist.T[1])
    #np.ones(len(fm_ORIG))
    # APPLY RATIO TEST
    fm_RAT, fs_RAT, fm_norm_RAT = ratio_test(fx2_to_fx1, fx2_to_dist, ratio_thresh)
    # SPATIAL VERIFICATION FILTER
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_RAT, sver_xy_thresh, dlen_sqrd2)
    if svtup is not None:
        (homog_inliers, homog_errors, H_RAT) = svtup[0:3]
    else:
        H_RAT = np.eye(3)
        homog_inliers = []
    fm_SV = fm_RAT[homog_inliers]
    fs_SV = fs_RAT[homog_inliers]
    fm_norm_SV = fm_norm_RAT[homog_inliers]

    base_tup = (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT)
    base_meta = (fm_norm_RAT, fm_norm_SV)
    return base_tup, base_meta


def spatially_constrianed_matcher(testtup, basetup, cfgdict={}):
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
        >>> simp = SimpleMatcher()
        >>> # execute function
        >>> nexttup, next_meta = spatially_constrianed_matcher(testtup, basetup)
        >>> # verify results
        >>> print(nexttup)
    """
    #import vtool as vt
    (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2) = testtup
    (fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT) = basetup

    #match_xy_thresh = .1
    #sver_xy_thresh = .01
    #ratio_thresh2 = .8
    # Observation, scores don't change above K=7
    # on easy test case
    #search_K = 7  # 3
    search_K = cfgdict.get('search_K', 7)
    ratio_thresh2   = cfgdict.get('ratio_thresh2', .8)
    sver_xy_thresh2 = cfgdict.get('sver_xy_thresh2', .01)
    normalizer_mode = cfgdict.get('normalizer_mode', 'nearby')
    match_xy_thresh = cfgdict.get('match_xy_thresh', .1)

    # ASSIGN CANDIDATES
    # Get candidate nearest neighbors
    fx2_to_fx1, fx2_to_dist = assign_nearest_neighbors(vecs1, vecs2, K=search_K)

    # COMPUTE CONSTRAINTS
    #normalizer_mode = 'far'
    constrain_tup = constrain_matches(dlen_sqrd2, kpts1, kpts2, H_RAT, fx2_to_fx1,
                                      fx2_to_dist, match_xy_thresh,
                                      normalizer_mode=normalizer_mode)
    (fm_SC, fm_norm_SC, match_dist_list, norm_dist_list) = constrain_tup
    fs_SC = 1 - np.divide(match_dist_list, norm_dist_list)   # NOQA

    fm_SCR, fs_SCR, fm_norm_SCR = ratio_test2(match_dist_list, norm_dist_list, fm_SC,
                                                    fm_norm_SC, ratio_thresh2)

    # Another round of verification
    svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_SCR, sver_xy_thresh2, dlen_sqrd2)
    if svtup is not None:
        (homog_inliers, homog_errors, H_SCR) = svtup[0:3]
    else:
        H_SCR = np.eye(3)
        homog_inliers = []
    fm_SCRSV = fm_SCR[homog_inliers]
    fs_SCRSV = fs_SCR[homog_inliers]

    fm_norm_SVSCR = fm_norm_SCR[homog_inliers]

    nexttup = (fm_SC, fs_SC, fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR)
    next_meta = (fm_norm_SC, fm_norm_SCR, fm_norm_SVSCR)
    return nexttup, next_meta


def constrain_matches(dlen_sqrd2, kpts1, kpts2, H_RAT, fx2_to_fx1, fx2_to_dist, match_xy_thresh, normalizer_mode='far'):
    r"""
    Args:
        dlen_sqrd2 (?):
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints
        H_RAT (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        fx2_to_fx1 (ndarray):
        fx2_to_dist (ndarray):
        match_xy_thresh (?): threshold is specified as a fraction of the diagonal chip length
        normalizer_mode (str):
    """
    # Find the normalized spatial error of all candidate matches
    fx2_to_xyerr_sqrd = ktool.get_match_spatial_squared_error(kpts1, kpts2, H_RAT, fx2_to_fx1)
    fx2_to_xyerr = np.sqrt(fx2_to_xyerr_sqrd)
    fx2_to_xyerr_norm = np.divide(fx2_to_xyerr, np.sqrt(dlen_sqrd2))

    # Find matches and normalizers which are within the spatial constraints

    fx2_to_valid_match = ut.inbounds(fx2_to_xyerr_norm, 0, match_xy_thresh)
    fx2_to_fx1_kmatch = ut.find_first_true_indicies(fx2_to_valid_match)

    #if normalizer_mode == 'plus':
    #    normalizer_xy_bounds = (0, np.inf)
    #    #maxk = fx2_to_dist.shape[1]
    #    #fx2_to_fx1_knorm = [None if fx1 is None else min(fx1 + 1, maxk)
    #    #                    for fx1 in fx2_to_fx1_kmatch]
    #else:
    if normalizer_mode == 'plus':
        normalizer_xy_bounds = (0, np.inf)
    # Set normalizer constraints
    elif normalizer_mode == 'far':
        normalizer_xy_bounds = (match_xy_thresh, np.inf)
    elif normalizer_mode == 'nearby':
        normalizer_xy_bounds = (0, match_xy_thresh)
    else:
        raise AssertionError('normalizer_mode=%r' % (normalizer_mode,))
    fx2_to_valid_normalizer = ut.inbounds(fx2_to_xyerr_norm, *normalizer_xy_bounds)
    fx2_to_fx1_knorm = ut.find_next_true_indicies(fx2_to_valid_normalizer, fx2_to_fx1_kmatch)

    # Filter out matches that could not be constrained
    assert fx2_to_fx1_kmatch != fx2_to_fx1_knorm
    fx2_to_hasmatch = [pos is not None for pos in fx2_to_fx1_knorm]
    fx2_list = np.where(fx2_to_hasmatch)[0]
    k_match_list = np.array(ut.list_take(fx2_to_fx1_kmatch, fx2_list))
    k_norm_list = np.array(ut.list_take(fx2_to_fx1_knorm, fx2_list))

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

    fm_constrained = np.vstack((fx1_list, fx2_list)).T
    # return noramlizers as well
    fm_norm_constrained = np.vstack((fx1_norm_list, fx2_list)).T

    try:
        assert not np.any(match_index_1d == norm_index_1d), 'index is same'
        #assert not np.any(match_dist_list == norm_dist_list), 'dist is same'
    except Exception as ex:
        ut.printex(ex)
        issame_pos = np.where(match_dist_list == norm_dist_list)[0]
        match_index_1d[issame_pos]
        norm_index_1d[issame_pos]
        fx1 = fx1_list[issame_pos]
        fx1_norm = fx1_norm_list[issame_pos]
        print(kpts1[fx1])
        print(kpts1[fx1_norm])
        print(str())
        print(str(match_dist_list[match_dist_list == norm_dist_list]))
        #ut.embed()
        raise

    #ut.embed()
    constraintup = fm_constrained, fm_norm_constrained, match_dist_list, norm_dist_list
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
    fs_RAT = np.subtract(1.0, fx2_to_ratio.take(fx2_m))
    fm_RAT = np.vstack((fx1_m, fx2_m)).T
    # return normalizer info as well
    fx1_m_normalizer = fx2_to_fx1.T[1].take(fx2_m)
    fm_norm_RAT = np.vstack((fx1_m_normalizer, fx2_m)).T
    return fm_RAT, fs_RAT, fm_norm_RAT


def ratio_test2(match_dist_list, norm_dist_list, fm_SC, fm_norm_SC, ratio_thresh2=.8):
    ratio_list = np.divide(match_dist_list, norm_dist_list)
    #ratio_thresh = .625
    #ratio_thresh = .725
    isvalid_list = np.less(ratio_list, ratio_thresh2)
    valid_ratios = ratio_list[isvalid_list]
    fm_SCR = fm_SC[isvalid_list]
    fs_SCR = np.subtract(1.0, valid_ratios)  # NOQA
    fm_norm_SCR = fm_norm_SC[isvalid_list]
    #fm_SCR = np.vstack((fx1_m, fx2_m)).T  # NOQA
    return fm_SCR, fs_SCR, fm_norm_SCR


def show_matches(fm, fs, fnum=1, pnum=None, title='', cmap='hot', **locals_):
    #locals_ = locals()
    import plottool as pt
    # hack keys out of namespace
    keys = 'rchip1, rchip2, kpts1, kpts2'.split(', ')
    rchip1, rchip2, kpts1, kpts2 = ut.dict_take(locals_, keys)
    pt.figure(fnum=fnum, pnum=pnum)
    #doclf=True, docla=True)
    pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs, fnum=fnum, cmap=cmap)
    title = title + '\n num=%d, sum=%.2f' % (len(fm), sum(fs))
    pt.set_title(title)
    #pt.set_figtitle(title)
    # if update:
    #pt.iup()


def show_example():
    r"""
    CommandLine:
        python -m vtool.constrained_matching --test-show_example --show

    Example:
        >>> # DISABLE_DOCTEST
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
    simp1 = SimpleMatcher(testtup1)
    simp2 = SimpleMatcher(testtup2)
    simp1.run_matching()
    simp2.run_matching()
    #simp1.visualize_matches()
    #simp2.visualize_matches()
    simp1.visualize_normalizers()
    simp2.visualize_normalizers()
    #simp1.param_interaction()


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
