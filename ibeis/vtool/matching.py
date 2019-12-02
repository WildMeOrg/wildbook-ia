# -*- coding: utf-8 -*-
"""
    vt
    python -m utool.util_inspect check_module_usage --pat="matching.py"

"""
from __future__ import absolute_import, division, print_function
from vtool import _rhomb_dist
import six
import warnings
import pandas as pd
import utool as ut
import ubelt as ub
import numpy as np
import parse
from collections import namedtuple
try:
    import pyhesaff
except ImportError:
    pyhesaff = None
    pass


AssignTup = namedtuple('AssignTup', ('fm', 'match_dist', 'norm_fx1', 'norm_dist'))

TAU = 2 * np.pi  # tauday.org


class MatchingError(Exception):
    pass


if pyhesaff is not None:
    VSONE_FEAT_CONFIG = [
        ut.ParamInfo(key, val)
        for key, val in pyhesaff.get_hesaff_default_params().items()
    ]
else:
    VSONE_FEAT_CONFIG = []

VSONE_ASSIGN_CONFIG = [
    ut.ParamInfo('checks', 20),
    ut.ParamInfo('symmetric', False),
    ut.ParamInfo('weight', None, valid_values=[None, 'fgweights'],),
    ut.ParamInfo('K', 1, min_=1),
    ut.ParamInfo('Knorm', 1, min_=1),
]

VSONE_RATIO_CONFIG = [
    ut.ParamInfo('ratio_thresh', .625, min_=0.0, max_=1.0),
]


VSONE_SVER_CONFIG = [
    ut.ParamInfo('sv_on', True),
    ut.ParamInfo('thresh_bins', tuple(), type_=eval,
                 hideif=lambda cfg: not cfg['sv_on']),
    ut.ParamInfo('refine_method', 'homog', valid_values=['homog', 'affine'],
                 hideif=lambda cfg: not cfg['sv_on']),
    ut.ParamInfo('sver_xy_thresh', .01, min_=0.0, max_=None,
                 hideif=lambda cfg: not cfg['sv_on']),
    ut.ParamInfo('sver_ori_thresh', TAU / 4.0, min_=0.0, max_=TAU,
                 hideif=lambda cfg: not cfg['sv_on']),
    ut.ParamInfo('sver_scale_thresh', 2.0, min_=1.0, max_=None,
                 hideif=lambda cfg: not cfg['sv_on']),

]


NORM_CHIP_CONFIG = [
    # ---
    ut.ParamInfo('histeq', False, hideif=False),
    # ---
    ut.ParamInfo('adapteq', False, hideif=False),
    ut.ParamInfo('adapteq_ksize', 8, hideif=lambda cfg: not cfg['adapteq']),
    ut.ParamInfo('adapteq_limit', 2.0, hideif=lambda cfg: not cfg['adapteq']),
    # ---
    ut.ParamInfo('medianblur', False, hideif=False),
    ut.ParamInfo('medianblur_thresh', 50, hideif=lambda cfg: not cfg['medianblur']),
    ut.ParamInfo('medianblur_ksize1', 3, hideif=lambda cfg: not cfg['medianblur']),
    ut.ParamInfo('medianblur_ksize2', 5, hideif=lambda cfg: not cfg['medianblur']),
    # ---
]


VSONE_DEFAULT_CONFIG = (
    VSONE_ASSIGN_CONFIG + VSONE_RATIO_CONFIG + VSONE_SVER_CONFIG
)

VSONE_PI_DICT = {
    pi.varname: pi for pi in VSONE_DEFAULT_CONFIG
}


def demodata_match(cfgdict={}, apply=True, use_cache=True, recompute=False):
    import vtool as vt
    from vtool.inspect_matches import lazy_test_annot
    # hashid based on the state of the code
    if not apply:
        use_cache = False
    function_sig = ut.get_func_sourcecode(vt.matching.PairwiseMatch)
    hashid = ut.hash_data(function_sig)
    cfgstr = ub.hash_data(cfgdict.items()) + hashid
    cacher = ub.Cacher(
        'test_match_v5',
        cfgstr=cfgstr,
        appname='vtool',
        enabled=use_cache
    )
    match = cacher.tryload()
    annot1 = lazy_test_annot('easy1.png')
    annot2 = lazy_test_annot('easy2.png')
    if match is None or recompute:
        match = vt.PairwiseMatch(annot1, annot2)
        if apply:
            match.apply_all(cfgdict)
        cacher.save(match)
    else:
        match.annot1 = annot1
        match.annot2 = annot2
    return match


class PairwiseMatch(ub.NiceRepr):
    """
    Newest (Sept-16-2016) object oriented one-vs-one matching interface

    Creates an object holding two annotations
    Then a pipeline of operations can be applied to
    generate score and refine the matches

    Note:
        The annotation dictionaries are required to have certain attributes.

        Required annotation attributes:
            (kpts, vecs) OR rchip OR rchip_fpath

        Optional annotation attributes:
            aid, nid, flann, rchip, dlen_sqrd, weight
    """
    def __init__(match, annot1=None, annot2=None):
        match.annot1 = annot1
        match.annot2 = annot2
        match.fm = None
        match.fs = None
        match.H_21 = None
        match.H_12 = None

        match.verbose = False

        match.local_measures = ub.odict([])
        match.global_measures = ub.odict([])
        match._inplace_default = False

    @staticmethod
    def _available_params():
        return VSONE_PI_DICT

    @staticmethod
    def _take_params(config, keys):
        """
        take parameter info from config using default values defined in module
        constants.
        """
        # if isinstance(keys, six.string_types):
        #     keys = keys.split(', ')
        for key in keys:
            yield config[key] if key in config else VSONE_PI_DICT[key].default

    def __nice__(match):
        parts = []
        if 'aid' in match.annot1:
            aid1 = match.annot1['aid']
            aid2 = match.annot2['aid']
            vsstr = '%s-vs-%s' % (aid1, aid2)
            parts.append(vsstr)
        parts.append('None' if match.fm is None else
                     six.text_type(len(match.fm)))
        return ' '.join(parts)

    def __len__(match):
        if match.fm is not None:
            return len(match.fm)
        else:
            return 0

    def __getstate__(match):
        """
        The state of Pariwise Match ignores most of the annotation objects.

        This means that if you need properties of annots, you must reapply
        them after you load a PairwiseMatch object.
        """
        def _prepare_annot(annot):
            if isinstance(annot, ut.LazyDict):
                _annot = {}
                if 'aid' in annot:
                    _annot['aid'] = annot['aid']
                return _annot
            return annot

        _annot1 = _prepare_annot(match.annot1)
        _annot2 = _prepare_annot(match.annot2)

        state = {
            'annot1': _annot1,
            'annot2': _annot2,
            'fm': match.fm,
            'fs': match.fs,
            'H_21': match.H_21,
            'H_12': match.H_12,
            'global_measures': match.global_measures,
            'local_measures': match.local_measures,
        }
        return state

    def __setstate__(match, state):
        match.__dict__.update(state)
        match.verbose = False

    def show(match, ax=None, show_homog=False, show_ori=False, show_ell=True,
             show_pts=False, show_lines=True, show_rect=False, show_eig=False,
             show_all_kpts=False, mask_blend=0, ell_alpha=.6, line_alpha=.35,
             modifysize=False, vert=None, overlay=True, heatmask=False,
             line_lw=1.4):

        if match.verbose:
            print('[match] show')

        import plottool as pt
        annot1 = match.annot1
        annot2 = match.annot2
        try:
            rchip1 = annot1['nchip']
            rchip2 = annot2['nchip']
        except KeyError:
            print('Warning: nchip not set. fallback to rchip')
            rchip1 = annot1['rchip']
            rchip2 = annot2['rchip']

        if overlay:
            kpts1 = annot1['kpts']
            kpts2 = annot2['kpts']
        else:
            kpts1 = kpts2 = None
            show_homog = False
            show_ori = False
            show_ell = False
            show_pts = False
            show_lines = False
            show_rect = False
            show_eig = False
            # show_all_kpts = False
            # mask_blend = 0

        if mask_blend:
            import vtool as vt
            mask1 = vt.resize(annot1['probchip_img'], vt.get_size(rchip1))
            mask2 = vt.resize(annot2['probchip_img'], vt.get_size(rchip2))
            # vt.blend_images_average(vt.mask1, 1.0, alpha=mask_blend)
            rchip1 = vt.blend_images_mult_average(rchip1, mask1, alpha=mask_blend)
            rchip2 = vt.blend_images_mult_average(rchip2, mask2, alpha=mask_blend)
        fm = match.fm
        fs = match.fs

        H1 = match.H_12 if show_homog else None
        # H2 = match.H_21 if show_homog else None

        ax, xywh1, xywh2 = pt.show_chipmatch2(
            rchip1, rchip2, kpts1, kpts2, fm, fs, colorbar_=False,
            H1=H1, ax=ax,
            modifysize=modifysize,
            ori=show_ori, rect=show_rect, eig=show_eig, ell=show_ell,
            pts=show_pts, draw_lines=show_lines,
            all_kpts=show_all_kpts, line_alpha=line_alpha,
            ell_alpha=ell_alpha, vert=vert, heatmask=heatmask,
            line_lw=line_lw,
        )
        return ax, xywh1, xywh2

    def ishow(match):
        """
        CommandLine:
            python -m vtool.matching ishow --show

        Example:
            >>> # SCRIPT
            >>> from vtool.matching import *  # NOQA
            >>> import vtool as vt
            >>> import guitool as gt
            >>> gt.ensure_qapp()
            >>> match = demodata_match(use_cache=False)
            >>> self = match.ishow()
            >>> ut.quit_if_noshow()
            >>> gt.qtapp_loop(qwin=self, freq=10)
        """
        from vtool.inspect_matches import MatchInspector
        self = MatchInspector(match=match)
        self.show()
        return self

    def add_global_measures(match, global_keys):
        for key in global_keys:
            match.global_measures[key] = (match.annot1[key],
                                          match.annot2[key])

    def add_local_measures(match, xy=True, scale=True):
        import vtool as vt
        if xy:
            key_ = 'norm_xys'
            norm_xy1 = match.annot1[key_].take(match.fm.T[0], axis=1)
            norm_xy2 = match.annot2[key_].take(match.fm.T[1], axis=1)
            match.local_measures['norm_x1'] = norm_xy1[0]
            match.local_measures['norm_y1'] = norm_xy1[1]
            match.local_measures['norm_x2'] = norm_xy2[0]
            match.local_measures['norm_y2'] = norm_xy2[1]
        if scale:
            kpts1_m = match.annot1['kpts'].take(match.fm.T[0], axis=0)
            kpts2_m = match.annot2['kpts'].take(match.fm.T[1], axis=0)
            match.local_measures['scale1'] = vt.get_scales(kpts1_m)
            match.local_measures['scale2'] = vt.get_scales(kpts2_m)

    def matched_vecs2(match):
        return match.annot2['vecs'].take(match.fm.T[1], axis=0)

    def _next_instance(match, inplace=None):
        """
        Returns either the same or a new instance of a match object with the
        same global attributes.
        """
        if inplace is None:
            inplace = match._inplace_default
        if inplace:
            match_ = match
        else:
            match_ = match.__class__(match.annot1, match.annot2)
            match_.H_21 = match.H_21
            match_.H_12 = match.H_12
            match_._inplace_default = match._inplace_default
            match_.global_measures = match.global_measures.copy()
        return match_

    def copy(match):
        match_ = match._next_instance(inplace=False)
        if match.fm is not None:
            match_.fm = match.fm.copy()
            match_.fs = match.fs.copy()
        match_.local_measures = ub.map_vals(
                lambda a: a.copy(), match.local_measures)
        return match_

    def compress(match, flags, inplace=None):
        if match.verbose:
            print('[match] compressing {}/{}'.format(sum(flags), len(flags)))
        match_ = match._next_instance(inplace)
        match_.fm = match.fm.compress(flags, axis=0)
        match_.fs = match.fs.compress(flags, axis=0)
        match_.local_measures = ub.map_vals(
                lambda a: a.compress(flags), match.local_measures)
        return match_

    def take(match, indicies, inplace=None):
        match_ = match._next_instance(inplace)
        match_.fm = match.fm.take(indicies, axis=0)
        match_.fs = match.fs.take(indicies, axis=0)
        match_.local_measures = ub.map_vals(
                lambda a: a.take(indicies), match.local_measures)
        return match_

    def assign(match, cfgdict={}, verbose=None):
        """
        Assign feature correspondences between annots

        Example:
            >>> from vtool.matching import *  # NOQA
            >>> cfgdict = {'symmetric': True}
            >>> match = demodata_match({}, apply=False)
            >>> m1 = match.copy().assign({'symmetric': False})
            >>> m2 = match.copy().assign({'symmetric': True})

        Grid:
            from vtool.matching import *  # NOQA
            grid = {
                'symmetric': [True, False],
            }
            for cfgdict in ut.all_dict_combinations(grid):
                match = demodata_match(cfgdict, apply=False)
                match.assign()
        """
        params = match._take_params(cfgdict, ['K', 'Knorm', 'symmetric',
                                              'checks', 'weight'])
        params = list(params)
        K, Knorm, symmetric, checks, weight_key = params
        annot1 = match.annot1
        annot2 = match.annot2

        if match.verbose:
            print('[match] assign')
            print('[match] params = ' + ut.repr2(params))

        ensure_metadata_vsone(annot1, annot2, cfgdict)

        allow_shrink = True  # TODO: parameterize?

        # Search for nearest neighbors
        if symmetric:
            tup = symmetric_correspondence(annot1, annot2, K, Knorm, checks,
                                           allow_shrink)
        else:
            tup = asymmetric_correspondence(annot1, annot2, K, Knorm, checks,
                                            allow_shrink)
        fm, match_dist, norm_dist, fx1_norm, fx2_norm = tup

        ratio = np.divide(match_dist, norm_dist)
        # convert so bigger is better
        ratio_score = (1.0 - ratio)

        # remove local measure that can no longer apply
        match.local_measures = ub.dict_diff(match.local_measures, [
            'sver_err_xy', 'sver_err_scale', 'sver_err_ori'])

        match.local_measures['match_dist'] = match_dist
        match.local_measures['norm_dist'] = norm_dist
        match.local_measures['ratio'] = ratio
        match.local_measures['ratio_score'] = ratio_score

        if weight_key is None:
            match.fs = ratio_score
        else:
            # Weight the scores with specified attribute
            # (e.g. foregroundness weights)
            weight1 = annot1[weight_key].take(fm.T[0], axis=0)
            weight2 = annot2[weight_key].take(fm.T[1], axis=0)
            weight = np.sqrt(weight1 * weight2)
            weighted_ratio_score = ratio_score * weight

            match.local_measures[weight_key] = weight
            match.local_measures['weighted_ratio_score'] = weighted_ratio_score
            match.local_measures['weighted_norm_dist'] = norm_dist * weight
            match.fs = weighted_ratio_score

        match.fm = fm
        match.fm_norm1 = np.vstack([fx1_norm, fm.T[1]]).T
        if fx2_norm is None:
            match.fm_norm2 = None
        else:
            match.fm_norm2 = np.vstack([fm.T[1], fx2_norm]).T
        return match

    def ratio_test_flags(match, cfgdict={}):
        ratio_thresh, = match._take_params(cfgdict, ['ratio_thresh'])
        if match.verbose:
            print('[match] apply_ratio_test')
            print('[match] ratio_thresh = {!r}'.format(ratio_thresh))
        ratio = match.local_measures['ratio']
        flags = ratio < ratio_thresh
        return flags

    def sver_flags(match, cfgdict={}, return_extra=False):
        """
        Example:
            >>> from vtool.matching import *  # NOQA
            >>> cfgdict = {'symmetric': True, 'newsym': True}
            >>> match = demodata_match(cfgdict, apply=False)
            >>> cfgbase = {'symmetric': True, 'ratio_thresh': .8}
            >>> cfgdict = ut.dict_union(cfgbase, dict(thresh_bins=[.5, .6, .7, .8]))
            >>> match = match.assign(cfgbase)
            >>> match.apply_ratio_test(cfgdict, inplace=True)
            >>> flags1 = match.sver_flags(cfgdict)
            >>> flags2 = match.sver_flags(cfgbase)
        """
        from vtool import spatial_verification as sver
        import vtool as vt

        def _run_sver(kpts1, kpts2, fm, match_weights, **sver_kw):
            svtup = sver.spatially_verify_kpts(
                kpts1, kpts2, fm, match_weights=match_weights, **sver_kw)
            if svtup is None:
                errors = [np.empty(0), np.empty(0), np.empty(0)]
                inliers = []
                H_12 =  np.eye(3)
            else:
                (inliers, errors, H_12) = svtup[0:3]
            svtup = (inliers, errors, H_12)
            return svtup

        (sver_xy_thresh, sver_ori_thresh,
         sver_scale_thresh, refine_method, thresh_bins) = match._take_params(
             cfgdict, [
                 'sver_xy_thresh', 'sver_ori_thresh', 'sver_scale_thresh',
                 'refine_method', 'thresh_bins'
             ])

        kpts1 = match.annot1['kpts']
        kpts2 = match.annot2['kpts']
        dlen_sqrd2 = match.annot2['dlen_sqrd']

        sver_kw = dict(
            xy_thresh=sver_xy_thresh, ori_thresh=sver_ori_thresh,
            scale_thresh=sver_scale_thresh,
            refine_method=refine_method,
            dlen_sqrd2=dlen_sqrd2,
        )

        if thresh_bins:
            sver_tups = []

            n_fm = len(match.fm)

            xy_err = np.full(n_fm, fill_value=np.inf)
            scale_err = np.full(n_fm, fill_value=np.inf)
            ori_err = np.full(n_fm, fill_value=np.inf)
            agg_errors = (xy_err, scale_err, ori_err)

            agg_inlier_flags = np.zeros(n_fm, dtype=np.bool)

            agg_H_12 = None
            prev_best = 50

            for thresh in thresh_bins:
                ratio = match.local_measures['ratio']

                # These are of len(match.fm)=1000
                # 100 of these are True
                ratio_flags = ratio < thresh
                ratio_idxs = np.where(ratio_flags)[0]

                if len(ratio_idxs) == 0:
                    continue

                # Filter matches at this level of the ratio test
                fm = match.fm[ratio_flags]
                match_weights = match.fs[ratio_flags]

                svtup = _run_sver(kpts1, kpts2, fm, match_weights, **sver_kw)
                (inliers, errors, H_12) = svtup
                n_inliers = len(inliers)

                if agg_H_12 is None or (n_inliers < 100 and
                                        n_inliers > prev_best):
                    # pick a homography from a lower ratio threshold if
                    # possible. TODO: check for H_12 = np.eye
                    agg_H_12 = H_12
                    prev_best = n_inliers

                # these are of len(fm)=100
                # flags = vt.index_to_boolmask(inliers, len(fm))
                # print(errors[0][inliers].mean())

                # Find places that passed the ratio and were inliers
                agg_inlier_flags[ratio_idxs[inliers]] = True

                for agg_err, err in zip(agg_errors, errors):
                    if len(err):
                        current_err = agg_err[ratio_idxs]
                        agg_err[ratio_idxs] = np.minimum(current_err, err)

                sver_tups.append(svtup)

            if return_extra:
                return agg_inlier_flags, agg_errors, agg_H_12
            else:
                return agg_inlier_flags

        else:
            # match_weights = np.ones(len(fm))
            fm = match.fm
            match_weights = match.fs

            svtup = _run_sver(kpts1, kpts2, fm, match_weights, **sver_kw)
            (inliers, errors, H_12) = svtup

            flags = vt.index_to_boolmask(inliers, len(fm))

            if return_extra:
                return flags, errors, H_12
            else:
                return flags

    def apply_all(match, cfgdict):
        if match.verbose:
            print('[match] apply_all')
        match.H_21 = None
        match.H_12 = None
        match.local_measures = ut.odict([])
        match.assign(cfgdict)
        match.apply_ratio_test(cfgdict, inplace=True)
        sv_on, = match._take_params(cfgdict, ['sv_on'])

        if sv_on:
            match.apply_sver(cfgdict, inplace=True)

    def apply_ratio_test(match, cfgdict={}, inplace=None):
        flags = match.ratio_test_flags(cfgdict)
        match_ = match.compress(flags, inplace=inplace)
        return match_

    def apply_sver(match, cfgdict={}, inplace=None):
        """
        Example:
            >>> from vtool.matching import *  # NOQA
            >>> cfgdict = {'symmetric': True, 'ratio_thresh': .8,
            >>>            'thresh_bins': [.5, .6, .7, .8]}
            >>> match = demodata_match(cfgdict, apply=False)
            >>> match = match.assign(cfgbase)
            >>> match.apply_ratio_test(cfgdict, inplace=True)
            >>> flags1 = match.apply_sver(cfgdict)
        """
        if match.verbose:
            print('[match] apply_sver')
        flags, errors, H_12 = match.sver_flags(cfgdict,
                                               return_extra=True)
        match_ = match.compress(flags, inplace=inplace)
        errors_ = [e.compress(flags) for e in errors]
        match_.local_measures['sver_err_xy'] = errors_[0]
        match_.local_measures['sver_err_scale'] = errors_[1]
        match_.local_measures['sver_err_ori'] = errors_[2]
        match_.H_12 = H_12
        return match_

    def _make_global_feature_vector(match, global_keys=None):
        """ Global annotation properties and deltas """
        import vtool as vt
        feat = ut.odict([])

        if global_keys is None:
            # speed should need to be requested
            global_keys = sorted(match.global_measures.keys())
        global_measures = ut.dict_subset(match.global_measures, global_keys)

        for k, v in global_measures.items():
            v1, v2 = v
            if v1 is None:
                v1 = np.nan
            if v2 is None:
                v2 = np.nan
            if ut.isiterable(v1):
                for i in range(len(v1)):
                    feat['global({}_1[{}])'.format(k, i)] = v1[i]
                    feat['global({}_2[{}])'.format(k, i)] = v2[i]
                if k == 'gps':
                    delta = vt.haversine(v1, v2)
                else:
                    delta = np.abs(v1 - v2)
            else:
                feat['global({}_1)'.format(k)] = v1
                feat['global({}_2)'.format(k)] = v2
                # if k == 'yaw':
                #     # delta = vt.ori_distance(v1, v2)
                #     delta = vt.cyclic_distance(v1, v2, modulo=vt.TAU)
                if k == 'view':
                    delta = _rhomb_dist.VIEW_INT_DIST[(v1, v2)]
                    # delta = vt.cyclic_distance(v1, v2, modulo=8)
                else:
                    delta = np.abs(v1 - v2)
            feat['global(delta_{})'.format(k)] = delta
            assert k != 'yaw', 'yaw is depricated'

        # Impose ordering on these keys to add symmetry
        keys_to_order = ['qual', 'view']
        for key in keys_to_order:
            k1 = 'global({}_1)'.format(key)
            k2 = 'global({}_2)'.format(key)
            if k1 in feat and k2 in feat:
                minv, maxv = np.sort([feat[k1], feat[k2]])
                feat['global(min_{})'.format(key)] = minv
                feat['global(max_{})'.format(key)] = maxv

        if 'global(delta_gps)' in feat and 'global(delta_time)' in feat:
            hour_delta = feat['global(delta_time)'] / 360
            km_delta = feat['global(delta_gps)']
            if hour_delta == 0:
                if km_delta == 0:
                    feat['global(speed)'] = 0
                else:
                    feat['global(speed)'] = np.nan
            else:
                feat['global(speed)'] = km_delta / hour_delta
        return feat

    def _make_local_summary_feature_vector(match, local_keys=None,
                                           summary_ops=None, bin_key=None,
                                           bins=None):
        r"""
        Summary statistics of local features

        CommandLine:
            python -m vtool.matching _make_local_summary_feature_vector

        Example:
            >>> # ENABLE_DOCTEST
            >>> from vtool.matching import *  # NOQA
            >>> import vtool as vt
            >>> cfgdict = {}
            >>> match = demodata_match(cfgdict, recompute=0)
            >>> match.apply_all(cfgdict)
            >>> summary_ops = {'len', 'sum'}
            >>> bin_key = 'ratio'
            >>> bins = 2
            >>> #bins = [.625, .725, .9]
            >>> bins = [.625, .9]
            >>> local_keys = ['ratio', 'norm_dist', 'ratio_score']
            >>> bin_key = None
            >>> local_keys = None
            >>> summary_ops = 'all'
            >>> feat = match._make_local_summary_feature_vector(
            >>>     local_keys=local_keys,
            >>>     bin_key=bin_key, summary_ops=summary_ops, bins=bins)
            >>> result = ('feat = %s' % (ut.repr2(feat, nl=2),))
            >>> print(result)
        """

        if summary_ops is None:
            summary_ops = {'sum', 'mean', 'std', 'len'}
        if summary_ops == 'all':
            summary_ops = set(SUM_OPS.keys()).union({'len'})

        if local_keys is None:
            local_measures = match.local_measures
        else:
            local_measures = ut.dict_subset(match.local_measures, local_keys)

        feat = ut.odict([])
        if bin_key is not None:
            if bins is None:
                raise ValueError('must choose bins')
                # bins = [.625, .725, .9]
            # binned ratio feature vectors
            if isinstance(bins, int):
                bins = np.linspace(0, 1.0, bins + 1)
            else:
                bins = list(bins)
            bin_ids = np.searchsorted(bins, match.local_measures[bin_key])

            dimkey_fmt = '{opname}({measure}[{bin_key}<{binval}])'
            for binid, binval in enumerate(bins, start=1):
                fxs = np.where(bin_ids <= binid)[0]
                if 'len' in summary_ops:
                    dimkey = dimkey_fmt.format(
                        opname='len', measure='matches',
                        bin_key=bin_key,
                        binval=binval,
                    )
                    feat[dimkey] = len(fxs)
                for opname in sorted(summary_ops - {'len'}):
                    op = SUM_OPS[opname]
                    for k, vs in local_measures.items():
                        dimkey = dimkey_fmt.format(
                            opname=opname, measure=k,
                            bin_key=bin_key,
                            binval=binval,
                        )
                        feat[dimkey] = op(vs[fxs])

        else:
            if True:
                dimkey_fmt = '{opname}({measure})'
                if 'len' in summary_ops:
                    dimkey = dimkey_fmt.format(opname='len', measure='matches')
                    feat[dimkey] = len(match.fm)
                for opname in sorted(summary_ops - {'len'}):
                    op = SUM_OPS[opname]
                    for k, vs in local_measures.items():
                        dimkey = dimkey_fmt.format(opname=opname, measure=k)
                        feat[dimkey] = op(vs)
            else:
                # OLD
                if 'len' in summary_ops:
                    feat['len(matches)'] = len(match.fm)
                if 'sum' in summary_ops:
                    for k, vs in local_measures.items():
                        feat['sum(%s)' % (k,)] = vs.sum()
                if 'mean' in summary_ops:
                    for k, vs in local_measures.items():
                        feat['mean(%s)' % (k,)] = np.mean(vs)
                if 'std' in summary_ops:
                    for k, vs in local_measures.items():
                        feat['std(%s)' % (k,)] = np.std(vs)
                if 'med' in summary_ops:
                    for k, vs in local_measures.items():
                        feat['med(%s)' % (k,)] = np.median(vs)
        return feat

    def _make_local_top_feature_vector(match, local_keys=None, sorters='ratio',
                                       indices=3):
        """ Selected subsets of top features """
        if local_keys is None:
            local_measures = match.local_measures
        else:
            local_measures = ut.dict_subset(match.local_measures, local_keys)

        # Convert indices to an explicit list
        if isinstance(indices, int):
            indices = slice(indices)
        if isinstance(indices, slice):
            # assert indices.stop is not None, 'indices must have maximum value'
            indices = list(range(*indices.indices(len(match.fm))))
            # indices = list(range(*indices.indices(indices.stop)))
        if len(indices) == 0:
            return {}
        # TODO: some sorters might want descending orders
        sorters = ut.ensure_iterable(sorters)
        chosen_xs = [
            match.local_measures[sorter].argsort()[::-1][indices]
            for sorter in sorters
        ]
        loc_fmt = 'loc[{sorter},{rank}]({measure})'
        feat = ut.odict([
            (loc_fmt.format(sorter=sorter, rank=rank, measure=k), v)
            for sorter, topxs in zip(sorters, chosen_xs)
            for k, vs in local_measures.items()
            for rank, v in zip(indices, vs[topxs])
        ])
        return feat

    def make_feature_vector(match, local_keys=None, global_keys=None,
                            summary_ops=None, sorters='ratio', indices=3,
                            bin_key=None, bins=None):
        """
        Constructs the pairwise feature vector that represents a match

        Args:
            local_keys (None): (default = None)
            global_keys (None): (default = None)
            summary_ops (None): (default = None)
            sorters (str): (default = 'ratio')
            indices (int): (default = 3)

        Returns:
            dict: feat

        CommandLine:
            python -m vtool.matching make_feature_vector

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.matching import *  # NOQA
            >>> import vtool as vt
            >>> match = demodata_match({})
            >>> feat = match.make_feature_vector(indices=[0, 1])
            >>> result = ('feat = %s' % (ut.repr2(feat, nl=2),))
            >>> print(result)
        """
        feat = ut.odict([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat.update(match._make_global_feature_vector(global_keys))
            feat.update(match._make_local_summary_feature_vector(
                local_keys, summary_ops, bin_key=bin_key, bins=bins))
            feat.update(match._make_local_top_feature_vector(
                local_keys, sorters=sorters, indices=indices))
        return feat


def invsum(x):
    return np.sum(1 / x)


def csum(x):
    return (1 - x).sum()


# Different summary options available for pairwise feature vecs
SUM_OPS = {
    # 'len'    : len,
    'invsum' : invsum,
    # 'csum'   : csum,
    'sum'    : np.sum,
    'mean'   : np.mean,
    'std'    : np.std,
    'med' : np.median,
}


@ut.reloadable_class
class AnnotPairFeatInfo(object):
    """
    Information class about feature dimensions of PairwiseMatch.

    Notes:
        * Can be used to compute marginal importances over groups of features
        used in the pairwise one-vs-one scoring algorithm

        * Can be used to construct an appropriate cfgdict for a new
        PairwiseMatch.

    CommandLine:
        python -m vtool.matching AnnotPairFeatInfo

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> import vtool as vt
        >>> match = demodata_match({})
        >>> match.add_global_measures(['time', 'gps'])
        >>> index = pd.MultiIndex.from_tuples([(1, 2)], names=('aid1', 'aid2'))
        >>> # Feat info without bins
        >>> feat = match.make_feature_vector()
        >>> X = pd.DataFrame(feat, index=index)
        >>> print(X.keys())
        >>> featinfo = AnnotPairFeatInfo(X)
        >>> pairfeat_cfg, global_keys = featinfo.make_pairfeat_cfg()
        >>> print('pairfeat_cfg = %r' % (pairfeat_cfg,))
        >>> print('global_keys = %r' % (global_keys,))
        >>> assert 'delta' not in global_keys
        >>> assert 'max' not in global_keys
        >>> ut.cprint(featinfo.get_infostr(), 'blue')
        >>> # Feat info with bins
        >>> feat = match.make_feature_vector(indices=0, bins=[.7, .8], bin_key='ratio')
        >>> X = pd.DataFrame(feat, index=index)
        >>> print(X.keys())
        >>> featinfo = AnnotPairFeatInfo(X)
        >>> pairfeat_cfg, global_keys = featinfo.make_pairfeat_cfg()
        >>> print('pairfeat_cfg = %s' % (ut.repr4(pairfeat_cfg),))
        >>> print('global_keys = %r' % (global_keys,))
        >>> ut.cprint(featinfo.get_infostr(), 'blue')
    """
    def __init__(featinfo, columns=None, importances=None):
        featinfo.importances = importances
        if columns is not None:
            if hasattr(columns, 'columns'):
                featinfo.columns = columns.columns
            else:
                featinfo.columns = columns
        else:
            featinfo.columns = list(importances.keys())
        if importances is not None:
            assert isinstance(importances, dict), 'must be a dict'
        featinfo._summary_keys = sorted(SUM_OPS.keys()) + ['len']

    def make_pairfeat_cfg(featinfo):
        criteria = [('measure_type', '==', 'local')]
        indices = sorted(map(int, set(map(
            featinfo.local_rank, featinfo.select_columns(criteria)))))
        sorters = sorted(set(map(
            featinfo.local_sorter, featinfo.select_columns(criteria))))

        criteria = [('measure_type', '==', 'global')]
        global_measures = sorted(set(map(
            featinfo.global_measure, featinfo.select_columns(criteria))))

        global_keys = []
        for key in global_measures:
            parts = key.split('_')
            offset = parts[0] in {'max', 'min', 'delta'}
            global_keys.append(parts[offset])
        global_keys = sorted(set(global_keys))
        if 'speed' in global_keys:
            global_keys.remove('speed')  # hack

        summary_cols = featinfo.select_columns([
            ('measure_type', '==', 'summary')])
        summary_ops = sorted(set(map(featinfo.summary_op, summary_cols)))
        summary_measures = sorted(set(map(featinfo.summary_measure,
                                          summary_cols)))
        summary_binvals = sorted(set(map(featinfo.summary_binval,
                                         summary_cols)))
        summary_binvals = ut.filter_Nones(summary_binvals)
        summary_binkeys = sorted(set(map(featinfo.summary_binkey,
                                         summary_cols)))
        summary_binkeys = ut.filter_Nones(summary_binkeys)
        if 'matches' in summary_measures:
            summary_measures.remove('matches')

        if len(summary_binkeys) == 0:
            bin_key = None
        else:
            assert len(summary_binkeys) == 1
            bin_key = summary_binkeys[0]

        pairfeat_cfg = {
            'summary_ops': summary_ops,
            'local_keys': summary_measures,
            'sorters': sorters,
            'indices': indices,
            'bins': ut.lmap(float, summary_binvals),
            'bin_key': bin_key,
        }
        return pairfeat_cfg, global_keys

    def select_columns(featinfo, criteria, op='and'):
        """
        Args:
            criteria (list): list of tokens denoting selection constraints
               can be one of:
               measure_type global_measure, local_sorter, local_rank,
               local_measure, summary_measure, summary_op, summary_bin,
               summary_binval, summary_binkey,

        Examples:
            >>> featinfo.select_columns([
            >>>     ('measure_type', '==', 'local'),
            >>>     ('local_sorter', 'in', ['weighted_ratio_score', 'lnbnn_norm_dist']),
            >>> ], op='and')
        """
        if op == 'and':
            cols = set(featinfo.columns)
            update = cols.intersection_update
        elif op == 'or':
            cols = set([])
            update = cols.update
        else:
            raise Exception(op)
        for group_id, op, value in criteria:
            found = featinfo.find(group_id, op, value)
            update(found)
        return cols

    def find(featinfo, group_id, op, value, hack=False):
        """
        groupid options:
           measure_type global_measure, local_sorter, local_rank,
           local_measure, summary_measure, summary_op, summary_bin,
           summary_binval, summary_binkey,

        Ignore:
            group_id = 'summary_op'
            op = '=='
            value = 'len'
        """
        import six
        if isinstance(op, six.text_type):
            opdict = ut.get_comparison_operators()
            op = opdict.get(op)
        grouper = getattr(featinfo, group_id)
        found = []
        for col in featinfo.columns:
            value1 = grouper(col)
            if value1 is None:
                # TODO: turn hack off and ensure that doesn't break anything
                if hack:
                    # Only filter out/in comparable things
                    # why? Is this just a hack? I forgot why I wrote this.
                    found.append(col)
            else:
                try:
                    if value1 is not None:
                        if isinstance(value, int):
                            value1 = int(value1)
                        elif isinstance(value, float):
                            value1 = float(value1)
                        elif isinstance(value, list):
                            if len(value) > 0 and isinstance(value[0], int):
                                value1 = int(value1)
                    if op(value1, value):
                        found.append(col)
                except Exception:
                    pass
        return found

    def group_importance(featinfo, item):
        name, keys = item
        num = len(keys)
        weight = sum(ut.take(featinfo.importances, keys))
        ave_w = weight / num
        tup = ave_w, weight, num
        # return tup
        df = pd.DataFrame([tup], columns=['ave_w', 'weight', 'num'],
                          index=[name])
        return df

    def print_margins(featinfo, group_id, ignore_trivial=True):
        columns = featinfo.columns
        if isinstance(group_id, list):
            cols = featinfo.select_columns(criteria=group_id)
            _keys = [(c, [c]) for c in cols]
            try:
                _weights = pd.concat(ut.lmap(featinfo.group_importance, _keys))
            except ValueError:
                _weights = []
                pass
            nice = str(group_id)
        else:
            grouper = getattr(featinfo, group_id)
            _keys = ut.group_items(columns, ut.lmap(grouper, columns))
            _weights = pd.concat(ut.lmap(featinfo.group_importance, _keys.items()))
            nice = ut.get_funcname(grouper).replace('_', ' ')
            nice = ut.pluralize(nice)
        try:
            _weights = _weights.iloc[_weights['ave_w'].argsort()[::-1]]
        except Exception:
            pass
        if not ignore_trivial or len(_weights) > 1:
            ut.cprint('\nMarginal importance of ' + nice, 'white')
            print(_weights)

    def group_counts(featinfo, item):
        name, keys = item
        num = len(keys)
        tup = (num,)
        # return tup
        df = pd.DataFrame([tup], columns=['num'], index=[name])
        return df

    def print_counts(featinfo, group_id):
        columns = featinfo.columns
        grouper = getattr(featinfo, group_id)
        _keys = ut.group_items(columns, ut.lmap(grouper, columns))
        _weights = pd.concat(ut.lmap(featinfo.group_counts, _keys.items()))
        _weights = _weights.iloc[_weights['num'].argsort()[::-1]]
        nice = ut.get_funcname(grouper).replace('_', ' ')
        nice = ut.pluralize(nice)
        print('\nCounts of ' + nice)
        print(_weights)

    def feature(featinfo, key):
        return key

    def measure(featinfo, key):
        parsed = parse.parse('{type}({measure})', key)
        if parsed is None:
            parsed = parse.parse('{type}({measure}[{bin}])', key)
        if parsed is not None:
            return parsed['measure']

    def global_measure(featinfo, key):
        parsed = parse.parse('global({measure})', key)
        if parsed is not None:
            return parsed['measure']

    loc_fmt = 'loc[{sorter},{rank}]({measure})'

    def local_measure(featinfo, key):
        parsed = parse.parse(featinfo.loc_fmt, key)
        if parsed is not None:
            return parsed['measure']

    def local_sorter(featinfo, key):
        parsed = parse.parse(featinfo.loc_fmt, key)
        if parsed is not None:
            return parsed['sorter']

    def local_rank(featinfo, key):
        parsed = parse.parse(featinfo.loc_fmt, key)
        if parsed is not None:
            return parsed['rank']

    sum_fmt = '{op}({measure})'
    binsum_fmt = '{op}({measure}[{bin_key}<{binval}])'

    def summary_measure(featinfo, key):
        parsed = parse.parse(featinfo.binsum_fmt, key)
        if parsed is None:
            parsed = parse.parse(featinfo.sum_fmt, key)
        if parsed is not None:
            if parsed['op'] in featinfo._summary_keys:
                return parsed['measure']

    def summary_op(featinfo, key):
        parsed = parse.parse(featinfo.binsum_fmt, key)
        if parsed is None:
            parsed = parse.parse(featinfo.sum_fmt, key)
        if parsed is not None:
            if parsed['op'] in featinfo._summary_keys:
                return parsed['op']

    # def summary_bin(featinfo, key):
    #     parsed = parse.parse(featinfo.binsum_fmt, key)
    #     if parsed is not None:
    #         return parsed['bin_key'] + '<' + parsed['binval']

    def summary_binkey(featinfo, key):
        parsed = parse.parse(featinfo.binsum_fmt, key)
        if parsed is not None:
            return parsed['bin_key']

    def summary_binval(featinfo, key):
        parsed = parse.parse(featinfo.binsum_fmt, key)
        if parsed is not None:
            return parsed['binval']

    def dimkey_grammar(featinfo):
        """
        CommandLine:
            python -m vtool.matching AnnotPairFeatInfo.dimkey_grammar

        Example:
            >>> # ENABLE_DOCTEST
            >>> from vtool.matching import *  # NOQA
            >>> import vtool as vt
            >>> match = demodata_match({})
            >>> match.add_global_measures(['view', 'qual', 'gps', 'time'])
            >>> index = pd.MultiIndex.from_tuples([(1, 2)], names=('aid1', 'aid2'))
            >>> # Feat info without bins
            >>> feat = match.make_feature_vector()
            >>> X = pd.DataFrame(feat, index=index)
            >>> featinfo = AnnotPairFeatInfo(X)
            >>> feat_grammar = featinfo.dimkey_grammar()
            >>> for key in X.keys():
            >>>     print(key)
            >>>     print(feat_grammar.parseString(key))
        """
        # https://stackoverflow.com/questions/18706631/pyparsing-get-token-location-in-results-name
        # Here is a start of a grammer if we need to get serious
        import pyparsing as pp
        # locator = pp.Empty().setParseAction(lambda s, l, t: l)
        # with feature dimension name encoding
        # _summary_keys = ['sum', 'mean', 'med', 'std', 'len']
        _summary_keys = featinfo._summary_keys
        S = pp.Suppress
        class Nestings(object):
            """ allows for a bit of syntactic sugar """
            def __call__(self, x):
                return pp.Suppress('(') + x + pp.Suppress(')')
            def __getitem__(self, x):
                return pp.Suppress('[') + x + pp.Suppress(']')
        brak = paren = Nestings()
        unary_id = pp.Regex('[12]')
        unary_id = (unary_id)('unary_id')
        # takes care of non-greedy matching of underscores
        # http://stackoverflow.com/questions/1905278/keyword-matching-in-
        unary_measure = pp.Combine(
            pp.Word(pp.alphanums) +
            pp.ZeroOrMore('_' + ~unary_id + pp.Word(pp.alphanums)))
        unary_sub = brak[pp.Word(pp.nums)]
        global_unary = (unary_measure + S('_') + unary_id + pp.ZeroOrMore(unary_sub))
        global_unary = (global_unary)('global_unary')

        global_relation = pp.Word(pp.alphas + '_')
        global_relation = (global_relation)('global_relation')

        global_measure = global_unary | global_relation
        global_feature = ('global' + paren(global_measure))('global_feature')
        # Local
        local_measure = pp.Word(pp.alphas + '_')
        local_sorter = pp.Word(pp.alphas + '_')
        local_rank = pp.Word(pp.nums)
        local_rank = (local_rank)('local_rank')
        local_sorter = (local_sorter)('local_sorter')
        local_measure = (local_measure)('local_measure')
        local_feature = (
            'loc' + brak[local_sorter + S(',') + local_rank] +
            paren(local_measure)
        )('local_feature')
        # Summary
        summary_measure = pp.Word(pp.alphas + '_')('summary_measure')
        summary_binkey = pp.Word(pp.alphas + '_')('summary_binkey')
        summary_binval = pp.Word(pp.nums + '.')('summary_binval')

        summary_bin = brak[summary_binkey + '<' + summary_binval]('summary_bin')
        summary_op = pp.Or(_summary_keys)('summary_op')

        summary_feature = (
            summary_op + paren(summary_measure + pp.Optional(summary_bin))
        )('summary_feature')
        feat_grammar = local_feature | global_feature | summary_feature
        if False:
            z = global_feature.parseString('global(qual_1)')
            z = feat_grammar.parseString('global(min_qual)')
            z = feat_grammar.parseString('loc[ratio,1](norm_dist)')
            z = feat_grammar.parseString('mean(dist[ratio<1])')
            z
        return feat_grammar

    def measure_type(featinfo, key):
        if key.startswith('global'):
            return 'global'
        if key.startswith('loc'):
            return 'local'
        if any(key.startswith(p) for p in featinfo._summary_keys):
            # if '[b' in key:  # ]
            #     return 'binned_summary'
            # else:
            return 'summary'

    def get_infostr(featinfo):
        """
        Summarizes the types (global, local, summary) of features in X based on
        standardized dimension names.
        """
        grouped_keys = ut.ddict(list)
        for key in featinfo.columns:
            type_ = featinfo.measure_type(key)
            grouped_keys[type_].append(key)

        info_items = ut.odict([
            ('global_measures', ut.lmap(featinfo.global_measure,
                                        grouped_keys['global'])),

            ('local_sorters', set(map(featinfo.local_sorter,
                                       grouped_keys['local']))),
            ('local_ranks', set(map(featinfo.local_rank,
                                     grouped_keys['local']))),
            ('local_measures', set(map(featinfo.local_measure,
                                        grouped_keys['local']))),

            ('summary_measures', set(map(featinfo.summary_measure,
                                          grouped_keys['summary']))),
            ('summary_ops', set(map(featinfo.summary_op,
                                     grouped_keys['summary']))),
            # ('summary_bins', set(map(featinfo.summary_bin,
            #                          grouped_keys['summary']))),
            ('summary_binvals', set(map(featinfo.summary_binval,
                                        grouped_keys['summary']))),
            ('summary_binkeys', set(map(featinfo.summary_binkey,
                                        grouped_keys['summary']))),
        ])

        import textwrap
        def _wrap(list_):
            unwrapped = ', '.join(sorted(list_))
            indent = (' ' * 4)
            lines_ = textwrap.wrap(unwrapped, width=80 - len(indent))
            lines = ['    ' + line for line in lines_]
            return lines

        lines = []
        lines.append('Feature Dimensions: %d' % (len(featinfo.columns)))
        for item  in info_items.items():
            key, list_ = item
            list_ = {a for a in list_ if a is not None}
            if len(list_):
                title = key.replace('_', ' ').title()
                if key.endswith('_measures'):
                    groupid = key.replace('_measures', '')
                    num = len(grouped_keys[groupid])
                    title = title + ' (%d)' % (num,)
                lines.append(title + ':')
                if key == 'summary_measures':
                    other = info_items['local_measures']
                    if other.issubset(list_) and len(other) > 0:
                        remain = list_ - other
                        lines.extend(_wrap(['<same as local_measures>'] + list(remain)))
                    else:
                        lines.extend(_wrap(list_))
                else:
                    lines.extend(_wrap(list_))

        infostr = '\n'.join(lines)
        return infostr
        # print(infostr)


def testdata_annot_metadata(rchip_fpath, cfgdict={}):
    metadata = ut.LazyDict({'rchip_fpath': rchip_fpath})
    ensure_metadata_feats(metadata, '', cfgdict)
    return metadata


def ensure_metadata_vsone(annot1, annot2, cfgdict={}):
    ensure_metadata_feats(annot1, cfgdict=cfgdict)
    ensure_metadata_feats(annot2, cfgdict=cfgdict)

    symmetric, = PairwiseMatch._take_params(cfgdict, ['symmetric'])

    ensure_metadata_flann(annot1, cfgdict=cfgdict)

    if symmetric:
        ensure_metadata_flann(annot2, cfgdict=cfgdict)
    ensure_metadata_dlen_sqrd(annot2)
    pass


def ensure_metadata_normxy(annot, cfgdict={}):
    import vtool as vt
    if 'norm_xys' not in annot:
        def eval_normxy():
            xys = vt.get_xys(annot['kpts'])
            chip_wh = np.array(annot['chip_size'])[:, None]
            return xys / chip_wh
        annot.set_lazy_func('norm_xys', eval_normxy)


def ensure_metadata_feats(annot, cfgdict={}):
    r"""
    Adds feature evaluation keys to a lazy dictionary

    Args:
        annot (utool.LazyDict):
        suffix (str): (default = '')
        cfgdict (dict): (default = {})

    CommandLine:
        python -m vtool.matching --exec-ensure_metadata_feats

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> rchip_fpath = ut.grab_test_imgpath('easy1.png')
        >>> annot = ut.LazyDict({'rchip_fpath': rchip_fpath})
        >>> cfgdict = {}
        >>> ensure_metadata_feats(annot, cfgdict)
        >>> assert len(annot._stored_results) == 1
        >>> annot['kpts']
        >>> assert len(annot._stored_results) == 4
        >>> annot['vecs']
        >>> assert len(annot._stored_results) == 5
    """
    import vtool as vt
    rchip_key = 'rchip'
    nchip_key = 'nchip'
    _feats_key = '_feats'
    kpts_key = 'kpts'
    vecs_key = 'vecs'
    rchip_fpath_key = 'rchip_fpath'

    if rchip_key not in annot:
        def eval_rchip():
            rchip_fpath = annot[rchip_fpath_key]
            return vt.imread(rchip_fpath)
        annot.set_lazy_func(rchip_key, eval_rchip)

    if nchip_key not in annot:
        def eval_normchip():
            print('EVAL NORMCHIP')
            # Hack in normalization (hack because rchip might already
            # be normalized)
            filter_list = []
            config = {
                pi.varname: (cfgdict[pi.varname] if pi.varname in cfgdict else
                             pi.default)
                for pi in NORM_CHIP_CONFIG
            }
            # new way
            if config['histeq']:
                filter_list.append(
                    ('histeq', {})
                )
            if config['medianblur']:
                filter_list.append(
                    ('medianblur', {
                        'noise_thresh': config['medianblur_thresh'],
                        'ksize1': config['medianblur_ksize1'],
                        'ksize2': config['medianblur_ksize2'],
                    }))
            if config['adapteq']:
                ksize = config['adapteq_ksize']
                filter_list.append(
                    ('adapteq', {
                        'tileGridSize': (ksize, ksize),
                        'clipLimit': config['adapteq_limit'],
                    })
                )
            rchip = annot[rchip_key]
            if filter_list:
                from vtool import image_filters
                ipreproc = image_filters.IntensityPreproc()
                nchip = ipreproc.preprocess(rchip, filter_list)
            else:
                nchip = rchip
            return nchip
        annot.set_lazy_func(nchip_key, eval_normchip)

    if kpts_key not in annot or vecs_key not in annot:
        def eval_feats():
            rchip = annot[nchip_key]
            feat_cfgkeys = [pi.varname for pi in VSONE_FEAT_CONFIG]
            feat_cfgdict = {key: cfgdict[key] for key in feat_cfgkeys if
                            key in cfgdict}
            _feats = vt.extract_features(rchip, **feat_cfgdict)
            return _feats

        def eval_kpts():
            _feats = annot[_feats_key]
            kpts = _feats[0]
            return kpts

        def eval_vecs():
            _feats = annot[_feats_key]
            vecs = _feats[1]
            return vecs
        annot.set_lazy_func(_feats_key, eval_feats)
        annot.set_lazy_func(kpts_key, eval_kpts)
        annot.set_lazy_func(vecs_key, eval_vecs)
    return annot


def ensure_metadata_dlen_sqrd(annot):
    if 'dlen_sqrd' not in annot:
        def eval_dlen_sqrd(annot):
            rchip = annot['rchip']
            dlen_sqrd = rchip.shape[0] ** 2 + rchip.shape[1] ** 2
            return dlen_sqrd
        annot.set_lazy_func('dlen_sqrd', lambda: eval_dlen_sqrd(annot))
    return annot


def ensure_metadata_flann(annot, cfgdict):
    """ setup lazy flann evaluation """
    import vtool as vt
    flann_params = {'algorithm': 'kdtree', 'trees': 8}
    if 'flann' not in annot:
        def eval_flann():
            vecs = annot['vecs']
            if len(vecs) == 0:
                _flann = None
            else:
                _flann = vt.flann_cache(vecs, flann_params=flann_params,
                                        verbose=False)
            return _flann
        annot.set_lazy_func('flann', eval_flann)
    return annot


def empty_neighbors(num_vecs=0, K=0):
    shape = (num_vecs, K)
    fx2_to_fx1 = np.empty(shape, dtype=np.int32)
    _fx2_to_dist_sqrd = np.empty(shape, dtype=np.float64)
    return fx2_to_fx1, _fx2_to_dist_sqrd


# maximum SIFT matching distance based using uint8 trick from hesaff
PSEUDO_MAX_VEC_COMPONENT = 512
PSEUDO_MAX_DIST_SQRD = 2 * (PSEUDO_MAX_VEC_COMPONENT ** 2)
PSEUDO_MAX_DIST = np.sqrt(2) * (PSEUDO_MAX_VEC_COMPONENT)


def empty_assign():
    fm = np.empty((0, 2), dtype=np.int32)
    match_dist = np.array([])
    norm_dist = np.array([])
    fx1_norm = np.array([], dtype=np.int32)
    fx2_norm = np.array([], dtype=np.int32)
    return fm, match_dist, norm_dist, fx1_norm, fx2_norm


def symmetric_correspondence(annot1, annot2, K, Knorm, checks, allow_shrink=True):
    """
    Find symmetric feature corresopndences
    """
    if allow_shrink:
        # Reduce K to allow some correspondences to be established
        n_have = min(len(annot1['vecs']), len(annot2['vecs']))
        if n_have < 2:
            return empty_assign()
        elif n_have < K + Knorm:
            K, Knorm = n_have - 1, 1

    num_neighbors = K + Knorm

    fx1_to_fx2, fx1_to_dist = normalized_nearest_neighbors(
        annot2['flann'], annot1['vecs'], num_neighbors, checks)

    fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(
        annot1['flann'], annot2['vecs'], num_neighbors, checks)

    # fx2_to_flags = flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2, K)

    assigntup2 = assign_symmetric_matches(
        fx2_to_fx1, fx2_to_dist, fx1_to_fx2, fx1_to_dist, K, Knorm)

    (fm, match_dist, fx1_norm, norm_dist1, fx2_norm,
     norm_dist2) = assigntup2
    norm_dist = np.minimum(norm_dist1, norm_dist2)

    return fm, match_dist, norm_dist, fx1_norm, fx2_norm


def asymmetric_correspondence(annot1, annot2, K, Knorm, checks, allow_shrink=True):
    """
    Find symmetric feature corresopndences
    """
    if allow_shrink:
        # Reduce K to allow some correspondences to be established
        n_have = len(annot1['vecs'])
        if n_have < 2:
            return empty_assign()
        elif n_have < K + Knorm:
            K, Knorm = n_have - 1, 1

    num_neighbors = K + Knorm

    fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(
        annot1['flann'], annot2['vecs'], num_neighbors, checks)
    fx2_to_flags = np.ones((len(fx2_to_fx1), K), dtype=np.bool)
    # Assign correspondences
    assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist, K,
                                             Knorm, fx2_to_flags)
    fm, match_dist, fx1_norm, norm_dist = assigntup
    fx2_norm = None

    return fm, match_dist, norm_dist, fx1_norm, fx2_norm


def normalized_nearest_neighbors(flann1, vecs2, K, checks=800):
    """
    Computes matches from vecs2 to flann1.

    uses flann index to return nearest neighbors with distances normalized
    between 0 and 1 using sifts uint8 trick
    """
    import vtool as vt
    if K == 0:
        (fx2_to_fx1, _fx2_to_dist_sqrd) = empty_neighbors(len(vecs2), 0)
    elif len(vecs2) == 0:
        (fx2_to_fx1, _fx2_to_dist_sqrd) = empty_neighbors(0, K)
    elif flann1 is None:
        (fx2_to_fx1, _fx2_to_dist_sqrd) = empty_neighbors(0, 0)
    elif K > flann1.get_indexed_shape()[0]:
        # Corner case, may be better to throw an assertion error
        raise MatchingError('not enough database features')
        #(fx2_to_fx1, _fx2_to_dist_sqrd) = empty_neighbors(len(vecs2), 0)
    else:
        fx2_to_fx1, _fx2_to_dist_sqrd = flann1.nn_index(
            vecs2, num_neighbors=K, checks=checks)
    _fx2_to_dist = np.sqrt(_fx2_to_dist_sqrd.astype(np.float64))
    # normalized SIFT dist
    fx2_to_dist = np.divide(_fx2_to_dist, PSEUDO_MAX_DIST)
    fx2_to_fx1 = vt.atleast_nd(fx2_to_fx1, 2)
    fx2_to_dist = vt.atleast_nd(fx2_to_dist, 2)
    return fx2_to_fx1, fx2_to_dist


def assign_symmetric_matches(fx2_to_fx1, fx2_to_dist, fx1_to_fx2, fx1_to_dist,
                             K, Knorm=None):
    r"""
    import vtool as vt
    from vtool.matching import *
    K = 2
    Knorm = 1
    feat1 = np.random.rand(5, 3)
    feat2 = np.random.rand(7, 3)

    # Assign distances
    distmat = vt.L2(feat1[:, None], feat2[None, :])

    # Find nearest K
    fx1_to_fx2 = distmat.argsort()[:, 0:K + Knorm]
    fx2_to_fx1 = distmat.T.argsort()[:, 0:K + Knorm]
    # and order their distances
    fx1_to_dist = np.array([distmat[i].take(col) for i, col in enumerate(fx1_to_fx2)])
    fx2_to_dist = np.array([distmat.T[j].take(row) for j, row in enumerate(fx2_to_fx1)])

    # flat_matx1 = fx1_to_fx2 + np.arange(distmat.shape[0])[:, None] * distmat.shape[1]
    # fx1_to_dist = distmat.take(flat_matx1).reshape(fx1_to_fx2.shape)

    fx21 = pd.DataFrame(fx2_to_fx1)
    fx21.columns.name = 'K'
    fx21.index.name = 'fx1'

    fx12 = pd.DataFrame(fx1_to_fx2)
    fx12.columns.name = 'K'
    fx12.index.name = 'fx2'

    fx12 = fx12.T[0:K].T.astype(np.float)
    fx21 = fx21.T[0:K].T.astype(np.float)

    fx12.values[~fx1_to_flags] = np.nan
    fx21.values[~fx2_to_flags] = np.nan

    print('fx12.values =\n%r' % (fx12,))
    print('fm_ =\n%r' % (fm_,))

    print('fx21.values =\n%r' % (fx21,))
    print('fm =\n%r' % (fm,))

    unflat_match_idx2 = -np.ones(fx2_to_fx1.shape)
    unflat_match_idx2.ravel()[flat_match_idx2] = flat_match_idx2
    inv_lookup21 = unflat_match_idx2.T[0:K].T

    for fx2 in zip(fx12.values[fx1_to_flags]:

    for fx1, fx2 in zip(match_fx1_, match_fx2_):
        cx = np.where(fx2_to_fx1[fx2][0:K] == fx1)[0][0]
        inv_idx = inv_lookup21[fx2][cx]
        print('inv_idx = %r' % (inv_idx,))
    """
    # Infer the valid internal query feature indexes and ranks
    if Knorm is None:
        basic_norm_rank = -1
    else:
        basic_norm_rank = K + Knorm - 1

    index_dtype = fx2_to_fx1.dtype

    fx2_to_flags = flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2, K)
    flat_validx2 = np.flatnonzero(fx2_to_flags)
    match_fx2   = np.floor_divide(flat_validx2, K, dtype=index_dtype)
    match_rank2 = np.mod(flat_validx2, K, dtype=index_dtype)
    flat_match_idx2 = np.ravel_multi_index((match_fx2, match_rank2),
                                           dims=fx2_to_fx1.shape)
    match_fx1 = fx2_to_fx1.take(flat_match_idx2)
    match_dist1 = fx2_to_dist.take(flat_match_idx2)

    # assert np.all(match_dist1 == fx2_to_dist[:, 0:K][fx2_to_flags])

    fm = np.vstack((match_fx1, match_fx2)).T

    # TODO: for each position in fm need to find corresponding position in fm_

    # Currently just use the last one as a normalizer
    norm_rank = np.array([basic_norm_rank] * len(match_fx2),
                         dtype=match_fx2.dtype)
    # norm_rank = basic_norm_rank
    flat_norm_idx1 = np.ravel_multi_index((match_fx2, norm_rank),
                                          dims=fx2_to_fx1.shape)
    norm_fx1 = fx2_to_fx1.take(flat_norm_idx1)
    norm_dist1 = fx2_to_dist.take(flat_norm_idx1)
    norm_fx1 = fx2_to_fx1[match_fx2, norm_rank]
    norm_dist1 = fx2_to_dist[match_fx2, norm_rank]

    # ---------
    # REVERSE DIRECTION
    fx1_to_flags = flag_symmetric_matches(fx1_to_fx2, fx2_to_fx1, K)
    flat_validx1 = np.flatnonzero(fx1_to_flags)
    match_fx1_   = np.floor_divide(flat_validx1, K, dtype=index_dtype)
    match_rank1 = np.mod(flat_validx1, K, dtype=index_dtype)
    flat_match_idx1 = np.ravel_multi_index((match_fx1_, match_rank1),
                                           dims=fx1_to_fx2.shape)
    match_fx2_ = fx1_to_fx2.take(flat_match_idx1)
    # match_dist2_ = fx1_to_dist.take(flat_match_idx1)
    fm_ = np.vstack((match_fx1_, match_fx2_)).T

    flat_norm_idx2_ = np.ravel_multi_index((match_fx1_, norm_rank),
                                           dims=fx1_to_fx2.shape)
    norm_fx2_ = fx1_to_fx2.take(flat_norm_idx2_)
    norm_dist2_ = fx1_to_dist.take(flat_norm_idx2_)

    # ---------
    # Align matches with the reverse direction
    # lookup = {(fx1, fx2): mx for mx, (fx1, fx2) in enumerate(fm)}
    # idx_lookup = np.array([lookup[(i, j)] for i, j in fm_])
    # idx_lookup_ = idx_lookup.argsort()

    # if False:
    #     assert fx1_to_flags.sum() == fx2_to_flags.sum()
    #     assert np.all(fm[idx_lookup] == fm_)
    #     assert np.all(fm == fm_[idx_lookup_])

    #     assert match_dist2.sum() == match_dist1.sum()
    #     assert set(ut.emap(frozenset, fm_)) == set(ut.emap(frozenset, fm))
    # norm_fx2 = norm_fx2_[idx_lookup_]
    # norm_dist2 = norm_dist2_[idx_lookup_]
    # norm_dist1

    # Do this by enforcing a constant sorting. No lookup necessary
    sortx = np.lexsort(fm.T)
    sortx_ = np.lexsort(fm_.T)
    # fm2_ = fm_.take(sortx_, axis=0)
    # assert np.all(fm2 == fm2_)

    a_fm = fm.take(sortx, axis=0)

    a_match_dist = match_dist1[sortx]
    # assert np.all(match_dist1[sortx] == match_dist2_[sortx_])

    a_norm_fx1 = norm_fx1[sortx]
    a_norm_dist1 = norm_dist1[sortx]

    a_norm_fx2_ = norm_fx2_[sortx_]
    a_norm_dist2_ = norm_dist2_[sortx_]

    np.minimum(a_norm_dist2_, a_norm_dist1)

    assigntup = (a_fm, a_match_dist, a_norm_fx1, a_norm_dist1, a_norm_fx2_,
                 a_norm_dist2_)
    return assigntup


def assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist, K, Knorm=None,
                                 fx2_to_flags=None):
    """
    assigns vsone matches using results of nearest neighbors.

    Ignore:
        fx2_to_dist = np.arange(fx2_to_fx1.size).reshape(fx2_to_fx1.shape)

    CommandLine:
        python -m vtool.matching --test-assign_unconstrained_matches --show
        python -m vtool.matching assign_unconstrained_matches:0
        python -m vtool.matching assign_unconstrained_matches:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> fx2_to_fx1, fx2_to_dist = empty_neighbors(0, 0)
        >>> K = 1
        >>> Knorm = 1
        >>> fx2_to_flags = None
        >>> assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist, K,
        >>>                                          Knorm, fx2_to_flags)
        >>> fm, match_dist, norm_fx1, norm_dist = assigntup
        >>> result = ut.repr4(assigntup, precision=3, nobr=True, with_dtype=True)
        >>> print(result)
        np.array([], shape=(0, 2), dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> fx2_to_fx1 = np.array([[ 77,   971, 22],
        >>>                        [116,   120, 34],
        >>>                        [122,   128, 99],
        >>>                        [1075,  692, 102],
        >>>                        [ 530,   45, 120],
        >>>                        [  45,  530, 77]], dtype=np.int32)
        >>> fx2_to_dist = np.array([[ 0.059,  0.238, .3],
        >>>                         [ 0.021,  0.240, .4],
        >>>                         [ 0.039,  0.247, .5],
        >>>                         [ 0.149,  0.151, .6],
        >>>                         [ 0.226,  0.244, .7],
        >>>                         [ 0.215,  0.236, .8]], dtype=np.float32)
        >>> K = 1
        >>> Knorm = 1
        >>> fx2_to_flags = np.array([[1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [1, 1]])
        >>> fx2_to_flags = fx2_to_flags[:, 0:K]
        >>> assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist, K,
        >>>                                          Knorm, fx2_to_flags)
        >>> fm, match_dist, norm_fx1, norm_dist = assigntup
        >>> result = ut.repr3(assigntup, precision=3, nobr=True, with_dtype=True)
        >>> print(result)
        >>> assert len(fm.shape) == 2 and fm.shape[1] == 2
        >>> assert ut.allsame(list(map(len, assigntup)))
    """
    # Infer the valid internal query feature indexes and ranks
    index_dtype = fx2_to_fx1.dtype

    if fx2_to_flags is None:
        # make everything valid
        flat_validx = np.arange(len(fx2_to_fx1) * K, dtype=index_dtype)
    else:
        #fx2_to_flags = np.ones((len(fx2_to_fx1), K), dtype=np.bool)
        flat_validx = np.flatnonzero(fx2_to_flags)

    match_fx2  = np.floor_divide(flat_validx, K, dtype=index_dtype)
    match_rank = np.mod(flat_validx, K, dtype=index_dtype)

    flat_match_idx = np.ravel_multi_index((match_fx2, match_rank),
                                          dims=fx2_to_fx1.shape)
    match_fx1 = fx2_to_fx1.take(flat_match_idx)
    match_dist = fx2_to_dist.take(flat_match_idx)

    fm = np.vstack((match_fx1, match_fx2)).T

    if Knorm is None:
        basic_norm_rank = -1
    else:
        basic_norm_rank = K + Knorm - 1

    # Currently just use the last one as a normalizer
    norm_rank = np.array([basic_norm_rank] * len(match_fx2),
                         dtype=match_fx2.dtype)
    flat_norm_idx = np.ravel_multi_index((match_fx2, norm_rank),
                                         dims=fx2_to_fx1.shape)
    norm_fx1 = fx2_to_fx1.take(flat_norm_idx)
    norm_dist = fx2_to_dist.take(flat_norm_idx)
    norm_fx1 = fx2_to_fx1[match_fx2, norm_rank]
    norm_dist = fx2_to_dist[match_fx2, norm_rank]

    assigntup = AssignTup(fm, match_dist, norm_fx1, norm_dist)
    return assigntup


def flag_sym_slow(fx1_to_fx2, fx2_to_fx1, K):
    """
    Returns flags indicating if the matches in fx1_to_fx2 are reciprocal
    with the matches in fx2_to_fx1.

    Much slower version of flag_symmetric_matches, but more clear
    """
    fx1_to_flags = []
    for fx1 in range(len(fx1_to_fx2)):
        # Find img2 features in fx1's top K
        fx2_m = fx1_to_fx2[fx1, :K]
        # Find img2 features that have fx1 in their top K
        reverse_m = fx2_to_fx1[fx2_m, :K]
        is_recip = (reverse_m == fx1).sum(axis=1).astype(np.bool)
        fx1_to_flags.append(is_recip)
    fx1_to_flags = np.array(fx1_to_flags)
    return fx1_to_flags


def flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2, K=2):
    """
    Returns flags indicating if the matches in fx2_to_fx1 are reciprocal
    with the matches in fx1_to_fx2.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.matching import *  # NOQA
        >>> K = 2
        >>> fx2_to_fx1 = np.array([[ 0,  1], # 0
        >>>                        [ 1,  4], # 1
        >>>                        [ 3,  4], # 2
        >>>                        [ 2,  3]], dtype=np.int32) # 3
        >>> fx1_to_fx2 = np.array([[ 0, 1], # 0
        >>>                        [ 2, 1], # 1
        >>>                        [ 0, 1], # 2
        >>>                        [ 3, 1], # 3
        >>>                        [ 0, 1]], dtype=np.int32) # 4
        >>> fx2_to_flagsA = flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2, K)
        >>> fx2_to_flagsB = flag_sym_slow(fx2_to_fx1, fx1_to_fx2, K)
        >>> assert np.all(fx2_to_flagsA == fx2_to_flagsB)
        >>> result = ut.repr2(fx2_to_flagsB)
        >>> print(result)
        np.array([[ True, False],
                  [ True,  True],
                  [False, False],
                  [False,  True]])

    Ignore:
        %timeit flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2, K)
        %timeit flag_sym_slow(fx2_to_fx1, fx1_to_fx2, K)
    """
    match_12 = fx1_to_fx2.T[:K].T
    match_21 = fx2_to_fx1.T[:K].T
    fx2_list = np.arange(len(match_21))
    # Lookup the reciprocal neighbors of each img2 feature.
    matched = match_12[match_21.ravel()]
    # Group the reciprocal matches such that
    # matches[fx2, k, i] is the i-th recip neighbor of the k-th neighbor of fx
    matched = matched.reshape((len(fx2_to_fx1), K, K))
    # If a reciprocal neighbor is itself, then the feature is good
    flags = (matched == fx2_list[:, None, None])
    fx2_to_flags = np.any(flags, axis=2)
    return fx2_to_flags


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/vtool/vtool/matching.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
