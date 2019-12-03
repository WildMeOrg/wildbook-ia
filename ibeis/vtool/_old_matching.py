# def vsone_matching(metadata, cfgdict={}, verbose=None):
#     """
#     DEPRICATE in favor of PairwiseMatch

#     Metadata is a dictionary that contains either computed information
#     necessary for matching or the dependenceis of those computations.

#     Args:
#         metadata (utool.LazyDict):
#         cfgdict (dict): (default = {})
#         verbose (bool):  verbosity flag(default = None)

#     Returns:
#         tuple: (matches, metadata)
#     """
#     # import vtool as vt
#     #assert isinstance(metadata, ut.LazyDict), 'type(metadata)=%r' % (type(metadata),)

#     annot1 = metadata['annot1']
#     annot2 = metadata['annot2']

#     ensure_metadata_feats(annot1, cfgdict=cfgdict)
#     ensure_metadata_feats(annot2, cfgdict=cfgdict)
#     ensure_metadata_dlen_sqrd(annot2)

#     # Exceute relevant dependencies
#     kpts1 = annot1['kpts']
#     vecs1 = annot1['vecs']
#     kpts2 = annot2['kpts']
#     vecs2 = annot2['vecs']
#     dlen_sqrd2 = annot2['dlen_sqrd']
#     flann1 = annot1.get('flann', None)
#     flann2 = annot2.get('flann', None)

#     matches, output_metdata = vsone_feature_matching(
#         kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict=cfgdict,
#         flann1=flann1, flann2=flann2, verbose=verbose)
#     metadata.update(output_metdata)
#     match = SingleMatch(matches, metadata)
#     return match


# def vsone_feature_matching(kpts1, vecs1, kpts2, vecs2, dlen_sqrd2, cfgdict={},
#                            flann1=None, flann2=None, verbose=None):
#     r"""
#     DEPRICATE

#     logic for matching

#     Args:
#         vecs1 (ndarray[uint8_t, ndim=2]): SIFT descriptors
#         vecs2 (ndarray[uint8_t, ndim=2]): SIFT descriptors
#         kpts1 (ndarray[float32_t, ndim=2]):  keypoints
#         kpts2 (ndarray[float32_t, ndim=2]):  keypoints

#     Ignore:
#         >>> from vtool.matching import *  # NOQA
#         >>> ut.qtensure()
#         >>> import plottool as pt
#         >>> pt.imshow(rchip1)
#         >>> pt.draw_kpts2(kpts1)
#         >>> pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
#         >>> pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
#     """
#     import vtool as vt
#     #import vtool as vt
#     sv_on = cfgdict.get('sv_on', True)
#     sver_xy_thresh = cfgdict.get('sver_xy_thresh', .01)
#     ratio_thresh   = cfgdict.get('ratio_thresh', .625)
#     refine_method  = cfgdict.get('refine_method', 'homog')
#     symmetric      = cfgdict.get('symmetric', False)
#     K              = cfgdict.get('K', 1)
#     Knorm          = cfgdict.get('Knorm', 1)
#     checks = cfgdict.get('checks', 800)
#     if verbose is None:
#         verbose = True

#     flann_params = {'algorithm': 'kdtree', 'trees': 8}
#     if flann1 is None:
#         flann1 = vt.flann_cache(vecs1, flann_params=flann_params,
#                                 verbose=verbose)
#     if symmetric:
#         if flann2 is None:
#             flann2 = vt.flann_cache(vecs2, flann_params=flann_params,
#                                     verbose=verbose)
#     try:
#         num_neighbors = K + Knorm
#         # Search for nearest neighbors
#         fx2_to_fx1, fx2_to_dist = normalized_nearest_neighbors(
#             flann1, vecs2, num_neighbors, checks)
#         if symmetric:
#             fx1_to_fx2, fx1_to_dist = normalized_nearest_neighbors(
#                 flann2, vecs1, K, checks)

#         if symmetric:
#             valid_flags = flag_symmetric_matches(fx2_to_fx1, fx1_to_fx2, K)
#         else:
#             valid_flags = np.ones((len(fx2_to_fx1), K), dtype=np.bool)

#         # Assign matches
#         assigntup = assign_unconstrained_matches(fx2_to_fx1, fx2_to_dist, K,
#                                                  Knorm, valid_flags)
#         fm, match_dist, fx1_norm, norm_dist = assigntup
#         fs = 1 - np.divide(match_dist, norm_dist)

#         fm_ORIG = fm
#         fs_ORIG = fs

#         ratio_on = sv_on
#         if ratio_on:
#             # APPLY RATIO TEST
#             fm, fs, fm_norm = ratio_test(fm_ORIG, fx1_norm, match_dist, norm_dist,
#                                          ratio_thresh)
#             fm_RAT, fs_RAT, fm_norm_RAT = (fm, fs, fm_norm)

#         if sv_on:
#             fm, fs, fm_norm, H_RAT = match_spatial_verification(
#                 kpts1, kpts2, fm, fs, fm_norm, sver_xy_thresh, dlen_sqrd2,
#                 refine_method)
#             fm_RAT_SV, fs_RAT_SV, fm_norm_RAT_SV = (fm, fs, fm_norm)

#         #top_percent = .5
#         #top_idx = ut.take_percentile(match_dist.T[0].argsort(), top_percent)
#         #fm_TOP = fm_ORIG.take(top_idx, axis=0)
#         #fs_TOP = match_dist.T[0].take(top_idx)
#         #match_weights = 1 - fs_TOP
#         #svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm_TOP, sver_xy_thresh,
#         #                                   dlen_sqrd2, match_weights=match_weights,
#         #                                   refine_method=refine_method)
#         #if svtup is not None:
#         #    (homog_inliers, homog_errors, H_TOP) = svtup[0:3]
#         #    np.sqrt(homog_errors[0] / dlen_sqrd2)
#         #else:
#         #    H_TOP = np.eye(3)
#         #    homog_inliers = []
#         #fm_TOP_SV = fm_TOP.take(homog_inliers, axis=0)
#         #fs_TOP_SV = fs_TOP.take(homog_inliers, axis=0)

#         matches = {
#             'ORIG'   : MatchTup2(fm_ORIG, fs_ORIG),
#         }
#         output_metdata = {}
#         if ratio_on:
#             matches['RAT'] = MatchTup3(fm_RAT, fs_RAT, fm_norm_RAT)
#         if sv_on:
#             matches['RAT+SV'] = MatchTup3(fm_RAT_SV, fs_RAT_SV, fm_norm_RAT_SV)
#             output_metdata['H_RAT'] = H_RAT
#             #output_metdata['H_TOP'] = H_TOP
#             #'TOP'    : MatchTup2(fm_TOP, fs_TOP),
#             #'TOP+SV' : MatchTup2(fm_TOP_SV, fs_TOP_SV),

#     except MatchingError:
#         fm_ERR = np.empty((0, 2), dtype=np.int32)
#         fs_ERR = np.empty((0, 1), dtype=np.float32)
#         H_ERR = np.eye(3)
#         matches = {
#             'ORIG'   : MatchTup2(fm_ERR, fs_ERR),
#             'RAT'    : MatchTup3(fm_ERR, fs_ERR, fm_ERR),
#             'RAT+SV' : MatchTup3(fm_ERR, fs_ERR, fm_ERR),
#             #'TOP'    : MatchTup2(fm_ERR, fs_ERR),
#             #'TOP+SV' : MatchTup2(fm_ERR, fs_ERR),
#         }
#         output_metdata = {
#             'H_RAT': H_ERR,
#             #'H_TOP': H_ERR,
#         }

#     return matches, output_metdata


# def match_spatial_verification(kpts1, kpts2, fm, fs, fm_norm, sver_xy_thresh,
#                                dlen_sqrd2, refine_method):
#     from vtool import spatial_verification as sver
#     # SPATIAL VERIFICATION FILTER
#     match_weights = np.ones(len(fm))
#     svtup = sver.spatially_verify_kpts(kpts1, kpts2, fm,
#                                        xy_thresh=sver_xy_thresh,
#                                        ori_thresh=ori_thresh,
#                                        dlen_sqrd2=dlen_sqrd2,
#                                        match_weights=match_weights,
#                                        refine_method=refine_method)
#     if svtup is not None:
#         (homog_inliers, homog_errors, H_RAT) = svtup[0:3]
#     else:
#         H_RAT = np.eye(3)
#         homog_inliers = []
#     fm_SV = fm.take(homog_inliers, axis=0)
#     fs_SV = fs.take(homog_inliers, axis=0)
#     fm_norm_SV = fm_norm[homog_inliers]
#     return fm_SV, fs_SV, fm_norm_SV, H_RAT

# class SingleMatch(ut.NiceRepr):
#     """
#     DEPRICATE in favor of PairwiseMatch
#     """

#     def __init__(self, matches, metadata):
#         self.matches = matches
#         self.metadata = metadata

#     def show(self, *args, **kwargs):
#         from vtool import inspect_matches
#         inspect_matches.show_matching_dict(
#             self.matches, self.metadata, *args, **kwargs)

#     def make_interaction(self, *args, **kwargs):
#         from vtool import inspect_matches
#         return inspect_matches.make_match_interaction(
#             self.matches, self.metadata, *args, **kwargs)

#     def __nice__(self):
#         parts = [key + '=%d' % (len(m.fm)) for key, m in self.matches.items()]
#         return ' ' + ', '.join(parts)

#     def __getstate__(self):
#         state_dict = self.__dict__
#         return state_dict

#     def __setstate__(self, state_dict):
#         self.__dict__.update(state_dict)


# def vsone_image_fpath_matching(rchip_fpath1, rchip_fpath2, cfgdict={}, metadata_=None):
#     r"""
#     Args:
#         rchip_fpath1 (str):
#         rchip_fpath2 (str):
#         cfgdict (dict): (default = {})

#     CommandLine:
#         python -m vtool --tf vsone_image_fpath_matching --show
#         python -m vtool --tf vsone_image_fpath_matching --show --helpx
#         python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+siam128
#         python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+siam128 --ratio-thresh=.9
#         python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+sift --ratio-thresh=.8
#         python -m vtool --tf vsone_image_fpath_matching --show --feat-type=hesaff+sift --ratio-thresh=.8

#     Example:
#         >>> # ENABLE_DOCTEST
#         >>> from vtool.matching import *  # NOQA
#         >>> import vtool as vt
#         >>> rchip_fpath1 = ut.grab_test_imgpath('easy1.png')
#         >>> rchip_fpath2 = ut.grab_test_imgpath('easy2.png')
#         >>> import pyhesaff
#         >>> metadata_ = None
#         >>> default_cfgdict = dict(feat_type='hesaff+sift', ratio_thresh=.625,
#         >>>                        **pyhesaff.get_hesaff_default_params())
#         >>> cfgdict = ut.parse_dict_from_argv(default_cfgdict)
#         >>> match = vsone_image_fpath_matching(rchip_fpath1, rchip_fpath2, cfgdict)
#         >>> # xdoctest: +REQUIRES(--show)
#         >>> match.show(mode=1)
#         >>> ut.show_if_requested()
#     """
#     metadata = ut.LazyDict()
#     annot1 = metadata['annot1'] = ut.LazyDict()
#     annot2 = metadata['annot2'] = ut.LazyDict()
#     if metadata_ is not None:
#         metadata.update(metadata_)
#     annot1['rchip_fpath'] = rchip_fpath1
#     annot2['rchip_fpath'] = rchip_fpath2
#     match =  vsone_matching(metadata, cfgdict)
#     return match
