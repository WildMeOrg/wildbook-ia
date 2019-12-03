# -*- coding: utf-8 -*-
"""
VTool - Computer vision tools
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
__version__ = '2.0.0'

# import utool as ut
# ut.noinject(__name__, '[vtool]')

"""
mkinit vtool
python -c "import vtool" --dump-vtool-init
python -c "import vtool" --update-vtool-init
"""


# TODO finish utoolifying this this
IMPORT_TUPLES = [
    ('image', None),
    ('histogram', None),
    ('image', None),
    ('exif', None),
    ('keypoint', None),
    ('features', None),
    ('linalg', None),
    ('patch', None),
    ('chip', None),
    ('spatial_verification', None),
    ('trig', None),
    ('util_math', None),
    ('matching', None),
    ('geometry', None),
    ('nearest_neighbors', None),
    ('clustering2', None),
    ('distance', None),
    ('other', None),
    ('numpy_utils', None),
    ('confusion', None),
    ('score_normalization', None),
    ('blend', None),
    ('symbolic', None),
    #('coverage_kpts', None),
    #('coverage_grid', None),
]

#import sys
from vtool import histogram
from vtool import linalg
from vtool import image_shared
from vtool import image
from vtool import exif
from vtool import keypoint
from vtool import ellipse
from vtool import patch
from vtool import chip
from vtool import spatial_verification
from vtool import trig
from vtool import util_math
from vtool import matching
from vtool import geometry
from vtool import nearest_neighbors
from vtool import clustering2
from vtool import other
from vtool import numpy_utils
from vtool import confusion
from vtool import score_normalization
from vtool import symbolic

# TODO: de-utoolificaiton: replace utool with ubelt
from vtool import histogram as htool
from vtool import linalg as ltool
from vtool import image as gtool
from vtool import exif as exiftool
from vtool import keypoint as ktool
from vtool import ellipse as etool
from vtool import patch as ptool
from vtool import chip as ctool
from vtool import spatial_verification as svtool
from vtool import clustering2 as clustertool
from vtool import trig
from vtool import util_math as mtool
from vtool.tests.dummy import get_dummy_kpts
from vtool.tests import dummy

import sys


# <AUTOGEN_INIT>

from vtool import image
from vtool import histogram
from vtool import image
from vtool import exif
from vtool import keypoint
from vtool import features
from vtool import linalg
from vtool import patch
from vtool import chip
from vtool import spatial_verification
from vtool import trig
from vtool import util_math
from vtool import matching
from vtool import geometry
from vtool import nearest_neighbors
from vtool import clustering2
from vtool import distance
from vtool import other
from vtool import numpy_utils
from vtool import confusion
from vtool import score_normalization
from vtool import blend
from vtool import symbolic
from vtool.image import (CV2_BORDER_TYPES, CV2_INTERPOLATION_TYPES,
                         CV2_WARP_KWARGS, EXIF_TAG_DATETIME, EXIF_TAG_GPS,
                         IMREAD_COLOR, LINE_AA, TAU,
                         affine_warp_around_center, clipwhite,
                         clipwhite_ondisk, combine_offset_lists,
                         convert_colorspace, convert_image_list_colorspace,
                         crop_out_imgfill, cvt_BGR2L, cvt_BGR2RGB,
                         draw_text, embed_channels, embed_in_square_image,
                         ensure_3channel, filterflags_valid_images,
                         find_pixel_value_index,
                         get_num_channels, get_pixel_dist,
                         get_round_scaled_dsize, get_scale_factor,
                         get_size, imread, imread_remote_s3,
                         imread_remote_url, imwrite, imwrite_fallback,
                         infer_vert, make_channels_comparable,
                         make_white_transparent, montage, open_image_size,
                         pad_image, pad_image_ondisk, padded_resize,
                         perlin_noise, rectify_to_float01,
                         rectify_to_square, rectify_to_uint8, resize,
                         resize_image_by_scale,
                         resize_mask,
                         resize_thumb, resize_to_maxdims,
                         resize_to_maxdims_ondisk,
                         resized_clamped_thumb_dims,
                         resized_dims_and_ratio, rotate_image,
                         rotate_image_ondisk, shear, stack_image_list,
                         stack_image_list_special, stack_image_recurse,
                         stack_images, stack_multi_images,
                         stack_multi_images2, stack_square_images,
                         subpixel_values, testdata_imglist, warpAffine,
                         warpHomog,)
from vtool.histogram import (argsubmax, argsubmaxima, get_histinfo_str,
                             hist_argmaxima, hist_argmaxima2,
                             hist_edges_to_centers, interpolate_submaxima,
                             interpolated_histogram, maxima_neighbors,
                             maximum_parabola_point, show_hist_submaxima,
                             show_ori_image, show_ori_image_ondisk,
                             subbin_bounds, wrap_histogram,)

from vtool.exif import (DATETIMEORIGINAL_TAGID, EXIF_TAG_TO_TAGID,
                        GPSINFO_CODE, GPSLATITUDEREF_CODE,
                        GPSLATITUDE_CODE, GPSLONGITUDEREF_CODE,
                        GPSLONGITUDE_CODE, GPSTAGS, GPS_TAG_TO_GPSID,
                        ORIENTATION_000, ORIENTATION_090, ORIENTATION_180,
                        ORIENTATION_270, ORIENTATION_CODE,
                        ORIENTATION_DICT, ORIENTATION_DICT_INVERSE,
                        ORIENTATION_UNDEFINED, SENSITIVITYTYPE_CODE, TAGS,
                        check_exif_keys, convert_degrees, get_exif_dict,
                        get_exif_dict2, get_exif_tagids, get_exist,
                        get_lat_lon, get_orientation, get_orientation_str,
                        get_unixtime, make_exif_dict_human_readable,
                        parse_exif_unixtime, read_all_exif_tags, read_exif,
                        read_exif_tags, read_one_exif_tag,)
from vtool.keypoint import (GRAVITY_THETA, KPTS_DTYPE, LOC_DIMS, ORI_DIM,
                            SCAX_DIM, SCAY_DIM, SHAPE_DIMS, SKEW_DIM, XDIM,
                            YDIM, augment_2x2_with_translation, cast_split,
                            convert_kptsZ_to_kpts,
                            decompose_Z_to_RV_mats2x2,
                            decompose_Z_to_V_2x2, decompose_Z_to_invV_2x2,
                            decompose_Z_to_invV_mats2x2,
                            flatten_invV_mats_to_kpts, get_RV_mats2x2,
                            get_RV_mats_3x3, get_V_mats, get_Z_mats,
                            get_even_point_sample, get_grid_kpts,
                            get_invVR_mats2x2, get_invVR_mats3x3,
                            get_invVR_mats_oris, get_invVR_mats_shape,
                            get_invVR_mats_sqrd_scale, get_invVR_mats_xys,
                            get_invV_mats, get_invV_mats2x2,
                            get_invV_mats3x3, get_invVs,
                            get_kpts_dlen_sqrd, get_kpts_eccentricity,
                            get_kpts_image_extent, get_kpts_strs,
                            get_kpts_wh, get_match_spatial_squared_error,
                            get_ori_mats, get_ori_strs, get_oris,
                            get_scales, get_shape_strs, get_sqrd_scales,
                            get_transforms_from_patch_image_kpts,
                            get_uneven_point_sample, get_xy_strs, get_xys,
                            invert_invV_mats, kp_cpp_infostr, kpts_docrepr,
                            kpts_repr, matrix_multiply, offset_kpts,
                            rectify_invV_mats_are_up, transform_kpts,
                            transform_kpts_to_imgspace,
                            transform_kpts_xys,)
from vtool.features import (detect_opencv_keypoints,
                            extract_feature_from_patch, extract_features,
                            get_extract_features_default_params,
                            test_mser,)
from vtool.linalg import (TRANSFORM_DTYPE, add_homogenous_coordinate,
                          affine_around_mat3x3, affine_mat3x3, det_ltri,
                          dot_ltri, gauss2d_pdf, inv_ltri, normalize,
                          normalize_rows, random_affine_args,
                          random_affine_transform,
                          remove_homogenous_coordinate,
                          rotation_around_bbox_mat3x3,
                          rotation_around_mat3x3, rotation_mat2x2,
                          rotation_mat3x3, scale_around_mat3x3,
                          scale_mat3x3, shear_mat3x3, svd,
                          transform_around,
                          transform_points_with_homography,
                          translation_mat3x3, whiten_xy_points,)
from vtool.patch import (GaussianBlurInplace, draw_kp_ori_steps,
                         find_dominant_kp_orientations,
                         find_kpts_direction,
                         find_patch_dominant_orientations,
                         gaussian_average_patch, gaussian_patch,
                         gaussian_weight_patch,
                         generate_to_patch_transforms, get_cross_patch,
                         get_no_symbol, get_orientation_histogram,
                         get_star2_patch, get_star_patch, get_stripe_patch,
                         get_test_patch, get_unwarped_patch,
                         get_unwarped_patches, get_warped_patch,
                         get_warped_patches, gradient_fill,
                         intern_warp_single_patch, inverted_sift_patch,
                         make_test_image_keypoints,
                         patch_gaussian_weighted_average_intensities,
                         patch_gradient, patch_mag, patch_ori,
                         show_gaussian_patch,
                         show_patch_orientation_estimation,
                         test_ondisk_find_patch_fpath_dominant_orientations,
                         test_show_gaussian_patches,
                         test_show_gaussian_patches2, testdata_patch,)
from vtool.chip import (apply_filter_funcs, compute_chip,
                        extract_chip_from_gpath,
                        extract_chip_from_gpath_into_square,
                        extract_chip_from_img, extract_chip_into_square,
                        get_extramargin_measures,
                        get_image_to_chip_transform,
                        get_scaled_size_with_area,
                        get_scaled_size_with_dlen,
                        get_scaled_size_with_width,
                        get_scaled_sizes_with_area, gridsearch_chipextract,
                        testshow_extramargin_info,)
from vtool.spatial_verification import (HAVE_SVER_C_WRAPPER, INDEX_DTYPE,
                                        SV_DTYPE, VERBOSE_SVER,
                                        build_affine_lstsqrs_Mx6,
                                        build_lstsqrs_Mx9, compute_affine,
                                        compute_homog,
                                        estimate_refined_transform,
                                        get_affine_inliers,
                                        get_best_affine_inliers,
                                        get_best_affine_inliers_,
                                        get_normalized_affine_inliers,
                                        refine_inliers,
                                        spatially_verify_kpts,
                                        test_affine_errors,
                                        test_homog_errors,
                                        testdata_matching_affine_inliers,
                                        testdata_matching_affine_inliers_normalized,
                                        try_svd, unnormalize_transform,)
from vtool.trig import (atan2,)
from vtool.util_math import (beaton_tukey_loss, beaton_tukey_weight,
                        breakup_equal_streak, ensure_monotone_decreasing,
                        ensure_monotone_increasing,
                        ensure_monotone_strictly_decreasing,
                        ensure_monotone_strictly_increasing, eps,
                        gauss_func1d, gauss_func1d_unnormalized,
                        gauss_parzen_est, group_consecutive, iceil,
                        interpolate_nans, iround, logistic_01, logit,
                        non_decreasing, non_increasing,
                        strictly_decreasing, strictly_increasing,
                        test_language_modulus,)
from vtool.matching import (AnnotPairFeatInfo, AssignTup, MatchingError,
                            PSEUDO_MAX_DIST, PSEUDO_MAX_DIST_SQRD,
                            PSEUDO_MAX_VEC_COMPONENT, PairwiseMatch,
                            SUM_OPS, VSONE_ASSIGN_CONFIG,
                            VSONE_DEFAULT_CONFIG, VSONE_FEAT_CONFIG,
                            VSONE_PI_DICT, VSONE_RATIO_CONFIG,
                            VSONE_SVER_CONFIG, assign_symmetric_matches,
                            assign_unconstrained_matches, csum,
                            demodata_match, empty_neighbors,
                            ensure_metadata_dlen_sqrd,
                            ensure_metadata_feats, ensure_metadata_flann,
                            ensure_metadata_normxy, ensure_metadata_vsone,
                            flag_sym_slow, flag_symmetric_matches, invsum,
                            namedtuple, normalized_nearest_neighbors,
                            testdata_annot_metadata,)
from vtool.geometry import (bbox_center, bbox_from_center_wh,
                            bbox_from_extent, bbox_from_verts,
                            bbox_from_xywh, bboxes_from_vert_list,
                            closest_point_on_bbox, closest_point_on_line,
                            closest_point_on_line_segment,
                            closest_point_on_vert_segments,
                            cvt_bbox_xywh_to_pt1pt2, distance_to_lineseg,
                            draw_border, draw_verts, extent_from_bbox,
                            extent_from_verts, get_pointset_extent_wh,
                            get_pointset_extents, point_inside_bbox,
                            scale_bbox, scale_extents,
                            scaled_verts_from_bbox,
                            scaled_verts_from_bbox_gen, union_extents,
                            verts_from_bbox, verts_list_from_bboxes_list,)
from vtool.nearest_neighbors import (AnnoyWrapper, ann_flann_once,
                                     assign_to_centroids, flann_augment,
                                     flann_cache,
                                     flann_index_time_experiment,
                                     get_flann_cfgstr, get_flann_fpath,
                                     get_flann_params,
                                     get_flann_params_cfgstr,
                                     get_kdtree_flann_params,
                                     invertible_stack, test_annoy,
                                     test_cv2_flann, tune_flann,)
from vtool.clustering2 import (AnnoyWraper, apply_grouping,
                               apply_grouping_, apply_grouping_iter,
                               apply_grouping_iter2, apply_jagged_grouping,
                               example_binary, find_duplicate_items,
                               group_indices, groupby, groupby_dict,
                               groupby_gen, groupedzip,
                               invert_apply_grouping,
                               invert_apply_grouping2,
                               invert_apply_grouping3, jagged_group,
                               plot_centroids, sorted_indices_ranges,
                               tune_flann2, uniform_sample_hypersphere,
                               unsupervised_multicut_labeling,)
from vtool.distance import (L1, L2, L2_root_sift, L2_sift, L2_sift_sqrd,
                            L2_sqrd, OrderedDict, TEMP_VEC_DTYPE,
                            VALID_DISTS, bar_L2_sift, bar_cos_sift,
                            closest_point, compute_distances, cos_sift,
                            cosine_dist, cyclic_distance, det_distance,
                            emd, haversine, hist_isect, nearest_point,
                            ori_distance, pdist_argsort, pdist_indicies,
                            safe_pdist, signed_ori_distance, testdata_hist,
                            testdata_sift2, understanding_pseudomax_props,
                            wrapped_distance,)
from vtool.other import (and_lists, argsort_groups, argsort_records,
                         assert_zipcompress, asserteq, axiswise_operation2,
                         bow_test, calc_error_bars_from_sample,
                         calc_sample_from_error_bars, check_sift_validity,
                         clipnorm, colwise_operation,
                         compare_implementations, compare_matrix_columns,
                         compare_matrix_to_rows, componentwise_dot,
                         compress2, compute_ndarray_unique_rowids_unsafe,
                         compute_unique_arr_dataids,
                         compute_unique_data_ids, compute_unique_data_ids_,
                         compute_unique_integer_data_ids, ensure_rng,
                         find_best_undirected_edge_indexes,
                         find_elbow_point, find_first_true_indices,
                         find_k_true_indicies, find_next_true_indices,
                         flag_intersection, get_consec_endpoint,
                         get_covered_mask, get_crop_slices,
                         get_uncovered_mask, get_undirected_edge_ids,
                         grab_webcam_image, greedy_setcover, inbounds,
                         index_partition, index_to_boolmask,
                         intersect1d_reduce, intersect2d_flags,
                         intersect2d_indices, intersect2d_numpy,
                         intersect2d_structured_numpy, iter_reduce_ufunc,
                         list_compress_, list_take_, make_video,
                         make_video2, median_abs_dev, mult_lists,
                         multiaxis_reduce, multigroup_lookup,
                         multigroup_lookup_naive, next,
                         nonunique_row_flags, nonunique_row_indexes,
                         norm01, or_lists, pad_vstack, rebuild_partition,
                         rowwise_operation, safe_cat, safe_div,
                         safe_extreme, safe_max, safe_min, safe_vstack,
                         structure_rows, take2, take_col_per_row,
                         to_undirected_edges, trytake, unique_row_indexes,
                         unique_rows, unstructure_rows,
                         weighted_average_scoring, weighted_geometic_mean,
                         weighted_geometic_mean_unnormalized, zipcat,
                         zipcompress, zipcompress_safe, ziptake,
                         zstar_value,)
from vtool.numpy_utils import (atleast_nd, ensure_shape, fromiter_nd,)
from vtool.confusion import (ConfusionMetrics, draw_precision_recall_curve,
                             draw_roc_curve, interact_roc_factory,
                             interpolate_precision_recall,
                             interpolate_replbounds, nan_to_num,
                             testdata_scores_labels,)
from vtool.score_normalization import (ScoreNormVisualizeClass,
                                       ScoreNormalizer,
                                       check_unused_kwargs, estimate_pdf,
                                       find_clip_range, flatten_scores,
                                       get_left_area, get_right_area,
                                       inspect_pdfs,
                                       learn_score_normalization,
                                       normalize_scores, partial,
                                       partition_scores,
                                       plot_postbayes_pdf,
                                       plot_prebayes_pdf,
                                       test_score_normalization,
                                       testdata_score_normalier,)
from vtool.blend import (blend_images, blend_images_average,
                         blend_images_average_stack,
                         blend_images_mult_average, blend_images_multiply,
                         gamma_adjust, gridsearch_addWeighted,
                         gridsearch_image_function, overlay_alpha_images,
                         testdata_blend,)
from vtool.symbolic import (check_expr_eq, custom_sympy_attrs, evalprint,
                            symbolic_randcheck, sympy_latex_repr,
                            sympy_mat, sympy_numpy_repr,)
import utool
print, rrr, profile = utool.inject2(__name__, '[vtool]')


def reassign_submodule_attributes(verbose=1):
    """
    Updates attributes in the __init__ modules with updated attributes
    in the submodules.
    """
    import sys
    if verbose and '--quiet' not in sys.argv:
        print('dev reimport')
    # Self import
    import vtool
    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        if len(tup) > 2 and tup[2]:
            continue  # dont import package names
        submodname, fromimports = tup[0:2]
        submod = getattr(vtool, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(vtool, attr, getattr(submod, attr))


def reload_subs(verbose=1):
    """ Reloads vtool and submodules """
    if verbose:
        print('Reloading vtool submodules')
    rrr(verbose > 1)
    def wrap_fbrrr(mod):
        def fbrrr(*args, **kwargs):
            """ fallback reload """
            if verbose > 0:
                print('Auto-reload (using rrr) not setup for mod=%r' % (mod,))
        return fbrrr
    def get_rrr(mod):
        if hasattr(mod, 'rrr'):
            return mod.rrr
        else:
            return wrap_fbrrr(mod)
    def get_reload_subs(mod):
        return getattr(mod, 'reload_subs', wrap_fbrrr(mod))
    get_rrr(image)(verbose > 1)
    get_rrr(histogram)(verbose > 1)
    get_rrr(image)(verbose > 1)
    get_rrr(exif)(verbose > 1)
    get_rrr(keypoint)(verbose > 1)
    get_rrr(features)(verbose > 1)
    get_rrr(linalg)(verbose > 1)
    get_rrr(patch)(verbose > 1)
    get_rrr(chip)(verbose > 1)
    get_rrr(spatial_verification)(verbose > 1)
    get_rrr(trig)(verbose > 1)
    get_rrr(util_math)(verbose > 1)
    get_rrr(matching)(verbose > 1)
    get_rrr(geometry)(verbose > 1)
    get_rrr(nearest_neighbors)(verbose > 1)
    get_rrr(clustering2)(verbose > 1)
    get_rrr(distance)(verbose > 1)
    get_rrr(other)(verbose > 1)
    get_rrr(numpy_utils)(verbose > 1)
    get_rrr(confusion)(verbose > 1)
    get_rrr(score_normalization)(verbose > 1)
    get_rrr(blend)(verbose > 1)
    get_rrr(symbolic)(verbose > 1)
    rrr(verbose > 1)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)
rrrr = reload_subs
# </AUTOGEN_INIT>
