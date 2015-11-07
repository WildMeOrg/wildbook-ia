# -*- coding: utf-8 -*-
"""
VTool - Computer vision tools
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
__version__ = '1.0.1.dev1'

import utool as ut
ut.noinject(__name__, '[vtool]')


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
    ('math', None),
    ('matching', None),
    ('geometry', None),
    ('nearest_neighbors', None),
    ('clustering2', None),
    ('distance', None),
    ('other', None),
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
from vtool import image
from vtool import exif
from vtool import keypoint
from vtool import ellipse
from vtool import patch
from vtool import chip
from vtool import spatial_verification
from vtool import trig
from vtool import math
from vtool import matching
from vtool import geometry
from vtool import clustering
from vtool import nearest_neighbors
from vtool import clustering2
from vtool import other
from vtool import confusion
from vtool import score_normalization
from vtool import symbolic

# TODO: incorporate into utoolification
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
from vtool import math as mtool
from vtool.tests.dummy import get_dummy_kpts
from vtool.tests import dummy

import sys
__DYNAMIC__ = not ut.get_argflag('--nodyn')


#__DYNAMIC__ = '--dyn' in sys.argv
"""
python -c "import vtool" --dump-vtool-init
python -c "import vtool" --update-vtool-init
"""


DOELSE = False
if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    from utool._internal import util_importer
    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    ignore_endswith = ['_cyth']
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES, ignore_endswith=ignore_endswith)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    pass
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
    from vtool import math
    from vtool import matching
    from vtool import geometry
    from vtool import nearest_neighbors
    from vtool import clustering2
    from vtool import distance
    from vtool import other
    from vtool import confusion
    from vtool import score_normalization
    from vtool import blend
    from vtool import symbolic
    from vtool.image import (CV2_INTERPOLATION_TYPES, CV2_WARP_KWARGS, 
                             EXIF_TAG_DATETIME, EXIF_TAG_GPS, IMREAD_COLOR, 
                             LINE_AA, TAU, affine_warp_around_center, 
                             clipwhite_ondisk, combine_offset_lists, 
                             convert_image_list_colorspace, crop_out_imgfill, 
                             cvt_BGR2L, cvt_BGR2RGB, cvt_bbox_xywh_to_pt1pt2, 
                             draw_text, embed_in_square_image, ensure_3channel, 
                             filterflags_valid_images, find_pixel_value_index, 
                             get_gpathlist_sizes, get_num_channels, 
                             get_pixel_dist, get_round_scaled_dsize, 
                             get_scale_factor, get_size, imread, imwrite, 
                             imwrite_fallback, infer_vert, 
                             make_channels_comparable, open_image_size, 
                             open_pil_image, pad_image_on_disk, padded_resize, 
                             perlin_noise, print_image_checks, 
                             rectify_to_float01, rectify_to_uint8, resize, 
                             resize_image_by_scale, resize_imagelist_generator, 
                             resize_imagelist_to_sqrtarea, resize_thumb, 
                             resize_to_maxdims, resize_worker, 
                             resized_clamped_thumb_dims, 
                             resized_dims_and_ratio, rotate_image, 
                             rotate_image_on_disk, scaled_verts_from_bbox, 
                             scaled_verts_from_bbox_gen, shear, 
                             stack_image_list, stack_image_list_special, 
                             stack_image_recurse, stack_images, 
                             stack_multi_images, stack_multi_images2, 
                             stack_square_images, subpixel_values, 
                             testdata_imglist, warpAffine, warpHomog,) 
    from vtool.histogram import (get_histinfo_str, hist_argmaxima, 
                                 hist_edges_to_centers, 
                                 hist_interpolated_submaxima, 
                                 interpolate_submaxima, interpolated_histogram, 
                                 maxima_neighbors, maximum_parabola_point, 
                                 readable_interpolate_submaxima, 
                                 show_hist_submaxima, show_ori_image, 
                                 show_ori_image_ondisk, subbin_bounds, 
                                 wrap_histogram,) 
     
    from vtool.exif import (DATETIMEORIGINAL_TAGID, EXIF_TAG_TO_TAGID, 
                            GPSINFO_CODE, GPSLATITUDEREF_CODE, 
                            GPSLATITUDE_CODE, GPSLONGITUDEREF_CODE, 
                            GPSLONGITUDE_CODE, GPSTAGS, GPS_TAG_TO_GPSID, 
                            SENSITIVITYTYPE_CODE, TAGS, check_exif_keys, 
                            convert_degrees, get_exif_dict, get_exif_dict2, 
                            get_exif_tagids, get_exist, get_lat_lon, 
                            get_unixtime, make_exif_dict_human_readable, 
                            parse_exif_unixtime, read_all_exif_tags, read_exif, 
                            read_exif_tags, read_one_exif_tag,) 
    from vtool.keypoint import (GRAVITY_THETA, KPTS_DTYPE, LOC_DIMS, ORI_DIM, 
                                SCAX_DIM, SCAY_DIM, SHAPE_DIMS, SKEW_DIM, XDIM, 
                                YDIM, augment_2x2_with_translation, cast_split, 
                                flatten_invV_mats_to_kpts, get_RV_mats_3x3, 
                                get_V_mats, get_Z_mats, get_even_point_sample, 
                                get_grid_kpts, get_invVR_mats2x2, 
                                get_invVR_mats3x3, get_invVR_mats_oris, 
                                get_invVR_mats_shape, 
                                get_invVR_mats_sqrd_scale, get_invVR_mats_xys, 
                                get_invV_mats, get_invV_mats2x2, 
                                get_invV_mats3x3, get_invV_xy_axis_extents, 
                                get_invVs, get_kpts_dlen_sqrd, 
                                get_kpts_image_extent, get_kpts_strs, 
                                get_match_spatial_squared_error, get_ori_mats, 
                                get_ori_strs, get_oris, get_scales, 
                                get_shape_strs, get_sqrd_scales, 
                                get_transforms_from_patch_image_kpts, 
                                get_uneven_point_sample, get_xy_axis_extents, 
                                get_xy_strs, get_xys, invert_invV_mats, 
                                kp_cpp_infostr, kpts_docrepr, kpts_repr, 
                                matrix_multiply, offset_kpts, 
                                rectify_invV_mats_are_up, transform_kpts, 
                                transform_kpts_to_imgspace, 
                                transform_kpts_xys,) 
    from vtool.features import (detect_opencv_keypoints, extract_features, 
                                get_extract_features_default_params,) 
    from vtool.linalg import (OLD_pdf_norm2d, TRANSFORM_DTYPE, 
                              add_homogenous_coordinate, affine_around_mat3x3, 
                              affine_mat3x3, det_ltri, dot_ltri, gauss2d_pdf, 
                              inv_ltri, normalize_rows, 
                              remove_homogenous_coordinate, 
                              rotation_around_bbox_mat3x3, 
                              rotation_around_mat3x3, rotation_mat2x2, 
                              rotation_mat3x3, scale_mat3x3, shear_mat3x3, svd, 
                              transform_points_with_homography, 
                              translation_mat3x3, whiten_xy_points,) 
    from vtool.patch import (GaussianBlurInplace, 
                             find_dominant_kp_orientations, 
                             find_kpts_direction, 
                             find_patch_dominant_orientations, 
                             gaussian_average_patch, gaussian_patch, 
                             gaussian_weight_patch, get_cross_patch, 
                             get_no_symbol, get_orientation_histogram, 
                             get_star2_patch, get_star_patch, get_stripe_patch, 
                             get_test_patch, get_unwarped_patch, 
                             get_unwarped_patches, get_warped_patch, 
                             get_warped_patches, intern_warp_single_patch, 
                             make_test_image_keypoints, patch_gradient, 
                             patch_mag, patch_ori, show_gaussian_patch, 
                             show_patch_orientation_estimation, 
                             test_find_kp_direction, 
                             test_ondisk_find_patch_fpath_dominant_orientations, 
                             test_show_gaussian_patches, 
                             test_show_gaussian_patches2,) 
    from vtool.chip import (apply_filter_funcs, compute_chip, 
                            extract_chip_from_gpath, 
                            extract_chip_from_gpath_into_square, 
                            extract_chip_from_img, extract_chip_into_square, 
                            get_filter_list, get_image_to_chip_transform, 
                            get_scaled_size_with_area, 
                            get_scaled_size_with_width, 
                            get_scaled_sizes_with_area, 
                            gridsearch_chipextract,) 
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
    from vtool.math import (breakup_equal_streak, ensure_monotone_decreasing, 
                            ensure_monotone_increasing, 
                            ensure_monotone_strictly_decreasing, 
                            ensure_monotone_strictly_increasing, eps, 
                            gauss_func1d, gauss_func1d_unnormalized, 
                            group_consecutive, iceil, iround, non_decreasing, 
                            non_increasing, strictly_decreasing, 
                            strictly_increasing, test_language_modulus,) 
    from vtool.matching import (PSEUDO_MAX_DIST, PSEUDO_MAX_DIST_SQRD, 
                                PSEUDO_MAX_VEC_COMPONENT, 
                                assign_spatially_constrained_matches, 
                                assign_unconstrained_matches, ensure_fsv_list, 
                                marge_matches, normalized_nearest_neighbors, 
                                ratio_test, show_matching_dict, 
                                spatially_constrained_ratio_match, 
                                unconstrained_ratio_match, 
                                vsone_feature_matching, 
                                vsone_image_fpath_matching, vsone_matching,) 
    from vtool.geometry import (bbox_center, bbox_from_verts, 
                                bboxes_from_vert_list, draw_border, draw_verts, 
                                verts_from_bbox, verts_list_from_bboxes_list,) 
    from vtool.nearest_neighbors import (ann_flann_once, assign_to_centroids, 
                                         build_flann_index, flann_augment, 
                                         flann_cache, 
                                         flann_index_time_experiment, 
                                         get_flann_cfgstr, get_flann_fpath, 
                                         get_flann_params, 
                                         get_flann_params_cfgstr, 
                                         get_kdtree_flann_params, 
                                         invertible_stack, test_cv2_flann, 
                                         tune_flann,) 
    from vtool.clustering2 import (CLUSTERS_FNAME, akmeans, akmeans_iterations, 
                                   akmeans_plusplus_init, apply_grouping, 
                                   apply_grouping_, apply_grouping_iter, 
                                   apply_grouping_iter2, apply_jagged_grouping, 
                                   approximate_assignments, 
                                   approximate_distances, assert_centroids, 
                                   cached_akmeans, compute_centroids, 
                                   double_group, find_duplicate_items, 
                                   get_akmeans_cfgstr, group_indices, groupby, 
                                   groupby_dict, groupby_gen, groupedzip, 
                                   initialize_centroids, invert_apply_grouping, 
                                   invert_apply_grouping2, jagged_group, 
                                   plot_centroids, refine_akmeans, 
                                   sparse_multiply_rows, sparse_normalize_rows, 
                                   tune_flann2, uniform_sample_hypersphere,) 
    from vtool.distance import (DEBUG_DIST, L1, L2, L2_sift, L2_sqrd, 
                                OrderedDict, TEMP_VEC_DTYPE, VALID_DISTS, 
                                bar_L2_sift, bar_cos_sift, closest_point, 
                                compute_distances, cos_sift, cosine_dist, 
                                det_distance, emd, hist_isect, nearest_point, 
                                ori_distance, signed_ori_distance, 
                                testdata_hist, understanding_pseudomax_props,) 
    from vtool.other import (and_lists, argsort_groups, argsort_multiarray, 
                             assert_zipcompress, axiswise_operation2, 
                             check_sift_validity, clipnorm, colwise_operation, 
                             compare_matrix_columns, compare_matrix_to_rows, 
                             componentwise_dot, compute_unique_arr_dataids, 
                             compute_unique_data_ids, compute_unique_data_ids_, 
                             compute_unique_integer_data_ids, 
                             find_best_undirected_edge_indexes, 
                             find_first_true_indices, find_next_true_indices, 
                             flag_intersection, get_consec_endpoint, 
                             get_covered_mask, get_crop_slices, 
                             get_uncovered_mask, get_undirected_edge_ids, 
                             grab_webcam_image, index_partition, 
                             index_to_boolmask, intersect1d_reduce, 
                             intersect2d_flags, intersect2d_indices, 
                             intersect2d_numpy, intersect2d_structured_numpy, 
                             iter_reduce_ufunc, list_compress_, list_take_, 
                             mult_lists, next, nonunique_row_flags, 
                             nonunique_row_indexes, norm01, or_lists, 
                             pdist_argsort, rebuild_partition, 
                             rowwise_operation, safe_vstack, trytake, 
                             unique_row_indexes, unique_rows, 
                             weighted_average_scoring, weighted_geometic_mean, 
                             weighted_geometic_mean_unnormalized, zipcompress, 
                             zipcompress_safe, ziptake,) 
    from vtool.confusion import (ConfusionMetrics, draw_precision_recall_curve, 
                                 draw_roc_curve, get_confusion_metrics, 
                                 interact_roc_factory, 
                                 interpolate_precision_recall, 
                                 interpolate_replbounds, 
                                 testdata_scores_labels,) 
    from vtool.score_normalization import (ScoreNormalizer, 
                                           check_unused_kwargs, estimate_pdf, 
                                           find_clip_range, flatten_scores, 
                                           inspect_pdfs, 
                                           learn_score_normalization, 
                                           normalize_scores, partial, 
                                           partition_scores, 
                                           plot_postbayes_pdf, 
                                           plot_prebayes_pdf, 
                                           test_score_normalization, 
                                           testdata_score_normalier,) 
    from vtool.blend import (blend_images, blend_images_average, 
                             blend_images_mult_average, blend_images_multiply, 
                             gamma_adjust, gridsearch_addWeighted, 
                             gridsearch_image_function, testdata_blend,) 
    from vtool.symbolic import (check_expr_eq, custom_sympy_attrs, evalprint, 
                                symbolic_randcheck, sympy_latex_repr, 
                                sympy_mat,) 
    import utool
    print, print_, printDBG, rrr, profile = utool.inject(
        __name__, '[vtool]')
    
    
    def reassign_submodule_attributes(verbose=True):
        """
        why reloading all the modules doesnt do this I don't know
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
    
    
    def reload_subs(verbose=True):
        """ Reloads vtool and submodules """
        rrr(verbose=verbose)
        def fbrrr(*args, **kwargs):
            """ fallback reload """
            pass
        getattr(image, 'rrr', fbrrr)(verbose=verbose)
        getattr(histogram, 'rrr', fbrrr)(verbose=verbose)
        getattr(image, 'rrr', fbrrr)(verbose=verbose)
        getattr(exif, 'rrr', fbrrr)(verbose=verbose)
        getattr(keypoint, 'rrr', fbrrr)(verbose=verbose)
        getattr(features, 'rrr', fbrrr)(verbose=verbose)
        getattr(linalg, 'rrr', fbrrr)(verbose=verbose)
        getattr(patch, 'rrr', fbrrr)(verbose=verbose)
        getattr(chip, 'rrr', fbrrr)(verbose=verbose)
        getattr(spatial_verification, 'rrr', fbrrr)(verbose=verbose)
        getattr(trig, 'rrr', fbrrr)(verbose=verbose)
        getattr(math, 'rrr', fbrrr)(verbose=verbose)
        getattr(matching, 'rrr', fbrrr)(verbose=verbose)
        getattr(geometry, 'rrr', fbrrr)(verbose=verbose)
        getattr(nearest_neighbors, 'rrr', fbrrr)(verbose=verbose)
        getattr(clustering2, 'rrr', fbrrr)(verbose=verbose)
        getattr(distance, 'rrr', fbrrr)(verbose=verbose)
        getattr(other, 'rrr', fbrrr)(verbose=verbose)
        getattr(confusion, 'rrr', fbrrr)(verbose=verbose)
        getattr(score_normalization, 'rrr', fbrrr)(verbose=verbose)
        getattr(blend, 'rrr', fbrrr)(verbose=verbose)
        getattr(symbolic, 'rrr', fbrrr)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # </AUTOGEN_INIT>
