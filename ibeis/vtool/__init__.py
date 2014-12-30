# flake8: noqa
from __future__ import absolute_import, division, print_function
__version__ = '1.0.0.dev1'

import utool as ut
ut.noinject(__name__, '[vtool]')


# TODO utoolify this
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
    ('geometry', None),
    ('nearest_neighbors', None),
    ('clustering2', None),
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
from vtool import geometry
from vtool import clustering
from vtool import nearest_neighbors
from vtool import clustering2

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

#try:
#    from . import _linalg_cyth
#    #print('[vtool] cython is on')
#except ImportError as ex:
#    #import utool
#    #utool.printex(ex, iswarning=True)
#    #print('[vtool] cython is off')
#    raise


r"""

ls vtool

rm -rf build ; rm vtool/*.pyd ; rm vtool/*.c

python setup.py build_ext --inplace

cyth.py vtool/_linalg_cyth.pyx

C:\Python27\Scripts\cython.exe C:\Users\joncrall\code\vtool\vtool\_linalg_cyth.pyx

#C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\_linalg_cyth.c -o vtool\_linalg_cyth.o
#C:\MinGW\bin\gcc.exe -shared -s build\temp.win32-2.7\Release\vtool\_linalg_cyth.o build\temp.win32-2.7\Release\vtool\_linalg_cyth.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o  build\lib.win32-2.7\_linalg_cyth.pyd

C:\MinGW\bin\gcc.exe -shared -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -static-libgcc -static-libstdc++ -c _linalg_cyth.c -o _linalg_cyth.pyd
python -c "import vtool"


C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\_linalg_cyth.c -o _linalg_cyth.o
C:\MinGW\bin\gcc.exe -shared -s _linalg_cyth.o _linalg_cyth.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o _linalg_cyth.pyd


C:\MinGW\bin\gcc.exe -w -Wall -m32 -lpython27 -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -IC:\Python27\Lib\site-packages\numpy\core\include -LC:\Python27\libs -o _linalg_cyth.pyd -c _linalg_cyth.c

python -c "import vtool"
"""

import sys
__DYNAMIC__ = not '--nodyn' in sys.argv

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
    from vtool import geometry
    from vtool import nearest_neighbors
    from vtool import clustering2
    from vtool.image import (CV2_INTERPOLATION_TYPES, CV2_WARP_KWARGS, 
                             EXIF_TAG_DATETIME, EXIF_TAG_GPS, IMREAD_COLOR, 
                             ThumbnailCacheContext, blend_images, cvt_BGR2L, 
                             cvt_BGR2RGB, cvt_bbox_xywh_to_pt1pt2, dummy_img, 
                             get_gpathlist_sizes, get_num_channels, 
                             get_scale_factor, get_size, imread, imwrite, 
                             imwrite_fallback, open_image_size, open_pil_image, 
                             print_image_checks, resize, 
                             resize_imagelist_generator, 
                             resize_imagelist_to_sqrtarea, resize_thumb, 
                             resize_worker, resized_clamped_thumb_dims, 
                             resized_dims_and_ratio, scale_bbox_to_verts_gen, 
                             subpixel_values, warpAffine, warpHomog,) 
    from vtool.histogram import (argrelextrema, hist_argmaxima, 
                                 hist_edges_to_centers, 
                                 hist_interpolated_submaxima, inject, 
                                 interpolate_submaxima, maxima_neighbors, 
                                 subbin_bounds, wrap_histogram,) 
     
    from vtool.exif import (DATETIMEORIGINAL_TAGID, EXIF_TAG_TO_TAGID, 
                            GPSINFO_CODE, GPSLATITUDEREF_CODE, 
                            GPSLATITUDE_CODE, GPSLONGITUDEREF_CODE, 
                            GPSLONGITUDE_CODE, GPSTAGS, GPS_TAG_TO_GPSID, 
                            SENSITIVITYTYPE_CODE, TAGS, check_exif_keys, 
                            convert_degrees, get_exif_dict, get_exif_dict2, 
                            get_exif_tagids, get_exist, get_lat_lon, 
                            get_lat_lon2, get_unixtime, 
                            make_exif_dict_human_readable, read_all_exif_tags, 
                            read_exif, read_exif_tags, read_one_exif_tag,) 
    from vtool.keypoint import (GRAVITY_THETA, KPTS_DTYPE, LOC_DIMS, ORI_DIM, 
                                SCAX_DIM, SCAY_DIM, SHAPE_DIMS, SKEW_DIM, TAU, 
                                XDIM, YDIM, cast_split, diag, 
                                flatten_invV_mats_to_kpts, get_V_mats, 
                                get_Z_mats, get_diag_extent_sqrd, 
                                get_grid_kpts, get_homog_xyzs, 
                                get_invVR_mats_oris, get_invVR_mats_shape, 
                                get_invVR_mats_sqrd_scale, get_invVR_mats_xys, 
                                get_invV_mats, get_invV_mats2x2, 
                                get_invV_xy_axis_extents, get_invVs, 
                                get_kpts_bounds, get_kpts_strs, get_ori_mats, 
                                get_ori_strs, get_oris, get_scales, 
                                get_shape_strs, get_sqrd_scales, 
                                get_xy_axis_extents, get_xy_strs, get_xys, 
                                invert_invV_mats, matrix_multiply, offset_kpts, 
                                ones, rectify_invV_mats_are_up, rollaxis, sqrt, 
                                transform_kpts, transform_kpts_to_imgspace, 
                                zeros,) 
    from vtool.features import (extract_features,) 
    from vtool.linalg import (L1, L2, L2_sqrd, OLD_pdf_norm2d, TRANSFORM_DTYPE, 
                              and_3lists, and_lists, axiswise_operation2, 
                              colwise_operation, compare_matrix_columns, 
                              compare_matrix_to_rows, cos, det_distance, 
                              det_ltri, dot_ltri, flag_intersection, 
                              gauss2d_pdf, get_covered_mask, 
                              get_uncovered_mask, hist_isect, homogonize, 
                              intersect2d_indicies, intersect2d_numpy, 
                              intersect2d_structured_numpy, inv_ltri, 
                              mult_lists, nearest_point, normalize_rows, 
                              normalize_vecs2d_inplace, or_lists, ori_distance, 
                              rotation_around_bbox_mat3x3, 
                              rotation_around_mat3x3, rotation_mat2x2, 
                              rotation_mat3x3, rowwise_division, 
                              rowwise_operation, rowwise_oridist, scale_mat3x3, 
                              scaleedoffset_mat3x3, sin, svd, 
                              translation_mat3x3, whiten_xy_points,) 
    from vtool.patch import (find_kpts_direction, gaussian_patch, 
                             get_orientation_histogram, get_unwarped_patch, 
                             get_unwarped_patches, get_warped_patch, 
                             get_warped_patches, lru_cache, patch_gradient, 
                             patch_mag, patch_ori,) 
    from vtool.chip import (compute_chip, get_filter_list, 
                            get_scaled_size_with_area, 
                            get_scaled_sizes_with_area,) 
    from vtool.spatial_verification import (SV_DTYPE, build_lstsqrs_Mx9, 
                                            compute_homog, get_affine_inliers, 
                                            get_best_affine_inliers, 
                                            get_homography_inliers, ibeis_test, 
                                            spatially_verify_kpts,) 
    from vtool.trig import (atan2,) 
    from vtool.math import (ensure_monotone_decreasing, 
                            ensure_monotone_increasing, 
                            ensure_monotone_strictly_increasing, eps, 
                            group_consecutive, non_decreasing, non_increasing, 
                            strictly_decreasing, strictly_increasing, tau,) 
    from vtool.geometry import (bbox_of_verts, bboxes_from_vert_list, 
                                draw_verts, homogonize_list, unhomogonize, 
                                unhomogonize_list, verts_from_bbox, 
                                verts_list_from_bboxes_list,) 
    from vtool.nearest_neighbors import (ann_flann_once, assign_to_centroids, 
                                         flann_augment, flann_cache, 
                                         get_flann_cfgstr, get_flann_fpath, 
                                         get_flann_params, 
                                         get_flann_params_cfgstr, 
                                         get_kdtree_flann_params, 
                                         invertable_stack, tune_flann,) 
    from vtool.clustering2 import (CLUSTERS_FNAME, akmeans, akmeans_iterations, 
                                   akmeans_plusplus_init, apply_grouping, 
                                   apply_grouping_iter, apply_grouping_iter2, 
                                   apply_jagged_grouping, 
                                   approximate_assignments, 
                                   approximate_distances, assert_centroids, 
                                   cached_akmeans, compute_centroids, 
                                   double_group, find_duplicate_items, 
                                   get_akmeans_cfgstr, group_indicies, 
                                   group_indicies_pandas, groupby, 
                                   groupby_dict, groupby_gen, 
                                   initialize_centroids, jagged_group, 
                                   plot_centroids, refine_akmeans, 
                                   sparse_multiply_rows, sparse_normalize_rows, 
                                   tune_flann2,) 
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
        getattr(geometry, 'rrr', fbrrr)(verbose=verbose)
        getattr(nearest_neighbors, 'rrr', fbrrr)(verbose=verbose)
        getattr(clustering2, 'rrr', fbrrr)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # </AUTOGEN_INIT>