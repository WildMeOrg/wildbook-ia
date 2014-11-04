# flake8: noqa
from __future__ import absolute_import, division, print_function
__version__ = '1.0.0.dev1'


# TODO utoolify this
IMPORT_TUPLES = [
    ('image', None),
    ('histogram', None),
    ('image', None),
    ('exif', None),
    ('keypoint', None),
    ('patch', None),
    ('chip', None),
    ('spatial_verification', None),
    ('trig', None),
    ('math', None),
    ('geometry', None),
    ('clustering2', None),
]

from . import histogram
from . import linalg
from . import image
from . import exif
from . import keypoint
from . import ellipse
from . import patch
from . import chip
from . import spatial_verification
from . import trig
from . import math
from . import geometry
from . import clustering
from . import nearest_neighbors
from . import clustering2

from . import histogram as htool
from . import linalg as ltool
from . import image as gtool
from . import exif as exiftool
from . import keypoint as ktool
from . import ellipse as etool
from . import patch as ptool
from . import chip as ctool
from . import spatial_verification as svtool
from . import clustering2 as clustertool
from . import trig
from . import math as mtool

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
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    pass
    # <AUTOGEN_INIT>

    from . import image
    from . import histogram
    from . import image
    from . import exif
    from . import keypoint
    from . import patch
    from . import chip
    from . import spatial_verification
    from . import trig
    from . import math
    from . import geometry
    from . import clustering2
    from .image import (CV2_INTERPOLATION_TYPES, CV2_WARP_KWARGS, 
                        EXIF_TAG_DATETIME, EXIF_TAG_GPS, IMREAD_COLOR, 
                        ThumbnailCacheContext, blend_images, cvt_BGR2L, 
                        cvt_BGR2RGB, cvt_bbox_xywh_to_pt1pt2, dummy_img, 
                        get_gpathlist_sizes, get_num_channels, 
                        get_scale_factor, get_size, imread, imwrite, 
                        imwrite_fallback, open_image_size, open_pil_image, 
                        print_image_checks, resize, resize_imagelist_generator, 
                        resize_imagelist_to_sqrtarea, resize_thumb, 
                        resize_worker, resized_clamped_thumb_dims, 
                        resized_dims_and_ratio, scale_bbox_to_verts_gen, 
                        subpixel_values, warpAffine, warpHomog,) 
    from .histogram import (argrelextrema, hist_argmaxima, 
                            hist_edges_to_centers, hist_interpolated_submaxima, 
                            inject, interpolate_submaxima, maxima_neighbors, 
                            subbin_bounds, wrap_histogram,) 
     
    from .exif import (DATETIMEORIGINAL_TAGID, EXIF_TAG_TO_TAGID, GPSINFO_CODE, 
                       GPSLATITUDEREF_CODE, GPSLATITUDE_CODE, 
                       GPSLONGITUDEREF_CODE, GPSLONGITUDE_CODE, GPSTAGS, 
                       GPS_TAG_TO_GPSID, SENSITIVITYTYPE_CODE, TAGS, 
                       check_exif_keys, convert_degrees, get_exif_dict, 
                       get_exif_dict2, get_exif_tagids, get_exist, get_lat_lon, 
                       get_lat_lon2, get_unixtime, 
                       make_exif_dict_human_readable, read_all_exif_tags, 
                       read_exif, read_exif_tags, read_one_exif_tag,) 
    from .keypoint import (CYTHONIZED, GRAVITY_THETA, KPTS_DTYPE, LOC_DIMS, 
                           ORI_DIM, SCAX_DIM, SCAY_DIM, SHAPE_DIMS, SKEW_DIM, 
                           TAU, XDIM, YDIM, array, cast_split, diag, 
                           flatten_invV_mats_to_kpts, get_V_mats, get_Z_mats, 
                           get_diag_extent_sqrd, get_grid_kpts, get_homog_xyzs, 
                           get_invVR_mats_oris, get_invVR_mats_oris_cyth, 
                           get_invVR_mats_shape, get_invVR_mats_shape_cyth, 
                           get_invVR_mats_sqrd_scale, 
                           get_invVR_mats_sqrd_scale_cyth, get_invVR_mats_xys, 
                           get_invVR_mats_xys_cyth, get_invV_mats, 
                           get_invV_mats2x2, get_invV_xy_axis_extents, 
                           get_invVs, get_kpts_bounds, get_kpts_strs, 
                           get_ori_mats, get_ori_strs, get_oris, get_scales, 
                           get_shape_strs, get_sqrd_scales, 
                           get_xy_axis_extents, get_xy_strs, get_xys, 
                           invert_invV_mats, matrix_multiply, offset_kpts, 
                           ones, rectify_invV_mats_are_up, 
                           rectify_invV_mats_are_up_cyth, rollaxis, sqrt, 
                           transform_kpts, transform_kpts_to_imgspace, zeros,) 
    from .patch import (find_kpts_direction, gaussian_patch, 
                        get_orientation_histogram, get_unwarped_patches, 
                        get_warped_patch, get_warped_patches, iprod, lru_cache, 
                        patch_gradient, patch_mag, patch_ori,) 
    from .chip import (compute_chip, get_filter_list, 
                       get_scaled_size_with_area, get_scaled_sizes_with_area,) 
    from .spatial_verification import (SV_DTYPE, build_lstsqrs_Mx9, 
                                       build_lstsqrs_Mx9_cyth, compute_homog, 
                                       compute_homog_cyth, 
                                       determine_best_inliers, 
                                       get_affine_inliers, 
                                       get_affine_inliers_cyth, 
                                       get_best_affine_inliers, 
                                       get_best_affine_inliers_cyth, 
                                       get_homography_inliers, 
                                       get_homography_inliers_cyth,) 
    from .trig import (atan2,) 
    from .math import (eps, tau,) 
    from .geometry import (bbox_of_verts, bboxes_from_vert_list, draw_verts, 
                           homogonize, homogonize_list, unhomogonize, 
                           unhomogonize_list, verts_from_bbox, 
                           verts_list_from_bboxes_list,) 
    from .clustering2 import (CLUSTERS_FNAME, akmeans, akmeans_iterations, 
                              akmeans_plusplus_init, apply_grouping, 
                              apply_grouping_iter, apply_grouping_iter2, 
                              apply_jagged_grouping, approximate_assignments, 
                              approximate_distances, assert_centroids, 
                              cached_akmeans, compute_centroids, double_group, 
                              get_akmeans_cfgstr, group_indicies, 
                              group_indicies_pandas, groupby, groupby_dict, 
                              groupby_gen, initialize_centroids, jagged_group, 
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
        for submodname, fromimports in IMPORT_TUPLES:
            submod = getattr(vtool, submodname)
            for attr in dir(submod):
                if attr.startswith('_'):
                    continue
                setattr(vtool, attr, getattr(submod, attr))
    
    
    def reload_subs(verbose=True):
        """ Reloads vtool and submodules """
        rrr(verbose=verbose)
        getattr(image, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(histogram, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(image, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(exif, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(keypoint, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(patch, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(chip, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(spatial_verification, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(trig, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(math, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(geometry, 'rrr', lambda verbose: None)(verbose=verbose)
        getattr(clustering2, 'rrr', lambda verbose: None)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception:
            pass
    rrrr = reload_subs
    # </AUTOGEN_INIT>