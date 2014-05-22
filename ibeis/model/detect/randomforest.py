from __future__ import absolute_import, division, print_function
# Python
#from os.path import exists, join, split  # UNUSED
from os.path import join, splitext
# UTool
from itertools import izip
import utool
from vtool import image as gtool
from ibeis.model.detect import grabmodels
import pyrf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[detect_randomforest]', DEBUG=False)


@utool.indent_func
def detect_rois(ibs, gid_list, species, quick=True, **kwargs):
    """
    kwargs can be:
        save_detection_images
        save_scales
        draw_supressed
        detection_width
        detection_height
        percentage_left
        percentage_top
        nms_margin_percentage
    """
    # Ensure all models downloaded and accounted for
    grabmodels.ensure_models()

    # Create detector
    if quick:
        config = {}
    else:
        config = {
            'scales': '11 2.0 1.75 1.5 1.33 1.15 1.0 0.75 0.55 0.40 0.30 0.20'
        }

    detector = pyrf.Random_Forest_Detector(rebuild=False, **config)

    rf_model_dir   = grabmodels.MODEL_DIRS['rf']
    trees_path     = join(rf_model_dir, species)
    tree_prefix = species + '-'

    detect_config = {
        'percentage_top':    0.40,
    }
    detect_config.update(kwargs)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, tree_prefix)

    # Get resized images to detect on
    src_gpath_list = ibs.get_image_detectpaths(gid_list)

    # Get sizes of the original and resized images for final scale correction
    new_sizes_list = [gtool.open_image_size(src_fpath)
                      for src_fpath in src_gpath_list]
    old_sizes_list = ibs.get_image_sizes(gid_list)
    scale_list = [oldsize[0] / newsize[0] for oldsize, newsize in
                  izip(old_sizes_list, new_sizes_list)]

    detected_gid_list  = []
    detected_bbox_list = []

    for ix in xrange(len(src_gpath_list)):
        gid = gid_list[ix]
        src_fpath = str(src_gpath_list[ix])
        dst_fpath = str(splitext(src_fpath)[0])
        scale = scale_list[ix]

        print('Processing [at scale %.2f]: ' % (scale) + src_fpath)
        results, timing = detector.detect(forest, src_fpath, dst_fpath,
                                          **detect_config)
        for res in results:
            # 0 centerx, 1 centery, 2 minx, 3 miny,
            # 4 maxx, 5 maxy, 6 0.0 [UNUSED], 7 supressed flag
            (centerx, centery, minx, miny, maxx, maxy, unused, supressed) = res
            if supressed == 0:
                # Perform final scale correction
                x = int(scale * (minx))
                y = int(scale * (miny))
                w = int(scale * (maxx - minx))
                h = int(scale * (maxy - miny))
                bbox = (x, y, w, h)
                # Append results
                detected_gid_list.append(gid)
                detected_bbox_list.append(bbox)

    detections = (detected_gid_list, detected_bbox_list)
    return detections
