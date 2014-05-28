from __future__ import absolute_import, division, print_function
# Python
#from os.path import exists, join, split  # UNUSED
from os.path import splitext
# UTool
from itertools import izip
import utool
from vtool import image as gtool
from ibeis.model.detect import grabmodels
import pyrf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[detect_randomforest]', DEBUG=False)


def scale_bbox(bbox, s):
    (x, y, w, h) = bbox
    bbox2 = (int(s * x), int(s * y), int(s * w), int(s * h))
    return bbox2


def generate_detections(ibs, gid_list, species, **kwargs):
    """ kwargs can be: save_detection_images, save_scales, draw_supressed,
        detection_width, detection_height, percentage_left, percentage_top,
        nms_margin_percentage

        Yeilds tuples of image ids and bounding boxes
    """
    #
    # Get resized images to detect on
    src_gpath_list = ibs.get_image_detectpaths(gid_list)

    # Get sizes of the original and resized images for final scale correction
    neww_list = [gtool.open_image_size(gpath)[0] for gpath in src_gpath_list]
    oldw_list = [oldw for (oldw, oldh) in ibs.get_image_sizes(gid_list)]
    scale_list = [oldw / neww for oldw, neww in izip(oldw_list, neww_list)]

    # Detect on scaled images
    bboxes_list = detect_species_bboxes(src_gpath_list, species, **kwargs)

    for gid, scale, bboxes in izip(gid_list, scale_list, bboxes_list):
        # Unscale results
        unscaled_bboxes = [scale_bbox(bbox_, scale) for bbox_ in bboxes]
        for bbox in unscaled_bboxes:
            yield gid, bbox


def detect_species_bboxes(src_gpath_list, species, quick=True, **kwargs):
    """
    Generates bounding boxes for each source image
    For each image yeilds a list of bounding boxes
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

    trees_path = grabmodels.get_species_trees_paths(species)

    detect_config = {
        'percentage_top':    0.40,
    }
    detect_config.update(kwargs)

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, species + '-')

    print('')
    print('Begining detection')
    detect_lbl = 'detect %s ' % species
    nImgs = len(src_gpath_list)
    mark_prog, end_prog = utool.progress_func(nImgs, detect_lbl, flush_after=1)
    for ix in xrange(len(src_gpath_list)):
        mark_prog(ix)
        src_fpath = str(src_gpath_list[ix])
        dst_fpath = str(splitext(src_fpath)[0])

        results, timing = detector.detect(forest, src_fpath, dst_fpath,
                                          **detect_config)
        # Unpack unsupressed bounding boxes
        bboxes = [(minx, miny, (maxx - minx), (maxy - miny))
                  for (centx, centy, minx, miny, maxx, maxy, _, supressed)
                  in results if supressed == 0]
        yield bboxes
    end_prog()
