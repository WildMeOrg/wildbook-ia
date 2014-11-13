"""
Interface to pyrf random forest object detection.
"""
from __future__ import absolute_import, division, print_function
# Python
#from os.path import exists, join, split  # UNUSED
from os.path import splitext, exists
# UTool
from six.moves import zip, map, range
import utool
import utool as ut
from vtool import image as gtool
from ibeis.model.detect import grabmodels
import pyrf
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[randomforest]', DEBUG=False)


"""

from ibeis.model.detect import randomforest
dir(randomforest)
import utool
func = randomforest.ibeis_generate_image_detections
print(utool.make_default_docstr(func))

"""


#=================
# IBEIS INTERFACE
#=================


def ibeis_generate_image_detections(ibs, gid_list, species, **detectkw):
    """
    detectkw can be: save_detection_images, save_scales, draw_supressed,
    detection_width, detection_height, percentage_left, percentage_top,
    nms_margin_percentage

    Args:
        ibs (IBEISController):
        gid_list (list):
        species (?):

    Yeilds:
        tuple: tuples of image ids and bounding boxes
    """
    if True or ut.VERBOSE:
        print('[randomforest] +--- BEGIN ibeis_generate_image_detections')
        print('[randomforest] * species = %r' % (species,))
        print('[randomforest] * len(gid_list) = %r' % (len(gid_list),))
    #
    # Resize to a standard image size prior to detection
    src_gpath_list = list(map(str, ibs.get_image_detectpaths(gid_list)))
    dst_gpath_list = [splitext(gpath)[0] + '_result' for gpath in src_gpath_list]
    utool.close_pool()

    # Get sizes of the original and resized images for final scale correction
    neww_list = [gtool.open_image_size(gpath)[0] for gpath in src_gpath_list]
    oldw_list = [oldw for (oldw, oldh) in ibs.get_image_sizes(gid_list)]
    scale_list = [oldw / neww for oldw, neww in zip(oldw_list, neww_list)]

    # Detect on scaled images
    #ibs.cfg.other_cfg.ensure_attr('detect_use_chunks', True)
    use_chunks = ibs.cfg.other_cfg.detect_use_chunks

    generator = detect_species_bboxes(src_gpath_list, dst_gpath_list, species,
                                      use_chunks=use_chunks, **detectkw)

    for gid, scale, detect_tup in zip(gid_list, scale_list, generator):
        (bboxes, confidences, img_conf) = detect_tup
        # Unscale results
        unscaled_bboxes = [_scale_bbox(bbox_, scale) for bbox_ in bboxes]
        for index in range(len(unscaled_bboxes)):
            bbox = unscaled_bboxes[index]
            confidence = float(confidences[index])
            yield gid, bbox, confidence, img_conf
    if True or ut.VERBOSE:
        print('[randomforest] L___ FINISH ibeis_generate_image_detections')


def compute_hough_images(src_gpath_list, dst_gpath_list, species, use_chunks=True, quick=True):
    """
    Args:
        src_gpath_list (list):
        dst_gpath_list (list):
        species (?):
        quick (bool):

    Returns:
        None

        import utool; utool.print_auto_docstr('ibeis.model.detect.randomforest', 'compute_hough_images')
    """
    # Define detect dict
    detectkw = {
        'quick': quick,
        'save_detection_images': True,
        'save_scales': False,
    }
    _compute_hough(src_gpath_list, dst_gpath_list, species, use_chunks=use_chunks, **detectkw)


def compute_probability_images(src_gpath_list, dst_gpath_list, species, use_chunks=True, quick=False):
    """
    Args:
        src_gpath_list (list):
        dst_gpath_list (list):
        species (?):
        quick (bool):

    Returns:
        None

    import utool; utool.print_auto_docstr('ibeis.model.detect.randomforest', 'compute_hough_images')

    """
    # Define detect dict
    detectkw = {
        'single': True,  # single is True = probability, False is hough
        'quick': quick,
        'save_detection_images': True,
        'save_scales': False,
    }
    _compute_hough(src_gpath_list, dst_gpath_list, species, **detectkw)


def _compute_hough(src_gpath_list, dst_gpath_list, species, use_chunks=True, **detectkw):
    """
    FIXME. This name is not accurate

    """
    assert len(src_gpath_list) == len(dst_gpath_list)

    # FIXME: images are not invalidated so this cache doesnt always work
    isvalid_list = [exists(gpath + '.png') for gpath in dst_gpath_list]
    dirty_src_gpaths = utool.get_dirty_items(src_gpath_list, isvalid_list)
    dirty_dst_gpaths = utool.get_dirty_items(dst_gpath_list, isvalid_list)
    num_dirty = len(dirty_src_gpaths)
    if num_dirty > 0:
        if utool.VERBOSE:
            print('[detect.rf] making hough images for %d images' % num_dirty)
        generator = detect_species_bboxes(dirty_src_gpaths, dirty_dst_gpaths, species,
                                          use_chunks=use_chunks, **detectkw)
        # Execute generator
        for tup in generator:
            pass


#=================
# HELPER FUNCTIONS
#=================


def _scale_bbox(bbox, s):
    """
    Args:
        bbox (tuple): bounding box
        s (float): scale factor

    Returns:
        bbox2
    """
    bbox_scaled = (s * _ for _ in bbox)
    bbox_round = list(map(round, bbox_scaled))
    bbox_int   = list(map(int,   bbox_round))
    bbox2      = tuple(bbox_int)
    return bbox2


def _get_detector(species, quick=True, single=False):
    """
    _get_detector

    Args:
        species (?):
        quick (bool):
        single (bool):

    Returns:
        tuple: (detector, forest)

    Example1:
        >>> # Test existing model
        >>> from ibeis.model.detect.randomforest import *  # NOQA
        >>> from ibeis.model.detect import randomforest
        >>> species = 'zebra_plains'
        >>> quick = True
        >>> single = False
        >>> (detector, forest) = randomforest._get_detector(species, quick, single)
        >>> print(list(map(type, (detector, forest))))
        [<class 'pyrf._pyrf.Random_Forest_Detector'>, <type 'int'>]

    Example2:
        >>> # Test non-existing model
        >>> from ibeis.model.detect.randomforest import *  # NOQA
        >>> from ibeis.model.detect import randomforest
        >>> species = 'dropbear'
        >>> quick = True
        >>> single = False
        >>> (detector, forest) = randomforest._get_detector(species, quick, single)
        >>> print(list(map(type, (detector, forest))))
        [<type 'NoneType'>, <type 'NoneType'>]

    """
    # Ensure all models downloaded and accounted for
    grabmodels.ensure_models()
    # Create detector
    if single:
        if quick:
            print("[detect.rf] Running quick (single scale) probability image")
            config = {
                'scales': '1 1.0',
                'out_scale': 255,
                'pos_like': 1,
            }
        else:
            config = {
                'out_scale': 255,
                'pos_like': 1,
            }
    else:
        if quick:
            config = {}
        else:
            config = {
                'scales': '11 2.0 1.75 1.5 1.33 1.15 1.0 0.75 0.55 0.40 0.30 0.20',
            }
    print('[randomforest] building detector')
    detector = pyrf.Random_Forest_Detector(rebuild=False, **config)
    trees_path = grabmodels.get_species_trees_paths(species)
    if utool.checkpath(trees_path, verbose=True):
        # Load forest, so we don't have to reload every time
        print('[randomforest] loading forest')
        forest = detector.load(trees_path, species + '-', num_trees=25)
        # TODO: WE NEED A WAY OF ASKING IF THE LOAD WAS SUCCESSFUL
        # SO WE CAN HANDLE IT GRACEFULLY FROM PYTHON
        return detector, forest
    else:
        # If the models do not exist return None
        return None, None


def _get_detect_config(**detectkw):
    detect_config = {
        'percentage_top':    0.40,
    }
    detect_config.update(detectkw)
    return detect_config


#=================
# PYRF INTERFACE
#=================


def detect_species_bboxes(src_gpath_list, dst_gpath_list, species, quick=True,
                          single=False, use_chunks=False, **detectkw):
    """
    Generates bounding boxes for each source image
    For each image yeilds a list of bounding boxes
    """
    nImgs = len(src_gpath_list)
    print('[detect.rf] Begining %s detection' % (species,))
    detect_lbl = 'detect %s: ' % species
    #mark_prog, end_prog = utool.progress_func(nImgs, detect_lbl, flush_after=1)

    detect_config = _get_detect_config(**detectkw)
    detector, forest = _get_detector(species, quick=quick, single=single)
    if detector is None:
        raise StopIteration('species=%s does not have models trained' % (species,))
    detector.set_detect_params(**detect_config)

    chunksize = 8
    use_chunks_ = use_chunks and nImgs >= chunksize

    print('[rf] src_gpath_list = ' + ut.truncate_str(ut.list_str(src_gpath_list)))
    print('[rf] dst_gpath_list = ' + ut.truncate_str(ut.list_str(dst_gpath_list)))
    if use_chunks_:
        print('[rf] detect in chunks')
        pathtup_iter = list(zip(src_gpath_list, dst_gpath_list))
        chunk_iter = utool.ichunks(pathtup_iter, chunksize)
        chunk_progiter = ut.ProgressIter(chunk_iter, lbl=detect_lbl,
                                         nTotal=int(len(pathtup_iter) / chunksize), freq=1)
        for ic, chunk in enumerate(chunk_progiter):
            src_gpath_list = [tup[0] for tup in chunk]
            dst_gpath_list = [tup[1] for tup in chunk]
            #mark_prog(ic * chunksize)
            results_list = detector.detect_many(forest, src_gpath_list, dst_gpath_list, use_openmp=True)

            for results in results_list:
                bboxes = [(minx, miny, (maxx - minx), (maxy - miny))
                          for (centx, centy, minx, miny, maxx, maxy, confidence, supressed)
                          in results if supressed == 0]

                #x_arr = results[:, 2]
                #y_arr = results[:, 3]
                #w_arr = results[:, 4] - results[:, 2]
                #h_arr = results[:, 5] - results[:, 3]
                #bboxes = np.hstack((x_arr, y_arr, w_arr, h_arr))
                # Unpack unsupressed bounding boxes

                confidences = [confidence
                               for (centx, centy, minx, miny, maxx, maxy, confidence, supressed)
                               in results if supressed == 0]

                if len(results) > 0:
                    image_confidence = max([float(result[6]) for result in results])
                else:
                    image_confidence = 0.0

                yield bboxes, confidences, image_confidence
    else:
        print('[rf] detect one image at a time')
        pathtup_iter = zip(src_gpath_list, dst_gpath_list)
        pathtup_progiter = ut.ProgressIter(pathtup_iter, lbl=detect_lbl,
                                           nTotal=len(pathtup_iter), freq=1)
        for ix, (src_gpath, dst_gpath) in enumerate(pathtup_progiter):
            #mark_prog(ix)
            results = detector.detect(forest, src_gpath, dst_gpath)
            bboxes = [(minx, miny, (maxx - minx), (maxy - miny))
                      for (centx, centy, minx, miny, maxx, maxy, confidence, supressed)
                      in results if supressed == 0]

            confidences = [confidence
                           for (centx, centy, minx, miny, maxx, maxy, confidence, supressed)
                           in results if supressed == 0]

            if len(results) > 0:
                image_confidence = max([float(result[6]) for result in results])
            else:
                image_confidence = 0.0

            yield bboxes, confidences, image_confidence
    #end_prog()


if __name__ == '__main__':
    """
    CommandLine:
        python ibeis/model/detect/randomforest.py --test-_get_detector

    """
    testable_list = [
        _get_detector
    ]
    ut.doctest_funcs(testable_list)
