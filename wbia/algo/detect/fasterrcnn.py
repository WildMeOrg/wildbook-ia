# -*- coding: utf-8 -*-
"""
Interface to Faster R-CNN object proposals.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import vtool as vt
from six.moves import zip, range
from os.path import abspath, dirname, expanduser, join, exists  # NOQA
import numpy as np
import sys
import cv2

(print, rrr, profile) = ut.inject2(__name__, '[faster r-cnn]')

# SCRIPT_PATH = abspath(dirname(__file__))
SCRIPT_PATH = abspath(expanduser(join('~', 'code', 'py-faster-rcnn')))

if not ut.get_argflag('--no-faster-rcnn'):
    try:
        assert exists(SCRIPT_PATH)

        def add_path(path):
            # if path not in sys.path:
            sys.path.insert(0, path)

        # Add pycaffe to PYTHONPATH
        pycaffe_path = join(SCRIPT_PATH, 'caffe-fast-rcnn', 'python')
        add_path(pycaffe_path)

        # Add caffe lib path to PYTHONPATH
        lib_path = join(SCRIPT_PATH, 'lib')
        add_path(lib_path)

        import caffe

        ut.reload_module(caffe)
        from fast_rcnn.config import cfg
        from fast_rcnn.test import im_detect

        # from fast_rcnn.nms_wrapper import nms
    except AssertionError:
        print('WARNING Failed to find py-faster-rcnn. ' 'Faster R-CNN is unavailable')
        # if ut.SUPER_STRICT:
        #     raise
    except ImportError:
        print('WARNING Failed to import fast_rcnn. ' 'Faster R-CNN is unavailable')
        # if ut.SUPER_STRICT:
        #     raise


VERBOSE_SS = ut.get_argflag('--verbdss') or ut.VERBOSE


CONFIG_URL_DICT = {
    # 'pretrained-fast-vgg-pascal' : 'https://wildbookiarepository.azureedge.net/models/pretrained.fastrcnn.vgg16.pascal.prototxt',  # Trained on PASCAL VOC 2007
    'pretrained-vgg-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.fasterrcnn.vgg16.pascal.prototxt',  # Trained on PASCAL VOC 2007
    'pretrained-zf-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.fasterrcnn.zf.pascal.prototxt',  # Trained on PASCAL VOC 2007
    'pretrained-vgg-ilsvrc': 'https://wildbookiarepository.azureedge.net/models/pretrained.fasterrcnn.vgg16.ilsvrc.prototxt',  # Trained on ILSVRC 2014
    'pretrained-zf-ilsvrc': 'https://wildbookiarepository.azureedge.net/models/pretrained.fasterrcnn.zf.ilsvrc.prototxt',  # Trained on ILSVRC 2014
    'default': 'https://wildbookiarepository.azureedge.net/models/pretrained.fasterrcnn.vgg16.pascal.prototxt',  # Trained on PASCAL VOC 2007
    None: 'https://wildbookiarepository.azureedge.net/models/pretrained.fasterrcnn.vgg16.pascal.prototxt',  # Trained on PASCAL VOC 2007
}


def _parse_weight_from_cfg(url):
    return url.replace('.prototxt', '.caffemodel')


def _parse_classes_from_cfg(url):
    return url.replace('.prototxt', '.classes')


def _parse_class_list(classes_filepath):
    # Load classes from file into the class list
    assert exists(classes_filepath)
    class_list = []
    with open(classes_filepath) as classes:
        for line in classes.readlines():
            line = line.strip()
            if len(line) > 0:
                class_list.append(line)
    return class_list


def detect_gid_list(ibs, gid_list, downsample=True, verbose=VERBOSE_SS, **kwargs):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the Faster R-CNN documentation for configuration settings

    Args:
        ibs (wbia.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
                sizes should be used; defaults to True

    Kwargs:
        detector, config_filepath, weights_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)

    CommandLine:
        python -m wbia.algo.detect.fasterrcnn detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.fasterrcnn import *  # NOQA
        >>> from wbia.core_images import LocalizerConfig
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> config = {'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
        >>> config = {'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()

    Yields:
        results (list of dict)
    """
    # Get new gpaths if downsampling
    if downsample:
        gpath_list = ibs.get_image_detectpaths(gid_list)
        neww_list = [vt.open_image_size(gpath)[0] for gpath in gpath_list]
        oldw_list = [oldw for (oldw, oldh) in ibs.get_image_sizes(gid_list)]
        downsample_list = [oldw / neww for oldw, neww in zip(oldw_list, neww_list)]
        orient_list = [1] * len(gid_list)
    else:
        gpath_list = ibs.get_image_paths(gid_list)
        downsample_list = [None] * len(gpath_list)
        orient_list = ibs.get_image_orientation(gid_list)
    # Run detection
    results_iter = detect(gpath_list, verbose=verbose, **kwargs)
    # Upscale the results
    _iter = zip(downsample_list, gid_list, orient_list, results_iter)
    for downsample, gid, orient, (gpath, result_list) in _iter:
        # Upscale the results back up to the original image size
        for result in result_list:
            if downsample is not None and downsample != 1.0:
                for key in ['xtl', 'ytl', 'width', 'height']:
                    result[key] = int(result[key] * downsample)
            bbox = (
                result['xtl'],
                result['ytl'],
                result['width'],
                result['height'],
            )
            bbox_list = [bbox]
            bbox = bbox_list[0]
            result['xtl'], result['ytl'], result['width'], result['height'] = bbox
        yield (gid, gpath, result_list)


def detect(
    gpath_list,
    config_filepath,
    weight_filepath,
    class_filepath,
    sensitivity,
    verbose=VERBOSE_SS,
    use_gpu=True,
    use_gpu_id=0,
    **kwargs,
):
    """
    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Faster R-CNN documentation for configuration settings

    Returns:
        iter
    """
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # Get correct config if specified with shorthand
    config_url = None
    if config_filepath in CONFIG_URL_DICT:
        config_url = CONFIG_URL_DICT[config_filepath]
        config_filepath = ut.grab_file_url(config_url, appname='wbia', check_hash=True)

    # Get correct weights if specified with shorthand
    if weight_filepath in CONFIG_URL_DICT:
        if weight_filepath is None and config_url is not None:
            config_url_ = config_url
        else:
            config_url_ = CONFIG_URL_DICT[weight_filepath]
        weight_url = _parse_weight_from_cfg(config_url_)
        weight_filepath = ut.grab_file_url(weight_url, appname='wbia', check_hash=True)

    if class_filepath is None:
        class_url = _parse_classes_from_cfg(config_url)
        class_filepath = ut.grab_file_url(
            class_url, appname='wbia', check_hash=True, verbose=verbose
        )
    class_list = _parse_class_list(class_filepath)

    # Need to convert unicode strings to Python strings to support Boost Python
    # call signatures in caffe
    prototxt_filepath = str(config_filepath)  # alias to Caffe nomenclature
    caffemodel_filepath = str(weight_filepath)  # alias to Caffe nomenclature

    assert exists(prototxt_filepath), 'Specified prototxt file not found'
    assert exists(caffemodel_filepath), 'Specified caffemodel file not found'

    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(use_gpu_id)
        cfg.GPU_ID = use_gpu_id
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(prototxt_filepath, caffemodel_filepath, caffe.TEST)

    # Warm-up network on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = im_detect(net, im)

    results_list_ = []
    for gpath in gpath_list:
        image = cv2.imread(gpath)
        score_list, bbox_list = im_detect(net, image)

        # Compile results
        result_list_ = []
        for class_index, class_name in enumerate(class_list[1:]):
            class_index += 1  # because we skipped background
            class_boxes = bbox_list[:, 4 * class_index : 4 * (class_index + 1)]
            class_scores = score_list[:, class_index]
            dets_list = np.hstack((class_boxes, class_scores[:, np.newaxis]))
            dets_list = dets_list.astype(np.float32)
            # # Perform NMS
            # keep_list = nms(dets_list, nms_sensitivity)
            # dets_list = dets_list[keep_list, :]
            # Perform sensitivity check
            keep_list = np.where(dets_list[:, -1] >= sensitivity)[0]
            dets_list = dets_list[keep_list, :]
            for (xtl, ytl, xbr, ybr, conf) in dets_list:
                xtl = int(np.around(xtl))
                ytl = int(np.around(ytl))
                xbr = int(np.around(xbr))
                ybr = int(np.around(ybr))
                confidence = float(conf)
                result_dict = {
                    'xtl': xtl,
                    'ytl': ytl,
                    'width': xbr - xtl,
                    'height': ybr - ytl,
                    'class': class_name,
                    'confidence': confidence,
                }
                result_list_.append(result_dict)
        results_list_.append(result_list_)

    results_list = zip(gpath_list, results_list_)
    return results_list
