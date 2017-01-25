# -*- coding: utf-8 -*-
"""
Interface to Faster R-CNN object proposals.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import vtool as vt
from six.moves import zip
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
            if path not in sys.path:
                sys.path.insert(0, path)

        # Add pycaffe to PYTHONPATH
        pycaffe_path = join(SCRIPT_PATH, 'caffe-fast-rcnn', 'python')
        add_path(pycaffe_path)

        # Add caffe lib path to PYTHONPATH
        lib_path = join(SCRIPT_PATH, 'lib')
        add_path(lib_path)

        import caffe
        from fast_rcnn.config import cfg
        from fast_rcnn.test import im_detect
        from fast_rcnn.nms_wrapper import nms
    except AssertionError as ex:
        print('WARNING Failed to find faster r-cnn. '
              'Faster R-CNN is unavailable')
        if ut.SUPER_STRICT:
            raise

CLASS_LIST = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']

# NETS = {'vgg16': ('VGG16',
#                   'VGG16_faster_rcnn_final.caffemodel'),
#         'zf': ('ZF',
#                   'ZF_faster_rcnn_final.caffemodel')}

VERBOSE_SS = ut.get_argflag('--verbdss') or ut.VERBOSE


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
        ibs (ibeis.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        downsample (bool, optional): a flag to indicate if the original image
                sizes should be used; defaults to True

    Kwargs:
        detector, config_filepath, weights_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)

    CommandLine:
        python -m ibeis.algo.detect.fasterrcnn detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.detect.fasterrcnn import *  # NOQA
        >>> from ibeis.core_images import LocalizerConfig
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> config = {'matlab_command': 'faster_rcnn', 'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
        >>> config = {'matlab_command': 'faster_rcnn_rcnn', 'verbose': True}
        >>> downsample = False
        >>> results_list = detect_gid_list(ibs, gid_list, downsample, **config)
        >>> results_list = list(results_list)
        >>> print('result lens = %r' % (map(len, list(results_list))))
        >>> print('result[0] = %r' % (len(list(results_list[0][2]))))
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
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
            bbox = (result['xtl'], result['ytl'], result['width'], result['height'], )
            bbox_list = [ bbox ]
            bbox_list = ibs.fix_horizontal_bounding_boxes_to_orient(gid, bbox_list)
            bbox = bbox_list[0]
            result['xtl'], result['ytl'], result['width'], result['height'] = bbox
        yield (gid, gpath, result_list)


def detect(gpath_list, config_filepath, weight_filepath, verbose=VERBOSE_SS,
           use_gpu=True, use_gpu_id=0, sensitivity=0.8, nms_sensitivity=0.2,
           **kwargs):
    """
    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Faster R-CNN documentation for configuration settings

    Returns:
        iter
    """
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt_filepath = config_filepath  # alias to Caffe nomenclature
    caffemodel_filepath = weight_filepath  # alias to Caffe nomenclature

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
    for i in xrange(2):
        _, _ = im_detect(net, im)

    results_list = []
    for gpath in gpath_list:
        image = cv2.imread(gpath)
        score_list, bbox_list = im_detect(net, image)

        ut.embed()

        for cls_ind, cls in enumerate(CLASS_LIST[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = bbox_list[:, 4 * cls_ind: 4 * (cls_ind + 1)]
            cls_scores = score_list[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, nms_sensitivity)
            dets = dets[keep, :]

    # Pack results
    results_list_ = []
    for result_list in results_list:
        result_list_ = []
        for result in result_list:
            xtl = int(np.around(result[0]))
            ytl = int(np.around(result[1]))
            xbr = int(np.around(result[2]))
            ybr = int(np.around(result[3]))
            result_dict = {
                'xtl'        : xtl,
                'ytl'        : ytl,
                'width'      : xbr - xtl,
                'height'     : ybr - ytl,
                'class'      : None,
                'confidence' : 1.0,
            }
            result_list_.append(result_dict)
        results_list_.append(result_list_)

    results_list = zip(gpath_list, results_list_)
    return results_list
