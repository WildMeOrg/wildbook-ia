# -*- coding: utf-8 -*-
"""
Interface to SSD object proposals.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import vtool as vt
from six.moves import zip
from os.path import abspath, dirname, expanduser, join, exists  # NOQA
import numpy as np
import sys

(print, rrr, profile) = ut.inject2(__name__, '[ssd]')

# SCRIPT_PATH = abspath(dirname(__file__))
SCRIPT_PATH = abspath(expanduser(join('~', 'code', 'ssd')))

if not ut.get_argflag('--no-ssd'):
    try:
        assert exists(SCRIPT_PATH)

        def add_path(path):
            # if path not in sys.path:
            sys.path.insert(0, path)

        # Add pycaffe to PYTHONPATH
        pycaffe_path = join(SCRIPT_PATH, 'python')
        add_path(pycaffe_path)

        import caffe

        rrr(caffe)
        from google.protobuf import text_format
        from caffe.proto import caffe_pb2
    except AssertionError:
        print('WARNING Failed to find ssd. ' 'SSD is unavailable')
        # if ut.SUPER_STRICT:
        #     raise
    except ImportError:
        print('WARNING Failed to import caffe. ' 'SSD is unavailable')
        # if ut.SUPER_STRICT:
        #     raise


VERBOSE_SS = ut.get_argflag('--verbssd') or ut.VERBOSE


CONFIG_URL_DICT = {
    'pretrained-300-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.300.pascal.prototxt',
    'pretrained-512-pascal': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.512.pascal.prototxt',
    'pretrained-300-pascal-plus': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.300.pascal.plus.prototxt',
    'pretrained-512-pascal-plus': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.512.pascal.plus.prototxt',
    'pretrained-300-coco': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.300.coco.prototxt',
    'pretrained-512-coco': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.512.coco.prototxt',
    'pretrained-300-ilsvrc': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.300.ilsvrc.prototxt',
    'pretrained-500-ilsvrc': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.500.ilsvrc.prototxt',
    'default': 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.512.pascal.plus.prototxt',
    None: 'https://wildbookiarepository.azureedge.net/models/pretrained.ssd.512.pascal.plus.prototxt',
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

    Kwargs (optional): refer to the SSD documentation for configuration settings

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
        python -m wbia.algo.detect.ssd detect_gid_list --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.ssd import *  # NOQA
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

    Kwargs (optional): refer to the SSD documentation for configuration settings

    Returns:
        iter
    """

    def _get_label_name(class_labelmap, label_list):
        if not isinstance(label_list, list):
            label_list = [label_list]
        item_list = class_labelmap.item
        name_list = []
        for label in label_list:
            found = False
            for i in range(len(item_list)):
                if label == item_list[i].label:
                    found = True
                    name_list.append(item_list[i].display_name)
                    break
            assert found
        return name_list

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

    # load class labels
    with open(class_filepath, 'r') as class_file:
        class_labelmap = caffe_pb2.LabelMap()
        class_str = str(class_file.read(class_file))
        text_format.Merge(class_str, class_labelmap)

    # Need to convert unicode strings to Python strings to support Boost Python
    # call signatures in caffe
    prototxt_filepath = str(config_filepath)  # alias to Caffe nomenclature
    caffemodel_filepath = str(weight_filepath)  # alias to Caffe nomenclature

    assert exists(prototxt_filepath), 'Specified prototxt file not found'
    assert exists(caffemodel_filepath), 'Specified caffemodel file not found'

    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(use_gpu_id)
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(prototxt_filepath, caffemodel_filepath, caffe.TEST)

    # Determine input size from prototext
    with open(prototxt_filepath, 'r') as prototxt_file:
        # load all lines
        line_list = prototxt_file.readlines()
        # look for dim size lines
        line_list = [line for line in line_list if 'dim:' in line]
        line_list = line_list[:4]
        # Get last line
        line = line_list[-1]
        line_ = line.strip().split(' ')
        # Filter empty spaces
        line_ = [_ for _ in line_ if len(_) > 0]
        # Get last value on line, which should be the image size
        image_resize = int(line_[-1])
        # Check to make sure
        assert image_resize in [300, 500, 512]
        print('FOUND image_resize = %r' % (image_resize,))

    # Input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # Mean pixel value
    transformer.set_mean('data', np.array([104, 117, 123]))
    # The reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_raw_scale('data', 255)
    # The reference model has channels in BGR order instead of RGB
    transformer.set_channel_swap('data', (2, 1, 0))
    # Set batch size to 1 and set testing image size
    net.blobs['data'].reshape(1, 3, image_resize, image_resize)

    results_list_ = []
    for gpath in gpath_list:
        image = caffe.io.load_image(gpath)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= sensitivity]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = _get_label_name(class_labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        height, width = image.shape[:2]

        # Compile results
        result_list_ = []
        zipped = zip(top_xmin, top_ymin, top_xmax, top_ymax, top_labels, top_conf)
        for (xmin, ymin, xmax, ymax, label, conf) in zipped:
            xtl = int(np.around(xmin * width))
            ytl = int(np.around(ymin * height))
            xbr = int(np.around(xmax * width))
            ybr = int(np.around(ymax * height))
            confidence = float(conf)
            result_dict = {
                'xtl': xtl,
                'ytl': ytl,
                'width': xbr - xtl,
                'height': ybr - ytl,
                'class': label,
                'confidence': confidence,
            }
            result_list_.append(result_dict)
        results_list_.append(result_list_)

    results_list = zip(gpath_list, results_list_)
    return results_list
