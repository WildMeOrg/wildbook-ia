# -*- coding: utf-8 -*-
"""Interface to Lightnet object proposals."""
from __future__ import absolute_import, division, print_function
import utool as ut
from six.moves import zip
import numpy as np
from os.path import abspath, dirname, expanduser, join, exists  # NOQA
import cv2
from tqdm import tqdm
(print, rrr, profile) = ut.inject2(__name__, '[lightnet]')


if not ut.get_argflag('--no-lightnet'):
    try:
        import torch
        from torchvision import transforms as tf
        import lightnet as ln
    except ImportError:
        print('WARNING Failed to import lightnet. '
              'PyDarknet YOLO detection is unavailable')
        if ut.SUPER_STRICT:
            raise


VERBOSE_LN = ut.get_argflag('--verbln') or ut.VERBOSE


WEIGHT_URL_DICT = {
    'seaturtle'     : 'https://lev.cs.rpi.edu/public/models/detect.lightnet.sea_turtle.weights',
    'hammerhead'    : 'https://lev.cs.rpi.edu/public/models/detect.lightnet.shark_hammerhead.weights',
    'ggr2'          : 'https://lev.cs.rpi.edu/public/models/detect.lightnet.ggr2.weights',
    'lynx'          : 'https://lev.cs.rpi.edu/public/models/detect.lightnet.lynx.weights',

    None            : 'https://lev.cs.rpi.edu/public/models/detect.lightnet.sea_turtle.weights',
}


def _parse_classes_from_weights(url):
    return url.replace('.weights', '.classes')


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


def detect_gid_list(ibs, gid_list, verbose=VERBOSE_LN, **kwargs):
    """Detect gid_list with lightnet.

    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs (optional): refer to the Lightnet documentation for configuration settings

    Args:
        ibs (ibeis.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs:
        detector, config_filepath, weights_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)
    """
    # Get new gpaths if downsampling
    gpath_list = ibs.get_image_paths(gid_list)

    # Run detection
    results_iter = detect(gpath_list, verbose=verbose, **kwargs)
    # Upscale the results
    _iter = zip(gid_list, results_iter)
    for gid, (gpath, result_list) in _iter:
        # Upscale the results back up to the original image size
        for result in result_list:
            bbox = (result['xtl'], result['ytl'], result['width'], result['height'], )
            bbox_list = [ bbox ]
            bbox = bbox_list[0]
            result['xtl'], result['ytl'], result['width'], result['height'] = bbox
        yield (gid, gpath, result_list)


def _create_network(weight_filepath, class_list, conf_thresh, nms_thresh, network_size):
    """Create the lightnet network."""
    net = ln.models.Yolo(len(class_list), weight_filepath, conf_thresh, nms_thresh)
    net.postprocess.append(ln.data.transform.TensorToBrambox(network_size, class_list))

    net.eval()
    if torch.cuda.is_available():
        net.cuda()

    return net


def _detect(net, img_path, network_size):
    """Perform a detection."""
    # Load image
    img = cv2.imread(img_path)
    im_h, im_w = img.shape[:2]

    img_tf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tf = ln.data.transform.Letterbox.apply(img_tf, dimension=network_size)
    img_tf = tf.ToTensor()(img_tf)
    img_tf.unsqueeze_(0)

    if torch.cuda.is_available():
        img_tf = img_tf.cuda()

    # Run detector
    if torch.__version__.startswith('0.3'):
        img_tf = torch.autograd.Variable(img_tf, volatile=True)
        out = net(img_tf)
    else:
        with torch.no_grad():
            out = net(img_tf)
    out = ln.data.transform.ReverseLetterbox.apply(out, network_size, (im_w, im_h))

    return img, out


def detect(gpath_list, config_filepath, weight_filepath, class_filepath, sensitivity,
           verbose=VERBOSE_LN, **kwargs):
    """Detect image filepaths with lightnet.

    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Lightnet documentation for configuration settings

    Returns:
        iter
    """
    assert config_filepath is None, 'lightnet does not have a config file'

    # Get correct weight if specified with shorthand
    weight_url = None
    if weight_filepath in WEIGHT_URL_DICT:
        weight_url = WEIGHT_URL_DICT[weight_filepath]
        weight_filepath = ut.grab_file_url(weight_url, appname='ibeis',
                                           check_hash=True)

    if class_filepath in WEIGHT_URL_DICT:
        if class_filepath is None and weight_url is not None:
            weight_url_ = weight_url
        else:
            weight_url_ = WEIGHT_URL_DICT[weight_filepath]
        class_url = _parse_classes_from_weights(weight_url_)
        class_filepath = ut.grab_file_url(class_url, appname='ibeis',
                                          check_hash=True, verbose=verbose)

    assert exists(weight_filepath)
    weight_filepath = ut.truepath(weight_filepath)
    assert exists(class_filepath)
    class_filepath = ut.truepath(class_filepath)

    class_list = _parse_class_list(class_filepath)

    network_size = (416, 416)
    conf_thresh = sensitivity
    nms_thresh = 1.0  # Turn off NMS
    network = _create_network(weight_filepath, class_list, conf_thresh,
                              nms_thresh, network_size)

    # Execute detector for each image
    results_list_ = []
    for gpath in tqdm(gpath_list):
        image, output_list = _detect(network, gpath, network_size)
        output_list = output_list[0]

        result_list_ = []
        for output in list(output_list):
            xtl = int(np.around(float(output.x_top_left)))
            ytl = int(np.around(float(output.y_top_left)))
            xbr = int(np.around(float(output.x_top_left + output.width)))
            ybr = int(np.around(float(output.y_top_left + output.height)))
            class_ = output.class_label
            conf = float(output.confidence)
            result_dict = {
                'xtl'        : xtl,
                'ytl'        : ytl,
                'width'      : xbr - xtl,
                'height'     : ybr - ytl,
                'class'      : class_,
                'confidence' : conf,
            }
            result_list_.append(result_dict)
        results_list_.append(result_list_)

    if len(results_list_) != len(gpath_list):
        raise ValueError('Lightnet did not return valid data')

    results_list = zip(gpath_list, results_list_)
    return results_list
