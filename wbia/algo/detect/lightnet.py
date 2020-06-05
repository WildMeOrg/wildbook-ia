# -*- coding: utf-8 -*-
"""Interface to Lightnet object proposals."""
from __future__ import absolute_import, division, print_function
import utool as ut
from six.moves import zip
import numpy as np
from os.path import abspath, dirname, expanduser, join, exists, splitext  # NOQA
from tqdm import tqdm
import cv2

(print, rrr, profile) = ut.inject2(__name__, '[lightnet]')


if not ut.get_argflag('--no-lightnet'):
    try:
        import torch
        from torchvision import transforms as tf
        import lightnet as ln
    except ImportError:
        print(
            'WARNING Failed to import lightnet. '
            'PyDarknet YOLO detection is unavailable'
        )
        if ut.SUPER_STRICT:
            raise


VERBOSE_LN = ut.get_argflag('--verbln') or ut.VERBOSE


CONFIG_URL_DICT = {
    'hammerhead': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.shark_hammerhead.py',
    'lynx': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.lynx.py',
    'manta': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.manta_ray_giant.py',
    'seaturtle': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.sea_turtle.py',
    'rightwhale': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.rightwhale.v1.py',
    'rightwhale_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.rightwhale.v1.py',
    'rightwhale_v2': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.rightwhale.v2.py',
    'rightwhale_v3': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.rightwhale.v3.py',
    'jaguar_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.jaguar.v1.py',
    'jaguar_v2': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.jaguar.v2.py',
    'jaguar': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.jaguar.v2.py',
    'giraffe_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.giraffe.v1.py',
    'zebra_mountain_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.zebra_mountain.v0.py',
    'hendrik_elephant': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.hendrik.elephant.py',
    'hendrik_elephant_ears': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.hendrik.elephant.ears.py',
    'hendrik_elephant_ears_left': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.hendrik.elephant.ears.left.py',
    'hendrik_dorsal': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.hendrik.dorsal.py',
    'humpback_dorsal': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.whale_humpback.dorsal.v0.py',
    'orca_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.whale_orca.v0.py',
    'fins_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.fins.v0.py',
    'nassau_grouper_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.grouper_nassau.v0.py',
    'spotted_dolphin_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.dolphin_spotted.v0.py',
    'spotted_skunk_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.skunk_spotted.v0.py',
    'spotted_skunk_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.skunk_spotted.v1.py',
    'spotted_dolphin_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.dolphin_spotted.v1.py',
    'seadragon_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.seadragon.v0.py',
    'seadragon_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.seadragon.v1.py',
    'iot_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.iot.v0.py',
    'wilddog_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.wild_dog.v0.py',
    'leopard_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.leopard.v0.py',
    'cheetah_v1': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.cheetah.v1.py',
    'kitsci_v0': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.kitsci.v0.py',
    'candidacy': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.candidacy.py',
    'ggr2': 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.ggr2.py',
    None: 'https://wildbookiarepository.azureedge.net/models/detect.lightnet.candidacy.py',
    'training_kit': 'https://wildbookiarepository.azureedge.net/data/lightnet-training-kit.zip',
}


def _download_training_kit():
    training_kit_url = CONFIG_URL_DICT['training_kit']
    training_kit_path = ut.grab_zipped_url(training_kit_url, appname='lightnet')
    return training_kit_path


def _parse_weights_from_cfg(url):
    return url.replace('.py', '.weights')


def _parse_class_list(config_filepath):
    # Load classes from file into the class list
    params = ln.engine.HyperParameters.from_file(config_filepath)
    class_list = params.class_label_map
    return class_list


def detect_gid_list(ibs, gid_list, verbose=VERBOSE_LN, **kwargs):
    """Detect gid_list with lightnet.

    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs (optional): refer to the Lightnet documentation for configuration settings

    Args:
        ibs (wbia.IBEISController):  image analysis api
        gid_list (list of int): the list of IBEIS image_rowids that need detection

    Kwargs:
        detector, config_filepath, weight_filepath, verbose

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


def _create_network(
    config_filepath, weight_filepath, conf_thresh, nms_thresh, multi=False
):
    """Create the lightnet network."""
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('[lightnet] CUDA enabled')
        device = torch.device('cuda')
    else:
        print('[lightnet] CUDA not available')

    params = ln.engine.HyperParameters.from_file(config_filepath)
    params.load(weight_filepath)
    params.device = device

    # Update conf_thresh and nms_thresh in postpsocess
    params.network.postprocess[0].conf_thresh = conf_thresh
    params.network.postprocess[1].nms_thresh = nms_thresh

    if multi:
        import torch.nn as nn
        import lightnet.data as lnd

        # Add serialization to Brambox Detections for DataParallel
        postprocess_list = list(params.network.postprocess)
        postprocess_list.append(lnd.transform.SerializeBrambox())
        params.network.postprocess = lnd.transform.Compose(postprocess_list)

        # Make mult-GPU
        params.network = nn.DataParallel(params.network)

    params.network.eval()
    params.network.to(params.device)

    return params


def _detect(params, gpath_list, flip=False):
    """Perform a detection."""
    # Load image
    imgs = []
    img_sizes = []
    for gpath in gpath_list:
        img = cv2.imread(gpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip:
            img = cv2.flip(img, 1)

        img_h, img_w = img.shape[:2]
        img_size = (
            img_w,
            img_h,
        )
        img_sizes.append(img_size)

        img = ln.data.transform.Letterbox.apply(img, dimension=params.input_dimension)
        img = tf.ToTensor()(img)
        imgs.append(img)

    imgs = torch.stack(imgs)
    if len(imgs.shape) != 4:
        imgs.unsqueeze_(0)

    if torch.cuda.is_available():
        imgs = imgs.cuda()

    # ut.embed()

    # Run detector
    if torch.__version__.startswith('0.3'):
        imgs_tf = torch.autograd.Variable(imgs, volatile=True)
        out = params.network(imgs_tf)
    else:
        with torch.no_grad():
            out = params.network(imgs)

    result_list = []
    for result, img_size in zip(out, img_sizes):
        result = ln.data.transform.ReverseLetterbox.apply(
            [result], params.input_dimension, img_size
        )
        result = result[0]
        result_list.append(result)

    return result_list, img_sizes


def detect(
    gpath_list,
    config_filepath=None,
    weight_filepath=None,
    classes_filepath=None,
    sensitivity=0.0,
    verbose=VERBOSE_LN,
    flip=False,
    batch_size=192,
    **kwargs,
):
    """Detect image filepaths with lightnet.

    Args:
        gpath_list (list of str): the list of image paths that need proposal candidates

    Kwargs (optional): refer to the Lightnet documentation for configuration settings

    Returns:
        iter
    """
    # Get correct weight if specified with shorthand
    config_url = None
    if config_filepath in CONFIG_URL_DICT:
        config_url = CONFIG_URL_DICT[config_filepath]
        config_filepath = ut.grab_file_url(
            config_url, appname='lightnet', check_hash=True
        )

    # Get correct weights if specified with shorthand
    if weight_filepath in CONFIG_URL_DICT:
        if weight_filepath is None and config_url is not None:
            config_url_ = config_url
        else:
            config_url_ = CONFIG_URL_DICT[weight_filepath]
        weight_url = _parse_weights_from_cfg(config_url_)
        weight_filepath = ut.grab_file_url(
            weight_url, appname='lightnet', check_hash=True
        )

    assert exists(config_filepath)
    config_filepath = ut.truepath(config_filepath)
    assert exists(weight_filepath)
    weight_filepath = ut.truepath(weight_filepath)

    conf_thresh = sensitivity
    nms_thresh = 1.0  # Turn off NMS

    params = _create_network(config_filepath, weight_filepath, conf_thresh, nms_thresh)

    # Execute detector for each image
    results_list_ = []
    for gpath_batch_list in tqdm(list(ut.ichunks(gpath_list, batch_size))):
        try:
            result_list, img_sizes = _detect(params, gpath_batch_list, flip=flip)
        except cv2.error:
            result_list, img_sizes = [], []

        for result, img_size in zip(result_list, img_sizes):
            img_w, img_h = img_size

            result_list_ = []
            for output in list(result):
                xtl = int(np.around(float(output.x_top_left)))
                ytl = int(np.around(float(output.y_top_left)))
                xbr = int(np.around(float(output.x_top_left + output.width)))
                ybr = int(np.around(float(output.y_top_left + output.height)))
                width = xbr - xtl
                height = ybr - ytl
                class_ = output.class_label
                conf = float(output.confidence)
                if flip:
                    xtl = img_w - xbr
                result_dict = {
                    'xtl': xtl,
                    'ytl': ytl,
                    'width': width,
                    'height': height,
                    'class': class_,
                    'confidence': conf,
                }
                result_list_.append(result_dict)
            results_list_.append(result_list_)

    if len(results_list_) != len(gpath_list):
        raise ValueError('Lightnet did not return valid data')

    results_list = zip(gpath_list, results_list_)
    return results_list
