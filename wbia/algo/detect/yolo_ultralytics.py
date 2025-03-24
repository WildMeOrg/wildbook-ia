# -*- coding: utf-8 -*-
"""Interface to Lightnet object proposals."""
import logging
from os.path import abspath, dirname, exists, expanduser, join, splitext  # NOQA

import cv2
import numpy as np
import utool as ut
import vtool as vt
from tqdm import tqdm

from ultralytics import YOLO
import torch


(print, rrr, profile) = ut.inject2(__name__, '[yolo_ultralytics]')
logger = logging.getLogger('wbia')


if not ut.get_argflag('--no-yolo-ultralytics'):
    try:
        import lightnet as ln
        import torch
        from torchvision import transforms as tf
    except ImportError:
        logger.info(
            'WARNING Failed to import lightnet. '
            'PyDarknet YOLO detection is unavailable'
        )
        if ut.SUPER_STRICT:
            raise


VERBOSE_LN = ut.get_argflag('--verb-yolo-ultralytics') or ut.VERBOSE


CONFIG_URL_DICT = {
'msv3': 'https://cthulhu.dyn.wildme.io/public/models/detect.yolov11.msv3.pt',
'sea_turtle_new_v0': 'https://cthulhu.dyn.wildme.io/public/models/detect.yolov11.msv3.pt',
}



def detect_gid_list(ibs, gid_list, verbose=VERBOSE_LN, **kwargs):
    """Detect gid_list with YOLO model and return formatted results.

    Args:
        ibs (wbia.IBEISController): Image analysis API
        gid_list (list of int): List of IBEIS image_rowids that need detection

    Kwargs:
        detector, config_filepath, weight_filepath, verbose

    Yields:
        tuple: (gid, gpath, result_list)
    """
    # Get image paths
    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)
    weight_filepath = kwargs.get("weight_filepath", None)
    
    config_url = None
    if weight_filepath in CONFIG_URL_DICT:
        weight_url = CONFIG_URL_DICT[weight_filepath]
        weight_filepath = ut.grab_file_url(
            weight_url, appname='lightnet', check_hash=False
        )
    
    assert exists(weight_filepath), weight_filepath
    weight_filepath = ut.truepath(weight_filepath)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## Need to do to device?
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt") #('weight_filepath')
    print(weight_filepath)
    model = YOLO(weight_filepath)
    model = model.to(device)
    
    # Run detection
    results_iter = []
    for gpath in tqdm(gpath_list, desc="Running YOLO detections"):
        result = model.predict(gpath, conf=0.25, iou=0.7, imgsz=512, device=device)
        
        detection_results = []
        for box, cls, conf in zip(result[0].boxes.xywh, result[0].boxes.cls, result[0].boxes.conf):
            x_center, y_center, width, height = box.tolist()
            xtl = int(np.around(x_center - width / 2))
            ytl = int(np.around(y_center - height / 2))
            
            result_dict = {
                'xtl': xtl,
                'ytl': ytl,
                'width': int(np.around(width)),
                'height': int(np.around(height)),
                'class': int(cls.item()),
                'confidence': float(conf.item()),
            }
            detection_results.append(result_dict)
        
        results_iter.append((gpath, detection_results))
    
    # Upscale results and yield
    for gid, (gpath, result_list) in zip(gid_list, results_iter):
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
