# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def dummy_bbox(img, shiftxy=(0.0, 0.0), scale=0.25):
    """Default to rectangle that has a quarter-width/height border."""
    (gh, gw) = img.shape[0:2]
    half_w = gw / 2
    half_h = gh / 2
    w = gw * scale
    h = gh * scale
    x = half_w - (w / 2) + shiftxy[0] * gw
    y = half_h - (h / 2) + shiftxy[1] * gh
    bbox = tuple(map(int, map(round, [x, y, w, h])))
    return bbox


def imread_many(imgpaths):
    import cv2

    img_list = [cv2.imread(img_fpath) for img_fpath in imgpaths]
    return img_list
