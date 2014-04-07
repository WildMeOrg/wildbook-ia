# LICENCE
from __future__ import print_function, division
# Science
import numpy as np
import numpy.linalg as npl
# VTool
from . import linalg as ltool
from . import image as gtool
from . import image_filters as gfilt_tool
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[chip]', DEBUG=False)


def _get_image_to_chip_transform(bbox, chipsz, theta):
    """ transforms image space into chipspace
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box
    """
    (x, y, w, h) = bbox
    (w_, h_)     = chipsz
    sx = (w_ / w)
    sy = (h_ / h)
    tx = -(x + (w / 2))
    ty = -(y + (h / 2))

    T1 = ltool.translation_mat(tx, ty)
    S  = ltool.scale_mat(sx, sy)
    R  = ltool.rotation_mat(-theta)
    T2 = ltool.translation_mat((w_ / 2),  (h_ / 2))
    C = T2.dot(R.dot(S.dot(T1)))
    return C


def _get_chip_to_image_transform(bbox, chipsz, theta):
    """ transforms chip space into imgspace
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box
    """
    C    = _get_image_to_chip_transform(bbox, chipsz, theta)
    invC = npl.inv(C)
    return invC


def _extract_chip(img_fpath, bbox, theta, new_size):
    """ Crops chip from image ; Rotates and scales; """
    imgBGR = gtool.imread(img_fpath)  # Read parent image
    M = _get_image_to_chip_transform(bbox, new_size, theta)  # Build transformation
    chipBGR = gtool.warpAffine(imgBGR, M, new_size)  # Rotate and scale
    return chipBGR


def _filter_chip(chipBGR, filter_funcs):
    """ applies a list of preprocessing filters to a chip """
    chipBGR_ = chipBGR
    for func in filter_funcs:
        chipBGR_ = func(chipBGR)
    return chipBGR_


def get_scaled_size_with_area(target_area, w, h):
    """ returns new_size which scales (w, h) as close to target_area as possible
    and maintains aspect ratio
    """
    ht = np.sqrt(target_area * h / w)
    wt = w * ht / h
    new_size = (int(round(wt)), int(round(ht)))
    return new_size


def get_scaled_sizes_with_area(target_area, size_list):
    return [get_scaled_size_with_area(target_area, w, h) for (w, h) in size_list]


def compute_chip(img_fpath, bbox, theta, new_size, filter_list=[]):
    """ Extracts a chip and applies filters """
    chipBGR = _extract_chip(img_fpath, bbox, theta, new_size)
    chipBGR = _filter_chip(chipBGR, filter_list)
    return chipBGR


def get_filter_list(chipcfg_dict):
    filter_list = []
    if chipcfg_dict.get('adapteq'):
        filter_list.append(gfilt_tool.adapteq_fn)
    if chipcfg_dict.get('histeq'):
        filter_list.append(gfilt_tool.histeq_fn)
    #if chipcfg_dict.get('maxcontrast'):
        #filter_list.append(maxcontr_fn)
    #if chipcfg_dict.get('rank_eq'):
        #filter_list.append(rankeq_fn)
    #if chipcfg_dict.get('local_eq'):
        #filter_list.append(localeq_fn)
    if chipcfg_dict.get('grabcut'):
        filter_list.append(gfilt_tool.grabcut_fn)
    return filter_list
