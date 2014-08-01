# LICENCE
from __future__ import absolute_import, division, print_function
# Science
import numpy as np
import numpy.linalg as npl
# VTool
from . import linalg as ltool
from . import image as gtool
from . import image_filters as gfilt_tool
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[chip]', DEBUG=False)


@profile
def _get_image_to_chip_transform(bbox, chipsz, theta):
    """ transforms image space into chipspace
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box
    """
    (x, y, w, h) = bbox
    (cw_, ch_)     = chipsz
    # Translate from bbox center to (0, 0)
    tx1 = -(x + (w / 2))
    ty1 = -(y + (h / 2))
    T1 = ltool.translation_mat3x3(tx1, ty1)
    # Scale to chip height
    sx = (cw_ / w)
    sy = (ch_ / h)
    S  = ltool.scale_mat3x3(sx, sy)
    # Rotate to chip orientation
    R  = ltool.rotation_mat3x3(-theta)
    # Translate from (0, 0) to chip center
    tx2 = (cw_ / 2)
    ty2 = (ch_ / 2)
    T2 = ltool.translation_mat3x3(tx2, ty2)
    # Merge into single transformation (operate left-to-right aka data on left)
    C = T2.dot(R.dot(S.dot(T1)))
    return C


@profile
def _get_chip_to_image_transform(bbox, chipsz, theta):
    """ transforms chip space into imgspace
        bbox   - bounding box of chip in image space
        chipsz - size of the chip
        theta  - rotation of the bounding box
    """
    C    = _get_image_to_chip_transform(bbox, chipsz, theta)
    invC = npl.inv(C)
    return invC


@profile
def _extract_chip(gfpath, bbox, theta, new_size):
    """ Crops chip from image ; Rotates and scales; """
    imgBGR = gtool.imread(gfpath)  # Read parent image
    M = _get_image_to_chip_transform(bbox, new_size, theta)  # Build transformation
    chipBGR = gtool.warpAffine(imgBGR, M, new_size)  # Rotate and scale
    return chipBGR


@profile
def _filter_chip(chipBGR, filter_funcs):
    """ applies a list of preprocessing filters to a chip """
    chipBGR_ = chipBGR
    for func in filter_funcs:
        chipBGR_ = func(chipBGR)
    return chipBGR_


@profile
def get_scaled_size_with_area(target_area, w, h):
    """ returns new_size which scales (w, h) as close to target_area as possible
    and maintains aspect ratio
    """
    ht = np.sqrt(target_area * h / w)
    wt = w * ht / h
    new_size = (int(round(wt)), int(round(ht)))
    return new_size


@profile
def get_scaled_sizes_with_area(target_area, size_list):
    return [get_scaled_size_with_area(target_area, w, h) for (w, h) in size_list]


@profile
def compute_chip(gfpath, bbox, theta, new_size, filter_list=[]):
    """ Extracts a chip and applies filters """
    chipBGR = _extract_chip(gfpath, bbox, theta, new_size)
    chipBGR = _filter_chip(chipBGR, filter_list)
    return chipBGR


@profile
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
