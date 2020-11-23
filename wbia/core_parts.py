# -*- coding: utf-8 -*-

"""
Extracts parts chips from image and applies optional image normalizations.
"""
import logging
import utool as ut
import numpy as np
from wbia import dtool
from wbia.control.controller_inject import register_preprocs, register_subprops
from wbia import core_annots

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


derived_attribute = register_preprocs['part']
register_subprop = register_subprops['part']


PartChipConfig = core_annots.ChipConfig
PartChipImgType = core_annots.ChipImgType


@derived_attribute(
    tablename='pchips',
    parents=['parts'],
    colnames=['img', 'width', 'height', 'M'],
    coltypes=[PartChipImgType, int, int, np.ndarray],
    configclass=PartChipConfig,
    fname='partchipcache4',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_part_chip(depc, part_rowid_list, config=None):
    r"""
    Extracts the part chip from the bounding box

    Args:
        depc (wbia.depends_cache.DependencyCache):
        part_rowid_list (list):  list of part rowids
        config (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        wbia --tf compute_part_chip

    Doctest:
        >>> from wbia.core_parts import *  # NOQA
        >>> import wbia
        >>> import random
        >>> defaultdb = 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> depc = ibs.depc_part
        >>> config = {'dim_size': None}
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:10]
        >>> bbox_list = ibs.get_annot_bboxes(aid_list)
        >>> bbox_list = [
        >>>     (xtl + 100, ytl + 100, w - 100, h - 100)
        >>>     for xtl, ytl, w, h in bbox_list
        >>> ]
        >>> part_rowid_list = ibs.add_parts(aid_list, bbox_list=bbox_list)
        >>> chips = depc.get_property('pchips', part_rowid_list, 'img', config=config)
        >>> for (xtl, ytl, w, h), chip in zip(bbox_list, chips):
        >>>     assert chip.shape == (h, w, 3)
        >>> ibs.delete_parts(part_rowid_list)
    """
    logger.info('Preprocess Part Chips')
    logger.info('config = %r' % (config,))

    ibs = depc.controller

    aid_list = ibs.get_part_aids(part_rowid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    bbox_list = ibs.get_part_bboxes(part_rowid_list)
    theta_list = ibs.get_part_thetas(part_rowid_list)

    result_list = core_annots.gen_chip_configure_and_compute(
        ibs, gid_list, part_rowid_list, bbox_list, theta_list, config
    )
    for result in result_list:
        yield result
    logger.info('Done Preprocessing Part Chips')


class PartAssignmentFeatureConfig(dtool.Config):
    _param_info_list = []


@derived_attribute(
    tablename='part_assignment_features',
    parents=['parts', 'annotations'],
    colnames=[
        'p_xtl', 'p_ytl', 'p_w', 'p_h',
        'a_xtl', 'a_ytl', 'a_w', 'a_h',
        'int_xtl', 'int_ytl', 'int_w', 'int_h',
        'intersect_area_relative_part',
        'intersect_area_relative_annot',
        'part_area_relative_annot'
    ],
    coltypes=[
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='part_assignment_features',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_assignment_features(depc, part_rowid_list, aid_list, config=None):
    assert len(part_rowid_list) == len(aid_list)
    ibs = depc.controller

    part_bboxes  = ibs.get_part_bboxes(part_rowid_list)
    annot_bboxes = ibs.get_annot_bboxes(aid_list)

    part_areas  = [bbox[2] * bbox[3] for bbox in part_bboxes]
    annot_areas = [bbox[2] * bbox[3] for bbox in annot_bboxes]
    p_area_relative_annot = [part_area / annot_area
                        for (part_area, annot_area) in zip(part_areas, annot_areas)]

    intersect_bboxes = _bbox_intersections(part_bboxes, annot_bboxes)
    intersect_areas = [w * h if w > 0 and h > 0 else 0
                       for (_,_,w,h) in intersect_bboxes]

    int_area_relative_part  = [int_area / part_area  for int_area, part_area
                              in zip(intersect_areas, part_areas)]
    int_area_relative_annot = [int_area / annot_area for int_area, annot_area
                               in zip(intersect_areas, annot_areas)]

    result_list = list(zip(part_bboxes, annot_bboxes, intersect_bboxes,
        int_area_relative_part, int_area_relative_annot, p_area_relative_annot))

    for (part_bbox, annot_bbox, intersect_bbox, int_area_relative_part,
        int_area_relative_annot, p_area_relative_annot) in result_list:
        yield (part_bbox[0], part_bbox[1], part_bbox[2], part_bbox[3],
               annot_bbox[0], annot_bbox[1], annot_bbox[2], annot_bbox[3],
               intersect_bbox[0], intersect_bbox[1], intersect_bbox[2], intersect_bbox[3],
               intersect_area_relative_part,
               intersect_area_relative_annot,
               part_area_relative_annot)



def _bbox_intersections(bboxes_a, bboxes_b):
    corner_bboxes_a = _bbox_to_corner_format(bboxes_a)
    corner_bboxes_b = _bbox_to_corner_format(bboxes_b)

    intersect_xtls = [max(xtl_a, xtl_b)
                      for ((xtl_a, _, _, _),(xtl_b, _, _, _))
                      in zip(corner_bboxes_a, corner_bboxes_b)]

    intersect_ytls = [max(ytl_a, ytl_b)
                      for ((_, ytl_a, _, _),(_, ytl_b, _, _))
                      in zip(corner_bboxes_a, corner_bboxes_b)]

    intersect_xbrs = [min(xbr_a, xbr_b)
                      for ((_, _, xbr_a, _),(_, _, xbr_b, _))
                      in zip(corner_bboxes_a, corner_bboxes_b)]

    intersect_ybrs = [min(ybr_a, ybr_b)
                      for ((_, _, _, ybr_a),(_, _, _, ybr_b))
                      in zip(corner_bboxes_a, corner_bboxes_b)]

    intersect_widths = [int_xbr - int_xtl for int_xbr, int_xtl
                        in zip(intersect_xbrs, intersect_xtls)]

    intersect_heights = [int_ybr - int_ytl for int_ybr, int_ytl
                        in zip(intersect_ybrs, intersect_ytls)]

    intersect_bboxes = list(zip(intersect_xtls, intersect_ytls,
                    intersect_widths, intersect_heights))

    return intersect_bboxes



# converts bboxes from (xtl, ytl, w, h) to (xtl, ytl, xbr, ybr)
def _bbox_to_corner_format(bboxes):
    corner_bboxes = [(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
                     for bbox in bboxes]
    return corner_bboxes







