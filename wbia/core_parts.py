# -*- coding: utf-8 -*-

"""
Extracts parts chips from image and applies optional image normalizations.
"""
import logging

import numpy as np
import utool as ut

from wbia import core_annots
from wbia.constants import ANNOTATION_TABLE, PART_TABLE  # NOQA
from wbia.control.controller_inject import register_preprocs, register_subprops

# from wbia.constants import ANNOTATION_TABLE, PART_TABLE

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
    logger.info('config = {!r}'.format(config))

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
