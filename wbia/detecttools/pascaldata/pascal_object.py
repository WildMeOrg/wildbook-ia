#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import math
from . import common as com
from .pascal_part import PASCAL_Part


BINS = [
    'left',
    'front_left',
    'front',
    'front_right',
    'right',
    'back_right',
    'back',
    'back_left',
]


class PASCAL_Object(object):
    def __init__(pascalo, _xml, width, height, name=None, **kwargs):
        if name is None:
            pascalo.name = com.get(_xml, 'name')
            pascalo.pose = com.get(_xml, 'pose')
            pascalo.truncated = com.get(_xml, 'truncated') == '1'
            pascalo.difficult = com.get(_xml, 'difficult') == '1'

            bndbox = com.get(_xml, 'bndbox', text=False)
            pascalo.xmax = min(width, int(float(com.get(bndbox, 'xmax'))))
            pascalo.xmin = max(0, int(float(com.get(bndbox, 'xmin'))))
            pascalo.ymax = min(height, int(float(com.get(bndbox, 'ymax'))))
            pascalo.ymin = max(0, int(float(com.get(bndbox, 'ymin'))))

            pascalo.parts = [
                PASCAL_Part(part)
                for part in com.get(_xml, 'part', text=False, singularize=False)
            ]
        else:
            pascalo.name = name
            pascalo.pose = -1
            pascalo.truncated = False
            pascalo.difficult = False

            pascalo.xmax = min(width, int(_xml['xmax']))
            pascalo.xmin = max(0, int(_xml['xmin']))
            pascalo.ymax = min(height, int(_xml['ymax']))
            pascalo.ymin = max(0, int(_xml['ymin']))

            pascalo.parts = []
        # Pose
        if isinstance(pascalo.pose, str):
            pascalo.pose_str = pascalo.pose
        elif pascalo.pose < 0:
            pascalo.pose_str = 'Unspecified'
        else:
            bin_size = 2.0 * math.pi / len(BINS)
            temp = float(pascalo.pose) + 0.5 * bin_size
            temp %= 2.0 * math.pi
            pascalo.pose_str = BINS[int(temp / bin_size)]

        pascalo.width = pascalo.xmax - pascalo.xmin
        pascalo.height = pascalo.ymax - pascalo.ymin
        pascalo.xcenter = int(pascalo.xmin + (pascalo.width / 2))
        pascalo.ycenter = int(pascalo.ymin + (pascalo.height / 2))
        pascalo.area = pascalo.width * pascalo.height

    def __len__(pascalo):
        return len(pascalo.parts)

    def bounding_box(pascalo, parts=False):
        _parts = [part.bounding_box() for part in pascalo.parts]
        retval = [
            pascalo.name,
            pascalo.xmax,
            pascalo.xmin,
            pascalo.ymax,
            pascalo.ymin,
            _parts,
        ]
        return retval
