#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import math
from . import common as com
from .wbia_part import IBEIS_Part
import six


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


class IBEIS_Object(object):
    def __init__(ibso, _xml, width, height, name=None, **kwargs):
        if name is None:
            ibso.name = com.get(_xml, 'name')
            ibso.pose = com.get(_xml, 'pose')
            ibso.truncated = com.get(_xml, 'truncated') == '1'
            ibso.difficult = com.get(_xml, 'difficult') == '1'

            bndbox = com.get(_xml, 'bndbox', text=False)
            ibso.xmax = min(width, int(float(com.get(bndbox, 'xmax'))))
            ibso.xmin = max(0, int(float(com.get(bndbox, 'xmin'))))
            ibso.ymax = min(height, int(float(com.get(bndbox, 'ymax'))))
            ibso.ymin = max(0, int(float(com.get(bndbox, 'ymin'))))

            ibso.parts = [
                IBEIS_Part(part)
                for part in com.get(_xml, 'part', text=False, singularize=False)
            ]
        else:
            ibso.name = name
            ibso.pose = -1
            ibso.truncated = False
            ibso.difficult = False

            ibso.xmax = min(width, int(_xml['xmax']))
            ibso.xmin = max(0, int(_xml['xmin']))
            ibso.ymax = min(height, int(_xml['ymax']))
            ibso.ymin = max(0, int(_xml['ymin']))

            ibso.parts = []

        # Pose
        if isinstance(ibso.pose, six.string_types):
            ibso.pose_str = ibso.pose
        elif ibso.pose < 0 or ibso.pose == []:
            ibso.pose_str = 'Unspecified'
        else:
            bin_size = 2.0 * math.pi / len(BINS)
            temp = float(ibso.pose) + 0.5 * bin_size
            temp %= 2.0 * math.pi
            ibso.pose_str = BINS[int(temp / bin_size)]

        ibso.width = ibso.xmax - ibso.xmin
        ibso.height = ibso.ymax - ibso.ymin
        ibso.xcenter = int(ibso.xmin + (ibso.width / 2))
        ibso.ycenter = int(ibso.ymin + (ibso.height / 2))
        ibso.area = ibso.width * ibso.height

    def __len__(ibso):
        return len(ibso.parts)

    def bounding_box(ibso, parts=False):
        _parts = [part.bounding_box() for part in ibso.parts]
        return [ibso.name, ibso.xmax, ibso.xmin, ibso.ymax, ibso.ymin, _parts]
