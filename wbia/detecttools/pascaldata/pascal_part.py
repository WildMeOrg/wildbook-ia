#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from . import common as com


class PASCAL_Part(object):
    def __init__(pascalp, _xml, **kwargs):
        pascalp.name = com.get(_xml, 'name')

        bndbox = com.get(_xml, 'bndbox', text=False)
        pascalp.xmax = int(float(com.get(bndbox, 'xmax')))
        pascalp.xmin = int(float(com.get(bndbox, 'xmin')))
        pascalp.ymax = int(float(com.get(bndbox, 'ymax')))
        pascalp.ymin = int(float(com.get(bndbox, 'ymin')))
        pascalp.width = pascalp.xmax - pascalp.xmin
        pascalp.height = pascalp.ymax - pascalp.ymin
        pascalp.area = pascalp.width * pascalp.height

    def bounding_box(pascalp):
        retval = [pascalp.name, pascalp.xmax, pascalp.xmin, pascalp.ymax, pascalp.ymin]
        return retval
