#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from . import common as com


class IBEIS_Part(object):
    def __init__(ibsp, _xml, **kwargs):
        ibsp.name = com.get(_xml, 'name')

        bndbox = com.get(_xml, 'bndbox', text=False)
        ibsp.xmax = int(float(com.get(bndbox, 'xmax')))
        ibsp.xmin = int(float(com.get(bndbox, 'xmin')))
        ibsp.ymax = int(float(com.get(bndbox, 'ymax')))
        ibsp.ymin = int(float(com.get(bndbox, 'ymin')))
        ibsp.width = ibsp.xmax - ibsp.xmin
        ibsp.height = ibsp.ymax - ibsp.ymin
        ibsp.area = ibsp.width * ibsp.height

    def bounding_box(ibsp):
        return [ibsp.name, ibsp.xmax, ibsp.xmin, ibsp.ymax, ibsp.ymin]
