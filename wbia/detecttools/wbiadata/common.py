#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import cv2
import os
import random

# import xml.etree.ElementTree as xml

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def _kwargs(kwargs, key, value):
    if key not in kwargs.keys():
        kwargs[key] = value


def get(et, category, text=True, singularize=True):
    temp = [(_object.text if text else _object) for _object in et.findall(category)]
    if len(temp) == 1 and singularize:
        temp = temp[0]
    return temp


def histogram(_list):
    retDict = {}
    for value in _list:
        if value in retDict:
            retDict[value] += 1
        else:
            retDict[value] = 1
    return retDict


def openImage(filename, color=False, alpha=False):
    if not os.path.exists(filename):
        return None

    if not color:
        mode = 0  # Greyscale by default
    elif not alpha:
        mode = 1  # Color without alpha channel
    else:
        mode = -1  # Color with alpha channel

    return cv2.imread(filename, mode)


def randInt(lower, upper):
    return random.randint(lower, upper)


def randColor():
    return [randInt(50, 205), randInt(50, 205), randInt(50, 205)]
