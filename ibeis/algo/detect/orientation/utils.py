#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import cv2


def resample(image, width=None, height=None):
    if width is None and height is None:
        return None
    if width is not None and height is None:
        height = int((float(width) / len(image[0])) * len(image))
    if height is not None and width is None:
        width = int((float(height) / len(image)) * len(image[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
