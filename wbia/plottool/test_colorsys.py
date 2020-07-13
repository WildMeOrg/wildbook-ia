# -*- coding: utf-8 -*-
import colorsys
import numpy as np
from wbia.plottool import color_funcs


def TEST_COLORSYS():
    rgb = [1.0, 0.2, 0.1]
    hsv = colorsys.rgb_to_hsv(*rgb)
    rgb2 = colorsys.hsv_to_rgb(*hsv)  # NOQA

    new_rgb = color_funcs.brighten_rgb(rgb, 0.0)  # NOQA

    rgba = np.array([0.0, 1.0, 0.0, 1.0])
    rbg = rgba  # NOQA
    new_rgba1 = color_funcs.brighten_rgb(rgba, 0.0)  # NOQA
    new_rgba2 = color_funcs.brighten_rgb(rgba, 0.5)  # NOQA


if __name__ == '__main__':
    TEST_COLORSYS()
