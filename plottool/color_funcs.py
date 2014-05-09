from __future__ import absolute_import, division, print_function
import colorsys
import numpy as np  # NOQA
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[colorfuncs]', DEBUG=False)


def brighten_rgb(rgb, amount):
    alpha = None
    if len(rgb) == 4:
        (R, G, B, alpha) = rgb
    else:
        (R, G, B) = rgb
    hsv = colorsys.rgb_to_hsv(R, G, B)
    (H, S, V) = hsv
    rgb_new = colorsys.hsv_to_rgb(H, S + amount, V + amount)
    if alpha is not None:
        rgb_new = rgb + [alpha]
    return rgb_new


def brighten(*args, **kwargs):
    brighten_rgb(*args, **kwargs)


def distinct_colors(N, brightness=.878):
    # http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html
    sat = brightness
    val = brightness
    HSV_tuples = [(x * 1.0 / N, sat, val) for x in xrange(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    utool.deterministic_shuffle(RGB_tuples)
    return RGB_tuples


def add_alpha(colors):
    return [list(color) + [1] for color in colors]
