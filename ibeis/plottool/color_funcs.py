from __future__ import absolute_import, division, print_function
import colorsys
import numpy as np  # NOQA
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[colorfuncs]', DEBUG=False)


def assert_base01(channels):
    try:
        assert all([utool.is_float(channel) for channel in channels]), 'channels must be floats'
        assert all([channel <= 1.0 for channel in channels]), 'channels must be in 0-1'
    except AssertionError as ex:
        utool.printex(ex, key_list=['channels'])
        raise


def to_base255(color01):
    assert_base01(color01)
    color255 = map(int, [round(channel * 255.0) for channel in color01])
    return color255


def brighten_rgb(rgb, amount):
    return adjust_sat_and_val_rgb(rgb, amount, amount)


def lighten_rgb(rgb, amount):
    return adjust_sat_and_val_rgb(rgb, -amount, amount)


def adjust_sat_and_val_rgb(rgb, sat_adjust, val_adjust):
    assert_base01(rgb)
    assert_base01([sat_adjust, val_adjust])
    numpy_input = isinstance(rgb, np.ndarray)
    # For some reason numpy input does not work well
    if numpy_input:
        dtype = rgb.dtype
        rgb = rgb.tolist()
    #print('rgb=%r' % (rgb,))
    alpha = None
    if len(rgb) == 4:
        (R, G, B, alpha) = rgb
    else:
        (R, G, B) = rgb
    hsv = colorsys.rgb_to_hsv(R, G, B)
    (H, S, V) = hsv
    S_new = max(min(S + sat_adjust, 1.0), 0.0)
    V_new = max(min(V + val_adjust, 1.0), 0.0)
    #print('hsv=%r' % (hsv,))
    hsv_new = (H, S_new, V_new)
    #print('hsv_new=%r' % (hsv_new,))
    rgb_new = colorsys.hsv_to_rgb(*hsv_new)
    if alpha is not None:
        rgb_new = list(rgb_new) + [alpha]
    #print('rgb_new=%r' % (rgb_new,))
    assert_base01(rgb_new)
    # Return numpy if given as numpy
    if numpy_input:
        rgb_new = np.array(rgb_new, dtype=dtype)
    return rgb_new


def brighten(*args, **kwargs):
    return brighten_rgb(*args, **kwargs)


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
