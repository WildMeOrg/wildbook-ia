from __future__ import absolute_import, division, print_function
from six.moves import range
import colorsys
import numpy as np  # NOQA
import utool
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[colorfuncs]', DEBUG=False)
utool.noinject(__name__, '[colorfuncs]')


def assert_base01(channels):
    try:
        assert all([utool.is_float(channel) for channel in channels]), 'channels must be floats'
        assert all([channel <= 1.0 for channel in channels]), 'channels must be in 0-1'
    except AssertionError as ex:
        utool.printex(ex, key_list=['channels'])
        raise


def to_base01(color255):
    color01 = [channel / 255.0 for channel in color255]
    return color01


def to_base255(color01):
    assert_base01(color01)
    color255 = map(int, [round(channel * 255.0) for channel in color01])
    return color255


def brighten_rgb(rgb, amount):
    hue_adjust = 0.0
    sat_adjust = amount
    val_adjust = amount
    return adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)


def desaturate_rgb(rgb, amount):
    r"""
    CommandLine:
        python -m plottool.color_funcs --test-desaturate_rgb

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> rgb = (255.0 / 255.0, 100 / 255.0, 0 / 255.0)
        >>> amount = 10.0 / 255.0
        >>> # execute function
        >>> result = desaturate_rgb(rgb, amount)
        >>> # verify results
        >>> print(result)
        (1.0, 0.41599384851980004, 0.039215686274509776)
    """
    hue_adjust = 0.0
    sat_adjust = -amount
    val_adjust = 0.0
    new_rgb = adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)
    return new_rgb


def lighten_rgb(rgb, amount):
    r"""
    CommandLine:
        python -m plottool.color_funcs --test-lighten_rgb

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> rgb = np.array((255.0 / 255.0, 100 / 255.0, 0 / 255.0))
        >>> amount = 20.0 / 255.0
        >>> # execute function
        >>> rgb_new = lighten_rgb(rgb, amount)
        >>> # verify results
        >>> result = str(rgb_new)
        >>> print(result)
        [ 1.          0.43983083  0.07843137]
    """
    hue_adjust = 0.0
    sat_adjust = -amount
    val_adjust = amount
    rgb_new = adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)
    return rgb_new


def adjust_hsv_of_rgb255(rgb255, *args, **kwargs):
    rgb = to_base01(rgb255)
    new_rgb = adjust_hsv_of_rgb(rgb, *args, **kwargs)
    new_rgb255 = to_base255(new_rgb)
    return new_rgb255


def adjust_hsv_of_rgb(rgb, hue_adjust=0.0, sat_adjust=0.0, val_adjust=0.0):
    """ works on a single rgb tuple """
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
    H_new = (H + hue_adjust) % 1.0
    S_new = max(min(S + sat_adjust, 1.0), 0.0)
    V_new = max(min(V + val_adjust, 1.0), 0.0)
    #print('hsv=%r' % (hsv,))
    hsv_new = (H_new, S_new, V_new)
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
    HSV_tuples = [(x * 1.0 / N, sat, val) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    utool.deterministic_shuffle(RGB_tuples)
    return RGB_tuples


def add_alpha(colors):
    return [list(color) + [1] for color in colors]


if __name__ == '__main__':
    """
    CommandLine:
        python -m plottool.color_funcs
        python -m plottool.color_funcs --allexamples
        python -m plottool.color_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
