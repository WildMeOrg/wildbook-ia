from __future__ import absolute_import, division, print_function
from six.moves import range, zip, map  # NOQA
from plottool import custom_constants  # NOQA
import colorsys
import numpy as np  # NOQA
import utool as ut
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[colorfuncs]', DEBUG=False)
ut.noinject(__name__, '[colorfuncs]')


def assert_base01(channels):
    try:
        assert all([ut.is_float(channel) for channel in channels]), 'channels must be floats'
        assert all([channel <= 1.0 for channel in channels]), 'channels must be in 0-1'
    except AssertionError as ex:
        ut.printex(ex, key_list=['channels'])
        raise


def to_base01(color255):
    color01 = [channel / 255.0 for channel in color255]
    return color01


def to_base255(color01):
    assert_base01(color01)
    color255 = list(map(int, [round(channel * 255.0) for channel in color01]))
    return color255


def brighten_rgb(rgb, amount):
    hue_adjust = 0.0
    sat_adjust = amount
    val_adjust = amount
    return adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)


def testshow_colors(rgb_list):
    import plottool as pt
    block = np.zeros((5, 5, 3))
    block_list = [block + color[0:3] for color in rgb_list]
    #print(ut.list_str(block_list))
    #print(ut.list_str(rgb_list))
    stacked_block = pt.stack_image_list(block_list, vert=False)
    # convert to bgr
    stacked_block = stacked_block[:, :, ::-1]
    pt.imshow((255 * stacked_block).astype(np.uint8))
    pt.show_if_requested()


def desaturate_rgb(rgb, amount):
    r"""
    CommandLine:
        python -m plottool.color_funcs --test-desaturate_rgb --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> rgb = (255.0 / 255.0, 100 / 255.0, 0 / 255.0)
        >>> amount = .5
        >>> # execute function
        >>> new_rgb = desaturate_rgb(rgb, amount)
        >>> if ut.show_was_requested():
        >>>     color_list = [rgb, new_rgb, desaturate_rgb(rgb, .7)]
        >>>     testshow_colors(color_list)
        >>> # verify results
        >>> new_rgb = str(result)
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
        python -m plottool.color_funcs --test-lighten_rgb --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> rgb = np.array((255.0 / 255.0, 100 / 255.0, 0 / 255.0))
        >>> amount = .1
        >>> # execute function
        >>> new_rgb = lighten_rgb(rgb, amount)
        >>> if ut.show_was_requested():
        >>>     color_list = [rgb, new_rgb, lighten_rgb(rgb, .5)]
        >>>     testshow_colors(color_list)
        >>> # verify results
        >>> result = str(new_rgb)
        >>> print(result)
        [ 1.          0.43983083  0.07843137]
    """
    hue_adjust = 0.0
    sat_adjust = -amount
    val_adjust = amount
    new_rgb = adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)
    return new_rgb


def adjust_hsv_of_rgb255(rgb255, *args, **kwargs):
    """
    CommandLine:
        python -m plottool.color_funcs --test-adjust_hsv_of_rgb255 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> rgb = (220, 220, 255)
        >>> hue_adjust =  0.0
        >>> sat_adjust = -0.05
        >>> val_adjust =  0.0
        >>> # execute function
        >>> new_rgb = adjust_hsv_of_rgb255(rgb, hue_adjust, sat_adjust, val_adjust)
        >>> # verify results
        >>> result = str(new_rgb)
        >>> print(result)
        >>> if ut.show_was_requested():
        >>>     color_list = [to_base01(rgb), to_base01(new_rgb)]
        >>>     testshow_colors(color_list)
    """
    rgb = to_base01(rgb255)
    new_rgb = adjust_hsv_of_rgb(rgb, *args, **kwargs)
    new_rgb255 = to_base255(new_rgb)
    return new_rgb255


def adjust_hsv_of_rgb(rgb, hue_adjust=0.0, sat_adjust=0.0, val_adjust=0.0):
    """ works on a single rgb tuple

    Args:
        rgb (tuple):
        hue_adjust (float):
        sat_adjust (float):
        val_adjust (float):

    Returns:
        ?: new_rgb

    CommandLine:
        python -m plottool.color_funcs --test-adjust_hsv_of_rgb --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> rgb_list = [pt.DEEP_PINK[0:3], pt.DARK_YELLOW[0:3], pt.DARK_GREEN[0:3]]
        >>> hue_adjust = -0.1
        >>> sat_adjust = +0.5
        >>> val_adjust = -0.1
        >>> # execute function
        >>> new_rgb_list = [adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust) for rgb in rgb_list]
        >>> if ut.show_was_requested():
        >>>     color_list = rgb_list + new_rgb_list
        >>>     testshow_colors(color_list)
        >>> # verify results
        >>> result = str(new_rgb)
        >>> print(result)

    Ignore:
        print(np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]))
        print(np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]) % 1.0)
        print(divmod(np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]), 1.0))
        print(1 + np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]) % 1.0)
    """
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
    H_new = (H + hue_adjust)
    if H_new > 0 or H_new < 1:
        # is there a way to more ellegantly get this?
        H_new %= 1.0
    S_new = max(min(S + sat_adjust, 1.0), 0.0)
    V_new = max(min(V + val_adjust, 1.0), 0.0)
    #print('hsv=%r' % (hsv,))
    hsv_new = (H_new, S_new, V_new)
    #print('hsv_new=%r' % (hsv_new,))
    new_rgb = colorsys.hsv_to_rgb(*hsv_new)
    if alpha is not None:
        new_rgb = list(new_rgb) + [alpha]
    #print('new_rgb=%r' % (new_rgb,))
    assert_base01(new_rgb)
    # Return numpy if given as numpy
    if numpy_input:
        new_rgb = np.array(new_rgb, dtype=dtype)
    return new_rgb


def brighten(*args, **kwargs):
    return brighten_rgb(*args, **kwargs)


def distinct_colors(N, brightness=.878, randomize=True, hue_range=(0.0, 1.0)):
    r"""
    Args:
        N (int):
        brightness (float):

    Returns:
        list: RGB_tuples

    CommandLine:
        python -m plottool.color_funcs --test-distinct_colors --N 2 --show --hue-range=0.05,.95
        python -m plottool.color_funcs --test-distinct_colors --N 3 --show --hue-range=0.05,.95
        python -m plottool.color_funcs --test-distinct_colors --N 4 --show --hue-range=0.05,.95
        python -m plottool.color_funcs --test-distinct_colors --N 3 --show --no-randomize
        python -m plottool.color_funcs --test-distinct_colors --N 4 --show --no-randomize
        python -m plottool.color_funcs --test-distinct_colors --N 20 --show

    References:
        http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> N = ut.get_argval('--N', int, 2)
        >>> randomize = not ut.get_argflag('--no-randomize')
        >>> brightness = 0.878
        >>> # execute function
        >>> hue_range = ut.get_argval('--hue-range', list, default=(0.00, 1.0))
        >>> RGB_tuples = distinct_colors(N, brightness, randomize, hue_range)
        >>> if ut.show_was_requested():
        >>>     color_list = RGB_tuples
        >>>     testshow_colors(color_list)
        >>> # verify results
        >>> assert len(RGB_tuples) == N
        >>> result = str(RGB_tuples)
        >>> print(result)
    """
    # TODO: Add sin wave modulation to the sat and value
    sat = brightness
    val = brightness
    hmin, hmax = hue_range
    hue_list = np.linspace(hmin, hmax, N, endpoint=False, dtype=np.float)
    HSV_tuples = [(hue, sat, val) for hue in hue_list]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    if randomize:
        ut.deterministic_shuffle(RGB_tuples)
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
