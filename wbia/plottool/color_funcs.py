# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import range, zip, map  # NOQA
from wbia.plottool import custom_constants  # NOQA
import six
from matplotlib import colors as mcolors
import colorsys
import numpy as np  # NOQA
import utool as ut

# from wbia.plottool import colormaps as cmaps2
# (print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[colorfuncs]', DEBUG=False)
ut.noinject(__name__)
# '[colorfuncs]')


def _test_base01(channels):
    tests01 = {
        'is_float': all([ut.is_float(c) for c in channels]),
        'is_01': all([c >= 0.0 and c <= 1.0 for c in channels]),
    }
    return tests01


def _test_base255(channels):
    tests255 = {
        # 'is_int': all([ut.is_int(c) for c in channels]),
        'is_255': all([c >= 0.0 and c <= 255.0 for c in channels]),
    }
    return tests255


def is_base01(channels):
    """ check if a color is in base 01 """
    if isinstance(channels, six.string_types):
        return False
    return all(_test_base01(channels).values())


def is_base255(channels):
    """ check if a color is in base 01 """
    if isinstance(channels, six.string_types):
        return False
    return all(_test_base255(channels).values())


def assert_base01(channels):
    try:
        tests01 = _test_base01(channels)
        assert tests01['is_float'], 'channels must be floats'
        assert tests01['is_01'], 'channels must be in 0-1'
    except AssertionError as ex:
        ut.printex(ex, key_list=['channels', 'tests01'])
        raise


def assert_base255(channels):
    try:
        tests255 = _test_base255(channels)
        assert tests255['is_255'], 'channels must be in 0-255'
    except AssertionError as ex:
        ut.printex(ex, key_list=['channels', 'tests255'])
        raise


def to_base01(color255):
    """ converts base 255 color to base 01 color """
    color01 = [channel / 255.0 for channel in color255]
    return color01


def to_base255(color01, assume01=False):
    """ converts base 01 color to base 255 color """
    if not assume01:
        assert_base01(color01)
    color255 = list(map(int, [round(channel * 255.0) for channel in color01]))
    return color255


def ensure_base01(color):
    """ always returns a base 01 color

    Note, some colors cannot be determined to be either 255 or 01 if they are
    in float format.

    Args:
        color (?):

    Returns:
        ?: color01

    CommandLine:
        python -m wbia.plottool.color_funcs ensure_base01

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> ensure_base01('g')
        >>> ensure_base01('orangered')
        >>> ensure_base01('#AAAAAA')
        >>> ensure_base01([0, 0, 0])
        >>> ensure_base01([1, 1, 0, 0])
        >>> ensure_base01([1., 1., 0., 0.])
        >>> ensure_base01([.7, .2, 0., 0.])
    """
    if is_base01(color):
        color01 = color
    else:
        if isinstance(color, six.string_types) and color in mcolors.BASE_COLORS:
            # base colors are 01 based
            color01 = mcolors.BASE_COLORS[color]
            color01 = [float(c) for c in color01]
        else:
            color255 = ensure_base255(color)
            color01 = to_base01(color255)
    return color01


def convert_255_to_hex(color255):
    """
    >>> color255 = [255, 51, 0]

    target_rgb01 = pt.FALSE_RED[0:3]
    target_rgb = np.array([[target_rgb01]]).astype(np.float32) / 25
    target_lab = vt.convert_colorspace(target_rgb, 'lab', 'rgb')

    # Find closest CSS color in LAB space
    dist_lab = {}
    dist_rgb = {}
    css_colors = ub.map_vals(convert_hex_to_255, mcolors.CSS4_COLORS)
    for k, c in css_colors.items():
        rgb = np.array([[c]]).astype(np.float32) / 255
        lab = vt.convert_colorspace(rgb, 'lab', 'rgb')
        dist_lab[k] = np.sqrt(((target_lab - lab) ** 2).sum())
        dist_rgb[k] = np.sqrt(((target_rgb - rgb) ** 2).sum())

    best_keys = ub.argsort(dist_lab)
    ub.odict(zip(best_keys, ub.take(dist_lab, best_keys)))
    """
    colorhex = '0x' + ''.join(['%02x' % c for c in color255])
    return colorhex


def convert_hex_to_255(hex_color):
    """
    hex_color = '#6A5AFFAF'
    """
    assert hex_color.startswith('#'), 'not a hex string %r' % (hex_color,)
    parts = hex_color[1:].strip()
    color255 = tuple(int(parts[i : i + 2], 16) for i in range(0, len(parts), 2))
    assert len(color255) in [3, 4], 'must be length 3 or 4'
    # # color = mcolors.hex2color(hex_color[0:7])
    # if len(hex_color) > 8:
    #     alpha_hex = hex_color[7:9]
    #     alpha_float = int(alpha_hex, 16) / 255.0
    #     color = color + (alpha_float,)
    return color255


def ensure_base255(color):
    """
    always returns a base 255 color

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> ensure_base255('g')
        >>> ensure_base255('orangered')
        >>> ensure_base255('#AAAAAA')
        >>> ensure_base255([0, 0, 0])
        >>> ensure_base255([1, 1, 0, 0])
        >>> ensure_base255([.9, 1., 0., 0.])
        >>> ensure_base255([1., 1., 0., 0.])  # FIXME
        >>> ensure_base255([.7, .2, 0., 0.])
    """
    if isinstance(color, six.string_types):
        if color in mcolors.BASE_COLORS:
            # base colors are 01 based
            color01 = mcolors.BASE_COLORS[color]
            color255 = to_base255(color01, assume01=True)
        elif color in mcolors.CSS4_COLORS:
            # cs4 are hex based
            color_hex = mcolors.CSS4_COLORS[color]
            color255 = convert_hex_to_255(color_hex)
        elif color.startswith('#'):
            color255 = convert_hex_to_255(color)
        else:
            raise ValueError('unknown color=%r' % (color,))
    elif is_base01(color):
        color255 = to_base255(color)
    else:
        color255 = color
    assert_base255(color255)
    return color255


def brighten_rgb(rgb, amount):
    hue_adjust = 0.0
    sat_adjust = amount
    val_adjust = amount
    return adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)


def testshow_colors(rgb_list, gray=ut.get_argflag('--gray')):
    """

    colors = ['r', 'b', 'purple', 'orange', 'deeppink', 'g']

    colors = list(mcolors.CSS4_COLORS.keys())

    CommandLine:
        python -m wbia.plottool.color_funcs testshow_colors --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> colors = ut.get_argval('--colors', type_=list, default=['k', 'r'])
        >>> ut.quit_if_noshow()
        >>> rgb_list = ut.emap(ensure_base01, colors)
        >>> testshow_colors(rgb_list)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    import wbia.plottool as pt
    import vtool as vt

    block = np.zeros((5, 5, 3))
    block_list = [block + color[0:3] for color in rgb_list]
    # print(ut.repr2(block_list))
    # print(ut.repr2(rgb_list))
    chunks = ut.ichunks(block_list, 10)
    stacked_chunk = []
    for chunk in chunks:
        stacked_chunk.append(vt.stack_image_list(chunk, vert=False))
    stacked_block = vt.stack_image_list(stacked_chunk, vert=True)
    # convert to bgr
    stacked_block = stacked_block[:, :, ::-1]
    uint8_img = (255 * stacked_block).astype(np.uint8)
    if gray:
        import cv2

        uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
    pt.imshow(uint8_img)
    # pt.show_if_requested()


def desaturate_rgb(rgb, amount):
    r"""
    CommandLine:
        python -m wbia.plottool.color_funcs --test-desaturate_rgb --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> rgb = (255.0 / 255.0, 100 / 255.0, 0 / 255.0)
        >>> amount = .5
        >>> new_rgb = desaturate_rgb(rgb, amount)
        >>> # xdoctest: +REQUIRES(--show)
        >>> color_list = [rgb, new_rgb, desaturate_rgb(rgb, .7)]
        >>> testshow_colors(color_list)
        >>> # verify results
        >>> result = ut.repr2(new_rgb)
        >>> print(result)
        (1.0, 0.696078431372549, 0.5)

        (1.0, 0.41599384851980004, 0.039215686274509776)
    """
    hue_adjust = 0.0
    sat_adjust = -amount
    val_adjust = 0.0
    new_rgb = adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)
    return new_rgb


def darken_rgb(rgb, amount):
    hue_adjust = 0.0
    sat_adjust = 0.0
    val_adjust = -amount
    new_rgb = adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)
    return new_rgb


def lighten_rgb(rgb, amount):
    r"""
    CommandLine:
        python -m wbia.plottool.color_funcs --test-lighten_rgb --show
        python -m wbia.plottool.color_funcs --test-lighten_rgb

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> rgb = np.array((255.0 / 255.0, 100 / 255.0, 0 / 255.0))
        >>> amount = .1
        >>> # execute function
        >>> new_rgb = lighten_rgb(rgb, amount)
        >>> import wbia.plottool as pt
        >>> if pt.show_was_requested():
        >>>     color_list = [rgb, new_rgb, lighten_rgb(rgb, .5)]
        >>>     testshow_colors(color_list)
        >>> # verify results
        >>> result = ut.repr2(new_rgb, with_dtype=False)
        >>> print(result)
    """
    hue_adjust = 0.0
    sat_adjust = -amount
    val_adjust = amount
    new_rgb = adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust)
    return new_rgb


def adjust_hsv_of_rgb255(rgb255, *args, **kwargs):
    """
    CommandLine:
        python -m wbia.plottool.color_funcs --test-adjust_hsv_of_rgb255 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> import wbia.plottool as pt
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
        >>> import wbia.plottool as pt
        >>> if pt.show_was_requested():
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
        python -m wbia.plottool.color_funcs --test-adjust_hsv_of_rgb --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> import wbia.plottool as pt
        >>> # build test data
        >>> rgb_list = [pt.DEEP_PINK[0:3], pt.DARK_YELLOW[0:3], pt.DARK_GREEN[0:3]]
        >>> hue_adjust = -0.1
        >>> sat_adjust = +0.5
        >>> val_adjust = -0.1
        >>> # execute function
        >>> new_rgb_list = [adjust_hsv_of_rgb(rgb, hue_adjust, sat_adjust, val_adjust) for rgb in rgb_list]
        >>> import wbia.plottool as pt
        >>> if pt.show_was_requested():
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
    # assert_base01([sat_adjust, val_adjust])
    numpy_input = isinstance(rgb, np.ndarray)
    # For some reason numpy input does not work well
    if numpy_input:
        dtype = rgb.dtype
        rgb = rgb.tolist()
    # print('rgb=%r' % (rgb,))
    alpha = None
    if len(rgb) == 4:
        (R, G, B, alpha) = rgb
    else:
        (R, G, B) = rgb
    hsv = colorsys.rgb_to_hsv(R, G, B)
    (H, S, V) = hsv
    H_new = H + hue_adjust
    if H_new > 0 or H_new < 1:
        # is there a way to more ellegantly get this?
        H_new %= 1.0
    S_new = max(min(S + sat_adjust, 1.0), 0.0)
    V_new = max(min(V + val_adjust, 1.0), 0.0)
    # print('hsv=%r' % (hsv,))
    hsv_new = (H_new, S_new, V_new)
    # print('hsv_new=%r' % (hsv_new,))
    new_rgb = colorsys.hsv_to_rgb(*hsv_new)
    if alpha is not None:
        new_rgb = list(new_rgb) + [alpha]
    # print('new_rgb=%r' % (new_rgb,))
    assert_base01(new_rgb)
    # Return numpy if given as numpy
    if numpy_input:
        new_rgb = np.array(new_rgb, dtype=dtype)
    return new_rgb


def brighten(*args, **kwargs):
    return brighten_rgb(*args, **kwargs)


def distinct_colors(
    N, brightness=0.878, randomize=True, hue_range=(0.0, 1.0), cmap_seed=None
):
    r"""
    Args:
        N (int):
        brightness (float):

    Returns:
        list: RGB_tuples

    CommandLine:
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 2 --show --hue-range=0.05,.95
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 3 --show --hue-range=0.05,.95
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 4 --show --hue-range=0.05,.95
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 3 --show --no-randomize
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 4 --show --no-randomize
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 6 --show --no-randomize
        python -m wbia.plottool.color_funcs --test-distinct_colors --N 20 --show

    References:
        http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html

    CommandLine:
        python -m wbia.plottool.color_funcs --exec-distinct_colors --show
        python -m wbia.plottool.color_funcs --exec-distinct_colors --show --no-randomize --N 50
        python -m wbia.plottool.color_funcs --exec-distinct_colors --show --cmap_seed=foobar

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> # build test data
        >>> N = ut.get_argval('--N', int, 2)
        >>> randomize = not ut.get_argflag('--no-randomize')
        >>> brightness = 0.878
        >>> # execute function
        >>> cmap_seed = ut.get_argval('--cmap_seed', str, default=None)
        >>> hue_range = ut.get_argval('--hue-range', list, default=(0.00, 1.0))
        >>> RGB_tuples = distinct_colors(N, brightness, randomize, hue_range, cmap_seed=cmap_seed)
        >>> # verify results
        >>> assert len(RGB_tuples) == N
        >>> result = str(RGB_tuples)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> color_list = RGB_tuples
        >>> testshow_colors(color_list)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    # TODO: Add sin wave modulation to the sat and value
    # import wbia.plottool as pt
    if True:
        import wbia.plottool as pt

        # HACK for white figures
        remove_yellow = not pt.is_default_dark_bg()
        # if not pt.is_default_dark_bg():
        #    brightness = .8

    use_jet = False
    if use_jet:
        import wbia.plottool as pt

        cmap = pt.plt.cm.jet
        RGB_tuples = list(map(tuple, cmap(np.linspace(0, 1, N))))
    elif cmap_seed is not None:
        # Randomized map based on a seed
        # cmap_ = 'Set1'
        # cmap_ = 'Dark2'
        choices = [
            # 'Set1', 'Dark2',
            'jet',
            # 'gist_rainbow',
            # 'rainbow',
            # 'gnuplot',
            # 'Accent'
        ]
        cmap_hack = ut.get_argval('--cmap-hack', type_=str, default=None)
        ncolor_hack = ut.get_argval('--ncolor-hack', type_=int, default=None)
        if cmap_hack is not None:
            choices = [cmap_hack]
        if ncolor_hack is not None:
            N = ncolor_hack
            N_ = N
        seed = sum(list(map(ord, ut.hashstr27(cmap_seed))))
        rng = np.random.RandomState(seed + 48930)
        cmap_str = rng.choice(choices, 1)[0]
        # print('cmap_str = %r' % (cmap_str,))
        cmap = pt.plt.cm.get_cmap(cmap_str)
        # ut.hashstr27(cmap_seed)
        # cmap_seed = 0
        # pass
        jitter = (rng.randn(N) / (rng.randn(100).max() / 2)).clip(-1, 1) * (
            (1 / (N ** 2))
        )
        range_ = np.linspace(0, 1, N, endpoint=False)
        # print('range_ = %r' % (range_,))
        range_ = range_ + jitter
        # print('range_ = %r' % (range_,))
        while not (np.all(range_ >= 0) and np.all(range_ <= 1)):
            range_[range_ < 0] = np.abs(range_[range_ < 0])
            range_[range_ > 1] = 2 - range_[range_ > 1]
        # print('range_ = %r' % (range_,))
        shift = rng.rand()
        range_ = (range_ + shift) % 1
        # print('jitter = %r' % (jitter,))
        # print('shift = %r' % (shift,))
        # print('range_ = %r' % (range_,))
        if ncolor_hack is not None:
            range_ = range_[0:N_]
        RGB_tuples = list(map(tuple, cmap(range_)))
    else:
        sat = brightness
        val = brightness
        hmin, hmax = hue_range
        if remove_yellow:
            hue_skips = [(0.13, 0.24)]
        else:
            hue_skips = []
        hue_skip_ranges = [_[1] - _[0] for _ in hue_skips]
        total_skip = sum(hue_skip_ranges)
        hmax_ = hmax - total_skip
        hue_list = np.linspace(hmin, hmax_, N, endpoint=False, dtype=np.float)
        # Remove colors (like hard to see yellows) in specified ranges
        for skip, range_ in zip(hue_skips, hue_skip_ranges):
            hue_list = [hue if hue <= skip[0] else hue + range_ for hue in hue_list]
        HSV_tuples = [(hue, sat, val) for hue in hue_list]
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    if randomize:
        ut.deterministic_shuffle(RGB_tuples)
    return RGB_tuples


def add_alpha(colors):
    return [list(color) + [1] for color in colors]


CMAP_DICT = dict(
    [
        ('Perceptually Uniform Sequential', ['viridis', 'inferno', 'plasma', 'magma']),
        (
            'Sequential',
            [
                'Blues',
                'BuGn',
                'BuPu',
                'GnBu',
                'Greens',
                'Greys',
                'Oranges',
                'OrRd',
                'PuBu',
                'PuBuGn',
                'PuRd',
                'Purples',
                'RdPu',
                'Reds',
                'YlGn',
                'YlGnBu',
                'YlOrBr',
                'YlOrRd',
            ],
        ),
        (
            'Sequential (2)',
            [
                'afmhot',
                'autumn',
                'bone',
                'cool',
                'copper',
                'gist_heat',
                'gray',
                'hot',
                'pink',
                'spring',
                'summer',
                'winter',
            ],
        ),
        (
            'Diverging',
            [
                'BrBG',
                'bwr',
                'coolwarm',
                'PiYG',
                'PRGn',
                'PuOr',
                'RdBu',
                'RdGy',
                'RdYlBu',
                'RdYlGn',
                'Spectral',
                'seismic',
            ],
        ),
        (
            'Qualitative',
            ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
        ),
        (
            'Miscellaneous',
            [
                'gist_earth',
                'terrain',
                'ocean',
                'gist_stern',
                'brg',
                'CMRmap',
                'cubehelix',
                'gnuplot',
                'gnuplot2',
                'gist_ncar',
                'nipy_spectral',
                'jet',
                'rainbow',
                'gist_rainbow',
                'hsv',
                'flag',
                'prism',
            ],
        ),
        # ('New', ['magma', 'inferno', 'plasma', 'viridis']),
    ]
)


def show_all_colormaps():
    """
    Displays at a 90 degree angle. Weird

    FIXME: Remove call to pylab

    References:
        http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        http://matplotlib.org/examples/color/colormaps_reference.html

    Notes:
        cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])
                             ]


    CommandLine:
        python -m wbia.plottool.color_funcs --test-show_all_colormaps --show
        python -m wbia.plottool.color_funcs --test-show_all_colormaps --show --type=Miscellaneous
        python -m wbia.plottool.color_funcs --test-show_all_colormaps --show --cmap=RdYlBu

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.color_funcs import *  # NOQA
        >>> import wbia.plottool as pt
        >>> show_all_colormaps()
        >>> pt.show_if_requested()
    """
    from matplotlib import pyplot as plt
    import pylab
    import numpy as np

    pylab.rc('text', usetex=False)
    TRANSPOSE = True
    a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
    if TRANSPOSE:
        a = a.T
    pylab.figure(figsize=(10, 5))
    if TRANSPOSE:
        pylab.subplots_adjust(right=0.8, left=0.05, bottom=0.01, top=0.99)
    else:
        pylab.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.99)

    type_ = ut.get_argval('--type', str, default=None)
    if type_ is None:
        maps = [m for m in pylab.cm.datad if not m.endswith('_r')]
        # maps += cmaps2.__all__
        maps.sort()
    else:
        maps = CMAP_DICT[type_]
        print('CMAP_DICT = %s' % (ut.repr3(CMAP_DICT),))

    cmap_ = ut.get_argval('--cmap', default=None)
    if cmap_ is not None:
        maps = [getattr(plt.cm, cmap_)]

    length = len(maps) + 1
    for i, m in enumerate(maps):
        if TRANSPOSE:
            pylab.subplot(length, 1, i + 1)
        else:
            pylab.subplot(1, length, i + 1)

        # pylab.axis("off")
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        # try:
        cmap = pylab.get_cmap(m)
        # except Exception:
        #    cmap = getattr(cmaps2, m)

        pylab.imshow(a, aspect='auto', cmap=cmap)  # , origin="lower")
        if TRANSPOSE:
            ax.set_ylabel(
                m,
                rotation=0,
                fontsize=10,
                horizontalalignment='right',
                verticalalignment='center',
            )
        else:
            pylab.title(m, rotation=90, fontsize=10)
    # pylab.savefig("colormaps.png", dpi=100, facecolor='gray')


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.plottool.color_funcs
        python -m wbia.plottool.color_funcs --allexamples
        python -m wbia.plottool.color_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
