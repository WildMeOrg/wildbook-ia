from __future__ import absolute_import, division, print_function
from six.moves import range, zip, map  # NOQA
from plottool import custom_constants  # NOQA
import colorsys
import numpy as np  # NOQA
import utool as ut
#from plottool import colormaps as cmaps2
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[colorfuncs]', DEBUG=False)
ut.noinject(__name__, '[colorfuncs]')


def assert_base01(channels):
    try:
        assert all([ut.is_float(channel) for channel in channels]), (
            'channels must be floats')
        assert all([channel <= 1.0 for channel in channels]), (
            'channels must be in 0-1')
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


def testshow_colors(rgb_list, gray=ut.get_argflag('--gray')):
    import plottool as pt
    import vtool as vt
    block = np.zeros((5, 5, 3))
    block_list = [block + color[0:3] for color in rgb_list]
    #print(ut.list_str(block_list))
    #print(ut.list_str(rgb_list))
    stacked_block = vt.stack_image_list(block_list, vert=False)
    # convert to bgr
    stacked_block = stacked_block[:, :, ::-1]
    uint8_img = (255 * stacked_block).astype(np.uint8)
    if gray:
        import cv2
        uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
    pt.imshow(uint8_img)
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
        python -m plottool.color_funcs --test-lighten_rgb --show
        python -m plottool.color_funcs --test-lighten_rgb

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
        >>> result = ut.repr2(new_rgb, with_dtype=False)
        >>> print(result)
        np.array([ 1.        ,  0.45294118,  0.1       ])
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


def distinct_colors(N, brightness=.878, randomize=True, hue_range=(0.0, 1.0), cmap_seed=None):
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

    CommandLine:
        python -m plottool.color_funcs --exec-distinct_colors --show
        python -m plottool.color_funcs --exec-distinct_colors --show --no-randomize --N 50
        python -m plottool.color_funcs --exec-distinct_colors --show --cmap_seed=foobar

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
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
        >>> ut.show_if_requested()
    """
    # TODO: Add sin wave modulation to the sat and value
    #import plottool as pt
    if True:
        import plottool as pt
        # HACK for white figures
        remove_yellow = not pt.is_default_dark_bg()
        #if not pt.is_default_dark_bg():
        #    brightness = .8

    use_jet = False
    if use_jet:
        import plottool as pt
        cmap = pt.plt.cm.jet
        RGB_tuples = list(map(tuple, cmap(np.linspace(0, 1, N))))
    elif cmap_seed is not None:
        # Randomized map based on a seed
        #cmap_ = 'Set1'
        #cmap_ = 'Dark2'
        choices = [
            #'Set1', 'Dark2',
            'jet',
            #'gist_rainbow',
            #'rainbow',
            #'gnuplot',
            #'Accent'
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
        #print('cmap_str = %r' % (cmap_str,))
        cmap = pt.plt.cm.get_cmap(cmap_str)
        #ut.hashstr27(cmap_seed)
        #cmap_seed = 0
        #pass
        jitter = (rng.randn(N) / (rng.randn(100).max() / 2)).clip(-1, 1) * ((1 / (N ** 2)))
        range_ = np.linspace(0, 1, N, endpoint=False)
        #print('range_ = %r' % (range_,))
        range_ = range_ + jitter
        #print('range_ = %r' % (range_,))
        while not (np.all(range_ >= 0) and np.all(range_ <= 1)):
            range_[range_ < 0] = np.abs(range_[range_ < 0] )
            range_[range_ > 1] = 2 - range_[range_ > 1]
        #print('range_ = %r' % (range_,))
        shift = rng.rand()
        range_ = (range_ + shift) % 1
        #print('jitter = %r' % (jitter,))
        #print('shift = %r' % (shift,))
        #print('range_ = %r' % (range_,))
        if ncolor_hack is not None:
            range_ = range_[0:N_]
        RGB_tuples = list(map(tuple, cmap(range_)))
    else:
        sat = brightness
        val = brightness
        hmin, hmax = hue_range
        if remove_yellow:
            hue_skips = [(.13, .24)]
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


CMAP_DICT = dict([
    ('Perceptually Uniform Sequential',
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
                        'gist_rainbow', 'hsv', 'flag', 'prism']),
    #('New', ['magma', 'inferno', 'plasma', 'viridis']),
])


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
        python -m plottool.color_funcs --test-show_all_colormaps --show
        python -m plottool.color_funcs --test-show_all_colormaps --show --type=Miscellaneous

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.color_funcs import *  # NOQA
        >>> import plottool as pt
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

    type_ =  ut.get_argval('--type', str, default=None)
    if type_ is None:
        maps = [m for m in pylab.cm.datad if not m.endswith("_r")]
        #maps += cmaps2.__all__
        maps.sort()
    else:
        maps = CMAP_DICT[type_]
        print('CMAP_DICT = %s' % (ut.repr3(CMAP_DICT),))

    l = len(maps) + 1
    for i, m in enumerate(maps):
        if TRANSPOSE:
            pylab.subplot(l, 1, i + 1)
        else:
            pylab.subplot(1, l, i + 1)

        #pylab.axis("off")
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        #try:
        cmap = pylab.get_cmap(m)
        #except Exception:
        #    cmap = getattr(cmaps2, m)

        pylab.imshow(a, aspect='auto', cmap=cmap)  # , origin="lower")
        if TRANSPOSE:
            ax.set_ylabel(m, rotation=0, fontsize=10,
                          horizontalalignment='right', verticalalignment='center')
        else:
            pylab.title(m, rotation=90, fontsize=10)
    #pylab.savefig("colormaps.png", dpi=100, facecolor='gray')


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
