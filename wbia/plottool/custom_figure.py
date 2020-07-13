# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import exists, splitext, join, split
import six
import utool as ut
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import functools
from wbia.plottool import custom_constants
import matplotlib.gridspec as gridspec  # NOQA

ut.noinject(__name__, '[customfig]')


# LABEL_SIZE = ut.get_argval('--labelsize', default=None)
# TITLE_SIZE = ut.get_argval('--titlesize', default=None)
# FIGTITLE_SIZE = ut.get_argval('--figtitlesize', default=None) # figure.titlesize
# LEGEND_SIZE = ut.get_argval('--legendsize', default=None)
# TICK_SIZE = ut.get_argval('--ticksize', default=None)

# CUSTOM_RC = {
#     'legend.fontsize': LEGEND_SIZE,
#     'axes.titlesize': TITLE_SIZE,
#     'axes.labelsize': LABEL_SIZE,
#     # 'legend.facecolor': 'w',
#     # 'font.family': 'DejaVu Sans',
#     'xtick.labelsize': TICK_SIZE,
#     'ytick.labelsize': TICK_SIZE,
# }
# for key, val in CUSTOM_RC.items():
#     print(key)
#     print(val)
#     if val is not None:
#         mpl.rcParams[key] = val


def customize_figure(fig, docla):
    # if 'user_stat_list' not in fig.__dict__.keys() or docla:
    #    fig.user_stat_list = []
    #    fig.user_notes = []
    fig.df2_closed = False
    fig.pt_save = functools.partial(save_figure, fig=fig)
    fig.pt_save_and_view = lambda *args, **kwargs: ut.startfile(
        fig.pt_save(*args, **kwargs)
    )


def gcf():
    return plt.gcf()


def gca():
    return plt.gca()


def cla():
    return plt.cla()


def clf():
    return plt.clf()


def get_fig(fnum=None):
    """
    DEPRICATE use ensure_fig
    """
    fig_kwargs = dict(figsize=custom_constants.FIGSIZE, dpi=custom_constants.DPI)
    if fnum is None:
        try:
            fig = gcf()
        except Exception as ex:
            ut.printex(ex, '1 in get_fig', iswarning=True)
            fig = plt.figure(**fig_kwargs)
        fnum = fig.number
    else:
        try:
            fig = plt.figure(fnum, **fig_kwargs)
        except Exception as ex:
            ut.printex(ex, '2 in get_fig', iswarning=True)
            fig = gcf()
    return fig


def ensure_fig(fnum=None):
    if fnum is None:
        try:
            fig = gcf()
        except Exception:
            fig = plt.figure()
    else:
        try:
            fig = plt.figure(fnum)
        except Exception:
            fig = gcf()
    return fig


def get_ax(fnum=None, pnum=None):
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    return ax


def _convert_pnum_int_to_tup(int_pnum):
    # Convert pnum to tuple format if in integer format
    nr = int_pnum // 100
    nc = int_pnum // 10 - (nr * 10)
    px = int_pnum - (nr * 100) - (nc * 10)
    pnum = (nr, nc, px)
    return pnum


def _pnum_to_subspec(pnum):
    if isinstance(pnum, six.string_types):
        pnum = list(pnum)
    nrow, ncols, plotnum = pnum
    # if kwargs.get('use_gridspec', True):
    # Convert old pnums to gridspec
    gs = gridspec.GridSpec(nrow, ncols)
    if isinstance(plotnum, (tuple, slice, list)):
        subspec = gs[plotnum]
    else:
        subspec = gs[plotnum - 1]
    return (subspec,)
    # else:
    #     return (nrow, ncols, plotnum)


def figure(
    fnum=None,
    pnum=(1, 1, 1),
    docla=False,
    title=None,
    figtitle=None,
    doclf=False,
    projection=None,
    **kwargs,
):
    """
    http://matplotlib.org/users/gridspec.html

    Args:
        fnum (int): fignum = figure number
        pnum (int, str, or tuple(int, int, int)): plotnum = plot tuple
        docla (bool): (default = False)
        title (str):  (default = None)
        figtitle (None): (default = None)
        doclf (bool): (default = False)
        projection (None): (default = None)

    Returns:
        ?: fig

    CommandLine:
        python -m wbia.plottool.custom_figure --exec-figure:0 --show
        python -m wbia.plottool.custom_figure --exec-figure:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.custom_figure import *  # NOQA
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.custom_figure import *  # NOQA
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> fig = figure(fnum, (2, 4, (1, slice(1, None))))
        >>> gca().text(0.5, 0.5, "ax3", va="center", ha="center")
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    # mpl.pyplot.xkcd()
    fig = ensure_fig(fnum)
    axes_list = fig.get_axes()
    # Ensure my customized settings
    customize_figure(fig, docla)
    if pnum is None:
        return fig
    if ut.is_int(pnum):
        pnum = _convert_pnum_int_to_tup(pnum)
    if doclf:
        fig.clf()
    if docla or len(axes_list) == 0:
        if pnum is not None:
            assert pnum[0] > 0, 'nRows must be > 0: pnum=%r' % (pnum,)
            assert pnum[1] > 0, 'nCols must be > 0: pnum=%r' % (pnum,)
            subspec = _pnum_to_subspec(pnum)
            ax = fig.add_subplot(*subspec, projection=projection)
            if docla:
                ax.cla()
        else:
            ax = gca()
    else:
        if pnum is not None:
            subspec = _pnum_to_subspec(pnum)
            ax = plt.subplot(*subspec)
        else:
            ax = gca()
    # Set the title / figtitle
    if title is not None:
        ax = gca()
        set_title(title, ax=ax)
    if figtitle is not None:
        set_figtitle(figtitle, incanvas=False, fig=fig)
    return fig


def prepare_figure_for_save(fnum, dpi=None, figsize=None, fig=None):
    """ so bad """
    if fig is not None:
        # HACK; doesnt set DPI this might cause issues
        if dpi is not None:
            fig.set_dpi(dpi)
        if figsize is not None and figsize is not False:
            # Enforce inches and DPI
            figw, figh = figsize[0], figsize[1]
            print('fig w,h (inches) = %r, %r' % (figw, figh))
            fig.set_size_inches(figw, figh)
        return fig, fig.number
    else:
        if dpi is None:
            dpi = custom_constants.DPI
        if figsize is None and figsize is not False:
            figsize = custom_constants.FIGSIZE
        # Resizes the figure for quality saving
        if fnum is None:
            fig = gcf()
        else:
            fig = plt.figure(fnum, figsize=figsize, dpi=dpi)
        # Enforce inches and DPI
        if figsize is not False:
            figw, figh = figsize[0], figsize[1]
            fig.set_size_inches(figw, figh)
        fnum = fig.number
        return fig, fnum


def sanitize_img_fname(fname):
    """ Removes bad characters from images fnames """
    # Replace bad chars
    fname_clean = fname.replace('/', 'slash')
    search_replace_list = [(' ', '_'), ('\n', '--'), ('\\', ''), ('/', '')]
    for old, new in search_replace_list:
        fname_clean = fname_clean.replace(old, new)
    return fname_clean


def sanitize_img_ext(ext, defaultext=None):
    # Find good ext
    if defaultext is None:
        if mpl.get_backend().lower() == 'pdf':
            defaultext = '.pdf'
        else:
            defaultext = '.jpg'
    ext = ext.lower()
    if ext not in ut.IMG_EXTENSIONS and ext != '.pdf':
        ext += defaultext
    return ext


def prepare_figure_fpath(fig, fpath, fnum, usetitle, defaultext, verbose, dpath=None):
    if fpath is None or usetitle:
        if fig._suptitle is not None:
            # safer than using the canvas window title
            # that only works in qt
            title = 'fig(%r) ' % (fig.number,) + fig._suptitle.get_text()
        else:
            title = fig.canvas.get_window_title()
    if fpath is None:
        fpath = sanitize_img_fname(title)
    elif usetitle:
        title = sanitize_img_fname(title)
        fpath = join(fpath, title)
    # Split into dpath, fname, and extension
    dpath_, fname_ = split(fpath)
    if dpath is None:
        dpath = dpath_
    fname, ext = splitext(fname_)
    # Add the extension back if it wasnt a real extension
    if ext not in ut.IMG_EXTENSIONS and ext != '.pdf':
        fname += ext
        ext = ''
    # Add in DPI information
    # size_suffix = 'DPI=%r_WH=%d,%d' % (custom_constants.DPI, custom_constants.FIGSIZE[0], custom_constants.FIGSIZE[1])
    add_render_suffix = False
    if add_render_suffix:
        size_suffix = 'DPI=%r_WH=%d,%d' % (
            fig.dpi,
            int(fig.get_figwidth()),
            int(fig.get_figheight()),
        )
    else:
        size_suffix = ''
    # Sanatize
    fname = sanitize_img_fname(fname)
    ext = sanitize_img_ext(ext, defaultext)
    # Format safely
    fname_fmt = '{fname}_{size_suffix}{ext}'
    fmt_dict = dict(fname=fname, ext=ext, size_suffix=size_suffix)
    if verbose > 1:
        print('[custom_figure] Formating long name')
    fname_clean = ut.long_fname_format(
        fname_fmt, fmt_dict, ['size_suffix', 'fname'], max_len=155, hashlen=8
    )
    # Normalize extension
    fpath_clean = join(dpath, fname_clean)
    return fpath_clean


def get_image_from_figure(fig):
    """
    saves figure data to an ndarray

    References:
        http://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    """
    import numpy as np
    import cv2

    fig.canvas.draw()
    shape = fig.canvas.get_width_height()[::-1] + (3,)
    imgRGB = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    imgRGB = imgRGB.reshape(shape)
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    return imgBGR


def save_figure(
    fnum=None,
    fpath=None,
    fpath_strict=None,
    usetitle=False,
    overwrite=True,
    defaultext=None,
    verbose=1,
    dpi=None,
    figsize=None,
    saveax=None,
    fig=None,
    dpath=None,
):
    """
    Helper to save the figure image to disk. Tries to be smart about filename
    lengths, extensions, overwrites, etc...

    DEPCIATE

    Args:
        fnum (int):  figure number
        fpath (str): file path string
        fpath_strict (str): uses this exact path
        usetitle (bool): uses title as the fpath
        overwrite (bool): default=True
        defaultext (str): default extension
        verbose (int):  verbosity flag
        dpi (int): dots per inch
        figsize (tuple(int, int)): figure size
        saveax (bool or Axes): specifies if the axes should be saved instead of
            the figure

    References:
        for saving only a specific Axes
        http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
        http://robotics.usc.edu/~ampereir/wordpress/?p=626
        http://stackoverflow.com/questions/1271023/resize-a-figure-automatically-in-matplotlib
    """
    if dpi is None:
        dpi = custom_constants.DPI

    if defaultext is None:
        if mpl.get_backend().lower() == 'pdf':
            defaultext = '.pdf'
        else:
            defaultext = '.jpg'
    # print('figsize = %r' % (figsize,))
    fig, fnum = prepare_figure_for_save(fnum, dpi, figsize, fig)
    if fpath_strict is None:
        fpath_clean = prepare_figure_fpath(
            fig, fpath, fnum, usetitle, defaultext, verbose, dpath
        )
    else:
        fpath_clean = fpath_strict
    savekw = {'dpi': dpi}
    if verbose > 1:
        # print('verbose = %r' % (verbose,))
        print('[pt.save_figure] saveax = %r' % (saveax,))

    if False:
        import wbia.plottool as pt

        extent = pt.extract_axes_extents(fig)
        savekw['bbox_inches'] = extent

    if saveax is not None and saveax is not False:
        if verbose > 0:
            print('\n[pt.save_figure] SAVING ONLY EXTENT saveax=%r\n' % (saveax,))
        if saveax is True:
            saveax = plt.gca()
        # ut.embed()
        # saveax.set_aspect('auto')
        import wbia.plottool as pt
        import numpy as np

        xy, w, h = pt.get_axis_xy_width_height(saveax)
        ar = np.abs(w / h)
        if verbose == 2:
            print('[pt.save_figure] saveax xywh = %r' % ((xy, w, h),))
            print('[pt.save_figure] saveax ar = %.2f' % (ar,))
        saveax.set_aspect('equal')
        # extent is bbox in the form [[x0, y0], [x1, y1]]
        extent = saveax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if verbose == 2:
            print(
                '[pt.save_figure] bbox ar = %.2f'
                % np.abs((extent.width / extent.height,))
            )
        # extent = saveax.get_window_extent().transformed(fig.transFigure.inverted())
        # print('[df2] bbox ar = %.2f' % np.abs((extent.width / extent.height,)))
        savekw['bbox_inches'] = extent.expanded(1.0, 1.0)
        if verbose == 2:
            print('[pt.save_figure] savekw = ' + ut.repr2(savekw))
        # ut.embed()

    # fname_clean = split(fpath_clean)[1]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        if overwrite or not exists(fpath_clean):
            if verbose == 2:
                print('[pt.save_figure] save_figure() full=%r' % (fpath_clean,))
            elif verbose == 1:
                fpathndir = ut.path_ndir_split(fpath_clean, 5)
                print('[pt.save_figure] save_figure() ndir=%r' % (fpathndir))
            # fig.savefig(fpath_clean)
            if verbose > 1 or ut.VERBOSE:
                print(']pt.save_figure] fpath_clean = %s' % (fpath_clean,))
                print('[pt.save_figure] savekw = ' + ut.repr2(savekw))
            # savekw['bbox_inches'] = 'tight'
            # print('savekw = %r' % (savekw,))
            if fpath_clean.endswith('.png'):
                savekw['transparent'] = True
                savekw['edgecolor'] = 'none'
                # savekw['axes.edgecolor'] = 'none'
            fig.savefig(fpath_clean, **savekw)
        else:
            if verbose > 0:
                print('[pt.save_figure] not overwriteing')
    return fpath_clean


def set_ticks(xticks, yticks):
    ax = gca()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


def set_xticks(tick_set):
    ax = gca()
    ax.set_xticks(tick_set)


def set_yticks(tick_set):
    ax = gca()
    ax.set_yticks(tick_set)


def customize_fontprop(font_prop, **fontkw):
    font_keys = ['size', 'weight']
    valid_keys = ut.dict_keysubset(fontkw, font_keys)
    if len(valid_keys) > 0:
        font_prop2 = font_prop.copy()
        for key in valid_keys:
            setter = getattr(font_prop, 'set_' + key)
            setter(fontkw[key])
    else:
        font_prop2 = font_prop
    return font_prop2


def set_title(title='', ax=None, **fontkw):
    if ax is None:
        ax = gca()
    if fontkw:
        fontsize = fontkw.get('fontsize', mpl.rcParams['axes.titlesize'])
        titlesize = fontkw.get('titlesize', fontsize)
        titlekw = {
            'fontproperties': mpl.font_manager.FontProperties(
                weight=fontkw.get('weight', 'light'), size=titlesize
            )
        }
    else:
        titlekw = {}
    # font_prop = customize_fontprop(custom_constants.FONTS.axtitle, **fontkw)
    ax.set_title(title, **titlekw)


def set_xlabel(lbl, ax=None, **kwargs):
    r"""
    Args:
        lbl (?):
        ax (None): (default = None)
        **kwargs:

    CommandLine:
        python -m wbia.plottool.custom_figure set_xlabel

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.custom_figure import *  # NOQA
        >>> import wbia.plottool as pt
        >>> fig = pt.figure()
        >>> pt.adjust_subplots(fig=fig, bottom=.5)
        >>> ax = pt.gca()
        >>> lbl = 'a\nab\nabc'
        >>> result = set_xlabel(lbl, ax)
        >>> xaxis = ax.get_xaxis()
        >>> xlabel = xaxis.get_label()
        >>> xlabel.set_horizontalalignment('left')
        >>> xlabel.set_x(0)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    if ax is None:
        ax = gca()
    # labelsize = kwargs.get('labelsize', mpl.rcParams['axes.labelsize'])
    fontkw = {}
    labelkw = {}
    if 'labelsize' in kwargs:
        fontkw['labelsize'] = kwargs.get('labelsize')
    if 'weight' in kwargs:
        fontkw['weight'] = kwargs.get('weight')
    if fontkw:
        labelkw['fontproperties'] = mpl.font_manager.FontProperties(**fontkw)
    # Have to strip for tex output to work with mpl. uggg
    ax.set_xlabel(lbl.strip('\n'), **labelkw)


def set_ylabel(lbl, ax=None, **kwargs):
    if ax is None:
        ax = gca()
    # labelsize = kwargs.get('labelsize', mpl.rcParams['axes.labelsize'])
    # labelkw = {
    #     'fontproperties': mpl.font_manager.FontProperties(
    #         weight=kwargs.get('weight', 'light'), size=labelsize)
    # }
    fontkw = {}
    labelkw = {}
    if 'labelsize' in kwargs:
        fontkw['labelsize'] = kwargs.get('labelsize')
    if 'weight' in kwargs:
        fontkw['weight'] = kwargs.get('weight')
    if fontkw:
        labelkw['fontproperties'] = mpl.font_manager.FontProperties(**fontkw)
    ax.set_ylabel(lbl, **labelkw)


def set_figtitle(
    figtitle,
    subtitle='',
    forcefignum=True,
    incanvas=True,
    size=None,
    fontfamily=None,
    fontweight=None,
    fig=None,
    font=None,
):
    r"""
    Args:
        figtitle (?):
        subtitle (str): (default = '')
        forcefignum (bool): (default = True)
        incanvas (bool): (default = True)
        fontfamily (None): (default = None)
        fontweight (None): (default = None)
        size (None): (default = None)
        fig (None): (default = None)

    CommandLine:
        python -m wbia.plottool.custom_figure set_figtitle --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.custom_figure import *  # NOQA
        >>> import wbia.plottool as pt
        >>> fig = pt.figure(fnum=1, doclf=True)
        >>> result = pt.set_figtitle(figtitle='figtitle', fig=fig)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    # if size is None:
    #     size = FIGTITLE_SIZE
    if font is not None:
        print('WARNING set_figtitle font kwarg is DEPRICATED')
    if figtitle is None:
        figtitle = ''
    if fig is None:
        fig = gcf()
    figtitle = ut.ensure_unicode(figtitle)
    subtitle = ut.ensure_unicode(subtitle)
    if incanvas:
        if subtitle != '':
            subtitle = '\n' + subtitle
        prop = {
            'family': fontfamily,
            'weight': fontweight,
            'size': size,
        }
        prop = {k: v for k, v in prop.items() if v is not None}
        sup = fig.suptitle(figtitle + subtitle)

        if prop:
            fontproperties = sup.get_fontproperties().copy()
            for key, val in prop.items():
                getattr(fontproperties, 'set_' + key)(val)
            sup.set_fontproperties(fontproperties)
            # fontproperties = mpl.font_manager.FontProperties(**prop)
    else:
        fig.suptitle('')
    # Set title in the window
    window_figtitle = ('fig(%d) ' % fig.number) + figtitle
    window_figtitle = window_figtitle.replace('\n', ' ')
    fig.canvas.set_window_title(window_figtitle)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.plottool.custom_figure
        python -m wbia.plottool.custom_figure --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
