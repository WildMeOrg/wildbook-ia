from __future__ import absolute_import, division, print_function
from os.path import exists, splitext, join, split
import utool
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from plottool.custom_constants import FIGSIZE, DPI, FONTS
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[customfig]')
utool.noinject(__name__, '[customfig]')


def customize_figure(fig, docla):
    #if 'user_stat_list' not in fig.__dict__.keys() or docla:
    #    fig.user_stat_list = []
    #    fig.user_notes = []
    fig.df2_closed = False


def gcf():
    return plt.gcf()


def gca():
    return plt.gca()


def cla():
    return plt.cla()


def clf():
    return plt.clf()


def get_fig(fnum=None):
    #printDBG('[df2] get_fig(fnum=%r)' % fnum)
    fig_kwargs = dict(figsize=FIGSIZE, dpi=DPI)
    if fnum is None:
        try:
            fig = gcf()
        except Exception as ex:
            #printDBG('[df2] get_fig(): ex=%r' % ex)
            utool.printex(ex, '1 in get_fig', iswarning=True)
            fig = plt.figure(**fig_kwargs)
        fnum = fig.number
    else:
        try:
            fig = plt.figure(fnum, **fig_kwargs)
        except Exception as ex:
            #print(repr(ex))
            utool.printex(ex, '2 in get_fig', iswarning=True)
            #warnings.warn(repr(ex))
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


def figure(fnum=None, docla=False, title=None, pnum=(1, 1, 1), figtitle=None,
           doclf=False, **kwargs):
    """
    fnum = fignum = figure number
    pnum = plotnum = plot tuple
    """
    #mpl.pyplot.xkcd()
    fig = get_fig(fnum)
    axes_list = fig.get_axes()
    # Ensure my customized settings
    customize_figure(fig, docla)
    if utool.is_int(pnum):
        pnum = _convert_pnum_int_to_tup(pnum)
    if doclf:  # a bit hacky. Need to rectify docla and doclf
        fig.clf()
        # <HACK TO CLEAR AXES>
        #for ax in axes_list:
        #    ax.clear()
        #for ax in fig.get_axes:
        #    fig.delaxes(ax)
        #axes_list = []
        # </HACK TO CLEAR AXES>
    # Get the subplot
    if docla or len(axes_list) == 0:
        #printDBG('[df2] *** NEW FIGURE %r.%r ***' % (fnum, pnum))
        if pnum is not None:
            assert pnum[0] > 0, 'nRows must be > 0: pnum=%r' % (pnum,)
            assert pnum[1] > 0, 'nCols must be > 0: pnum=%r' % (pnum,)
            #ax = plt.subplot(*pnum)
            ax = fig.add_subplot(*pnum)
            ax.cla()
        else:
            ax = gca()
    else:
        #printDBG('[df2] *** OLD FIGURE %r.%r ***' % (fnum, pnum))
        if pnum is not None:
            ax = plt.subplot(*pnum)  # fig.add_subplot fails here
            #ax = fig.add_subplot(*pnum)
        else:
            ax = gca()
        #ax  = axes_list[0]
    # Set the title
    if title is not None:
        ax = gca()
        set_title(title, ax=ax)
        # Add title to figure
        if figtitle is None and pnum == (1, 1, 1):
            figtitle = title
        if figtitle is not None:
            set_figtitle(figtitle, incanvas=False)
    return fig


def prepare_figure_for_save(fnum, dpi=None, figsize=None):
    if dpi is None:
        dpi = DPI
    if figsize is None:
        figsize = FIGSIZE
    # Resizes the figure for quality saving
    if fnum is None:
        fig = gcf()
    else:
        fig = plt.figure(fnum, figsize=figsize, dpi=dpi)
    # Enforce inches and DPI
    fig.set_size_inches(figsize[0], figsize[1])
    fnum = fig.number
    return fig, fnum


def sanatize_img_fname(fname):
    """ Removes bad characters from images fnames """
    # Replace bad chars
    fname_clean = fname.replace('/', 'slash')
    search_replace_list = [(' ', '_'), ('\n', '--'), ('\\', ''), ('/', '')]
    for old, new in search_replace_list:
        fname_clean = fname_clean.replace(old, new)
    return fname_clean


def sanatize_img_ext(ext, defaultext=None):
    # Find good ext
    if defaultext is None:
        if mpl.get_backend().lower() == 'pdf':
            defaultext = '.pdf'
        else:
            defaultext = '.jpg'
    ext = ext.lower()
    if ext not in utool.IMG_EXTENSIONS and ext != '.pdf':
        ext += defaultext
    return ext


def prepare_figure_fpath(fig, fpath, fnum, usetitle, defaultext):
    if fpath is None:
        # Find the title
        fpath = sanatize_img_fname(fig.canvas.get_window_title())
    elif usetitle:
        title = sanatize_img_fname(fig.canvas.get_window_title())
        fpath = join(fpath, title)
    # Split into dpath, fname, and extension
    dpath, fname_ = split(fpath)
    fname, ext = splitext(fname_)
    # Add the extension back if it wasnt a real extension
    if ext not in utool.IMG_EXTENSIONS and ext != '.pdf':
        fname += ext
        ext = ''
    # Add in DPI information
    size_suffix = 'DPI=%r_WH=%d,%d' % (DPI, FIGSIZE[0], FIGSIZE[1])
    # Sanatize
    fname = sanatize_img_fname(fname)
    ext = sanatize_img_ext(ext, defaultext)
    # Format safely
    fname_fmt = '{fname}_{size_suffix}{ext}'
    fmt_dict = dict(fname=fname, ext=ext, size_suffix=size_suffix)
    fname_clean = utool.long_fname_format(fname_fmt, fmt_dict, ['size_suffix'], max_len=255, hashlen=4)
    # Normalize extension
    fpath_clean = join(dpath, fname_clean)
    return fpath_clean


def save_figure(fnum=None, fpath=None, usetitle=False, overwrite=True,
                defaultext=None, verbose=2, dpi=None, figsize=None, saveax=None):
    """
    Helper to save the figure image to disk. Tries to be smart about filename
    lengths, extensions, overwrites, etc...

    Args:
        fnum (int):  figure number
        fpath (None):
        usetitle (bool): uses title as the fpath
        overwrite (bool): default=True
        defaultext (None):
        verbose (int):  verbosity flag
        dpi (None): dots per inch
        figsize (None):
        saveax (bool or Axes): specifies if the axes should be saved instead of the figure

    References:
        for saving only a specific Axes
        http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
    """
    if dpi is None:
        dpi = DPI

    if defaultext is None:
        if mpl.get_backend().lower() == 'pdf':
            defaultext = '.pdf'
        else:
            defaultext = '.jpg'
    fig, fnum = prepare_figure_for_save(fnum, dpi, figsize)
    fpath_clean = prepare_figure_fpath(fig, fpath, fnum, usetitle, defaultext)

    savekw = {'dpi': dpi}

    if saveax is not None:
        if saveax is True:
            saveax = plt.gca()
        extent = saveax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        savekw['bbox_inches'] = extent.expanded(1.0, 1.0)

    #fname_clean = split(fpath_clean)[1]
    #adjust_subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        if overwrite or not exists(fpath_clean):
            if verbose == 2:
                print('[df2] save_figure() full=%r' % (fpath_clean,))
            elif verbose == 1:
                fpathndir = utool.util_path.path_ndir_split(fpath_clean, 5)
                print('[df2] save_figure() ndir=%r' % (fpathndir))
            fig.savefig(fpath_clean, **savekw)
        else:
            print('[df2] not overwriteing')
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


def set_xlabel(lbl, ax=None):
    if ax is None:
        ax = gca()
    ax.set_xlabel(lbl, fontproperties=FONTS.xlabel)


def set_title(title='', ax=None):
    if ax is None:
        ax = gca()
    ax.set_title(title, fontproperties=FONTS.axtitle)


def set_ylabel(lbl):
    ax = gca()
    ax.set_ylabel(lbl, fontproperties=FONTS.xlabel)


def set_figtitle(figtitle, subtitle='', forcefignum=True, incanvas=True,
                 font='figtitle'):
    if figtitle is None:
        figtitle = ''
    fig = gcf()
    if incanvas:
        if subtitle != '':
            subtitle = '\n' + subtitle
        #fig.suptitle(figtitle + subtitle, fontsize=14, fontweight='bold')
        fontprop = getattr(FONTS, font)
        fig.suptitle(figtitle + subtitle, fontproperties=fontprop)
        #fig_relative_text(.5, .96, subtitle, fontproperties=FONTS.subtitle)
    else:
        fig.suptitle('')
    # Set title in the window
    window_figtitle = ('fig(%d) ' % fig.number) + figtitle
    window_figtitle = window_figtitle.replace('\n', ' ')
    fig.canvas.set_window_title(window_figtitle)
