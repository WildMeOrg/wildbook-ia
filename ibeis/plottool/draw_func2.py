# -*- coding: utf-8 -*-
""" Lots of functions for drawing and plotting visiony things """
# TODO: New naming scheme
# viz_<funcname> should clear everything. The current axes and fig: clf, cla.
# # Will add annotations
# interact_<funcname> should clear everything and start user interactions.
# show_<funcname> should always clear the current axes, but not fig: cla #
# Might # add annotates?  plot_<funcname> should not clear the axes or figure.
# More useful for graphs draw_<funcname> same as plot for now. More useful for
# images
from __future__ import absolute_import, division, print_function
from six.moves import range, zip, map
import six
import utool as ut  # NOQA
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError as ex:
    ut.printex(ex,
               'try pip install mpl_toolkits.axes_grid1 or something.  idk yet',
               iswarning=False)
    raise
#import colorsys
import pylab
import warnings
import numpy as np
import cv2
from plottool import mpl_keypoint as mpl_kp
from plottool import color_funcs as color_fns  # NOQA
from plottool import custom_constants  # NOQA
from plottool import custom_figure
from plottool import fig_presenter
#from plottool.custom_figure import *     # NOQA  # TODO: FIXME THIS FILE NEEDS TO BE PARTITIONED
#from plottool.custom_constants import *  # NOQA  # TODO: FIXME THIS FILE NEEDSTO BE PARTITIONED
#from plottool.fig_presenter import *     # NOQA  # TODO: FIXME THIS FILE NEEDS TO BE PARTITIONED
#import operator

DEBUG = False
# Try not injecting into plotting things
ut.noinject(__name__, '[df2]')
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[df2]', DEBUG=DEBUG)


def printDBG(*args):
    pass


# Bring over moved functions that still have dependants elsewhere

TAU = np.pi * 2
distinct_colors = color_fns.distinct_colors
lighten_rgb = color_fns.lighten_rgb
to_base255 = color_fns.to_base255

DARKEN = ut.get_argval(
    '--darken', type_=float, default=(.3 if ut.get_argflag('--darken') else None))


all_figures_bring_to_front = fig_presenter.all_figures_bring_to_front
all_figures_tile           = fig_presenter.all_figures_tile
close_all_figures          = fig_presenter.close_all_figures
close_figure               = fig_presenter.close_figure
iup                        = fig_presenter.iup
iupdate                    = fig_presenter.iupdate
present                    = fig_presenter.present
reset                      = fig_presenter.reset
update                     = fig_presenter.update


ORANGE       = custom_constants.ORANGE
RED          = custom_constants.RED
GREEN        = custom_constants.GREEN
BLUE         = custom_constants.BLUE
YELLOW       = custom_constants.YELLOW
BLACK        = custom_constants.BLACK
WHITE        = custom_constants.WHITE
GRAY         = custom_constants.GRAY
LIGHTGRAY    = custom_constants.LIGHTGRAY
DEEP_PINK    = custom_constants.DEEP_PINK
PINK         = custom_constants.PINK
FALSE_RED    = custom_constants.FALSE_RED
TRUE_GREEN   = custom_constants.TRUE_GREEN
TRUE_BLUE    = custom_constants.TRUE_BLUE
DARK_GREEN   = custom_constants.DARK_GREEN
DARK_BLUE    = custom_constants.DARK_BLUE
DARK_RED     = custom_constants.DARK_RED
DARK_ORANGE  = custom_constants.DARK_ORANGE
DARK_YELLOW  = custom_constants.DARK_YELLOW
PURPLE       = custom_constants.PURPLE
LIGHT_BLUE   = custom_constants.LIGHT_BLUE
UNKNOWN_PURP = custom_constants.UNKNOWN_PURP

TRUE = TRUE_BLUE
FALSE = FALSE_RED

figure       = custom_figure.figure
gca          = custom_figure.gca
gcf          = custom_figure.gcf
get_fig      = custom_figure.get_fig
save_figure  = custom_figure.save_figure
set_figtitle = custom_figure.set_figtitle
set_title    = custom_figure.set_title
set_xlabel   = custom_figure.set_xlabel
set_xticks   = custom_figure.set_xticks
set_ylabel   = custom_figure.set_ylabel
set_yticks   = custom_figure.set_yticks

VERBOSE = ut.get_argflag(('--verbose-df2', '--verb-pt'))

#================
# GLOBALS
#================

TMP_mevent = None
plotWidget = None


def show_was_requested():
    """
    returns True if --show is specified on the commandline or you are in
    IPython (and presumably want some sort of interaction
    """
    return (
        not ut.get_argflag(('--noshow')) and
        (ut.get_argflag(('--show', '--save')) or ut.inIPython()))
    #return ut.show_was_requested()


def show_if_requested(N=1):
    if ut.NOT_QUIET:
        print('[pt] ' + str(ut.get_caller_name(range(3))) + ' show_if_requested()')

    # Process figures adjustments from command line before a show or a save

    adjust_list = ut.get_argval('--adjust', type_=list, default=None)
    if adjust_list is not None:
        # --adjust=[.02,.02,.05]
        keys = ['left', 'bottom', 'wspace', 'right', 'top', 'hspace']
        if len(adjust_list) == 1:
            # [all]
            vals = adjust_list * 3 + [1 - adjust_list[0]] * 2 + adjust_list
        elif len(adjust_list) == 3:
            # [left, bottom, wspace]
            vals = adjust_list + [1 - adjust_list[0], 1 - adjust_list[1],
                                  adjust_list[2]]
        elif len(adjust_list) == 4:
            # [left, bottom, wspace, hspace]
            vals = adjust_list[0:3] + [1 - adjust_list[0], 1 - adjust_list[1],
                                       adjust_list[3]]
        elif len(adjust_list) == 6:
            vals = adjust_list
        else:
            raise NotImplementedError(
                ('vals must be len (1, 3, or 6) not %d, adjust_list=%r. '
                 'Expects keys=%r') % (len(adjust_list), adjust_list, keys))
        adjust_kw = dict(zip(keys, vals))
        print('**adjust_kw = %s' % (ut.dict_str(adjust_kw),))
        adjust_subplots(**adjust_kw)

    figsize = ut.get_argval('--figsize', type_=list, default=None)
    if figsize is not None:
        # Enforce inches and DPI
        fig = gcf()
        figsize = [eval(term) if isinstance(term, str) else term
                   for term in figsize]
        figw, figh = figsize[0], figsize[1]
        #print('get_size_inches = %r' % (fig.get_size_inches(),))
        #print('fig w,h (inches) = %r, %r' % (figw, figh))
        fig.set_size_inches(figw, figh)
        #print('get_size_inches = %r' % (fig.get_size_inches(),))

    dpi = ut.get_argval('--dpi', type_=int, default=custom_constants.DPI)

    fpath_ = ut.get_argval('--save', type_=str, default=None)

    if fpath_ is not None:
        print('Figure save was requested')
        arg_dict = ut.get_arg_dict(prefix_list=['--', '-'],
                                   type_hints={'t': list, 'a': list})
        #import sys
        from os.path import basename, splitext, join
        import plottool as pt
        import vtool as vt

        #print(sys.argv)
        #ut.print_dict(arg_dict)
        # HACK
        arg_dict = {
            key: (val[0] if len(val) == 1 else '[' + ']['.join(val) + ']')
            if isinstance(val, list) else val
            for key, val in arg_dict.items()
        }
        fpath_ = fpath_.format(**arg_dict)
        fpath_ = ut.remove_chars(fpath_, ' \'"')
        #dpath = ut.get_argval('--dpath', type_=str, default=None)
        dpath = ut.get_argval('--dpath', type_=str, default='.')
        fpath = join(dpath, fpath_)

        fig = pt.gcf()

        dpi = ut.get_argval('--dpi', type_=int, default=custom_constants.DPI)

        #ut.embed()

        absfpath_ = pt.save_figure(fig=fig, fpath_strict=ut.truepath(fpath),
                                   figsize=False, dpi=dpi)

        CLIP_WHITE = ut.get_argflag('--clipwhite')
        if CLIP_WHITE:
            # remove white borders
            fpath_in = fpath_out = absfpath_
            vt.clipwhite_ondisk(fpath_in, fpath_out)
            #img = vt.imread(absfpath_)
            #thresh = 128
            #fillval = [255, 255, 255]
            #cropped_img = vt.crop_out_imgfill(img, fillval=fillval, thresh=thresh)
            #print('img.shape = %r' % (img.shape,))
            #print('cropped_img.shape = %r' % (cropped_img.shape,))
            #vt.imwrite(absfpath_, cropped_img)

        default_label = splitext(basename(fpath))[0]  # [0].replace('_', '')
        caption_list = ut.get_argval('--caption', type_=str,
                                     default=basename(fpath).replace('_', ' '))
        if isinstance(caption_list, six.string_types):
            caption_str = caption_list
        else:
            caption_str = ' '.join(caption_list)
        #caption_str = ut.get_argval('--caption', type_=str,
        #default=basename(fpath).replace('_', ' '))
        label_str   = ut.get_argval('--label', type_=str, default=default_label)
        width_str = ut.get_argval('--width', type_=str, default=r'\textwidth')
        width_str = ut.get_argval('--width', type_=str, default=r'\textwidth')
        print('width_str = %r' % (width_str,))
        height_str  = ut.get_argval('--height', type_=str, default=None)
        #if dpath is not None:
        #    fpath_ = ut.unixjoin(dpath, basename(absfpath_))
        #else:
        #    fpath_ = fpath
        fpath_list = [fpath_]

        if len(fpath_list) == 1 and ut.is_developer():
            latex_block = (
                '\ImageCommandII{' + ''.join(fpath_list) + '}{' +
                width_str + '}{\n' + caption_str + '\n}{' + label_str + '}')
            # HACK
        else:
            figure_str  = ut.util_latex.get_latex_figure_str(fpath_list,
                                                             label_str=label_str,
                                                             caption_str=caption_str,
                                                             width_str=width_str,
                                                             height_str=height_str)
            #import sys
            #print(sys.argv)
            latex_block = figure_str
            latex_block = ut.latex_newcommand(label_str, latex_block)
        #latex_block = ut.codeblock(
        #    r'''
        #    \newcommand{\%s}{
        #    %s
        #    }
        #    '''
        #) % (label_str, latex_block,)
        try:
            import os
            import psutil
            import pipes
            #import shlex
            # TODO: separate into get_process_cmdline_str
            # TODO: replace home with ~
            proc = psutil.Process(pid=os.getpid())
            home = os.path.expanduser('~')
            cmdline_str = ' '.join([
                pipes.quote(_).replace(home, '~')
                for _ in proc.cmdline()])
            latex_block = ut.codeblock(
                r'''
                \begin{comment}
                %s
                \end{comment}
                '''
            ) % (cmdline_str,) + '\n' + latex_block
        except OSError:
            pass

        #latex_indent = ' ' * (4 * 2)
        latex_indent = ' ' * (0)

        latex_block_ = (ut.indent(latex_block, latex_indent))
        ut.print_code(latex_block_, 'latex')

        if 'append' in arg_dict:
            append_fpath = arg_dict['append']
            ut.write_to(append_fpath, '\n\n' + latex_block_, mode='a')

        if ut.get_argflag(('--diskshow', '--ds')):
            # show what we wrote
            ut.startfile(absfpath_)

        # Hack write the corresponding logfile next to the output
        log_fpath = ut.get_current_log_fpath()
        if log_fpath is not None:
            ut.copy(log_fpath, splitext(absfpath_)[0] + '.txt')
        else:
            print('Cannot copy log file because none exists')
    if ut.inIPython():
        import plottool as pt
        pt.iup()
    elif ut.get_argflag('--cmd'):
        import plottool as pt
        pt.draw()
        ut.embed(N=N)
    elif ut.get_argflag('--show'):
        if ut.get_argflag('--present'):
            fig_presenter.present()
        for fig in fig_presenter.get_all_figures():
            fig.set_dpi(80)
        plt.show()


def distinct_markers(num, style='astrisk', total=None, offset=0):
    r"""
    Args:
        num (?):

    CommandLine:
        python -m plottool.draw_func2 --exec-distinct_markers --show
        python -m plottool.draw_func2 --exec-distinct_markers --style=star --show
        python -m plottool.draw_func2 --exec-distinct_markers --style=polygon --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import plottool as pt
        >>> style = ut.get_argval('--style', type_=str, default='astrisk')
        >>> marker_list = distinct_markers(10, style)
        >>> x_data = np.arange(0, 3)
        >>> for count, (marker) in enumerate(marker_list):
        >>>     pt.plot(x_data, [count] * len(x_data), marker=marker, markersize=10, linestyle='', label=str(marker))
        >>> pt.legend()
        >>> ut.show_if_requested()
    """
    num_sides = 3
    style_num = {
        'astrisk': 2,
        'star': 1,
        'polygon': 0,
        'circle': 3
    }[style]
    if total is None:
        total = num
    total_degrees = 360 / num_sides
    marker_list = [
        (num_sides, style_num,  total_degrees * (count + offset) / total)
        for count in range(num)
    ]
    return marker_list


def get_all_markers():
    r"""
    CommandLine:
        python -m plottool.draw_func2 --exec-get_all_markers --show

    References:
        http://matplotlib.org/1.3.1/examples/pylab_examples/line_styles.html
        http://matplotlib.org/api/markers_api.html#matplotlib.markers.MarkerStyle.markers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import plottool as pt
        >>> marker_dict = get_all_markers()
        >>> x_data = np.arange(0, 3)
        >>> for count, (marker, name) in enumerate(marker_dict.items()):
        >>>     pt.plot(x_data, [count] * len(x_data), marker=marker, linestyle='', label=name)
        >>> pt.legend()
        >>> ut.show_if_requested()
    """
    marker_dict = {
        0: u'tickleft',
        1: u'tickright',
        2: u'tickup',
        3: u'tickdown',
        4: u'caretleft',
        5: u'caretright',
        6: u'caretup',
        7: u'caretdown',
        #None: u'nothing',
        #u'None': u'nothing',
        #u' ': u'nothing',
        #u'': u'nothing',
        u'*': u'star',
        u'+': u'plus',
        u',': u'pixel',
        u'.': u'point',
        u'1': u'tri_down',
        u'2': u'tri_up',
        u'3': u'tri_left',
        u'4': u'tri_right',
        u'8': u'octagon',
        u'<': u'triangle_left',
        u'>': u'triangle_right',
        u'D': u'diamond',
        u'H': u'hexagon2',
        u'^': u'triangle_up',
        u'_': u'hline',
        u'd': u'thin_diamond',
        u'h': u'hexagon1',
        u'o': u'circle',
        u'p': u'pentagon',
        u's': u'square',
        u'v': u'triangle_down',
        u'x': u'x',
        u'|': u'vline',
    }
    #marker_list = marker_dict.keys()
    #marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*',
    #               'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'TICKLEFT', 'TICKRIGHT', 'TICKUP',
    #               'TICKDOWN', 'CARETLEFT', 'CARETRIGHT', 'CARETUP', 'CARETDOWN']
    return marker_dict


def get_pnum_func(nRows=1, nCols=1, base=0):
    offst = 0 if base == 1 else 1
    def pnum_(px):
        return (nRows, nCols, px + offst)
    return pnum_


def pnum_generator(nRows=1, nCols=1, base=0, nSubplots=None):
    pnum_func = get_pnum_func(nRows, nCols, base)
    total_plots = nRows * nCols
    # TODO: have the last pnums fill in the whole figure
    # when there are less subplots than rows * cols
    #if nSubplots is not None:
    #    if nSubplots < total_plots:
    #        pass
    for px in range(total_plots):
        yield pnum_func(px)


def make_pnum_nextgen(nRows=1, nCols=1, base=0, nSubplots=None):
    import functools
    pnum_gen = pnum_generator(nRows=nRows, nCols=nCols, base=base, nSubplots=nSubplots)
    pnum_next = functools.partial(six.next, pnum_gen)
    return pnum_next


def fnum_generator(base=1):
    fnum = base - 1
    while True:
        fnum += 1
        yield fnum


def make_fnum_nextgen(base=1):
    import functools
    fnum_gen = fnum_generator(base=base)
    fnum_next = functools.partial(six.next, fnum_gen)
    return fnum_next


BASE_FNUM = 9001


def next_fnum(new_base=None):
    global BASE_FNUM
    if new_base is not None:
        BASE_FNUM = new_base
    BASE_FNUM += 1
    return BASE_FNUM


def ensure_fnum(fnum):
    if fnum is None:
        return next_fnum()
    return fnum


def execstr_global():
    execstr = ['global' + key for key in globals().keys()]
    return execstr


def label_to_colors(labels_):
    """
    returns a unique and distinct color corresponding to each label
    """
    unique_labels = list(set(labels_))
    unique_colors = distinct_colors(len(unique_labels))
    label2_color = dict(zip(unique_labels, unique_colors))
    color_list = [label2_color[label] for label in labels_]
    return color_list


#def distinct_colors(N, brightness=.878, shuffle=True):
#    """
#    Args:
#        N (int): number of distinct colors
#        brightness (float): brightness of colors (maximum distinctiveness is .5) default is .878
#    Returns:
#        RGB_tuples
#    Example:
#        >>> from plottool.draw_func2 import *  # NOQA
#    """
#    # http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html
#    sat = brightness
#    val = brightness
#    HSV_tuples = [(x * 1.0 / N, sat, val) for x in range(N)]
#    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
#    if shuffle:
#        ut.deterministic_shuffle(RGB_tuples)
#    return RGB_tuples


def add_alpha(colors):
    return [list(color) + [1] for color in colors]


def get_axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    """ gets geometry of a subplot """
    if ax is None:
        ax = gca()
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


def get_axis_bbox(ax=None, **kwargs):
    """
    # returns in figure coordinates?
    """

    xy, width, height = get_axis_xy_width_height(ax=ax, **kwargs)
    return (xy[0], xy[1], width, height)


def draw_border(ax, color=GREEN, lw=2, offset=None, adjust=True):
    'draws rectangle border around a subplot'
    if adjust:
        xy, width, height = get_axis_xy_width_height(ax, -.7, -.2, 1, .4)
    else:
        xy, width, height = get_axis_xy_width_height(ax)
    if offset is not None:
        xoff, yoff = offset
        xy = [xoff, yoff]
        height = - height - yoff
        width = width - xoff
    rect = mpl.patches.Rectangle(xy, width, height, lw=lw)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(False)
    rect.set_edgecolor(color)


TAU = np.pi * 2


def rotate_plot(theta=TAU / 8, ax=None):
    r"""
    Args:
        theta (?):
        ax (None):

    CommandLine:
        python -m plottool.draw_func2 --test-rotate_plot

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> # build test data
        >>> ax = gca()
        >>> theta = TAU / 8
        >>> plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 2, 2])
        >>> # execute function
        >>> result = rotate_plot(theta, ax)
        >>> # verify results
        >>> print(result)
        >>> show_if_requested()
    """
    import vtool as vt
    if ax is None:
        ax = gca()
    #import vtool as vt
    xy, width, height = get_axis_xy_width_height(ax)
    bbox = [xy[0], xy[1], width, height]
    M = mpl.transforms.Affine2D(vt.rotation_around_bbox_mat3x3(theta, bbox))
    propname = 'transAxes'
    #propname = 'transData'
    T = getattr(ax, propname)
    T.transform_affine(M)
    #T = ax.get_transform()
    #Tnew = T + M
    #ax.set_transform(Tnew)
    #setattr(ax, propname, Tnew)
    iup()


# TODO SEPARTE THIS INTO DRAW BBOX AND DRAW_ANNOTATION
def draw_bbox(bbox, lbl=None, bbox_color=(1, 0, 0), lbl_bgcolor=(0, 0, 0),
              lbl_txtcolor=(1, 1, 1), draw_arrow=True, theta=0, ax=None, lw=2):
    if ax is None:
        ax = gca()
    (rx, ry, rw, rh) = bbox
    # Transformations are specified in backwards order.
    trans_annotation = mpl.transforms.Affine2D()
    trans_annotation.scale(rw, rh)
    trans_annotation.rotate(theta)
    trans_annotation.translate(rx + rw / 2, ry + rh / 2)
    t_end = trans_annotation + ax.transData
    bbox = mpl.patches.Rectangle((-.5, -.5), 1, 1, lw=lw, transform=t_end)
    bbox.set_fill(False)
    #bbox.set_transform(trans)
    bbox.set_edgecolor(bbox_color)
    ax.add_patch(bbox)
    # Draw overhead arrow indicating the top of the ANNOTATION
    if draw_arrow:
        arw_xydxdy = (-0.5, -0.5, 1.0, 0.0)
        arw_kw = dict(head_width=.1, transform=t_end, length_includes_head=True)
        arrow = mpl.patches.FancyArrow(*arw_xydxdy, **arw_kw)
        arrow.set_edgecolor(bbox_color)
        arrow.set_facecolor(bbox_color)
        ax.add_patch(arrow)
    # Draw a label
    if lbl is not None:
        ax_absolute_text(rx, ry, lbl, ax=ax,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=lbl_txtcolor,
                         backgroundcolor=lbl_bgcolor)


def plot(*args, **kwargs):
    yscale = kwargs.pop('yscale', 'linear')
    xscale = kwargs.pop('xscale', 'linear')
    logscale_kwargs = kwargs.pop('logscale_kwargs', {'nonposx': 'clip'})
    plot = plt.plot(*args, **kwargs)
    ax = plt.gca()

    yscale_kwargs = logscale_kwargs if yscale in ['log', 'symlog'] else {}
    xscale_kwargs = logscale_kwargs if xscale in ['log', 'symlog'] else {}

    ax.set_yscale(yscale, **yscale_kwargs)
    ax.set_xscale(xscale, **xscale_kwargs)
    return plot


def plot2(x_data, y_data, marker='o', title_pref='', x_label='x', y_label='y',
          unitbox=False, flipx=False, flipy=False, title=None, dark=None,
          equal_aspect=True, pad=0, label='', fnum=None, pnum=None, *args,
          **kwargs):
    """
    don't forget to call pt.legend

    Kwargs:
        linewidth (float):
    """
    if x_data is None:
        warnstr = '[df2] ! Warning:  x_data is None'
        print(warnstr)
        x_data = np.arange(len(y_data))
    if fnum is not None or pnum is not None:
        figure(fnum=fnum, pnum=pnum)
    do_plot = True
    # ensure length
    if len(x_data) != len(y_data):
        warnstr = '[df2] ! Warning:  len(x_data) != len(y_data). Cannot plot2'
        warnings.warn(warnstr)
        draw_text(warnstr)
        do_plot = False
    if len(x_data) == 0:
        warnstr = '[df2] ! Warning:  len(x_data) == 0. Cannot plot2'
        warnings.warn(warnstr)
        draw_text(warnstr)
        do_plot = False
    # ensure in ndarray
    if isinstance(x_data, list):
        x_data = np.array(x_data)
    if isinstance(y_data, list):
        y_data = np.array(y_data)
    ax = gca()
    if do_plot:
        ax.plot(x_data, y_data, marker, label=label, *args, **kwargs)

        min_x = x_data.min()
        min_y = y_data.min()
        max_x = x_data.max()
        max_y = y_data.max()
        min_ = min(min_x, min_y)
        max_ = max(max_x, max_y)

        if equal_aspect:
            # Equal aspect ratio
            if unitbox is True:
                # Just plot a little bit outside  the box
                ax.set_xlim(-.01, 1.01)
                ax.set_ylim(-.01, 1.01)
                ax.grid(True)
            else:
                ax.set_xlim(min_, max_)
                ax.set_ylim(min_, max_)
                #aspect_opptions = ['auto', 'equal', num]
                ax.set_aspect('equal')
        else:
            ax.set_aspect('auto')
        if pad > 0:
            ax.set_xlim(min_x - pad, max_x + pad)
            ax.set_ylim(min_y - pad, max_y + pad)
        ax.grid(True, color='w' if dark else 'k')
        if flipx:
            ax.invert_xaxis()
        if flipy:
            ax.invert_yaxis()

        use_darkbackground = dark
        if use_darkbackground is None:
            import plottool as pt
            use_darkbackground = pt.is_default_dark_bg()
        if use_darkbackground:
            dark_background(ax)
    else:
        # No data, draw big red x
        draw_boxedX()

    presetup_axes(x_label, y_label, title_pref, title, ax=None)


def pad_axes(pad, xlim=None, ylim=None):
    ax = gca()
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    min_x, max_x = xlim
    min_y, max_y = ylim
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)


def presetup_axes(x_label='x', y_label='y', title_pref='', title=None,
                  equal_aspect=False, ax=None, **kwargs):
    if ax is None:
        ax = gca()
    set_xlabel(x_label, **kwargs)
    set_ylabel(y_label, **kwargs)
    if title is None:
        title = x_label + ' vs ' + y_label
    set_title(title_pref + ' ' + title, ax=None, **kwargs)
    if equal_aspect:
        ax.set_aspect('equal')


def postsetup_axes(use_legend=True, bg='dark'):
    if bg == 'dark':
        dark_background()
    if use_legend:
        legend()


def adjust_subplots_xlabels():
    adjust_subplots(left=.03, right=.97, bottom=.2, top=.9, hspace=.15)


def adjust_subplots_xylabels():
    adjust_subplots(left=.03, right=1, bottom=.1, top=.9, hspace=.15)


SAFE_POS = {
    'left': .1,
    'right': .9,
    'top': .9,
    'bottom': .1,
    'wspace': .3,
    'hspace': .5,
}


def adjust_subplots_safe(**kwargs):
    for key in six.iterkeys(SAFE_POS):
        if key not in kwargs:
            kwargs[key] = SAFE_POS[key]
    adjust_subplots(**kwargs)


def adjust_subplots(left=0.02,  bottom=0.02,
                    right=None,   top=None,
                    wspace=0.1,   hspace=None):
    """
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2
    """
    if right is None:
        right = 1 - left
    if top is None:
        top = 1 - bottom
    if hspace is None:
        hspace = wspace
    print('[df2] adjust_subplots(**%s)' % ut.dict_str(locals()))
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)


def adjust_subplots2(**kwargs):
    subplotpars = gcf().subplotpars
    adjust_dict = {}
    valid_kw = ['left', 'right', 'top', 'bottom', 'wspace', 'hspace']
    for key in valid_kw:
        adjust_dict[key] = kwargs.get(key, subplotpars.__dict__[key])
    if kwargs.get('use_argv', False):
        adjust_dict = ut.parse_dict_from_argv(adjust_dict)
    plt.subplots_adjust(**adjust_dict)


#=======================
# TEXT FUNCTIONS
# TODO: I have too many of these. Need to consolidate
#=======================


def upperleft_text(txt, alpha=.6, color=None):
    txtargs = dict(horizontalalignment='left',
                   verticalalignment='top',
                   backgroundcolor=(0, 0, 0, alpha),
                   color=ORANGE if color is None else color)
    ax_relative_text(.02, .02, txt, **txtargs)


def upperright_text(txt, offset=None, alpha=.6):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   backgroundcolor=(0, 0, 0, alpha),
                   color=ORANGE,
                   offset=offset)
    ax_relative_text(.98, .02, txt, **txtargs)


def lowerright_text(txt):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='bottom',
                   backgroundcolor=(0, 0, 0, .6),
                   color=ORANGE)
    ax_relative_text(.98, .92, txt, **txtargs)


def absolute_lbl(x_, y_, txt, roffset=(-.02, -.02), alpha=.6, **kwargs):
    """ alternative to relative text """
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   backgroundcolor=(0, 0, 0, alpha),
                   color=ORANGE)
    txtargs.update(kwargs)
    ax_absolute_text(x_, y_, txt, roffset=roffset, **txtargs)


def ax_relative_text(x, y, txt, ax=None, offset=None, **kwargs):
    if ax is None:
        ax = gca()
    xy, width, height = get_axis_xy_width_height(ax)
    x_, y_ = ((xy[0]) + x * width, (xy[1] + height) - y * height)
    if offset is not None:
        xoff, yoff = offset
        x_ += xoff
        y_ += yoff
    ax_absolute_text(x_, y_, txt, ax=ax, **kwargs)


def ax_absolute_text(x_, y_, txt, ax=None, roffset=None, **kwargs):
    """ Base function for text """
    if ax is None:
        ax = gca()
    if 'fontproperties' in kwargs:
        kwargs['fontproperties'] = custom_constants.FONTS.relative
    if roffset is not None:
        xroff, yroff = roffset
        xy, width, height = get_axis_xy_width_height(ax)
        x_ += xroff * width
        y_ += yroff * height

    ax.text(x_, y_, txt, **kwargs)


def fig_relative_text(x, y, txt, **kwargs):
    kwargs['horizontalalignment'] = 'center'
    kwargs['verticalalignment'] = 'center'
    fig = gcf()
    #xy, width, height = get_axis_xy_width_height(ax)
    #x_, y_ = ((xy[0]+width)+x*width, (xy[1]+height)-y*height)
    fig.text(x, y, txt, **kwargs)


def draw_text(text_str, rgb_textFG=(0, 0, 0), rgb_textBG=(1, 1, 1)):
    ax = gca()
    xy, width, height = get_axis_xy_width_height(ax)
    text_x = xy[0] + (width / 2)
    text_y = xy[1] + (height / 2)
    ax.text(text_x, text_y, text_str,
            horizontalalignment='center',
            verticalalignment='center',
            color=rgb_textFG,
            backgroundcolor=rgb_textBG)


#def convert_keypress_event_mpl_to_qt4(mevent):
#    global TMP_mevent
#    TMP_mevent = mevent
#    # Grab the key from the mpl.KeyPressEvent
#    key = mevent.key
#    print('[df2] convert event mpl -> qt4')
#    print('[df2] key=%r' % key)
#    # dicts modified from backend_qt4.py
#    mpl2qtkey = {'control': Qt.Key_Control, 'shift': Qt.Key_Shift,
#                 'alt': Qt.Key_Alt, 'super': Qt.Key_Meta,
#                 'enter': Qt.Key_Return, 'left': Qt.Key_Left, 'up': Qt.Key_Up,
#                 'right': Qt.Key_Right, 'down': Qt.Key_Down,
#                 'escape': Qt.Key_Escape, 'f1': Qt.Key_F1, 'f2': Qt.Key_F2,
#                 'f3': Qt.Key_F3, 'f4': Qt.Key_F4, 'f5': Qt.Key_F5,
#                 'f6': Qt.Key_F6, 'f7': Qt.Key_F7, 'f8': Qt.Key_F8,
#                 'f9': Qt.Key_F9, 'f10': Qt.Key_F10, 'f11': Qt.Key_F11,
#                 'f12': Qt.Key_F12, 'home': Qt.Key_Home, 'end': Qt.Key_End,
#                 'pageup': Qt.Key_PageUp, 'pagedown': Qt.Key_PageDown}
#    # Reverse the control and super (aka cmd/apple) keys on OSX
#    if sys.platform == 'darwin':
#        mpl2qtkey.update({'super': Qt.Key_Control, 'control': Qt.Key_Meta, })

#    # Try to reconstruct QtGui.KeyEvent
#    type_ = QtCore.QEvent.Type(QtCore.QEvent.KeyPress)  # The type should always be KeyPress
#    text = ''
#    # Try to extract the original modifiers
#    modifiers = QtCore.Qt.NoModifier  # initialize to no modifiers
#    if key.find(u'ctrl+') >= 0:
#        modifiers = modifiers | QtCore.Qt.ControlModifier
#        key = key.replace(u'ctrl+', u'')
#        print('[df2] has ctrl modifier')
#        text += 'Ctrl+'
#    if key.find(u'alt+') >= 0:
#        modifiers = modifiers | QtCore.Qt.AltModifier
#        key = key.replace(u'alt+', u'')
#        print('[df2] has alt modifier')
#        text += 'Alt+'
#    if key.find(u'super+') >= 0:
#        modifiers = modifiers | QtCore.Qt.MetaModifier
#        key = key.replace(u'super+', u'')
#        print('[df2] has super modifier')
#        text += 'Super+'
#    if key.isupper():
#        modifiers = modifiers | QtCore.Qt.ShiftModifier
#        print('[df2] has shift modifier')
#        text += 'Shift+'
#    # Try to extract the original key
#    try:
#        if key in mpl2qtkey:
#            key_ = mpl2qtkey[key]
#        else:
#            key_ = ord(key.upper())  # Qt works with uppercase keys
#            text += key.upper()
#    except Exception as ex:
#        print('[df2] ERROR key=%r' % key)
#        print('[df2] ERROR %r' % ex)
#        raise
#    autorep = False  # default false
#    count   = 1  # default 1
#    text = str(text)  # The text is somewhat arbitrary
#    # Create the QEvent
#    print('----------------')
#    print('[df2] Create event')
#    print('[df2] type_ = %r' % type_)
#    print('[df2] text = %r' % text)
#    print('[df2] modifiers = %r' % modifiers)
#    print('[df2] autorep = %r' % autorep)
#    print('[df2] count = %r ' % count)
#    print('----------------')
#    qevent = QtGui.QKeyEvent(type_, key_, modifiers, text, autorep, count)
#    return qevent


#def test_build_qkeyevent():
#    import draw_func2 as df2
#    qtwin = df2.QT4_WINS[0]
#    # This reconstructs an test mplevent
#    canvas = df2.figure(1).canvas
#    mevent = mpl.backend_bases.KeyEvent('key_press_event', canvas, u'ctrl+p', x=672, y=230.0)
#    qevent = df2.convert_keypress_event_mpl_to_qt4(mevent)
#    app = qtwin.backend.app
#    app.sendEvent(qtwin.ui, mevent)
#    #type_ = QtCore.QEvent.Type(QtCore.QEvent.KeyPress)  # The type should always be KeyPress
#    #text = str('A')  # The text is somewhat arbitrary
#    #modifiers = QtCore.Qt.NoModifier  # initialize to no modifiers
#    #modifiers = modifiers | QtCore.Qt.ControlModifier
#    #modifiers = modifiers | QtCore.Qt.AltModifier
#    #key_ = ord('A')  # Qt works with uppercase keys
#    #autorep = False  # default false
#    #count   = 1  # default 1
#    #qevent = QtGui.QKeyEvent(type_, key_, modifiers, text, autorep, count)
#    return qevent


def show_histogram(data, bins=None, **kwargs):
    """
    CommandLine:
        python -m plottool.draw_func2 --test-show_histogram --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> # build test data
        >>> data = np.array([1, 24, 0, 0, 3, 4, 5, 9, 3, 0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 0, 0, 0, 3,])
        >>> bins = None
        >>> # execute function
        >>> result = show_histogram(data, bins)
        >>> # verify results
        >>> print(result)
        >>> ut.show_if_requested()
    """
    print('[df2] show_histogram()')
    dmin = int(np.floor(data.min()))
    dmax = int(np.ceil(data.max()))
    if bins is None:
        bins = dmax - dmin
    fig = figure(**kwargs)
    ax  = gca()
    ax.hist(data, bins=bins, range=(dmin, dmax))
    #dark_background()

    use_darkbackground = None
    if use_darkbackground is None:
        use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background(ax)
    return fig
    #help(np.bincount)
    #fig.show()


def show_signature(sig, **kwargs):
    fig = figure(**kwargs)
    plt.plot(sig)
    fig.show()


def draw_stems(x_data=None, y_data=None, setlims=True, color=None,
               markersize=None, bottom=None, marker=None, linestyle='-'):
    """
    Draws stem plot

    Args:
        x_data (None):
        y_data (None):
        setlims (bool):
        color (None):
        markersize (None):
        bottom (None):

    References:
        http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-multiple-line.html

    CommandLine:
        python -m plottool.draw_func2 --test-draw_stems --show
        python -m plottool.draw_func2 --test-draw_stems

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> x_data = np.append(np.arange(1, 10), np.arange(1, 10))
        >>> rng = np.random.RandomState(0)
        >>> y_data = sorted(rng.rand(len(x_data)) * 10)
        >>> # y_data = np.array([ut.get_nth_prime(n) for n in x_data])
        >>> setlims = False
        >>> color = [1.0, 0.0, 0.0, 1.0]
        >>> markersize = 2
        >>> marker = 'o'
        >>> bottom = None
        >>> result = draw_stems(x_data, y_data, setlims, color, markersize, bottom, marker)
        >>> ut.show_if_requested()
    """
    if y_data is not None and x_data is None:
        x_data = np.arange(len(y_data))
        pass
    if len(x_data) != len(y_data):
        print('[df2] WARNING plot_stems(): len(x_data)!=len(y_data)')
    if len(x_data) == 0:
        print('[df2] WARNING plot_stems(): len(x_data)=len(y_data)=0')
    x_data_ = np.array(x_data)
    y_data_ = np.array(y_data)
    y_data_sortx = y_data_.argsort()[::-1]
    x_data_sort = x_data_[y_data_sortx]
    y_data_sort = y_data_[y_data_sortx]
    if color is None:
        color =  [1.0, 0.0, 0.0, 1.0]

    OLD = False
    if not OLD:
        if bottom is None:
            bottom = 0
        # Faster way of drawing stems
        #with ut.Timer('new stem'):
        stemlines = []
        ax = gca()
        x_segments = ut.flatten([[thisx, thisx, None] for thisx in x_data_sort])
        if linestyle == '':
            y_segments = ut.flatten([[thisy, thisy, None] for thisy in y_data_sort])
        else:
            y_segments = ut.flatten([[bottom, thisy, None] for thisy in y_data_sort])
        ax.plot(x_segments, y_segments, linestyle, color=color, marker=marker)
    else:
        with ut.Timer('old stem'):
            markerline, stemlines, baseline = pylab.stem(
                x_data_sort, y_data_sort, linefmt='-', bottom=bottom)
            if markersize is not None:
                markerline.set_markersize(markersize)

            pylab.setp(markerline, 'markerfacecolor', 'w')
            pylab.setp(stemlines, 'markerfacecolor', 'w')
            if color is not None:
                for l in stemlines:
                    l.set_color(color)
            pylab.setp(baseline, 'linewidth', 0)  # baseline should be invisible
    if setlims:
        ax = gca()
        ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
        ax.set_ylim(min(y_data) - 1, max(max(y_data), max(x_data)) + 1)


def plot_sift_signature(sift, title='', fnum=None, pnum=None):
    """
    Plots a SIFT descriptor as a histogram and distinguishes different bins
    into different colors

    Args:
        sift (ndarray[dtype=np.uint8]):
        title (str):  (default = '')
        fnum (int):  figure number(default = None)
        pnum (tuple):  plot number(default = None)

    Returns:
        AxesSubplot: ax

    CommandLine:
        python -m plottool.draw_func2 --test-plot_sift_signature --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> sift = vt.dummy.testdata_dummy_sift(1, np.random.RandomState(0))[0]
        >>> title = 'test sift histogram'
        >>> fnum = None
        >>> pnum = None
        >>> ax = plot_sift_signature(sift, title, fnum, pnum)
        >>> result = ('ax = %s' % (str(ax),))
        >>> print(result)
        >>> ut.show_if_requested()
    """
    fnum = ensure_fnum(fnum)
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    plot_bars(sift, 16)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 256)
    space_xticks(9, 16)
    space_yticks(5, 64)
    set_title(title, ax=ax)
    #dark_background(ax)

    use_darkbackground = None
    if use_darkbackground is None:
        use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background(ax)
    return ax


def plot_descriptor_signature(vec, title='', fnum=None, pnum=None):
    """
    signature general for for any descriptor vector.

    Args:
        vec (ndarray):
        title (str):  (default = '')
        fnum (int):  figure number(default = None)
        pnum (tuple):  plot number(default = None)

    Returns:
        AxesSubplot: ax

    CommandLine:
        python -m plottool.draw_func2 --test-plot_descriptor_signature --show

    SeeAlso:
        plot_sift_signature

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> vec = ((np.random.RandomState(0).rand(258) - .2) * 4)
        >>> title = 'test sift histogram'
        >>> fnum = None
        >>> pnum = None
        >>> ax = plot_descriptor_signature(vec, title, fnum, pnum)
        >>> result = ('ax = %s' % (str(ax),))
        >>> print(result)
        >>> ut.show_if_requested()
    """
    fnum = ensure_fnum(fnum)
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    plot_bars(vec, vec.size // 8)
    ax.set_xlim(0, vec.size)
    ax.set_ylim(vec.min(), vec.max())
    #space_xticks(9, 16)
    #space_yticks(5, 64)
    set_title(title, ax=ax)

    use_darkbackground = None
    if use_darkbackground is None:
        use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background(ax)

    return ax


def dark_background(ax=None, doubleit=False):
    bgcolor = BLACK * .9
    if ax is None:
        ax = gca()
    from mpl_toolkits.mplot3d import Axes3D
    if isinstance(ax, Axes3D):
        ax.set_axis_bgcolor(bgcolor)
        ax.tick_params(colors='white')
        return
    xy, width, height = get_axis_xy_width_height(ax)
    if doubleit:
        halfw = (doubleit) * (width / 2)
        halfh = (doubleit) * (height / 2)
        xy = (xy[0] - halfw, xy[1] - halfh)
        width *= (doubleit + 1)
        height *= (doubleit + 1)
    rect = mpl.patches.Rectangle(xy, width, height, lw=0, zorder=0)
    rect.set_clip_on(True)
    rect.set_fill(True)
    rect.set_color(bgcolor)
    rect = ax.add_patch(rect)


def space_xticks(nTicks=9, spacing=16, ax=None):
    if ax is None:
        ax = gca()
    ax.set_xticks(np.arange(nTicks) * spacing)
    small_xticks(ax)


def space_yticks(nTicks=9, spacing=32, ax=None):
    if ax is None:
        ax = gca()
    ax.set_yticks(np.arange(nTicks) * spacing)
    small_yticks(ax)


def small_xticks(ax=None):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)


def small_yticks(ax=None):
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)


def plot_bars(y_data, nColorSplits=1):
    width = 1
    nDims = len(y_data)
    nGroup = nDims // nColorSplits
    ori_colors = distinct_colors(nColorSplits)
    x_data = np.arange(nDims)
    ax = gca()
    for ix in range(nColorSplits):
        xs = np.arange(nGroup) + (nGroup * ix)
        color = ori_colors[ix]
        x_dat = x_data[xs]
        y_dat = y_data[xs]
        ax.bar(x_dat, y_dat, width, color=color, edgecolor=np.array(color) * .8)


def append_phantom_legend_label(label, color, type_='circle', alpha=1.0):
    """
    adds a legend label without displaying an actor

    Args:
        label (?):
        color (?):
        loc (str):

    CommandLine:
        python -m plottool.draw_func2 --test-append_phantom_legend_label

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> # build test data
        >>> label = 'some label'
        >>> color = 'b'
        >>> loc = 'upper right'
        >>> # execute function
        >>> result = append_phantom_legend_label(label, color, loc)
        >>> # verify results
        >>> print(result)
    """
    #pass
    #, loc=loc
    ax = gca()
    _phantom_legend_list = getattr(ax, '_phantom_legend_list', None)
    if _phantom_legend_list is None:
        _phantom_legend_list = []
        setattr(ax, '_phantom_legend_list', _phantom_legend_list)
    if type_ == 'line':
        phantom_actor = plt.Line2D((0, 0), (1, 1), color=color, label=label,
                                   alpha=alpha)
    else:
        phantom_actor = plt.Circle((0, 0), 1, fc=color, label=label, alpha=alpha)
    #, prop=custom_constants.FONTS.legend)
    #legend_tups = []
    _phantom_legend_list.append(phantom_actor)
    #ax.legend(handles=[phantom_actor], framealpha=.2)
    #plt.legend(*zip(*legend_tups), framealpha=.2)


def show_phantom_legend_labels(loc='upper right'):
    ax = gca()
    _phantom_legend_list = getattr(ax, '_phantom_legend_list', None)
    if _phantom_legend_list is None:
        _phantom_legend_list = []
        setattr(ax, '_phantom_legend_list', _phantom_legend_list)
    #print(_phantom_legend_list)
    ax.legend(handles=_phantom_legend_list, framealpha=.2)


LEGEND_LOCATION = {
    'upper right':  1,
    'upper left':   2,
    'lower left':   3,
    'lower right':  4,
    'right':        5,
    'center left':  6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center':      10,
}


#def legend(loc='upper right', fontproperties=None):
def legend(loc='best', fontproperties=None, size=None, fc='w', alpha=1):
    r"""
    Args:
        loc (str): (default = 'best')
        fontproperties (None): (default = None)
        size (None): (default = None)

    CommandLine:
        python -m plottool.draw_func2 --exec-legend --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> loc = 'best'
        >>> import plottool as pt
        >>> xdata = np.linspace(-6, 6)
        >>> ydata = np.sin(xdata)
        >>> pt.plot(xdata, ydata, label='sin')
        >>> fontproperties = None
        >>> size = None
        >>> result = legend(loc, fontproperties, size)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    assert loc in LEGEND_LOCATION or loc == 'best', (
        'invalid loc. try one of %r' % (LEGEND_LOCATION,))
    ax = gca()
    if fontproperties is None:
        fontproperties = custom_constants.FONTS.legend
    prop = {}
    if size is not None:
        prop['size'] = size
    legend = ax.legend(loc=loc, prop=prop)
    legend.get_frame().set_fc(fc)
    legend.get_frame().set_alpha(alpha)


def plot_histpdf(data, label=None, draw_support=False, nbins=10):
    freq, _ = plot_hist(data, nbins=nbins)
    from plottool import plots
    plots.plot_pdf(data, draw_support=draw_support, scale_to=freq.max(), label=label)


def plot_hist(data, bins=None, nbins=10, weights=None):
    if isinstance(data, list):
        data = np.array(data)
    dmin = data.min()
    dmax = data.max()
    if bins is None:
        bins = dmax - dmin
    ax  = gca()
    freq, bins_, patches = ax.hist(data, bins=nbins, weights=weights, range=(dmin, dmax))
    return freq, bins_


def variation_trunctate(data):
    ax = gca()
    data = np.array(data)
    if len(data) == 0:
        warnstr = '[df2] ! Warning: len(data) = 0. Cannot variation_truncate'
        warnings.warn(warnstr)
        return
    trunc_max = data.mean() + data.std() * 2
    trunc_min = np.floor(data.min())
    ax.set_xlim(trunc_min, trunc_max)
    #trunc_xticks = np.linspace(0, int(trunc_max),11)
    #trunc_xticks = trunc_xticks[trunc_xticks >= trunc_min]
    #trunc_xticks = np.append([int(trunc_min)], trunc_xticks)
    #no_zero_yticks = ax.get_yticks()[ax.get_yticks() > 0]
    #ax.set_xticks(trunc_xticks)
    #ax.set_yticks(no_zero_yticks)
#_----------------- HELPERS ^^^ ---------


def scores_to_color(score_list, cmap_='hot', logscale=False, reverse_cmap=False,
                    custom=False, val2_customcolor=None, scale_min=.1,
                    scale_max=.9):
    """
    Other good colormaps are 'spectral', 'gist_rainbow', 'gist_ncar', 'Set1', 'Set2', 'Accent'

    Args:
        score_list (list):
        cmap_ (str): defaults to hot
        logscale (bool):

    Returns:
        <class '_ast.ListComp'>

    Example:
        >>> from plottool.draw_func2 import *  # NOQA
        >>> score_list = np.array([-1, -2, 1, 1, 2, 10])
        >>> cmap_ = 'hot'
        >>> logscale = False
        >>> reverse_cmap = True
        >>> custom = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
    """
    if DEBUG:
        print('scores_to_color()')
    assert len(score_list.shape) == 1, 'score must be 1d'
    if len(score_list) == 0:
        return []
    if logscale:
        score_list = np.log2(np.log2(score_list + 2) + 1)
    cmap = plt.get_cmap(cmap_)
    if reverse_cmap:
        cmap = reverse_colormap(cmap)
    #if custom:
    #    base_colormap = cmap
    #    data = score_list
    #    cmap = customize_colormap(score_list, base_colormap)
    min_ = score_list.min()
    range_ = score_list.max() - min_
    if range_ == 0:
        colors = [cmap(.5) for fx in range(len(score_list))]
    else:
        if logscale:
            def score2_01(score):
                return np.log2(
                    1 + scale_min + scale_max *
                    (float(score) - min_) / (range_))
            score_list = np.array(score_list)
            #rank_multiplier = score_list.argsort() / len(score_list)
            #normscore = np.array(list(map(score2_01, score_list))) * rank_multiplier
            normscore = np.array(list(map(score2_01, score_list)))
            colors =  list(map(cmap, normscore))
        else:
            def score2_01(score):
                return scale_min + scale_max * (float(score) - min_) / (range_)
        colors = [cmap(score2_01(score)) for score in score_list]
        if val2_customcolor is not None:
            colors = [
                np.array(val2_customcolor.get(score, color))
                for color, score in zip(colors, score_list)]
    return colors


def customize_colormap(data, base_colormap):
    unique_scalars = np.array(sorted(np.unique(data)))
    max_ = unique_scalars.max()
    min_ = unique_scalars.min()
    range_ = max_ - min_
    bounds = np.linspace(min_, max_ + 1, range_ + 2)

    # Get a few more colors than we actually need so we don't hit the bottom of
    # the cmap
    colors_ix = np.concatenate((np.linspace(0, 1., range_ + 2), (0., 0., 0., 0.)))
    colors_rgba = base_colormap(colors_ix)
    # TODO: parametarize
    val2_special_rgba = {
        -1: UNKNOWN_PURP,
        -2: LIGHT_BLUE,
    }
    def get_new_color(ix, val):
        if val in val2_special_rgba:
            return val2_special_rgba[val]
        else:
            return colors_rgba[ix - len(val2_special_rgba) + 1]
    special_colors = [get_new_color(ix, val) for ix, val in enumerate(bounds)]

    cmap = mpl.colors.ListedColormap(special_colors)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sm.set_clim(-.5, range_ + 0.5)
    #colorbar = plt.colorbar(sm)

    #missing_ixs = find_nonconsec_indices(unique_scalars, bounds)
    #sel_bounds = np.array([x for ix, x in enumerate(bounds) if ix not in missing_ixs])

    #ticks = sel_bounds + .5
    #ticklabels = sel_bounds
    #colorbar.set_ticks(ticks)  # tick locations
    #colorbar.set_ticklabels(ticklabels)  # tick labels
    return cmap


def unique_rows(arr):
    """
    References:
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """
    rowblocks = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idx = np.unique(rowblocks, return_index=True)
    unique_arr = arr[idx]
    return unique_arr


def scores_to_cmap(scores, colors=None, cmap_='hot'):
    if DEBUG:
        print('scores_to_cmap()')
    if colors is None:
        colors = scores_to_color(scores, cmap_=cmap_)
    scores = np.array(scores)
    colors = np.array(colors)
    sortx = scores.argsort()
    sorted_colors = colors[sortx]
    # Make a listed colormap and mappable object
    listed_cmap = mpl.colors.ListedColormap(sorted_colors)
    return listed_cmap


DF2_DIVIDER_KEY = '_df2_divider'


def ensure_divider(ax):
    """ Returns previously constructed divider or creates one """
    from plottool import plot_helpers as ph
    divider = ph.get_plotdat(ax, DF2_DIVIDER_KEY, None)
    if divider is None:
        divider = make_axes_locatable(ax)
        ph.set_plotdat(ax, DF2_DIVIDER_KEY, divider)
    return divider


def get_binary_svm_cmap():
    # useful for svms
    return reverse_colormap(plt.get_cmap('bwr'))


def reverse_colormap(cmap):
    """
    References:
        http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Matteo_colourmaps.ipynb
    """
    if isinstance(cmap,  mpl.colors.ListedColormap):
        return mpl.colors.ListedColormap(cmap.colors[::-1])
    else:
        reverse = []
        k = []
        for key, channel in six.iteritems(cmap._segmentdata):
            data = []
            for t in channel:
                data.append((1 - t[0], t[1], t[2]))
            k.append(key)
            reverse.append(sorted(data))
        cmap_reversed = mpl.colors.LinearSegmentedColormap(
            cmap.name + '_reversed', dict(zip(k, reverse)))
        return cmap_reversed


def print_valid_cmaps():
    import pylab
    import utool as ut
    maps = [m for m in pylab.cm.datad if not m.endswith("_r")]
    print(ut.list_str(sorted(maps)))


def show_all_colormaps():
    """
    Displays at a 90 degree angle. Weird

    FIXME: Remove call to pylab

    References:
        http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        http://matplotlib.org/examples/color/colormaps_reference.html

    CommandLine:
        python -m plottool.draw_func2 --test-show_all_colormaps --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import plottool as pt
        >>> show_all_colormaps()
        >>> pt.show_if_requested()
    """
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
    maps = [m for m in pylab.cm.datad if not m.endswith("_r")]
    maps.sort()
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
        pylab.imshow(a, aspect='auto', cmap=pylab.get_cmap(m))  # , origin="lower")
        if TRANSPOSE:
            ax.set_ylabel(m, rotation=0, fontsize=10,
                          horizontalalignment='right', verticalalignment='center')
        else:
            pylab.title(m, rotation=90, fontsize=10)
    #pylab.savefig("colormaps.png", dpi=100, facecolor='gray')


def colorbar(scalars, colors, custom=False, lbl=None):
    """
    adds a color bar next to the axes based on specific scalars

    Args:
        scalars (ndarray):
        colors (ndarray):

    Returns:
        cb : matplotlib colorbar object

    CommandLine:
        python -m plottool.draw_func2 --exec-colorbar --show

    Example:
        >>> from plottool.draw_func2 import *  # NOQA
        >>> from plottool import draw_func2 as df2
        >>> from plottool.draw_func2 import *  # NOQA
        >>> scalars = np.array([-1, -2, 1, 1, 2, 7, 10])
        >>> cmap_ = 'hot'
        >>> logscale = False
        >>> custom = True
        >>> reverse_cmap = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale, reverse_cmap=reverse_cmap, val2_customcolor=val2_customcolor)
        >>> colorbar(scalars, colors, custom=custom)
        >>> df2.present()
        >>> import plottool as pt
        >>> pt.show_if_requested()
    """
    assert len(scalars) == len(colors), 'scalars and colors must be corresponding'
    if len(scalars) == 0:
        return None
    # Parameters
    ax = gca()
    divider = ensure_divider(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    xy, width, height = get_axis_xy_width_height(ax)
    #orientation = ['vertical', 'horizontal'][0]
    TICK_FONTSIZE = 8
    #
    # Create scalar mappable with cmap
    if custom:
        # FIXME: clean this code up and change the name custom
        # to be meaningful. It is more like: display unique colors
        unique_scalars, unique_idx = np.unique(scalars, return_index=True)
        unique_colors = np.array(colors)[unique_idx]
        #max_, min_ = unique_scalars.max(), unique_scalars.min()
        #range_ = max_ - min_
        #bounds = np.linspace(min_, max_ + 1, range_ + 2)
        listed_cmap = mpl.colors.ListedColormap(unique_colors)
        #norm = mpl.colors.BoundaryNorm(bounds, listed_cmap.N)
        #sm = mpl.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
        sm = mpl.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(np.linspace(0, 1, len(unique_scalars) + 1))
    else:
        sorted_scalars = sorted(scalars)
        listed_cmap = scores_to_cmap(scalars, colors)
        sm = plt.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(sorted_scalars)
    # Use mapable object to create the colorbar
    #COLORBAR_SHRINK = .42  # 1
    #COLORBAR_PAD = .01  # 1
    #COLORBAR_ASPECT = np.abs(20 * height / (width))  # 1

    cb = plt.colorbar(sm, cax=cax)

    ## Add the colorbar to the correct label
    #axis = cb.ax.yaxis  # if orientation == 'horizontal' else cb.ax.yaxis
    #position = 'bottom' if orientation == 'horizontal' else 'right'
    #axis.set_ticks_position(position)

    # This line alone removes data
    # axis.set_ticks([0, .5, 1])
    if custom:
        #missing_ixs = ut.find_nonconsec_indices(unique_scalars, bounds)
        #sel_bounds = np.array([x for ix, x in enumerate(bounds) if ix not in missing_ixs])
        #ticks = sel_bounds + .5 - (sel_bounds.min())
        #ticklabels = sel_bounds
        #ticks = bounds
        ticks = np.linspace(0, 1, len(unique_scalars) + 1)
        if len(ticks) < 2:
            ticks += .5
        else:
            # SO HACKY
            ticks += (ticks[1] - ticks[0]) / 2

        if isinstance(unique_scalars, np.ndarray) and ut.is_float(unique_scalars):
            ticklabels = ['%.2f' % scalar for scalar in unique_scalars]
        else:
            ticklabels = unique_scalars
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)  # tick labels

    # FIXME: Figure out how to make a maximum number of ticks
    # and to enforce them to be inside the data bounds
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    # Sets current axis
    plt.sca(ax)
    if lbl is not None:
        cb.set_label(lbl)
    return cb


def draw_lines2(kpts1, kpts2, fm=None, fs=None, kpts2_offset=(0, 0),
                color_list=None, scale_factor=1, lw=1.4, line_alpha=.35,
                H1=None, H2=None, scale_factor1=None, scale_factor2=None, **kwargs):
    import vtool as vt
    if scale_factor1 is None:
        scale_factor1 = 1.0, 1.0
    if scale_factor2 is None:
        scale_factor2 = 1.0, 1.0
    # input data
    if fm is None:  # assume kpts are in director correspondence
        assert kpts1.shape == kpts2.shape, 'bad shape'
    if len(fm) == 0:
        return
    ax = gca()
    woff, hoff = kpts2_offset
    # Draw line collection
    kpts1_m = kpts1[fm[:, 0]].T
    kpts2_m = kpts2[fm[:, 1]].T
    xy1_m = (kpts1_m[0:2])
    xy2_m = (kpts2_m[0:2])
    if H1 is not None:
        xy1_m = vt.transform_points_with_homography(H1, xy1_m)
    if H2 is not None:
        xy2_m = vt.transform_points_with_homography(H2, xy2_m)
    xy1_m = xy1_m * scale_factor * np.array(scale_factor1)[:, None]
    xy2_m = (
        (xy2_m * scale_factor * np.array(scale_factor2)[:, None]) +
        np.array([woff, hoff])[:, None])
    if color_list is None:
        if fs is None:  # Draw with solid color
            color_list    = [RED for fx in range(len(fm))]
        else:  # Draw with colors proportional to score difference
            color_list = scores_to_color(fs)
    segments = [
        ((x1, y1), (x2, y2))
        for (x1, y1), (x2, y2) in zip(xy1_m.T, xy2_m.T)
    ]
    linewidth = [lw for fx in range(len(fm))]
    line_alpha = line_alpha

    line_group = mpl.collections.LineCollection(
        segments, linewidth, color_list, alpha=line_alpha)
    #plt.colorbar(line_group, ax=ax)
    ax.add_collection(line_group)
    #figure(100)
    #plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)


def draw_line_segments(segments_list, **kwargs):
    """
    segments_list - list of [xs,ys,...] defining the segments
    """
    import plottool as pt
    marker = '.-'
    for data in segments_list:
        pt.plot(data.T[0], data.T[1], marker, **kwargs)

    #from matplotlib.collections import LineCollection
    #points_list = [np.array([pts[0], pts[1]]).T.reshape(-1, 1, 2) for pts in segments_list]
    #segments_list = [np.concatenate([points[:-1], points[1:]], axis=1) for points in points_list]
    #linewidth = 2
    #alpha = 1.0
    #lc_list = [LineCollection(segments, linewidth=linewidth, alpha=alpha)
    #           for segments in segments_list]
    #ax = plt.gca()
    #for lc in lc_list:
    #    ax.add_collection(lc)


def draw_patches_and_sifts(patch_list, sift_list, fnum=None, pnum=(1, 1, 1)):
    # Hacked together will not work on inputs of all sizes
    #raise NotImplementedError('unfinished')
    import plottool as pt
    num, width, height = patch_list.shape[0:3]
    rows = int(np.sqrt(num))
    cols = num // rows
    # TODO: recursive stack
    #stacked_img = patch_list.transpose(2, 0, 1).reshape(height * rows, width * cols)
    stacked_img = np.vstack([np.hstack(chunk) for chunk in ut.ichunks(patch_list, rows)])

    x_base = ((np.arange(rows) + .5 ) * width) - .5
    y_base = ((np.arange(cols) + .5 ) * height) - .5
    xs, ys = np.meshgrid(x_base, y_base)

    tmp_kpts = np.vstack(
        (xs.flatten(),
         ys.flatten(),
         width / 2 * np.ones(len(patch_list)),
         np.zeros(len(patch_list)),
         height / 2 * np.ones(len(patch_list)),
         np.zeros(len(patch_list)))).T

    pt.figure(fnum=fnum, pnum=pnum, docla=True)
    pt.imshow(stacked_img, pnum=pnum, fnum=fnum)
    #ax = pt.gca()
    #ax.invert_yaxis()
    #ax.invert_xaxis()
    if sift_list is not None:
        pt.draw_kpts2(tmp_kpts, sifts=sift_list)
    return gca()
    #pt.iup()


def draw_kpts2(kpts, offset=(0, 0), scale_factor=1,
               ell=True, pts=False, rect=False, eig=False, ori=False,
               pts_size=2, ell_alpha=.6, ell_linewidth=1.5,
               ell_color=None, pts_color=ORANGE, color_list=None, pts_alpha=1.0,
               siftkw={}, H=None, weights=None, cmap_='hot', **kwargs):
    """
    thin wrapper around mpl_keypoint.draw_keypoints

    FIXME: seems to be off by (.5, .5) translation

    Args:
        kpts (?):
        offset (tuple):
        scale_factor (int):
        ell (bool):
        pts (bool):
        rect (bool):
        eig (bool):
        ori (bool):
        pts_size (int):
        ell_alpha (float):
        ell_linewidth (float):
        ell_color (None):
        pts_color (ndarray):
        color_list (list):

    Example:
        >>> from plottool.draw_func2 import *  # NOQA
        >>> from plottool import draw_func2 as df2
        >>> offset = (0, 0)
        >>> scale_factor = 1
        >>> ell = True
        >>> ell=True
        >>> pts=False
        >>> rect=False
        >>> eig=False
        >>> ell=True
        >>> pts=False
        >>> rect=False
        >>> eig=False
        >>> ori=False
        >>> pts_size=2
        >>> ell_alpha=.6
        >>> ell_linewidth=1.5
        >>> ell_color=None
        >>> pts_color=df2.ORANGE
        >>> color_list=None
    """
    if ell_color is None:
        ell_color = kwargs.get('color', BLUE)

    if isinstance(kpts, list):
        # ensure numpy
        kpts = np.array(kpts)

    #if ut.DEBUG2:
    #    printDBG('-------------')
    #    printDBG('draw_kpts2():')
    #    #printDBG(' * kwargs.keys()=%r' % (kwargs.keys(),))
    #    printDBG(' * kpts.shape=%r:' % (kpts.shape,))
    #    printDBG(' * ell=%r pts=%r' % (ell, pts))
    #    printDBG(' * rect=%r eig=%r, ori=%r' % (rect, eig, ori))
    #    printDBG(' * scale_factor=%r' % (scale_factor,))
    #    printDBG(' * offset=%r' % (offset,))
    #    printDBG(' * drawing kpts.shape=%r' % (kpts.shape,))
    try:
        assert len(kpts) > 0, 'len(kpts) < 0'
    except AssertionError as ex:
        ut.printex(ex)
        return
    ax = gca()
    if color_list is None and weights is not None:
        # hack to turn into a color map
        color_list = scores_to_color(weights, cmap_=cmap_, reverse_cmap=False)
    if color_list is not None:
        ell_color = color_list
        pts_color = color_list
    #else:
        #pts_color = [pts_color for _ in range(len(kpts))]
    if ell_color == 'distinct':
        ell_color = distinct_colors(len(kpts))  # , randomize=True)
        #print(len(kpts))

    _kwargs = kwargs.copy()
    _kwargs.update({
        # offsets
        'offset': offset,
        'scale_factor': scale_factor,
        # flags
        'pts': pts,
        'ell': ell,
        'ori': ori,
        'rect': rect,
        'eig': eig,
        # properties
        'ell_color': ell_color,
        'ell_alpha': ell_alpha,
        'ell_linewidth': ell_linewidth,
        'pts_color': pts_color,
        'pts_alpha': pts_alpha,
        'pts_size': pts_size,
    })

    mpl_kp.draw_keypoints(ax, kpts, siftkw=siftkw, H=H, **_kwargs)
    return color_list


def draw_keypoint_gradient_orientations(rchip, kpt, sift=None, mode='vec',
                                        kptkw={}, siftkw={}, **kwargs):
    """
    Extracts a keypoint patch from a chip, extract the gradient, and visualizes
    it with respect to the current mode.

    """
    import vtool as vt
    wpatch, wkp  = vt.get_warped_patch(rchip, kpt, gray=True)
    try:
        gradx, grady = vt.patch_gradient(wpatch)
    except Exception as ex:
        print('!!!!!!!!!!!!')
        print('[df2!] Exception = ' + str(ex))
        print('---------')
        print('type(wpatch) = ' + str(type(wpatch)))
        print('repr(wpatch) = ' + str(repr(wpatch)))
        print('wpatch = ' + str(wpatch))
        raise
    if mode == 'vec' or mode == 'vecfield':
        draw_vector_field(gradx, grady, **kwargs)
    elif mode == 'col' or mode == 'colors':
        import plottool
        gmag = vt.patch_mag(gradx, grady)
        gori = vt.patch_ori(gradx, grady)
        gorimag = plottool.color_orimag(gori, gmag)
        imshow(gorimag, **kwargs)
    wkpts = np.array([wkp])
    sifts = np.array([sift]) if sift is not None else None
    draw_kpts2(wkpts, sifts=sifts, siftkw=siftkw, **kptkw)


#@ut.indent_func('[df2.dkp]')
def draw_keypoint_patch(rchip, kp, sift=None, warped=False, patch_dict={}, **kwargs):
    r"""
    Args:
        rchip (ndarray[uint8_t, ndim=2]):  rotated annotation image data
        kp (ndarray[float32_t, ndim=1]):  a single keypoint
        sift (None): (default = None)
        warped (bool): (default = False)
        patch_dict (dict): (default = {})

    Returns:
        ?: ax

    CommandLine:
        python -m plottool.draw_func2 --test-draw_keypoint_patch --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> rchip = vt.imread(ut.grab_test_imgpath('lena.png'))
        >>> kp = [100, 100, 20, 0, 20, 0]
        >>> sift = None
        >>> warped = True
        >>> patch_dict = {}
        >>> ax = draw_keypoint_patch(rchip, kp, sift, warped, patch_dict)
        >>> result = ('ax = %s' % (str(ax),))
        >>> print(result)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    #print('--------------------')
    kpts = np.array([kp])
    if warped:
        patches, subkpts = vt.get_warped_patches(rchip, kpts)
    else:
        patches, subkpts = vt.get_unwarped_patches(rchip, kpts)
    #print('[df2] kpts[0]    = %r' % (kpts[0]))
    #print('[df2] subkpts[0] = %r' % (subkpts[0]))
    #print('[df2] patches[0].shape = %r' % (patches[0].shape,))
    patch = patches[0]
    subkpts_ = np.array(subkpts)
    patch_dict_ = {
        'sifts': None if sift is None else np.array([sift]),
        'ell_color':  (0, 0, 1),
        'pts': kwargs.get('pts', True),
        'ori': kwargs.get('ori', True),
        'ell': True,
        'eig': False,
        'rect': kwargs.get('rect', True),
        'multicolored_arms': kwargs.get('multicolored_arms', False),
    }
    patch_dict_.update(patch_dict)
    # Draw patch with keypoint overlay
    fig, ax = imshow(patch, **kwargs)
    draw_kpts2(subkpts_, **patch_dict_)
    return ax


# ---- CHIP DISPLAY COMMANDS ----
def imshow(img, fnum=None, title=None, figtitle=None, pnum=None,
           interpolation='nearest', cmap=None, heatmap=False,
           data_colorbar=False, darken=DARKEN, update=False,
           redraw_image=True, **kwargs):
    """
    Args:
        img (ndarray):  image data
        fnum (int):  figure number
        title (str):
        figtitle (None):
        pnum (tuple):  plot number
        interpolation (str): other interpolations = nearest, bicubic, bilinear
        cmap (None):
        heatmap (bool):
        data_colorbar (bool):
        darken (None):
        redraw_image (bool): used when calling imshow over and over. if false
                            doesnt do the image part.

    Returns:
        tuple: (fig, ax)
    """
    fig = figure(fnum=fnum, pnum=pnum, title=title, figtitle=figtitle, **kwargs)
    ax = gca()

    if not redraw_image:
        return fig, ax

    if isinstance(img, six.string_types):
        # Allow for path to image to be specified
        img_fpath = img
        ut.assertpath(img_fpath)
        import vtool as vt
        img = vt.imread(img_fpath)
    #darken = .4
    if darken is not None:
        if darken is True:
            darken = .5
        # Darken the shown picture
        imgdtype = img.dtype
        img = np.array(img, dtype=float) * darken
        img = np.array(img, dtype=imgdtype)

    plt_imshow_kwargs = {
        'interpolation': interpolation,
        #'cmap': plt.get_cmap('gray'),
    }
    if cmap is None and not heatmap:
        plt_imshow_kwargs['vmin'] = 0
        plt_imshow_kwargs['vmax'] = 255
    if heatmap:
        cmap = 'hot'
    try:
        if len(img.shape) == 3 and (img.shape[2] == 3 or img.shape[2] == 4):
            # img is in a color format
            imgBGR = img
            if imgBGR.dtype == np.float64:
                if imgBGR.max() <= 1:
                    imgBGR = np.array(imgBGR, dtype=np.float32)
                else:
                    imgBGR = np.array(imgBGR, dtype=np.uint8)
            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            ax.imshow(imgRGB, **plt_imshow_kwargs)
        elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # img is in grayscale
            if len(img.shape) == 3:
                imgGRAY = img.reshape(img.shape[0:2])
            else:
                imgGRAY = img
            if cmap is None:
                cmap = plt.get_cmap('gray')
            if isinstance(cmap, six.string_types):
                cmap = plt.get_cmap(cmap)
            ax.imshow(imgGRAY, cmap=cmap, **plt_imshow_kwargs)
        else:
            raise AssertionError(
                'unknown image format. img.dtype=%r, img.shape=%r' %
                (img.dtype, img.shape))
    except TypeError as te:
        print('[df2] imshow ERROR %r' % (te,))
        raise
    except Exception as ex:
        print('!!!!!!!!!!!!!!WARNING!!!!!!!!!!!')
        print('[df2] type(img) = %r' % type(img))
        if not isinstance(img, np.ndarray):
            print('!!!!!!!!!!!!!!ERRROR!!!!!!!!!!!')
            pass
            #print('img = %r' % (img,))
        print('[df2] img.dtype = %r' % (img.dtype,))
        print('[df2] type(img) = %r' % (type(img),))
        print('[df2] img.shape = %r' % (img.shape,))
        print('[df2] imshow ERROR %r' % ex)
        raise
    #plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

    if data_colorbar is True:
        scores = np.unique(img.flatten())
        if cmap is None:
            cmap = 'hot'
        colors = scores_to_color(scores, cmap)
        colorbar(scores, colors)

    if figtitle is not None:
        custom_figure.set_figtitle(figtitle)
    if update:
        fig_presenter.update()
    return fig, ax


def draw_vector_field(gx, gy, fnum=None, pnum=None, title=None, invert=True):
    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.quiver
    quiv_kw = {
        'units': 'xy',
        'scale_units': 'xy',
        #'angles': 'uv',
        #'scale': 80,
        #'width':
        'headaxislength': 4.5,
        'headlength': 5,
        'headwidth': 3,
        'minshaft': 1,
        'minlength': 1,
        #'color': 'r',
        #'edgecolor': 'k',
        'linewidths': (.5,),
        'pivot': 'tail',  # 'middle',
    }
    stride = 1
    #TAU = 2 * np.pi
    x_grid = np.arange(0, len(gx), 1)
    y_grid = np.arange(0, len(gy), 1)
    # Vector locations and directions
    X, Y = np.meshgrid(x_grid, y_grid)
    #X += .5
    #Y += .5
    U, V = gx, -gy
    # Apply stride
    X_ = X[::stride, ::stride]
    Y_ = Y[::stride, ::stride]
    U_ = U[::stride, ::stride]
    V_ = V[::stride, ::stride]
    # Draw arrows
    figure(fnum=fnum, pnum=pnum)
    plt.quiver(X_, Y_, U_, V_, **quiv_kw)
    # Plot properties
    ax = gca()
    ax.set_xticks([])
    ax.set_yticks([])
    if invert:
        ax.invert_yaxis()
    ax.set_aspect('equal')
    if title is not None:
        set_title(title)


def show_chipmatch2(rchip1, rchip2, kpts1=None, kpts2=None, fm=None, fs=None,
                    fm_norm=None, title=None,
                    vert=None, fnum=None, pnum=None, heatmap=False,
                    draw_fmatch=True, darken=DARKEN, H1=None, H2=None,
                    sel_fm=[], **kwargs):
    """
    Draws two chips and the feature matches between them. feature matches
    kpts1 and kpts2 use the (x,y,a,c,d)

    Args:
        rchip1 (ndarray): rotated annotation 1 image data
        rchip2 (ndarray): rotated annotation 2 image data
        kpts1 (ndarray): keypoints for annotation 1 [x, y, a=1, c=0, d=1, theta=0]
        kpts2 (ndarray): keypoints for annotation 2 [x, y, a=1, c=0, d=1, theta=0]
        fm (list):  list of feature matches as tuples (qfx, dfx)
        fs (list):  list of feature scores
        title (str):
        vert (None):
        fnum (int):  figure number
        pnum (tuple):  plot number
        heatmap (bool):
        draw_fmatch (bool):

     Returns:
         ax, xywh1, xywh2

    CommandLine:
        python -m plottool.draw_func2 --test-show_chipmatch2 --show

    Example:
        >>> # DISABLE_DOCTEST (TODO REMOVE IBEIS DOCTEST)
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import plottool as pt
        >>> import vtool as vt
        >>> # build test data
        >>> fname1 = ut.get_argval('--fname1', type_=str, default='easy1.png')
        >>> fname2 = ut.get_argval('--fname2', type_=str, default='easy2.png')
        >>> rchip1 = vt.imread(ut.grab_test_imgpath(fname1))
        >>> rchip2 = vt.imread(ut.grab_test_imgpath(fname2))
        >>> kpts1 = np.array([[ 430.84,  124.51,   12.98,   -1.54,    8.51,    0.  ],
        ...                   [ 355.89,  142.95,   10.46,   -0.63,    8.59,    0.  ],
        ...                   [ 356.35,  147.  ,    8.38,    1.08,   11.68,    0.  ],
        ...                   [ 361.4 ,  150.64,    7.44,    3.45,   13.63,    0.  ]], dtype=np.float64)
        >>> kpts2 = np.array([[ 466.01,   18.15,   13.24,   -3.74,    8.85,    0.  ],
        ...                   [ 376.98,   50.61,   11.91,   -2.9 ,    9.77,    0.  ],
        ...                   [ 377.59,   54.89,    9.7 ,   -1.4 ,   13.72,    0.  ],
        ...                   [ 382.8 ,   58.2 ,    7.87,   -0.31,   15.23,    0.  ]], dtype=np.float64)
        >>> fm = None
        >>> fs = None
        >>> H1 = np.array([[ -4.68815126e-01,   7.80306795e-02,  -2.23674587e+01],
        ...                [  4.54394231e-02,  -7.67438835e-01,   5.92158624e+01],
        ...                [  2.12918867e-04,  -8.64851418e-05,  -6.21472492e-01]])
        >>> H2 = None
        >>> H2 = None
        >>> # execute function
        >>> result = show_chipmatch2(rchip1, rchip2, kpts1, kpts2, H1=H1, H2=H2, fm=fm, ell_alpha=.9, ell_linewidth=5)
        >>> # verify results
        >>> print(result)
        >>> pt.show_if_requested()

    Ignore:
        print(ut.doctest_repr(kpts1[fm.T[0]][0:4], precision=2, varname='kpts1'))
        print(ut.doctest_repr(kpts2[fm.T[1]][0:4], precision=2, varname='kpts2'))
        print(ut.numpy_str2(kpts2[fm.T[1]][0:4], precision=1))
        #>>> fm = np.array([[ 244,  132], [ 604,  602], [ 187,  604], [ 200,  610],
        #...                [ 243,  627], [ 831,  819], [ 601,  851], [ 602,  852],
        #...                [ 610,  855], [ 609,  857], [ 865,  860], [ 617,  863],
        #...                [ 979,  984], [ 860, 1013], [ 605, 1020], [ 866, 1027],
        #...                [ 667, 1071], [1022, 1163], [1135, 1165]])
        #>>> ibs = ibeis.opendb('testdb1')
        #>>> aid1 = 1
        #>>> aid2 = 3
        #>>> chip1, chip2 = ibs.get_annot_chips((aid1, aid2))
        #>>> kpts1, kpts2 = ibs.get_annot_kpts((aid1, aid2))
    """
    if ut.VERBOSE:
        print('[df2] show_chipmatch2() fnum=%r, pnum=%r' % (fnum, pnum))
    import vtool as vt
    wh1 = vt.get_size(rchip1)
    wh2 = vt.get_size(rchip2)
    # Warp if homography is specified
    rchip1_ = vt.warpHomog(rchip1, H1, wh2) if H1 is not None else rchip1
    rchip2_ = vt.warpHomog(rchip2, H2, wh1) if H2 is not None else rchip2
    # get matching keypoints + offset
    (h1, w1) = rchip1_.shape[0:2]  # get chip (h, w) dimensions
    (h2, w2) = rchip2_.shape[0:2]
    # Stack the compared chips
    match_img, offset_tup, sf_tup = vt.stack_images(rchip1_, rchip2_, vert, return_sf=True)
    (woff, hoff) = offset_tup[1]
    xywh1 = (0, 0, w1, h1)
    xywh2 = (woff, hoff, w2, h2)
    # Show the stacked chips
    fig, ax = imshow(match_img, title=title, fnum=fnum, pnum=pnum, heatmap=heatmap, darken=darken)
    # Overlay feature match nnotations
    if draw_fmatch and kpts1 is not None and kpts2 is not None:
        plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs, fm_norm=fm_norm, H1=H1,
                    H2=H2, **kwargs)
        if len(sel_fm) > 0:
            # Draw any selected matches in blue
            sm_kw = dict(rect=True, colors=BLUE)
            plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    return ax, xywh1, xywh2


# plot feature match
def plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs=None, fm_norm=None, lbl1=None, lbl2=None,
                fnum=None, pnum=None, rect=False, colorbar_=True,
                draw_border=False, cmap=None, H1=None, H2=None,
                scale_factor1=None, scale_factor2=None,
                **kwargs):
    """
    Overlays the matching features over chips that were previously plotted.

    Args:
        xywh1 (tuple): location of rchip1 in the axes
        xywh2 (tuple): location or rchip2 in the axes
        kpts1 (ndarray):  keypoints in rchip1
        kpts2 (ndarray):  keypoints in rchip1
        fm (list): feature matches
        fs (list): features scores
        lbl1 (None): rchip1 label
        lbl2 (None): rchip2 label
        fnum (None): figure number
        pnum (None): plot number
        rect (bool):
        colorbar_ (bool):
        draw_border (bool):
    """
    if fm is None and fm_norm is None:
        assert kpts1.shape == kpts2.shape, 'shapes different or fm not none'
        fm = np.tile(np.arange(0, len(kpts1)), (2, 1)).T
    pts       = kwargs.get('draw_pts', False)
    ell       = kwargs.get('draw_ell', True)
    lines     = kwargs.get('draw_lines', True)
    ell_alpha = kwargs.get('ell_alpha', .4)
    nMatch = len(fm)
    x2, y2, w2, h2 = xywh2
    offset1 = (0., 0.)
    offset2 = (x2, y2)
    # THIS IS NOT WHERE THIS CODE BELONGS
    if False:
        # Custom user label for chips 1 and 2
        if lbl1 is not None:
            x1, y1, w1, h1 = xywh1
            absolute_lbl(x1 + w1, y1, lbl1)
        if lbl2 is not None:
            absolute_lbl(x2 + w2, y2, lbl2)
    # Plot the number of matches
    #if kwargs.get('show_nMatches', False):
    #    upperleft_text('#match=%d' % nMatch)
    # Draw all keypoints in both chips as points
    if kwargs.get('all_kpts', False):
        all_args = dict(ell=False, pts=pts, pts_color=GREEN, pts_size=2,
                        ell_alpha=ell_alpha, rect=rect)
        all_args.update(kwargs)
        draw_kpts2(kpts1, offset=offset1, H=H1, **all_args)
        draw_kpts2(kpts2, offset=offset2, H=H2, **all_args)
    if draw_border:
        draw_bbox(xywh1, bbox_color=BLACK, draw_arrow=False)
        draw_bbox(xywh2, bbox_color=BLACK, draw_arrow=False)

    if nMatch > 0:
        # draw lines and ellipses and points
        colors = [kwargs['colors']] * nMatch if 'colors' in kwargs else distinct_colors(nMatch)
        if fs is not None:
            if cmap is None:
                cmap = 'hot'
            colors = scores_to_color(fs, cmap)

        acols = add_alpha(colors)

        # Helper functions
        def _drawkpts(**_kwargs):
            _kwargs.update(kwargs)
            fxs1 = fm.T[0]
            fxs2 = fm.T[1]
            if kpts1 is not None:
                draw_kpts2(kpts1[fxs1], offset=offset1,
                           scale_factor=scale_factor1, rect=rect, H=H1,
                           **_kwargs)
            draw_kpts2(kpts2[fxs2], offset=offset2, scale_factor=scale_factor2,
                       rect=rect, H=H2, **_kwargs)

        def _drawlines(**_kwargs):
            _kwargs.update(kwargs)
            draw_lines2(kpts1, kpts2, fm, fs,
                        kpts2_offset=offset2,
                        scale_factor1=scale_factor1,
                        scale_factor2=scale_factor2,
                        H1=H1, H2=H2, **_kwargs)
            if fm_norm is not None:
                # NORMALIZING MATCHES IF GIVEN
                _kwargs_norm = _kwargs.copy()
                if fs is not None:
                    cmap = 'cool'
                    colors = scores_to_color(fs, cmap)
                _kwargs_norm['color_list'] = colors
                draw_lines2(kpts1, kpts2, fm_norm, fs, kpts2_offset=offset2,
                            H1=H1, H2=H2, scale_factor1=scale_factor1,
                            scale_factor2=scale_factor2, **_kwargs_norm)

        if ell:
            _drawkpts(pts=False, ell=True, color_list=colors)
        if pts:
            _drawkpts(pts_size=8, pts=True, ell=False, pts_color=BLACK)
            _drawkpts(pts_size=6, pts=True, ell=False, color_list=acols)
        if lines and kpts1 is not None:
            _drawlines(color_list=colors)
    else:
        # if not matches draw a big red X
        draw_boxedX(xywh2)
    # Turn off colorbar if there are no features being drawn
    # or the user doesnt want a colorbar
    drew_anything = fs is not None and (ell or pts or lines)
    has_colors = nMatch > 0 and colors is not None  # 'colors' in vars()
    if drew_anything and has_colors and colorbar_:
        colorbar(fs, colors)
    #legend()
    return None


def draw_boxedX(xywh=None, color=RED, lw=2, alpha=.5, theta=0):
    'draws a big red x. redx'
    ax = gca()
    if xywh is None:
        xy, w, h = get_axis_xy_width_height(ax)
        xywh = (xy[0], xy[1], w, h)
    x1, y1, w, h = xywh
    x2, y2 = x1 + w, y1 + h
    segments = [((x1, y1), (x2, y2)),
                ((x1, y2), (x2, y1))]
    trans = mpl.transforms.Affine2D()
    trans.rotate(theta)
    trans = trans + ax.transData
    width_list = [lw] * len(segments)
    color_list = [color] * len(segments)
    line_group = mpl.collections.LineCollection(segments, width_list,
                                                color_list, alpha=alpha,
                                                transOffset=trans)
    ax.add_collection(line_group)


def color_orimag(gori, gmag=None, gmag_is_01=None, encoding='rgb', p=.5):
    r"""
    Args:
        gori (ndarray): orientation values at pixels between 0 and tau
        gmag (ndarray): orientation magnitude
        gmag_is_01 (bool): True if gmag is in the 0 and 1 range. if None we try to guess
        p (float): power to raise normalized weights to for visualization purposes

    Returns:
        ndarray: rgb_ori or bgr_ori

    CommandLine:
        python -m plottool.draw_func2 --test-color_orimag --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import plottool as pt
        >>> import vtool as vt
        >>> # build test data
        >>> gori = np.array([[ 0.        ,  0.        ,  3.14159265,  3.14159265,  0.        ],
        ...                  [ 1.57079633,  3.92250052,  1.81294053,  3.29001537,  1.57079633],
        ...                  [ 4.71238898,  6.15139659,  0.76764078,  1.75632531,  1.57079633],
        ...                  [ 4.71238898,  4.51993581,  6.12565345,  3.87978382,  1.57079633],
        ...                  [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
        >>> gmag = np.array([[ 0.        ,  0.02160321,  0.00336692,  0.06290751,  0.        ],
        ...                      [ 0.02363726,  0.04195344,  0.29969492,  0.53007415,  0.0426679 ],
        ...                      [ 0.00459386,  0.32086307,  0.02844123,  0.24623816,  0.27344167],
        ...                      [ 0.04204251,  0.52165989,  0.25800464,  0.14568752,  0.023614  ],
        ...                      [ 0.        ,  0.05143869,  0.2744546 ,  0.01582246,  0.        ]])
        >>> # execute function
        >>> p = 1
        >>> bgr_ori1 = color_orimag(gori, gmag, encoding='bgr', p=p)
        >>> bgr_ori2 = color_orimag(gori, None, encoding='bgr')
        >>> legendimg = pt.make_ori_legend_img().astype(np.float32) / 255.0
        >>> gweights_color = np.dstack([gmag] * 3).astype(np.float32)
        >>> img, _, _ = vt.stack_images(bgr_ori2, gweights_color, vert=False)
        >>> img, _, _ = vt.stack_images(img, bgr_ori1, vert=False)
        >>> img, _, _ = vt.stack_images(img, legendimg, vert=True, modifysize=True)
        >>> # verify results
        >>> pt.imshow(img, pnum=(1, 2, 1))
        >>> # Hack orientation offset so 0 is downward
        >>> gradx, grady = np.cos(gori + TAU / 4.0), np.sin(gori + TAU / 4.0)
        >>> pt.imshow(bgr_ori2, pnum=(1, 2, 2))
        >>> pt.draw_vector_field(gradx, grady, pnum=(1, 2, 2), invert=False)
        >>> color_orimag_colorbar(gori)
        >>> pt.set_figtitle('weighted and unweighted orientaiton colors')
        >>> pt.update()
        >>> pt.show_if_requested()
    """
    # Turn a 0 to 1 orienation map into hsv colors
    #gori_01 = (gori - gori.min()) / (gori.max() - gori.min())
    if gori.max() > TAU or gori.min() < 0:
        print('WARNING: [color_orimag] gori might not be in radians')
    flat_rgb = get_orientation_color(gori.flatten())
    #flat_rgb = np.array(cmap_(), dtype=np.float32)
    rgb_ori_alpha = flat_rgb.reshape(np.hstack((gori.shape, [4])))
    rgb_ori = cv2.cvtColor(rgb_ori_alpha, cv2.COLOR_RGBA2RGB)
    hsv_ori = cv2.cvtColor(rgb_ori,       cv2.COLOR_RGB2HSV)
    # Darken colors based on magnitude
    if gmag is not None:
        # Hueristic hack
        if gmag_is_01 is None:
            gmag_is_01 = gmag.max() <= 1.0
        gmag_ = gmag if gmag_is_01 else gmag / max(255.0, gmag.max())
        # Weights modify just value
        gmag_ = gmag_ ** p
        #ut.embed()
        #SAT_CHANNEL = 1
        VAL_CHANNEL = 2
        #hsv_ori[:, :, SAT_CHANNEL] = gmag_
        hsv_ori[:, :, VAL_CHANNEL] = gmag_
    # Convert back to bgr
    #bgr_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2BGR)
    if encoding == 'rgb':
        rgb_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2RGB)
        return rgb_ori
    elif encoding == 'bgr':
        bgr_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2BGR)
        return bgr_ori
    else:
        raise AssertionError('unkonwn encoding=%r' % (encoding,))


def get_orientation_color(radians_list):
    r"""
    Args:
        radians_list (list):

    CommandLine:
        python -m plottool.draw_func2 --test-get_orientation_color

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> # build test data
        >>> radians_list = np.linspace(-1, 10, 10)
        >>> # execute function
        >>> result = get_orientation_color(radians_list)
        >>> # verify results
        >>> print(result)
    """
    TAU = np.pi * 2
    # Map radians to 0 to 1
    ori01_list = (radians_list % TAU) / TAU
    cmap_ = plt.get_cmap('hsv')
    color_list = cmap_(ori01_list)
    ori_colors_rgb = np.array(color_list, dtype=np.float32)
    return ori_colors_rgb


def color_orimag_colorbar(gori):
    TAU = np.pi * 2
    ori_list = np.linspace(0, TAU, 8)
    color_list = get_orientation_color(ori_list)
    colorbar(ori_list, color_list, lbl='orientation (radians)', custom=True)


def make_ori_legend_img():
    r"""

    creates a figure that shows which colors are associated with which keypoint
    rotations.

    a rotation of 0 should point downward (becuase it is relative the the (0, 1)
    keypoint eigenvector. and its color should be red due to the hsv mapping

    CommandLine:
        python -m plottool.draw_func2 --test-make_ori_legend_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.draw_func2 import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> # execute function
        >>> img_BGR = make_ori_legend_img()
        >>> # verify results
        >>> pt.imshow(img_BGR)
        >>> pt.iup()
        >>> pt.show_if_requested()
    """
    import plottool as pt
    TAU = 2 * np.pi
    NUM = 36
    NUM = 36 * 2
    domain = np.linspace(0, 1, NUM, endpoint=False)
    theta_list = domain * TAU
    relative_theta_list = theta_list + (TAU / 4)
    color_rgb_list = pt.get_orientation_color(theta_list)
    c_list = np.cos(relative_theta_list)
    r_list = np.sin(relative_theta_list)
    rc_list = list(zip(r_list, c_list))
    size = 1024
    radius =  (size / 5) * ut.PHI
    #size_root = size / 4
    half_size = size / 2
    img_BGR = np.zeros((size, size, 3), dtype=np.uint8)
    basis = np.arange(-7, 7)
    x_kernel_offset, y_kernel_offset = np.meshgrid(basis, basis)
    x_kernel_offset = x_kernel_offset.ravel()
    y_kernel_offset = y_kernel_offset.ravel()
    #x_kernel_offset = np.array([0, 1,  0, -1, -1, -1,  0,  1, 1])
    #y_kernel_offset = np.array([0, 1,  1,  1,  0, -1, -1, -1, 0])
    #new_data_weight = np.ones(x_kernel_offset.shape, dtype=np.int32)
    for color_rgb, (r, c) in zip(color_rgb_list, rc_list):
        row = x_kernel_offset + int(r * radius + half_size)
        col = y_kernel_offset + int(c * radius + half_size)
        #old_data = img[row, col, :]
        color = color_rgb[0:3] * 255
        color_bgr = color[::-1]
        #img_BGR[row, col, :] = color
        img_BGR[row, col, :] = color_bgr
        #new_data = img_BGR[row, col, :]
        #old_data_weight = np.array(list(map(np.any, old_data > 0)), dtype=np.int32)
        #total_weight = old_data_weight + 1
    import cv2
    for color_rgb, theta, (r, c) in list(zip(color_rgb_list, theta_list, rc_list))[::8]:
        row = int(r * (radius * 1.2) + half_size)
        col = int(c * (radius * 1.2) + half_size)
        text = str('t=%.2f' % (theta))
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        textcolor = [255, 255, 255]
        text_pt, text_sz = cv2.getTextSize(text, fontFace, fontScale, thickness)
        text_w, text_h = text_pt
        org = (int(col - text_w / 2), int(row + text_h / 2))
        #print(row)
        #print(col)
        #print(color_rgb)
        #print(text)
        cv2.putText(img_BGR, text, org, fontFace, fontScale, textcolor,
                    thickness, bottomLeftOrigin=False)
        #img_BGR[row, col, :] = ((old_data * old_data_weight[:, None] +
        #new_data) / total_weight[:, None])
    #print(img_BGR)
    return img_BGR


def remove_patches(ax=None):
    """ deletes patches from axes """
    if ax is None:
        ax = gca()
    for patch in ax.patches:
        del patch


def imshow_null(**kwargs):
    imshow(np.zeros((10, 10), dtype=np.uint8), **kwargs)
    draw_boxedX()


def axes_bottom_button_bar(ax, text_list=[]):
    # Method 2
    divider = make_axes_locatable(ax)
    ax_list = []
    but_list = []

    for text in text_list:
        ax = divider.append_axes('bottom', size='5%', pad=0.05)
        but = mpl.widgets.Button(ax, text)
        ax_list.append(ax)
        but_list.append(but)

    return but_list, ax_list
    """
    # Method 1
    (x1, y1), (x2, y2) = ax.get_position().get_points()
    # Parent axes props
    root_left   = x1
    root_bottom = y1
    root_height = y2 - y1
    root_width  = x2 - x1

    # Build axes for buttons
    num = len(text_list)
    pad_percent = .05
    rect_list = []
    xpad = root_width * pad_percent
    width = (root_width - (xpad * num)) / num
    height = root_height * .05
    left = root_left
    bottom = root_bottom - height
    for ix in range(num):
        rect = [left, bottom, width, height]
        rect_list.append(rect)
        left += width + xpad
    ax_list = [plt.axes(rect) for rect in rect_list]
    but_list = [mpl.widgets.Button(ax_, text) for ax_, text in zip(ax_list, text_list)]
    return but_list
    """


def make_bbox_positioners(y=.02, w=.08, h=.02, xpad=.05, startx=0, stopx=1):
    def hl_slot(ix):
        x = startx + (xpad * (ix + 1)) + ix * w
        return (x, y, w, h)

    def hr_slot(ix):
        x = stopx - ((xpad * (ix + 1)) + (ix + 1) * w)
        return (x, y, w, h)
    return hl_slot, hr_slot


def width_from(num, pad=.05, start=0, stop=1):
    return ((stop - start) - ((num + 1) * pad)) / num


#+-----
# From vtool.patch
def param_plot_iterator(param_list, fnum=None, projection=None):
    from plottool import plot_helpers
    nRows, nCols = plot_helpers.get_square_row_cols(len(param_list), fix=True)
    #next_pnum = make_pnum_nextgen(nRows=nRows, nCols=nCols)
    pnum_gen = pnum_generator(nRows, nCols)
    pnum = (nRows, nCols, 1)
    fig = figure(fnum=fnum, pnum=pnum)
    for param, pnum in zip(param_list, pnum_gen):
        # get next figure ready
        #print('fnum=%r, pnum=%r' % (fnum, pnum))
        if projection is not None:
            subplot_kw = {'projection': projection}
        else:
            subplot_kw = {}
        fig.add_subplot(*pnum, **subplot_kw)
        #figure(fnum=fnum, pnum=pnum)
        yield param


def plot_surface3d(xgrid, ygrid, zdata, xlabel=None, ylabel=None, zlabel=None,
                   wire=False, mode=None, contour=False, dark=False, rstride=1,
                   cstride=1, pnum=None, labelkw=None, xlabelkw=None,
                   ylabelkw=None, zlabelkw=None, titlekw=None, *args, **kwargs):
    """
    References:
        http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    """
    if titlekw is None:
        titlekw = {}
    if labelkw is None:
        labelkw = {}
    if xlabelkw is None:
        xlabelkw = labelkw.copy()
    if ylabelkw is None:
        ylabelkw = labelkw.copy()
    if zlabelkw is None:
        zlabelkw = labelkw.copy()
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    if pnum is None:
        ax = plt.gca(projection='3d')
    else:
        fig = plt.gcf()
        #print('pnum = %r' % (pnum,))
        ax = fig.add_subplot(*pnum, projection='3d')
    title = kwargs.pop('title', None)
    if mode is None:
        mode = 'wire' if wire else 'surface'

    if mode == 'wire':
        ax.plot_wireframe(xgrid, ygrid, zdata, rstride=rstride,
                          cstride=cstride, *args, **kwargs)
        #ax.contour(xgrid, ygrid, zdata, rstride=rstride, cstride=cstride,
        #extend3d=True, *args, **kwargs)
    elif mode == 'surface' :
        ax.plot_surface(xgrid, ygrid, zdata, rstride=rstride, cstride=cstride,
                        linewidth=.1, *args, **kwargs)
    else:
        raise NotImplementedError('mode=%r' % (mode,))
    if contour:
        import matplotlib.cm as cm
        xoffset = xgrid.min() - ((xgrid.max() - xgrid.min()) * .1)
        yoffset = ygrid.max() + ((ygrid.max() - ygrid.min()) * .1)
        zoffset = zdata.min() - ((zdata.max() - zdata.min()) * .1)
        cmap = kwargs.get('cmap', cm.coolwarm)
        ax.contour(xgrid, ygrid, zdata, zdir='x', offset=xoffset, cmap=cmap)
        ax.contour(xgrid, ygrid, zdata, zdir='y', offset=yoffset, cmap=cmap)
        ax.contour(xgrid, ygrid, zdata, zdir='z', offset=zoffset, cmap=cmap)
        #ax.plot_trisurf(xgrid.flatten(), ygrid.flatten(), zdata.flatten(), *args, **kwargs)
    if title is not None:
        ax.set_title(title, **titlekw)
    if xlabel is not None:
        ax.set_xlabel(xlabel, **xlabelkw)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **ylabelkw)
    if zlabel is not None:
        ax.set_zlabel(zlabel, **zlabelkw)
    use_darkbackground = dark
    if use_darkbackground is None:
        use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background()
    return ax
#L_____

if __name__ == '__main__':
    """
    CommandLine:
        python -m plottool.draw_func2
        python -m plottool.draw_func2 --allexamples
        python -m plottool.draw_func2 --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
