""" Lots of functions for drawing and plotting visiony things """
# TODO: New naming scheme
# viz_<func_name> will clear everything. The current axes and fig: clf, cla.  # Will add annotations
# interact_<func_name> will clear everything and start user interactions.
# show_<func_name> will always clear the current axes, but not fig: cla # Might # add annotates?
# plot_<func_name> will not clear the axes or figure. More useful for graphs
# draw_<func_name> same as plot for now. More useful for images
from __future__ import absolute_import, division, print_function
import os
import sys
import utool
# Matplotlib
import matplotlib as mpl
if not sys.platform.startswith('win32') and not sys.platform.startswith('darwin') and os.environ.get('DISPLAY', None) is None:
    # Write to files if we cannot display
    TARGET_BACKEND = 'PDF'
else:
    TARGET_BACKEND = 'Qt4Agg'
mpl.use(TARGET_BACKEND)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Python
import colorsys
import pylab
import sys
import warnings
#from itertools import izip
# Qt
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
# Scientific
import numpy as np
import scipy.stats
import cv2
# VTool
import vtool.patch as ptool
import vtool.image as gtool
# Drawtool
from . import mpl_keypoint as mpl_kp
from .custom_figure import *     # NOQA  # TODO: FIXME THIS FILE NEEDS TO BE PARTITIONED
from .custom_constants import *  # NOQA  # TODO: FIXME THIS FILE NEEDS TO BE PARTITIONED
from .fig_presenter import *     # NOQA  # TODO: FIXME THIS FILE NEEDS TO BE PARTITIONED
from . import color_funcs as color_fns  # NOQA

lighten_rgb = color_fns.lighten_rgb
to_base255 = color_fns.to_base255


(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[df2]', DEBUG=False)

#================
# GLOBALS
#================

TMP_mevent = None
plotWidget = None


def get_pnum_func(nRows, nCols, base=0):
    offst = 0 if base == 1 else 1
    def pnum_(px):
        return (nRows, nCols, px + offst)
    return pnum_


def kwargs_fnum(kwargs):
    fnum = kwargs.get('fnum', None)
    if fnum is None:
        fnum = next_fnum()
        kwargs['fnum'] = fnum
    return fnum


def next_fnum(new_base=None):
    global base_fnum
    if new_base is not None:
        base_fnum = new_base
    base_fnum += 1
    return base_fnum


def execstr_global():
    execstr = ['global' + key for key in globals().keys()]
    return execstr


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


def _axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    if ax is None:
        ax = gca()
    'gets geometry of a subplot'
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


def draw_border(ax, color=GREEN, lw=2, offset=None):
    'draws rectangle border around a subplot'
    xy, width, height = _axis_xy_width_height(ax, -.7, -.2, 1, .4)
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


# TODO SEPARTE THIS INTO DRAW BBOX AND DRAW_ANNOTATION
def draw_annotation(annotation, label=None, bbox_color=(1, 0, 0),
             lbl_bgcolor=(0, 0, 0), lbl_txtcolor=(1, 1, 1),
             draw_arrow=True, theta=0, ax=None):
    if ax is None:
        ax = gca()
    (rx, ry, rw, rh) = annotation
    # Transformations are specified in backwards order.
    trans_annotation = mpl.transforms.Affine2D()
    trans_annotation.scale(rw, rh)
    trans_annotation.rotate(theta)
    trans_annotation.translate(rx + rw / 2, ry + rh / 2)
    t_end = trans_annotation + ax.transData
    bbox = mpl.patches.Rectangle((-.5, -.5), 1, 1, lw=2, transform=t_end)
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
    if label is not None:
        ax_absolute_text(rx, ry, label, ax=ax,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=lbl_txtcolor,
                         backgroundcolor=lbl_bgcolor)


def get_good_logyscale_kwargs(y_data, adaptive_knee_scaling=False):
    # Attempts to detect knee points by looking for
    # log derivatives way past the normal standard deviations
    # The input data is assumed to be sorted and y_data
    basey = 10
    nStdDevs_thresh = 10
    # Take the log of the data
    logy = np.log(y_data)
    logy[np.isnan(logy)] = 0
    logy[np.isinf(logy)] = 0
    # Find the derivative of data
    dy = np.diff(logy)
    dy_sortx = dy.argsort()
    # Get mean and standard deviation
    dy_stats = utool.mystats(dy)
    dy_sorted = dy[dy_sortx]
    # Find the number of standard deveations past the mean each datapoint is
    nStdDevs = np.abs(dy_sorted - dy_stats['mean']) / dy_stats['std']
    # Mark any above a threshold as knee points
    knee_indexes = np.where(nStdDevs > nStdDevs_thresh)[0]
    knee_mag = nStdDevs[knee_indexes]
    knee_points = dy_sortx[knee_indexes]
    print('[df2] knee_points = %r' % (knee_points,))
    # Check to see that we have found a knee
    if len(knee_points) > 0 and adaptive_knee_scaling:
        # Use linear scaling up the the knee points and
        # scale it by the magnitude of the knee
        kneex = knee_points.argmin()
        linthreshx = knee_points[kneex] + 1
        linthreshy = y_data[linthreshx] * basey
        linscaley = min(2, max(1, (knee_mag[kneex] / (basey * 2))))
    else:
        linthreshx = 1E2
        linthreshy = 1E2
        linscaley = 1
    logscale_kwargs = {
        'basey': basey,
        'nonposx': 'clip',
        'nonposy': 'clip',
        'linthreshy': linthreshy,
        'linthreshx': linthreshx,
        'linscalex': 1,
        'linscaley': linscaley,
    }
    #print(logscale_kwargs)
    return logscale_kwargs


def set_logyscale_from_data(y_data):
    logscale_kwargs = get_good_logyscale_kwargs(y_data)
    ax = gca()
    ax.set_yscale('symlog', **logscale_kwargs)


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


def plot2(x_data, y_data, marker='o', title_pref='', x_label='x', y_label='y', *args,
          **kwargs):
    do_plot = True
    ax = gca()
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
    if do_plot:
        ax.plot(x_data, y_data, marker, *args, **kwargs)

    min_ = min(x_data.min(), y_data.min())
    max_ = max(x_data.max(), y_data.max())
    # Equal aspect ratio
    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)
    ax.set_aspect('equal')
    ax.set_xlabel(x_label, fontproperties=FONTS.xlabel)
    ax.set_ylabel(y_label, fontproperties=FONTS.xlabel)
    set_title(title_pref + ' ' + x_label + ' vs ' + y_label, ax=None)


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
    for key in SAFE_POS.iterkeys():
        if key not in kwargs:
            kwargs[key] = SAFE_POS[key]
    adjust_subplots(**kwargs)


def adjust_subplots(left=0.02,  bottom=0.02,
                    right=0.98,     top=0.90,
                    wspace=0.1,   hspace=0.15):
    """
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2
    """
    #print('[df2] adjust_subplots(%r)' % locals())
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)


#=======================
# TEXT FUNCTIONS
# TODO: I have too many of these. Need to consolidate
#=======================


def upperleft_text(txt, alpha=.6):
    txtargs = dict(horizontalalignment='left',
                   verticalalignment='top',
                   backgroundcolor=(0, 0, 0, alpha),
                   color=ORANGE)
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
    xy, width, height = _axis_xy_width_height(ax)
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
        kwargs['fontproperties'] = FONTS.relative
    if roffset is not None:
        xroff, yroff = roffset
        xy, width, height = _axis_xy_width_height(ax)
        x_ += xroff * width
        y_ += yroff * height

    ax.text(x_, y_, txt, **kwargs)


def fig_relative_text(x, y, txt, **kwargs):
    kwargs['horizontalalignment'] = 'center'
    kwargs['verticalalignment'] = 'center'
    fig = gcf()
    #xy, width, height = _axis_xy_width_height(ax)
    #x_, y_ = ((xy[0]+width)+x*width, (xy[1]+height)-y*height)
    fig.text(x, y, txt, **kwargs)


def draw_text(text_str, rgb_textFG=(0, 0, 0), rgb_textBG=(1, 1, 1)):
    ax = gca()
    xy, width, height = _axis_xy_width_height(ax)
    text_x = xy[0] + (width / 2)
    text_y = xy[1] + (height / 2)
    ax.text(text_x, text_y, text_str,
            horizontalalignment='center',
            verticalalignment='center',
            color=rgb_textFG,
            backgroundcolor=rgb_textBG)


def convert_keypress_event_mpl_to_qt4(mevent):
    global TMP_mevent
    TMP_mevent = mevent
    # Grab the key from the mpl.KeyPressEvent
    key = mevent.key
    print('[df2] convert event mpl -> qt4')
    print('[df2] key=%r' % key)
    # dicts modified from backend_qt4.py
    mpl2qtkey = {'control': Qt.Key_Control, 'shift': Qt.Key_Shift,
                 'alt': Qt.Key_Alt, 'super': Qt.Key_Meta,
                 'enter': Qt.Key_Return, 'left': Qt.Key_Left, 'up': Qt.Key_Up,
                 'right': Qt.Key_Right, 'down': Qt.Key_Down,
                 'escape': Qt.Key_Escape, 'f1': Qt.Key_F1, 'f2': Qt.Key_F2,
                 'f3': Qt.Key_F3, 'f4': Qt.Key_F4, 'f5': Qt.Key_F5,
                 'f6': Qt.Key_F6, 'f7': Qt.Key_F7, 'f8': Qt.Key_F8,
                 'f9': Qt.Key_F9, 'f10': Qt.Key_F10, 'f11': Qt.Key_F11,
                 'f12': Qt.Key_F12, 'home': Qt.Key_Home, 'end': Qt.Key_End,
                 'pageup': Qt.Key_PageUp, 'pagedown': Qt.Key_PageDown}
    # Reverse the control and super (aka cmd/apple) keys on OSX
    if sys.platform == 'darwin':
        mpl2qtkey.update({'super': Qt.Key_Control, 'control': Qt.Key_Meta, })

    # Try to reconstruct QtGui.KeyEvent
    type_ = QtCore.QEvent.Type(QtCore.QEvent.KeyPress)  # The type should always be KeyPress
    text = ''
    # Try to extract the original modifiers
    modifiers = QtCore.Qt.NoModifier  # initialize to no modifiers
    if key.find(u'ctrl+') >= 0:
        modifiers = modifiers | QtCore.Qt.ControlModifier
        key = key.replace(u'ctrl+', u'')
        print('[df2] has ctrl modifier')
        text += 'Ctrl+'
    if key.find(u'alt+') >= 0:
        modifiers = modifiers | QtCore.Qt.AltModifier
        key = key.replace(u'alt+', u'')
        print('[df2] has alt modifier')
        text += 'Alt+'
    if key.find(u'super+') >= 0:
        modifiers = modifiers | QtCore.Qt.MetaModifier
        key = key.replace(u'super+', u'')
        print('[df2] has super modifier')
        text += 'Super+'
    if key.isupper():
        modifiers = modifiers | QtCore.Qt.ShiftModifier
        print('[df2] has shift modifier')
        text += 'Shift+'
    # Try to extract the original key
    try:
        if key in mpl2qtkey:
            key_ = mpl2qtkey[key]
        else:
            key_ = ord(key.upper())  # Qt works with uppercase keys
            text += key.upper()
    except Exception as ex:
        print('[df2] ERROR key=%r' % key)
        print('[df2] ERROR %r' % ex)
        raise
    autorep = False  # default false
    count   = 1  # default 1
    text = QtCore.QString(text)  # The text is somewhat arbitrary
    # Create the QEvent
    print('----------------')
    print('[df2] Create event')
    print('[df2] type_ = %r' % type_)
    print('[df2] text = %r' % text)
    print('[df2] modifiers = %r' % modifiers)
    print('[df2] autorep = %r' % autorep)
    print('[df2] count = %r ' % count)
    print('----------------')
    qevent = QtGui.QKeyEvent(type_, key_, modifiers, text, autorep, count)
    return qevent


def test_build_qkeyevent():
    import draw_func2 as df2
    qtwin = df2.QT4_WINS[0]
    # This reconstructs an test mplevent
    canvas = df2.figure(1).canvas
    mevent = mpl.backend_bases.KeyEvent('key_press_event', canvas, u'ctrl+p', x=672, y=230.0)
    qevent = df2.convert_keypress_event_mpl_to_qt4(mevent)
    app = qtwin.backend.app
    app.sendEvent(qtwin.ui, mevent)
    #type_ = QtCore.QEvent.Type(QtCore.QEvent.KeyPress)  # The type should always be KeyPress
    #text = QtCore.QString('A')  # The text is somewhat arbitrary
    #modifiers = QtCore.Qt.NoModifier  # initialize to no modifiers
    #modifiers = modifiers | QtCore.Qt.ControlModifier
    #modifiers = modifiers | QtCore.Qt.AltModifier
    #key_ = ord('A')  # Qt works with uppercase keys
    #autorep = False  # default false
    #count   = 1  # default 1
    #qevent = QtGui.QKeyEvent(type_, key_, modifiers, text, autorep, count)
    return qevent


def plot_pdf(data, draw_support=True, scale_to=None, label=None, color=0,
             nYTicks=3):
    fig = gcf()
    ax = gca()
    data = np.array(data)
    if len(data) == 0:
        warnstr = '[df2] ! Warning: len(data) = 0. Cannot visualize pdf'
        warnings.warn(warnstr)
        draw_text(warnstr)
        return
    if len(data) == 1:
        warnstr = '[df2] ! Warning: len(data) = 1. Cannot visualize pdf'
        warnings.warn(warnstr)
        draw_text(warnstr)
        return
    bw_factor = .05
    if isinstance(color, (int, float)):
        colorx = color
        line_color = plt.get_cmap('gist_rainbow')(colorx)
    else:
        line_color = color

    # Estimate a pdf
    data_pdf = estimate_pdf(data, bw_factor)
    # Get probability of seen data
    prob_x = data_pdf(data)
    # Get probability of unseen data data
    x_data = np.linspace(0, data.max(), 500)
    y_data = data_pdf(x_data)
    # Scale if requested
    if scale_to is not None:
        scale_factor = scale_to / y_data.max()
        y_data *= scale_factor
        prob_x *= scale_factor
    #Plot the actual datas on near the bottom perterbed in Y
    if draw_support:
        pdfrange = prob_x.max() - prob_x.min()
        perb   = (np.random.randn(len(data))) * pdfrange / 30.
        preb_y_data = np.abs([pdfrange / 50. for _ in data] + perb)
        ax.plot(data, preb_y_data, 'o', color=line_color, figure=fig, alpha=.1)
    # Plot the pdf (unseen data)
    ax.plot(x_data, y_data, color=line_color, label=label)
    if nYTicks is not None:
        yticks = np.linspace(min(y_data), max(y_data), nYTicks)
        ax.set_yticks(yticks)


def estimate_pdf(data, bw_factor):
    try:
        data_pdf = scipy.stats.gaussian_kde(data, bw_factor)
        data_pdf.covariance_factor = bw_factor
    except Exception as ex:
        print('[df2] ! Exception while estimating kernel density')
        print('[df2] data=%r' % (data,))
        print('[df2] ex=%r' % (ex,))
        raise
    return data_pdf


def show_histogram(data, bins=None, **kwargs):
    print('[df2] show_histogram()')
    dmin = int(np.floor(data.min()))
    dmax = int(np.ceil(data.max()))
    if bins is None:
        bins = dmax - dmin
    fig = figure(**kwargs)
    ax  = gca()
    ax.hist(data, bins=bins, range=(dmin, dmax))
    #help(np.bincount)
    fig.show()


def show_signature(sig, **kwargs):
    fig = figure(**kwargs)
    plt.plot(sig)
    fig.show()


def plot_stems(x_data=None, y_data=None):
    if y_data is not None and x_data is None:
        x_data = np.arange(len(y_data))
        pass
    if len(x_data) != len(y_data):
        print('[df2] WARNING plot_stems(): len(x_data)!=len(y_data)')
    if len(x_data) == 0:
        print('[df2] WARNING plot_stems(): len(x_data)=len(y_data)=0')
    x_data_ = np.array(x_data)
    y_data_ = np.array(y_data)
    x_data_sort = x_data_[y_data_.argsort()[::-1]]
    y_data_sort = y_data_[y_data_.argsort()[::-1]]

    markerline, stemlines, baseline = pylab.stem(x_data_sort, y_data_sort, linefmt='-')
    pylab.setp(markerline, 'markerfacecolor', 'b')
    pylab.setp(baseline, 'linewidth', 0)
    ax = gca()
    ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
    ax.set_ylim(min(y_data) - 1, max(max(y_data), max(x_data)) + 1)


def plot_sift_signature(sift, title='', fnum=None, pnum=None):
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    plot_bars(sift, 16)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 256)
    space_xticks(9, 16)
    space_yticks(5, 64)
    set_title(title, ax=ax)
    dark_background(ax)
    return ax


def dark_background(ax=None, doubleit=False):
    bgcolor = BLACK * .9
    if ax is None:
        ax = gca()
    from mpl_toolkits.mplot3d import Axes3D
    if isinstance(ax, Axes3D):
        ax.set_axis_bgcolor(bgcolor)
        return
    xy, width, height = _axis_xy_width_height(ax)
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
    for ix in xrange(nColorSplits):
        xs = np.arange(nGroup) + (nGroup * ix)
        color = ori_colors[ix]
        x_dat = x_data[xs]
        y_dat = y_data[xs]
        ax.bar(x_dat, y_dat, width, color=color, edgecolor=np.array(color) * .8)


def phantom_legend_label(label, color, loc='upper right'):
    'adds a legend label without displaying an actor'
    pass
    #phantom_actor = plt.Circle((0, 0), 1, fc=color, prop=FONTS.legend, loc=loc)
    #plt.legend(phant_actor, label, framealpha=.2)
    #plt.legend(*zip(*legend_tups), framealpha=.2)
    #legend_tups = []
    #legend_tups.append((phantom_actor, label))


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


def legend(loc='upper right'):
    ax = gca()
    ax.legend(prop=FONTS.legend, loc=loc)


def plot_histpdf(data, label=None, draw_support=False, nbins=10):
    freq, _ = plot_hist(data, nbins=nbins)
    plot_pdf(data, draw_support=draw_support, scale_to=freq.max(), label=label)


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


def scores_to_color(score_list, cmap_='hot', logscale=False):
    printDBG('scores_to_color()')
    assert len(score_list.shape) == 1, 'score must be 1d'
    if logscale:
        score_list = np.log2(np.log2(score_list + 2) + 1)
    cmap = plt.get_cmap(cmap_)
    mins = score_list.min()
    rnge = score_list.max() - mins
    if rnge == 0:
        return [cmap(.5) for fx in xrange(len(score_list))]
    else:
        if logscale:
            score2_01 = lambda score: np.log2(1.1 + .9 * (float(score) - mins) / (rnge))
            score_list = np.array(score_list)
            #rank_multiplier = score_list.argsort() / len(score_list)
            #normscore = np.array(map(score2_01, score_list)) * rank_multiplier
            normscore = np.array(map(score2_01, score_list))
            colors =  map(cmap, normscore)
        else:
            score2_01 = lambda score: .1 + .9 * (float(score) - mins) / (rnge)
        colors    = [cmap(score2_01(score)) for score in score_list]
        return colors


def scores_to_cmap(scores, colors=None, cmap_='hot'):
    printDBG('scores_to_cmap()')
    if colors is None:
        colors = scores_to_color(scores, cmap_=cmap_)
    sorted_colors = [x for (y, x) in sorted(zip(scores, colors))]
    # Make a listed colormap and mappable object
    listed_cmap = mpl.colors.ListedColormap(sorted_colors)
    return listed_cmap


def ensure_divider(ax):
    """ Returns previously constructed divider or creates one """
    if not hasattr(ax, '_df2_divider'):
        divider = make_axes_locatable(ax)
        ax._df2_divider = divider
    return ax._df2_divider


def colorbar(scalars, colors):
    """ adds a color bar next to the axes """
    printDBG('colorbar()')
    # Parameters
    ax = gca()
    divider = ensure_divider(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    xy, width, height = _axis_xy_width_height(ax)
    #orientation = ['vertical', 'horizontal'][0]
    TICK_FONTSIZE = 8
    #
    listed_cmap = scores_to_cmap(scalars, colors)
    # Create scalar mappable with cmap
    sorted_scalars = sorted(scalars)
    sm = plt.cm.ScalarMappable(cmap=listed_cmap)
    sm.set_array(sorted_scalars)
    # Use mapable object to create the colorbar
    #COLORBAR_SHRINK = .42  # 1
    #COLORBAR_PAD = .01  # 1
    #COLORBAR_ASPECT = np.abs(20 * height / (width))  # 1
    #printDBG('[df] COLORBAR_ASPECT = %r' % COLORBAR_ASPECT)

    cb = plt.colorbar(sm, cax=cax)

    ## Add the colorbar to the correct label
    axis = cb.ax.yaxis  # if orientation == 'horizontal' else cb.ax.yaxis
    #position = 'bottom' if orientation == 'horizontal' else 'right'
    #axis.set_ticks_position(position)
    axis.set_ticks([0, .5, 1])
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.sca(ax)


def draw_lines2(kpts1, kpts2, fm=None, fs=None, kpts2_offset=(0, 0),
                color_list=None, scale_factor=1, lw=1.4, line_alpha=.35, **kwargs):
    printDBG('-------------')
    printDBG('draw_lines2()')
    printDBG(' * len(fm) = %r' % len(fm))
    printDBG(' * scale_factor = %r' % scale_factor)
    # input data
    if fm is None:  # assume kpts are in director correspondence
        assert kpts1.shape == kpts2.shape
    if len(fm) == 0:
        return
    ax = gca()
    woff, hoff = kpts2_offset
    # Draw line collection
    kpts1_m = kpts1[fm[:, 0]].T
    kpts2_m = kpts2[fm[:, 1]].T
    xxyy_iter = iter(zip(kpts1_m[0] * scale_factor,
                         kpts2_m[0] * scale_factor + woff,
                         kpts1_m[1] * scale_factor,
                         kpts2_m[1] * scale_factor + hoff))
    if color_list is None:
        if fs is None:  # Draw with solid color
            color_list    = [RED for fx in xrange(len(fm))]
        else:  # Draw with colors proportional to score difference
            color_list = scores_to_color(fs)
    segments  = [((x1, y1), (x2, y2)) for (x1, x2, y1, y2) in xxyy_iter]
    linewidth = [lw for fx in xrange(len(fm))]
    line_alpha = line_alpha
    line_group = LineCollection(segments, linewidth, color_list, alpha=line_alpha)
    #plt.colorbar(line_group, ax=ax)
    ax.add_collection(line_group)
    #figure(100)
    #plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)


def draw_kpts2(kpts, offset=(0, 0), scale_factor=1,
               ell=True, pts=False, rect=False, eig=False, ori=False,
               pts_size=2, ell_alpha=.6, ell_linewidth=1.5,
               ell_color=None, pts_color=ORANGE, color_list=None, **kwargs):
    if ell_color is None:
        ell_color = kwargs.get('color', BLUE)
    printDBG('-------------')
    printDBG('draw_kpts2():')
    #printDBG(' * kwargs.keys()=%r' % (kwargs.keys(),))
    printDBG(' * kpts.shape=%r:' % (kpts.shape,))
    printDBG(' * ell=%r pts=%r' % (ell, pts))
    printDBG(' * rect=%r eig=%r, ori=%r' % (rect, eig, ori))
    printDBG(' * scale_factor=%r' % (scale_factor,))
    printDBG(' * offset=%r' % (offset,))
    printDBG(' * drawing kpts.shape=%r' % (kpts.shape,))
    assert len(kpts) > 0, 'len(kpts) < 0'
    ax = gca()
    ell_alpha = ell_alpha
    if color_list is not None:
        ell_color = color_list
        pts_color = color_list
    #else:
        #pts_color = [pts_color for _ in xrange(len(kpts))]
    if ell_color == 'distinct':
        ell_color = distinct_colors(len(kpts))

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
        'pts_color': pts_color,
        'ell_linewidth': ell_linewidth,
    })

    mpl_kp.draw_keypoints(ax, kpts, **_kwargs)


def draw_keypoint_gradient_orientations(rchip, kp, sift=None, mode='vec', **kwargs):
    wpatch, wkp  = ptool.get_warped_patch(rchip, kp, gray=True)
    try:
        gradx, grady = ptool.patch_gradient(wpatch)
    except Exception as ex:
        print('!!!!!!!!!!!!')
        print('[df2!] Exception = ' + str(ex))
        print('---------')
        print('type(wpatch) = ' + str(type(wpatch)))
        print('repr(wpatch) = ' + str(repr(wpatch)))
        print('wpatch = ' + str(wpatch))
        raise
    if mode == 'vec':
        draw_vector_field(gradx, grady, **kwargs)
    elif mode == 'col':
        import plottool as dtool
        gmag = ptool.patch_mag(gradx, grady)
        gori = ptool.patch_ori(gradx, grady)
        gorimag = dtool.color_orimag(gori, gmag)
        imshow(gorimag, **kwargs)
    wkpts = np.array([wkp])
    sifts = np.array([sift])
    draw_kpts2(wkpts, sifts=sifts, ori=True)


@utool.indent_func('[df2.dkp]')
def draw_keypoint_patch(rchip, kp, sift=None, warped=False, patch_dict={}, **kwargs):
    #print('--------------------')
    printDBG('[df2] draw_keypoint_patch()')
    kpts = np.array([kp])
    if warped:
        patches, subkpts = ptool.get_warped_patches(rchip, kpts)
    else:
        patches, subkpts = ptool.get_unwarped_patches(rchip, kpts)
    #print('[df2] kpts[0]    = %r' % (kpts[0]))
    #print('[df2] subkpts[0] = %r' % (subkpts[0]))
    #print('[df2] patches[0].shape = %r' % (patches[0].shape,))
    patch = patches[0]
    subkpts_ = np.array(subkpts)
    patch_dict_ = {
        'sifts': np.array([sift]),
        'ell_color':  (0, 0, 1),
        'pts': True,
        'ori': True,
        'ell': True,
        'eig': False,
        'rect': True,
    }
    patch_dict_.update(patch_dict)
    # Draw patch with keypoint overlay
    fig, ax = imshow(patch, **kwargs)
    draw_kpts2(subkpts_, **patch_dict_)
    return ax


# ---- CHIP DISPLAY COMMANDS ----
def imshow(img, fnum=None, title=None, figtitle=None, pnum=None,
           interpolation='nearest', cmap=None, heatmap=False,
           data_colorbar=False, darken=DARKEN, **kwargs):
    'other interpolations = nearest, bicubic, bilinear'
    printDBG('imshow()')
    #printDBG('[df2] ----- IMSHOW ------ ')
    #printDBG('[***df2.imshow] fnum=%r pnum=%r title=%r *** ' % (fnum, pnum, title))
    #printDBG('[***df2.imshow] img.shape = %r ' % (img.shape,))
    #printDBG('[***df2.imshow] img.stats = %r ' % (utool.common_stats(img),))
    fig = figure(fnum=fnum, pnum=pnum, title=title, figtitle=figtitle, **kwargs)
    ax = gca()
    if darken is not None:
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
                    printDBG('Drawing Float Color Image < 1')
                    imgBGR = np.array(imgBGR, dtype=np.float32)
                else:
                    printDBG('Drawing Float Color Image > 1')
                    imgBGR = np.array(imgBGR, dtype=np.uint8)
            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            ax.imshow(imgRGB, **plt_imshow_kwargs)
        elif len(img.shape) == 2:
            # img is in grayscale
            imgGRAY = img
            if cmap is None:
                cmap = plt.get_cmap('gray')
            if isinstance(cmap, (str, unicode)):
                cmap = plt.get_cmap(cmap)
            ax.imshow(imgGRAY, cmap=cmap, **plt_imshow_kwargs)
        else:
            raise Exception('unknown image format')
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
        colors = scores_to_color(scores, cmap)
        colorbar(scores, colors)
    return fig, ax


def draw_vector_field(gx, gy, fnum=None, pnum=None, title=None):
    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.quiver
    printDBG('[df2] draw_vector_vield()')
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
    np.tau = 2 * np.pi
    x_grid = np.arange(0, len(gx), 1)
    y_grid = np.arange(0, len(gy), 1)
    # Vector locations and directions
    X, Y = np.meshgrid(x_grid, y_grid)
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
    ax.invert_yaxis()
    ax.set_aspect('equal')
    if title is not None:
        set_title(title)


def show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=None, fs=None, title=None,
                    vert=None, fnum=None, pnum=None, heatmap=False,
                    draw_fmatch=True, **kwargs):
    """Draws two chips and the feature matches between them. feature matches
    kpts1 and kpts2 use the (x,y,a,c,d)
    """
    printDBG('[df2] draw_matches2() fnum=%r, pnum=%r' % (fnum, pnum))
    # get matching keypoints + offset
    (h1, w1) = rchip1.shape[0:2]  # get chip (h, w) dimensions
    (h2, w2) = rchip2.shape[0:2]
    # Stack the compared chips
    match_img, woff, hoff = stack_images(rchip1, rchip2, vert)
    xywh1 = (0, 0, w1, h1)
    xywh2 = (woff, hoff, w2, h2)
    # Show the stacked chips
    fig, ax = imshow(match_img, title=title, fnum=fnum, pnum=pnum, heatmap=heatmap)
    # Overlay feature match nnotations
    if draw_fmatch and kpts1 is not None and kpts2 is not None:
        plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs, **kwargs)
    return ax, xywh1, xywh2


# plot feature match
def plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs=None, lbl1=None, lbl2=None,
                fnum=None, pnum=None, rect=False, colorbar_=True,
                draw_border=False, **kwargs):
    """Draws the matching features. This is draw because it is an overlay
    xywh1 - location of rchip1 in the axes
    xywh2 - location or rchip2 in the axes
    """
    printDBG('[df2] plot_fmatch')
    if fm is None:
        assert kpts1.shape == kpts2.shape, 'shapes different or fm not none'
        fm = np.tile(np.arange(0, len(kpts1)), (2, 1)).T
    pts       = kwargs.get('draw_pts', False)
    ell       = kwargs.get('draw_ell', True)
    lines     = kwargs.get('draw_lines', True)
    ell_alpha = kwargs.get('ell_alpha', .4)
    nMatch = len(fm)
    #printDBG('[df2.draw_fnmatch] nMatch=%r' % nMatch)
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    offset2 = (x2, y2)
    # THIS IS NOT WHERE THIS CODE BELONGS
    if False:
        # Custom user label for chips 1 and 2
        if lbl1 is not None:
            absolute_lbl(x1 + w1, y1, lbl1)
        if lbl2 is not None:
            absolute_lbl(x2 + w2, y2, lbl2)
    # Plot the number of matches
    if kwargs.get('show_nMatches', False):
        upperleft_text('#match=%d' % nMatch)
    # Draw all keypoints in both chips as points
    if kwargs.get('all_kpts', False):
        all_args = dict(ell=False, pts=pts, pts_color=GREEN, pts_size=2,
                        ell_alpha=ell_alpha, rect=rect)
        all_args.update(kwargs)
        draw_kpts2(kpts1, **all_args)
        draw_kpts2(kpts2, offset=offset2, **all_args)
    if draw_border:
        draw_annotation(xywh1, bbox_color=BLACK, draw_arrow=False)
        draw_annotation(xywh2, bbox_color=BLACK, draw_arrow=False)

    # Draw Lines and Ellipses and Points oh my
    if nMatch > 0:
        colors = [kwargs['colors']] * nMatch if 'colors' in kwargs else distinct_colors(nMatch)
        if fs is not None:
            colors = scores_to_color(fs, 'hot')

        acols = add_alpha(colors)

        # Helper functions
        def _drawkpts(**_kwargs):
            _kwargs.update(kwargs)
            fxs1 = fm[:, 0]
            fxs2 = fm[:, 1]
            draw_kpts2(kpts1[fxs1], rect=rect, **_kwargs)
            draw_kpts2(kpts2[fxs2], offset=offset2, rect=rect, **_kwargs)

        def _drawlines(**_kwargs):
            _kwargs.update(kwargs)
            draw_lines2(kpts1, kpts2, fm, fs, kpts2_offset=offset2, **_kwargs)

        # User utool
        if ell:
            _drawkpts(pts=False, ell=True, color_list=colors)
        if pts:
            _drawkpts(pts_size=8, pts=True, ell=False, pts_color=BLACK)
            _drawkpts(pts_size=6, pts=True, ell=False, color_list=acols)
        if lines:
            _drawlines(color_list=colors)
    else:
        draw_boxedX(xywh2)
    if fs is not None and colorbar_ and 'colors' in vars() and colors is not None:
        colorbar(fs, colors)
    #legend()
    return None


def draw_boxedX(xywh=None, color=RED, lw=2, alpha=.5, theta=0):
    'draws a big red x. redx'
    ax = gca()
    if xywh is None:
        xy, w, h = _axis_xy_width_height(ax)
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
    line_group = LineCollection(segments, width_list, color_list, alpha=alpha,
                                transOffset=trans)
    ax.add_collection(line_group)


def color_orimag(gori, gmag):
    # Turn a 0 to 1 orienation map into hsv colors
    gori_01 = (gori - gori.min()) / (gori.max() - gori.min())
    cmap_ = plt.get_cmap('hsv')
    flat_rgb = np.array(cmap_(gori_01.flatten()), dtype=np.float32)
    rgb_ori_alpha = flat_rgb.reshape(np.hstack((gori.shape, [4])))
    rgb_ori = cv2.cvtColor(rgb_ori_alpha, cv2.COLOR_RGBA2RGB)
    hsv_ori = cv2.cvtColor(rgb_ori, cv2.COLOR_RGB2HSV)
    # Desaturate colors based on magnitude
    hsv_ori[:, :, 1] = (gmag / 255.0)
    hsv_ori[:, :, 2] = (gmag / 255.0)
    # Convert back to bgr
    bgr_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2RGB)
    return bgr_ori


def stack_image_list(img_list, **kwargs):
    if len(img_list) == 0:
        return None
    imgB = img_list[0]
    offset_list = []
    for count, img2 in enumerate(img_list):
        if count == 0:
            continue
        imgB, woff, hoff = stack_images(imgB, img2, **kwargs)
        offset_list.append((woff, hoff))
    return imgB


def stack_image_recurse(img_list1, img_list2=None, vert=True):
    if img_list2 is None:
        # Initialization and error checking
        if len(img_list1) == 0:
            return None
        if len(img_list1) == 1:
            return img_list1[0]
        return stack_image_recurse(img_list1[0::2], img_list1[1::2], vert=vert)
    if len(img_list1) == 1:
        # Left base case
        img1 = img_list1[0]
    else:
        # Left recurse
        img1 = stack_image_recurse(img_list1[0::2], img_list1[1::2], vert=not vert)
    if len(img_list2) == 1:
        # Right base case
        img2 = img_list2[0]
    else:
        # Right Recurse
        img2 = stack_image_recurse(img_list2[0::2], img_list2[1::2], vert=not vert)
    imgB, woff, hoff = stack_images(img1, img2, vert=vert)
    return imgB


def stack_images(img1, img2, vert=None):
    # TODO: move this to the same place I'm doing the color gradient
    nChannels = gtool.get_num_channels(img1)
    nChannels2 = gtool.get_num_channels(img2)
    assert nChannels == nChannels2
    (h1, w1) = img1.shape[0: 2]  # get chip dimensions
    (h2, w2) = img2.shape[0: 2]
    woff, hoff = 0, 0
    vert_wh  = max(w1, w2), h1 + h2
    horiz_wh = w1 + w2, max(h1, h2)
    if vert is None:
        # Display the orientation with the better (closer to 1) aspect ratio
        vert_ar  = max(vert_wh) / min(vert_wh)
        horiz_ar = max(horiz_wh) / min(horiz_wh)
        vert = vert_ar < horiz_ar
    if vert:
        wB, hB = vert_wh
        hoff = h1
    else:
        wB, hB = horiz_wh
        woff = w1
    # concatentate images
    dtype = img1.dtype
    assert img1.dtype == img2.dtype
    if nChannels == 3:
        imgB = np.zeros((hB, wB, 3), dtype)
        imgB[0:h1, 0:w1, :] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2), :] = img2
    elif nChannels == 1:
        imgB = np.zeros((hB, wB), dtype)
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2
    return imgB, woff, hoff


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
    for ix in xrange(num):
        rect = [left, bottom, width, height]
        rect_list.append(rect)
        left += width + xpad
    ax_list = [plt.axes(rect) for rect in rect_list]
    but_list = [mpl.widgets.Button(ax_, text) for ax_, text in izip(ax_list, text_list)]
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
