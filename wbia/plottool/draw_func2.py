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
import itertools as it
import utool as ut  # NOQA
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError as ex:
    ut.printex(
        ex,
        'try pip install mpl_toolkits.axes_grid1 or something.  idk yet',
        iswarning=False,
    )
    raise
# import colorsys
import pylab
import warnings
import numpy as np
from os.path import relpath

try:
    import cv2
except ImportError as ex:
    print('ERROR PLOTTOOL CANNOT IMPORT CV2')
    print(ex)
from wbia.plottool import mpl_keypoint as mpl_kp
from wbia.plottool import color_funcs as color_fns
from wbia.plottool import custom_constants
from wbia.plottool import custom_figure
from wbia.plottool import fig_presenter

DEBUG = False
(print, rrr, profile) = ut.inject2(__name__)


def is_texmode():
    return mpl.rcParams['text.usetex']


# Bring over moved functions that still have dependants elsewhere

TAU = np.pi * 2
distinct_colors = color_fns.distinct_colors
lighten_rgb = color_fns.lighten_rgb
to_base255 = color_fns.to_base255

DARKEN = ut.get_argval(
    '--darken', type_=float, default=(0.7 if ut.get_argflag('--darken') else None)
)

# print('DARKEN = %r' % (DARKEN,))

all_figures_bring_to_front = fig_presenter.all_figures_bring_to_front
all_figures_tile = fig_presenter.all_figures_tile
close_all_figures = fig_presenter.close_all_figures
close_figure = fig_presenter.close_figure
iup = fig_presenter.iup
iupdate = fig_presenter.iupdate
present = fig_presenter.present
reset = fig_presenter.reset
update = fig_presenter.update


ORANGE = custom_constants.ORANGE
RED = custom_constants.RED
GREEN = custom_constants.GREEN
BLUE = custom_constants.BLUE
YELLOW = custom_constants.YELLOW
BLACK = custom_constants.BLACK
WHITE = custom_constants.WHITE
GRAY = custom_constants.GRAY
LIGHTGRAY = custom_constants.LIGHTGRAY
DEEP_PINK = custom_constants.DEEP_PINK
PINK = custom_constants.PINK
FALSE_RED = custom_constants.FALSE_RED
TRUE_GREEN = custom_constants.TRUE_GREEN
TRUE_BLUE = custom_constants.TRUE_BLUE
DARK_GREEN = custom_constants.DARK_GREEN
DARK_BLUE = custom_constants.DARK_BLUE
DARK_RED = custom_constants.DARK_RED
DARK_ORANGE = custom_constants.DARK_ORANGE
DARK_YELLOW = custom_constants.DARK_YELLOW
PURPLE = custom_constants.PURPLE
LIGHT_BLUE = custom_constants.LIGHT_BLUE
UNKNOWN_PURP = custom_constants.UNKNOWN_PURP

TRUE = TRUE_BLUE
FALSE = FALSE_RED

figure = custom_figure.figure
gca = custom_figure.gca
gcf = custom_figure.gcf
get_fig = custom_figure.get_fig
save_figure = custom_figure.save_figure
set_figtitle = custom_figure.set_figtitle
set_title = custom_figure.set_title
set_xlabel = custom_figure.set_xlabel
set_xticks = custom_figure.set_xticks
set_ylabel = custom_figure.set_ylabel
set_yticks = custom_figure.set_yticks

VERBOSE = ut.get_argflag(('--verbose-df2', '--verb-pt'))

# ================
# GLOBALS
# ================

TMP_mevent = None
plotWidget = None


def show_was_requested():
    """
    returns True if --show is specified on the commandline or you are in
    IPython (and presumably want some sort of interaction
    """
    return not ut.get_argflag(('--noshow')) and (
        ut.get_argflag(('--show', '--save')) or ut.inIPython()
    )
    # return ut.show_was_requested()


class OffsetImage2(mpl.offsetbox.OffsetBox):
    """
    TODO: If this works reapply to mpl
    """

    def __init__(
        self,
        arr,
        zoom=1,
        cmap=None,
        norm=None,
        interpolation=None,
        origin=None,
        filternorm=1,
        filterrad=4.0,
        resample=False,
        dpi_cor=True,
        **kwargs
    ):

        mpl.offsetbox.OffsetBox.__init__(self)
        self._dpi_cor = dpi_cor

        self.image = mpl.offsetbox.BboxImage(
            bbox=self.get_window_extent,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
            origin=origin,
            filternorm=filternorm,
            filterrad=filterrad,
            resample=resample,
            **kwargs
        )

        self._children = [self.image]

        self.set_zoom(zoom)
        self.set_data(arr)

    def set_data(self, arr):
        self._data = np.asarray(arr)
        self.image.set_data(self._data)
        self.stale = True

    def get_data(self):
        return self._data

    def set_zoom(self, zoom):
        self._zoom = zoom
        self.stale = True

    def get_zoom(self):
        return self._zoom

    #     def set_axes(self, axes):
    #         self.image.set_axes(axes)
    #         martist.Artist.set_axes(self, axes)

    #     def set_offset(self, xy):
    #         """
    #         set offset of the container.

    #         Accept : tuple of x,y coordinate in disokay units.
    #         """
    #         self._offset = xy

    #         self.offset_transform.clear()
    #         self.offset_transform.translate(xy[0], xy[1])

    def get_offset(self):
        """
        return offset of the container.
        """
        return self._offset

    def get_children(self):
        return [self.image]

    def get_window_extent(self, renderer):
        """
        get the bounding box in display space.
        """
        import matplotlib.transforms as mtransforms

        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset()
        return mtransforms.Bbox.from_bounds(ox - xd, oy - yd, w, h)

    def get_extent(self, renderer):

        # FIXME dpi_cor is never used
        if self._dpi_cor:  # True, do correction
            # conversion (px / pt)
            dpi_cor = renderer.points_to_pixels(1.0)
        else:
            dpi_cor = 1.0  # NOQA

        zoom = self.get_zoom()
        data = self.get_data()

        # Data width and height in pixels
        ny, nx = data.shape[:2]

        # w /= dpi_cor
        # h /= dpi_cor
        # import utool
        # if self.axes:
        # Hack, find right axes
        ax = self.figure.axes[0]
        ax.get_window_extent()
        # bbox = mpl.transforms.Bbox.union([ax.get_window_extent()])
        # xmin, xmax = ax.get_xlim()
        # ymin, ymax = ax.get_ylim()
        # https://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg25931.html
        fig = self.figure
        # dpi = fig.dpi  # (pt / in)
        fw_in, fh_in = fig.get_size_inches()
        # divider = make_axes_locatable(ax)

        # fig_ppi = dpi * dpi_cor
        # fw_px = fig_ppi * fw_in
        # fh_px = fig_ppi * fh_in
        # bbox.width

        # transforms data to figure coordinates
        # pt1 = ax.transData.transform_point([nx, ny])
        pt1 = ax.transData.transform_point([1, 20])
        pt2 = ax.transData.transform_point([0, 0])
        w, h = pt1 - pt2

        # zoom_factor = max(fw_px, )
        # print('fw_px = %r' % (fw_px,))
        # print('pos = %r' % (pos,))
        # w = h = .2 * fw_px * pos[2]
        # .1 * fig_dpi * fig_size[0] / data.shape[0]
        # print('zoom = %r' % (zoom,))

        w, h = w * zoom, h * zoom
        return w, h, 0, 0
        # return 30, 30, 0, 0

    def draw(self, renderer):
        """
        Draw the children
        """
        self.image.draw(renderer)
        # bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        self.stale = False


def overlay_icon(
    icon,
    coords=(0, 0),
    coord_type='axes',
    bbox_alignment=(0, 0),
    max_asize=None,
    max_dsize=None,
    as_artist=True,
):
    """
    Overlay a species icon

    References:
        http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        http://matplotlib.org/users/annotations_guide.html
        /usr/local/lib/python2.7/dist-packages/matplotlib/offsetbox.py

    Args:
        icon (ndarray or str): image icon data or path
        coords (tuple): (default = (0, 0))
        coord_type (str): (default = 'axes')
        bbox_alignment (tuple): (default = (0, 0))
        max_dsize (None): (default = None)

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-overlay_icon --show --icon zebra.png
        python -m wbia.plottool.draw_func2 --exec-overlay_icon --show --icon lena.png
        python -m wbia.plottool.draw_func2 --exec-overlay_icon --show --icon lena.png --artist

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> pt.plot2(np.arange(100), np.arange(100))
        >>> icon = ut.get_argval('--icon', type_=str, default='lena.png')
        >>> coords = (0, 0)
        >>> coord_type = 'axes'
        >>> bbox_alignment = (0, 0)
        >>> max_dsize = None  # (128, None)
        >>> max_asize = (60, 40)
        >>> as_artist = not ut.get_argflag('--noartist')
        >>> result = overlay_icon(icon, coords, coord_type, bbox_alignment,
        >>>                       max_asize, max_dsize, as_artist)
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    # from mpl_toolkits.axes_grid.anchored_artists import AnchoredAuxTransformBox
    import vtool as vt

    ax = gca()
    if isinstance(icon, six.string_types):
        # hack because icon is probably a url
        icon_url = icon
        icon = vt.imread(ut.grab_file_url(icon_url))
    if max_dsize is not None:
        icon = vt.resize_to_maxdims(icon, max_dsize)
    icon = vt.convert_colorspace(icon, 'RGB', 'BGR')

    # imagebox = OffsetImage2(icon, zoom=.3)
    if coord_type == 'axes':
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xy = [
            xlim[0] * (1 - coords[0]) + xlim[1] * (coords[0]),
            ylim[0] * (1 - coords[1]) + ylim[1] * (coords[1]),
        ]
    else:
        raise NotImplementedError('')

    # ab = AnchoredAuxTransformBox(ax.transData, loc=2)
    # ab.drawing_area.add_artist(imagebox)

    # *xycoords* and *textcoords* are strings that indicate the
    # coordinates of *xy* and *xytext*, and may be one of the
    # following values:

    # 'figure points' #'figure pixels' #'figure fraction' #'axes points'
    # 'axes pixels' #'axes fraction' #'data' #'offset points' #'polar'

    if as_artist:
        # Hack while I am trying to get constant size images working
        if ut.get_argval('--save'):
            # zoom = 1.0
            zoom = 1.0
        else:
            zoom = 0.5
        zoom = ut.get_argval('--overlay-zoom', default=zoom)
        if False:
            # TODO: figure out how to make axes fraction work
            imagebox = mpl.offsetbox.OffsetImage(icon)
            imagebox.set_width(1)
            imagebox.set_height(1)
            ab = mpl.offsetbox.AnnotationBbox(
                imagebox,
                xy,
                xybox=(0.0, 0.0),
                xycoords='data',
                boxcoords=('axes fraction', 'data'),
                # boxcoords="offset points",
                box_alignment=bbox_alignment,
                pad=0.0,
            )
        else:
            imagebox = mpl.offsetbox.OffsetImage(icon, zoom=zoom)
            ab = mpl.offsetbox.AnnotationBbox(
                imagebox,
                xy,
                xybox=(0.0, 0.0),
                xycoords='data',
                # xycoords='axes fraction',
                boxcoords='offset points',
                box_alignment=bbox_alignment,
                pad=0.0,
            )
        ax.add_artist(ab)
    else:
        img_size = vt.get_size(icon)
        print('img_size = %r' % (img_size,))
        if max_asize is not None:
            dsize, ratio = vt.resized_dims_and_ratio(img_size, max_asize)
            width, height = dsize
        else:
            width, height = img_size
        print('width, height= %r, %r' % (width, height,))
        x1 = xy[0] + width * bbox_alignment[0]
        y1 = xy[1] + height * bbox_alignment[1]
        x2 = xy[0] + width * (1 - bbox_alignment[0])
        y2 = xy[1] + height * (1 - bbox_alignment[1])

        ax = plt.gca()
        prev_aspect = ax.get_aspect()
        # FIXME: adjust aspect ratio of extent to match the axes
        print('icon.shape = %r' % (icon.shape,))
        print('prev_aspect = %r' % (prev_aspect,))
        extent = [x1, x2, y1, y2]
        print('extent = %r' % (extent,))
        ax.imshow(icon, extent=extent)
        print('current_aspect = %r' % (ax.get_aspect(),))
        ax.set_aspect(prev_aspect)
        print('current_aspect = %r' % (ax.get_aspect(),))
        # x - width // 2, x + width // 2,
        # y - height // 2, y + height // 2])


def update_figsize():
    """ updates figsize based on command line """
    figsize = ut.get_argval('--figsize', type_=list, default=None)
    if figsize is not None:
        # Enforce inches and DPI
        fig = gcf()
        figsize = [eval(term) if isinstance(term, str) else term for term in figsize]
        figw, figh = figsize[0], figsize[1]
        print('get_size_inches = %r' % (fig.get_size_inches(),))
        print('fig w,h (inches) = %r, %r' % (figw, figh))
        fig.set_size_inches(figw, figh)
        # print('get_size_inches = %r' % (fig.get_size_inches(),))


def udpate_adjust_subplots():
    """
    DEPRICATE

    updates adjust_subplots based on command line
    """
    adjust_list = ut.get_argval('--adjust', type_=list, default=None)
    if adjust_list is not None:
        # --adjust=[.02,.02,.05]
        keys = ['left', 'bottom', 'wspace', 'right', 'top', 'hspace']
        if len(adjust_list) == 1:
            # [all]
            vals = adjust_list * 3 + [1 - adjust_list[0]] * 2 + adjust_list
        elif len(adjust_list) == 3:
            # [left, bottom, wspace]
            vals = adjust_list + [1 - adjust_list[0], 1 - adjust_list[1], adjust_list[2]]
        elif len(adjust_list) == 4:
            # [left, bottom, wspace, hspace]
            vals = adjust_list[0:3] + [
                1 - adjust_list[0],
                1 - adjust_list[1],
                adjust_list[3],
            ]
        elif len(adjust_list) == 6:
            vals = adjust_list
        else:
            raise NotImplementedError(
                (
                    'vals must be len (1, 3, or 6) not %d, adjust_list=%r. '
                    'Expects keys=%r'
                )
                % (len(adjust_list), adjust_list, keys)
            )
        adjust_kw = dict(zip(keys, vals))
        print('**adjust_kw = %s' % (ut.repr2(adjust_kw),))
        adjust_subplots(**adjust_kw)


def render_figure_to_image(fig, **savekw):
    import io
    import cv2
    import wbia.plottool as pt

    # Pop save kwargs from kwargs
    # save_keys = ['dpi', 'figsize', 'saveax', 'verbose']
    # Write matplotlib axes to an image
    axes_extents = pt.extract_axes_extents(fig)
    # assert len(axes_extents) == 1, 'more than one axes'
    # if len(axes_extents) == 1:
    #     extent = axes_extents[0]
    # else:
    extent = mpl.transforms.Bbox.union(axes_extents)
    with io.BytesIO() as stream:
        # This call takes 23% - 15% of the time depending on settings
        fig.savefig(stream, bbox_inches=extent, **savekw)
        stream.seek(0)
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, 1)
    return image


class RenderingContext(object):
    def __init__(self, **savekw):
        self.image = None
        self.fig = None
        self.was_interactive = None
        self.savekw = savekw

    def __enter__(self):
        import wbia.plottool as pt

        tmp_fnum = -1
        import matplotlib as mpl

        self.fig = pt.figure(fnum=tmp_fnum)
        self.was_interactive = mpl.is_interactive()
        if self.was_interactive:
            mpl.interactive(False)
        return self

    def __exit__(self, type_, value, trace):
        if trace is not None:
            # print('[util_time] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        # Ensure that this figure will not pop up
        import wbia.plottool as pt

        self.image = pt.render_figure_to_image(self.fig, **self.savekw)
        pt.plt.close(self.fig)
        if self.was_interactive:
            mpl.interactive(self.was_interactive)


def extract_axes_extents(fig, combine=False, pad=0.0):
    """

    CommandLine:
        python -m wbia.plottool.draw_func2 extract_axes_extents
        python -m wbia.plottool.draw_func2 extract_axes_extents --save foo.jpg

     Notes:
          contour does something weird to axes
          with contour:
            axes_extents = Bbox([[-0.839827203337, -0.00555555555556], [7.77743055556, 6.97227277762]])

          without contour
            axes_extents = Bbox([[0.0290607810781, -0.00555555555556], [7.77743055556, 5.88]])

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import matplotlib.gridspec as gridspec
        >>> import matplotlib.pyplot as plt
        >>> import six
        >>> if six.PY2:
        >>>     import pytest
        >>>     pytest.skip()
        >>> pt.qtensure()
        >>> fig = plt.figure()
        >>> gs = gridspec.GridSpec(17, 17)
        >>> specs = [
        >>>     gs[0:8,  0:8], gs[0:8,  8:16],
        >>>     gs[9:17, 0:8], gs[9:17, 8:16],
        >>> ]
        >>> rng = np.random.RandomState(0)
        >>> X = (rng.rand(100, 2) * [[8, 8]]) + [[6, -14]]
        >>> x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        >>> y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        >>> xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        >>> yynan = np.full(yy.shape, fill_value=np.nan)
        >>> xxnan = np.full(yy.shape, fill_value=np.nan)
        >>> cmap = plt.cm.RdYlBu
        >>> norm = plt.Normalize(vmin=0, vmax=1)
        >>> for count, spec in enumerate(specs):
        >>>     fig.add_subplot(spec)
        >>>     plt.plot(X.T[0], X.T[1], 'o', color='r', markeredgecolor='w')
        >>>     Z = rng.rand(*xx.shape)
        >>>     plt.contourf(xx, yy, Z, cmap=cmap, norm=norm, alpha=1.0)
        >>>     plt.title('full-nan decision point')
        >>>     plt.gca().set_aspect('equal')
        >>> gs = gridspec.GridSpec(1, 16)
        >>> subspec = gs[:, -1:]
        >>> cax = plt.subplot(subspec)
        >>> sm = plt.cm.ScalarMappable(cmap=cmap)
        >>> sm.set_array(np.linspace(0, 1))
        >>> plt.colorbar(sm, cax)
        >>> cax.set_ylabel('ColorBar')
        >>> fig.suptitle('SupTitle')
        >>> subkw = dict(left=.001, right=.9, top=.9, bottom=.05, hspace=.2, wspace=.1)
        >>> plt.subplots_adjust(**subkw)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    import wbia.plottool as pt

    # Make sure we draw the axes first so we can
    # extract positions from the text objects
    fig.canvas.draw()

    # Group axes that belong together
    atomic_axes = []
    seen_ = set([])
    for ax in fig.axes:
        div = pt.get_plotdat(ax, DF2_DIVIDER_KEY, None)
        if div is not None:
            df2_div_axes = pt.get_plotdat_dict(ax).get('df2_div_axes', [])
            seen_.add(ax)
            seen_.update(set(df2_div_axes))
            atomic_axes.append([ax] + df2_div_axes)
            # TODO: pad these a bit
        else:
            if ax not in seen_:
                atomic_axes.append([ax])
                seen_.add(ax)

    hack_axes_group_row = ut.get_argflag('--grouprows')
    if hack_axes_group_row:
        groupid_list = []
        for axs in atomic_axes:
            for ax in axs:
                groupid = ax.colNum
            groupid_list.append(groupid)

        groupxs = ut.group_indices(groupid_list)[1]
        new_groups = ut.lmap(ut.flatten, ut.apply_grouping(atomic_axes, groupxs))
        atomic_axes = new_groups
        # [[(ax.rowNum, ax.colNum) for ax in axs] for axs in atomic_axes]
        # save all rows of each column

    dpi_scale_trans_inv = fig.dpi_scale_trans.inverted()
    axes_bboxes_ = [axes_extent(axs, pad) for axs in atomic_axes]
    axes_extents_ = [extent.transformed(dpi_scale_trans_inv) for extent in axes_bboxes_]
    # axes_extents_ = axes_bboxes_
    if combine:
        if True:
            # Grab include extents of figure text as well
            # FIXME: This might break on OSX
            # http://stackoverflow.com/questions/22667224/bbox-backend
            renderer = fig.canvas.get_renderer()
            for mpl_text in fig.texts:
                bbox = mpl_text.get_window_extent(renderer=renderer)
                extent_ = bbox.expanded(1.0 + pad, 1.0 + pad)
                extent = extent_.transformed(dpi_scale_trans_inv)
                # extent = extent_
                axes_extents_.append(extent)
        axes_extents = mpl.transforms.Bbox.union(axes_extents_)
    else:
        axes_extents = axes_extents_
    # if True:
    #     axes_extents.x0 = 0
    #     # axes_extents.y1 = 0
    return axes_extents


def axes_extent(axs, pad=0.0):
    """
    Get the full extent of a group of axes, including axes labels, tick labels,
    and titles.
    """

    def axes_parts(ax):
        yield ax
        for label in ax.get_xticklabels():
            if label.get_text():
                yield label
        for label in ax.get_yticklabels():
            if label.get_text():
                yield label
        xlabel = ax.get_xaxis().get_label()
        ylabel = ax.get_yaxis().get_label()
        for label in (xlabel, ylabel, ax.title):
            if label.get_text():
                yield label

    # def axes_parts2(ax):
    #     yield ('ax', ax)
    #     for c, label in enumerate(ax.get_xticklabels()):
    #         if label.get_text():
    #             yield ('xtick{}'.format(c), label)
    #     for label in ax.get_yticklabels():
    #         if label.get_text():
    #             yield ('ytick{}'.format(c), label)
    #     xlabel = ax.get_xaxis().get_label()
    #     ylabel = ax.get_yaxis().get_label()
    #     for key, label in (('xlabel', xlabel), ('ylabel', ylabel),
    #                   ('title', ax.title)):
    #         if label.get_text():
    #             yield (key, label)
    # yield from ax.lines
    # yield from ax.patches
    items = it.chain.from_iterable(axes_parts(ax) for ax in axs)
    extents = [item.get_window_extent() for item in items]
    # mpl.transforms.Affine2D().scale(1.1)
    extent = mpl.transforms.Bbox.union(extents)
    extent = extent.expanded(1.0 + pad, 1.0 + pad)
    return extent


def save_parts(fig, fpath, grouped_axes=None, dpi=None):
    """
    FIXME: this works in mpl 2.0.0, but not 2.0.2

    Args:
        fig (?):
        fpath (str):  file path string
        dpi (None): (default = None)

    Returns:
        list: subpaths

    CommandLine:
        python -m wbia.plottool.draw_func2 save_parts

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import matplotlib as mpl
        >>> import matplotlib.pyplot as plt
        >>> def testimg(fname):
        >>>     return plt.imread(mpl.cbook.get_sample_data(fname))
        >>> fnames = ['grace_hopper.png', 'ada.png'] * 4
        >>> fig = plt.figure(1)
        >>> for c, fname in enumerate(fnames, start=1):
        >>>     ax = fig.add_subplot(3, 4, c)
        >>>     ax.imshow(testimg(fname))
        >>>     ax.set_title(fname[0:3] + str(c))
        >>>     ax.set_xticks([])
        >>>     ax.set_yticks([])
        >>> ax = fig.add_subplot(3, 1, 3)
        >>> ax.plot(np.sin(np.linspace(0, np.pi * 2)))
        >>> ax.set_xlabel('xlabel')
        >>> ax.set_ylabel('ylabel')
        >>> ax.set_title('title')
        >>> fpath = 'test_save_parts.png'
        >>> adjust_subplots(fig=fig, wspace=.3, hspace=.3, top=.9)
        >>> subpaths = save_parts(fig, fpath, dpi=300)
        >>> fig.savefig(fpath)
        >>> ut.startfile(subpaths[0])
        >>> ut.startfile(fpath)
    """
    if dpi:
        # Need to set figure dpi before we draw
        fig.dpi = dpi
    # We need to draw the figure before calling get_window_extent
    # (or we can figure out how to set the renderer object)
    # if getattr(fig.canvas, 'renderer', None) is None:
    fig.canvas.draw()

    # Group axes that belong together
    if grouped_axes is None:
        grouped_axes = []
        for ax in fig.axes:
            grouped_axes.append([ax])

    subpaths = []
    _iter = enumerate(grouped_axes, start=0)
    _iter = ut.ProgIter(list(_iter), label='save subfig')
    for count, axs in _iter:
        subpath = ut.augpath(fpath, chr(count + 65))
        extent = axes_extent(axs).transformed(fig.dpi_scale_trans.inverted())
        savekw = {}
        savekw['transparent'] = ut.get_argflag('--alpha')
        if dpi is not None:
            savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        fig.savefig(subpath, bbox_inches=extent, **savekw)
        subpaths.append(subpath)
    return subpaths


def quit_if_noshow():
    import utool as ut

    saverequest = ut.get_argval('--save', default=None)
    if not (saverequest or ut.get_argflag(('--show', '--save')) or ut.inIPython()):
        raise ut.ExitTestException('This should be caught gracefully by ut.run_test')


def show_if_requested(N=1):
    """
    Used at the end of tests. Handles command line arguments for saving figures

    Referencse:
        http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib

    """
    if ut.NOT_QUIET:
        print('[pt] ' + str(ut.get_caller_name(range(3))) + ' show_if_requested()')

    # Process figures adjustments from command line before a show or a save

    # udpate_adjust_subplots()
    adjust_subplots(use_argv=True)

    update_figsize()

    dpi = ut.get_argval('--dpi', type_=int, default=custom_constants.DPI)
    SAVE_PARTS = ut.get_argflag('--saveparts')

    fpath_ = ut.get_argval('--save', type_=str, default=None)
    if fpath_ is None:
        fpath_ = ut.get_argval('--saveparts', type_=str, default=None)
        SAVE_PARTS = True

    if fpath_ is not None:
        from os.path import expanduser

        fpath_ = expanduser(fpath_)
        print('Figure save was requested')
        arg_dict = ut.get_arg_dict(
            prefix_list=['--', '-'], type_hints={'t': list, 'a': list}
        )
        # import sys
        from os.path import basename, splitext, join, dirname
        import wbia.plottool as pt
        import vtool as vt

        # HACK
        arg_dict = {
            key: (val[0] if len(val) == 1 else '[' + ']['.join(val) + ']')
            if isinstance(val, list)
            else val
            for key, val in arg_dict.items()
        }
        fpath_ = fpath_.format(**arg_dict)
        fpath_ = ut.remove_chars(fpath_, ' \'"')
        dpath, gotdpath = ut.get_argval(
            '--dpath', type_=str, default='.', return_specified=True
        )

        fpath = join(dpath, fpath_)
        if not gotdpath:
            dpath = dirname(fpath_)
        print('dpath = %r' % (dpath,))

        fig = pt.gcf()
        fig.dpi = dpi

        fpath_strict = ut.truepath(fpath)
        CLIP_WHITE = ut.get_argflag('--clipwhite')

        if SAVE_PARTS:
            # TODO: call save_parts instead, but we still need to do the
            # special grouping.

            # Group axes that belong together
            atomic_axes = []
            seen_ = set([])
            for ax in fig.axes:
                div = pt.get_plotdat(ax, DF2_DIVIDER_KEY, None)
                if div is not None:
                    df2_div_axes = pt.get_plotdat_dict(ax).get('df2_div_axes', [])
                    seen_.add(ax)
                    seen_.update(set(df2_div_axes))
                    atomic_axes.append([ax] + df2_div_axes)
                    # TODO: pad these a bit
                else:
                    if ax not in seen_:
                        atomic_axes.append([ax])
                        seen_.add(ax)

            hack_axes_group_row = ut.get_argflag('--grouprows')
            if hack_axes_group_row:
                groupid_list = []
                for axs in atomic_axes:
                    for ax in axs:
                        groupid = ax.colNum
                    groupid_list.append(groupid)

                groups = ut.group_items(atomic_axes, groupid_list)
                new_groups = ut.emap(ut.flatten, groups.values())
                atomic_axes = new_groups
                # [[(ax.rowNum, ax.colNum) for ax in axs] for axs in atomic_axes]
                # save all rows of each column

            subpath_list = save_parts(
                fig=fig, fpath=fpath_strict, grouped_axes=atomic_axes, dpi=dpi
            )
            absfpath_ = subpath_list[-1]
            fpath_list = [relpath(_, dpath) for _ in subpath_list]

            if CLIP_WHITE:
                for subpath in subpath_list:
                    # remove white borders
                    pass
                    vt.clipwhite_ondisk(subpath, subpath)
        else:
            savekw = {}
            # savekw['transparent'] = fpath.endswith('.png') and not noalpha
            savekw['transparent'] = ut.get_argflag('--alpha')
            savekw['dpi'] = dpi
            savekw['edgecolor'] = 'none'
            savekw['bbox_inches'] = extract_axes_extents(
                fig, combine=True
            )  # replaces need for clipwhite
            absfpath_ = ut.truepath(fpath)
            fig.savefig(absfpath_, **savekw)

            if CLIP_WHITE:
                # remove white borders
                fpath_in = fpath_out = absfpath_
                vt.clipwhite_ondisk(fpath_in, fpath_out)
                # img = vt.imread(absfpath_)
                # thresh = 128
                # fillval = [255, 255, 255]
                # cropped_img = vt.crop_out_imgfill(img, fillval=fillval, thresh=thresh)
                # print('img.shape = %r' % (img.shape,))
                # print('cropped_img.shape = %r' % (cropped_img.shape,))
                # vt.imwrite(absfpath_, cropped_img)
            # if dpath is not None:
            #    fpath_ = ut.unixjoin(dpath, basename(absfpath_))
            # else:
            #    fpath_ = fpath
            fpath_list = [fpath_]

        # Print out latex info
        default_caption = '\n% ---\n' + basename(fpath).replace('_', ' ') + '\n% ---\n'
        default_label = splitext(basename(fpath))[0]  # [0].replace('_', '')
        caption_list = ut.get_argval('--caption', type_=str, default=default_caption)
        if isinstance(caption_list, six.string_types):
            caption_str = caption_list
        else:
            caption_str = ' '.join(caption_list)
        # caption_str = ut.get_argval('--caption', type_=str,
        # default=basename(fpath).replace('_', ' '))
        label_str = ut.get_argval('--label', type_=str, default=default_label)
        width_str = ut.get_argval('--width', type_=str, default='\\textwidth')
        width_str = ut.get_argval('--width', type_=str, default='\\textwidth')
        print('width_str = %r' % (width_str,))
        height_str = ut.get_argval('--height', type_=str, default=None)
        caplbl_str = label_str

        if False and ut.is_developer() and len(fpath_list) <= 4:
            if len(fpath_list) == 1:
                latex_block = (
                    '\\ImageCommand{'
                    + ''.join(fpath_list)
                    + '}{'
                    + width_str
                    + '}{\n'
                    + caption_str
                    + '\n}{'
                    + label_str
                    + '}'
                )
            else:
                width_str = '1'
                latex_block = (
                    '\\MultiImageCommandII'
                    + '{'
                    + label_str
                    + '}'
                    + '{'
                    + width_str
                    + '}'
                    + '{'
                    + caplbl_str
                    + '}'
                    + '{\n'
                    + caption_str
                    + '\n}'
                    '{' + '}{'.join(fpath_list) + '}'
                )
            # HACK
        else:
            RESHAPE = ut.get_argval('--reshape', type_=tuple, default=None)
            if RESHAPE:

                def list_reshape(list_, new_shape):
                    for dim in reversed(new_shape):
                        list_ = list(map(list, zip(*[list_[i::dim] for i in range(dim)])))
                    return list_

                newshape = (2,)
                unflat_fpath_list = ut.list_reshape(fpath_list, newshape, trail=True)
                fpath_list = ut.flatten(ut.list_transpose(unflat_fpath_list))

            caption_str = '\\caplbl{' + caplbl_str + '}' + caption_str
            figure_str = ut.util_latex.get_latex_figure_str(
                fpath_list,
                label_str=label_str,
                caption_str=caption_str,
                width_str=width_str,
                height_str=height_str,
            )
            # import sys
            # print(sys.argv)
            latex_block = figure_str
            latex_block = ut.latex_newcommand(label_str, latex_block)
        # latex_block = ut.codeblock(
        #    r'''
        #    \newcommand{\%s}{
        #    %s
        #    }
        #    '''
        # ) % (label_str, latex_block,)
        try:
            import os
            import psutil
            import pipes

            # import shlex
            # TODO: separate into get_process_cmdline_str
            # TODO: replace home with ~
            proc = psutil.Process(pid=os.getpid())
            home = os.path.expanduser('~')
            cmdline_str = ' '.join(
                [pipes.quote(_).replace(home, '~') for _ in proc.cmdline()]
            )
            latex_block = (
                ut.codeblock(
                    r"""
                \begin{comment}
                %s
                \end{comment}
                """
                )
                % (cmdline_str,)
                + '\n'
                + latex_block
            )
        except OSError:
            pass

        # latex_indent = ' ' * (4 * 2)
        latex_indent = ' ' * (0)

        latex_block_ = ut.indent(latex_block, latex_indent)
        ut.print_code(latex_block_, 'latex')

        if 'append' in arg_dict:
            append_fpath = arg_dict['append']
            ut.write_to(append_fpath, '\n\n' + latex_block_, mode='a')

        if ut.get_argflag(('--diskshow', '--ds')):
            # show what we wrote
            ut.startfile(absfpath_)

        # Hack write the corresponding logfile next to the output
        log_fpath = ut.get_current_log_fpath()
        if ut.get_argflag('--savelog'):
            if log_fpath is not None:
                ut.copy(log_fpath, splitext(absfpath_)[0] + '.txt')
            else:
                print('Cannot copy log file because none exists')
    if ut.inIPython():
        import wbia.plottool as pt

        pt.iup()
    # elif ut.get_argflag('--cmd'):
    #     import wbia.plottool as pt
    #     pt.draw()
    #     ut.embed(N=N)
    elif ut.get_argflag('--cmd'):
        # cmd must handle show I think
        pass
    elif ut.get_argflag('--show'):
        if ut.get_argflag('--tile'):
            if ut.get_computer_name().lower() in ['hyrule']:
                fig_presenter.all_figures_tile(percent_w=0.5, monitor_num=0)
            else:
                fig_presenter.all_figures_tile()
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
        python -m wbia.plottool.draw_func2 --exec-distinct_markers --show
        python -m wbia.plottool.draw_func2 --exec-distinct_markers --mstyle=star --show
        python -m wbia.plottool.draw_func2 --exec-distinct_markers --mstyle=polygon --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> style = ut.get_argval('--mstyle', type_=str, default='astrisk')
        >>> marker_list = distinct_markers(10, style)
        >>> x_data = np.arange(0, 3)
        >>> for count, (marker) in enumerate(marker_list):
        >>>     pt.plot(x_data, [count] * len(x_data), marker=marker, markersize=10, linestyle='', label=str(marker))
        >>> pt.legend()
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    num_sides = 3
    style_num = {'astrisk': 2, 'star': 1, 'polygon': 0, 'circle': 3}[style]
    if total is None:
        total = num
    total_degrees = 360 / num_sides
    marker_list = [
        (num_sides, style_num, total_degrees * (count + offset) / total)
        for count in range(num)
    ]
    return marker_list


def get_all_markers():
    r"""
    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-get_all_markers --show

    References:
        http://matplotlib.org/1.3.1/examples/pylab_examples/line_styles.html
        http://matplotlib.org/api/markers_api.html#matplotlib.markers.MarkerStyle.markers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> marker_dict = get_all_markers()
        >>> x_data = np.arange(0, 3)
        >>> for count, (marker, name) in enumerate(marker_dict.items()):
        >>>     pt.plot(x_data, [count] * len(x_data), marker=marker, linestyle='', label=name)
        >>> pt.legend()
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
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
        # None: u'nothing',
        # u'None': u'nothing',
        # u' ': u'nothing',
        # u'': u'nothing',
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
    # marker_list = marker_dict.keys()
    # marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*',
    #               'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'TICKLEFT', 'TICKRIGHT', 'TICKUP',
    #               'TICKDOWN', 'CARETLEFT', 'CARETRIGHT', 'CARETUP', 'CARETDOWN']
    return marker_dict


def get_pnum_func(nRows=1, nCols=1, base=0):
    assert base in [0, 1], 'use base 0'
    offst = 0 if base == 1 else 1

    def pnum_(px):
        return (nRows, nCols, px + offst)

    return pnum_


def pnum_generator(nRows=1, nCols=1, base=0, nSubplots=None, start=0):
    r"""
    Args:
        nRows (int): (default = 1)
        nCols (int): (default = 1)
        base (int): (default = 0)
        nSubplots (None): (default = None)

    Yields:
        tuple : pnum

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-pnum_generator --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> nRows = 3
        >>> nCols = 2
        >>> base = 0
        >>> pnum_ = pnum_generator(nRows, nCols, base)
        >>> result = ut.repr2(list(pnum_), nl=1, nobr=True)
        >>> print(result)
        (3, 2, 1),
        (3, 2, 2),
        (3, 2, 3),
        (3, 2, 4),
        (3, 2, 5),
        (3, 2, 6),

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> nRows = 3
        >>> nCols = 2
        >>> pnum_ = pnum_generator(nRows, nCols, start=3)
        >>> result = ut.repr2(list(pnum_), nl=1, nobr=True)
        >>> print(result)
        (3, 2, 4),
        (3, 2, 5),
        (3, 2, 6),
    """
    pnum_func = get_pnum_func(nRows, nCols, base)
    total_plots = nRows * nCols
    # TODO: have the last pnums fill in the whole figure
    # when there are less subplots than rows * cols
    # if nSubplots is not None:
    #    if nSubplots < total_plots:
    #        pass
    for px in range(start, total_plots):
        yield pnum_func(px)


def make_pnum_nextgen(nRows=None, nCols=None, base=0, nSubplots=None, start=0):
    r"""
    Args:
        nRows (None): (default = None)
        nCols (None): (default = None)
        base (int): (default = 0)
        nSubplots (None): (default = None)
        start (int): (default = 0)

    Returns:
        iterator: pnum_next

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-make_pnum_nextgen --show

    GridParams:
        >>> param_grid = dict(
        >>>     nRows=[None, 3],
        >>>     nCols=[None, 3],
        >>>     nSubplots=[None, 9],
        >>> )
        >>> combos = ut.all_dict_combinations(param_grid)

    GridExample:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> base, start = 0, 0
        >>> pnum_next = make_pnum_nextgen(nRows, nCols, base, nSubplots, start)
        >>> pnum_list = list( (pnum_next() for _ in it.count()) )
        >>> print((nRows, nCols, nSubplots))
        >>> result = ('pnum_list = %s' % (ut.repr2(pnum_list),))
        >>> print(result)
    """
    import functools

    nRows, nCols = get_num_rc(nSubplots, nRows, nCols)

    pnum_gen = pnum_generator(
        nRows=nRows, nCols=nCols, base=base, nSubplots=nSubplots, start=start
    )
    pnum_next = functools.partial(six.next, pnum_gen)
    return pnum_next


def get_num_rc(nSubplots=None, nRows=None, nCols=None):
    r"""
    Gets a constrained row column plot grid

    Args:
        nSubplots (None): (default = None)
        nRows (None): (default = None)
        nCols (None): (default = None)

    Returns:
        tuple: (nRows, nCols)

    CommandLine:
        python -m wbia.plottool.draw_func2 get_num_rc

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> cases = [
        >>>     dict(nRows=None, nCols=None, nSubplots=None),
        >>>     dict(nRows=2, nCols=None, nSubplots=5),
        >>>     dict(nRows=None, nCols=2, nSubplots=5),
        >>>     dict(nRows=None, nCols=None, nSubplots=5),
        >>> ]
        >>> for kw in cases:
        >>>     print('----')
        >>>     size = get_num_rc(**kw)
        >>>     if kw['nSubplots'] is not None:
        >>>         assert size[0] * size[1] >= kw['nSubplots']
        >>>     print('**kw = %s' % (ut.repr2(kw),))
        >>>     print('size = %r' % (size,))
    """
    if nSubplots is None:
        if nRows is None:
            nRows = 1
        if nCols is None:
            nCols = 1
    else:
        if nRows is None and nCols is None:
            from wbia.plottool import plot_helpers

            nRows, nCols = plot_helpers.get_square_row_cols(nSubplots)
        elif nRows is not None:
            nCols = int(np.ceil(nSubplots / nRows))
        elif nCols is not None:
            nRows = int(np.ceil(nSubplots / nCols))
    return nRows, nCols


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


# def distinct_colors(N, brightness=.878, shuffle=True):
#    """
#    Args:
#        N (int): number of distinct colors
#        brightness (float): brightness of colors (maximum distinctiveness is .5) default is .878
#    Returns:
#        RGB_tuples
#    Example:
#        >>> from wbia.plottool.draw_func2 import *  # NOQA
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
    xy = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


def get_axis_bbox(ax=None, **kwargs):
    """
    # returns in figure coordinates?
    """
    xy, width, height = get_axis_xy_width_height(ax=ax, **kwargs)
    return (xy[0], xy[1], width, height)


def draw_border(ax, color=GREEN, lw=2, offset=None, adjust=True):
    """draws rectangle border around a subplot"""
    if adjust:
        xy, width, height = get_axis_xy_width_height(ax, -0.7, -0.2, 1, 0.4)
    else:
        xy, width, height = get_axis_xy_width_height(ax)
    if offset is not None:
        xoff, yoff = offset
        xy = [xoff, yoff]
        height = -height - yoff
        width = width - xoff
    rect = mpl.patches.Rectangle(xy, width, height, lw=lw)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(False)
    rect.set_edgecolor(color)
    return rect


TAU = np.pi * 2


def rotate_plot(theta=TAU / 8, ax=None):
    r"""
    Args:
        theta (?):
        ax (None):

    CommandLine:
        python -m wbia.plottool.draw_func2 --test-rotate_plot

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
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
    # import vtool as vt
    xy, width, height = get_axis_xy_width_height(ax)
    bbox = [xy[0], xy[1], width, height]
    M = mpl.transforms.Affine2D(vt.rotation_around_bbox_mat3x3(theta, bbox))
    propname = 'transAxes'
    # propname = 'transData'
    T = getattr(ax, propname)
    T.transform_affine(M)
    # T = ax.get_transform()
    # Tnew = T + M
    # ax.set_transform(Tnew)
    # setattr(ax, propname, Tnew)
    iup()


def cartoon_stacked_rects(xy, width, height, num=4, shift=None, **kwargs):
    """
    pt.figure()
    xy = (.5, .5)
    width = .2
    height = .2
    ax = pt.gca()
    ax.add_collection(col)
    """
    if shift is None:
        shift = np.array([-width, height]) * (0.1 / num)
    xy = np.array(xy)
    rectkw = dict(
        ec=kwargs.pop('ec', None),
        lw=kwargs.pop('lw', None),
        linestyle=kwargs.pop('linestyle', None),
    )
    patch_list = [
        mpl.patches.Rectangle(xy + shift * count, width, height, **rectkw)
        for count in reversed(range(num))
    ]
    col = mpl.collections.PatchCollection(patch_list, **kwargs)
    return col


def make_bbox(
    bbox,
    theta=0,
    bbox_color=None,
    ax=None,
    lw=2,
    alpha=1.0,
    align='center',
    fill=None,
    **kwargs
):
    if ax is None:
        ax = gca()
    (rx, ry, rw, rh) = bbox
    # Transformations are specified in backwards order.
    trans_annotation = mpl.transforms.Affine2D()
    if align == 'center':
        trans_annotation.scale(rw, rh)
    elif align == 'outer':
        trans_annotation.scale(rw + (lw / 2), rh + (lw / 2))
    elif align == 'inner':
        trans_annotation.scale(rw - (lw / 2), rh - (lw / 2))

    trans_annotation.rotate(theta)
    trans_annotation.translate(rx + rw / 2, ry + rh / 2)
    t_end = trans_annotation + ax.transData
    bbox = mpl.patches.Rectangle((-0.5, -0.5), 1, 1, lw=lw, transform=t_end, **kwargs)
    bbox.set_fill(fill if fill else None)
    bbox.set_alpha(alpha)
    # bbox.set_transform(trans)
    bbox.set_edgecolor(bbox_color)
    return bbox


# TODO SEPARTE THIS INTO DRAW BBOX AND DRAW_ANNOTATION
def draw_bbox(
    bbox,
    lbl=None,
    bbox_color=(1, 0, 0),
    lbl_bgcolor=(0, 0, 0),
    lbl_txtcolor=(1, 1, 1),
    draw_arrow=True,
    theta=0,
    ax=None,
    lw=2,
):
    if ax is None:
        ax = gca()
    (rx, ry, rw, rh) = bbox
    # Transformations are specified in backwards order.
    trans_annotation = mpl.transforms.Affine2D()
    trans_annotation.scale(rw, rh)
    trans_annotation.rotate(theta)
    trans_annotation.translate(rx + rw / 2, ry + rh / 2)
    t_end = trans_annotation + ax.transData
    bbox = mpl.patches.Rectangle((-0.5, -0.5), 1, 1, lw=lw, transform=t_end)
    bbox.set_fill(False)
    # bbox.set_transform(trans)
    bbox.set_edgecolor(bbox_color)
    ax.add_patch(bbox)
    # Draw overhead arrow indicating the top of the ANNOTATION
    if draw_arrow:
        arw_xydxdy = (-0.5, -0.5, 1.0, 0.0)
        arw_kw = dict(head_width=0.1, transform=t_end, length_includes_head=True)
        arrow = mpl.patches.FancyArrow(*arw_xydxdy, **arw_kw)
        arrow.set_edgecolor(bbox_color)
        arrow.set_facecolor(bbox_color)
        ax.add_patch(arrow)
    # Draw a label
    if lbl is not None:
        ax_absolute_text(
            rx,
            ry,
            lbl,
            ax=ax,
            horizontalalignment='center',
            verticalalignment='center',
            color=lbl_txtcolor,
            backgroundcolor=lbl_bgcolor,
        )


def plot(*args, **kwargs):
    yscale = kwargs.pop('yscale', 'linear')
    xscale = kwargs.pop('xscale', 'linear')
    logscale_kwargs = kwargs.pop('logscale_kwargs', {})  # , {'nonposx': 'clip'})
    plot = plt.plot(*args, **kwargs)
    ax = plt.gca()

    yscale_kwargs = logscale_kwargs if yscale in ['log', 'symlog'] else {}
    xscale_kwargs = logscale_kwargs if xscale in ['log', 'symlog'] else {}

    ax.set_yscale(yscale, **yscale_kwargs)
    ax.set_xscale(xscale, **xscale_kwargs)
    return plot


def plot2(
    x_data,
    y_data,
    marker='o',
    title_pref='',
    x_label='x',
    y_label='y',
    unitbox=False,
    flipx=False,
    flipy=False,
    title=None,
    dark=None,
    equal_aspect=True,
    pad=0,
    label='',
    fnum=None,
    pnum=None,
    *args,
    **kwargs
):
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
                set_axis_limit(-0.01, 1.01, -0.01, 1.01, ax)
                # ax.grid(True)
            else:
                set_axis_limit(min_, max_, min_, max_, ax)
                # aspect_opptions = ['auto', 'equal', num]
                ax.set_aspect('equal')
        else:
            ax.set_aspect('auto')
        if pad > 0:
            ax.set_xlim(min_x - pad, max_x + pad)
            ax.set_ylim(min_y - pad, max_y + pad)
        # ax.grid(True, color='w' if dark else 'k')
        if flipx:
            ax.invert_xaxis()
        if flipy:
            ax.invert_yaxis()

        use_darkbackground = dark
        if use_darkbackground is None:
            import wbia.plottool as pt

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


def presetup_axes(
    x_label='x',
    y_label='y',
    title_pref='',
    title=None,
    equal_aspect=False,
    ax=None,
    **kwargs
):
    if ax is None:
        ax = gca()
    set_xlabel(x_label, **kwargs)
    set_ylabel(y_label, **kwargs)
    if title is None:
        title = x_label + ' vs ' + y_label
    set_title(title_pref + ' ' + title, ax=None, **kwargs)
    if equal_aspect:
        ax.set_aspect('equal')


def postsetup_axes(use_legend=True, bg=None):
    import wbia.plottool as pt

    if bg is None:
        if pt.is_default_dark_bg():
            bg = 'dark'

    if bg == 'dark':
        dark_background()
    if use_legend:
        legend()


def adjust_subplots(
    left=None,
    right=None,
    bottom=None,
    top=None,
    wspace=None,
    hspace=None,
    use_argv=False,
    fig=None,
):
    """
    Kwargs:
        left (float): left side of the subplots of the figure
        right (float): right side of the subplots of the figure
        bottom (float): bottom of the subplots of the figure
        top (float): top of the subplots of the figure
        wspace (float): width reserved for blank space between subplots
        hspace (float): height reserved for blank space between subplots
    """
    kwargs = dict(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if fig is None:
        fig = gcf()
    subplotpars = fig.subplotpars
    adjust_dict = subplotpars.__dict__.copy()
    del adjust_dict['validate']
    adjust_dict.update(kwargs)
    if use_argv:
        # hack to take args from commandline
        adjust_dict = ut.parse_dict_from_argv(adjust_dict)
    fig.subplots_adjust(**adjust_dict)


# =======================
# TEXT FUNCTIONS
# TODO: I have too many of these. Need to consolidate
# =======================


def upperleft_text(txt, alpha=0.6, color=None):
    txtargs = dict(
        horizontalalignment='left',
        verticalalignment='top',
        backgroundcolor=(0, 0, 0, alpha),
        color=ORANGE if color is None else color,
    )
    relative_text((0.02, 0.02), txt, **txtargs)


def upperright_text(txt, offset=None, alpha=0.6):
    txtargs = dict(
        horizontalalignment='right',
        verticalalignment='top',
        backgroundcolor=(0, 0, 0, alpha),
        color=ORANGE,
        offset=offset,
    )
    relative_text((0.98, 0.02), txt, **txtargs)


def lowerright_text(txt):
    txtargs = dict(
        horizontalalignment='right',
        verticalalignment='bottom',
        backgroundcolor=(0, 0, 0, 0.6),
        color=ORANGE,
    )
    relative_text((0.98, 0.92), txt, **txtargs)


def absolute_lbl(x_, y_, txt, roffset=(-0.02, -0.02), alpha=0.6, **kwargs):
    """ alternative to relative text """
    txtargs = dict(
        horizontalalignment='right',
        verticalalignment='top',
        backgroundcolor=(0, 0, 0, alpha),
        color=ORANGE,
    )
    txtargs.update(kwargs)
    ax_absolute_text(x_, y_, txt, roffset=roffset, **txtargs)


def absolute_text(pos, text, ax=None, **kwargs):
    x, y = pos
    ax_absolute_text(x, y, text, ax=ax, **kwargs)


def relative_text(pos, text, ax=None, offset=None, **kwargs):
    """
    Places text on axes in a relative position

    Args:
        pos (tuple): relative xy position
        text (str): text
        ax (None): (default = None)
        offset (None): (default = None)
        **kwargs: horizontalalignment, verticalalignment, roffset, ha, va,
                  fontsize, fontproperties, fontproperties, clip_on

    CommandLine:
        python -m wbia.plottool.draw_func2 relative_text --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> x = .5
        >>> y = .5
        >>> txt = 'Hello World'
        >>> pt.figure()
        >>> ax = pt.gca()
        >>> family = 'monospace'
        >>> family = 'CMU Typewriter Text'
        >>> fontproperties = mpl.font_manager.FontProperties(family=family,
        >>>                                                  size=42)
        >>> result = relative_text((x, y), txt, ax, halign='center',
        >>>                           fontproperties=fontproperties)
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    if pos == 'lowerleft':
        pos = (0.01, 0.99)
        kwargs['halign'] = 'left'
        kwargs['valign'] = 'bottom'
    elif pos == 'upperleft':
        pos = (0.01, 0.01)
        kwargs['halign'] = 'left'
        kwargs['valign'] = 'top'
    x, y = pos
    if ax is None:
        ax = gca()
    if 'halign' in kwargs:
        kwargs['horizontalalignment'] = kwargs.pop('halign')
    if 'valign' in kwargs:
        kwargs['verticalalignment'] = kwargs.pop('valign')
    xy, width, height = get_axis_xy_width_height(ax)
    x_, y_ = ((xy[0]) + x * width, (xy[1] + height) - y * height)
    if offset is not None:
        xoff, yoff = offset
        x_ += xoff
        y_ += yoff
    absolute_text((x_, y_), text, ax=ax, **kwargs)


def parse_fontkw(**kwargs):
    r"""
    Kwargs:
        fontsize, fontfamilty, fontproperties
    """
    from matplotlib.font_manager import FontProperties

    if 'fontproperties' not in kwargs:
        size = kwargs.get('fontsize', 14)
        weight = kwargs.get('fontweight', 'normal')
        fontname = kwargs.get('fontname', None)
        if fontname is not None:
            # TODO catch user warning
            '/usr/share/fonts/truetype/'
            '/usr/share/fonts/opentype/'
            fontpath = mpl.font_manager.findfont(fontname, fallback_to_default=False)
            font_prop = FontProperties(fname=fontpath, weight=weight, size=size)
        else:
            family = kwargs.get('fontfamilty', 'monospace')
            font_prop = FontProperties(family=family, weight=weight, size=size)
    else:
        font_prop = kwargs['fontproperties']
    return font_prop


def ax_absolute_text(x_, y_, txt, ax=None, roffset=None, **kwargs):
    """ Base function for text

    Kwargs:
        horizontalalignment in ['right', 'center', 'left'],
        verticalalignment in ['top']
        color

    """
    kwargs = kwargs.copy()
    if ax is None:
        ax = gca()
    if 'ha' in kwargs:
        kwargs['horizontalalignment'] = kwargs['ha']
    if 'va' in kwargs:
        kwargs['verticalalignment'] = kwargs['va']

    if 'fontproperties' not in kwargs:
        if 'fontsize' in kwargs:
            fontsize = kwargs['fontsize']
            font_prop = mpl.font_manager.FontProperties(
                family='monospace',
                # weight='light',
                size=fontsize,
            )
            kwargs['fontproperties'] = font_prop
        else:
            kwargs['fontproperties'] = mpl.font_manager.FontProperties(family='monospace')
            # custom_constants.FONTS.relative

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if roffset is not None:
        xroff, yroff = roffset
        xy, width, height = get_axis_xy_width_height(ax)
        x_ += xroff * width
        y_ += yroff * height

    return ax.text(x_, y_, txt, **kwargs)


def fig_relative_text(x, y, txt, **kwargs):
    kwargs['horizontalalignment'] = 'center'
    kwargs['verticalalignment'] = 'center'
    fig = gcf()
    # xy, width, height = get_axis_xy_width_height(ax)
    # x_, y_ = ((xy[0]+width)+x*width, (xy[1]+height)-y*height)
    fig.text(x, y, txt, **kwargs)


def draw_text(text_str, rgb_textFG=(0, 0, 0), rgb_textBG=(1, 1, 1)):
    ax = gca()
    xy, width, height = get_axis_xy_width_height(ax)
    text_x = xy[0] + (width / 2)
    text_y = xy[1] + (height / 2)
    ax.text(
        text_x,
        text_y,
        text_str,
        horizontalalignment='center',
        verticalalignment='center',
        color=rgb_textFG,
        backgroundcolor=rgb_textBG,
    )


# def convert_keypress_event_mpl_to_qt4(mevent):
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


# def test_build_qkeyevent():
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
        python -m wbia.plottool.draw_func2 --test-show_histogram --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> # build test data
        >>> data = np.array([1, 24, 0, 0, 3, 4, 5, 9, 3, 0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 0, 0, 0, 3,])
        >>> bins = None
        >>> # execute function
        >>> result = show_histogram(data, bins)
        >>> # verify results
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    print('[df2] show_histogram()')
    dmin = int(np.floor(data.min()))
    dmax = int(np.ceil(data.max()))
    if bins is None:
        bins = dmax - dmin
    fig = figure(**kwargs)
    ax = gca()
    ax.hist(data, bins=bins, range=(dmin, dmax))
    # dark_background()

    use_darkbackground = None
    if use_darkbackground is None:
        use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background(ax)
    return fig
    # help(np.bincount)
    # fig.show()


def show_signature(sig, **kwargs):
    fig = figure(**kwargs)
    plt.plot(sig)
    fig.show()


def draw_stems(
    x_data=None,
    y_data=None,
    setlims=True,
    color=None,
    markersize=None,
    bottom=None,
    marker=None,
    linestyle='-',
):
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
        python -m wbia.plottool.draw_func2 --test-draw_stems --show
        python -m wbia.plottool.draw_func2 --test-draw_stems

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
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
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
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
        color = [1.0, 0.0, 0.0, 1.0]

    OLD = False
    if not OLD:
        if bottom is None:
            bottom = 0
        # Faster way of drawing stems
        # with ut.Timer('new stem'):
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
                x_data_sort, y_data_sort, linefmt='-', bottom=bottom
            )
            if markersize is not None:
                markerline.set_markersize(markersize)

            pylab.setp(markerline, 'markerfacecolor', 'w')
            pylab.setp(stemlines, 'markerfacecolor', 'w')
            if color is not None:
                for line in stemlines:
                    line.set_color(color)
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
        python -m wbia.plottool.draw_func2 --test-plot_sift_signature --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> sift = vt.demodata.testdata_dummy_sift(1, np.random.RandomState(0))[0]
        >>> title = 'test sift histogram'
        >>> fnum = None
        >>> pnum = None
        >>> ax = plot_sift_signature(sift, title, fnum, pnum)
        >>> result = ('ax = %s' % (str(ax),))
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
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
    # dark_background(ax)

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
        python -m wbia.plottool.draw_func2 --test-plot_descriptor_signature --show

    SeeAlso:
        plot_sift_signature

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> vec = ((np.random.RandomState(0).rand(258) - .2) * 4)
        >>> title = 'test sift histogram'
        >>> fnum = None
        >>> pnum = None
        >>> ax = plot_descriptor_signature(vec, title, fnum, pnum)
        >>> result = ('ax = %s' % (str(ax),))
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    fnum = ensure_fnum(fnum)
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    plot_bars(vec, vec.size // 8)
    ax.set_xlim(0, vec.size)
    ax.set_ylim(vec.min(), vec.max())
    # space_xticks(9, 16)
    # space_yticks(5, 64)
    set_title(title, ax=ax)

    use_darkbackground = None
    if use_darkbackground is None:
        use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background(ax)

    return ax


def dark_background(ax=None, doubleit=False, force=False):
    r"""
    Args:
        ax (None): (default = None)
        doubleit (bool): (default = False)

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-dark_background --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> fig = pt.figure()
        >>> pt.dark_background()
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """

    def is_using_style(style):
        style_dict = mpl.style.library[style]
        return len(ut.dict_isect(style_dict, mpl.rcParams)) == len(style_dict)

    # is_using_style('classic')
    # is_using_style('ggplot')
    # HARD_DISABLE = force is not True
    HARD_DISABLE = False
    if not HARD_DISABLE and force:
        # Should use mpl style dark background instead
        bgcolor = BLACK * 0.9
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
            width *= doubleit + 1
            height *= doubleit + 1
        rect = mpl.patches.Rectangle(xy, width, height, lw=0, zorder=0)
        rect.set_clip_on(True)
        rect.set_fill(True)
        rect.set_color(bgcolor)
        rect.set_zorder(-99999999999)
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
        ax.bar(x_dat, y_dat, width, color=color, edgecolor=np.array(color) * 0.8)


def append_phantom_legend_label(label, color, type_='circle', alpha=1.0, ax=None):
    """
    adds a legend label without displaying an actor

    Args:
        label (?):
        color (?):
        loc (str):

    CommandLine:
        python -m wbia.plottool.draw_func2 --test-append_phantom_legend_label --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> label = 'some label'
        >>> color = 'b'
        >>> loc = 'upper right'
        >>> fig = pt.figure()
        >>> ax = pt.gca()
        >>> result = append_phantom_legend_label(label, color, loc, ax=ax)
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.quit_if_noshow()
        >>> pt.show_phantom_legend_labels(ax=ax)
        >>> pt.show_if_requested()
    """
    # pass
    # , loc=loc
    if ax is None:
        ax = gca()
    _phantom_legend_list = getattr(ax, '_phantom_legend_list', None)
    if _phantom_legend_list is None:
        _phantom_legend_list = []
        setattr(ax, '_phantom_legend_list', _phantom_legend_list)
    if type_ == 'line':
        phantom_actor = plt.Line2D((0, 0), (1, 1), color=color, label=label, alpha=alpha)
    else:
        phantom_actor = plt.Circle((0, 0), 1, fc=color, label=label, alpha=alpha)
    # , prop=custom_constants.FONTS.legend)
    # legend_tups = []
    _phantom_legend_list.append(phantom_actor)
    # ax.legend(handles=[phantom_actor], framealpha=.2)
    # plt.legend(*zip(*legend_tups), framealpha=.2)


def show_phantom_legend_labels(ax=None, **kwargs):
    if ax is None:
        ax = gca()
    _phantom_legend_list = getattr(ax, '_phantom_legend_list', None)
    if _phantom_legend_list is None:
        _phantom_legend_list = []
        setattr(ax, '_phantom_legend_list', _phantom_legend_list)
    # print(_phantom_legend_list)
    legend(handles=_phantom_legend_list, ax=ax, **kwargs)
    # ax.legend(handles=_phantom_legend_list, framealpha=.2)


LEGEND_LOCATION = {
    'upper right': 1,
    'upper left': 2,
    'lower left': 3,
    'lower right': 4,
    'right': 5,
    'center left': 6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center': 10,
}


# def legend(loc='upper right', fontproperties=None):
def legend(
    loc='best', fontproperties=None, size=None, fc='w', alpha=1, ax=None, handles=None
):
    r"""
    Args:
        loc (str): (default = 'best')
        fontproperties (None): (default = None)
        size (None): (default = None)

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-legend --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> loc = 'best'
        >>> import wbia.plottool as pt
        >>> xdata = np.linspace(-6, 6)
        >>> ydata = np.sin(xdata)
        >>> pt.plot(xdata, ydata, label='sin')
        >>> fontproperties = None
        >>> size = None
        >>> result = legend(loc, fontproperties, size)
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    assert loc in LEGEND_LOCATION or loc == 'best', 'invalid loc. try one of %r' % (
        LEGEND_LOCATION,
    )
    if ax is None:
        ax = gca()
    if fontproperties is None:
        prop = {}
        if size is not None:
            prop['size'] = size
        # prop['weight'] = 'normal'
        # prop['family'] = 'sans-serif'
    else:
        prop = fontproperties
    legendkw = dict(loc=loc)
    if prop:
        legendkw['prop'] = prop
    if handles is not None:
        legendkw['handles'] = handles
    legend = ax.legend(**legendkw)
    if legend:
        legend.get_frame().set_fc(fc)
        legend.get_frame().set_alpha(alpha)


def plot_histpdf(data, label=None, draw_support=False, nbins=10):
    freq, _ = plot_hist(data, nbins=nbins)
    from wbia.plottool import plots

    plots.plot_pdf(data, draw_support=draw_support, scale_to=freq.max(), label=label)


def plot_hist(data, bins=None, nbins=10, weights=None):
    if isinstance(data, list):
        data = np.array(data)
    dmin = data.min()
    dmax = data.max()
    if bins is None:
        bins = dmax - dmin
    ax = gca()
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
    # trunc_xticks = np.linspace(0, int(trunc_max),11)
    # trunc_xticks = trunc_xticks[trunc_xticks >= trunc_min]
    # trunc_xticks = np.append([int(trunc_min)], trunc_xticks)
    # no_zero_yticks = ax.get_yticks()[ax.get_yticks() > 0]
    # ax.set_xticks(trunc_xticks)
    # ax.set_yticks(no_zero_yticks)


# _----------------- HELPERS ^^^ ---------


def scores_to_color(
    score_list,
    cmap_='hot',
    logscale=False,
    reverse_cmap=False,
    custom=False,
    val2_customcolor=None,
    score_range=None,
    cmap_range=(0.1, 0.9),
):
    """
    Other good colormaps are 'spectral', 'gist_rainbow', 'gist_ncar', 'Set1',
    'Set2', 'Accent'
    # TODO: plasma

    Args:
        score_list (list):
        cmap_ (str): defaults to hot
        logscale (bool):
        cmap_range (tuple): restricts to only a portion of the cmap to avoid extremes

    Returns:
        <class '_ast.ListComp'>

    SeeAlso:
        python -m wbia.plottool.color_funcs --test-show_all_colormaps --show --type "Perceptually Uniform Sequential"

    CommandLine:
        python -m wbia.plottool.draw_func2 scores_to_color --show

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> ut.exec_funckw(pt.scores_to_color, globals())
        >>> score_list = np.array([-1, -2, 1, 1, 2, 10])
        >>> # score_list = np.array([0, .1, .11, .12, .13, .8])
        >>> # score_list = np.linspace(0, 1, 100)
        >>> cmap_ = 'plasma'
        >>> colors = pt.scores_to_color(score_list, cmap_)
        >>> import vtool as vt
        >>> imgRGB = vt.atleast_nd(np.array(colors)[:, 0:3], 3, tofront=True)
        >>> imgRGB = imgRGB.astype(np.float32)
        >>> imgBGR = vt.convert_colorspace(imgRGB, 'BGR', 'RGB')
        >>> pt.imshow(imgBGR)
        >>> pt.show_if_requested()

    Example:
        >>> from wbia.plottool.draw_func2 import *  # NOQA
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
    assert len(score_list.shape) == 1, 'score must be 1d'
    if len(score_list) == 0:
        return []

    def apply_logscale(scores):
        scores = np.array(scores)
        above_zero = scores >= 0
        scores_ = scores.copy()
        scores_[above_zero] = scores_[above_zero] + 1
        scores_[~above_zero] = scores_[~above_zero] - 1
        scores_ = np.log2(scores_)
        return scores_

    if logscale:
        # Hack
        score_list = apply_logscale(score_list)
        # if loglogscale
        # score_list = np.log2(np.log2(score_list + 2) + 1)
    # if isinstance(cmap_, six.string_types):
    cmap = plt.get_cmap(cmap_)
    # else:
    #    cmap = cmap_
    if reverse_cmap:
        cmap = reverse_colormap(cmap)
    # if custom:
    #    base_colormap = cmap
    #    data = score_list
    #    cmap = customize_colormap(score_list, base_colormap)
    if score_range is None:
        min_ = score_list.min()
        max_ = score_list.max()
    else:
        min_ = score_range[0]
        max_ = score_range[1]
        if logscale:
            min_, max_ = apply_logscale([min_, max_])
    if cmap_range is None:
        cmap_scale_min, cmap_scale_max = 0.0, 1.0
    else:
        cmap_scale_min, cmap_scale_max = cmap_range
    extent_ = max_ - min_
    if extent_ == 0:
        colors = [cmap(0.5) for fx in range(len(score_list))]
    else:
        if False and logscale:
            # hack
            def score2_01(score):
                return np.log2(
                    1
                    + cmap_scale_min
                    + cmap_scale_max * (float(score) - min_) / (extent_)
                )

            score_list = np.array(score_list)
            # rank_multiplier = score_list.argsort() / len(score_list)
            # normscore = np.array(list(map(score2_01, score_list))) * rank_multiplier
            normscore = np.array(list(map(score2_01, score_list)))
            colors = list(map(cmap, normscore))
        else:

            def score2_01(score):
                return cmap_scale_min + cmap_scale_max * (float(score) - min_) / (extent_)

        colors = [cmap(score2_01(score)) for score in score_list]
        if val2_customcolor is not None:
            colors = [
                np.array(val2_customcolor.get(score, color))
                for color, score in zip(colors, score_list)
            ]
    return colors


def customize_colormap(data, base_colormap):
    unique_scalars = np.array(sorted(np.unique(data)))
    max_ = unique_scalars.max()
    min_ = unique_scalars.min()
    extent_ = max_ - min_
    bounds = np.linspace(min_, max_ + 1, extent_ + 2)

    # Get a few more colors than we actually need so we don't hit the bottom of
    # the cmap
    colors_ix = np.concatenate((np.linspace(0, 1.0, extent_ + 2), (0.0, 0.0, 0.0, 0.0)))
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
    sm.set_clim(-0.5, extent_ + 0.5)
    # colorbar = plt.colorbar(sm)

    return cmap


def unique_rows(arr):
    """
    References:
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """
    rowblocks = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    )
    _, idx = np.unique(rowblocks, return_index=True)
    unique_arr = arr[idx]
    return unique_arr


def scores_to_cmap(scores, colors=None, cmap_='hot'):
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
    from wbia.plottool import plot_helpers as ph

    divider = ph.get_plotdat(ax, DF2_DIVIDER_KEY, None)
    if divider is None:
        divider = make_axes_locatable(ax)
        ph.set_plotdat(ax, DF2_DIVIDER_KEY, divider)
        orig_append_axes = divider.append_axes

        def df2_append_axes(
            divider, position, size, pad=None, add_to_figure=True, **kwargs
        ):
            """ override divider add axes to register the divided axes """
            div_axes = ph.get_plotdat(ax, 'df2_div_axes', [])
            new_ax = orig_append_axes(
                position, size, pad=pad, add_to_figure=add_to_figure, **kwargs
            )
            div_axes.append(new_ax)
            ph.set_plotdat(ax, 'df2_div_axes', div_axes)
            return new_ax

        ut.inject_func_as_method(
            divider, df2_append_axes, 'append_axes', allow_override=True
        )
    return divider


def get_binary_svm_cmap():
    # useful for svms
    return reverse_colormap(plt.get_cmap('bwr'))


def reverse_colormap(cmap):
    """
    References:
        http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Matteo_colourmaps.ipynb
    """
    if isinstance(cmap, mpl.colors.ListedColormap):
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
            cmap.name + '_reversed', dict(zip(k, reverse))
        )
        return cmap_reversed


def interpolated_colormap(color_frac_list, resolution=64, space='lch-ab'):
    """
    http://stackoverflow.com/questions/12073306/customize-colorbar-in-matplotlib

    CommandLine:
        python -m wbia.plottool.draw_func2 interpolated_colormap --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> color_frac_list = [
        >>>     (pt.TRUE_BLUE, 0),
        >>>     #(pt.WHITE, .5),
        >>>     (pt.YELLOW, .5),
        >>>     (pt.FALSE_RED, 1.0),
        >>> ]
        >>> color_frac_list = [
        >>>     (pt.RED, 0),
        >>>     (pt.PINK, .1),
        >>>     (pt.ORANGE, .2),
        >>>     (pt.GREEN, .5),
        >>>     (pt.TRUE_BLUE, .7),
        >>>     (pt.PURPLE, 1.0),
        >>> ]
        >>> color_frac_list = [
        >>>     (pt.RED,      0/6),
        >>>     (pt.YELLOW,   1/6),
        >>>     (pt.GREEN,    2/6),
        >>>     (pt.CYAN,     3/6),
        >>>     (pt.BLUE,     4/6),  # FIXME doesn't go in correct direction
        >>>     (pt.MAGENTA,  5/6),
        >>>     (pt.RED,      6/6),
        >>> ]
        >>> color_frac_list = [
        >>>     ((1, 0, 0, 0),  0/6),
        >>>     ((1, 0, .001/255, 0),  6/6), # hack
        >>> ]
        >>> space = 'hsv'
        >>> color_frac_list = [
        >>>     (pt.BLUE,   0.0),
        >>>     (pt.GRAY,   0.5),
        >>>     (pt.YELLOW, 1.0),
        >>> ]
        >>> color_frac_list = [
        >>>     (pt.GREEN,  0.0),
        >>>     (pt.GRAY,   0.5),
        >>>     (pt.RED,    1.0),
        >>> ]
        >>> space = 'lab'
        >>> #resolution = 16 + 1
        >>> resolution = 256 + 1
        >>> cmap = interpolated_colormap(color_frac_list, resolution, space)
        >>> import wbia.plottool as pt
        >>> pt.quit_if_noshow()
        >>> a = np.linspace(0, 1, resolution).reshape(1, -1)
        >>> pylab.imshow(a, aspect='auto', cmap=cmap, interpolation='nearest')  # , origin="lower")
        >>> plt.grid(False)
        >>> pt.show_if_requested()
    """
    import colorsys

    if len(color_frac_list[0]) != 2:
        color_frac_list = list(
            zip(color_frac_list, np.linspace(0, 1, len(color_frac_list)))
        )

    colors = ut.take_column(color_frac_list, 0)
    fracs = ut.take_column(color_frac_list, 1)

    # resolution = 17
    basis = np.linspace(0, 1, resolution)
    fracs = np.array(fracs)
    indices = np.searchsorted(fracs, basis)
    indices = np.maximum(indices, 1)
    cpool = []

    # vt.convert_colorspace((c[None, None, 0:3] * 255).astype(np.uint8), 'RGB', 'HSV') / 255

    # import colorspacious
    # import colormath
    from colormath import color_conversions

    # FIXME: need to ensure monkeypatch for networkx 2.0 in colormath
    # color_conversions._conversion_manager = color_conversions.GraphConversionManager()

    from colormath import color_objects

    # from colormath import color_conversions

    def new_convertor(target_obj):
        source_obj = color_objects.sRGBColor

        def to_target(src_tup):
            src_tup = src_tup[0:3]
            src_co = source_obj(*src_tup)
            target_co = color_conversions.convert_color(src_co, target_obj)
            target_tup = target_co.get_value_tuple()
            return target_tup

        def from_target(target_tup):
            target_co = target_obj(*target_tup)
            src_co = color_conversions.convert_color(target_co, source_obj)
            src_tup = src_co.get_value_tuple()
            return src_tup

        return to_target, from_target
        # colorspacious.cspace_convert(rgb, "sRGB255", "CIELCh")

    def from_hsv(rgb):
        return colorsys.rgb_to_hsv(*rgb[0:3])

    def to_hsv(hsv):
        return colorsys.hsv_to_rgb(*hsv[0:3].tolist())

    classnames = {
        # 'AdobeRGBColor',
        # 'BaseRGBColor',
        'cmk': 'CMYColor',
        'cmyk': 'CMYKColor',
        'hsl': 'HSLColor',
        'hsv': 'HSVColor',
        'ipt': 'IPTColor',
        'lch-ab': 'LCHabColor',
        'lch-uv': 'LCHuvColor',
        'lab': 'LabColor',
        'luv': 'LuvColor',
        # 'SpectralColor',
        'xyz': 'XYZColor',
        # 'sRGBColor',
        'xyy': 'xyYColor',
    }

    conversions = {
        k: new_convertor(getattr(color_objects, v)) for k, v in classnames.items()
    }

    # conversions = {
    #     'lch': new_convertor(color_objects.LCHabColor),
    #     'lch-uv': new_convertor(color_objects.LCHuvColor),
    #     'lab': new_convertor(color_objects.LabColor),
    #     'hsv': new_convertor(color_objects.HSVColor),
    #     'xyz': new_convertor(color_objects.XYZColor)
    # }
    from_rgb, to_rgb = conversions['hsv']
    from_rgb, to_rgb = conversions['xyz']
    from_rgb, to_rgb = conversions['lch-uv']
    from_rgb, to_rgb = conversions['lch-ab']
    from_rgb, to_rgb = conversions[space]
    # from_rgb, to_rgb = conversions['lch']
    # from_rgb, to_rgb = conversions['lab']
    # from_rgb, to_rgb = conversions['lch-uv']

    for idx2, b in zip(indices, basis):
        idx1 = idx2 - 1
        f1 = fracs[idx1]
        f2 = fracs[idx2]

        c1 = colors[idx1]
        c2 = colors[idx2]
        # from_rgb, to_rgb = conversions['lch']
        h1 = np.array(from_rgb(c1))
        h2 = np.array(from_rgb(c2))
        alpha = (b - f1) / (f2 - f1)
        new_h = h1 * (1 - alpha) + h2 * (alpha)
        new_c = np.clip(to_rgb(new_h), 0, 1)
        # print('new_c = %r' % (new_c,))
        cpool.append(new_c)

    cpool = np.array(cpool)
    # print('cpool = %r' % (cpool,))
    cmap = mpl.colors.ListedColormap(cpool, 'indexed')
    return cmap

    # cm.register_cmap(cmap=cmap3)
    # pass


def print_valid_cmaps():
    import pylab
    import utool as ut

    maps = [m for m in pylab.cm.datad if not m.endswith('_r')]
    print(ut.repr2(sorted(maps)))


def colorbar(
    scalars,
    colors,
    custom=False,
    lbl=None,
    ticklabels=None,
    float_format='%.2f',
    **kwargs
):
    """
    adds a color bar next to the axes based on specific scalars

    Args:
        scalars (ndarray):
        colors (ndarray):
        custom (bool): use custom ticks

    Kwargs:
        See plt.colorbar

    Returns:
        cb : matplotlib colorbar object

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-colorbar --show
        python -m wbia.plottool.draw_func2 --exec-colorbar:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> from wbia.plottool import draw_func2 as df2
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> scalars = np.array([-1, -2, 1, 1, 2, 7, 10])
        >>> cmap_ = 'plasma'
        >>> logscale = False
        >>> custom = True
        >>> reverse_cmap = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale, reverse_cmap=reverse_cmap, val2_customcolor=val2_customcolor)
        >>> colorbar(scalars, colors, custom=custom)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> from wbia.plottool import draw_func2 as df2
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> scalars = np.linspace(0, 1, 100)
        >>> cmap_ = 'plasma'
        >>> logscale = False
        >>> custom = False
        >>> reverse_cmap = False
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale,
        >>>                          reverse_cmap=reverse_cmap)
        >>> colors = [pt.lighten_rgb(c, .3) for c in colors]
        >>> colorbar(scalars, colors, custom=custom)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    from wbia.plottool import plot_helpers as ph

    assert len(scalars) == len(colors), 'scalars and colors must be corresponding'
    if len(scalars) == 0:
        return None
    # Parameters
    ax = gca()
    divider = ensure_divider(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    xy, width, height = get_axis_xy_width_height(ax)
    # orientation = ['vertical', 'horizontal'][0]
    TICK_FONTSIZE = 8
    #
    # Create scalar mappable with cmap
    if custom:
        # FIXME: clean this code up and change the name custom
        # to be meaningful. It is more like: display unique colors
        unique_scalars, unique_idx = np.unique(scalars, return_index=True)
        unique_colors = np.array(colors)[unique_idx]
        # max_, min_ = unique_scalars.max(), unique_scalars.min()
        # extent_ = max_ - min_
        # bounds = np.linspace(min_, max_ + 1, extent_ + 2)
        listed_cmap = mpl.colors.ListedColormap(unique_colors)
        # norm = mpl.colors.BoundaryNorm(bounds, listed_cmap.N)
        # sm = mpl.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
        sm = mpl.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(np.linspace(0, 1, len(unique_scalars) + 1))
    else:
        sorted_scalars = sorted(scalars)
        listed_cmap = scores_to_cmap(scalars, colors)
        sm = plt.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(sorted_scalars)
    # Use mapable object to create the colorbar
    # COLORBAR_SHRINK = .42  # 1
    # COLORBAR_PAD = .01  # 1
    # COLORBAR_ASPECT = np.abs(20 * height / (width))  # 1

    cb = plt.colorbar(sm, cax=cax, **kwargs)

    # # Add the colorbar to the correct label
    # axis = cb.ax.yaxis  # if orientation == 'horizontal' else cb.ax.yaxis
    # position = 'bottom' if orientation == 'horizontal' else 'right'
    # axis.set_ticks_position(position)

    # This line alone removes data
    # axis.set_ticks([0, .5, 1])
    if custom:
        ticks = np.linspace(0, 1, len(unique_scalars) + 1)
        if len(ticks) < 2:
            ticks += 0.5
        else:
            # SO HACKY
            ticks += (ticks[1] - ticks[0]) / 2

        if isinstance(unique_scalars, np.ndarray) and ut.is_float(unique_scalars):
            ticklabels = [float_format % scalar for scalar in unique_scalars]
        else:
            ticklabels = unique_scalars
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)  # tick labels
    elif ticklabels is not None:
        ticks_ = cb.ax.get_yticks()
        mx = ticks_.max()
        mn = ticks_.min()
        ticks = np.linspace(mn, mx, len(ticklabels))
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)
        # cb.ax.get_yticks()
        # cb.set_ticks(ticks)  # tick locations
        # cb.set_ticklabels(ticklabels)  # tick labels
    ph.set_plotdat(cb.ax, 'viztype', 'colorbar-%s' % (lbl,))
    ph.set_plotdat(cb.ax, 'sm', sm)
    # FIXME: Figure out how to make a maximum number of ticks
    # and to enforce them to be inside the data bounds
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    # Sets current axis
    plt.sca(ax)
    if lbl is not None:
        cb.set_label(lbl)
    return cb


def draw_lines2(
    kpts1,
    kpts2,
    fm=None,
    fs=None,
    kpts2_offset=(0, 0),
    color_list=None,
    scale_factor=1,
    lw=1.4,
    line_alpha=0.35,
    H1=None,
    H2=None,
    scale_factor1=None,
    scale_factor2=None,
    ax=None,
    **kwargs
):
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
    if ax is None:
        ax = gca()
    woff, hoff = kpts2_offset
    # Draw line collection
    kpts1_m = kpts1[fm[:, 0]].T
    kpts2_m = kpts2[fm[:, 1]].T
    xy1_m = kpts1_m[0:2]
    xy2_m = kpts2_m[0:2]
    if H1 is not None:
        xy1_m = vt.transform_points_with_homography(H1, xy1_m)
    if H2 is not None:
        xy2_m = vt.transform_points_with_homography(H2, xy2_m)
    xy1_m = xy1_m * scale_factor * np.array(scale_factor1)[:, None]
    xy2_m = (xy2_m * scale_factor * np.array(scale_factor2)[:, None]) + np.array(
        [woff, hoff]
    )[:, None]
    if color_list is None:
        if fs is None:  # Draw with solid color
            color_list = [RED for fx in range(len(fm))]
        else:  # Draw with colors proportional to score difference
            color_list = scores_to_color(fs)
    segments = [((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in zip(xy1_m.T, xy2_m.T)]
    linewidth = [lw for fx in range(len(fm))]
    # line_alpha = line_alpha
    # line_alpha = np.linspace(0, 1, len(fm))

    if ut.isiterable(line_alpha):
        # Hack for multiple alphas
        for segment, alpha, color in zip(segments, line_alpha, color_list):
            line_group = mpl.collections.LineCollection(
                [segment], linewidth, color, alpha=alpha
            )
            ax.add_collection(line_group)
    else:
        line_group = mpl.collections.LineCollection(
            segments, linewidth, color_list, alpha=line_alpha
        )
        # plt.colorbar(line_group, ax=ax)
        ax.add_collection(line_group)
    # figure(100)
    # plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)


def draw_line_segments2(pts1, pts2, ax=None, **kwargs):
    """
    draws `N` line segments

    Args:
        pts1 (ndarray): Nx2
        pts2 (ndarray): Nx2
        ax (None): (default = None)

    CommandLine:
        python -m wbia.plottool.draw_func2 draw_line_segments2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> pts1 = np.array([(1, 1), (0, 0)])
        >>> pts2 = np.array([(2, 2), (1, 0)])
        >>> pt.figure(fnum=None)
        >>> #segments = [np.array((xy1, xy2)) for xy1, xy2 in zip(pts1, pts2)]
        >>> #draw_line_segments(segments)
        >>> draw_line_segments2(pts1, pts2)
        >>> import wbia.plottool as pt
        >>> pt.quit_if_noshow()
        >>> ax = pt.gca()
        >>> pt.set_axis_limit(-1, 3, -1, 3, ax)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    if ax is None:
        ax = gca()
    assert len(pts1) == len(pts2), 'unaligned'
    # if len(pts1) == 0:
    #     return
    segments = [(xy1, xy2) for xy1, xy2 in zip(pts1, pts2)]
    linewidth = kwargs.pop('lw', kwargs.pop('linewidth', 1.0))
    alpha = kwargs.pop('alpha', 1.0)
    # if 'color' in kwargs:
    #     kwargs['color'] = mpl.colors.ColorConverter().to_rgb(kwargs['color'])
    line_group = mpl.collections.LineCollection(
        segments, linewidth, alpha=alpha, **kwargs
    )
    ax.add_collection(line_group)


def draw_line_segments(segments_list, **kwargs):
    """
    segments_list - list of [xs,ys,...] defining the segments
    """
    import wbia.plottool as pt

    marker = '.-'
    for data in segments_list:
        pt.plot(data.T[0], data.T[1], marker, **kwargs)

    # from matplotlib.collections import LineCollection
    # points_list = [np.array([pts[0], pts[1]]).T.reshape(-1, 1, 2) for pts in segments_list]
    # segments_list = [np.concatenate([points[:-1], points[1:]], axis=1) for points in points_list]
    # linewidth = 2
    # alpha = 1.0
    # lc_list = [LineCollection(segments, linewidth=linewidth, alpha=alpha)
    #           for segments in segments_list]
    # ax = plt.gca()
    # for lc in lc_list:
    #    ax.add_collection(lc)


def draw_patches_and_sifts(patch_list, sift_list, fnum=None, pnum=(1, 1, 1)):
    # Hacked together will not work on inputs of all sizes
    # raise NotImplementedError('unfinished')
    import wbia.plottool as pt

    num, width, height = patch_list.shape[0:3]
    rows = int(np.sqrt(num))
    cols = num // rows
    # TODO: recursive stack
    # stacked_img = patch_list.transpose(2, 0, 1).reshape(height * rows, width * cols)
    stacked_img = np.vstack([np.hstack(chunk) for chunk in ut.ichunks(patch_list, rows)])

    x_base = ((np.arange(rows) + 0.5) * width) - 0.5
    y_base = ((np.arange(cols) + 0.5) * height) - 0.5
    xs, ys = np.meshgrid(x_base, y_base)

    tmp_kpts = np.vstack(
        (
            xs.flatten(),
            ys.flatten(),
            width / 2 * np.ones(len(patch_list)),
            np.zeros(len(patch_list)),
            height / 2 * np.ones(len(patch_list)),
            np.zeros(len(patch_list)),
        )
    ).T

    pt.figure(fnum=fnum, pnum=pnum, docla=True)
    pt.imshow(stacked_img, pnum=pnum, fnum=fnum)
    # ax = pt.gca()
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    if sift_list is not None:
        pt.draw_kpts2(tmp_kpts, sifts=sift_list)
    return gca()
    # pt.iup()


def show_kpts(kpts, fnum=None, pnum=None, **kwargs):
    r"""
    Show keypoints in a new figure. Note: use draw_kpts2 to overlay keypoints on a existing figure.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        xdoctest -m ~/code/plottool/plottool/draw_func2.py show_kpts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> kpts = vt.demodata.get_dummy_kpts()
        >>> result = show_kpts(kpts)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    import vtool as vt
    import wbia.plottool as pt

    pt.figure(doclf=True, fnum=pt.ensure_fnum(fnum), pnum=pnum)
    pt.draw_kpts2(kpts, **kwargs)
    ax = pt.gca()
    extents = vt.get_kpts_image_extent(kpts)
    set_axis_extent(extents)
    ax.set_aspect('equal')


def set_axis_extent(extents, ax=None):
    """

    Args:
        extents: xmin, xmax, ymin, ymax
    """
    if ax is None:
        ax = gca()
    ax.set_xlim(*extents[0:2])
    ax.set_ylim(*extents[2:4])


def set_axis_limit(xmin, xmax, ymin, ymax, ax=None):
    return set_axis_extent((xmin, xmax, ymin, ymax), ax=ax)


def draw_kpts2(
    kpts,
    offset=(0, 0),
    scale_factor=1,
    ell=True,
    pts=False,
    rect=False,
    eig=False,
    ori=False,
    pts_size=2,
    ell_alpha=0.6,
    ell_linewidth=1.5,
    ell_color=None,
    pts_color=ORANGE,
    color_list=None,
    pts_alpha=1.0,
    siftkw={},
    H=None,
    weights=None,
    cmap_='hot',
    ax=None,
    **kwargs
):
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
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> from wbia.plottool import draw_func2 as df2
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

    # if ut.DEBUG2:
    #    print('-------------')
    #    print('draw_kpts2():')
    #    #print(' * kwargs.keys()=%r' % (kwargs.keys(),))
    #    print(' * kpts.shape=%r:' % (kpts.shape,))
    #    print(' * ell=%r pts=%r' % (ell, pts))
    #    print(' * rect=%r eig=%r, ori=%r' % (rect, eig, ori))
    #    print(' * scale_factor=%r' % (scale_factor,))
    #    print(' * offset=%r' % (offset,))
    #    print(' * drawing kpts.shape=%r' % (kpts.shape,))
    try:
        assert len(kpts) > 0, 'len(kpts) < 0'
    except AssertionError as ex:
        ut.printex(ex)
        return
    if ax is None:
        ax = gca()
    if color_list is None and weights is not None:
        # hack to turn into a color map
        color_list = scores_to_color(weights, cmap_=cmap_, reverse_cmap=False)
    if color_list is not None:
        ell_color = color_list
        pts_color = color_list
    # else:
    # pts_color = [pts_color for _ in range(len(kpts))]
    if isinstance(ell_color, six.string_types) and ell_color == 'distinct':
        ell_color = distinct_colors(len(kpts))  # , randomize=True)
        # print(len(kpts))

    _kwargs = kwargs.copy()
    _kwargs.update(
        {
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
        }
    )

    mpl_kp.draw_keypoints(ax, kpts, siftkw=siftkw, H=H, **_kwargs)
    return color_list


def draw_keypoint_gradient_orientations(
    rchip, kpt, sift=None, mode='vec', kptkw={}, siftkw={}, **kwargs
):
    """
    Extracts a keypoint patch from a chip, extract the gradient, and visualizes
    it with respect to the current mode.

    """
    import vtool as vt

    wpatch, wkp = vt.get_warped_patch(rchip, kpt, gray=True)
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
        fig = draw_vector_field(gradx, grady, **kwargs)
    elif mode == 'col' or mode == 'colors':
        import wbia.plottool as pt

        gmag = vt.patch_mag(gradx, grady)
        gori = vt.patch_ori(gradx, grady)
        gorimag = pt.color_orimag(gori, gmag)
        fig, ax = imshow(gorimag, **kwargs)
    wkpts = np.array([wkp])
    sifts = np.array([sift]) if sift is not None else None
    draw_kpts2(wkpts, sifts=sifts, siftkw=siftkw, **kptkw)
    return fig


# @ut.indent_func('[df2.dkp]')
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
        python -m wbia.plottool.draw_func2 --test-draw_keypoint_patch --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> rchip = vt.imread(ut.grab_test_imgpath('lena.png'))
        >>> kp = [100, 100, 20, 0, 20, 0]
        >>> sift = None
        >>> warped = True
        >>> patch_dict = {}
        >>> ax = draw_keypoint_patch(rchip, kp, sift, warped, patch_dict)
        >>> result = ('ax = %s' % (str(ax),))
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    import vtool as vt

    # print('--------------------')
    kpts = np.array([kp])
    if warped:
        patches, subkpts = vt.get_warped_patches(rchip, kpts)
    else:
        patches, subkpts = vt.get_unwarped_patches(rchip, kpts)
    # print('[df2] kpts[0]    = %r' % (kpts[0]))
    # print('[df2] subkpts[0] = %r' % (subkpts[0]))
    # print('[df2] patches[0].shape = %r' % (patches[0].shape,))
    patch = patches[0]
    subkpts_ = np.array(subkpts)
    patch_dict_ = {
        'sifts': None if sift is None else np.array([sift]),
        'ell_color': kwargs.get('ell_color', (0, 0, 1)),
        'pts': kwargs.get('pts', True),
        'ori': kwargs.get('ori', True),
        'ell': True,
        'eig': False,
        'rect': kwargs.get('rect', True),
        'stroke': kwargs.get('stroke', 1),
        'arm1_lw': kwargs.get('arm1_lw', 2),
        'multicolored_arms': kwargs.get('multicolored_arms', False),
    }
    patch_dict_.update(patch_dict)
    if 'ell_alpha' in kwargs:
        patch_dict['ell_alpha'] = kwargs['ell_alpha']

    # Draw patch with keypoint overlay
    fig, ax = imshow(patch, **kwargs)
    draw_kpts2(subkpts_, **patch_dict_)
    return ax


# ---- CHIP DISPLAY COMMANDS ----
def imshow(
    img,
    fnum=None,
    title=None,
    figtitle=None,
    pnum=None,
    interpolation='nearest',
    cmap=None,
    heatmap=False,
    data_colorbar=False,
    darken=DARKEN,
    update=False,
    xlabel=None,
    redraw_image=True,
    ax=None,
    alpha=None,
    norm=None,
    **kwargs
):
    r"""
    Args:
        img (ndarray): image data
        fnum (int): figure number
        title (str):
        figtitle (None):
        pnum (tuple): plot number
        interpolation (str): other interpolations = nearest, bicubic, bilinear
        cmap (None):
        heatmap (bool):
        data_colorbar (bool):
        darken (None):
        update (bool): (default = False)
        redraw_image (bool): used when calling imshow over and over. if false
                                doesnt do the image part.

    Returns:
        tuple: (fig, ax)

    Kwargs:
        docla, doclf, projection

    Returns:
        tuple: (fig, ax)

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-imshow --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> (fig, ax) = imshow(img)
        >>> result = ('(fig, ax) = %s' % (str((fig, ax)),))
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    if ax is not None:
        fig = ax.figure
        nospecial = True
    else:
        fig = figure(fnum=fnum, pnum=pnum, title=title, figtitle=figtitle, **kwargs)
        ax = gca()
        nospecial = False
        # ax.set_xticks([])
        # ax.set_yticks([])
        # return fig, ax

    if not redraw_image:
        return fig, ax

    if isinstance(img, six.string_types):
        # Allow for path to image to be specified
        img_fpath = img
        ut.assertpath(img_fpath)
        import vtool as vt

        img = vt.imread(img_fpath)
    # darken = .4
    if darken is not None:
        if darken is True:
            darken = 0.5
        # Darken the shown picture
        imgdtype = img.dtype
        img = np.array(img, dtype=float) * (1 - darken)
        img = np.array(img, dtype=imgdtype)

    plt_imshow_kwargs = {
        'interpolation': interpolation,
        # 'cmap': plt.get_cmap('gray'),
    }
    if alpha is not None:
        plt_imshow_kwargs['alpha'] = alpha

    if norm is not None:
        if norm is True:
            norm = mpl.colors.Normalize()
        plt_imshow_kwargs['norm'] = norm
    else:
        if cmap is None and not heatmap and not nospecial:
            plt_imshow_kwargs['vmin'] = 0
            plt_imshow_kwargs['vmax'] = 255
    if heatmap:
        cmap = 'hot'
    try:
        if len(img.shape) == 3 and (img.shape[2] == 3 or img.shape[2] == 4):
            # img is in a color format
            imgBGR = img

            if imgBGR.dtype == np.float64:
                if imgBGR.max() <= 1.01:
                    imgBGR = np.array(imgBGR, dtype=np.float32)
                else:
                    imgBGR = np.array(imgBGR, dtype=np.uint8)
            if imgBGR.dtype == np.float32:
                # print('[imshow] imgBGR.dtype = %r' % (imgBGR.dtype,))
                # print('[imshow] imgBGR.max() = %r' % (imgBGR.max(),))
                pass
                # imgBGR *= 255
                # if imgBGR.max() <= 1.0001:
                #    plt_imshow_kwargs['vmax'] = 1
                #    #del plt_imshow_kwargs['vmin']
                #    #del plt_imshow_kwargs['vmax']
            if img.shape[2] == 3:
                imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
                # print('plt_imshow_kwargs = %r' % (plt_imshow_kwargs,))
                ax.imshow(imgRGB, **plt_imshow_kwargs)
            else:
                imgBGRA = imgBGR
                imgRGBA = cv2.cvtColor(imgBGRA, cv2.COLOR_BGRA2RGBA)
                # print('plt_imshow_kwargs = %r' % (plt_imshow_kwargs,))
                ax.imshow(imgRGBA, **plt_imshow_kwargs)
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

            if not plt_imshow_kwargs.get('norm'):
                # for some reason gray floats aren't working right
                if imgGRAY.max() <= 1.01 and imgGRAY.min() >= -1e-9:
                    imgGRAY = (imgGRAY * 255).astype(np.uint8)

            ax.imshow(imgGRAY, cmap=cmap, **plt_imshow_kwargs)
        else:
            raise AssertionError(
                'unknown image format. img.dtype=%r, img.shape=%r'
                % (img.dtype, img.shape)
            )
    except TypeError as te:
        print('[df2] imshow ERROR %r' % (te,))
        raise
    except Exception as ex:
        print('!!!!!!!!!!!!!!WARNING!!!!!!!!!!!')
        print('[df2] type(img) = %r' % type(img))
        if not isinstance(img, np.ndarray):
            print('!!!!!!!!!!!!!!ERRROR!!!!!!!!!!!')
            pass
            # print('img = %r' % (img,))
        print('[df2] img.dtype = %r' % (img.dtype,))
        print('[df2] type(img) = %r' % (type(img),))
        print('[df2] img.shape = %r' % (img.shape,))
        print('[df2] imshow ERROR %r' % ex)
        raise
    # plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

    if data_colorbar is True:
        scores = np.unique(img.flatten())
        if cmap is None:
            cmap = 'hot'
        colors = scores_to_color(scores, cmap)
        colorbar(scores, colors)

    if xlabel is not None:
        custom_figure.set_xlabel(xlabel)

    if figtitle is not None:
        custom_figure.set_figtitle(figtitle)
    if update:
        fig_presenter.update()
    return fig, ax


def draw_vector_field(gx, gy, fnum=None, pnum=None, title=None, invert=True, stride=1):
    r"""
    CommandLine:
        python -m wbia.plottool.draw_func2 draw_vector_field --show
        python -m wbia.plottool.draw_func2 draw_vector_field --show --fname=zebra.png --fx=121 --stride=3

    Example:
        >>> # DISABLE_DOCTEST
        >>> import wbia.plottool as pt
        >>> import utool as ut
        >>> import vtool as vt
        >>> patch = vt.testdata_patch()
        >>> gx, gy = vt.patch_gradient(patch, gaussian_weighted=False)
        >>> stride = ut.get_argval('--stride', default=1)
        >>> pt.draw_vector_field(gx, gy, stride=stride)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    # https://stackoverflow.com/questions/1843194/plotting-vector-fields-in-python-matplotlib
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.quiver
    quiv_kw = {
        'units': 'xy',
        'scale_units': 'xy',
        # 'angles': 'uv',
        # 'scale': 80,
        # 'width':
        'headaxislength': 4.5,
        # 'headlength': 5,
        'headlength': 5,
        # 'headwidth': 3,
        'headwidth': 10,
        'minshaft': 1,
        'minlength': 1,
        # 'color': 'r',
        # 'edgecolor': 'k',
        'linewidths': (0.5,),
        'pivot': 'tail',  # 'middle',
    }
    # TAU = 2 * np.pi
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
    fig = figure(fnum=fnum, pnum=pnum)
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
    return fig


def show_chipmatch2(
    rchip1,
    rchip2,
    kpts1=None,
    kpts2=None,
    fm=None,
    fs=None,
    fm_norm=None,
    title=None,
    vert=None,
    fnum=None,
    pnum=None,
    heatmap=False,
    modifysize=False,
    new_return=False,
    draw_fmatch=True,
    darken=DARKEN,
    H1=None,
    H2=None,
    sel_fm=[],
    ax=None,
    heatmask=False,
    white_background=False,
    **kwargs
):
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
        fm_norm (None): (default = None)
        title (str):  (default = None)
        vert (None): (default = None)
        fnum (int):  figure number(default = None)
        pnum (tuple):  plot number(default = None)
        heatmap (bool): (default = False)
        modifysize (bool): (default = False)
        new_return (bool): (default = False)
        draw_fmatch (bool): (default = True)
        darken (None): (default = None)
        H1 (None): (default = None)
        H2 (None): (default = None)
        sel_fm (list): (default = [])
        ax (None): (default = None)
        heatmask (bool): (default = False)
        **kwargs: all_kpts, lbl1, lbl2, rect, colorbar_, draw_border, cmap,
                  scale_factor1, scale_factor2, draw_pts, draw_ell,
                  draw_lines, ell_alpha, colors

    Returns:
        tuple: (xywh1, xywh2, sf_tup)

    CommandLine:
        python -m wbia.plottool.draw_func2 show_chipmatch2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import vtool as vt
        >>> rchip1 = vt.imread(ut.grab_test_imgpath('easy1.png'))
        >>> rchip2 = vt.imread(ut.grab_test_imgpath('easy2.png'))
        >>> kpts1 = np.array([
        >>>     [10,  10,   30,   0,    30,    0.  ],
        >>>     [ 355.89,  142.95,   10.46,   -0.63,    8.59,    0.  ],
        >>>     [ 356.35,  147.  ,    8.38,    1.08,   11.68,    0.  ],
        >>>     [ 361.4 ,  150.64,    7.44,    3.45,   13.63,    0.  ]
        >>> ], dtype=np.float64)
        >>> kpts2 = np.array([
        >>>     [ 10,   10,   30,   0,    30,    0.  ],
        >>>     [ 376.98,   50.61,   11.91,   -2.9 ,    9.77,    0.  ],
        >>>     [ 377.59,   54.89,    9.7 ,   -1.4 ,   13.72,    0.  ],
        >>>     [ 382.8 ,   58.2 ,    7.87,   -0.31,   15.23,    0.  ]
        >>> ], dtype=np.float64)
        >>> fm = None
        >>> fs = None
        >>> H1 = np.array([
        >>>     [ -4.68815126e-01,   7.80306795e-02,  -2.23674587e+01],
        >>>     [  4.54394231e-02,  -7.67438835e-01,   5.92158624e+01],
        >>>     [  2.12918867e-04,  -8.64851418e-05,  -6.21472492e-01]])
        >>> H1 = None
        >>> H2 = None
        >>> #H_half = np.array([[.2, 0, 0], [0, .2, 0], [0, 0, 1]])
        >>> #H1 = H_half
        >>> #H2 = H_half
        >>> kwargs = dict(H1=H1, H2=H2, fm=fm, draw_lines=True, draw_ell=True)
        >>> kwargs.update(ell_linewidth=5, lw=10, line_alpha=[1, .3, .3, .3])
        >>> result = show_chipmatch2(rchip1, rchip2, kpts1, kpts2, **kwargs)
        >>> pt.show_if_requested()
    """
    import vtool as vt

    if ut.VERBOSE:
        print('[df2] show_chipmatch2() fnum=%r, pnum=%r, ax=%r' % (fnum, pnum, ax))
    wh1 = vt.get_size(rchip1)
    wh2 = vt.get_size(rchip2)
    if True:  # if H1 is None and H2 is not None or H2 is None and H1 is not None:
        # We are warping one chip into the space of the other
        dsize1 = wh2
        dsize2 = wh1

    if heatmask:
        from vtool.coverage_kpts import make_kpts_heatmask

        if not kwargs.get('all_kpts', False) and fm is not None:
            kpts1_m = kpts1[fm.T[0]]
            kpts2_m = kpts2[fm.T[1]]
        else:
            kpts1_m = kpts1
            kpts2_m = kpts2

        heatmask1 = make_kpts_heatmask(kpts1_m, wh1)
        heatmask2 = make_kpts_heatmask(kpts2_m, wh2)
        rchip1 = vt.overlay_alpha_images(heatmask1, rchip1)
        rchip2 = vt.overlay_alpha_images(heatmask2, rchip2)

    # Warp if homography is specified
    rchip1_ = vt.warpHomog(rchip1, H1, dsize1) if H1 is not None else rchip1
    rchip2_ = vt.warpHomog(rchip2, H2, dsize2) if H2 is not None else rchip2
    # get matching keypoints + offset
    (w1, h1) = vt.get_size(rchip1_)
    (w2, h2) = vt.get_size(rchip2_)
    # Stack the compared chips
    # modifysize = True
    match_img, offset_tup, sf_tup = vt.stack_images(
        rchip1_,
        rchip2_,
        vert,
        modifysize=modifysize,
        return_sf=True,
        white_background=white_background,
    )

    (woff, hoff) = offset_tup[1]
    xywh1 = (0, 0, w1, h1)
    xywh2 = (woff, hoff, w2, h2)
    # Show the stacked chips
    fig, ax = imshow(
        match_img,
        title=title,
        fnum=fnum,
        pnum=pnum,
        ax=ax,
        heatmap=heatmap,
        darken=darken,
    )
    # Overlay feature match nnotations
    if draw_fmatch and kpts1 is not None and kpts2 is not None:
        sf1, sf2 = sf_tup
        plot_fmatch(
            xywh1,
            xywh2,
            kpts1,
            kpts2,
            fm,
            fs,
            fm_norm=fm_norm,
            H1=H1,
            scale_factor1=sf1,
            scale_factor2=sf2,
            H2=H2,
            ax=ax,
            **kwargs
        )
        if len(sel_fm) > 0:
            # Draw any selected matches in blue
            sm_kw = dict(rect=True, colors=BLUE)
            plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, ax=ax, **sm_kw)
    if new_return:
        return xywh1, xywh2, sf_tup
    else:
        return ax, xywh1, xywh2


# plot feature match
def plot_fmatch(
    xywh1,
    xywh2,
    kpts1,
    kpts2,
    fm,
    fs=None,
    fm_norm=None,
    lbl1=None,
    lbl2=None,
    fnum=None,
    pnum=None,
    rect=False,
    colorbar_=True,
    draw_border=False,
    cmap=None,
    H1=None,
    H2=None,
    scale_factor1=None,
    scale_factor2=None,
    ax=None,
    **kwargs
):
    """
    Overlays the matching features over chips that were previously plotted.

    Args:
        xywh1 (tuple): location of rchip1 in the axes
        xywh2 (tuple): location or rchip2 in the axes
        kpts1 (ndarray): keypoints in rchip1
        kpts2 (ndarray): keypoints in rchip1
        fm (list): feature matches
        fs (list): features scores
        fm_norm (None): (default = None)
        lbl1 (None): rchip1 label
        lbl2 (None): rchip2 label
        fnum (None): figure number
        pnum (None): plot number
        rect (bool):
        colorbar_ (bool):
        draw_border (bool):
        cmap (None): (default = None)
        H1 (None): (default = None)
        H2 (None): (default = None)
        scale_factor1 (None): (default = None)
        scale_factor2 (None): (default = None)

    Kwargs:
        draw_pts, draw_ell, draw_lines, show_nMatches, all_kpts

    Returns:
        ?: None

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-plot_fmatch

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> xywh1 = '?'
        >>> xywh2 = '?'
        >>> kpts1 = '?'
        >>> kpts2 = '?'
        >>> fm = '?'
        >>> fs = None
        >>> fm_norm = None
        >>> lbl1 = None
        >>> lbl2 = None
        >>> fnum = None
        >>> pnum = None
        >>> rect = False
        >>> colorbar_ = True
        >>> draw_border = False
        >>> cmap = None
        >>> H1 = None
        >>> H2 = None
        >>> scale_factor1 = None
        >>> scale_factor2 = None
        >>> plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs, fm_norm, lbl1, lbl2,
        >>>             fnum, pnum, rect, colorbar_, draw_border, cmap, h1, h2,
        >>>             scale_factor1, scale_factor2)
        >>> result = ('None = %s' % (str(None),))
        >>> print(result)
    """
    if fm is None and fm_norm is None:
        assert kpts1.shape == kpts2.shape, 'shapes different or fm not none'
        fm = np.tile(np.arange(0, len(kpts1)), (2, 1)).T
    pts = kwargs.get('draw_pts', False)
    ell = kwargs.get('draw_ell', True)
    lines = kwargs.get('draw_lines', True)
    ell_alpha = kwargs.get('ell_alpha', 0.4)
    nMatch = len(fm)
    x2, y2, w2, h2 = xywh2
    offset1 = (0.0, 0.0)
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
    # if kwargs.get('show_nMatches', False):
    #    upperleft_text('#match=%d' % nMatch)
    # Draw all keypoints in both chips as points
    if kwargs.get('all_kpts', False):
        all_args = dict(
            ell=False,
            pts=pts,
            pts_color=GREEN,
            pts_size=2,
            ell_alpha=ell_alpha,
            rect=rect,
        )
        all_args.update(kwargs)
        draw_kpts2(kpts1, offset=offset1, H=H1, ax=ax, **all_args)
        draw_kpts2(kpts2, offset=offset2, H=H2, ax=ax, **all_args)
    if draw_border:
        draw_bbox(xywh1, bbox_color=BLACK, ax=ax, draw_arrow=False)
        draw_bbox(xywh2, bbox_color=BLACK, ax=ax, draw_arrow=False)

    if nMatch > 0:
        # draw lines and ellipses and points
        colors = (
            [kwargs['colors']] * nMatch if 'colors' in kwargs else distinct_colors(nMatch)
        )
        if fs is not None:
            if cmap is None:
                cmap = 'hot'
            colors = scores_to_color(fs, cmap)

        # acols = add_alpha(colors)

        # Helper functions
        def _drawkpts(**_kwargs):
            _kwargs.update(kwargs)
            fxs1 = fm.T[0]
            fxs2 = fm.T[1]
            if kpts1 is not None:
                draw_kpts2(
                    kpts1[fxs1],
                    offset=offset1,
                    scale_factor=scale_factor1,
                    rect=rect,
                    H=H1,
                    ax=ax,
                    **_kwargs
                )
            draw_kpts2(
                kpts2[fxs2],
                offset=offset2,
                scale_factor=scale_factor2,
                ax=ax,
                rect=rect,
                H=H2,
                **_kwargs
            )

        def _drawlines(**_kwargs):
            _kwargs.update(kwargs)
            if 'line_lw' in _kwargs:
                _kwargs['lw'] = _kwargs.pop('line_lw')
            draw_lines2(
                kpts1,
                kpts2,
                fm,
                fs,
                kpts2_offset=offset2,
                scale_factor1=scale_factor1,
                scale_factor2=scale_factor2,
                H1=H1,
                H2=H2,
                ax=ax,
                **_kwargs
            )
            if fm_norm is not None:
                # NORMALIZING MATCHES IF GIVEN
                _kwargs_norm = _kwargs.copy()
                if fs is not None:
                    cmap = 'cool'
                    colors = scores_to_color(fs, cmap)
                _kwargs_norm['color_list'] = colors
                draw_lines2(
                    kpts1,
                    kpts2,
                    fm_norm,
                    fs,
                    kpts2_offset=offset2,
                    H1=H1,
                    H2=H2,
                    scale_factor1=scale_factor1,
                    scale_factor2=scale_factor2,
                    ax=ax,
                    **_kwargs_norm
                )

        if ell:
            _drawkpts(pts=False, ell=True, color_list=colors)
        if pts:
            # TODO: just draw points with a stroke
            _drawkpts(pts_size=8, pts=True, ell=False, pts_color=BLACK)
            _drawkpts(pts_size=6, pts=True, ell=False, color_list=colors)
        if lines and kpts1 is not None:
            _drawlines(color_list=colors)
    else:
        # if not matches draw a big red X
        # draw_boxedX(xywh2)
        pass
    # Turn off colorbar if there are no features being drawn
    # or the user doesnt want a colorbar
    drew_anything = fs is not None and (ell or pts or lines)
    has_colors = nMatch > 0 and colors is not None  # 'colors' in vars()
    if drew_anything and has_colors and colorbar_:
        colorbar(fs, colors)
    # legend()
    return None


def draw_boxedX(xywh=None, color=RED, lw=2, alpha=0.5, theta=0, ax=None):
    """ draws a big red x """
    if ax is None:
        ax = gca()
    if xywh is None:
        xy, w, h = get_axis_xy_width_height(ax)
        xywh = (xy[0], xy[1], w, h)
    x1, y1, w, h = xywh
    x2, y2 = x1 + w, y1 + h
    segments = [((x1, y1), (x2, y2)), ((x1, y2), (x2, y1))]
    trans = mpl.transforms.Affine2D()
    trans.rotate(theta)
    trans = trans + ax.transData
    width_list = [lw] * len(segments)
    color_list = [color] * len(segments)
    line_group = mpl.collections.LineCollection(
        segments, width_list, color_list, alpha=alpha, transOffset=trans
    )
    ax.add_collection(line_group)


def color_orimag(gori, gmag=None, gmag_is_01=None, encoding='rgb', p=0.5):
    r"""
    Args:
        gori (ndarray): orientation values at pixels between 0 and tau
        gmag (ndarray): orientation magnitude
        gmag_is_01 (bool): True if gmag is in the 0 and 1 range. if None we try to guess
        p (float): power to raise normalized weights to for visualization purposes

    Returns:
        ndarray: rgb_ori or bgr_ori

    CommandLine:
        python -m wbia.plottool.draw_func2 --test-color_orimag --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
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
    # gori_01 = (gori - gori.min()) / (gori.max() - gori.min())
    if gori.max() > TAU or gori.min() < 0:
        print('WARNING: [color_orimag] gori might not be in radians')
    flat_rgb = get_orientation_color(gori.flatten())
    # flat_rgb = np.array(cmap_(), dtype=np.float32)
    rgb_ori_alpha = flat_rgb.reshape(np.hstack((gori.shape, [4])))
    rgb_ori = cv2.cvtColor(rgb_ori_alpha, cv2.COLOR_RGBA2RGB)
    hsv_ori = cv2.cvtColor(rgb_ori, cv2.COLOR_RGB2HSV)
    # Darken colors based on magnitude
    if gmag is not None:
        # Hueristic hack
        if gmag_is_01 is None:
            gmag_is_01 = gmag.max() <= 1.0
        gmag_ = gmag if gmag_is_01 else gmag / max(255.0, gmag.max())
        # Weights modify just value
        gmag_ = gmag_ ** p
        # SAT_CHANNEL = 1
        VAL_CHANNEL = 2
        # hsv_ori[:, :, SAT_CHANNEL] = gmag_
        hsv_ori[:, :, VAL_CHANNEL] = gmag_
    # Convert back to bgr
    # bgr_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2BGR)
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
        python -m wbia.plottool.draw_func2 --test-get_orientation_color

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
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
    # colorbar(ori_list, color_list, lbl='orientation (radians)', custom=True)
    colorbar(ori_list, color_list, lbl='radians', float_format='%.1f', custom=True)


def make_ori_legend_img():
    r"""

    creates a figure that shows which colors are associated with which keypoint
    rotations.

    a rotation of 0 should point downward (becuase it is relative the the (0, 1)
    keypoint eigenvector. and its color should be red due to the hsv mapping

    CommandLine:
        python -m wbia.plottool.draw_func2 --test-make_ori_legend_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> # build test data
        >>> # execute function
        >>> img_BGR = make_ori_legend_img()
        >>> # verify results
        >>> pt.imshow(img_BGR)
        >>> pt.iup()
        >>> pt.show_if_requested()
    """
    import wbia.plottool as pt

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
    radius = (size / 5) * ut.PHI
    # size_root = size / 4
    half_size = size / 2
    img_BGR = np.zeros((size, size, 3), dtype=np.uint8)
    basis = np.arange(-7, 7)
    x_kernel_offset, y_kernel_offset = np.meshgrid(basis, basis)
    x_kernel_offset = x_kernel_offset.ravel()
    y_kernel_offset = y_kernel_offset.ravel()
    # x_kernel_offset = np.array([0, 1,  0, -1, -1, -1,  0,  1, 1])
    # y_kernel_offset = np.array([0, 1,  1,  1,  0, -1, -1, -1, 0])
    # new_data_weight = np.ones(x_kernel_offset.shape, dtype=np.int32)
    for color_rgb, (r, c) in zip(color_rgb_list, rc_list):
        row = x_kernel_offset + int(r * radius + half_size)
        col = y_kernel_offset + int(c * radius + half_size)
        # old_data = img[row, col, :]
        color = color_rgb[0:3] * 255
        color_bgr = color[::-1]
        # img_BGR[row, col, :] = color
        img_BGR[row, col, :] = color_bgr
        # new_data = img_BGR[row, col, :]
        # old_data_weight = np.array(list(map(np.any, old_data > 0)), dtype=np.int32)
        # total_weight = old_data_weight + 1
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
        # print(row)
        # print(col)
        # print(color_rgb)
        # print(text)
        cv2.putText(
            img_BGR,
            text,
            org,
            fontFace,
            fontScale,
            textcolor,
            thickness,
            bottomLeftOrigin=False,
        )
        # img_BGR[row, col, :] = ((old_data * old_data_weight[:, None] +
        # new_data) / total_weight[:, None])
    # print(img_BGR)
    return img_BGR


def remove_patches(ax=None):
    """ deletes patches from axes """
    if ax is None:
        ax = gca()
    for patch in ax.patches:
        del patch


def imshow_null(msg=None, ax=None, **kwargs):
    r"""
    Args:
        msg (None): (default = None)
        ax (None): (default = None)
        **kwargs: fnum, title, figtitle, pnum, interpolation, cmap, heatmap,
                  data_colorbar, darken, update, xlabel, redraw_image, alpha,
                  docla, doclf, projection, use_gridspec

    CommandLine:
        python -m wbia.plottool.draw_func2 imshow_null --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> msg = None
        >>> ax = None
        >>> result = imshow_null(msg, ax)
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    if ax is None:
        ax = gca()
    subkeys = [key for key in ['fontsize'] if key in kwargs]
    print('kwargs = %r' % (kwargs,))
    kwargs_ = ut.dict_subset(kwargs, subkeys)
    print('kwargs_ = %r' % (kwargs_,))
    imshow(np.zeros((10, 10), dtype=np.uint8), ax=ax, **kwargs)
    if msg is None:
        draw_boxedX(ax=ax)
    else:
        relative_text(
            (0.5, 0.5), msg, color='r', horizontalalignment='center', ax=ax, **kwargs_
        )


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


def make_bbox_positioners(y=0.02, w=0.08, h=0.02, xpad=0.05, startx=0, stopx=1):
    def hl_slot(ix):
        x = startx + (xpad * (ix + 1)) + ix * w
        return (x, y, w, h)

    def hr_slot(ix):
        x = stopx - ((xpad * (ix + 1)) + (ix + 1) * w)
        return (x, y, w, h)

    return hl_slot, hr_slot


def width_from(num, pad=0.05, start=0, stop=1):
    return ((stop - start) - ((num + 1) * pad)) / num


# +-----
# From vtool.patch
def param_plot_iterator(param_list, fnum=None, projection=None):
    from wbia.plottool import plot_helpers

    nRows, nCols = plot_helpers.get_square_row_cols(len(param_list), fix=True)
    # next_pnum = make_pnum_nextgen(nRows=nRows, nCols=nCols)
    pnum_gen = pnum_generator(nRows, nCols)
    pnum = (nRows, nCols, 1)
    fig = figure(fnum=fnum, pnum=pnum)
    for param, pnum in zip(param_list, pnum_gen):
        # get next figure ready
        # print('fnum=%r, pnum=%r' % (fnum, pnum))
        if projection is not None:
            subplot_kw = {'projection': projection}
        else:
            subplot_kw = {}
        fig.add_subplot(*pnum, **subplot_kw)
        # figure(fnum=fnum, pnum=pnum)
        yield param


def plot_surface3d(
    xgrid,
    ygrid,
    zdata,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    wire=False,
    mode=None,
    contour=False,
    dark=False,
    rstride=1,
    cstride=1,
    pnum=None,
    labelkw=None,
    xlabelkw=None,
    ylabelkw=None,
    zlabelkw=None,
    titlekw=None,
    *args,
    **kwargs
):
    r"""
    References:
        http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-plot_surface3d --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import vtool as vt
        >>> shape=(19, 19)
        >>> sigma1, sigma2 = 2.0, 1.0
        >>> ybasis = np.arange(shape[0])
        >>> xbasis = np.arange(shape[1])
        >>> xgrid, ygrid = np.meshgrid(xbasis, ybasis)
        >>> sigma = [sigma1, sigma2]
        >>> gausspatch = vt.gaussian_patch(shape, sigma=sigma)
        >>> title = 'ksize=%r, sigma=%r' % (shape, (sigma1, sigma2),)
        >>> pt.plot_surface3d(xgrid, ygrid, gausspatch, rstride=1, cstride=1,
        >>>                   cmap=mpl.cm.coolwarm, title=title)
        >>> pt.show_if_requested()
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
        # print('pnum = %r' % (pnum,))
        ax = fig.add_subplot(*pnum, projection='3d')
    title = kwargs.pop('title', None)
    if mode is None:
        mode = 'wire' if wire else 'surface'

    if mode == 'wire':
        ax.plot_wireframe(
            xgrid, ygrid, zdata, rstride=rstride, cstride=cstride, *args, **kwargs
        )
        # ax.contour(xgrid, ygrid, zdata, rstride=rstride, cstride=cstride,
        # extend3d=True, *args, **kwargs)
    elif mode == 'surface':
        ax.plot_surface(
            xgrid,
            ygrid,
            zdata,
            rstride=rstride,
            cstride=cstride,
            linewidth=0.1,
            *args,
            **kwargs
        )
    else:
        raise NotImplementedError('mode=%r' % (mode,))
    if contour:
        import matplotlib.cm as cm

        xoffset = xgrid.min() - ((xgrid.max() - xgrid.min()) * 0.1)
        yoffset = ygrid.max() + ((ygrid.max() - ygrid.min()) * 0.1)
        zoffset = zdata.min() - ((zdata.max() - zdata.min()) * 0.1)
        cmap = kwargs.get('cmap', cm.coolwarm)
        ax.contour(xgrid, ygrid, zdata, zdir='x', offset=xoffset, cmap=cmap)
        ax.contour(xgrid, ygrid, zdata, zdir='y', offset=yoffset, cmap=cmap)
        ax.contour(xgrid, ygrid, zdata, zdir='z', offset=zoffset, cmap=cmap)
        # ax.plot_trisurf(xgrid.flatten(), ygrid.flatten(), zdata.flatten(), *args, **kwargs)
    if title is not None:
        ax.set_title(title, **titlekw)
    if xlabel is not None:
        ax.set_xlabel(xlabel, **xlabelkw)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **ylabelkw)
    if zlabel is not None:
        ax.set_zlabel(zlabel, **zlabelkw)
    use_darkbackground = dark
    # if use_darkbackground is None:
    #    use_darkbackground = not ut.get_argflag('--save')
    if use_darkbackground:
        dark_background()
    return ax


# L_____


def draw_text_annotations(
    text_list,
    pos_list,
    bbox_offset_list=[0, 0],
    pos_offset_list=[0, 0],
    bbox_align_list=[0, 0],
    color_list=None,
    textprops={},
):
    """
    Hack fixes to issues in text annotations
    """
    import wbia.plottool as pt

    artist_list = []
    offset_box_list = []

    if not isinstance(bbox_offset_list[0], (list, tuple)):
        bbox_offset_list = [bbox_offset_list] * len(text_list)
    if not isinstance(pos_offset_list[0], (list, tuple)):
        pos_offset_list = [pos_offset_list] * len(text_list)
    if not isinstance(bbox_align_list[0], (list, tuple)):
        bbox_align_list = [bbox_align_list] * len(text_list)

    ax = pt.gca()

    textkw = dict(
        xycoords='data',
        boxcoords='offset points',
        pad=0.25,
        framewidth=True,
        arrowprops=dict(arrowstyle='->', ec='black'),
        # bboxprops=dict(fc=node_attr['fillcolor']),
    )

    _iter = zip(text_list, pos_list, pos_offset_list, bbox_offset_list, bbox_align_list)
    for count, tup in enumerate(_iter):
        (text, pos, pos_offset, bbox_offset, bbox_align) = tup
        if color_list is not None:
            color = color_list[count]
        else:
            color = None
        if color is None:
            color = pt.WHITE
        x, y = pos
        dpx, dpy = pos_offset

        if text is not None:
            offset_box = mpl.offsetbox.TextArea(text, textprops)
            artist = mpl.offsetbox.AnnotationBbox(
                offset_box,
                (x + dpx, y + dpy),
                xybox=bbox_offset,
                box_alignment=bbox_align,
                bboxprops=dict(fc=color),
                **textkw
            )
            offset_box_list.append(offset_box)
            artist_list.append(artist)

    for artist in artist_list:
        ax.add_artist(artist)

    def hack_fix_centeralign():
        """
        Caller needs to call this after limits are set up
        to fixe issue in matplotlib
        """
        if textprops.get('horizontalalignment', None) == 'center':
            print('Fixing centeralign')
            fig = pt.gcf()
            fig.canvas.draw()

            # Superhack for centered text. Fix bug in
            # /usr/local/lib/python2.7/dist-packages/matplotlib/offsetbox.py
            # /usr/local/lib/python2.7/dist-packages/matplotlib/text.py
            for offset_box in offset_box_list:
                offset_box.set_offset
                z = offset_box._text.get_window_extent()
                (z.x1 - z.x0) / 2
                offset_box._text
                T = offset_box._text.get_transform()
                A = mpl.transforms.Affine2D()
                A.clear()
                A.translate((z.x1 - z.x0) / 2, 0)
                offset_box._text.set_transform(T + A)

    return hack_fix_centeralign


def set_figsize(w, h, dpi):
    fig = plt.gcf()
    fig.set_size_inches(w, h)
    fig.set_dpi(dpi)


def plot_func(funcs, start=0, stop=1, num=100, setup=None, fnum=None, pnum=None):
    r"""
    plots a numerical function in a given range

    Args:
        funcs (list of function):  live python function
        start (int): (default = 0)
        stop (int): (default = 1)
        num (int): (default = 100)

    CommandLine:
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=-1,1 --func=np.exp
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=-1,1 --func=scipy.special.logit
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=0,1 --func="lambda x: scipy.special.expit(((x * 2) - 1.0) * 6)"
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=0,1 --func="lambda x: scipy.special.expit(-6 + 12 * x)"
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=0,4 --func="lambda x: vt.logistic_01((-1 + x) * 2)"
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=0,1 --func="lambda x: np.tan((x - .5) * np.pi)" --ylim=-10,10
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=0,3 --func=np.tan
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=0,50 --func="lambda x: np.exp(-x / 50)"
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=-8,8 --func=vt.beaton_tukey_loss
        python -m wbia.plottool.draw_func2 --exec-plot_func --show --range=-8,8 --func=vt.beaton_tukey_weight,vt.beaton_tukey_loss

        python -m wbia.plottool plot_func --show --range=-1,1 \
                --setup="from wbia.algo.smk.smk_pipeline import SMK" \
                --func=lambda u: SMK.selectivity(u, 3.0, 0)

        python -m wbia.plottool plot_func --show --range=-1,1 \
                --func \
                "lambda u: sign(u) * abs(u)**3.0 * greater_equal(u, 0)" \
                "lambda u: (sign((u+1)/2) * abs((u+1)/2)**3.0 * greater_equal(u, 0+.5))"

        alpha=3
        thresh=-1

        python -m wbia.plottool plot_func --show --range=-1,1 \
                --func \
                "lambda u: sign(u) * abs(u)**$alpha * greater_equal(u, $thresh)" \
                "lambda u: (sign(u) * abs(u)**$alpha * greater_equal(u, $thresh) + 1) / 2" \
                "lambda u: sign((u+1)/2) * abs((u+1)/2)**$alpha * greater_equal(u, $thresh)"

        python -m wbia.plottool plot_func --show --range=4,100 \
                --func \
                "lambda n: log2(n)"\
                "lambda n: log2(log2(n))"\
                "lambda n: log2(n)/log2(log2(n))"\
                "lambda n: log2(n) ** 2"\
                "lambda n: n"\

        python -m wbia.plottool plot_func --show --range=4,1000000 \
                --func \
                "lambda n: log2(n)"\
                "lambda n: n ** (1/3)"

        python -m wbia.plottool plot_func --show --range=0,10 \
                --func \
                "lambda x: (3 * (x ** 2) - 18 * (x) - 81) / ((x ** 2) - 54) "

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.draw_func2 import *  # NOQA
        >>> import scipy
        >>> import scipy.special  # NOQA
        >>> func_list = ut.get_argval('--func', type_=list, default=['np.exp'])
        >>> setup = ut.get_argval('--setup', type_=str, default=None)
        >>> #funcs = [eval(f) for f in func_list]
        >>> funcs = func_list
        >>> start, stop = ut.get_argval('--range', type_=list, default=[-1, 1])
        >>> start, stop = eval(str(start)), eval(str(stop))
        >>> num = 1000
        >>> result = plot_func(funcs, start, stop, num, setup=setup)
        >>> print(result)
        >>> import plottool as pt
        >>> pt.quit_if_noshow()
        >>> ylim = ut.get_argval('--ylim', type_=list, default=None)
        >>> import wbia.plottool as pt
        >>> None if ylim is None else plt.gca().set_ylim(*ylim)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    import wbia.plottool as pt

    xdata = np.linspace(start, stop, num)
    if not ut.isiterable(funcs):
        funcs = [funcs]
    import scipy  # NOQA
    import scipy.special  # NOQA

    labels = [
        func if isinstance(func, six.string_types) else ut.get_callable_name(func)
        for func in funcs
    ]
    try:
        funcs_ = [
            eval(func) if isinstance(func, six.string_types) else func for func in funcs
        ]
        ydatas = [func(xdata) for func in funcs_]
    except NameError:
        locals_ = locals()
        if setup is not None:
            exec(setup, locals_, locals_)
        locals_.update(**np.__dict__)
        funcs_ = [
            eval(func, locals_) if isinstance(func, six.string_types) else func
            for func in funcs
        ]
        ydatas = [func(xdata) for func in funcs_]
    except Exception:
        print(ut.repr3(funcs))
        raise
    fnum = pt.ensure_fnum(fnum)
    pt.multi_plot(
        xdata, ydatas, label_list=labels, marker='', fnum=fnum, pnum=pnum
    )  # yscale='log')


def test_save():
    """
    CommandLine:
        python -m wbia.plottool.draw_func2 test_save --show
        python -m wbia.plottool.draw_func2 test_save
    """
    import wbia.plottool as pt
    import utool as ut
    from os.path import join

    fig = pt.figure(fnum=1)
    ax = pt.plt.gca()
    ax.plot([1, 2, 3], [4, 5, 7])
    dpath = ut.ensure_app_cache_dir('plottool')
    fpath = join(dpath, 'test.png')
    fig.savefig(fpath)
    return fpath


if __name__ == '__main__':
    """
    commandline:
        python -m wbia.plottool.draw_func2
        python -m wbia.plottool.draw_func2 --allexamples
        python -m wbia.plottool.draw_func2 --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
