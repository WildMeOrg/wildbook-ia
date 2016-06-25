# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import plottool as pt
from plottool import abstract_interaction
import plottool.interact_helpers as ih


def check_if_subinteract(func):
    try:
        if ut.VERBOSE:
            print('Checking if subinteraction')
            print('func = %r' % (func,))
        is_sub = issubclass(
            func, abstract_interaction.AbstractInteraction)
    except TypeError:
        is_sub = False
    if ut.VERBOSE:
        if is_sub:
            print('... yup')
        else:
            print('... nope')

    return is_sub


class ExpandableInteraction(abstract_interaction.AbstractInteraction):
    """
    Append a list of functions that draw plots and this interaction will plot
    them in appropriate subplots and let you click on them to zoom in.

    CommandLine:
        python -m plottool.interactions --exec-ExpandableInteraction --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.interactions import *  # NOQA
        >>> import numpy as np
        >>> import plottool as pt
        >>> inter = pt.interactions.ExpandableInteraction()
        >>> inter.append_plot(ut.partial(pt.plot_func, np.sin, stop=np.pi * 2))
        >>> inter.append_plot(ut.partial(pt.plot_func, np.cos, stop=np.pi * 2))
        >>> inter.append_plot(ut.partial(pt.plot_func, np.tan, stop=np.pi * 2))
        >>> inter.start()
        >>> pt.show_if_requested()
    """
    def __init__(self, fnum=None, _pnumiter=None, interactive=None, **kwargs):
        self.nRows = kwargs.get('nRows', None)
        self.nCols = kwargs.get('nCols', None)
        self._pnumiter = _pnumiter
        self.pnum_list = []
        self.interactive = interactive
        self.ishow_func_list = []
        self.func_list = []
        self.fnum = pt.ensure_fnum(fnum)
        self.fig = None
        autostart = False
        super(ExpandableInteraction, self).__init__(autostart=autostart, **kwargs)

    def append_plot(self, func, pnum=None, ishow_func=None, px=None):
        """
        Func must take fnum and pnum as inputs
        """
        if pnum is None:
            if px is not None:
                if isinstance(px, tuple):
                    rx, cx = px
                    px = (rx * self.nCols) + cx + 1
                pnum = (self.nRows, self.nCols, px)
            else:
                if self._pnumiter is None:
                    pnum = None
                else:
                    pnum = self._pnumiter()
        self.pnum_list.append(pnum)
        self.func_list.append(func)
        self.ishow_func_list.append(ishow_func)

    def append_partial(self, func, *args, **kwargs):
        """
        Func must take fnum and pnum as inputs
        """
        pnum = None
        if pnum is None:
            if self._pnumiter is None:
                pnum = None
            else:
                pnum = self._pnumiter()
        def _partial(fnum=None, pnum=None):
            pt.figure(fnum=fnum, pnum=pnum)
            func(*args, **kwargs)
        self.pnum_list.append(pnum)
        self.func_list.append(_partial)
        self.ishow_func_list.append(None)

    def show_page(self):
        if self.fig is None:
            raise AssertionError('fig is None, did you run interction.start()?')
        import plottool as pt
        fig = ih.begin_interaction('expandable', self.fnum)
        if not any(self.pnum_list) and self.nRows is None and self.nRows is None:
            # Hack if no pnum was given
            self.nRows, self.nCols = pt.get_num_rc(len(self.pnum_list),
                                                   nRows=self.nRows,
                                                   nCols=self.nCols)
            nSubplots = len(self.func_list)
            pnum_ = pt.make_pnum_nextgen(self.nRows, self.nCols, nSubplots=nSubplots)
            self.pnum_list = [pnum_() for _ in self.pnum_list]

        for index, (pnum, func) in enumerate(zip(self.pnum_list, self.func_list)):
            if check_if_subinteract(func):
                # Hack
                interclass = func
                interclass.static_plot(fnum=self.fnum, pnum=pnum)
            elif hasattr(func, 'plot'):
                inter = func
                inter.plot(fnum=self.fnum, pnum=pnum)
            else:
                func(fnum=self.fnum, pnum=pnum)
            ax = pt.gca()
            pt.set_plotdat(ax, 'plot_func', func)
            pt.set_plotdat(ax, 'expandable_index', index)
        #if self.interactive is None or self.interactive:
        #    ih.connect_callback(fig, 'button_press_event', self.onclick)
        self.connect_callbacks()
        self.fig = fig
        return fig

    def on_click(self, event):
        print('[inter] clicked in expandable interact')
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            func = pt.get_plotdat(ax, 'plot_func', None)
            if ut.VERBOSE:
                print('func = %r' % (func,))
            if func is not None:
                if ut.VERBOSE:
                    print('calling func = %r' % (func,))
                fnum = pt.next_fnum()
                #pt.figure(fnum=fnum)
                pnum = (1, 1, 1)
                index = pt.get_plotdat(ax, 'expandable_index', None)
                if index is not None:
                    ishow_func = self.ishow_func_list[index]
                else:
                    ishow_func = None
                if ishow_func is not None:
                    inter = ishow_func(fnum=fnum)
                else:
                    if check_if_subinteract(func):
                        inter = func(fnum=fnum)
                        inter.show_page()
                    elif hasattr(func, 'plot'):
                        inter = func
                        inter.start()
                        #func.plot(fnum=self.fnum, pnum=pnum)
                    else:
                        func(fnum=fnum, pnum=pnum)
                    #inter.show_page()
                fig = pt.gcf()
                pt.show_figure(fig)
                #extra


def zoom_factory(ax=None, zoomable_list=[], base_scale=1.1):
    """
    References:
        https://gist.github.com/tacaswell/3144287
        http://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
    """
    if ax is None:
        ax = pt.gca()
    def zoom_fun(event):
        #print('zooming')
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if xdata is None or ydata is None:
            return
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            raise NotImplementedError('event.button=%r' % (event.button,))
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        for zoomable in zoomable_list:
            zoom = zoomable.get_zoom()
            new_zoom = zoom / (scale_factor ** (1.2))
            zoomable.set_zoom(new_zoom)
        # Get distance from the cursor to the edge of the figure frame
        x_left = xdata - cur_xlim[0]
        x_right = cur_xlim[1] - xdata
        y_top = ydata - cur_ylim[0]
        y_bottom = cur_ylim[1] - ydata
        ax.set_xlim([xdata - x_left * scale_factor, xdata + x_right * scale_factor])
        ax.set_ylim([ydata - y_top * scale_factor, ydata + y_bottom * scale_factor])

        # ----
        ax.figure.canvas.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    #return the function
    return zoom_fun


def pan_factory(ax):
    self = PanEvents(ax)
    ax = self.ax
    fig = ax.get_figure()  # get the figure of interest
    self.cidBP = fig.canvas.mpl_connect('button_press_event', self.pan_on_press)
    self.cidBR = fig.canvas.mpl_connect('button_release_event', self.pan_on_release)
    self.cidBM = fig.canvas.mpl_connect('motion_notify_event', self.pan_on_motion)
    # attach the call back
    return self


class PanEvents(object):
    def __init__(self, ax=None):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        self.xzoom = True
        self.yzoom = True
        self.cidBP = None
        self.cidBR = None
        self.cidBM = None
        self.cidKeyP = None
        self.cidKeyR = None
        self.cidScroll = None
        self.ax = ax
        #if ax is None:
        #    import plottool as pt
        #    ax = pt.gca()
        #self.ax = ax
        #self.connect()

    def pan_on_press(self, event):
        ax = self.ax
        if event.inaxes != ax:
            return
        self.cur_xlim = ax.get_xlim()
        self.cur_ylim = ax.get_ylim()
        self.press = self.x0, self.y0, event.xdata, event.ydata
        self.x0, self.y0, self.xpress, self.ypress = self.press

    def pan_on_release(self, event):
        ax = self.ax
        self.press = None
        ax.figure.canvas.draw()

    def pan_on_motion(self, event):
        ax = self.ax
        if self.press is None:
            return
        if event.inaxes != ax:
            return
        dx = event.xdata - self.xpress
        dy = event.ydata - self.ypress
        self.cur_xlim -= dx
        self.cur_ylim -= dy
        ax.set_xlim(self.cur_xlim)
        ax.set_ylim(self.cur_ylim)

        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m plottool.interactions
        python -m plottool.interactions --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
