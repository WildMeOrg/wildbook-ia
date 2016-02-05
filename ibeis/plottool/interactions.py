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
    """
    def __init__(self, fnum=None, _pnumiter=None, interactive=None, **kwargs):
        self.nRows = kwargs.get('nRows', None)
        self.nCols = kwargs.get('nCols', None)
        self._pnumiter = _pnumiter
        self.pnum_list = []
        self.interactive = interactive
        self.ishow_func_list = []
        self.func_list = []
        if fnum is None:
            fnum = pt.next_fnum()
        self.fnum = fnum

        autostart = False
        super(ExpandableInteraction, self).__init__(autostart=autostart, **kwargs)

    def append_plot(self, func, extra=None, pnum=None, ishow_func=None, px=None):
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

    def show_page(self):
        import plottool as pt
        fig = ih.begin_interaction('expandable', self.fnum)
        if not any(self.pnum_list) and self.nRows is None and self.nRows is None:
            # Hack if no pnum was given
            self.nRows, self.nCols = pt.get_square_row_cols(len(self.pnum_list))
            pnum_ = pt.make_pnum_nextgen(self.nRows, self.nCols)
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
