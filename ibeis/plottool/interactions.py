# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import plottool as pt
from plottool import abstract_interaction
import plottool.interact_helpers as ih


def check_if_subinteract(func):
    try:
        print('Checking if subinteraction')
        print('func = %r' % (func,))
        is_sub = issubclass(
            func, abstract_interaction.AbstractInteraction)
    except TypeError:
        is_sub = False
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
    def __init__(self, fnum=None, _pnumiter=None, interactive=None):
        self._pnumiter = _pnumiter
        self.pnum_list = []
        self.interactive = interactive
        self.func_list = []
        if fnum is None:
            fnum = pt.next_fnum()
        self.fnum = fnum

    def append_plot(self, func, extra=None):
        pnum = self._pnumiter()
        self.pnum_list.append(pnum)
        self.func_list.append(func)

    def show_page(self):
        fig = ih.begin_interaction('expandable', self.fnum)
        for pnum, func in zip(self.pnum_list, self.func_list):
            if check_if_subinteract(func):
                # Hack
                func.static_plot(self.fnum, pnum)
            else:
                func(self.fnum, pnum)
            ax = pt.gca()
            pt.set_plotdat(ax, 'plot_func', func)
        if self.interactive is None or self.interactive:
            ih.connect_callback(fig, 'button_press_event', self.onclick)
        self.fig = fig
        return fig

    def onclick(self, event):
        print('[inter] clicked in expandable interact')
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            func = pt.get_plotdat(ax, 'plot_func', None)
            print('func = %r' % (func,))
            if func is not None:
                print('calling func = %r' % (func,))
                fnum = pt.next_fnum()
                #pt.figure(fnum=fnum)
                pnum = (1, 1, 1)
                if not check_if_subinteract(func):
                    func(fnum, pnum)
                else:
                    inter = func(fnum=fnum)
                    inter.show_page()
                fig = pt.gcf()
                pt.show_figure(fig)
                #extra
