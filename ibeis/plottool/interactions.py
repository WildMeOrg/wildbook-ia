import plottool as pt
from plottool import abstract_interaction
import plottool.interact_helpers as ih


class ExpandableInteraction(abstract_interaction.AbstractInteraction):
    def __init__(self, fnum=None, _pnumiter=None):
        self._pnumiter = _pnumiter
        self.pnum_list = []
        self.func_list = []
        if fnum is None:
            fnum = pt.next_fnum()
        self.fnum = fnum

    def append_plot(self, func):
        pnum = self._pnumiter()
        self.pnum_list.append(pnum)
        self.func_list.append(func)

    def show_page(self):
        fig = ih.begin_interaction('expandable', self.fnum)
        for pnum, func in zip(self.pnum_list, self.func_list):
            func(self.fnum, pnum)
            ax = pt.gca()
            pt.set_plotdat(ax, 'plot_func', func)
        ih.connect_callback(fig, 'button_press_event', self.onclick)

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
                func(fnum, pnum)
                fig = pt.gcf()
                pt.show_figure(fig)
