from __future__ import absolute_import, division, print_function
from plottool import plot_helpers as ph
import six
import utool as ut
import plottool.draw_func2 as df2
from plottool import fig_presenter
import matplotlib as mpl
ut.noinject(__name__, '[abstract_iteract]')

#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[abstract_iteract]')


class AbstractInteraction(object):
    def __init__(self, **kwargs):
        self.fnum            = kwargs.get('fnum', None)
        if self.fnum  is None:
            self.fnum  = df2.next_fnum()
        self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)
        self.scope = []  # for keeping those widgets alive!

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        self.scope = []

    def append_button(self, text, divider=None, rect=None, callback=None,
                      size='9%', **kwargs):
        """ Adds a button to the current page """
        if divider is not None:
            new_ax = divider.append_axes('bottom', size=size, pad=.05)
        if rect is not None:
            new_ax = df2.plt.axes(rect)
        new_but = mpl.widgets.Button(new_ax, text)
        if callback is not None:
            new_but.on_clicked(callback)
        ph.set_plotdat(new_ax, 'viztype', 'button')
        ph.set_plotdat(new_ax, 'text', text)
        for key, val in six.iteritems(kwargs):
            ph.set_plotdat(new_ax, key, val)
        # Keep buttons from losing scrop
        tup = (new_but, new_ax)
        self.scope.append(tup)
        return tup

    # def make_hud(self):

    # def prepare_page(self, pagenum):

    # def show_page(self, *args):

    def bring_to_front(self):
        fig_presenter.bring_to_front(self.fig)

    def draw(self):
        self.fig.canvas.draw()

    def show(self):
        self.fig.show()

    def update(self):
        fig_presenter.update()

    def close(self):
        assert isinstance(self.fig, mpl.figure.Figure)
        fig_presenter.close_figure(self.fig)
