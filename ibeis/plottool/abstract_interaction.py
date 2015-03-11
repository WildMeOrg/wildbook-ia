from __future__ import absolute_import, division, print_function
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
import six
import utool as ut
import plottool.draw_func2 as df2
from plottool import fig_presenter
import matplotlib as mpl
ut.noinject(__name__, '[abstract_iteract]')

#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[abstract_iteract]')


# for scoping
__REGISTERED_INTERACTIONS__ = []


def register_interaction(self):
    global __REGISTERED_INTERACTIONS__
    if ut.VERBOSE:
        print('Registering intearction: self=%r' % (self,))
    __REGISTERED_INTERACTIONS__.append(self)


def unregister_interaction(self):
    global __REGISTERED_INTERACTIONS__
    if ut.VERBOSE:
        print('Unregistering intearction: self=%r' % (self,))
    __REGISTERED_INTERACTIONS__


class AbstractInteraction(object):
    def __init__(self, **kwargs):
        self.fnum = kwargs.get('fnum', None)
        if self.fnum  is None:
            self.fnum  = df2.next_fnum()
        self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)
        ih.connect_callback(self.fig, 'close_event', self.on_close)
        # Careful this might cause memory leaks
        self.scope = []  # for keeping those widgets alive!
        register_interaction(self)

    def clear_parent_axes(self, ax):
        """ for clearing axes that we appended anything to """
        child_axes = ph.get_plotdat(ax, 'child_axes', [])
        ph.set_plotdat(ax, 'child_axes', [])
        for subax in child_axes:
            to_remove = None
            for tup in self.scope:
                if tup[1] is subax:
                    to_remove = tup
                    break
            if to_remove is not None:
                self.scope.remove(to_remove)
            subax.cla()
            self.fig.delaxes(subax)
        ph.del_plotdat(ax, df2.DF2_DIVIDER_KEY)
        ax.cla()

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        self.scope = []

    def append_button(self, text, divider=None, rect=None, callback=None,
                      size='9%', location='bottom', ax=None, **kwargs):
        """ Adds a button to the current page """

        if rect is not None:
            new_ax = df2.plt.axes(rect)
        if rect is None and divider is None:
            if ax is None:
                ax = df2.gca()
            divider = df2.ensure_divider(ax)
        if divider is not None:
            new_ax = divider.append_axes(location, size=size, pad=.05)
        new_but = mpl.widgets.Button(new_ax, text)
        if callback is not None:
            new_but.on_clicked(callback)
        ph.set_plotdat(new_ax, 'viztype', 'button')
        ph.set_plotdat(new_ax, 'text', text)
        #ph.set_plotdat(new_ax, 'parent_axes', ax)
        if ax is not None:
            child_axes = ph.get_plotdat(ax, 'child_axes', [])
            child_axes.append(new_ax)
            ph.set_plotdat(ax, 'child_axes', child_axes)
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

    def on_close(self, event=None):
        print('[pt] handling close')
        unregister_interaction(self)
