from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[abstract_iteract]')


class AbstractInteraction(object):
    def __init__(self, **kwargs):
        self.fnum            = kwargs.get('fnum', None)
        if self.fnum  is None:
            self.fnum  = df2.next_fnum()
        self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)
        self.scope           = []  # for keeping those widgets alive!

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        self.scope = []

    # def make_hud(self):

    # def prepare_page(self, pagenum):

    # def show_page(self, *args):

    def bring_to_front(self):
        df2.bring_to_front(self.fig)

    def draw(self):
        self.fig.canvas.draw()

    def show(self):
        self.draw()
        self.bring_to_front()

    def update(self):
        df2.update()
