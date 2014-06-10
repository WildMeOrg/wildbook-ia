from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih
import matplotlib as mpl
from plottool import plot_helpers as ph

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[interact_qres2]')


class AbstractPagedInteraction(object):
    def __init__(self, **kwargs):
        # Initialize variables. No logic
        self.fnum            = None
        self.nIndexes        = 0  # number of total indexes
        self.start_index     = 0
        self.stop_index      = -1
        self.nPerPage        = None
        self.current_pagenum = -1
        self.nPages          = 0
        self.scope           = []  # for keeping those widgets alive!

    def append_button(self, text, divider=None, rect=None, callback=None, **kwargs):
        """ Adds a button to the current page """
        if divider is not None:
            new_ax = divider.append_axes('bottom', size='9%', pad=.05)
        if rect is not None:
            new_ax = df2.plt.axes(rect)
        new_but = mpl.widgets.Button(new_ax, text)
        if callback is not None:
            new_but.on_clicked(callback)
        ph.set_plotdat(new_ax, 'viztype', 'button')
        ph.set_plotdat(new_ax, 'text', text)
        for key, val in kwargs.iteritems():
            ph.set_plotdat(new_ax, key, val)
        # Keep buttons from losing scrop
        self.scope.append((new_but, new_ax))

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        self.scope = []

    def prepare_page(self, pagenum):
        """ Gets indexes for the pagenum ready to be displayed """
        # Set the start index
        self.start_index = pagenum * self.nPerPage
        # Clip based on nCands
        self.nDisplay = min(self.nCands - self.start_index, self.nPerPage)
        nRows, nCols = ph.get_square_row_cols(self.nDisplay)
        # Create a grid to hold nPerPage
        self.pnum_ = df2.get_pnum_func(nRows, nCols)
        printDBG('[iqr2*] r=%r, c=%r' % (nRows, nCols))
        # Adjust stop index
        self.stop_index = self.start_index + self.nDisplay
        # Clear current figure
        self.clean_scope()
        self.fig = df2.figure(fnum=self.fnum, pnum=self.pnum_(0), doclf=True, docla=True)
        ih.disconnect_callback(self.fig, 'button_press_event')
        ih.connect_callback(self.fig, 'button_press_event', self.on_figure_clicked)
        printDBG(self.fig)

    def show_page(self, pagenum=None):
        """ Displays a page of matches """
        if pagenum is None:
            pagenum = self.current_pagenum
        print('[iqr2] show page: %r' % pagenum)
        self.current_pagenum = pagenum
        self.prepare_page(pagenum)
        # Begin showing matches
        index = self.start_index
        for index in xrange(self.start_index, self.stop_index):
            self.plot_index(index, draw=False)
        self.make_hud()
        self.draw()

    def plot_index(self, index, draw=True, make_buttons=True):
        # Get index relative to the page
        px = index - self.start_index
        pnum = self.pnum_(px)
        # Setup figure
        fnum = self.fnum
        printDBG('\n<<<<  BEGIN INTERACTION >>>>')
        df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=False)

        # do plotting

        if draw:
            vh.draw()

    def make_hud(self):
        """ Creates heads up display """
        # Button positioning
        hl_slot, hr_slot = df2.make_bbox_positioners(y=.02, w=.08, h=.04,
                                                     xpad=.05, startx=0, stopx=1)
        prev_rect = hl_slot(0)
        next_rect = hr_slot(0)

        # Create buttons
        if self.current_pagenum != 0:
            self.append_button('prev', callback=self.prev_page, rect=prev_rect)
        if self.current_pagenum != self.nPages - 1:
            self.append_button('next', callback=self.next_page, rect=next_rect)

        figtitle_fmt = '''
        PageMembers: ({start_index}-{stop_index}) / {nCands}
        page {current_pagenum} / {nPages}
        '''
        # sexy: using object dict as format keywords
        figtitle = figtitle_fmt.format(**self.__dict__)
        df2.set_figtitle(figtitle)

    def next_page(self, event):
        print('next')
        self.show_page(self.current_pagenum + 1)
        pass

    def prev_page(self, event):
        self.show_page(self.current_pagenum - 1)
        pass

    def bring_to_front(self):
        df2.bring_to_front(self.fig)

    def draw(self):
        self.fig.canvas.draw()

    def show(self):
        self.draw()
        self.bring_to_front()
