from __future__ import absolute_import, division, print_function
#from ibeis.viz.interact.interact_matches import ishow_matches
from six.moves import range
import functools
import six
from collections import OrderedDict as odict
import utool as ut
import vtool as vt
from plottool import interact_helpers as ih
from plottool import plot_helpers as ph
import matplotlib as mpl
import plottool.draw_func2 as df2
from ibeis import ibsfuncs
from ibeis.dev import results_organizer
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_matches
from ibeis.viz.interact.interact_sver import ishow_sver

(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[interact_qres2]')


BREAK_MATCH_PREF = 'break match'
NEW_MATCH_PREF   = 'new match'
RENAME1_PREF     = 'rename query: '
RENAME2_PREF     = 'rename result: '


def default_interact_qres_params():
    params = {
        'fnum'               : 512,
        'nPerPage'           : 6,
        'ranks_lt'           : 3,
        'on_change_callback' : None
    }
    return params


class Interact_QueryResult(object):
    def __init__(self, ibs, qaid2_qres, **kwargs):
        # Initialize variables. No logic
        self.fnum               = None
        self.nPerPage           = None
        self.ranks_lt           = None
        self.on_change_callback = None
        self.ibs = None
        self.nCands = 0  # number of candidate matches
        self.qaid2_qres = {}
        self.cand_match_list = []
        self.start_index = 0
        self.current_pagenum = -1
        self.current_match_aids = None
        self.current_qres       = None
        self.scope = []  # for keeping those widgets alive!
        self.nPages = 0
        self.stop_index  = -1
        self.interactkw = {
            'draw_fmatches': False,
            'draw_ell': True,
            'draw_rect': True,
            'draw_lines': True,
            'in_image': False,
            'draw_lbl': True,
            'show_timedelta': False,
        }
        self.toggleable_kws = odict([
            ('TOG: fmatch', 'draw_fmatches'),
            ('TOG: in_image', 'in_image'),
            ('TOG: timedelta', 'show_timedelta'),
            ('TOG: lbl', 'draw_lbl'),
        ])
        # Initialize Logic
        # main data
        self.ibs = ibs
        self.qaid2_qres = qaid2_qres
        # update keyword args
        params = default_interact_qres_params()
        ut.updateif_haskey(params, kwargs)
        self.__dict__.update(**params)
        # initialize matches
        self.init_candidates(qaid2_qres)
        # show first page
        self.show_page(0)

    def get_default_params(self):
        return default_interact_qres_params()

    def init_candidates(self, qaid2_qres):
        self.qaid2_qres = qaid2_qres
        get_candidates = results_organizer.get_automatch_candidates
        self.cand_match_list = get_candidates(self.qaid2_qres,
                                              ranks_lt=self.ranks_lt,
                                              directed=False)
        (qaids, aids, scores, ranks) = self.cand_match_list
        self.qaids = qaids
        self.aids = aids
        self.nCands = len(self.qaids)
        self.nPages = vt.iceil(self.nCands / self.nPerPage)
        #if self.nCands > 0:
        #    index = 0
        #    self.select_candidate_match(index)

    def select_candidate_match(self, index):
        #if not ut.isiterable(index_list):
        #    index = index_list
        #if index < 0 or index >= len(self.cand_match_list): raise AssertionError('no results')
            #return None
        (qaid, aid, rank, score) = [list_[index] for list_ in self.cand_match_list]
        self.current_match_aids = (self.qaids[index], self.aids[index])
        self.current_qres       = self.qaid2_qres[qaid]

    def append_button(self, text, divider=None, rect=None, callback=None, size='9%', **kwargs):
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
        for key, val in six.iteritems(kwargs):
            ph.set_plotdat(new_ax, key, val)
        # Keep buttons from losing scrop
        self.scope.append((new_but, new_ax))

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        #for (but, ax) in self.scope:
        #    but.disconnect_events()
        #    ax.set_visible(False)
        #    assert len(ax.callbacks.callbacks) == 0
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
        for index in range(self.start_index, self.stop_index):
            self.plot_annotationmatch(index, draw=False)
        self.make_hud()
        self.draw()

    def plot_annotationmatch(self, index, draw=True, make_buttons=True):
        printDBG('[ishow_qres] starting interaction')
        self.select_candidate_match(index)
        # Get index relative to the page
        px = index - self.start_index
        pnum = self.pnum_(px)
        #printDBG('[inter] starting %s interaction' % type_)
        # Setup figure
        fnum = self.fnum
        printDBG('\n<<<<  BEGIN %s INTERACTION >>>>' % (str('qres').upper()))
        fig = df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=False)
        printDBG(fig)
        #self.ax = ax = df2.gca()
        # Get viz params
        qres = self.current_qres
        aid1, aid2 = self.current_match_aids
        ibs = self.ibs
        kwargs = self.interactkw
        # Vizualize
        ax = viz_matches.show_matches(ibs, qres, aid2, self_fm=[], fnum=fnum,
                                      pnum=pnum, **kwargs)[0]

        divider = df2.ensure_divider(ax)

        name1, name2 = ibs.get_annot_names([aid1, aid2])
        #truth = self.ibs.get_match_truth(aid1, aid2)

        if make_buttons:
            butkw = {
                'divider': divider,
                'callback': self.match_reviewed,
                'index': index,
            }
            if name1 == name2 and not name1.startswith('____'):
                self.append_button(BREAK_MATCH_PREF, **butkw)
            else:
                if not name1.startswith('____'):
                    self.append_button(RENAME2_PREF + name1, **butkw)
                if not name2.startswith('____'):
                    self.append_button(RENAME1_PREF + name2, **butkw)
                if name1.startswith('____') and name2.startswith('____'):
                    self.append_button(NEW_MATCH_PREF, **butkw)

        if draw:
            vh.draw()

    def make_hud(self):
        """ Creates heads up display """
        # Button positioning
        nToggle = len(self.toggleable_kws)

        # horizontal left, horizonal right
        hl_slot, hr_slot = df2.make_bbox_positioners(y=.02, w=.08, h=.04,
                                                     xpad=.05, startx=0, stopx=1)
        prev_rect = hl_slot(0)  # left button
        next_rect = hr_slot(0)  # right button

        tw = df2.width_from(nToggle, pad=.05, start=.13, stop=.87)
        hlt_slot, hrt_slot = df2.make_bbox_positioners(y=.02, w=tw, h=.04,
                                                       xpad=.05, startx=.13,
                                                       stopx=.87)

        # Create buttons
        if self.current_pagenum != 0:
            self.append_button('prev', callback=self.prev_page, rect=prev_rect)
        if self.current_pagenum != self.nPages - 1:
            self.append_button('next', callback=self.next_page, rect=next_rect)
        for count, (text, keyword) in enumerate(six.iteritems(self.toggleable_kws)):
            callback = functools.partial(self.toggle_kw, keyword=keyword)
            rect = hlt_slot(count)
            self.append_button(text, callback=callback, rect=rect)

        figtitle_fmt = '''
        Match Candidates ({start_index}-{stop_index}) / {nCands}
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

    def toggle_kw(self, event, keyword=None):
        print('toggle %r' % keyword)
        self.interactkw[keyword] = not self.interactkw[keyword]
        self.show_page()

    def match_reviewed(self, event):
        ax = event.inaxes
        viztype = ph.get_plotdat(ax, 'viztype', '')
        assert viztype == 'button', 'bad mpl button slot'
        # The change name button was clicked
        index = ph.get_plotdat(ax, 'index', -1)
        text  = ph.get_plotdat(ax, 'text', -1)
        self.select_candidate_match(index)
        aid1, aid2 = self.current_match_aids
        print(index)
        print(text)
        ibs = self.ibs
        if text.startswith(BREAK_MATCH_PREF):
            ibs.set_annot_names([aid1, aid2], ['____', '____'])
        elif text.startswith(NEW_MATCH_PREF):
            next_name = ibsfuncs.make_next_name(ibs)
            ibs.set_annot_names([aid1, aid2], [next_name, next_name])
        elif text.startswith(RENAME1_PREF):
            name2 = ibs.get_annot_names(aid2)
            ibs.set_annot_names([aid1], [name2])
        elif text.startswith(RENAME2_PREF):
            name1 = ibs.get_annot_names(aid1)
            ibs.set_annot_names([aid2], [name1])
        # Emit that something has changed
        self.on_change_callback()
        self.show_page()

    def on_figure_clicked(self, event):
        """ Clicked a match between query annotation and result annotation:
            parses the type of click it was and execute the correct
            visualiztion
        """
        print('[viz] clicked result')
        if ih.clicked_outside_axis(event):
            #self.toggle_fmatch()
            pass
        else:
            ax = event.inaxes
            viztype = ph.get_plotdat(ax, 'viztype', '')
            #printDBG(str(event.__dict__))
            printDBG('viztype=%r' % viztype)
            # Clicked a specific matches
            if viztype == 'matches':
                aid1 = ph.get_plotdat(ax, 'aid1', None)
                aid2 = ph.get_plotdat(ax, 'aid2', None)
                # Ctrl-Click
                key = '' if event.key is None else event.key
                print('key = %r' % key)
                if key.find('control') == 0:
                    print('[viz] result control clicked')
                    self.on_ctrl_clicked_match(aid1, aid2)
                # Left-Click
                else:
                    print('[viz] result clicked')
                    self.on_clicked_match(aid1, aid2)

    def on_ctrl_clicked_match(self, aid1, aid2):
        """ HELPER:  Executed when a result ANNOTATION is control-clicked """
        printDBG('ctrl+clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        ishow_sver(self.ibs, aid1, aid2, fnum=fnum_)
        fig = df2.gcf()
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def on_clicked_match(self, aid1, aid2):
        """ HELPER: Executed when a result ANNOTATION is clicked """
        printDBG('clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        qres = self.qaid2_qres[aid1]
        qres.ishow_matches(self.ibs, aid2, fnum=fnum_)
        fig = df2.gcf()
        fig.canvas.draw()
        df2.bring_to_front(fig)
        #self.draw()
        #self.bring_to_front()

    def bring_to_front(self):
        df2.bring_to_front(self.fig)

    def draw(self):
        self.fig.canvas.draw()

    def show(self):
        self.draw()
        self.bring_to_front()


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_qres2
        python -m ibeis.viz.interact.interact_qres2 --allexamples
        python -m ibeis.viz.interact.interact_qres2 --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
