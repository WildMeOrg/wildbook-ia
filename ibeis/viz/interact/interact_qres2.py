from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import plottool.draw_func2 as df2
# IBEIS
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih
from ibeis.viz import viz_matches
from ibeis.viz.interact.interact_matches import ishow_matches
from ibeis.viz.interact.interact_sver import ishow_sver
from ibeis.dev import results_organizer
import matplotlib as mpl
# mpl.widgets.Button
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
import utool
import cv2

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[interact_qres2]')


class Interact_QueryResult(object):
    def __init__(self, ibs, qrid2_qres, **kwargs):
        self.interactkw = {
            'draw_fmatches': False,
            'draw_ell': False,
            'draw_rect': False,
            'draw_lines': False,
        }
        self.ibs = ibs
        self.nCandidates = 0
        self.qrid2_qres = {}
        self.cand_match_list = []
        self.fnum = 512
        self.nPerPage = 6
        self.ranks_lt = 3
        self.start_index = 0
        self.current_buttons = {}
        self.stop_index  = self.start_index + self.nPerPage
        self.init_candidates(qrid2_qres)
        self.show_page()

    def init_candidates(self, qrid2_qres):
        self.qrid2_qres = qrid2_qres
        get_candidates = results_organizer.get_automatch_candidates
        self.cand_match_list = get_candidates(self.qrid2_qres,
                                              ranks_lt=self.ranks_lt,
                                              directed=False)
        #utool.embed()
        (qrids, rids, scores, ranks) = self.cand_match_list
        self.qrids = qrids
        self.rids = rids
        self.nCandidates = len(self.qrids)
        #if self.nCandidates > 0:
        #    index = 0
        #    self.select_candidate_match(index)

    def select_candidate_match(self, index):
        #if not utool.isiterable(index_list):
        #    index = index_list
        #if index < 0 or index >= len(self.cand_match_list): raise AssertionError('no results')
            #return None
        (qrid, rid, rank, score) = [list_[index] for list_ in self.cand_match_list]
        self.current_match_rids = (self.qrids[index], self.rids[index])
        self.current_qres = self.qrid2_qres[qrid]

    def show_page(self):
        nLeft = self.nCandidates - self.start_index
        nDisplay = min(nLeft, self.nPerPage)
        self.nDisplay = nDisplay
        nRows, nCols = ph.get_square_row_cols(nDisplay)
        print('[viz*] r=%r, c=%r' % (nRows, nCols))
        self.pnum_ = df2.get_pnum_func(nRows, nCols)
        self.stop_index = self.start_index + nDisplay
        fig = df2.figure(fnum=self.fnum, pnum=self.pnum_(0), doclf=True, docla=True)
        printDBG(fig)
        index = 0
        for index in xrange(self.start_index, self.stop_index):
            # Clear the figure for the new page of data
            self.show_match(index, draw=False)

    def show_match(self, index, draw=True):
        printDBG('[ishow_qres] starting interaction')
        self.select_candidate_match(index)
        pnum = self.pnum_(index)
        #fnum = df2.kwargs_fnum(kwargs)
        #printDBG('[inter] starting %s interaction' % type_)
        # Setup figure
        fnum = self.fnum
        printDBG('\n<<<<  BEGIN %s INTERACTION >>>>' % (str('qres').upper()))
        self.fig = fig = df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=False)
        printDBG(fig)
        #self.ax = ax = df2.gca()
        # Get viz params
        qres = self.current_qres
        rid1, rid2 = self.current_match_rids
        ibs = self.ibs
        kwargs = self.interactkw
        # Vizualize
        ax, _0, _1  = viz_matches.show_matches(ibs, qres, rid2, self_fm=[], fnum=fnum, pnum=pnum, **kwargs)
        #(x1, y1), (x2, y2) = ax.get_position().get_points()
        #(x1, y1, x2, y2) = divider.get_position()
        #butsize1 = ([0.75, 0.025, 0.15, 0.075])

        divider = df2.make_axes_locatable(ax)

        name1, name2 = ibs.get_roi_names([rid1, rid2])
        truth = vh.get_match_truth(self.ibs, rid1, rid2)

        ax1 = divider.append_axes('bottom', size='5%', pad=0.05)
        but1 = mpl.widgets.Button(ax1, 'same')
        ax2 = divider.append_axes('bottom', size='5%', pad=0.05)
        but2 = mpl.widgets.Button(ax2, 'different')

        self.current_buttons[index] = ([but1, but2], [ax1, ax2])

        #divider = df2.make_axes_locatable(ax)
        #ax.yes_but_ax = divider.append_axes('bottom', size='5%', pad=0.05)
        #ax.next_but = mpl.widgets.Button(ax.yes_but_ax, 'different')
        #df2.iup()
        if draw:
            vh.draw()
        #ih.connect_callback(self.fig, 'button_press_event', self.on_match_clicked)
        #printDBG('[ishow_qres] Finished')
        return self.fig

    def mark_as_same(self, event):
        print(event)
        pass

    def mark_as_different(self, event):
        print(event)
        pass

    def on_match_clicked(self, event):
        """ Clicked a match between query roi and result roi:
            parses the type of click it was and execute the correct
            visualiztion
        """
        print('[viz] clicked result')
        if ih.clicked_outside_axis(event):
            self.view_top_matches(toggle=1)
        else:
            ax = event.inaxes
            viztype = vh.get_ibsdat(ax, 'viztype', '')
            #printDBG(str(event.__dict__))
            printDBG('viztype=%r' % viztype)
            # Clicked a specific matches
            if viztype.startswith('matches'):
                rid2 = vh.get_ibsdat(ax, 'rid2', None)
                # Ctrl-Click
                key = '' if event.key is None else event.key
                print('key = %r' % key)
                if key.find('control') == 0:
                    print('[viz] result control clicked')
                    self.on_ctrl_clicked_rid(rid2)
                # Left-Click
                else:
                    print('[viz] result clicked')
                    self.on_clicked_rid(rid2)

    def on_ctrl_clicked_rid(self, rid2):
        """ HELPER:  Executed when a result ROI is control-clicked """
        printDBG('ctrl+clicked rid2=%r' % rid2)
        fnum_ = df2.next_fnum()
        ishow_sver(self.ibs, self._qres.qrid, rid2, fnum=fnum_)
        self.fig.canvas.draw()
        df2.bring_to_front(self.fig)

    def on_clicked_rid(self, rid2):
        """ HELPER: Executed when a result ROI is clicked """
        printDBG('clicked rid2=%r' % rid2)
        fnum_ = df2.next_fnum()
        ishow_matches(self.ibs, self._qres, rid2, fnum=fnum_)
        self.fig = df2.gcf()
        self.fig.canvas.draw()
        df2.bring_to_front(self.fig)

    def view_top_matches(self, toggle=0):
        """  HELPER: Displays query chip, groundtruth matches, and top 5 matches"""
        # Toggle if the click is not in any axis
        printDBG('clicked none')
        kwargs = self.interactkw
        kwargs['annote_mode'] = kwargs.get('annote_mode', 0) + toggle
        self.fig = viz.show_qres(self.ibs, self._qres, **kwargs)
        return self.fig
