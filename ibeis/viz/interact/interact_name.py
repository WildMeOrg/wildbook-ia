from __future__ import absolute_import, division, print_function
import utool
from plottool import interact_helpers as ih
from plottool import draw_func2 as df2
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from ibeis.dev import ibsfuncs
from plottool.abstract_interaction import AbstractInteraction
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)

from ibeis.viz import viz_chip
plot_chip = viz_chip.show_chip


#==========================
# Name Interaction
#==========================


def ishow_name(ibs, nid, sel_rids=[], select_rid_callback=None, fnum=5, **kwargs):
    fig = ih.begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            print_(' viztype=%r' % viztype)
            if viztype == 'chip':
                rid = vh.get_ibsdat(ax, 'rid')
                print('... rid=%r' % rid)
                viz.show_name(ibs, nid, fnum=fnum, sel_rids=[rid])
                if select_rid_callback is not None:
                    select_rid_callback(rid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_rids=sel_rids)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_name_click)
    pass


class MatchVerificationInteraction(AbstractInteraction):
    def __init__(self, ibs, rid1, rid2, **kwargs):
        print('[matchver] __init__')
        super(MatchVerificationInteraction, self).__init__(**kwargs)
        self.ibs = ibs
        self.rid1 = rid1
        self.rid2 = rid2
        self.infer_data()
        self.show_page()

    def infer_data(self):
        ibs = self.ibs
        rid1, rid2 = self.rid1, self.rid2
        self.nid1, self.nid2 = ibs.get_roi_nids((rid1, rid2))
        if self.nid1 != self.nid2:
            groundtruth1, groundtruth2 = ibs.get_roi_groundtruth((rid1, rid2))
            self.gt1, self.gt2 = groundtruth1 + [rid1], groundtruth2 + [rid2]
            self.nCols = max(len(self.gt1), len(self.gt2))
            self.nRows = 2
        else:
            groundtruth, = ibs.get_roi_groundtruth((rid1,))
            self.gt1 = self.gt2 = groundtruth + [rid1]
            self.nCols = len(self.gt1)
            self.nRows = 1
        self.nids = utool.unique_ordered([self.nid1, self.nid2])
        self.colors = df2.distinct_colors(len(self.nids))

    def prepare_page(self):
        figkw = {'fnum': self.fnum,
                 'doclf': True,
                 'docla': True, }
        self.fig = df2.figure(**figkw)

    def show_page(self):
        print('[matchver] show_page()')
        self.prepare_page()
        # Variables we will work with to paint a pretty picture
        gt1, gt2 = self.gt1, self.gt2
        # Compare two sets of names

        nRows = self.nRows
        nCols = self.nCols

        for px, rid in enumerate(gt1):
            self.plot_chip(rid, nRows, nCols, px + 1, nokpts=True)
            ax = df2.gca()
            df2.draw_border(ax, color=self.colors[0])
        # show name2 rois
        if self.nid1 != self.nid2:
            for px, rid in enumerate(gt2):
                self.plot_chip(rid, nRows, nCols, px + nCols + 1, nokpts=True)
                ax = df2.gca()
                df2.draw_border(ax, color=self.colors[1])

        self.show_hud()
        self.draw()
        self.update()

    def break_match(self):
        self.ibs.set_roi_names([self.rid1, self.rid2], ['____', '____'])

    def new_match(self):
        new_name = ibsfuncs.make_new_name(self.ibs)
        self.ibs.set_roi_names([self.rid1, self.rid2], [new_name, new_name])

    def merge(self):
        self.ibs.set_roi_names(self.gt1 , [self.name2] * len(self.gt1))

    def show_hud(self):
        df2.set_figtitle('Review Match: ' + ibsfuncs.vsstr(self.rid1, self.rid2))

    def plot_chip(self, rid, nRows, nCols, px, **kwargs):
        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': (nRows, nCols, px),
            'nokpts': True,
        }
        viz_chip.show_chip(self.ibs, rid, **viz_chip_kw)

        make_buttons = True
        if make_buttons:
            ax = df2.gca()
            divider = df2.ensure_divider(ax)
            butkw = {
                'divider': divider,
                'callback': self.break_match,
                'index': 0,
            }
            self.append_button('break', **butkw)
        #    if name1 == name2 and not name1.startswith('____'):
        #        self.append_button(BREAK_MATCH_PREF, **butkw)
        #    else:
        #        if not name1.startswith('____'):
        #            self.append_button(RENAME2_PREF + name1, **butkw)
        #        if not name2.startswith('____'):
        #            self.append_button(RENAME1_PREF + name2, **butkw)
        #        if name1.startswith('____') and name2.startswith('____'):
        #            self.append_button(NEW_MATCH_PREF, **butkw)

        #if draw:
        #    vh.draw()

