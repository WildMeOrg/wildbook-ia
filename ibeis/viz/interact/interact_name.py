from __future__ import absolute_import, division, print_function
import utool
from itertools import izip
from plottool import interact_helpers as ih
from plottool import draw_func2 as df2
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from ibeis.dev import ibsfuncs
from functools import partial
from plottool.abstract_interaction import AbstractInteraction
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)

from ibeis.viz import viz_chip


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
        #self.infer_data()
        self.show_page()

    def infer_data(self):
        ibs = self.ibs
        self.rid_list = [self.rid1, self.rid2]
        self.nid_list = ibs.get_roi_nids(self.rid_list)
        groundtruth_list = ibs.get_roi_groundtruth(self.rid_list)
        self.gt_list = [gt + [rid] for gt, rid in
                        izip(groundtruth_list, self.rid_list)]
        self.nCols = max(map(len, self.gt_list))
        self.nRows = len(self.gt_list)

        # Build Groundtruth Map
        self.nid2_gt = utool.odict()
        for nid, groundtruth in izip(self.nid_list, self.gt_list):
            if nid not in self.nid2_gt:
                self.nid2_gt[nid] = groundtruth
        self.nid2_color = utool.odict()

        self.nid1, self.nid2 = self.nid_list
        self.gt1, self.gt2 = self.gt_list
        if self.nid1 == self.nid2:
            self.nRows = 1

    def prepare_page(self):
        figkw = {'fnum': self.fnum,
                 'doclf': True,
                 'docla': True, }
        self.fig = df2.figure(**figkw)

    def show_page(self):
        print('[matchver] show_page()')
        self.prepare_page()
        # Variables we will work with to paint a pretty picture
        # Compare two sets of names
        self.infer_data()
        nRows = self.nRows
        nCols = self.nCols

        # Distinct color for every unique name
        unique_colors = df2.distinct_colors(len(self.nid2_gt))

        for count, ((nid, gt), color) in enumerate(izip(self.nid2_gt.iteritems(), unique_colors)):
            offset = count * nCols + 1
            for px, rid in enumerate(gt):
                self.plot_chip(rid, nRows, nCols, px + offset, color=color)

        self.show_hud()
        self.draw()
        self.update()

    def plot_chip(self, rid, nRows, nCols, px, **kwargs):
        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': (nRows, nCols, px),
            'nokpts': True,
        }
        viz_chip.show_chip(self.ibs, rid, **viz_chip_kw)
        ax = df2.gca()
        df2.draw_border(ax, color=kwargs.get('color'))

        make_buttons = True
        if make_buttons:
            divider = df2.ensure_divider(ax)
            butkw = {
                'divider': divider,
            }
            self.append_button('unname', callback=partial(self.unname_roi, rid), **butkw)
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

    def unname_roi(self, rid, event=None):
        print('unname')
        self.ibs.set_roi_names([rid], [self.ibs.UNKNOWN_NAME])
        self.show_page()

    def break_match(self, event=None):
        self.ibs.set_roi_names([self.rid1, self.rid2], ['____', '____'])
        self.show_page()

    def new_match(self, event=None):
        new_name = ibsfuncs.make_new_name(self.ibs)
        self.ibs.set_roi_names([self.rid1, self.rid2], [new_name, new_name])
        self.show_page()

    def merge(self, event=None):
        self.ibs.set_roi_names(self.gt1 , [self.name2] * len(self.gt1))
        self.show_page()

    def show_hud(self):
        vsstr = ibsfuncs.vsstr(self.rid1, self.rid2)
        df2.set_figtitle('Review Match: ' + vsstr)
