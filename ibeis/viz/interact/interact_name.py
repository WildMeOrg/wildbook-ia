from __future__ import absolute_import, division, print_function
import utool
from plottool import interact_helpers as ih
from plottool import draw_func2 as df2
from ibeis import viz
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)
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


class MatchVerificationInteraction(object):
    def __init__(self, ibs, rid1, rid2, fnum=None, **kwargs):
        print('[matchver] __init__')
        self.ibs = ibs
        self.rid1 = rid1
        self.rid2 = rid2
        if fnum is None:
            fnum = df2.next_fnum()
        self.fnum = fnum
        self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)

        self.show_page()

    def prepare_page(self):
        self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)

    def show_page(self):
        print('[matchver] show_page()')
        self.prepare_page()
        ibs = self.ibs
        rid1 = self.rid1
        rid2 = self.rid2

        nid1, nid2 = ibs.get_roi_nids((rid1, rid2))
        self.draw()
        df2.update()

    def bring_to_front(self):
        df2.bring_to_front(self.fig)

    def draw(self):
        self.fig.canvas.draw()

    def show(self):
        self.draw()
        self.bring_to_front()
