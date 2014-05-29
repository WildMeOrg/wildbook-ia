from __future__ import absolute_import, division, print_function
import utool
from plottool import interact_helpers as ih
from ibeis import viz
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)
#==========================
# Name Interaction
#==========================


def ishow_name(ibs, nid, rids=[], select_rid_callback=None, fnum=5, **kwargs):
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
                viz.show_name(ibs, nid, fnum=fnum, rids=[rid])
                if select_rid_callback is not None:
                    select_rid_callback(rid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, rids=rids)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_name_click)
    pass
