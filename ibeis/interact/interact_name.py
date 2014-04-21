from __future__ import absolute_import, division, print_function
import utool
from . import interact_helpers as ih
from ibeis import viz
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)
#==========================
# Name Interaction
#==========================


def ishow_name(ibs, nid, sel_cids=[], select_cid_func=None, fnum=5, **kwargs):
    fig = ih.begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax is None or x is None:
            # The click is not in any axis
            print('... out of axis')
        else:
            hs_viztype = ax.__dict__.get('_hs_viztype', '')
            print_(' hs_viztype=%r' % hs_viztype)
            if hs_viztype == 'chip':
                cid = ax.__dict__.get('_hs_cid')
                print('... cid=%r' % cid)
                viz.show_name(ibs, nid, fnum=fnum, sel_cids=[cid])
                select_cid_func(cid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_cids=sel_cids)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_name_click)
    pass
