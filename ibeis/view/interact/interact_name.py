#==========================
# Name Interaction
#==========================

def interact_name(ibs, nid, sel_cids=[], select_cid_func=None, fnum=5, **kwargs):
    fig = ih.begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax is None or x is None:
            # The click is not in any axis
            print('... out of axis')
        else:
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            print_(' hs_viewtype=%r' % hs_viewtype)
            if hs_viewtype == 'chip':
                cid = ax.__dict__.get('_hs_cid')
                print('... cid=%r' % cid)
                viz.show_name(ibs, nid, fnum=fnum, sel_cids=[cid])
                select_cid_func(cid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_cids=sel_cids)
    viz.draw()
    df2.connect_callback(fig, 'button_press_event', _on_name_click)
    pass
