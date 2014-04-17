from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import plottool.draw_func2 as df2
# IBEIS
from ibeis.view import viz
from ibeis.view.viz import viz_helpers as vh
from . import interact_helpers as ih

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz-matches]', DEBUG=False)


@utool.indent_func
@profile
def interact_qres(ibs, res, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''

    # Result Interaction
    printDBG('[viz._show_res()] starting interaction')

    def _ctrlclicked_cid(cid):
        printDBG('ctrl+clicked cid=%r' % cid)
        fnum = df2.next_fnum()
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        df2.disconnect_callback(fig, 'button_press_event')
        viz.show_sv(ibs, res.qcid, cid2=cid, fnum=fnum)
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def _clicked_cid(cid):
        printDBG('clicked cid=%r' % cid)
        fnum = df2.next_fnum()
        res.interact_chipres(ibs, cid, fnum=fnum)
        fig = df2.gcf()
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def _top_matches_view(toggle=0):
        # Toggle if the click is not in any axis
        printDBG('clicked none')
        kwargs['annote'] = kwargs.get('annote', 0) + toggle
        fig = viz.show_qres(ibs, res, **kwargs)
        ih.connect_callback(fig, 'button_press_event', _on_res_click)
        vh.draw()

    def _on_res_click(event):
        'result interaction mpl event callback slot'
        print('[viz] clicked result')
        if event.xdata is None or event.inaxes is None:
            return _top_matches_view()
        ax = event.inaxes
        hs_viztype = ax.__dict__.get('_hs_viztype', '')
        printDBG(event.__dict__)
        printDBG('hs_viztype=%r' % hs_viztype)
        # Clicked a specific chipres
        if hs_viztype.find('chipres') == 0:
            cid = ax.__dict__.get('_hs_cid')
            # Ctrl-Click
            key = '' if event.key is None else event.key
            print('key = %r' % key)
            if key.find('control') == 0:
                print('[viz] result control clicked')
                return _ctrlclicked_cid(cid)
            # Left-Click
            else:
                print('[viz] result clicked')
                return _clicked_cid(cid)

    _top_matches_view()

    printDBG('[viz._show_res()] Finished')
