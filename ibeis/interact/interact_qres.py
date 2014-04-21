from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import plottool.draw_func2 as df2
# IBEIS
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from . import interact_helpers as ih
from . import interact_chipres

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz-matches]', DEBUG=False)


@utool.indent_func
@profile
def interact_qres(ibs, qres, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''

    # Result Interaction
    printDBG('[viz._show_res()] starting interaction')

    def _ctrlclicked_rid(rid):
        printDBG('ctrl+clicked rid=%r' % rid)
        fnum = df2.next_fnum()
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        ih.disconnect_callback(fig, 'button_press_event')
        viz.show_sv(ibs, qres.qrid, rid2=rid, fnum=fnum)
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def _clicked_rid(rid):
        printDBG('clicked rid=%r' % rid)
        fnum = df2.next_fnum()
        interact_chipres.interact_chipres(ibs, qres, rid, fnum=fnum)
        fig = df2.gcf()
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def _top_matches_view(toggle=0):
        # Toggle if the click is not in any axis
        printDBG('clicked none')
        kwargs['annote'] = kwargs.get('annote', 0) + toggle
        fig = viz.show_qres(ibs, qres, **kwargs)
        vh.draw()
        return fig

    def _on_match_click(event):
        'result interaction mpl event callback slot'
        print('[viz] clicked result')
        if event.xdata is None or event.inaxes is None:
            return _top_matches_view(toggle=1)
        ax = event.inaxes
        viztype = vh.get_ibsdat(ax, 'viztype', '')
        #printDBG(str(event.__dict__))
        printDBG('viztype=%r' % viztype)
        # Clicked a specific chipres
        if viztype.startswith('chipres'):
            rid = vh.get_ibsdat(ax, 'rid', None)
            # Ctrl-Click
            key = '' if event.key is None else event.key
            print('key = %r' % key)
            if key.find('control') == 0:
                print('[viz] result control clicked')
                return _ctrlclicked_rid(rid)
            # Left-Click
            else:
                print('[viz] result clicked')
                return _clicked_rid(rid)

    fig = _top_matches_view()
    ih.connect_callback(fig, 'button_press_event', _on_match_click)
    printDBG('[viz._show_res()] Finished')
