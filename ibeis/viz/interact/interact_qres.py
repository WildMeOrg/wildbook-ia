from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import plottool.draw_func2 as df2
# IBEIS
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih
from .interact_matches import ishow_matches
from .interact_sver import ishow_sver

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[interact_qres]', DEBUG=False)


#@utool.indent_func
@profile
def ishow_qres(ibs, qres, **kwargs):
    """
    Displays query chip, groundtruth matches, and top 5 matches
    """
    fnum = df2.kwargs_fnum(kwargs)
    fig = ih.begin_interaction('qres', fnum)
    # Result Interaction
    printDBG('[ishow_qres] starting interaction')

    def _ctrlclicked_aid(aid2):
        printDBG('ctrl+clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        ishow_sver(ibs, qres.qaid, aid2, fnum=fnum_)
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def _clicked_aid(aid2):
        printDBG('clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        ishow_matches(ibs, qres, aid2, fnum=fnum_)
        fig = df2.gcf()
        fig.canvas.draw()
        df2.bring_to_front(fig)

    def _top_matches_view(toggle=0):
        # Toggle if the click is not in any axis
        printDBG('clicked none')
        kwargs['annote_mode'] = kwargs.get('annote_mode', 0) + toggle
        fig = viz.show_qres(ibs, qres, **kwargs)
        return fig

    def _on_match_click(event):
        'result interaction mpl event callback slot'
        print('[viz] clicked result')
        if ih.clicked_outside_axis(event):
            _top_matches_view(toggle=1)
        else:
            ax = event.inaxes
            viztype = vh.get_ibsdat(ax, 'viztype', '')
            #printDBG(str(event.__dict__))
            printDBG('viztype=%r' % viztype)
            # Clicked a specific matches
            if viztype.startswith('matches'):
                aid2 = vh.get_ibsdat(ax, 'aid2', None)
                # Ctrl-Click
                key = '' if event.key is None else event.key
                print('key = %r' % key)
                if key.find('control') == 0:
                    print('[viz] result control clicked')
                    _ctrlclicked_aid(aid2)
                # Left-Click
                else:
                    print('[viz] result clicked')
                    _clicked_aid(aid2)
        vh.draw()

    fig = _top_matches_view()
    vh.draw()
    ih.connect_callback(fig, 'button_press_event', _on_match_click)
    printDBG('[ishow_qres] Finished')
    return fig
