from __future__ import absolute_import, division, print_function
import utool as ut
import plottool.draw_func2 as df2
from ibeis import viz
import plottool as pt
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
from ibeis.viz.interact.interact_sver import ishow_sver

(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[interact_qres]', DEBUG=False)


def ishow_analysis(ibs, qres, qreq_=None, **kwargs):
    """

    CommandLine:
        python -m ibeis.viz.interact.interact_qres --test-ishow_analysis:0 --show
        python -m ibeis.viz.interact.interact_qres --test-ishow_analysis:1 --show

    Example0:
        >>> # SLOW_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid = 2
        >>> qres = ibs._query_chips4([qaid], ibs.get_valid_aids(), cfgdict=dict())[qaid]
        >>> fig = ishow_analysis(ibs, qres)
        >>> pt.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid = 12
        >>> qres = ibs._query_chips4([qaid], ibs.get_valid_aids(), cfgdict=dict())[qaid]
        >>> fig = ishow_analysis(ibs, qres)
        >>> pt.show_if_requested()

    """
    return ishow_qres(ibs, qres, analysis=True, qreq_=qreq_, **kwargs)


def ishow_qres(ibs, qres, analysis=False, dodraw=True, qreq_=None, **kwargs):
    """
    Displays query chip, groundtruth matches, and top matches
    TODO: make this a class

    Args:
        ibs (IBEISController):  ibeis controller object
        qres (QueryResult):  object of feature correspondences and scores
        analysis (bool):

    CommandLine:
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> qreq_ = None
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
        >>> analysis = False
        >>> fig = ishow_qres(ibs, qres, analysis, dodraw=False, qreq_=qreq_)
        >>> pt.show_if_requested()
    """
    fnum = df2.ensure_fnum(kwargs.get('fnum', None))
    kwargs['fnum'] = fnum

    fig = ih.begin_interaction('qres', fnum)
    # Result Interaction
    printDBG('[ishow_qres] starting interaction')

    def _ctrlclicked_aid(aid2):
        printDBG('ctrl+clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        ishow_sver(ibs, qres.qaid, aid2, qreq_=qreq_, fnum=fnum_)
        fig.canvas.draw()
        pt.bring_to_front(fig)

    def _clicked_aid(aid2):
        printDBG('clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        qres.ishow_matches(ibs, aid2, qreq_=qreq_, fnum=fnum_)
        fig = df2.gcf()
        fig.canvas.draw()
        pt.bring_to_front(fig)

    def _top_matches_view(toggle=0):
        # Toggle if the click is not in any axis
        printDBG('clicked none')
        kwargs['annot_mode'] = kwargs.get('annot_mode', 0) + toggle
        fig = viz.show_qres(ibs, qres, qreq_=qreq_, **kwargs)
        return fig

    def _analysis_view(toggle=0):
        # Toggle if the click is not in any axis
        printDBG('clicked none')
        kwargs['annot_mode'] = kwargs.get('annot_mode', 0) + toggle
        fig = qres.show_analysis(ibs, qreq_=qreq_, **kwargs)
        return fig

    def _on_match_click(event):
        """ result interaction mpl event callback slot """
        print('[viz] clicked result')
        if ih.clicked_outside_axis(event):
            if analysis:
                _analysis_view(toggle=1)
            else:
                _top_matches_view(toggle=1)
        else:
            ax = event.inaxes
            viztype = ph.get_plotdat(ax, 'viztype', '')
            #printDBG(str(event.__dict__))
            printDBG('viztype=%r' % viztype)
            # Clicked a specific matches
            if viztype.startswith('matches'):
                aid2 = ph.get_plotdat(ax, 'aid2', None)
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
        ph.draw()

    if analysis:
        fig = _analysis_view()
    else:
        fig = _top_matches_view()
    if dodraw:
        ph.draw()
    ih.connect_callback(fig, 'button_press_event', _on_match_click)
    printDBG('[ishow_qres] Finished')
    return fig


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_qres
        python -m ibeis.viz.interact.interact_qres --allexamples
        python -m ibeis.viz.interact.interact_qres --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
