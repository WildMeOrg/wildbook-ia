from __future__ import absolute_import, division, print_function
import utool as ut
import plottool.draw_func2 as df2
from ibeis import viz
import plottool as pt
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
from ibeis.viz.interact.interact_sver import ishow_sver

(print, rrr, profile) = ut.inject2(__name__, '[interact_qres]')


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
    TODO; dodraw should be false by default

    Args:
        ibs (IBEISController):  ibeis controller object
        qres (QueryResult):  object of feature correspondences and scores
        analysis (bool):

    CommandLine:
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres --show
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> # EN-ABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> ibs, qreq_, qres = ibeis.testdata_qres()
        >>> analysis = True
        >>> fig = ishow_qres(ibs, qres, analysis, dodraw=False, qreq_=qreq_)
        >>> pt.show_if_requested()
    """
    # TODO: make this a class

    fnum = df2.ensure_fnum(kwargs.get('fnum', None))
    kwargs['fnum'] = fnum

    fig = ih.begin_interaction('qres', fnum)
    # Result Interaction
    #if verbose:
    #    print('[ishow_qres] starting interaction')

    # Start the transformation into a class
    self = ut.DynStruct()
    self.qreq_ = qreq_
    self.ibs = ibs
    verbose = False

    def show_sver_process_to_aid(aid2):
        if verbose:
            print('ctrl+clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        ishow_sver(ibs, qres.qaid, aid2, qreq_=self.qreq_, fnum=fnum_)
        fig.canvas.draw()
        pt.bring_to_front(fig)

    def show_matches_to_aid(aid2):
        if verbose:
            print('clicked aid2=%r' % aid2)
        fnum_ = df2.next_fnum()
        qres.ishow_matches(ibs, aid2, qreq_=self.qreq_, fnum=fnum_)
        fig = df2.gcf()
        fig.canvas.draw()
        pt.bring_to_front(fig)

    def _top_matches_view(toggle=0):
        # Toggle if the click is not in any axis
        if verbose:
            print('clicked none')
        kwargs['annot_mode'] = kwargs.get('annot_mode', 0) + toggle
        fig = viz.show_qres(ibs, qres, qreq_=self.qreq_, **kwargs)
        return fig

    def _analysis_view(toggle=0):
        # Toggle if the click is not in any axis
        if verbose:
            print('clicked none')
        kwargs['annot_mode'] = kwargs.get('annot_mode', 0) + toggle
        fig = qres.show_analysis(ibs, qreq_=self.qreq_, **kwargs)
        return fig

    def _refresh():
        if analysis:
            _analysis_view(toggle=1)
        else:
            _top_matches_view(toggle=1)

    def _on_match_click(event):
        """ result interaction mpl event callback slot """
        print('[viz] clicked result')
        if ih.clicked_outside_axis(event):
            _refresh()
        else:
            ax = event.inaxes
            viztype = ph.get_plotdat(ax, 'viztype', '')
            #if verbose:
            #    print(str(event.__dict__))
            print('viztype=%r' % viztype)
            # Clicked a specific matches
            print('plodat_dict = ' + ut.dict_str(ph.get_plotdat_dict(ax)))
            if viztype.startswith('chip'):
                import guitool
                from ibeis.viz.interact import interact_chip
                height = fig.canvas.geometry().height()
                qpoint = guitool.newQPoint(event.x, height - event.y)
                refresh_func = _analysis_view
                interact_chip.show_annot_context_menu(
                    ibs, qres.qaid, fig.canvas, qpoint, refresh_func=refresh_func,
                    with_interact_chip=False)
            if viztype.startswith('matches') or viztype == 'multi_match':  # why startswith?
                aid2 = ph.get_plotdat(ax, 'aid2', None)
                aid_list = ph.get_plotdat(ax, 'aid_list', None)
                if event.button == 3:   # right-click
                    print('right click')
                    height = fig.canvas.geometry().height()
                    import guitool
                    qpoint = guitool.newQPoint(event.x, height - event.y)
                    qwin = fig.canvas
                    # TODO; this functionality should be in viz.interact
                    from ibeis.gui import inspect_gui
                    update_callback = _refresh
                    backend_callback = None
                    print('qreq_ = %r' % (self.qreq_,))
                    inspect_gui.show_aidpair_context_menu(
                        ibs, qwin, qpoint, qres.qaid, aid2, qres, qreq_=self.qreq_,
                        update_callback=update_callback,
                        backend_callback=backend_callback, aid_list=aid_list)
                    #callback_list = [
                    #]
                    #guitool.popup_menu(qwin, qpoint, callback_list)
                else:
                    # Ctrl-Click
                    key = '' if event.key is None else event.key
                    print('key = %r' % key)
                    if key.find('control') == 0:
                        print('[viz] result control clicked')
                        show_sver_process_to_aid(aid2)
                    # Left-Click
                    else:
                        print('[viz] result clicked')
                        show_matches_to_aid(aid2)
                    #print('multimatches')
                    #aid2 = ph.get_plotdat(ax, 'aid2', None)
                    #key = '' if event.key is None else event.key
                    #print('key = %r' % key)
                    #if key == '':
                    #    show_matches_to_aid(aid2)
        ph.draw()

    if analysis:
        fig = _analysis_view()
    else:
        fig = _top_matches_view()
    if dodraw:
        ph.draw()
    ih.connect_callback(fig, 'button_press_event', _on_match_click)
    #if verbose:
    #    print('[ishow_qres] Finished')
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
