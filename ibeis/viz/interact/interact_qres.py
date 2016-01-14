# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
#from ibeis.algo.hots import chip_match
import plottool as pt
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih  # NOQA
#from plottool import interact_matches  # NOQA
from plottool import abstract_interaction  # NOQA
from ibeis import viz
from ibeis.viz.interact.interact_sver import ishow_sver

(print, rrr, profile) = ut.inject2(__name__, '[interact_qres]')


def ishow_analysis(ibs, cm, qreq_=None, **kwargs):
    """

    CommandLine:
        python -m ibeis.viz.interact.interact_qres --test-ishow_analysis:0 --show
        python -m ibeis.viz.interact.interact_qres --test-ishow_analysis:1 --show

    Example0:
        >>> # SLOW_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> cm, qreq_ = ibeis.testdata_cm()
        >>> fig = ishow_analysis(qreq_.ibs, cm, qreq_=qreq_)
        >>> pt.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> cm, qreq_ = ibeis.testdata_cm()
        >>> fig = ishow_analysis(qreq_.ibs, cm, qreq_=qreq_)
        >>> pt.show_if_requested()

    """
    if True:
        interact = InteractQres(ibs, cm, analysis=True, qreq_=qreq_, **kwargs)
        interact.show_page()
        interact.show()
        return interact
    else:
        return ishow_qres(ibs, cm, analysis=True, qreq_=qreq_, **kwargs)


BASE_CLASS = abstract_interaction.AbstractInteraction
#BASE_CLASS = interact_matches.MatchInteraction2


class InteractQres(BASE_CLASS):
    def __init__(self, ibs, cm, analysis=False, qreq_=None, **kwargs):
        self.ibs = ibs
        self.cm = cm
        self.analysis = analysis
        self.qreq_ = qreq_
        self.kwargs = kwargs.copy()
        self.verbose = True
        super(InteractQres, self).__init__(**kwargs)
        self.fnum
        print('self.fnum = %r' % (self.fnum,))

    def plot(self, *args, **kwargs):
        if self.analysis:
            self._analysis_view(toggle=1)
        else:
            self._top_matches_view(toggle=1)

    def _top_matches_view(self, toggle=0):
        # Toggle if the click is not in any axis
        self.kwargs['annot_mode'] = self.kwargs.get('annot_mode', 0) + toggle
        self.kwargs['fnum'] = self.fnum
        fig = viz.show_qres(self.ibs, self.cm, qreq_=self.qreq_, **self.kwargs)
        return fig

    def _analysis_view(self, toggle=0):
        # Toggle if the click is not in any axis
        if self.verbose:
            print('clicked none')
        self.kwargs['annot_mode'] = self.kwargs.get('annot_mode', 0) + toggle
        self.kwargs['fnum'] = self.fnum
        #if isinstance(self.cm, chip_match.ChipMatch):
        fig = self.cm.show_analysis(self.qreq_, **self.kwargs)
        #else:
        #    fig = self.cm.show_analysis(self.ibs, qreq_=self.qreq_, **self.kwargs)
        self.draw()
        return fig

    def show_sver_process_to_aid(self, aid2):
        if self.verbose:
            print('ctrl+clicked aid2=%r' % aid2)
        fnum_ = pt.next_fnum()
        ishow_sver(self.ibs, self.cm.qaid, aid2, qreq_=self.qreq_, fnum=fnum_)
        self.draw()
        self.bring_to_front()

    def show_matches_to_aid(self, aid2):
        if self.verbose:
            print('clicked aid2=%r' % aid2)
        fnum_ = pt.next_fnum()
        #if isinstance(self.cm, chip_match.ChipMatch):
        self.cm.ishow_match(self.qreq_, aid2, fnum=fnum_)
        #else:
        #    self.cm.ishow_matches(self.ibs, aid2, qreq_=self.qreq_, fnum=fnum_)
        self.draw()
        #self.bring_to_front()
        #fig = pt.gcf()
        #fig.canvas.draw()
        #pt.bring_to_front(fig)

    def on_click_outside(self, event):
        self.show_page()

    def on_click_inside(self, event, ax):
        ax = event.inaxes
        viztype = ph.get_plotdat(ax, 'viztype', '')
        #if verbose:
        #    print(str(event.__dict__))
        print('viztype=%r' % viztype)
        # Clicked a specific matches
        print('plodat_dict = ' + ut.dict_str(ph.get_plotdat_dict(ax)))
        if viztype.startswith('chip'):
            from ibeis.viz.interact import interact_chip
            options = interact_chip.build_annot_context_options(
                self.ibs, self.cm.qaid, refresh_func=self._analysis_view,
                with_interact_chip=False)
            self.show_popup_menu(options, event)

        if viztype.startswith('matches') or viztype == 'multi_match':  # why startswith?
            aid2 = ph.get_plotdat(ax, 'aid2', None)
            aid_list = ph.get_plotdat(ax, 'aid_list', None)
            if event.button == 3:   # right-click
                # TODO; this functionality should be in viz.interact
                from ibeis.gui import inspect_gui
                print('right click')
                print('qreq_ = %r' % (self.qreq_,))
                options = inspect_gui.get_aidpair_context_menu_options(
                    self.ibs, self.cm.qaid, aid2, self.cm,
                    qreq_=self.qreq_, update_callback=self.show_page,
                    backend_callback=None, aid_list=aid_list)
                self.show_popup_menu(options, event)
            else:
                # Ctrl-Click
                key = '' if event.key is None else event.key
                print('key = %r' % key)
                if key.find('control') == 0:
                    print('[viz] result control clicked')
                    self.show_sver_process_to_aid(aid2)
                # Left-Click
                else:
                    print('[viz] result clicked')
                    self.show_matches_to_aid(aid2)
        self.draw()


def ishow_qres(ibs, cm, analysis=False, dodraw=True, qreq_=None, **kwargs):
    """
    Displays query chip, groundtruth matches, and top matches

    THERE IS A DIFFERENCE BETWEEN THIS AND MATCH INTERACTION. THIS IS FOR DISPLAYING THE RANKED LIST
    MATCH INTERACTION IS LOOKING AT A SINGLE PAIR

    TODO: make this a class
    TODO; dodraw should be false by default

    SeeAlso:
        #interact_matches.MatchInteraction2
        #ibeis.viz.interact.MatchInteraction

    Args:
        ibs (IBEISController):  ibeis controller object
        cm (QueryResult):  object of feature correspondences and scores
        analysis (bool):

    CommandLine:
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres --show
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> # EN-ABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_qres import *  # NOQA
        >>> import ibeis
        >>> cm, qreq_ = ibeis.testdata_cm()
        >>> analysis = True
        >>> fig = ishow_qres(qreq_.ibs, cm, analysis, dodraw=False, qreq_=qreq_)
        >>> pt.show_if_requested()
    """
    if True:
        # TODO: make this a class
        interact = InteractQres(ibs, cm, analysis=True, qreq_=qreq_, **kwargs)
        interact.show_page()
        interact.show()
        return interact
    #else:
    #    fnum = pt.ensure_fnum(kwargs.get('fnum', None))
    #    kwargs['fnum'] = fnum

    #    fig = ih.begin_interaction('qres', fnum)
    #    # Result Interaction
    #    #if verbose:
    #    #    print('[ishow_qres] starting interaction')

    #    # Start the transformation into a class
    #    self = ut.DynStruct()
    #    self.qreq_ = qreq_
    #    self.ibs = ibs
    #    verbose = False

    #    def show_sver_process_to_aid(aid2):
    #        if verbose:
    #            print('ctrl+clicked aid2=%r' % aid2)
    #        fnum_ = pt.next_fnum()
    #        ishow_sver(ibs, cm.qaid, aid2, qreq_=self.qreq_, fnum=fnum_)
    #        fig.canvas.draw()
    #        pt.bring_to_front(fig)

    #    def show_matches_to_aid(aid2):
    #        if verbose:
    #            print('clicked aid2=%r' % aid2)
    #        fnum_ = pt.next_fnum()
    #        cm.ishow_matches(ibs, aid2, qreq_=self.qreq_, fnum=fnum_)
    #        fig = pt.gcf()
    #        fig.canvas.draw()
    #        #pt.bring_to_front(fig)

    #    def _top_matches_view(toggle=0):
    #        # Toggle if the click is not in any axis
    #        if verbose:
    #            print('clicked none')
    #        kwargs['annot_mode'] = kwargs.get('annot_mode', 0) + toggle
    #        fig = viz.show_qres(ibs, cm, qreq_=self.qreq_, **kwargs)
    #        return fig

    #    def _analysis_view(toggle=0):
    #        # Toggle if the click is not in any axis
    #        if verbose:
    #            print('clicked none')
    #        kwargs['annot_mode'] = kwargs.get('annot_mode', 0) + toggle
    #        fig = cm.show_analysis(ibs, qreq_=self.qreq_, **kwargs)
    #        return fig

    #    def _refresh():
    #        if analysis:
    #            fig = _analysis_view(toggle=1)
    #        else:
    #            fig = _top_matches_view(toggle=1)
    #        fig.canvas.draw()
    #        # TODO NEED TO MOVE TO CLASS STRUCTURE
    #        ph.draw()

    #    #def _on_keypress(event):
    #    #    if event.key == ',':
    #    #        print(event.key)
    #    #        from ibeis.gui import inspect_gui
    #    #        update_callback = _refresh
    #    #        backend_callback = None
    #    #        print('qreq_ = %r' % (self.qreq_,))
    #    #        print('right click')
    #    #        height = fig.canvas.geometry().height()
    #    #        import guitool
    #    #        qpoint = guitool.newQPoint(event.x, height - event.y)
    #    #        qwin = fig.canvas
    #    #        inspect_gui.show_aidpair_context_menu(
    #    #            ibs, qwin, qpoint, cm.qaid, aid2, cm, qreq_=self.qreq_,
    #    #            update_callback=update_callback,
    #    #            backend_callback=backend_callback, aid_list=aid_list)

    #    def _on_match_click(event):
    #        """ result interaction mpl event callback slot """
    #        print('[viz] clicked result')
    #        if ih.clicked_outside_axis(event):
    #            _refresh()
    #        else:
    #            ax = event.inaxes
    #            viztype = ph.get_plotdat(ax, 'viztype', '')
    #            #if verbose:
    #            #    print(str(event.__dict__))
    #            print('viztype=%r' % viztype)
    #            # Clicked a specific matches
    #            print('plodat_dict = ' + ut.dict_str(ph.get_plotdat_dict(ax)))
    #            if viztype.startswith('chip'):
    #                import guitool
    #                from ibeis.viz.interact import interact_chip
    #                height = fig.canvas.geometry().height()
    #                qpoint = guitool.newQPoint(event.x, height - event.y)
    #                refresh_func = _analysis_view
    #                interact_chip.show_annot_context_menu(
    #                    ibs, cm.qaid, fig.canvas, qpoint, refresh_func=refresh_func,
    #                    with_interact_chip=False)
    #            if viztype.startswith('matches') or viztype == 'multi_match':  # why startswith?
    #                aid2 = ph.get_plotdat(ax, 'aid2', None)
    #                aid_list = ph.get_plotdat(ax, 'aid_list', None)
    #                if event.button == 3:   # right-click
    #                    print('right click')
    #                    height = fig.canvas.geometry().height()
    #                    import guitool
    #                    qpoint = guitool.newQPoint(event.x, height - event.y)
    #                    qwin = fig.canvas
    #                    # TODO; this functionality should be in viz.interact
    #                    from ibeis.gui import inspect_gui
    #                    update_callback = _refresh
    #                    backend_callback = None
    #                    print('qreq_ = %r' % (self.qreq_,))
    #                    options = inspect_gui.get_aidpair_context_menu_options(
    #                        ibs, cm.qaid, aid2, cm, qreq_=qreq_,
    #                        update_callback=update_callback,
    #                        backend_callback=backend_callback, aid_list=aid_list)
    #                    guitool.popup_menu(qwin, qpoint, options)
    #                    #inspect_gui.show_aidpair_context_menu(
    #                    #    ibs, qwin, qpoint, cm.qaid, aid2, cm, qreq_=self.qreq_,
    #                    #    update_callback=update_callback,
    #                    #    backend_callback=backend_callback, aid_list=aid_list)
    #                    #callback_list = [
    #                    #]
    #                    #guitool.popup_menu(qwin, qpoint, callback_list)
    #                else:
    #                    # Ctrl-Click
    #                    key = '' if event.key is None else event.key
    #                    print('key = %r' % key)
    #                    if key.find('control') == 0:
    #                        print('[viz] result control clicked')
    #                        show_sver_process_to_aid(aid2)
    #                    # Left-Click
    #                    else:
    #                        print('[viz] result clicked')
    #                        show_matches_to_aid(aid2)
    #                    #print('multimatches')
    #                    #aid2 = ph.get_plotdat(ax, 'aid2', None)
    #                    #key = '' if event.key is None else event.key
    #                    #print('key = %r' % key)
    #                    #if key == '':
    #                    #    show_matches_to_aid(aid2)
    #        ph.draw()

    #    if analysis:
    #        fig = _analysis_view()
    #    else:
    #        fig = _top_matches_view()
    #    if dodraw:
    #        ph.draw()
    #    ih.connect_callback(fig, 'button_press_event', _on_match_click)
    #    #ih.connect_callback(fig, 'key_press_event', _on_keypress)
    #    #if verbose:
    #    #    print('[ishow_qres] Finished')
    #    return fig


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
