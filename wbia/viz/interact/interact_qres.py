# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import wbia.plottool as pt
from wbia.plottool import plot_helpers as ph
from wbia.plottool import abstract_interaction
from wbia import viz
from wbia.viz.interact.interact_sver import ishow_sver

(print, rrr, profile) = ut.inject2(__name__, '[interact_qres]')


def ishow_analysis(ibs, cm, qreq_=None, **kwargs):
    """

    CommandLine:
        python -m wbia.viz.interact.interact_qres --test-ishow_analysis:0 --show
        python -m wbia.viz.interact.interact_qres --test-ishow_analysis:1 --show

    Example0:
        >>> # SLOW_DOCTEST
        >>> from wbia.viz.interact.interact_qres import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> fig = ishow_analysis(qreq_.ibs, cm, qreq_=qreq_)
        >>> pt.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.interact.interact_qres import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> fig = ishow_analysis(qreq_.ibs, cm, qreq_=qreq_)
        >>> pt.show_if_requested()
    """
    interact = InteractQres(ibs, cm, analysis=True, qreq_=qreq_, **kwargs)
    interact.show_page()
    interact.show()
    return interact


BASE_CLASS = abstract_interaction.AbstractInteraction


class InteractQres(BASE_CLASS):
    """
    Displays query chip, groundtruth matches, and top matches

    THERE IS A DIFFERENCE BETWEEN THIS AND MATCH INTERACTION. THIS IS FOR
    DISPLAYING THE RANKED LIST MATCH INTERACTION IS LOOKING AT A SINGLE PAIR

    SeeAlso:
        #interact_matches.MatchInteraction2
        #wbia.viz.interact.MatchInteraction
    """

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
        # if isinstance(self.cm, chip_match.ChipMatch):
        fig = self.cm.show_analysis(self.qreq_, **self.kwargs)
        # else:
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
        # if isinstance(self.cm, chip_match.ChipMatch):
        self.cm.ishow_match(self.qreq_, aid2, fnum=fnum_)
        # else:
        #    self.cm.ishow_matches(self.ibs, aid2, qreq_=self.qreq_, fnum=fnum_)
        self.draw()
        # self.bring_to_front()
        # fig = pt.gcf()
        # fig.canvas.draw()
        # pt.bring_to_front(fig)

    def on_click_outside(self, event):
        self.show_page()

    def on_click_inside(self, event, ax):
        ax = event.inaxes
        viztype = ph.get_plotdat(ax, 'viztype', '')
        # if verbose:
        #    print(str(event.__dict__))
        print('viztype=%r' % viztype)
        # Clicked a specific matches
        print('plodat_dict = ' + ut.repr2(ph.get_plotdat_dict(ax)))
        if viztype.startswith('chip'):
            from wbia.viz.interact import interact_chip

            options = interact_chip.build_annot_context_options(
                self.ibs,
                self.cm.qaid,
                refresh_func=self._analysis_view,
                with_interact_chip=False,
            )
            self.show_popup_menu(options, event)

        if viztype.startswith('matches') or viztype == 'multi_match':  # why startswith?
            aid2 = ph.get_plotdat(ax, 'aid2', None)
            aid_list = ph.get_plotdat(ax, 'aid_list', None)
            if event.button == 3:  # right-click
                # TODO; this functionality should be in viz.interact
                from wbia.gui import inspect_gui

                print('right click')
                print('qreq_ = %r' % (self.qreq_,))
                options = inspect_gui.get_aidpair_context_menu_options(
                    self.ibs,
                    self.cm.qaid,
                    aid2,
                    self.cm,
                    qreq_=self.qreq_,
                    update_callback=self.show_page,
                    backend_callback=None,
                    aid_list=aid_list,
                )
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


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.interact.interact_qres
        python -m wbia.viz.interact.interact_qres --allexamples
        python -m wbia.viz.interact.interact_qres --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
