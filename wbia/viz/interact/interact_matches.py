# -*- coding: utf-8 -*-
"""
Single VsOne Chip Match Interface
For VsMany Interaction

Interaction for looking at matches between a single query and database annotation

Main development file

CommandLine:
    python -m wbia.viz.interact.interact_matches --test-show_coverage --show
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import wbia.plottool as pt
import six
from wbia.plottool import interact_helpers as ih
from wbia import viz
from wbia.algo.hots import scoring
from wbia.algo.hots import hstypes
from wbia.viz import viz_helpers as vh
from wbia.viz import viz_hough
from wbia.viz import viz_chip
from wbia.plottool import interact_matches
from wbia.viz.interact.interact_chip import ishow_chip

(print, rrr, profile) = ut.inject2(__name__, '[interact_matches]')


def testdata_match_interact(**kwargs):
    """
    CommandLine:
        python -m wbia.viz.interact.interact_matches --test-testdata_match_interact --show --db PZ_MTEST --qaid 3

    Example:
        >>> # VIZ_DOCTEST
        >>> from wbia.viz.interact.interact_matches import *  # NOQA
        >>> import wbia.plottool as pt
        >>> kwargs = {}
        >>> mx = ut.get_argval('--mx', type_=int, default=None)
        >>> self = testdata_match_interact(mx=mx, **kwargs)
        >>> pt.show_if_requested()
    """
    import wbia

    qreq_ = wbia.testdata_qreq_(defaultdb='testdb1', t=['default:Knorm=3'])
    ibs = qreq_.ibs
    cm = qreq_.execute()[0]
    cm.sortself()
    aid2 = None
    self = MatchInteraction(ibs, cm, aid2, mode=1, dodraw=False, qreq_=qreq_, **kwargs)
    self.start()
    return self


# TODO inherit from AbstractInteraction
@six.add_metaclass(ut.ReloadingMetaclass)
class MatchInteraction(interact_matches.MatchInteraction2):
    """
    Plots a chip result and sets up callbacks for interaction.

    SeeAlso:
        plottool.interact_matches.MatchInteraction2

    CommandLine:
        python -m wbia.viz.interact.interact_matches --test-testdata_match_interact --show --db PZ_MTEST --qaid 3
    """

    def __init__(
        self,
        ibs,
        cm,
        aid2=None,
        fnum=None,
        qreq_=None,
        figtitle='Match Interaction',
        **kwargs,
    ):
        # print('[ibs] MatchInteraction.__init__')
        self.ibs = ibs
        self.cm = cm
        self.qreq_ = qreq_
        # Unpack Args
        if aid2 is None:
            index = 0
            # FIXME: no sortself
            cm.sortself()
            self.rank = index
        else:
            index = cm.daid2_idx.get(aid2, None)
            # TODO: rank?
            self.rank = None
        if index is not None:
            self.qaid = self.cm.qaid
            self.daid = self.cm.daid_list[index]
            fm = self.cm.fm_list[index]
            fk = self.cm.fk_list[index]
            fsv = self.cm.fsv_list[index]
            if self.cm.fs_list is None:
                fs_list = self.cm.get_fsv_prod_list()
            else:
                fs_list = self.cm.fs_list
            fs = None if fs_list is None else fs_list[index]
            H1 = None if self.cm.H_list is None else cm.H_list[index]
            self.score = None if self.cm.score_list is None else self.cm.score_list[index]
        else:
            self.qaid = self.cm.qaid
            self.daid = aid2
            fm = np.empty((0, 2), dtype=hstypes.FM_DTYPE)
            fk = np.empty(0, dtype=hstypes.FK_DTYPE)
            fsv = np.empty((0, 2), dtype=hstypes.FS_DTYPE)
            fs = np.empty(0, dtype=hstypes.FS_DTYPE)
            H1 = None
            self.score = None

        # Read properties
        self.query_config2_ = (
            None if self.qreq_ is None else self.qreq_.extern_query_config2
        )
        self.data_config2_ = (
            None if self.qreq_ is None else self.qreq_.extern_data_config2
        )

        rchip1 = vh.get_chips(ibs, [self.qaid], config2_=self.query_config2_)[0]
        rchip2 = vh.get_chips(ibs, [self.daid], config2_=self.data_config2_)[0]

        kpts1 = ibs.get_annot_kpts([self.qaid], config2_=self.query_config2_)[0]
        kpts2 = ibs.get_annot_kpts([self.daid], config2_=self.data_config2_)[0]

        vecs1 = ibs.get_annot_vecs([self.qaid], config2_=self.query_config2_)[0]
        vecs2 = ibs.get_annot_vecs([self.daid], config2_=self.data_config2_)[0]

        self.figtitle = figtitle
        self.kwargs = kwargs
        self.fnum2 = pt.next_fnum()

        super(MatchInteraction, self).__init__(
            rchip1,
            rchip2,
            kpts1,
            kpts2,
            fm,
            fs,
            fsv,
            vecs1,
            vecs2,
            H1,
            H2=None,
            fk=fk,
            fnum=fnum,
            **kwargs,
        )

    # def plot(self, fnum, pnum):
    def chipmatch_view(self, fnum=None, pnum=(1, 1, 1), verbose=None, **kwargs_):
        """
        just visualizes the matches using some type of lines

        CommandLine:
            python -m wbia.viz.interact.interact_matches --test-chipmatch_view --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact()
            >>> self.chipmatch_view()
            >>> pt.show_if_requested()
        """
        if fnum is None:
            fnum = self.fnum
        if verbose is None:
            verbose = ut.VERBOSE

        ibs = self.ibs
        aid = self.daid
        qaid = self.qaid
        figtitle = self.figtitle

        # drawing mode draw: with/without lines/feats
        mode = kwargs_.get('mode', self.mode)
        draw_ell = mode >= 1
        draw_lines = mode == 2
        # self.mode = (self.mode + 1) % 3
        pt.figure(fnum=fnum, docla=True, doclf=True)
        show_matches_kw = self.kwargs.copy()
        show_matches_kw.update(
            dict(
                fnum=fnum,
                pnum=pnum,
                draw_lines=draw_lines,
                draw_ell=draw_ell,
                colorbar_=True,
                vert=self.vert,
            )
        )
        show_matches_kw.update(kwargs_)

        if self.warp_homog:
            show_matches_kw['H1'] = self.H1

        # show_matches_kw['score'] = self.score
        show_matches_kw['rawscore'] = self.score
        show_matches_kw['aid2_raw_rank'] = self.rank
        tup = viz.viz_matches.show_matches2(
            ibs,
            self.qaid,
            self.daid,
            self.fm,
            self.fs,
            qreq_=self.qreq_,
            **show_matches_kw,
        )
        ax, xywh1, xywh2 = tup
        self.xywh2 = xywh2

        pt.set_figtitle(figtitle + ' ' + vh.get_vsstr(qaid, aid))

    def sv_view(self, dodraw=True):
        """ spatial verification view

        """
        # fnum = viz.FNUMS['special']
        aid = self.daid
        fnum = pt.next_fnum()
        fig = pt.figure(fnum=fnum, docla=True, doclf=True)
        ih.disconnect_callback(fig, 'button_press_event')
        viz.viz_sver.show_sver(self.ibs, self.qaid, aid2=aid, fnum=fnum)
        if dodraw:
            # self.draw()
            pt.draw()

    def show_coverage(self, dodraw=True):
        """
        CommandLine:
            python -m wbia.viz.interact.interact_matches --test-show_coverage --show
            python -m wbia.viz.interact.interact_matches --test-show_coverage

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact(mx=1)
            >>> self.show_coverage(dodraw=False)
            >>> pt.show_if_requested()
        """
        masks_list = scoring.get_masks(self.qreq_, self.cm)
        scoring.show_coverage_mask(self.qreq_, self.cm, masks_list)
        if dodraw:
            # self.draw()
            pt.draw()

    def show_each_chip(self):
        viz_chip.show_chip(self.ibs, self.qaid, fnum=pt.next_fnum(), nokpts=True)
        viz_chip.show_chip(self.ibs, self.daid, fnum=pt.next_fnum(), nokpts=True)
        pt.draw()
        # self.draw()

    def show_each_fgweight_chip(self):
        viz_chip.show_chip(
            self.ibs, self.qaid, fnum=pt.next_fnum(), weight_label='fg_weights'
        )
        viz_chip.show_chip(
            self.ibs, self.daid, fnum=pt.next_fnum(), weight_label='fg_weights'
        )
        # self.draw()
        pt.draw()

    def show_each_dstncvs_chip(self, dodraw=True):
        """
        CommandLine:
            python -m wbia.viz.interact.interact_matches --test-show_each_dstncvs_chip --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact(mx=1)
            >>> self.show_each_dstncvs_chip(dodraw=False)
            >>> pt.show_if_requested()
        """
        dstncvs1, dstncvs2 = scoring.get_kpts_distinctiveness(
            self.ibs, [self.qaid, self.daid]
        )
        print('dstncvs1_stats = ' + ut.get_stats_str(dstncvs1))
        print('dstncvs2_stats = ' + ut.get_stats_str(dstncvs2))
        weight_label = 'dstncvs'
        showkw = dict(weight_label=weight_label, ell=False, pts=True)
        viz_chip.show_chip(
            self.ibs, self.qaid, weights=dstncvs1, fnum=pt.next_fnum(), **showkw
        )
        viz_chip.show_chip(
            self.ibs, self.daid, weights=dstncvs2, fnum=pt.next_fnum(), **showkw
        )
        if dodraw:
            # self.draw()
            pt.draw()

    def show_each_probchip(self):
        viz_hough.show_probability_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
        viz_hough.show_probability_chip(self.ibs, self.daid, fnum=pt.next_fnum())
        pt.draw()
        # self.draw()

    def dev_reload(self):
        ih.disconnect_callback(self.fig, 'button_press_event')
        self.rrr()
        self.set_callbacks()

    def dev_embed(self):
        ut.embed()

    def toggle_samefig(self):
        self.same_fig = not self.same_fig
        if self.mx is not None:
            self.select_ith_match(self.mx)
        self.draw()

    def query_last_feature(self):
        ibs = self.ibs
        qaid = self.qaid
        viz.show_nearest_descriptors(
            ibs, qaid, self.last_fx, pt.next_fnum(), qreq_=self.qreq_, draw_chip=True
        )
        fig3 = pt.gcf()
        ih.connect_callback(fig3, 'button_press_event', self.on_click)
        pt.draw()

    def get_popup_options(self):
        from wbia.gui import inspect_gui

        options = []

        ax = pt.gca()  # HACK

        from wbia.plottool import plot_helpers as ph

        viztype = ph.get_plotdat(ax, 'viztype', '')
        is_match_type = viztype in ['matches', 'multi_match']

        if is_match_type:
            options += inspect_gui.get_aidpair_context_menu_options(
                self.ibs,
                self.qaid,
                self.daid,
                self.cm,
                qreq_=self.qreq_,
                # update_callback=self.show_page,
                # backend_callback=None, aid_list=aid_list)
            )

        options += [
            # ('Toggle same_fig', self.toggle_samefig),
            # ('Toggle vert', self.toggle_vert),
            ('query last feature', self.query_last_feature),
            ('show each chip', self.show_each_chip),
            ('show each distinctiveness chip', self.show_each_dstncvs_chip),
            ('show each foreground weight chip', self.show_each_fgweight_chip),
            ('show each probchip', self.show_each_probchip),
            ('show coverage', self.show_coverage),
            # ('show each probchip', self.query_last_feature),
        ]

        # options.append(('name_interaction', self.name_interaction))
        # if self.H1 is not None:
        #    options.append(('Toggle homog', self.toggle_homog))
        if ut.is_developer():
            options.append(('dev_reload', self.dev_reload))
            options.append(('dev_embed', self.dev_embed))
        # options.append(('cancel', lambda: print('cancel')))
        options += super(MatchInteraction, self).get_popup_options()

        return options
        # self.show_popup_menu(options, event)

    # Callback
    def on_click_inside(self, event, ax):
        from wbia.plottool import plot_helpers as ph

        ibs = self.ibs
        viztype = ph.get_plotdat(ax, 'viztype', '')
        is_match_type = viztype in ['matches', 'multi_match']

        key = '' if event.key is None else event.key
        print('key=%r ' % key)
        ctrl_down = key.find('control') == 0
        # Click in match axes
        if event.button == 3:
            return super(MatchInteraction, self).on_click_inside(event, ax)
        if is_match_type and ctrl_down:
            # Ctrl-Click
            print('.. control click')
            return self.sv_view()
        elif viztype in ['warped', 'unwarped']:
            print('clicked at patch')
            ut.print_dict(ph.get_plotdat_dict(ax))
            hs_aid = {'aid1': self.qaid, 'aid2': self.daid}[
                vh.get_ibsdat(ax, 'aid', None)
            ]
            hs_fx = vh.get_ibsdat(ax, 'fx', None)
            print('hs_fx = %r' % (hs_fx,))
            print('hs_aid = %r' % (hs_aid,))
            if hs_aid is not None and viztype == 'unwarped':
                ishow_chip(ibs, hs_aid, fx=hs_fx, fnum=pt.next_fnum())
            elif hs_aid is not None and viztype == 'warped':
                viz.show_keypoint_gradient_orientations(
                    ibs, hs_aid, hs_fx, fnum=pt.next_fnum()
                )
        else:
            return super(MatchInteraction, self).on_click_inside(event, ax)
        self.draw()


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.interact.interact_matches
        python -m wbia.viz.interact.interact_matches --allexamples
        python -m wbia.viz.interact.interact_matches --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
