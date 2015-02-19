"""
Single VsOne Chip Match Interface
For VsMany Interaction

Interaction for looking at matches between a single query and database annotation

Main development file

CommandLine:
    python -m ibeis.viz.interact.interact_matches --test-begin --show

    python -m ibeis.viz.interact.interact_matches --test-show_coverage --show
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import plottool as pt
import six
import guitool
from plottool import draw_func2 as df2
from plottool import viz_featrow
from plottool import interact_helpers as ih
from plottool import plot_helpers as ph
from ibeis import viz
from ibeis.model.hots import scoring
from ibeis.model.hots import chip_match
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_hough
from ibeis.viz import viz_chip
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
from ibeis.viz.interact.interact_chip import ishow_chip
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[interact_matches]', DEBUG=False)


def testdata_match_interact(**kwargs):
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_matches --test-testdata_match_interact --show

    Example:
        >>> from ibeis.viz.interact.interact_matches import *  # NOQA
        >>> import plottool as pt
        >>> kwargs = {}
        >>> self = testdata_match_interact(**kwargs)
        >>> pt.show_if_requested()
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    cfgdict = {}
    ibs, qreq_ = plh.get_pipeline_testdata(cfgdict=cfgdict, defaultdb='testdb1', cmdline_ok=True)
    qres = ibs.query_chips(qreq_=qreq_)[0]
    aid2 = None
    self = MatchInteraction(ibs, qres, aid2, annot_mode=1, dodraw=False, **kwargs)
    return self


@six.add_metaclass(ut.ReloadingMetaclass)
class MatchInteraction(object):
    """
    TODO: replace functional version with this class

    Plots a chip result and sets up callbacks for interaction.

    """
    def __init__(self, *args, **kwargs):
        self.init_old(*args, **kwargs)
        if not kwargs.get('nobegin', False):
            dodraw = kwargs.get('dodraw', True)
            self.begin(dodraw=dodraw)

    def init_old(self, ibs, qres, *args, **kwargs):
        """ old begin function for working with qres objects """
        if not isinstance(qres, chip_match.ChipMatch2):
            self.qres = qres
            cm = chip_match.ChipMatch2.from_qres(self.qres)
        else:
            cm = qres
            self.qres = qres
        self.init_new(ibs, cm, *args, **kwargs)

    def init_new(self, ibs, cm, aid2=None, fnum=None, figtitle='Inspect Query Result', same_fig=True, **kwargs):
        r"""
        new init function for use with ChipMatch2 class
        """
        self.ibs = ibs
        self.cm = cm
        self.fnum = pt.ensure_fnum(fnum)
        # Unpack Args
        if aid2 is None:
            index = 0
            cm.sortself()
            self.rank = index
        else:
            index = cm.daid2_idx[aid2]
            # TODO: rank?
            self.rank = None
        self.qaid  = self.cm.qaid
        self.daid  = self.cm.daid_list[index]
        self.fm    = self.cm.fm_list[index]
        self.fk    = self.cm.fk_list[index]
        self.fsv   = self.cm.fsv_list[index]
        self.fs    = None if self.cm.fs_list is None else self.cm.fs_list[index]
        self.score = None if self.cm.score_list is None else self.cm.score_list[index]
        self.H1    = None if self.cm.H_list is None else cm.H_list[index]
        # Read properties
        self.rchip1, self.rchip2 = ibs.get_annot_chips([self.qaid, self.daid])
        # Begin Interaction
        self.fig = ih.begin_interaction('matches', self.fnum)  # call doclf docla and make figure
        self.xywh2_ptr  = [None]
        self.annote_ptr = [kwargs.pop('mode', 0)]
        # New state vars
        self.same_fig = same_fig
        self.use_homog = False
        self.vert = kwargs.pop('vert', None)
        self.mx   = kwargs.pop('mx', None)
        self.last_fx = 0
        self.fnum2 = pt.next_fnum()
        self.figtitle = figtitle
        self.kwargs = kwargs

    def begin(self, dodraw=True):
        r"""
        CommandLine:
            python -m ibeis.viz.interact.interact_matches --test-begin
            python -m ibeis.viz.interact.interact_matches --test-begin --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact()
            >>> self.begin(dodraw=False)
            >>> pt.show_if_requested()
        """
        if self.mx is None:
            self.chipmatch_view()
        else:
            self.select_ith_match(self.mx)

        self.set_callbacks()
        # FIXME: this should probably not be called here
        if dodraw:
            ph.draw()  # ph-> adjust stuff draw -> fig_presenter.draw -> all figures show

    def set_callbacks(self):
        # TODO: view probchip
        toggle_samefig_key = 'Toggle same_fig'
        opt2_callback = [
            (toggle_samefig_key, self.toggle_samefig),
            ('Toggle vert', self.toggle_vert),
            ('query last feature', self.query_last_feature),
            ('show each chip', self.show_each_chip),
            ('show each distinctiveness chip', self.show_each_dstncvs_chip),
            ('show each foreground weight chip', self.show_each_fgweight_chip),
            ('show each probchip', self.show_each_probchip),
            ('show coverage', self.show_coverage),
            #('show each probchip', self.query_last_feature),
        ]
        opt2_callback.append(('name_interaction', self.name_interaction))
        if self.H1 is not None:
            opt2_callback.append(('Toggle homog', self.toggle_homog))
        if ut.is_developer():
            opt2_callback.append(('dev_reload', self.dev_reload))
            opt2_callback.append(('dev_embed', self.dev_embed))
        opt2_callback.append(('cancel', lambda: print('cancel')))
        guitool.connect_context_menu(self.fig.canvas, opt2_callback)
        ih.connect_callback(self.fig, 'button_press_event', self._click_matches_click)

    # Callback
    def _click_matches_click(self, event):
        aid       = self.daid
        fm        = self.fm
        qaid      = self.qaid
        ibs       = self.ibs
        xywh2_ptr = self.xywh2_ptr
        print_('[inter] clicked matches')
        if event is None:
            return
        button = event.button
        is_right_click = button == 3
        if is_right_click:
            return
        (x, y, ax) = (event.xdata, event.ydata, event.inaxes)
        # Out of axes click
        if None in [x, y, ax]:
            print('... out of axis')
            self.chipmatch_view()
            viz.draw()
            return
        else:
            viztype = vh.get_ibsdat(ax, 'viztype', '')
            print_('[ir] viztype=%r ' % viztype)
            key = '' if event.key is None else event.key
            print_('key=%r ' % key)
            ctrl_down = key.find('control') == 0
            # Click in match axes
            if viztype == 'matches' and ctrl_down:
                # Ctrl-Click
                print('.. control click')
                return self.sv_view()
            elif viztype == 'matches':
                if len(fm) == 0:
                    print('[inter] no feature matches to click')
                else:
                    # Normal Click
                    # Select nearest feature match to the click
                    kpts1, kpts2 = ibs.get_annot_kpts([qaid, aid])
                    kpts1_m = kpts1[fm[:, 0]]
                    kpts2_m = kpts2[fm[:, 1]]
                    x2, y2, w2, h2 = xywh2_ptr[0]
                    _mx1, _dist1 = ut.nearest_point(x, y, kpts1_m)
                    _mx2, _dist2 = ut.nearest_point(x - x2, y - y2, kpts2_m)
                    mx = _mx1 if _dist1 < _dist2 else _mx2
                    print('... clicked mx=%r' % mx)
                    self.select_ith_match(mx)
            elif viztype in ['warped', 'unwarped']:
                hs_aid = ax.__dict__.get('_hs_aid', None)
                hs_fx = ax.__dict__.get('_hs_fx', None)
                if hs_aid is not None and viztype == 'unwarped':
                    ishow_chip(ibs, hs_aid, fx=hs_fx, fnum=df2.next_fnum())
                elif hs_aid is not None and viztype == 'warped':
                    viz.show_keypoint_gradient_orientations(ibs, hs_aid, hs_fx, fnum=df2.next_fnum())
            else:
                print('...Unknown viztype: %r' % viztype)
            viz.draw()

    def chipmatch_view(self, pnum=(1, 1, 1), **kwargs_):
        """
        just visualizes the matches using some type of lines

        CommandLine:
            python -m ibeis.viz.interact.interact_matches --test-chipmatch_view --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact()
            >>> self.chipmatch_view()
            >>> pt.show_if_requested()
        """
        ibs      = self.ibs
        aid      = self.daid
        qaid     = self.qaid
        fnum     = self.fnum
        figtitle = self.figtitle
        annote_ptr = self.annote_ptr
        xywh2_ptr  = self.xywh2_ptr

        mode = annote_ptr[0]  # drawing mode draw: with/without lines/feats
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        show_matches_kw = self.kwargs.copy()
        show_matches_kw.update(
            dict(fnum=fnum, pnum=pnum, draw_lines=draw_lines, draw_ell=draw_ell,
                 colorbar_=True, vert=self.vert))
        show_matches_kw.update(kwargs_)

        if self.use_homog:
            show_matches_kw['H1'] = self.H1

        # TODO RENAME This to remove qres and rectify with show_matches
        OLD = False
        if OLD:
            qres = self.qres
            tup = viz.viz_matches.show_matches(ibs, qres, aid, **show_matches_kw)
        else:
            #show_matches_kw['score'] = self.score
            show_matches_kw['rawscore'] = self.score
            #ut.embed()
            show_matches_kw['aid2_raw_rank'] = self.rank
            tup = viz.viz_matches.show_matches2(ibs, self.qaid, self.daid, self.fm, self.fs, **show_matches_kw)

        ax, xywh1, xywh2 = tup
        xywh2_ptr[0] = xywh2

        df2.set_figtitle(figtitle + ' ' + vh.get_vsstr(qaid, aid))

    # Draw clicked selection
    def select_ith_match(self, mx):
        """
        Selects the ith match and visualizes and prints information concerning
        features weights, keypoint details, and sift descriptions

        Args:
            mx (int) - the ith match to visualize
            qaid (int) - query annotation id
            aid (int) - database annotation id

        CommandLine:
            python -m ibeis.viz.interact.interact_matches --test-select_ith_match --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact(mx=1)
            >>> pt.show_if_requested()
        """
        ibs        = self.ibs
        #qres       = self.qres
        qaid       = self.qaid
        aid        = self.daid
        fnum       = self.fnum
        figtitle   = self.figtitle
        annote_ptr = self.annote_ptr
        rchip1     = self.rchip1
        rchip2     = self.rchip2
        aid        = self.daid
        same_fig   = self.same_fig
        self.mx    = mx
        print('+--- SELECT --- ')
        print('qaid=%r, daid=%r' % (qaid, aid))
        print('... selecting mx-th=%r feature match' % mx)
        #print('qres.filtkey_list = %r' % (qres.filtkey_list,))
        #fsv = qres.aid2_fsv[aid]
        #fs  = qres.aid2_fs[aid]
        fm  = self.fm
        fsv = self.fsv
        fk  = self.fk
        fs  = self.fs
        print('score stats:')
        print(ut.get_stats_str(fsv, axis=0, newlines=True))
        print('fsv[mx] = %r' % (fsv[mx],))
        print('fs[mx] = %r' % (fs[mx],))
        """
        # test feature weights of actual chips
        fx1, fx2 = fm[mx]
        daid = aid
        ibs.get_annot_fgweights([daid])[0][fx2]
        ibs.get_annot_fgweights([qaid])[0][fx1]
        """
        #----------------------
        # Get info for the select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        fx1, fx2 = fm[mx]
        fscore2  = fs[mx]
        fk2      = fk[mx]
        kpts1, kpts2 = ibs.get_annot_kpts([self.qaid, self.daid])
        desc1, desc2 = ibs.get_annot_vecs([self.qaid, self.daid])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        #last_state.last_fx = fx1
        self.last_fx = fx1
        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, self.qaid, info1),
                          (rchip2, kp2, sift2, fx2, self.daid, info2)]
        # Normalizng Keypoint
        #if hasattr(qres, 'filt2_meta') and 'lnbnn' in qres.filt2_meta:
        #    qfx2_norm = qres.filt2_meta['lnbnn']
        #    # Normalizing chip and feature
        #    (aid3, fx3, normk) = qfx2_norm[fx1]
        #    rchip3 = ibs.get_annot_chips(aid3)
        #    kp3 = ibs.get_annot_kpts(aid3)[fx3]
        #    sift3 = ibs.get_annot_vecs(aid3)[fx3]
        #    info3 = '\nnorm %s k=%r' % (vh.get_aidstrs(aid3), normk)
        #    extracted_list.append((rchip3, kp3, sift3, fx3, aid3, info3))
        #else:
        #    pass
        # print('WARNING: meta doesnt exist')

        #----------------------
        # Draw the select_ith_match plot
        nRows, nCols = len(extracted_list) + same_fig, 3
        # Draw matching chips and features
        sel_fm = np.array([(fx1, fx2)])
        pnum1 = (nRows, 1, 1) if same_fig else (1, 1, 1)
        vert = self.vert if self.vert is not None else False
        self.chipmatch_view(pnum1, ell_alpha=.4, ell_linewidth=1.8,
                            colors=df2.BLUE, sel_fm=sel_fm, vert=vert)
        # Draw selected feature matches
        px = nCols * same_fig  # plot offset
        prevsift = None
        if not same_fig:
            #fnum2 = fnum + len(viz.FNUMS)
            fnum2 = self.fnum2
            fig2 = df2.figure(fnum=fnum2, docla=True, doclf=True)
        else:
            fnum2 = fnum
        for (rchip, kp, sift, fx, aid, info) in extracted_list:
            px = viz_featrow.draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                                           prevsift=prevsift, aid=aid, info=info)
            prevsift = sift
        if not same_fig:
            ih.connect_callback(fig2, 'button_press_event', self._click_matches_click)
            df2.set_figtitle(figtitle + vh.get_vsstr(qaid, aid))

    def sv_view(self, dodraw=True):
        """ spatial verification view

        """
        #fnum = viz.FNUMS['special']
        aid = self.daid
        fnum = pt.next_fnum()
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        ih.disconnect_callback(fig, 'button_press_event')
        viz.viz_sver.show_sver(self.ibs, self.qaid, aid2=aid, fnum=fnum)
        if dodraw:
            viz.draw()

    def show_coverage(self, dodraw=True):
        """
        CommandLine:
            python -m ibeis.viz.interact.interact_matches --test-show_coverage --show
            python -m ibeis.viz.interact.interact_matches --test-show_coverage

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact(mx=1)
            >>> self.show_coverage(dodraw=False)
            >>> pt.show_if_requested()
        """
        masks_list = scoring.get_masks(self.ibs, self.cm)
        scoring.show_coverage_mask(self.ibs, self.cm, masks_list)
        if dodraw:
            viz.draw()

    def show_each_chip(self):
        viz_chip.show_chip(self.ibs, self.qaid, fnum=pt.next_fnum(), nokpts=True)
        viz_chip.show_chip(self.ibs, self.daid, fnum=pt.next_fnum(), nokpts=True)
        viz.draw()

    def show_each_fgweight_chip(self):
        viz_chip.show_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
        viz_chip.show_chip(self.ibs, self.daid, fnum=pt.next_fnum())
        viz.draw()

    def show_each_dstncvs_chip(self, dodraw=True):
        """
        CommandLine:
            python -m ibeis.viz.interact.interact_matches --test-show_each_dstncvs_chip --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_matches import *  # NOQA
            >>> self = testdata_match_interact(mx=1)
            >>> self.show_each_dstncvs_chip(dodraw=False)
            >>> pt.show_if_requested()
        """
        dstncvs1, dstncvs2 = scoring.get_kpts_distinctiveness(self.ibs, [self.qaid, self.daid])
        print('dstncvs1_stats = ' + ut.get_stats_str(dstncvs1))
        print('dstncvs2_stats = ' + ut.get_stats_str(dstncvs2))
        weight_label = 'dstncvs'
        showkw = dict(weight_label=weight_label, ell=False, pts=True)
        viz_chip.show_chip(self.ibs, self.qaid, weights=dstncvs1, fnum=pt.next_fnum(), **showkw)
        viz_chip.show_chip(self.ibs, self.daid, weights=dstncvs2, fnum=pt.next_fnum(), **showkw)
        if dodraw:
            viz.draw()

    def show_each_probchip(self):
        viz_hough.show_probability_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
        viz_hough.show_probability_chip(self.ibs, self.daid, fnum=pt.next_fnum())
        viz.draw()

    def dev_reload(self):
        ih.disconnect_callback(self.fig, 'button_press_event')
        self.rrr()
        self.set_callbacks()

    def dev_embed(self):
        ut.embed()

    def name_interaction(self):
        from ibeis.viz.interact import interact_name
        aid1 = self.qaid  # 14
        aid2 = self.daid  # 5546
        self = interact_name.MatchVerificationInteraction(self.ibs, aid1, aid2)
        #ut.embed()

    def toggle_vert(self):
        self.vert = not self.vert
        if self.mx is not None:
            self.select_ith_match(self.mx)

    def toggle_homog(self):
        self.use_homog = not self.use_homog
        self.chipmatch_view()
        viz.draw()

    def toggle_samefig(self):
        self.same_fig = not self.same_fig
        if self.mx is not None:
            self.select_ith_match(self.mx)

    def query_last_feature(self):
        ibs      = self.ibs
        qaid     = self.qaid
        viz.show_nearest_descriptors(ibs, qaid, self.last_fx, df2.next_fnum())
        fig3 = df2.gcf()
        ih.connect_callback(fig3, 'button_press_event', self._click_matches_click)
        viz.draw()
        #df2.update()

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_matches
        python -m ibeis.viz.interact.interact_matches --allexamples
        python -m ibeis.viz.interact.interact_matches --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
