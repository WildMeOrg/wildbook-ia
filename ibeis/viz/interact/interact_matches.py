"""
Interaction for looking at matches between two queries
"""
from __future__ import absolute_import, division, print_function
import utool
import guitool
import numpy as np
from plottool import draw_func2 as df2
import plottool as pt
import six
from ibeis import viz
import utool as ut
from plottool.viz_featrow import draw_feat_row
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih
from .interact_chip import ishow_chip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[interact_matches]', DEBUG=False)


class LastState(object):
    def __init__(last_state):
        last_state.same_fig = None
        last_state.last_fx = None


@six.add_metaclass(ut.ReloadingMetaclass)
class MatchInteraction(object):
    """
    TODO: replace functional version with this class

    Plots a chip result and sets up callbacks for interaction.
    """
    def __init__(self, *args, **kwargs):
        self.begin(*args, **kwargs)

    def begin(self, ibs, qres, aid=None, fnum=4,
              figtitle='Inspect Query Result', same_fig=True, **kwargs):
        fig = ih.begin_interaction('matches', fnum)  # call doclf docla and make figure
        qaid = qres.qaid
        if aid is None:
            aid = qres.get_top_aids(num=1)[0]
        rchip1, rchip2 = ibs.get_annot_chips([qaid, aid])
        fm = qres.aid2_fm[aid]
        mx = kwargs.pop('mx', None)
        xywh2_ptr = [None]
        annote_ptr = [kwargs.pop('mode', 0)]
        last_state = LastState()
        last_state.same_fig = same_fig
        last_state.last_fx = 0

        # SET CLOSURE VARS
        self.ibs      = ibs
        self.qres     = qres
        self.qaid     = qres.qaid
        self.aid      = aid
        self.fnum     = fnum
        self.figtitle = figtitle
        self.same_fig = same_fig
        self.kwargs   = kwargs
        self.fig = fig
        self.last_state = last_state
        self.fig        = fig
        self.annote_ptr = annote_ptr
        self.xywh2_ptr  = xywh2_ptr
        self.fm         = fm
        self.rchip1     = rchip1
        self.rchip2     = rchip2

        if mx is None:
            self.chipmatch_view()
        else:
            self.select_ith_match(mx)

        toggle_samefig_key = 'Toggle same_fig (currently %r)' % same_fig

        opt2_callback = [
            (toggle_samefig_key, self.toggle_samefig),
            ('query last feature', self.query_last_feature),
            ('cancel', lambda: print('cancel')), ]
        guitool.connect_context_menu(fig.canvas, opt2_callback)
        ih.connect_callback(fig, 'button_press_event', self._click_matches_click)
        # FIXME: this should probably not be called here
        viz.draw()  # ph-> adjust stuff draw -> fig_presenter.draw -> all figures show

    def chipmatch_view(self, pnum=(1, 1, 1), **kwargs):
        """
        just visualizes the matches using some type of lines
        """
        # <CLOSURE VARS>
        ibs      = self.ibs
        qres     = self.qres
        aid      = self.aid
        fnum     = self.fnum
        figtitle = self.figtitle
        #kwargs   = self.kwargs
        annote_ptr = self.annote_ptr
        xywh2_ptr  = self.xywh2_ptr
        # </CLOSURE VARS>
        qaid = qres.qaid

        mode = annote_ptr[0]  # drawing mode draw: with/without lines/feats
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        # TODO RENAME This to remove qres and rectify with show_matches
        tup = viz.show_matches(ibs, qres, aid, fnum=fnum, pnum=pnum,
                               draw_lines=draw_lines, draw_ell=draw_ell,
                               colorbar_=True, **kwargs)
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
        """
        # <CLOSURE VARS>
        ibs        = self.ibs
        qres       = self.qres
        aid        = self.aid
        fnum       = self.fnum
        figtitle   = self.figtitle
        kwargs     = self.kwargs
        same_fig   = self.same_fig
        last_state = self.last_state
        annote_ptr = self.annote_ptr
        rchip1     = self.rchip1
        rchip2     = self.rchip2
        aid        = self.aid
        # </CLOSURE VARS>
        qaid = qres.qaid
        print('+--- SELECT --- ')
        print('qaid=%r, daid=%r' % (qaid, aid))
        print('... selecting mx-th=%r feature match' % mx)
        print('qres.filtkey_list = %r' % (qres.filtkey_list,))
        fsv = qres.aid2_fsv[aid]
        fs  = qres.aid2_fs[aid]
        print('fsv[mx] = %r' % (fsv[mx],))
        print('fs[mx] = %r' % (fs[mx],))

        #----------------------
        # Get info for the select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        aid1, aid2 = qaid, aid
        fx1, fx2 = qres.aid2_fm[aid2][mx]

        # Older info
        fscore2  = qres.aid2_fs[aid2][mx]
        fk2      = qres.aid2_fk[aid2][mx]
        kpts1, kpts2 = ibs.get_annot_kpts([aid1, aid2])
        desc1, desc2 = ibs.get_annot_vecs([aid1, aid2])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        last_state.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, aid1, info1),
                          (rchip2, kp2, sift2, fx2, aid2, info2)]
        # Normalizng Keypoint
        if hasattr(qres, 'filt2_meta') and 'lnbnn' in qres.filt2_meta:
            qfx2_norm = qres.filt2_meta['lnbnn']
            # Normalizing chip and feature
            (aid3, fx3, normk) = qfx2_norm[fx1]
            rchip3 = ibs.get_annot_chips(aid3)
            kp3 = ibs.get_annot_kpts(aid3)[fx3]
            sift3 = ibs.get_annot_vecs(aid3)[fx3]
            info3 = '\nnorm %s k=%r' % (vh.get_aidstrs(aid3), normk)
            extracted_list.append((rchip3, kp3, sift3, fx3, aid3, info3))
        else:
            pass
            #print('WARNING: meta doesnt exist')

        #----------------------
        # Draw the select_ith_match plot
        nRows, nCols = len(extracted_list) + same_fig, 3
        # Draw matching chips and features
        sel_fm = np.array([(fx1, fx2)])
        pnum1 = (nRows, 1, 1) if same_fig else (1, 1, 1)
        self.chipmatch_view(pnum1, vert=False, ell_alpha=.4, ell_linewidth=1.8,
                             colors=df2.BLUE, sel_fm=sel_fm, **kwargs)
        # Draw selected feature matches
        px = nCols * same_fig  # plot offset
        prevsift = None
        if not same_fig:
            fnum2 = fnum + len(viz.FNUMS)
            fig2 = df2.figure(fnum=fnum2, docla=True, doclf=True)
        else:
            fnum2 = fnum
        for (rchip, kp, sift, fx, aid, info) in extracted_list:
            px = draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                               prevsift=prevsift, aid=aid, info=info)
            prevsift = sift
        if not same_fig:
            ih.connect_callback(fig2, 'button_press_event', self._click_matches_click)
            df2.set_figtitle(figtitle + vh.get_vsstr(qaid, aid))

    # Draw ctrl clicked selection
    def sv_view(self):
        """ spatial verification view """
        #fnum = viz.FNUMS['special']
        aid = self.aid
        fnum = pt.next_fnum()
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        ih.disconnect_callback(fig, 'button_press_event')
        viz.show_sv(self.ibs, self.qres.qaid, aid2=aid, fnum=fnum)
        viz.draw()

    # Callback
    def _click_matches_click(self, event):
        aid       = self.aid
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
                _mx1, _dist1 = utool.nearest_point(x, y, kpts1_m)
                _mx2, _dist2 = utool.nearest_point(x - x2, y - y2, kpts2_m)
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

    def toggle_samefig(self):
        ibs      = self.ibs
        qres     = self.qres
        aid      = self.aid
        fnum     = self.fnum
        figtitle = self.figtitle
        same_fig = self.same_fig
        # FIXME: Do not do recursive calls
        ishow_matches(ibs, qres, aid=aid, fnum=fnum, figtitle=figtitle, same_fig=not same_fig, **self.kwargs)

    def query_last_feature(self):
        ibs      = self.ibs
        qaid     = self.qaid
        viz.show_nearest_descriptors(ibs, qaid, self.last_state.last_fx, df2.next_fnum())
        fig3 = df2.gcf()
        ih.connect_callback(fig3, 'button_press_event', self._click_matches_click)
        df2.update()


# ===========================
# DEPRICATE IN FAVOR OF CLASS
# VVVVVVVVVVVVVVVVVVVVVVVVVVV

#def ishow_matches(*args, **kwargs):
#    match_interaction = MatchInteraction(*args, **kwargs)
#    return match_interaction.fig


def ishow_matches(ibs, qres, aid=None, fnum=4, figtitle='Inspect Query Result',
                  same_fig=True, **kwargs):
    """
    TODO: make this interaction a class instead of closure

    Plots a chip result and sets up callbacks for interaction.
    """
    fig = ih.begin_interaction('matches', fnum)  # call doclf docla and make figure
    qaid = qres.qaid
    if aid is None:
        aid = qres.get_top_aids(num=1)[0]
    rchip1, rchip2 = ibs.get_annot_chips([qaid, aid])
    fm = qres.aid2_fm[aid]
    mx = kwargs.pop('mx', None)
    xywh2_ptr = [None]
    annote_ptr = [kwargs.pop('mode', 0)]
    last_state = LastState()
    last_state.same_fig = same_fig
    last_state.last_fx = 0

    # Draw default
    def _chipmatch_view(pnum=(1, 1, 1), **kwargs):
        """
        just visualizes the matches using some type of lines
        """
        mode = annote_ptr[0]  # drawing mode draw: with/without lines/feats
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        # TODO RENAME This to remove qres and rectify with show_matches
        tup = viz.show_matches(ibs, qres, aid, fnum=fnum, pnum=pnum,
                               draw_lines=draw_lines, draw_ell=draw_ell,
                               colorbar_=True, **kwargs)
        ax, xywh1, xywh2 = tup
        xywh2_ptr[0] = xywh2

        df2.set_figtitle(figtitle + ' ' + vh.get_vsstr(qaid, aid))

    # Draw clicked selection
    def _select_ith_match(mx, qaid, aid):
        """
        Selects the ith match and visualizes and prints information concerning
        features weights, keypoint details, and sift descriptions

        Args:
            mx (int) - the ith match to visualize
            qaid (int) - query annotation id
            aid (int) - database annotation id
        """
        print('... selecting mx-th=%r feature match' % mx)
        print('qres.filtkey_list = %r' % (qres.filtkey_list,))
        fsv = qres.aid2_fsv[aid]
        print('fsv[mx] = %r' % (fsv[mx],))

        #----------------------
        # Get info for the _select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        aid1, aid2 = qaid, aid
        fx1, fx2 = fm[mx]
        fscore2  = qres.aid2_fs[aid2][mx]
        fk2      = qres.aid2_fk[aid2][mx]
        kpts1, kpts2 = ibs.get_annot_kpts([aid1, aid2])
        desc1, desc2 = ibs.get_annot_vecs([aid1, aid2])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        last_state.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, aid1, info1),
                          (rchip2, kp2, sift2, fx2, aid2, info2)]
        # Normalizng Keypoint
        if hasattr(qres, 'filt2_meta') and 'lnbnn' in qres.filt2_meta:
            qfx2_norm = qres.filt2_meta['lnbnn']
            # Normalizing chip and feature
            (aid3, fx3, normk) = qfx2_norm[fx1]
            rchip3 = ibs.get_annot_chips(aid3)
            kp3 = ibs.get_annot_kpts(aid3)[fx3]
            sift3 = ibs.get_annot_vecs(aid3)[fx3]
            info3 = '\nnorm %s k=%r' % (vh.get_aidstrs(aid3), normk)
            extracted_list.append((rchip3, kp3, sift3, fx3, aid3, info3))
        else:
            print('WARNING: meta doesnt exist')

        #----------------------
        # Draw the _select_ith_match plot
        nRows, nCols = len(extracted_list) + same_fig, 3
        # Draw matching chips and features
        sel_fm = np.array([(fx1, fx2)])
        pnum1 = (nRows, 1, 1) if same_fig else (1, 1, 1)
        _chipmatch_view(pnum1, vert=False, ell_alpha=.4, ell_linewidth=1.8,
                        colors=df2.BLUE, sel_fm=sel_fm, **kwargs)
        # Draw selected feature matches
        px = nCols * same_fig  # plot offset
        prevsift = None
        if not same_fig:
            fnum2 = fnum + len(viz.FNUMS)
            fig2 = df2.figure(fnum=fnum2, docla=True, doclf=True)
        else:
            fnum2 = fnum
        for (rchip, kp, sift, fx, aid, info) in extracted_list:
            px = draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                               prevsift=prevsift, aid=aid, info=info)
            prevsift = sift
        if not same_fig:
            ih.connect_callback(fig2, 'button_press_event', _click_matches_click)
            df2.set_figtitle(figtitle + vh.get_vsstr(qaid, aid))

    # Draw ctrl clicked selection
    def _sv_view(aid):
        """ spatial verification view """
        #fnum = viz.FNUMS['special']
        fnum = pt.next_fnum()
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        ih.disconnect_callback(fig, 'button_press_event')
        viz.show_sv(ibs, qres.qaid, aid2=aid, fnum=fnum)
        viz.draw()

    # Callback
    def _click_matches_click(event):
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
            _chipmatch_view()
            viz.draw()
            return
        viztype = vh.get_ibsdat(ax, 'viztype', '')
        print_('[ir] viztype=%r ' % viztype)
        key = '' if event.key is None else event.key
        print_('key=%r ' % key)
        ctrl_down = key.find('control') == 0
        # Click in match axes
        if viztype == 'matches' and ctrl_down:
            # Ctrl-Click
            print('.. control click')
            return _sv_view(aid)
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
                _mx1, _dist1 = utool.nearest_point(x, y, kpts1_m)
                _mx2, _dist2 = utool.nearest_point(x - x2, y - y2, kpts2_m)
                mx = _mx1 if _dist1 < _dist2 else _mx2
                print('... clicked mx=%r' % mx)
                _select_ith_match(mx, qaid, aid)
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

    if mx is None:
        _chipmatch_view()
    else:
        _select_ith_match(mx, qaid, aid)

    def toggle_samefig():
        # FIXME: Do not do recursive calls
        ishow_matches(ibs, qres, aid=aid, fnum=fnum, figtitle=figtitle, same_fig=not same_fig, **kwargs)

    def query_last_feature():
        viz.show_nearest_descriptors(ibs, qaid, last_state.last_fx, df2.next_fnum())
        fig3 = df2.gcf()
        ih.connect_callback(fig3, 'button_press_event', _click_matches_click)
        df2.update()

    toggle_samefig_key = 'Toggle same_fig (currently %r)' % same_fig

    opt2_callback = [
        (toggle_samefig_key, toggle_samefig),
        ('query last feature', query_last_feature),
        ('cancel', lambda: print('cancel')), ]
    guitool.connect_context_menu(fig.canvas, opt2_callback)
    ih.connect_callback(fig, 'button_press_event', _click_matches_click)
    # FIXME: this should probably not be called here
    viz.draw()  # ph-> adjust stuff draw -> fig_presenter.draw -> all figures show
    return fig
