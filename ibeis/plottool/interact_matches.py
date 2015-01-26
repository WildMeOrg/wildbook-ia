"""
Unfinished non-ibeis dependent version of interact matches
"""


# TODO: move to plottool and decouple with IBEIS
@six.add_metaclass(ut.ReloadingMetaclass)
class MatchInteraction2(object):
    """
    TODO: replace functional version with this class

    Plots a chip result and sets up callbacks for interaction.

    """
    def __init__(self, rchip1, rchip2, kpts1, kpts2, fm, fs, fsv, vecs1, vecs2, *args, **kwargs):
        self.rchip1 = rchip1
        self.rchip2 = rchip2
        self.kpts1 = kpts1
        self.kpts2 = kpts2
        self.fm = fm
        self.fs = fs
        self.fsv = fsv
        self.vecs1 = vecs1
        self.vecs2 = vecs2
        self.begin(*args, **kwargs)

    def begin(self, fnum=None, figtitle='Inspect Matches', same_fig=True, **kwargs):
        import plottool as pt
        from plottool import interact_helpers as ih
        from plottool import plot_helpers as ph
        if fnum is None:
            fnum = pt.next_fnum()
        fig = ih.begin_interaction('matches', fnum)  # call doclf docla and make figure

        rchip1, rchip2 = None, None
        fm = None
        mx = kwargs.pop('mx', None)
        xywh2_ptr = [None]
        annote_ptr = [kwargs.pop('mode', 0)]
        self.same_fig = same_fig
        self.last_fx = 0

        # New state vars
        self.vert = kwargs.pop('vert', None)
        self.mx = None

        # SET CLOSURE VARS
        self.fnum     = fnum
        self.fnum2    = pt.next_fnum()
        self.figtitle = figtitle
        self.same_fig = same_fig
        self.kwargs   = kwargs
        self.fig = fig
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

        self.set_callbacks()
        # FIXME: this should probably not be called here
        ph.draw()  # ph-> adjust stuff draw -> fig_presenter.draw -> all figures show

    def chipmatch_view(self, pnum=(1, 1, 1), **kwargs_):
        """
        just visualizes the matches using some type of lines
        """
        import plottool as pt
        # <CLOSURE VARS>
        fnum     = self.fnum
        #figtitle = self.figtitle
        #kwargs   = self.kwargs
        annote_ptr = self.annote_ptr
        #xywh2_ptr  = self.xywh2_ptr
        # </CLOSURE VARS>

        mode = annote_ptr[0]  # drawing mode draw: with/without lines/feats
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        pt.figure(fnum=fnum, docla=True, doclf=True)
        show_matches_kw = self.__dict__.copy()
        show_matches_kw.update(self.kwargs)
        show_matches_kw.update(
            dict(fnum=fnum, pnum=pnum, draw_lines=draw_lines, draw_ell=draw_ell,
                 colorbar_=True, vert=self.vert))
        show_matches_kw.update(kwargs_)

        #tup = show_matches(fm, fs, **show_matches_kw)
        #ax, xywh1, xywh2 = tup
        #xywh2_ptr[0] = xywh2

        #pt.set_figtitle(figtitle + ' ' + vh.get_vsstr(qaid, aid))

    # Draw clicked selection
    def select_ith_match(self, mx):
        """
        Selects the ith match and visualizes and prints information concerning
        features weights, keypoint details, and sift descriptions
        """
        import plottool as pt
        from plottool import viz_featrow
        from plottool import interact_helpers as ih
        # <CLOSURE VARS>
        fnum       = self.fnum
        #figtitle   = self.figtitle
        same_fig   = self.same_fig
        annote_ptr = self.annote_ptr
        rchip1     = self.rchip1
        rchip2     = self.rchip2
        # </CLOSURE VARS>
        self.mx    = mx
        print('+--- SELECT --- ')
        print('... selecting mx-th=%r feature match' % mx)
        fsv = self.fsv  # qres.aid2_fsv[aid]
        fs  = self.fs  # qres.aid2_fs[aid]
        print('score stats:')
        print(ut.get_stats_str(fsv, axis=0, newlines=True))
        print('fsv[mx] = %r' % (fsv[mx],))
        print('fs[mx] = %r' % (fs[mx],))
        #----------------------
        # Get info for the select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        fx1, fx2 = None, None  # qres.aid2_fm[aid2][mx]

        # Older info
        fscore2  = None, None  # qres.aid2_fs[aid2][mx]
        fk2      = None  # qres.aid2_fk[aid2][mx]
        kpts1, kpts2 = None, None  # ibs.get_annot_kpts([aid1, aid2])
        desc1, desc2 = None, None  # ibs.get_annot_vecs([aid1, aid2])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        #self.last_fx = fx1
        self.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, 'aid1', info1),
                          (rchip2, kp2, sift2, fx2, 'aid2', info2)]
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
        #    #print('WARNING: meta doesnt exist')

        #----------------------
        # Draw the select_ith_match plot
        nRows, nCols = len(extracted_list) + same_fig, 3
        # Draw matching chips and features
        sel_fm = np.array([(fx1, fx2)])
        pnum1 = (nRows, 1, 1) if same_fig else (1, 1, 1)
        vert = self.vert if self.vert is not None else False
        self.chipmatch_view(pnum1, ell_alpha=.4, ell_linewidth=1.8,
                            colors=pt.BLUE, sel_fm=sel_fm, vert=vert)
        # Draw selected feature matches
        px = nCols * same_fig  # plot offset
        prevsift = None
        if not same_fig:
            #fnum2 = fnum + len(viz.FNUMS)
            fnum2 = self.fnum2
            fig2 = pt.figure(fnum=fnum2, docla=True, doclf=True)
        else:
            fnum2 = fnum
        for (rchip, kp, sift, fx, aid, info) in extracted_list:
            px = viz_featrow.draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                                           prevsift=prevsift, aid=aid, info=info)
            prevsift = sift
        if not same_fig:
            ih.connect_callback(fig2, 'button_press_event', self._click_matches_click)
            #pt.set_figtitle(figtitle + vh.get_vsstr(qaid, aid))

    # Draw ctrl clicked selection
    #def sv_view(self):
    #    """ spatial verification view """
    #    #fnum = viz.FNUMS['special']
    #    aid = self.aid
    #    fnum = pt.next_fnum()
    #    fig = pt.figure(fnum=fnum, docla=True, doclf=True)
    #    ih.disconnect_callback(fig, 'button_press_event')
    #    viz.show_sv(self.ibs, self.qres.qaid, aid2=aid, fnum=fnum)
    #    ph.draw()

    # Callback
    def _click_matches_click(self, event):
        from plottool import plot_helpers as ph
        kpts1     = self.kpts1
        kpts2     = self.kpts2
        fm        = self.fm
        xywh2_ptr = self.xywh2_ptr
        #print_('[inter] clicked matches')
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
            ph.draw()
            return
        else:
            viztype = ph.get_plotdat(ax, 'viztype', '')
            key = '' if event.key is None else event.key
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
                    kpts1_m = kpts1[fm[:, 0]]
                    kpts2_m = kpts2[fm[:, 1]]
                    x2, y2, w2, h2 = xywh2_ptr[0]
                    _mx1, _dist1 = ut.nearest_point(x, y, kpts1_m)
                    _mx2, _dist2 = ut.nearest_point(x - x2, y - y2, kpts2_m)
                    mx = _mx1 if _dist1 < _dist2 else _mx2
                    print('... clicked mx=%r' % mx)
                    self.select_ith_match(mx)
            elif viztype in ['warped', 'unwarped']:
                pass
                #hs_aid = ax.__dict__.get('_hs_aid', None)
                #hs_fx = ax.__dict__.get('_hs_fx', None)
                #if hs_aid is not None and viztype == 'unwarped':
                #    ishow_chip(ibs, hs_aid, fx=hs_fx, fnum=pt.next_fnum())
                #elif hs_aid is not None and viztype == 'warped':
                #    viz.show_keypoint_gradient_orientations(ibs, hs_aid, hs_fx, fnum=pt.next_fnum())
            else:
                print('...Unknown viztype: %r' % viztype)
            ph.draw()

    #def show_each_chip(self):
    #    viz_chip.show_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
    #    viz_chip.show_chip(self.ibs, self.aid, fnum=pt.next_fnum())
    #    ph.draw()

    #def show_each_probchip(self):
    #    viz_hough.show_probability_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
    #    viz_hough.show_probability_chip(self.ibs, self.aid, fnum=pt.next_fnum())
    #    ph.draw()

    def set_callbacks(self):
        """
        CommandLine:
            python -m ibeis.viz.interact.interact_matches --test-begin --show
            python -m ibeis.viz.interact.interact_matches --test-begin

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_matches import *  # NOQA
            >>> code = ut.parse_doctest_from_docstr(MatchInteraction.begin.__doc__)[1][0]
            >>> ut.set_clipboard(code)
            >>> ut.send_keyboard_input(text='%paste')
            >>> ut.send_keyboard_input(key_list=['KP_Enter'])
        """
        from plottool import interact_helpers as ih
        #import guitool
        # TODO: view probchip
        #toggle_samefig_key = 'Toggle same_fig'
        #opt2_callback = [
        #    (toggle_samefig_key, self.toggle_samefig),
        #    ('Toggle vert', self.toggle_vert),
        #    ('query last feature', self.query_last_feature),
        #    ('show each chip', self.show_each_chip),
        #    ('show each probchip', self.show_each_probchip),
        #    #('show each probchip', self.query_last_feature),
        #    ('cancel', lambda: print('cancel')), ]
        #guitool.connect_context_menu(self.fig.canvas, opt2_callback)
        ih.connect_callback(self.fig, 'button_press_event', self._click_matches_click)

    #def toggle_vert(self):
    #    self.vert = not self.vert
    #    if self.mx is not None:
    #        self.select_ith_match(self.mx)

    #def toggle_samefig(self):
    #    self.same_fig = not self.same_fig
    #    if self.mx is not None:
    #        self.select_ith_match(self.mx)

    #def query_last_feature(self):
    #    ibs      = self.ibs
    #    qaid     = self.qaid
    #    viz.show_nearest_descriptors(ibs, qaid, self.last_fx, pt.next_fnum())
    #    fig3 = pt.gcf()
    #    ih.connect_callback(fig3, 'button_press_event', self._click_matches_click)
    #    pt.update()
