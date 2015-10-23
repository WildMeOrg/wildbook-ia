# -*- coding: utf-8 -*-
"""
Unfinished non-ibeis dependent version of interact matches
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import numpy as np

from plottool import abstract_interaction

BASE_CLASS = abstract_interaction.AbstractInteraction
#BASE_CLASS = object


# TODO: move to plottool and decouple with IBEIS
# TODO: abstract interaction
@six.add_metaclass(ut.ReloadingMetaclass)
class MatchInteraction2(BASE_CLASS):
    """
    TODO: replace functional version with this class

    Plots a chip result and sets up callbacks for interaction.

    SeeAlso:
        ibeis.viz.interact.interact_matches.MatchInteraction

    CommandLine:
        python -m plottool.interact_matches --test-MatchInteraction2 --show


    Example:
        >>> from plottool.interact_matches import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qreq_ = ibs.new_query_request([1], [2, 3, 4, 5], cfgdict=dict())
        >>> cm = ibs.query_chips(qreq_=qreq_, return_cm=True)[0]
        >>> qaid = cm.qaid
        >>> daid = cm.get_top_aids()[0]
        >>> rchip1 = ibs.get_annot_chips([qaid], config2_=qreq_.extern_query_config2)[0]
        >>> rchip2 = ibs.get_annot_chips([daid], config2_=qreq_.extern_data_config2)[0]
        >>> kpts1 = ibs.get_annot_kpts([qaid], config2_=qreq_.extern_query_config2)[0]
        >>> kpts2 = ibs.get_annot_kpts([daid], config2_=qreq_.extern_data_config2)[0]
        >>> vecs1 = ibs.get_annot_vecs([qaid], config2_=qreq_.extern_query_config2)[0]
        >>> vecs2 = ibs.get_annot_vecs([daid], config2_=qreq_.extern_data_config2)[0]
        >>> fm = cm.aid2_fm[daid]
        >>> fs = cm.aid2_fs[daid]
        >>> fsv = cm.aid2_fsv[daid]
        >>> H1 = cm.aid2_H[daid]
        >>> self = MatchInteraction2(rchip1, rchip2, kpts1, kpts2, fm, fs, fsv,
        >>>                          vecs1, vecs2, H1)
        >>> self.show_page()
        >>> ut.show_if_requested()

    """
    def __init__(self, rchip1, rchip2, kpts1, kpts2, fm, fs, fsv, vecs1, vecs2,
                 H1=None, H2=None,
                 **kwargs):
        # Drawing Data
        self.rchip1 = rchip1
        self.rchip2 = rchip2
        self.kpts1 = kpts1
        self.kpts2 = kpts2
        self.fm = fm
        self.fs = fs
        self.fsv = fsv
        self.vecs1 = vecs1
        self.vecs2 = vecs2
        self.H1 = H1
        self.H2 = H2

        # Drawing settings
        kwargs = kwargs.copy()
        self.warp_homog = False
        self.mode = kwargs.pop('mode', 0)
        self.mx = kwargs.pop('mx', None)
        self.vert = kwargs.pop('vert', None)
        self.same_fig = kwargs.get('same_fig', True)
        self.last_fx = 0
        self.figtitle = kwargs.get('figtitle', 'Inspect Matches')
        self.xywh2 = None
        import plottool as pt
        self.fnum2 = pt.next_fnum()

        if BASE_CLASS is not object:
            kwargs['interaction_name'] = 'matches'
            super(MatchInteraction2, self).__init__(**kwargs)

        #self.begin(**kwargs)

    def plot(self, *args, **kwargs):
        self.chipmatch_view(*args, **kwargs)

    def chipmatch_view(self, fnum=None, pnum=(1, 1, 1), **kwargs_):
        """
        just visualizes the matches using some type of lines
        """
        import plottool as pt
        from plottool import plot_helpers as ph
        if fnum is None:
            fnum     = self.fnum
        print('-- CHIPMATCH VIEW --')
        print('[ichipmatch_view] self.mode = %r' % (self.mode,))
        draw_ell = self.mode >= 1
        draw_lines = self.mode == 2
        print('[ichipmatch_view] draw_lines = %r' % (draw_lines,))
        print('[ichipmatch_view] draw_ell = %r' % (draw_ell,))
        pt.figure(fnum=fnum, docla=True, doclf=True)
        #show_matches_kw = self.__dict__.copy()
        show_matches_kw = dict(
            #fnum=fnum, pnum=pnum,
            draw_lines=draw_lines,
            draw_ell=draw_ell,
            colorbar_=True,
            vert=self.vert)
        show_matches_kw.update(kwargs_)

        print('self.warp_homog = %r' % (self.warp_homog,))
        if self.warp_homog:
            show_matches_kw['H1'] = self.H1
            show_matches_kw['H2'] = self.H2
        print('show_matches_kw = %s' % (ut.dict_str(show_matches_kw, truncate=True)))

        #tup = show_matches(fm, fs, **show_matches_kw)
        ax, xywh1, xywh2 = pt.show_chipmatch2(
            self.rchip1, self.rchip2,
            self.kpts1, self.kpts2,
            fm=self.fm, fs=self.fs,
            pnum=pnum, **show_matches_kw)
        self.xywh2 = xywh2
        ph.set_plotdat(ax, 'viztype', 'matches')
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
        self.mode = 1
        # Get the mx-th feature match
        fx1, fx2 = self.fm[mx]

        # Older info
        fscore2  = self.fs[mx]
        fk2      = None  # qres.aid2_fk[aid2][mx]
        kp1, kp2     = self.kpts1[fx1], self.kpts2[fx2]
        vecs1, vecs2 = self.vecs1[fx1], self.vecs2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        #self.last_fx = fx1
        self.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, vecs1, fx1, 'aid1', info1),
                          (rchip2, kp2, vecs2, fx2, 'aid2', info2)]
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
        self.chipmatch_view(pnum=pnum1, ell_alpha=.4, ell_linewidth=1.8,
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
            px = viz_featrow.draw_feat_row(
                rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                prevsift=prevsift, aid=aid, info=info)
            prevsift = sift
        if not same_fig:
            ih.connect_callback(fig2, 'button_press_event', self.on_click)
            #pt.set_figtitle(figtitle + vh.get_vsstr(qaid, aid))

    # Callback
    def on_click_inside(self, event, ax):
        from plottool import plot_helpers as ph
        (x, y) = (event.xdata, event.ydata)
        viztype = ph.get_plotdat(ax, 'viztype', '')

        if event.button == 3:
            self.show_popup_menu(self.get_popup_options(), event)
            return
        #key = '' if event.key is None else event.key
        #ctrl_down = key.find('control') == 0
        if viztype == 'matches':
            if len(self.fm) == 0:
                print('[inter] no feature matches to click')
            else:
                # Normal Click
                # Select nearest feature match to the click
                kpts1_m = self.kpts1[self.fm[:, 0]]
                kpts2_m = self.kpts2[self.fm[:, 1]]
                x2, y2, w2, h2 = self.xywh2
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
            #    viz.show_keypoint_gradient_orientations(ibs, hs_aid,
            #    hs_fx, fnum=pt.next_fnum())
        # Click in match axes
        #elif viztype == 'matches' and ctrl_down:
        #    # Ctrl-Click
        #    print('.. control click')
        #    return self.sv_view()
        else:
            print('...Unknown viztype: %r' % viztype)
        self.draw()

    def on_click_outside(self, event):
        if event.button != 1:
            return
        print('... out of axis')
        #self.warp_homog = not self.warp_homog
        self.mode = (self.mode + 1) % 3
        #self.chipmatch_view()
        self.show_page()
        self.draw()

    def get_popup_options(self):

        def toggle_attr_item(attr, num_states=2):
            value = getattr(self, attr)
            type_ = type(value)
            def toggle_attr():
                new_value = (value + 1) % (num_states)
                new_value = type_(new_value)
                print('new_value(%s) = %r' % (attr, new_value,))
                setattr(self, attr, new_value)
                self.show_page()
                self.draw()
            itemstr = 'Toggle %s=%r' % (attr, value)
            return (itemstr, toggle_attr)

        options = [
            toggle_attr_item('warp_homog'),
            toggle_attr_item('mode', 3),
        ]
        return options

    #def on_click(self, event):
    #    from plottool import plot_helpers as ph
    #    kpts1     = self.kpts1
    #    kpts2     = self.kpts2
    #    fm        = self.fm
    #    #print_('[inter] clicked matches')
    #    if event is None:
    #        return
    #    button = event.button
    #    is_right_click = button == 3
    #    if is_right_click:
    #        return
    #    (x, y, ax) = (event.xdata, event.ydata, event.inaxes)
    #    # Out of axes click
    #    if None in [x, y, ax]:
    #        return
    #    else:

    #def set_callbacks(self):
    #    """
    #    CommandLine:
    #        python -m ibeis.viz.interact.interact_matches --test-begin --show
    #        python -m ibeis.viz.interact.interact_matches --test-begin

    #    Example:
    #        >>> # DISABLE_DOCTEST
    #        >>> from ibeis.viz.interact.interact_matches import *  # NOQA
    #        >>> code = ut.parse_doctest_from_docstr(MatchInteraction.begin.__doc__)[1][0]
    #        >>> ut.set_clipboard(code)
    #        >>> ut.send_keyboard_input(text='%paste')
    #        >>> ut.send_keyboard_input(key_list=['KP_Enter'])
    #    """
    #    from plottool import interact_helpers as ih
    #    #import guitool
    #    # TODO: view probchip
    #    #toggle_samefig_key = 'Toggle same_fig'
    #    #opt2_callback = [
    #    #    (toggle_samefig_key, self.toggle_samefig),
    #    #    ('Toggle vert', self.toggle_vert),
    #    #    ('query last feature', self.query_last_feature),
    #    #    ('show each chip', self.show_each_chip),
    #    #    ('show each probchip', self.show_each_probchip),
    #    #    #('show each probchip', self.query_last_feature),
    #    #    ('cancel', lambda: print('cancel')), ]
    #    #guitool.connect_context_menu(self.fig.canvas, opt2_callback)
    #    ih.connect_callback(self.fig, 'button_press_event', self.on_click)

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
    #    ih.connect_callback(fig3, 'button_press_event', self.on_click)
    #    pt.update()

    #def show_each_chip(self):
    #    viz_chip.show_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
    #    viz_chip.show_chip(self.ibs, self.aid, fnum=pt.next_fnum())
    #    ph.draw()

    #def show_each_probchip(self):
    #    viz_hough.show_probability_chip(self.ibs, self.qaid, fnum=pt.next_fnum())
    #    viz_hough.show_probability_chip(self.ibs, self.aid, fnum=pt.next_fnum())
    #    ph.draw()

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

    #def show_page(self, *args):
    #    from plottool import interact_helpers as ih
    #    fig = ih.begin_interaction('matches', self.fnum)
    #    #self.set_callbacks()

    #def begin(self, fnum=None, figtitle='Inspect Matches', same_fig=True,
    #          **kwargs):
    #    import plottool as pt
    #    from plottool import interact_helpers as ih
    #    from plottool import plot_helpers as ph
    #    if fnum is None:
    #        fnum = pt.next_fnum()
    #    # call doclf docla and make figure
    #    fig = ih.begin_interaction('matches', fnum)

    #    # New state vars
    #    self.mx = None

    #    # SET CLOSURE VARS
    #    #self.fnum     = fnum
    #    self.kwargs   = kwargs
    #    self.fig = fig
    #    self.xywh2 = None
    #    #self.rchip1     = rchip1
    #    #self.rchip2     = rchip2

    #    if mx is None:
    #        self.chipmatch_view()
    #    else:
    #        self.select_ith_match(mx)

    #    self.set_callbacks()
    #    # FIXME: this should probably not be called here
    #    ph.draw()  # ph-> adjust stuff draw -> fig_presenter.draw -> all figures show


if __name__ == '__main__':
    """
    CommandLine:
        python -m plottool.interact_matches
        python -m plottool.interact_matches --allexamples
        python -m plottool.interact_matches --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
