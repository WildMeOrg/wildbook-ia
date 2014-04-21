from __future__ import absolute_import, division, print_function
import utool
import guitool
import numpy as np
from plottool import draw_func2 as df2
from ibeis.view import viz
from ibeis.view.viz import viz_helpers as vh
from . import interact_helpers as ih
from .interact_chip import interact_chip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[interact-chipres]', DEBUG=False)


def interact_chipres(ibs, qres, rid=None, fnum=4, figtitle='Inspect Query Result',
                     same_fig=True, **kwargs):
    'Plots a chip result and sets up callbacks for interaction.'
    fig = ih.begin_interaction('chipres', fnum)
    qrid = qres.qrid
    if rid is None:
        rid = qres.get_top_rids(num=1)[0]
    rchip1, rchip2 = ibs.get_roi_chips([qrid, rid])
    fm = qres.rid2_fm[rid]
    mx = kwargs.pop('mx', None)
    xywh2_ptr = [None]
    annote_ptr = [kwargs.pop('mode', 0)]
    last_state = utool.DynStruct()
    last_state.same_fig = same_fig
    last_state.last_fx = 0

    # Draw default
    def _chipmatch_view(pnum=(1, 1, 1), **kwargs):
        mode = annote_ptr[0]  # drawing mode draw: with/without lines/feats
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        # TODO RENAME This to remove qres and rectify with show_chipres
        tup = viz.show_chipres(ibs, qres, rid, fnum=fnum, pnum=pnum,
                               draw_lines=draw_lines, draw_ell=draw_ell,
                               colorbar_=True, **kwargs)
        ax, xywh1, xywh2 = tup
        xywh2_ptr[0] = xywh2

        df2.set_figtitle(figtitle + vh.get_vsstr(qrid, rid))

    # Draw clicked selection
    def _select_ith_match(mx, qrid, rid):
        #----------------------
        # Get info for the _select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        rid1, rid2 = qrid, rid
        fx1, fx2 = fm[mx]
        fscore2  = qres.rid2_fs[rid2][mx]
        fk2      = qres.rid2_fk[rid2][mx]
        kpts1, kpts2 = ibs.get_roi_kpts([rid1, rid2])
        desc1, desc2 = ibs.get_roi_desc([rid1, rid2])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        last_state.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, rid1, info1),
                          (rchip2, kp2, sift2, fx2, rid2, info2)]
        # Normalizng Keypoint
        if hasattr(qres, 'filt2_meta') and 'lnbnn' in qres.filt2_meta:
            qfx2_norm = qres.filt2_meta['lnbnn']
            # Normalizing chip and feature
            (rid3, fx3, normk) = qfx2_norm[fx1]
            rchip3 = ibs.get_roi_chips(rid3)
            kp3 = ibs.get_roi_kpts(rid3)[fx3]
            sift3 = ibs.get_roi_desc(rid3)[fx3]
            info3 = '\nnorm %s k=%r' % (vh.get_ridstrs(rid3), normk)
            extracted_list.append((rchip3, kp3, sift3, fx3, rid3, info3))
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
        for (rchip, kp, sift, fx, rid, info) in extracted_list:
            px = viz.draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                                   prevsift=prevsift, rid=rid, info=info)
            prevsift = sift
        if not same_fig:
            ih.connect_callback(fig2, 'button_press_event', _click_chipres_click)
            df2.set_figtitle(figtitle + vh.get_vsstr(qrid, rid))

    # Draw ctrl clicked selection
    def _sv_view(rid):
        fnum = viz.FNUMS['special']
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        ih.disconnect_callback(fig, 'button_press_event')
        viz.show_sv(ibs, qres.qrid, rid2=rid, fnum=fnum)
        viz.draw()

    # Callback
    def _click_chipres_click(event):
        print_('[inter] clicked chipres')
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
        if viztype == 'chipres' and ctrl_down:
            # Ctrl-Click
            print('.. control click')
            return _sv_view(rid)
        elif viztype == 'chipres':
            if len(fm) == 0:
                print('[inter] no feature matches to click')
            else:
                # Normal Click
                # Select nearest feature match to the click
                kpts1, kpts2 = ibs.get_roi_kpts([qrid, rid])
                kpts1_m = kpts1[fm[:, 0]]
                kpts2_m = kpts2[fm[:, 1]]
                x2, y2, w2, h2 = xywh2_ptr[0]
                _mx1, _dist1 = utool.nearest_point(x, y, kpts1_m)
                _mx2, _dist2 = utool.nearest_point(x - x2, y - y2, kpts2_m)
                mx = _mx1 if _dist1 < _dist2 else _mx2
                print('... clicked mx=%r' % mx)
                _select_ith_match(mx, qrid, rid)
        elif viztype in ['warped', 'unwarped']:
            hs_rid = ax.__dict__.get('_hs_rid', None)
            hs_fx = ax.__dict__.get('_hs_fx', None)
            if hs_rid is not None and viztype == 'unwarped':
                interact_chip(ibs, hs_rid, fx=hs_fx, fnum=df2.next_fnum())
            elif hs_rid is not None and viztype == 'warped':
                viz.show_keypoint_gradient_orientations(ibs, hs_rid, hs_fx, fnum=df2.next_fnum())
        else:
            print('...Unknown viztype: %r' % viztype)
        viz.draw()

    if mx is None:
        _chipmatch_view()
    else:
        _select_ith_match(mx, qrid, rid)

    def toggle_samefig():
        interact_chipres(ibs, qres, rid=rid, fnum=fnum, figtitle=figtitle, same_fig=not same_fig, **kwargs)

    def query_last_feature():
        viz.show_nearest_descriptors(ibs, qrid, last_state.last_fx, df2.next_fnum())
        fig3 = df2.gcf()
        ih.connect_callback(fig3, 'button_press_event', _click_chipres_click)
        df2.update()

    toggle_samefig_key = 'Toggle same_fig (currently %r)' % same_fig

    opt2_callback = [
        (toggle_samefig_key, toggle_samefig),
        ('query last feature', query_last_feature),
        ('cancel', lambda: print('cancel')), ]
    guitool.popup_menu(fig.canvas, opt2_callback, fig.canvas)
    ih.connect_callback(fig, 'button_press_event', _click_chipres_click)
    viz.draw()
