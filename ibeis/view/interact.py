from __future__ import division, print_function
# Scientific
import numpy as np
import utool
from drawtool import draw_func2 as df2
# IBEIS
from ibeis.view import viz
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[interact]', DEBUG=False)

from interact_helpers import begin_interaction

#==========================
# Image Interaction
#==========================

from interact_image import interact_image  # NOQA


#==========================
# Name Interaction
#==========================

def interact_name(ibs, nid, sel_cids=[], select_cid_func=None, fnum=5, **kwargs):
    fig = begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax is None or x is None:
            # The click is not in any axis
            print('... out of axis')
        else:
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            print_(' hs_viewtype=%r' % hs_viewtype)
            if hs_viewtype == 'chip':
                cid = ax.__dict__.get('_hs_cid')
                print('... cid=%r' % cid)
                viz.show_name(ibs, nid, fnum=fnum, sel_cids=[cid])
                select_cid_func(cid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_cids=sel_cids)
    viz.draw()
    df2.connect_callback(fig, 'button_press_event', _on_name_click)
    pass


#==========================
# Chip Interaction
#==========================


# CHIP INTERACTION 2
def interact_chip(ibs, cid, fnum=2, figtitle=None, fx=None, **kwargs):
    # TODO: Reconcile this with interact keypoints.
    # Preferably this will call that but it will set some fancy callbacks
    fig = begin_interaction('chip', fnum)
    # Get chip info (make sure get_chip is called first)
    rchip = ibs.get_chip(cid)
    annote_ptr = [False]

    def _select_ith_kpt(fx):
        # Get the fx-th keypiont
        kpts = ibs.get_kpts(cid)
        desc = ibs.get_desc(cid)
        kp, sift = kpts[fx], desc[fx]
        # Draw chip + keypoints + highlighted plots
        _chip_view(pnum=(2, 1, 1), sel_fx=fx)
        # Draw the selected feature plots
        nRows, nCols, px = (2, 3, 3)
        viz.draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _chip_view(pnum=(1, 1, 1), **kwargs):
        df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=True)
        # Toggle no keypoints view
        viz.show_chip(ibs, cid=cid, rchip=rchip, fnum=fnum, pnum=pnum, **kwargs)
        df2.set_figtitle(figtitle)

    def _on_chip_click(event):
        print_('[inter] clicked chip')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax is None or x is None:
            # The click is not in any axis
            print('... out of axis')
            annote_ptr[0] = (annote_ptr[0] + 1) % 3
            mode = annote_ptr[0]
            draw_ell = mode == 1
            draw_pts = mode == 2
            print('... default kpts view mode=%r' % mode)
            _chip_view(draw_ell=draw_ell, draw_pts=draw_pts)
        else:
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            print_('[ic] hs_viewtype=%r' % hs_viewtype)
            if hs_viewtype == 'chip' and event.key == 'shift':
                print('... masking')
                # TODO: Do better integration of masking
                _chip_view()
                df2.disconnect_callback(fig, 'button_press_event')
                #mc = mask_creator.MaskCreator(df2.gca())  # NOQA
            elif hs_viewtype == 'chip':
                kpts = ibs.get_kpts(cid)
                if len(kpts) > 0:
                    fx = utool.nearest_point(x, y, kpts)[0]
                    print('... clicked fx=%r' % fx)
                    _select_ith_kpt(fx)
                else:
                    print('... len(kpts) == 0')
            elif hs_viewtype in ['warped', 'unwarped']:
                hs_fx = ax.__dict__.get('_hs_fx', None)
                if hs_fx is not None and hs_viewtype == 'warped':
                    viz.show_keypoint_gradient_orientations(ibs, cid, hs_fx, fnum=df2.next_fnum())
            else:
                print('...Unknown viewtype: %r' % hs_viewtype)
        viz.draw()

    # Draw without keypoints the first time
    if fx is not None:
        _select_ith_kpt(fx)
    else:
        _chip_view(draw_ell=False, draw_pts=False)
    viz.draw()
    df2.connect_callback(fig, 'button_press_event', _on_chip_click)


def interact_keypoints(rchip, kpts, desc, fnum=0, figtitle=None, nodraw=False, **kwargs):
    fig = begin_interaction('keypoint', fnum)
    annote_ptr = [1]

    def _select_ith_kpt(fx):
        print_('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kp, sift = kpts[fx], desc[fx]
        # Draw the image with keypoint fx highlighted
        _viz_keypoints(fnum, (2, 1, 1), sel_fx=fx, **kwargs)  # MAYBE: remove kwargs
        # Draw the selected feature
        nRows, nCols, px = (2, 3, 3)
        viz.draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _viz_keypoints(fnum, pnum=(1, 1, 1), **kwargs):
        df2.figure(fnum=fnum, docla=True, doclf=True)
        viz.show_keypoints(rchip, kpts, fnum=fnum, pnum=pnum, **kwargs)
        if figtitle is not None:
            df2.set_figtitle(figtitle)

    def _on_keypoints_click(event):
        print_('[viz] clicked keypoint view')
        if event is None  or event.xdata is None or event.inaxes is None:
            annote_ptr[0] = (annote_ptr[0] + 1) % 3
            mode = annote_ptr[0]
            draw_ell = mode == 1
            draw_pts = mode == 2
            print('... default kpts view mode=%r' % mode)
            _viz_keypoints(fnum, draw_ell=draw_ell, draw_pts=draw_pts, **kwargs)    # MAYBE: remove kwargs
        else:
            ax = event.inaxes
            hs_viewtype = ax.__dict__.get('_hs_viewtype', None)
            print_('[ik] viewtype=%r' % hs_viewtype)
            if hs_viewtype == 'keypoints':
                kpts = ax.__dict__.get('_hs_kpts', [])
                if len(kpts) == 0:
                    print('...nokpts')
                else:
                    print('...nearest')
                    x, y = event.xdata, event.ydata
                    fx = utool.nearest_point(x, y, kpts)[0]
                    _select_ith_kpt(fx)
            elif hs_viewtype == 'warped':
                hs_fx = ax.__dict__.get('_hs_fx', None)
                if hs_fx is not None:
                    # Ugly. Interactions should be changed to classes.
                    kp = kpts[hs_fx]
                    sift = desc[hs_fx]
                    df2.draw_keypoint_gradient_orientations(rchip, kp, sift=sift, mode='vec', fnum=df2.next_fnum())
            else:
                print('...unhandled')
        viz.draw()

    # Draw without keypoints the first time
    _viz_keypoints(fnum, **kwargs)   # MAYBE: remove kwargs
    df2.connect_callback(fig, 'button_press_event', _on_keypoints_click)
    if not nodraw:
        viz.draw()

#==========================
# Chipres Interaction
#==========================


def interact_chipres(ibs, res, cid=None, fnum=4, figtitle='Inspect Query Result',
                     same_fig=True, **kwargs):
    'Plots a chip result and sets up callbacks for interaction.'
    fig = begin_interaction('chipres', fnum)
    qcid = res.qcid
    if cid is None:
        cid = res.topN_cids(ibs, 1)[0]
    rchip1, rchip2 = ibs.get_chip([qcid, cid])
    fm = res.cid2_fm[cid]
    mx = kwargs.pop('mx', None)
    xywh2_ptr = [None]
    annote_ptr = [kwargs.pop('mode', 0)]
    from hscom.Printable import DynStruct
    last_state = DynStruct()
    last_state.same_fig = same_fig
    last_state.last_fx = 0

    # Draw default
    def _chipmatch_view(pnum=(1, 1, 1), **kwargs):
        mode = annote_ptr[0]  # drawing mode draw: with/without lines/feats
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        # TODO RENAME This to remove res and rectify with show_chipres
        tup = viz.res_show_chipres(res, ibs, cid, fnum=fnum, pnum=pnum,
                                   draw_lines=draw_lines, draw_ell=draw_ell,
                                   colorbar_=True, **kwargs)
        ax, xywh1, xywh2 = tup
        xywh2_ptr[0] = xywh2

        df2.set_figtitle(figtitle + ibs.vs_str(qcid, cid))

    # Draw clicked selection
    def _select_ith_match(mx, qcid, cid):
        #----------------------
        # Get info for the _select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        cid1, cid2 = qcid, cid
        fx1, fx2 = fm[mx]
        fscore2  = res.cid2_fs[cid2][mx]
        fk2      = res.cid2_fk[cid2][mx]
        kpts1, kpts2 = ibs.get_kpts([cid1, cid2])
        desc1, desc2 = ibs.get_desc([cid1, cid2])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        last_state.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, cid1, info1),
                          (rchip2, kp2, sift2, fx2, cid2, info2)]
        # Normalizng Keypoint
        if hasattr(res, 'filt2_meta') and 'lnbnn' in res.filt2_meta:
            qfx2_norm = res.filt2_meta['lnbnn']
            # Normalizing chip and feature
            (cid3, fx3, normk) = qfx2_norm[fx1]
            rchip3 = ibs.get_chip(cid3)
            kp3 = ibs.get_kpts(cid3)[fx3]
            sift3 = ibs.get_desc(cid3)[fx3]
            info3 = '\nnorm %s k=%r' % (ibs.cidstr(cid3), normk)
            extracted_list.append((rchip3, kp3, sift3, fx3, cid3, info3))
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
        for (rchip, kp, sift, fx, cid, info) in extracted_list:
            px = viz.draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                                   prevsift=prevsift, cid=cid, info=info)
            prevsift = sift
        if not same_fig:
            df2.connect_callback(fig2, 'button_press_event', _click_chipres_click)
            df2.set_figtitle(figtitle + ibs.vs_str(qcid, cid))

    # Draw ctrl clicked selection
    def _sv_view(cid):
        fnum = viz.FNUMS['special']
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        df2.disconnect_callback(fig, 'button_press_event')
        viz.viz_spatial_verification(ibs, res.qcid, cid2=cid, fnum=fnum)
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
        hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
        print_('[ir] hs_viewtype=%r ' % hs_viewtype)
        key = '' if event.key is None else event.key
        print_('key=%r ' % key)
        ctrl_down = key.find('control') == 0
        # Click in match axes
        if hs_viewtype == 'chipres' and ctrl_down:
            # Ctrl-Click
            print('.. control click')
            return _sv_view(cid)
        elif hs_viewtype == 'chipres':
            if len(fm) == 0:
                print('[inter] no feature matches to click')
            else:
                # Normal Click
                # Select nearest feature match to the click
                kpts1, kpts2 = ibs.get_kpts([qcid, cid])
                kpts1_m = kpts1[fm[:, 0]]
                kpts2_m = kpts2[fm[:, 1]]
                x2, y2, w2, h2 = xywh2_ptr[0]
                _mx1, _dist1 = utool.nearest_point(x, y, kpts1_m)
                _mx2, _dist2 = utool.nearest_point(x - x2, y - y2, kpts2_m)
                mx = _mx1 if _dist1 < _dist2 else _mx2
                print('... clicked mx=%r' % mx)
                _select_ith_match(mx, qcid, cid)
        elif hs_viewtype in ['warped', 'unwarped']:
            hs_cid = ax.__dict__.get('_hs_cid', None)
            hs_fx = ax.__dict__.get('_hs_fx', None)
            if hs_cid is not None and hs_viewtype == 'unwarped':
                interact_chip(ibs, hs_cid, fx=hs_fx, fnum=df2.next_fnum())
            elif hs_cid is not None and hs_viewtype == 'warped':
                viz.show_keypoint_gradient_orientations(ibs, hs_cid, hs_fx, fnum=df2.next_fnum())
        else:
            print('...Unknown viewtype: %r' % hs_viewtype)
        viz.draw()

    if mx is None:
        _chipmatch_view()
    else:
        _select_ith_match(mx, qcid, cid)

    def toggle_samefig():
        interact_chipres(ibs, res, cid=cid, fnum=fnum, figtitle=figtitle, same_fig=not same_fig, **kwargs)

    def query_last_feature():
        viz.show_nearest_descriptors(ibs, qcid, last_state.last_fx, df2.next_fnum())
        fig3 = df2.gcf()
        df2.connect_callback(fig3, 'button_press_event', _click_chipres_click)
        df2.update()

    toggle_samefig_key = 'Toggle same_fig (currently %r)' % same_fig

    opt2_callback = [
        (toggle_samefig_key, toggle_samefig),
        ('query last feature', query_last_feature),
        ('cancel', lambda: print('cancel')), ]
    import guitool
    guitool.popup_menu(fig.canvas, opt2_callback, fig.canvas)
    df2.connect_callback(fig, 'button_press_event', _click_chipres_click)
    viz.draw()


#================
def select_bbox(ibs, gid, fnum=1, **kwargs):
    #from matplotlib.backend_bases import mplDeprecation
    print('[*interact] select_bbox(gid=%r, fnum=%r)' % (gid, fnum))
    print('[*interact] Define a Rectanglular ROI by clicking two points.')
    # Show the image
    fig = begin_interaction('select_bbox', fnum)
    viz.show_image(ibs, gid, **kwargs)
    try:
        viz.draw()
        fig = df2.gcf()
        pts = fig.ginput(2)
        print('[*guitools] ginput(2) = %r' % (pts,))
        [(x1, y1), (x2, y2)] = pts
        xm = min(x1, x2)
        xM = max(x1, x2)
        ym = min(y1, y2)
        yM = max(y1, y2)
        xywh = map(int, map(round, (xm, ym, xM - xm, yM - ym)))
        bbox = np.array(xywh, dtype=np.int32)
        # Reconnect the old button press events
        print('[*interact] bbox = %r ' % (bbox,))
        return bbox
    except Exception as ex:
        print('<!!!>')
        print('[*interact] Caught: %s %s' % (type(ex), ex))
        print('[*interact] ROI selection Failed:')
        print('</!!!>')
        raise
