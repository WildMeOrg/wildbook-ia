from __future__ import division, print_function
from ibeis.view import viz
import utool
from viz_helpers import get_ibsdat
from drawtool import draw_func2 as df2
import interact_helpers
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_chip]',
                                                       DEBUG=False)


# CHIP INTERACTION 2
def interact_chip(ibs, rid, fnum=2, figtitle='Chip View', fx=None, **kwargs):
    # TODO: Reconcile this with interact keypoints.
    # Preferably this will call that but it will set some fancy callbacks
    fig = interact_helpers.begin_interaction('chip', fnum)
    # Get chip info (make sure get_chips is called first)
    cid = ibs.get_roi_cids(rid)
    chip = ibs.get_roi_chips(rid)
    mode_ptr = [0]

    def _select_fxth_kpt(fx):
        # Get the fx-th keypiont
        kp = ibs.get_roi_kpts(rid)[fx]
        sift = ibs.get_roi_desc(rid)[fx]
        # Draw chip + keypoints + highlighted plots
        _chip_view(pnum=(2, 1, 1), sel_fx=fx)
        # Draw the selected feature plots
        nRows, nCols, px = (2, 3, 3)
        viz.draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _chip_view(pnum=(1, 1, 1), **kwargs):
        df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=True)
        # Toggle no keypoints view
        viz.show_chip(ibs, cid, fnum=fnum, pnum=pnum, **kwargs)
        df2.set_figtitle(figtitle)

    def _on_chip_click(event):
        print_('[inter] clicked chip')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if interact_helpers.clicked_outside_axis(event):
            print('... out of axis')
            mode_ptr[0] = (mode_ptr[0] + 1) % 3
            mode = mode_ptr[0]
            print('... default kpts view mode=%r' % mode)
            _chip_view(ell=(mode == 1), pts=(mode == 2))
        else:
            viztype = get_ibsdat(ax, 'viztype')
            print_('[ic] viztype=%r' % viztype)
            if viztype == 'chip' and event.key == 'shift':
                _chip_view()
                interact_helpers.disconnect_callback(fig, 'button_press_event')
            elif viztype == 'chip':
                kpts = ibs.get_roi_kpts(rid)
                if len(kpts) > 0:
                    fx = utool.nearest_point(x, y, kpts)[0]
                    print('... clicked fx=%r' % fx)
                    _select_fxth_kpt(fx)
                else:
                    print('... len(kpts) == 0')
            elif viztype in ['warped', 'unwarped']:
                fx = get_ibsdat(ax, 'fx')
                if fx is not None and viztype == 'warped':
                    viz.show_keypoint_gradient_orientations(ibs, rid, fx, fnum=df2.next_fnum())
            else:
                print('...Unknown viewtype: %r' % viztype)
        viz.draw()

    # Draw without keypoints the first time
    if fx is not None:
        _select_fxth_kpt(fx)
    else:
        _chip_view(ell=False, pts=False)
    viz.draw()
    interact_helpers.connect_callback(fig, 'button_press_event', _on_chip_click)
