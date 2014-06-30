from __future__ import absolute_import, division, print_function
from ibeis import viz
import utool
from plottool import draw_func2 as df2
from plottool.viz_featrow import draw_feat_row
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[interact_chip]', DEBUG=False)


# CHIP INTERACTION 2
def ishow_chip(ibs, aid, fnum=2, fx=None, **kwargs):
    vh.ibsfuncs.assert_valid_aids(ibs, (aid,))
    # TODO: Reconcile this with interact keypoints.
    # Preferably this will call that but it will set some fancy callbacks
    fig = ih.begin_interaction('chip', fnum)
    # Get chip info (make sure get_chips is called first)
    mode_ptr = [1]

    def _select_fxth_kpt(fx):
        # Get the fx-th keypiont
        chip = ibs.get_annotation_chips(aid)
        kp = ibs.get_annotation_kpts(aid)[fx]
        sift = ibs.get_annotation_desc(aid)[fx]
        # Draw chip + keypoints + highlighted plots
        _chip_view(pnum=(2, 1, 1), sel_fx=fx)
        # Draw the selected feature plots
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _chip_view(mode=0, pnum=(1, 1, 1), **kwargs):
        print('... _chip_view mode=%r' % mode_ptr[0])
        kwargs['ell'] = mode_ptr[0] == 1
        kwargs['pts'] = mode_ptr[0]  == 2
        df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=True)
        # Toggle no keypoints view
        viz.show_chip(ibs, aid, fnum=fnum, pnum=pnum, **kwargs)
        df2.set_figtitle('Chip View')

    def _on_chip_click(event):
        print_('[inter] clicked chip')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ih.clicked_outside_axis(event):
            print('... out of axis')
            mode_ptr[0] = (mode_ptr[0] + 1) % 3
            _chip_view(**kwargs)
        else:
            viztype = vh.get_ibsdat(ax, 'viztype')
            print_('[ic] viztype=%r' % viztype)
            if viztype == 'chip' and event.key == 'shift':
                _chip_view(**kwargs)
                ih.disconnect_callback(fig, 'button_press_event')
            elif viztype == 'chip':
                kpts = ibs.get_annotation_kpts(aid)
                if len(kpts) > 0:
                    fx = utool.nearest_point(x, y, kpts)[0]
                    print('... clicked fx=%r' % fx)
                    _select_fxth_kpt(fx)
                else:
                    print('... len(kpts) == 0')
            elif viztype in ['warped', 'unwarped']:
                fx = vh.get_ibsdat(ax, 'fx')
                if fx is not None and viztype == 'warped':
                    viz.show_keypoint_gradient_orientations(ibs, aid, fx, fnum=df2.next_fnum())
            else:
                print('...Unknown viztype: %r' % viztype)
        viz.draw()

    # Draw without keypoints the first time
    if fx is not None:
        _select_fxth_kpt(fx)
    else:
        _chip_view(**kwargs)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_chip_click)
