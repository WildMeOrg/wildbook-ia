from __future__ import absolute_import, division, print_function
from ibeis import viz
import utool
import plottool as pt  # NOQA
import functools
import six
from ibeis import constants as const
from plottool import draw_func2 as df2
from plottool.viz_featrow import draw_feat_row
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[interact_chip]', DEBUG=False)


def show_annot_context_menu(ibs, aid, qwin, qpoint, refresh_func=None):
    """
    Defines logic for poping up a context menu when viewing an annotation.
    Used in other interactions like name_interaction and interact_query_decision
    """
    import guitool
    is_exemplar = ibs.get_annot_exemplar_flags(aid)

    def refresh_wrp():
        if refresh_func is None:
            print('no refresh func')
        else:
            print('calling refresh_func=%r' % (refresh_func,))
            refresh_func()

    def toggle_exemplar_func():
        new_flag = not is_exemplar
        print('set_annot_exemplar(%r, %r)' % (aid, new_flag))
        ibs.set_annot_exemplar_flags(aid, new_flag)
        refresh_wrp()
    def set_yaw_func(yawtext):
        def _wrap_yaw():
            ibs.set_annot_yaw_texts([aid], [yawtext])
            print('set_annot_yaw(%r, %r)' % (aid, yawtext))
            refresh_wrp()
        return _wrap_yaw
    def set_quality_func(qualtext):
        def _wrp_qual():
            ibs.set_annot_quality_texts([aid], [qualtext])
            print('set_annot_quality(%r, %r=%r)' % (aid, qualtext))
            refresh_wrp()
        return _wrp_qual
    # Define popup menu
    angle_callback_list = [
        ('unset as exemplar' if is_exemplar else 'set as exemplar', toggle_exemplar_func),
    ]
    current_qualtext = ibs.get_annot_quality_texts([aid])[0]
    current_yawtext = ibs.get_annot_yaw_texts([aid])[0]
    # Nested viewpoints
    angle_callback_list += [
        #('Set Viewpoint: ' + key, set_yaw_func(key))
        ('Set Viewpoint: ',  [
            (('*' if current_yawtext == key else '') + key, set_yaw_func(key))
            for key in six.iterkeys(const.VIEWTEXT_TO_YAW_RADIANS)
        ]),
    ]
    # Nested qualities
    angle_callback_list += [
        #('Set Quality: ' + key, set_quality_func(key))
        ('Set Quality: ',  [
            (('*' if current_qualtext == key else '') + key, set_quality_func(key))
            for key in six.iterkeys(const.QUALITY_TEXT_TO_INT)
        ]),
    ]
    guitool.popup_menu(qwin, qpoint, angle_callback_list)


# CHIP INTERACTION 2
def ishow_chip(ibs, aid, fnum=2, fx=None, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid (int):  annotation id
        fnum (int):  figure number
        fx (None):

    CommandLine:
        python -m ibeis.viz.interact.interact_chip --test-ishow_chip --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_chip import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid = 1
        >>> fnum = 2
        >>> fx = None
        >>> # execute function
        >>> result = ishow_chip(ibs, aid, fnum, fx)
        >>> # verify results
        >>> pt.show_if_requested()
        >>> print(result)
    """
    vh.ibsfuncs.assert_valid_aids(ibs, (aid,))
    # TODO: Reconcile this with interact keypoints.
    # Preferably this will call that but it will set some fancy callbacks
    fig = ih.begin_interaction('chip', fnum)
    # Get chip info (make sure get_chips is called first)
    mode_ptr = [1]

    def _select_fxth_kpt(fx):
        # Get the fx-th keypiont
        chip = ibs.get_annot_chips(aid)
        kp = ibs.get_annot_kpts(aid)[fx]
        sift = ibs.get_annot_vecs(aid)[fx]
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
            if event.button == 3:   # right-click
                import guitool
                height = fig.canvas.geometry().height()
                qpoint = guitool.newQPoint(event.x, height - event.y)
                from ibeis.viz.interact import interact_chip
                refresh_func = functools.partial(_chip_view, **kwargs)
                interact_chip.show_annot_context_menu(
                    ibs, aid, fig.canvas, qpoint, refresh_func=refresh_func)
            else:
                viztype = vh.get_ibsdat(ax, 'viztype')
                print_('[ic] viztype=%r' % viztype)
                if viztype == 'chip' and event.key == 'shift':
                    _chip_view(**kwargs)
                    ih.disconnect_callback(fig, 'button_press_event')
                elif viztype == 'chip':
                    kpts = ibs.get_annot_kpts(aid)
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


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_chip
        python -m ibeis.viz.interact.interact_chip --allexamples
        python -m ibeis.viz.interact.interact_chip --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
