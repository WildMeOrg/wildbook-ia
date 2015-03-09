"""
Interaction for a single annoation.
Also defines annotation context menu.

CommandLine:
    python -m ibeis.viz.interact.interact_chip --test-ishow_chip --show --aid 2
"""
from __future__ import absolute_import, division, print_function
from ibeis import viz
import utool as ut
import vtool as vt
import plottool as pt  # NOQA
import functools
import six
from ibeis import constants as const
from plottool import draw_func2 as df2
from plottool.viz_featrow import draw_feat_row
from ibeis.viz import viz_helpers as vh
from plottool import interact_helpers as ih

(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[interact_chip]', DEBUG=False)


def show_annot_context_menu(ibs, aid, qwin, qpoint, refresh_func=None,
                            with_interact_name=True, with_interact_chip=True,
                            with_interact_image=True):
    """
    Defines logic for poping up a context menu when viewing an annotation.
    Used in other interactions like name_interaction and interact_query_decision

    CommandLine:
        python -m ibeis.viz.interact.interact_chip --test-ishow_chip --show
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
            #refresh_wrp()
        return _wrap_yaw
    def set_quality_func(qualtext):
        def _wrp_qual():
            ibs.set_annot_quality_texts([aid], [qualtext])
            print('set_annot_quality(%r, %r)' % (aid, qualtext))
            #refresh_wrp()
        return _wrp_qual
    # Define popup menu
    callback_list = [
        ('Unset as e&xemplar' if is_exemplar else 'Set as e&xemplar', toggle_exemplar_func),
    ]
    current_qualtext = ibs.get_annot_quality_texts([aid])[0]
    current_yawtext = ibs.get_annot_yaw_texts([aid])[0]
    # Nested viewpoints
    callback_list += [
        #('Set Viewpoint: ' + key, set_yaw_func(key))
        ('Set &Viewpoint: ',  [
            ('&' + str(count) + ' ' + ('*' if current_yawtext == key else '') + key, set_yaw_func(key))
            for count, key in enumerate(six.iterkeys(const.VIEWTEXT_TO_YAW_RADIANS), start=1)
        ]),
    ]
    # Nested qualities
    callback_list += [
        #('Set Quality: ' + key, set_quality_func(key))
        ('Set &Quality: ',  [
            ('&' + str(count) + ' ' + ('*' if current_qualtext == key else '') + '&' + key, set_quality_func(key))
            for count, key in enumerate(six.iterkeys(const.QUALITY_TEXT_TO_INT), start=1)
        ]),
    ]
    nid = ibs.get_annot_name_rowids(aid)

    if with_interact_chip:
        callback_list += [
            ('Interact chip', functools.partial(ishow_chip, ibs, aid, fnum=None))
        ]
    if with_interact_name and not ibs.is_nid_unknown(nid):
        from ibeis.viz.interact import interact_name
        callback_list.append(
            ('Interact name', functools.partial(interact_name.ishow_name, ibs, nid, fnum=None))
        )
    if with_interact_image:
        gid = ibs.get_annot_gids(aid)
        from ibeis.viz.interact import interact_annotations2
        callback_list.append(
            ('Interact image', functools.partial(interact_annotations2.ishow_image2, ibs, gid, fnum=None))
        )
    guitool.popup_menu(qwin, qpoint, callback_list)


# CHIP INTERACTION 2
def ishow_chip(ibs, aid, fnum=2, fx=None, dodraw=True, qreq_=None, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid (int):  annotation id
        fnum (int):  figure number
        fx (None):

    CommandLine:
        python -m ibeis.viz.interact.interact_chip --test-ishow_chip --show
        python -m ibeis.viz.interact.interact_chip --test-ishow_chip --show --aid 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_chip import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid = ut.get_argval('--aid', type_=int, default=1)
        >>> fnum = 2
        >>> fx = None
        >>> # execute function
        >>> dodraw = ut.show_was_requested()
        >>> result = ishow_chip(ibs, aid, fnum, fx, dodraw)
        >>> # verify results
        >>> pt.show_if_requested()
        >>> print(result)
    """
    if fnum is None:
        fnum = pt.next_fnum()
    vh.ibsfuncs.assert_valid_aids(ibs, (aid,))
    # TODO: Reconcile this with interact keypoints.
    # Preferably this will call that but it will set some fancy callbacks
    fig = ih.begin_interaction('chip', fnum)
    # Get chip info (make sure get_chips is called first)
    mode_ptr = [1]

    def _select_fxth_kpt(fx):
        # Get the fx-th keypiont
        chip = ibs.get_annot_chips(aid, qreq_=qreq_)
        kp = ibs.get_annot_kpts(aid, qreq_=qreq_)[fx]
        sift = ibs.get_annot_vecs(aid, qreq_=qreq_)[fx]
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
        viz.show_chip(ibs, aid, fnum=fnum, pnum=pnum, qreq_=qreq_, **kwargs)
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
                from ibeis.viz.interact import interact_chip
                height = fig.canvas.geometry().height()
                qpoint = guitool.newQPoint(event.x, height - event.y)
                refresh_func = functools.partial(_chip_view, **kwargs)
                interact_chip.show_annot_context_menu(
                    ibs, aid, fig.canvas, qpoint, refresh_func=refresh_func,
                    with_interact_chip=False)
            else:
                viztype = vh.get_ibsdat(ax, 'viztype')
                print_('[ic] viztype=%r' % viztype)
                if viztype == 'chip' and event.key == 'shift':
                    _chip_view(**kwargs)
                    ih.disconnect_callback(fig, 'button_press_event')
                elif viztype == 'chip':
                    kpts = ibs.get_annot_kpts(aid, qreq_=qreq_)
                    if len(kpts) > 0:
                        fx = vt.nearest_point(x, y, kpts, conflict_mode='next')[0]
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
    if dodraw:
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
