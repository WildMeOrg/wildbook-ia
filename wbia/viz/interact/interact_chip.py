# -*- coding: utf-8 -*-
"""
Interaction for a single annoation.
Also defines annotation context menu.

CommandLine:
    python -m wbia.viz.interact.interact_chip --test-ishow_chip --show --aid 2
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import wbia.plottool as pt  # NOQA
from functools import partial
from wbia import viz
from wbia.viz import viz_helpers as vh
from wbia.plottool import interact_helpers as ih

(print, rrr, profile) = ut.inject2(__name__)


def interact_multichips(ibs, aid_list, config2_=None, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids

    Returns:
        MultiImageInteraction: iteract_obj

    CommandLine:
        python -m wbia.viz.interact.interact_chip --exec-interact_multichips --show

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.viz.interact.interact_chip import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> iteract_obj = interact_multichips(ibs, aid_list)
        >>> iteract_obj.start()
        >>> result = ('iteract_obj = %s' % (str(iteract_obj),))
        >>> print(result)
        >>> ut.show_if_requested()
    """
    # FIXME: needs to be flushed out a little
    import wbia.plottool as pt

    show_chip_list = [
        partial(viz.show_chip, ibs, aid, config2_=config2_) for aid in aid_list
    ]
    vizkw = dict(ell=0, pts=1)
    context_option_funcs = [
        partial(build_annot_context_options, ibs, aid, config2_=config2_)
        for aid in aid_list
    ]
    iteract_obj = pt.interact_multi_image.MultiImageInteraction(
        show_chip_list, context_option_funcs=context_option_funcs, vizkw=vizkw, **kwargs
    )
    return iteract_obj


def show_annot_context_menu(
    ibs,
    aid,
    qwin,
    qpoint,
    refresh_func=None,
    with_interact_name=True,
    with_interact_chip=True,
    with_interact_image=True,
    config2_=None,
):
    """
    Defines logic for poping up a context menu when viewing an annotation.
    Used in other interactions like name_interaction and interact_query_decision

    CommandLine:
        python -m wbia.viz.interact.interact_chip --test-ishow_chip --show

    """
    import wbia.guitool as gt

    callback_list = build_annot_context_options(
        ibs,
        aid,
        refresh_func=refresh_func,
        with_interact_name=with_interact_name,
        with_interact_chip=with_interact_chip,
        with_interact_image=with_interact_image,
        config2_=config2_,
    )
    gt.popup_menu(qwin, qpoint, callback_list)


def build_annot_context_options(
    ibs,
    aid,
    refresh_func=None,
    with_interact_name=True,
    with_interact_chip=True,
    with_interact_image=True,
    config2_=None,
):
    r"""
    Build context options for things that select annotations in the IBEIS gui

    Args:
        ibs (IBEISController):  wbia controller object
        aid (int):  annotation id
        refresh_func (None): (default = None)
        with_interact_name (bool): (default = True)
        with_interact_chip (bool): (default = True)
        with_interact_image (bool): (default = True)
        config2_ (dict): (default = None)

    Returns:
        list: callback_list

    CommandLine:
        python -m wbia.viz.interact.interact_chip --exec-build_annot_context_options

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.interact.interact_chip import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid = ibs.get_valid_aids()[0]
        >>> refresh_func = None
        >>> with_interact_name = True
        >>> with_interact_chip = True
        >>> with_interact_image = True
        >>> config2_ = None
        >>> callback_list = build_annot_context_options(ibs, aid, refresh_func,
        >>>                                             with_interact_name,
        >>>                                             with_interact_chip,
        >>>                                             with_interact_image,
        >>>                                             config2_)
        >>> result = ('callback_list = %s' % (ut.repr2(callback_list, nl=4),))
        >>> print(result)
    """
    import wbia.guitool as gt

    is_exemplar = ibs.get_annot_exemplar_flags(aid)

    def refresh_wrp(func):
        def _wrp():
            ret = func()
            if refresh_func is None:
                print('no refresh func')
            else:
                print('calling refresh_func=%r' % (refresh_func,))
                refresh_func()
            return ret

        return _wrp

    def newplot_wrp(func):
        def _wrp():
            import wbia.plottool as pt

            ret = func()
            pt.draw()
            return ret

        return _wrp

    @refresh_wrp
    def toggle_exemplar_func():
        new_flag = not is_exemplar
        print('set_annot_exemplar(%r, %r)' % (aid, new_flag))
        ibs.set_annot_exemplar_flags(aid, new_flag)

    def set_viewpoint_func(view_code):
        # @refresh_wrp()
        def _wrap_view():
            ibs.set_annot_viewpoint_codes([aid], [view_code])
            print('set_annot_yaw(%r, %r)' % (aid, view_code))

        return _wrap_view

    def set_quality_func(qualtext):
        # @refresh_wrp()
        def _wrp_qual():
            ibs.set_annot_quality_texts([aid], [qualtext])
            print('set_annot_quality(%r, %r)' % (aid, qualtext))

        return _wrp_qual

    def set_multiple_func(flag):
        # @refresh_wrp()
        def _wrp():
            ibs.set_annot_multiple([aid], [flag])
            print('set_annot_multiple(%r, %r)' % (aid, flag))

        return _wrp

    # Define popup menu
    callback_list = []

    nid = ibs.get_annot_name_rowids(aid)

    if with_interact_chip:
        callback_list += [
            (
                'Interact chip',
                partial(ishow_chip, ibs, aid, fnum=None, config2_=config2_),
            )
        ]

    if with_interact_name and not ibs.is_nid_unknown(nid):
        # from wbia.viz.interact import interact_name
        # callback_list.append(
        #    ('Interact name', partial(interact_name.ishow_name, ibs,
        #                                        nid, fnum=None))
        # )
        from wbia.viz import viz_graph2

        nid = ibs.get_annot_nids(aid)
        callback_list.append(
            (
                'New Split Interact (Annots)',
                partial(viz_graph2.make_qt_graph_interface, ibs, nids=[nid]),
            ),
        )

    if with_interact_image:
        gid = ibs.get_annot_gids(aid)
        from wbia.viz.interact import interact_annotations2

        callback_list.append(
            (
                'Interact image',
                partial(interact_annotations2.ishow_image2, ibs, gid, fnum=None),
            )
        )

    if True:
        from wbia import viz

        callback_list.append(
            (
                'Show foreground mask',
                newplot_wrp(
                    lambda: viz.show_probability_chip(ibs, aid, config2_=config2_)
                ),
            ),
        )
        callback_list.append(
            (
                'Show foreground mask (blended)',
                newplot_wrp(
                    lambda: viz.show_probability_chip(
                        ibs, aid, config2_=config2_, blend=True
                    )
                ),
            ),
        )

    if True:
        # Edit mask
        callback_list.append(
            (
                'Edit mask',
                partial(ibs.depc_annot.get_property, 'annotmask', aid, recompute=True),
            )
        )

    current_qualtext = ibs.get_annot_quality_texts([aid])[0]
    current_viewcode = ibs.get_annot_viewpoint_code([aid])[0]
    current_multiple = ibs.get_annot_multiple([aid])[0]
    # Nested viewpoints
    callback_list += [
        # ('Set Viewpoint: ' + key, set_viewpoint_func(key))
        (
            'Set &Viewpoint (%s): ' % (current_viewcode,),
            [
                (
                    '&'
                    + str(count)
                    + ' '
                    + ('*' if current_viewcode == key else '')
                    + key,
                    set_viewpoint_func(key),
                )
                for count, key in enumerate(ibs.const.VIEW.CODE_TO_NICE.keys(), start=1)
            ],
        ),
    ]
    # Nested qualities
    callback_list += [
        # ('Set Quality: ' + key, set_quality_func(key))
        (
            'Set &Quality (%s): ' % (current_qualtext,),
            [
                (
                    '&'
                    + str(count)
                    + ' '
                    + ('*' if current_qualtext == key else '')
                    + '&'
                    + key,
                    set_quality_func(key),
                )
                for count, key in enumerate(ibs.const.QUALITY_TEXT_TO_INT.keys(), start=1)
            ],
        ),
    ]

    # TODO: add set species

    callback_list += [
        (
            'Set &multiple: %r' % (not current_multiple),
            set_multiple_func(not current_multiple),
        ),
    ]

    with_tags = True
    if with_tags:
        from wbia import tag_funcs

        case_list = tag_funcs.get_available_annot_tags()
        tags = ibs.get_annot_case_tags([aid])[0]
        tags = [_.lower() for _ in tags]

        case_hotlink_list = gt.make_word_hotlinks(case_list, after_colon=True)

        def _wrap_set_annot_prop(prop, toggle_val):
            if ut.VERBOSE:
                print('[SETTING] Clicked set prop=%r to val=%r' % (prop, toggle_val,))
            ibs.set_annot_prop(prop, [aid], [toggle_val])
            if ut.VERBOSE:
                print('[SETTING] done')

        annot_tag_options = []
        for case, case_hotlink in zip(case_list, case_hotlink_list):
            toggle_val = case.lower() not in tags
            fmtstr = 'Mark %s case' if toggle_val else 'Untag %s'
            annot_tag_options += [
                # (fmtstr % (case_hotlink,), lambda:
                # ibs.set_annotmatch_prop(case, _get_annotmatch_rowid(),
                #                        [toggle_val])),
                # (fmtstr % (case_hotlink,), partial(ibs.set_annotmatch_prop,
                # case, [annotmatch_rowid], [toggle_val])),
                (
                    fmtstr % (case_hotlink,),
                    partial(_wrap_set_annot_prop, case, toggle_val),
                ),
            ]

        callback_list += [
            ('Set Annot Ta&gs', annot_tag_options),
        ]

    callback_list += [('Remove name', lambda: ibs.set_annot_name_rowids([aid], [-aid]))]

    def _setname_callback():
        import wbia.guitool as gt

        name = ibs.get_annot_name_texts([aid])[0]
        newname = gt.user_input(title='edit name', msg=name, text=name)
        if newname is not None:
            print('[ctx] _setname_callback aid=%r resp=%r' % (aid, newname))
            ibs.set_annot_name_texts([aid], [newname])

    callback_list += [('Set name', _setname_callback)]

    callback_list += [
        (
            'Unset as e&xemplar' if is_exemplar else 'Set as e&xemplar',
            toggle_exemplar_func,
        ),
    ]

    annot_info = ibs.get_annot_info(
        aid, default=True, gname=False, name=False, notes=False, exemplar=False
    )

    def print_annot_info():
        print('[interact_chip] Annotation Info = ' + ut.repr2(annot_info, nl=4))
        print('config2_ = %r' % (config2_,))
        if config2_ is not None:
            print('config2_.__dict__ = %s' % (ut.repr3(config2_.__dict__),))

    dev_callback_list = []

    def dev_edit_annot_tags():
        print('ibs = %r' % (ibs,))
        text = ibs.get_annot_tag_text([aid])[0]
        resp = gt.user_input(title='edit tags', msg=text, text=text)
        if resp is not None:
            try:
                print('resp = %r' % (resp,))
                print('[ctx] set_annot_tag_text aid=%r resp=%r' % (aid, resp))
                ibs.set_annot_tag_text(aid, resp)
                new_text = ibs.get_annot_tag_text([aid])[0]
                print('new_text = %r' % (new_text,))
                assert new_text == resp, 'should have had text change'
            except Exception as ex:
                ut.printex(ex, 'error in dev edit tags')
                raise

    def dev_set_annot_species():
        text = ibs.get_annot_species([aid])[0]
        resp = gt.user_input(title='edit species', msg=text, text=text)
        if resp is not None:
            try:
                print('resp = %r' % (resp,))
                print('[ctx] set_annot_tag_text aid=%r resp=%r' % (aid, resp))
                ibs.set_annot_species(aid, resp)
                new_text = ibs.get_annot_species_texts([aid])[0]
                print('new_text = %r' % (new_text,))
                assert new_text == resp, 'should have had text change'
            except Exception as ex:
                ut.printex(ex, 'error in dev edit species')
                raise

    dev_callback_list += [
        ('dev Edit Annot Ta&gs', dev_edit_annot_tags),
        ('dev print annot info', print_annot_info),
        ('dev refresh', pt.update),
    ]

    if ut.is_developer():

        def dev_debug():
            print('aid = %r' % (aid,))
            print('config2_ = %r' % (config2_,))

        def dev_embed(ibs=ibs, aid=aid, config2_=config2_):
            # import wbia.plottool as pt
            # pt.plt.ioff()
            # TODO need to disable matplotlib callbacks?
            # Causes can't re-enter readline error
            ut.embed()
            # pt.plt.ion()
            pass

        dev_callback_list += [
            ('dev chip context embed', dev_embed),
            ('dev chip context debug', dev_debug),
        ]
    if len(dev_callback_list) > 0:
        callback_list += [('Dev', dev_callback_list)]
    return callback_list


# def custom_chip_click(event):
#    ax = event.inaxes
#    if ih.clicked_outside_axis(event):
#        pass
#    else:
#        viztype = vh.get_ibsdat(ax, 'viztype')
#        print('[ic] viztype=%r' % viztype)
#        if viztype == 'chip':
#            if event.button == 3:   # right-click
#                from wbia.viz.interact import interact_chip
#                height = fig.canvas.geometry().height()
#                qpoint = gt.newQPoint(event.x, height - event.y)
#                refresh_func = partial(_chip_view, **kwargs)
#                interact_chip.show_annot_context_menu(
#                    ibs, aid, fig.canvas, qpoint, refresh_func=refresh_func,
#                    with_interact_chip=False, config2_=config2_)


# CHIP INTERACTION 2
def ishow_chip(
    ibs, aid, fnum=2, fx=None, dodraw=True, config2_=None, ischild=False, **kwargs
):
    r"""

    # TODO:
        split into two interactions
        interact chip and interact chip features

    Args:
        ibs (IBEISController):  wbia controller object
        aid (int):  annotation id
        fnum (int):  figure number
        fx (None):

    CommandLine:
        python -m wbia.viz.interact.interact_chip --test-ishow_chip --show
        python -m wbia.viz.interact.interact_chip --test-ishow_chip --show --aid 2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.interact.interact_chip import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
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
    fnum = pt.ensure_fnum(fnum)
    vh.ibsfuncs.assert_valid_aids(ibs, (aid,))
    # TODO: Reconcile this with interact keypoints.
    # Preferably this will call that but it will set some fancy callbacks
    if not ischild:
        fig = ih.begin_interaction('chip', fnum)
    else:
        fig = pt.gcf()
        # fig = pt.figure(fnum=fnum, pnum=pnum)

    # Get chip info (make sure get_chips is called first)
    # mode_ptr = [1]
    mode_ptr = [0]

    def _select_fxth_kpt(fx):
        from wbia.plottool.viz_featrow import draw_feat_row

        # Get the fx-th keypiont
        chip = ibs.get_annot_chips(aid, config2_=config2_)
        kp = ibs.get_annot_kpts(aid, config2_=config2_)[fx]
        sift = ibs.get_annot_vecs(aid, config2_=config2_)[fx]
        # Draw chip + keypoints + highlighted plots
        _chip_view(pnum=(2, 1, 1), sel_fx=fx)
        # ishow_chip(ibs, aid, fnum=None, fx=fx, config2_=config2_, **kwargs)
        # Draw the selected feature plots
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _chip_view(mode=0, pnum=(1, 1, 1), **kwargs):
        print('... _chip_view mode=%r' % mode_ptr[0])
        kwargs['ell'] = mode_ptr[0] == 1
        kwargs['pts'] = mode_ptr[0] == 2

        if not ischild:
            pt.figure(fnum=fnum, pnum=pnum, docla=True, doclf=True)
        # Toggle no keypoints view
        viz.show_chip(ibs, aid, fnum=fnum, pnum=pnum, config2_=config2_, **kwargs)
        pt.set_figtitle('Chip View')

    def _on_chip_click(event):
        print('[inter] clicked chip')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ih.clicked_outside_axis(event):
            if not ischild:
                print('... out of axis')
                mode_ptr[0] = (mode_ptr[0] + 1) % 3
                _chip_view(**kwargs)
        else:
            if event.button == 3:  # right-click
                import wbia.guitool as gt

                # from wbia.viz.interact import interact_chip
                height = fig.canvas.geometry().height()
                qpoint = gt.newQPoint(event.x, height - event.y)
                refresh_func = partial(_chip_view, **kwargs)

                callback_list = build_annot_context_options(
                    ibs,
                    aid,
                    refresh_func=refresh_func,
                    with_interact_chip=False,
                    config2_=config2_,
                )
                qwin = fig.canvas
                gt.popup_menu(qwin, qpoint, callback_list)
                # interact_chip.show_annot_context_menu(
                #    ibs, aid, fig.canvas, qpoint, refresh_func=refresh_func,
                #    with_interact_chip=False, config2_=config2_)
            else:
                viztype = vh.get_ibsdat(ax, 'viztype')
                print('[ic] viztype=%r' % viztype)
                if viztype == 'chip' and event.key == 'shift':
                    _chip_view(**kwargs)
                    ih.disconnect_callback(fig, 'button_press_event')
                elif viztype == 'chip':
                    kpts = ibs.get_annot_kpts(aid, config2_=config2_)
                    if len(kpts) > 0:
                        fx = vt.nearest_point(x, y, kpts, conflict_mode='next')[0]
                        print('... clicked fx=%r' % fx)
                        _select_fxth_kpt(fx)
                    else:
                        print('... len(kpts) == 0')
                elif viztype in ['warped', 'unwarped']:
                    fx = vh.get_ibsdat(ax, 'fx')
                    if fx is not None and viztype == 'warped':
                        viz.show_keypoint_gradient_orientations(
                            ibs, aid, fx, fnum=pt.next_fnum()
                        )
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

    if not ischild:
        ih.connect_callback(fig, 'button_press_event', _on_chip_click)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.interact.interact_chip
        python -m wbia.viz.interact.interact_chip --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
