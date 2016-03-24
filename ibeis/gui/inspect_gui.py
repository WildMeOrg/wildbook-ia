# -*- coding: utf-8 -*-
"""
This module was never really finished. It is used in some cases
to display the results from a query in a qt window. It needs
some work if its to be re-integrated.

TODO:
    Refresh name table on inspect gui close
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from functools import partial
from guitool import (qtype, APIItemWidget, APIItemModel, FilterProxyModel,
                     ChangeLayoutContext)
from guitool.__PYQT__ import QtGui, QtCore
from ibeis.other import ibsfuncs
#from ibeis.viz import interact
from ibeis.viz import viz_helpers as vh
#from ibeis.algo.hots import chip_match
from plottool import fig_presenter
#from plottool import interact_helpers as ih
#import functools
import guitool
import numpy as np
import six
import utool
#from ibeis import constants as const
import utool as ut
from ibeis.gui import guiexcept
(print, rrr, profile) = utool.inject2(__name__, '[inspect_gui]')


MATCHED_STATUS_TEXT  = 'Matched'
REVIEWED_STATUS_TEXT = 'Reviewed'
USE_FILTER_PROXY = False


def get_aidpair_context_menu_options(ibs, aid1, aid2, cm, qreq_=None,
                                     marking_mode=False,
                                     aid_list=None, **kwargs):
    """ assert that the ampersand cannot have duplicate keys

    Args:
        ibs (ibeis.IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        cm (ibeis.ChipMatch):  object of feature correspondences and scores
        qreq_ (ibeis.QueryRequest):  query request object with hyper-parameters(default = None)
        aid_list (list):  list of annotation rowids(default = None)

    Returns:
        list: options

    CommandLine:
        python -m ibeis.gui.inspect_gui --exec-get_aidpair_context_menu_options
        python -m ibeis.gui.inspect_gui --exec-get_aidpair_context_menu_options --verbose
        python -m ibeis.gui.inspect_gui --exec-get_aidpair_context_menu_options --verbose -a timecontrolled -t invarbest --db PZ_Master1  --qaid 574

        # Other scripts that call this one;w
        python -m ibeis.dev -e cases --db PZ_Master1  -a timectrl -t best --filt :sortdsc=gfscore,fail=True,min_gtscore=.0001 --show
        python -m ibeis.dev -e cases --db PZ_MTEST  -a timectrl -t best --filt :sortdsc=gfscore,fail=True,min_gtscore=.0001 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qreq_ = ibeis.main_helpers.testdata_qreq_(t=['default:fg_on=False'])
        >>> cm_list = ibs.query_chips(qreq_=qreq_, return_cm=True)
        >>> cm = cm_list[0]
        >>> ibs = qreq_.ibs
        >>> aid1 = cm.qaid
        >>> aid2 = cm.get_top_aids()[0]
        >>> aid_list = None
        >>> options = get_aidpair_context_menu_options(ibs, aid1, aid2, cm, qreq_, aid_list)
        >>> result = ('options = %s' % (ut.list_str(options),))
        >>> print(result)
    """
    if ut.VERBOSE:
        print('[inspect_gui] Building AID pair context menu options')
    options = []

    #assert qreq_ is not None, 'must specify qreq_'

    if cm is not None:
        # MAKE SURE THIS IS ALL CM
        show_chip_match_features_option = (
            'Show chip feature matches',
            partial(cm.ishow_single_annotmatch, qreq_, aid2, mode=0))
        if aid_list is not None:
            # Give a subcontext menu for multiple options
            def partial_show_chip_matches_to(aid_):
                return lambda: cm.ishow_single_annotmatch(qreq_, aid_, mode=0)
            show_chip_match_features_option = (
                'Show chip feature matches',
                [
                    ('to aid=%r' % (aid_,), partial_show_chip_matches_to(aid_))
                    for aid_ in aid_list
                ]
            )

        def show_single_namematch():
            import plottool as pt
            ax = cm.show_single_namematch(qreq_, aid2, mode=0)
            ax = pt.gca()
            ax.figure.canvas.draw()
            pt.update()

        options += [
            show_chip_match_features_option,
            ('Show name feature matches', show_single_namematch),
        ]

    with_interact_chips = True

    if with_interact_chips:
        from ibeis.viz.interact import interact_chip

        aid_list2 = [aid1, aid2]
        if qreq_ is None:
            config2_list_ = [None, None]
        else:
            config2_list_ = [qreq_.get_external_query_config2(),
                             qreq_.get_external_data_config2()]

        #interact_chip_options = []
        #for count, (aid, config2_) in enumerate(zip(aid_list2,
        #                                            config2_list_),
        #                                        start=1):
        #    interact_chip_options += [
        #        ('Interact Annot&%d' % (count,),
        #         partial(interact_chip.ishow_chip, ibs, aid, config2_=config2_,
        #                 fnum=None, **kwargs)),
        #    ]
        #interact_chip_actions = ut.get_list_column(interact_chip_options, 1)
        #interact_chip_options.append(
        #    ('Interact &All Annots', lambda: [func() for func in
        #                                      interact_chip_actions]),
        #)

        chip_contex_options = []
        print('config2_list_ = %r' % (config2_list_,))
        for count, (aid, config2_) in enumerate(zip(aid_list2, config2_list_),
                                                start=1):
            chip_contex_options += [
                ('Annot&%d Options (aid=%r)' % (count, aid,),
                 interact_chip.build_annot_context_options(
                    ibs, aid, refresh_func=None, config2_=config2_))
            ]

        #options += [
        #    #('Interact Annots', interact_chip_options),
        #    #('Annot Conte&xt Options', chip_contex_options),
        #]
        if len(chip_contex_options) > 2:
            options += [
                ('Annot Conte&xt Options', chip_contex_options),
            ]
        else:
            options += chip_contex_options

    with_review_options = True

    from ibeis.viz import viz_graph
    if with_review_options:
        options += [
            ('Mark as &Reviewed', lambda: ibs.set_annot_pair_as_reviewed(aid1,
                                                                         aid2)),
            ('Mark as &True Match.',
             lambda: set_annot_pair_as_positive_match_(
                 ibs, aid1, aid2, cm, qreq_, **kwargs)),
            ('Mark as &False Match.',
             lambda:  set_annot_pair_as_negative_match_(
                 ibs, aid1, aid2, cm, qreq_, **kwargs)),

            #('Mark Disjoint Viewpoints.',
            # lambda:  set_annot_pair_as_unknown_match_(
            #     ibs, aid1, aid2, cm, qreq_, **kwargs)),

            ('Inspect Match Candidates',
             lambda: review_match(
                 ibs, aid1, aid2, qreq_=qreq_, cm=cm, **kwargs)),
            ('Interact Name Graph',
             partial(viz_graph.make_name_graph_interaction,
                     ibs, aids=aid_list2, selected_aids=aid_list2))
        ]

    with_vsone = True
    if with_vsone:
        from ibeis.algo.hots import vsone_pipeline

        #vsone_qreq_ = qreq_.shallowcopy(qaids=[aid1])
        def vsone_single_hack(ibs, qaid, daid, qreq_):
            import vtool as vt
            import plottool as pt
            if qreq_ is None:
                qreq2_ = ibs.new_query_request([qaid], [daid], cfgdict={})
            else:
                qreq2_ = ibs.new_query_request([qaid], [daid], cfgdict=qreq_.qparams)
            matches, metadata = vsone_pipeline.vsone_single(qaid, daid, qreq2_, use_ibscache=True)
            interact = vt.matching.show_matching_dict(matches, metadata, mode=1)  # NOQA
            pt.update()

        options += [
            ('VsOne', [
                ('Run Vsone(ib)', partial(vsone_pipeline.vsone_independant_pair_hack,
                                          ibs, aid1, aid2, qreq_=qreq_)),
                ('Run Vsone(vt)', partial(vsone_single_hack,
                                          ibs, aid1, aid2, qreq_=qreq_)),
            ]
            )
        ]
    with_vsmany = True
    if with_vsmany:
        def vsmany_load_and_show():
            if qreq_ is None:
                print('no qreq_ given')
                return None
                # qreq2_ = ibs.new_query_request([qaid], [daid], cfgdict={})
            else:
                qreq2_ = qreq_
            cm = qreq2_.execute_subset([aid1])[0]
            cm.ishow_single_annotmatch(qreq_, aid2, mode=0)
        options += [
            ('Load Vsmany', vsmany_load_and_show),
        ]
        pass

    with_match_tags = True
    if with_match_tags:
        annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey(
            [aid1], [aid2])[0]

        if annotmatch_rowid is None:
            tags = []
        else:
            tags = ibs.get_annotmatch_case_tags([annotmatch_rowid])[0]
            tags = [_.lower() for _ in tags]
        from ibeis import tag_funcs
        standard, other = tag_funcs.get_cate_categories()
        case_list = standard + other

        #used_chars = guitool.find_used_chars(ut.get_list_column(options, 0))
        used_chars = []
        case_hotlink_list = guitool.make_word_hotlinks(case_list, used_chars)
        case_options = []
        if True or ut.VERBOSE:
            print('[inspect_gui] aid1, aid2 = %r, %r' % (aid1, aid2,))
            print('[inspect_gui] annotmatch_rowid = %r' % (annotmatch_rowid,))
            print('[inspect_gui] tags = %r' % (tags,))
        if ut.VERBOSE:
            print('[inspect_gui] Making case hotlist: ' +
                  ut.list_str(case_hotlink_list))

        def _wrap_set_annotmatch_prop(prop, toggle_val):
            if ut.VERBOSE:
                print('[SETTING] Clicked set prop=%r to val=%r' %
                      (prop, toggle_val,))
            am_rowid = ibs.add_annotmatch([aid1], [aid2])[0]
            if ut.VERBOSE:
                print('[SETTING] aid1, aid2 = %r, %r' % (aid1, aid2,))
                print('[SETTING] annotmatch_rowid = %r' % (am_rowid,))
            ibs.set_annotmatch_prop(prop, [am_rowid], [toggle_val])
            if ut.VERBOSE:
                print('[SETTING] done')
            if True:
                # hack for reporting
                if annotmatch_rowid is None:
                    tags = []
                else:
                    tags = ibs.get_annotmatch_case_tags([annotmatch_rowid])[0]
                    tags = [_.lower() for _ in tags]
                print('[inspect_gui] aid1, aid2 = %r, %r' % (aid1, aid2,))
                print('[inspect_gui] annotmatch_rowid = %r' %
                      (annotmatch_rowid,))
                print('[inspect_gui] tags = %r' % (tags,))

        for case, case_hotlink in zip(case_list, case_hotlink_list):
            toggle_val = case.lower() not in tags
            fmtstr = 'Flag %s case' if toggle_val else 'Unflag %s case'
            case_options += [
                #(fmtstr % (case_hotlink,), lambda:
                #ibs.set_annotmatch_prop(case, _get_annotmatch_rowid(),
                #                        [toggle_val])),
                #(fmtstr % (case_hotlink,), partial(ibs.set_annotmatch_prop,
                #case, [annotmatch_rowid], [toggle_val])),
                (fmtstr % (case_hotlink,), partial(_wrap_set_annotmatch_prop,
                                                   case, toggle_val)),
            ]
        if ut.VERBOSE:
            print('Partial tag funcs:' +
                  ut.list_str(
                      [ut.func_str(func, func.args, func.keywords)
                       for func in ut.get_list_column(case_options, 1)]))
        options += [
            ('Match Ta&gs', case_options)
        ]

    if ut.is_developer():
        def dev_debug():
            print('=== DBG ===')
            print('ibs = %r' % (ibs,))
            print('cm = %r' % (cm,))
            print('aid1 = %r' % (aid1,))
            print('aid2 = %r' % (aid2,))
            print('qreq_ = %r' % (qreq_,))
            cm.print_inspect_str(qreq_)
            cm.print_rawinfostr()

            cm2 = cm.extend_results(qreq_)
            cm2.print_inspect_str(qreq_)
            cm2.print_rawinfostr()

        def dev_embed(ibs=ibs, aid1=aid1, aid2=aid2, cm=cm, qreq_=qreq_):
            ut.embed()
        options += [
            ('dev pair context embed', dev_embed),
            ('dev pair context debug', dev_debug),
        ]
    return options


class CustomFilterModel(FilterProxyModel):
    def __init__(model, headers=None, parent=None, *args):
        FilterProxyModel.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.imgsetid = -1  # negative one is an invalid imgsetid  # seems unused
        model.original_ider = None
        model.sourcemodel = APIItemModel(parent=parent)
        model.setSourceModel(model.sourcemodel)
        print('[ibs_model] just set the sourcemodel')

    def _update_headers(model, **headers):
        def _null_ider(**kwargs):
            return []
        model.original_iders = headers.get('iders', [_null_ider])
        if len(model.original_iders) > 0:
            model.new_iders = model.original_iders[:]
            model.new_iders[0] = model._ider
        headers['iders'] = model.new_iders
        model.sourcemodel._update_headers(**headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected imageset ids """
        return model.original_iders[0]()

    def _change_imageset(model, imgsetid):
        model.imgsetid = imgsetid
        # seems unused
        with ChangeLayoutContext([model]):
            FilterProxyModel._update_rows(model)


class QueryResultsWidget(APIItemWidget):
    """ Window for gui inspection

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show

    """

    def __init__(qres_wgt, ibs, cm_list, parent=None, callback=None,
                 name_scoring=False, qreq_=None, **kwargs):

        assert not isinstance(cm_list, dict)

        if ut.VERBOSE:
            print('[qres_wgt] Init QueryResultsWidget')
        # Uncomment below to turn on FilterProxyModel
        if USE_FILTER_PROXY:
            APIItemWidget.__init__(qres_wgt, parent=parent,
                                    model_class=CustomFilterModel)
        else:
            APIItemWidget.__init__(qres_wgt, parent=parent)

        #qres_wgt.altkey_shortcut =
        #QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.ALT), qres_wgt,
        #                qres_wgt.on_alt_pressed,
        #                context=QtCore..Qt.WidgetShortcut)
        qres_wgt.button_list = None
        qres_wgt.show_new = True
        qres_wgt.show_join = True
        qres_wgt.show_split = True
        qres_wgt.tt = utool.tic()
        # Set results data
        if USE_FILTER_PROXY:
            qres_wgt.add_checkboxes(qres_wgt.show_new, qres_wgt.show_join,
                                    qres_wgt.show_split)
        qres_wgt.set_query_results(ibs, cm_list, name_scoring=name_scoring,
                                   qreq_=qreq_, **kwargs)
        qres_wgt.connect_signals_and_slots()
        if callback is None:
            callback = partial(ut.identity, None)
        qres_wgt.callback = callback
        qres_wgt.view.setColumnHidden(0, False)
        qres_wgt.view.setColumnHidden(1, False)
        #qres_wgt.view.connect_single_key_to_slot(QtCore.Qt.ALT,
        #qres_wgt.on_alt_pressed)
        ALT_KEY = 16777251
        qres_wgt.view.connect_single_key_to_slot(ALT_KEY,
                                                 qres_wgt.on_alt_pressed)
        qres_wgt.view.connect_keypress_to_slot(qres_wgt.on_special_key_pressed)
        if parent is None:
            # Register parentless QWidgets
            fig_presenter.register_qt4_win(qres_wgt)

    def set_query_results(qres_wgt, ibs, cm_list, name_scoring=False,
                          qreq_=None, **kwargs):
        print('[qres_wgt] set_query_results()')
        tblnice = 'Query Results: ' + kwargs.get('query_title', '')
        ut.util_dict.delete_dict_keys(kwargs, ['query_title'])

        qres_wgt.ibs = ibs
        qres_wgt.qaid2_cm = dict([(cm.qaid, cm) for cm in cm_list])
        qres_wgt.qreq_ = qreq_
        qres_wgt.qres_api = make_qres_api(ibs, cm_list,
                                          name_scoring=name_scoring,
                                          qreq_=qreq_, **kwargs)
        qres_wgt.update_checkboxes()

        headers = qres_wgt.qres_api.make_headers(tblname='qres_api',
                                                 tblnice=tblnice)

        # HACK IN ROW SIZE
        vertical_header = qres_wgt.view.verticalHeader()
        vertical_header.setDefaultSectionSize(
            qres_wgt.qres_api.get_thumb_size())

        # super call
        APIItemWidget.change_headers(qres_wgt, headers)
        #qres_wgt.change_headers(headers)

        # HACK IN COL SIZE
        horizontal_header = qres_wgt.view.horizontalHeader()
        for col, width in six.iteritems(qres_wgt.qres_api.col_width_dict):
            #horizontal_header.defaultSectionSize()
            index = qres_wgt.qres_api.col_name_list.index(col)
            horizontal_header.resizeSection(index, width)

    @guitool.slot_()
    def closeEvent(qres_wgt, event):
        event.accept()
        if qres_wgt.callback is not None:
            # update names tree after closing
            qres_wgt.callback()

    def add_checkboxes(qres_wgt, show_new, show_join, show_split):
        _CHECK  = partial(guitool.newCheckBox, qres_wgt)
        qres_wgt.button_list = [
            [
                _CHECK('Show New Matches',
                        qres_wgt._check_changed,
                        checked=show_new),

                _CHECK('Show Join Matches',
                        qres_wgt._check_changed,
                        checked=show_join),

                _CHECK('Show Split Matches',
                        qres_wgt._check_changed,
                        checked=show_split),
            ]
        ]

        qres_wgt.buttonBars = []
        for row in qres_wgt.button_list:
            qres_wgt.buttonBars.append(QtGui.QHBoxLayout(qres_wgt))
            qres_wgt.vert_layout.addLayout(qres_wgt.buttonBars[-1])
            for button in row:
                qres_wgt.buttonBars[-1].addWidget(button)

    def update_checkboxes(qres_wgt):
        if qres_wgt.button_list is None:
            return
        show_new = qres_wgt.button_list[0][0].isChecked()
        show_join = qres_wgt.button_list[0][1].isChecked()
        show_split = qres_wgt.button_list[0][2].isChecked()
        if USE_FILTER_PROXY:
            qres_wgt.model.update_filterdict({
                'NEW Match ':   show_new,
                'JOIN Match ':  show_join,
                'SPLIT Match ': show_split,
            })
        qres_wgt.model._update_rows()

    def _check_changed(qres_wgt, value):
        qres_wgt.update_checkboxes()

    def sizeHint(qres_wgt):
        # should eventually improve this to use the widths of the header columns
        return QtCore.QSize(1000, 500)

    def connect_signals_and_slots(qres_wgt):
        qres_wgt.view.clicked.connect(qres_wgt._on_click)
        qres_wgt.view.doubleClicked.connect(qres_wgt._on_doubleclick)
        qres_wgt.view.pressed.connect(qres_wgt._on_pressed)
        qres_wgt.view.activated.connect(qres_wgt._on_activated)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_click(qres_wgt, qtindex):
        #print('[qres_wgt] _on_click: ')
        #print('[qres_wgt] _on_click: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)

        if colname == MATCHED_STATUS_TEXT:
            #qres_callback = partial(show_match_at_qtindex, qres_wgt, qtindex)
            #review_match_at_qtindex(qres_wgt, qtindex,
            #qres_callback=qres_callback)
            review_match_at_qtindex(qres_wgt, qtindex)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(qres_wgt, qtindex):
        print('[qres_wgt] _on_doubleclick: ')
        print('[qres_wgt] DoubleClicked: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        if qres_wgt.qres_api.col_edit_list[col]:
            print('do nothing special for editable columns')
            return
        model = qtindex.model()
        colname = model.get_header_name(col)
        if colname != MATCHED_STATUS_TEXT:
            return show_match_at_qtindex(qres_wgt, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(qres_wgt, qtindex):
        print('[qres_wgt] _on_pressed: ')
        def _check_for_double_click(qres_wgt, qtindex):
            threshold = 0.20  # seconds
            distance = utool.toc(qres_wgt.tt)
            #print('Pressed %r' % (distance,))
            col = qtindex.column()
            model = qtindex.model()
            colname = model.get_header_name(col)
            if distance <= threshold:
                if colname == MATCHED_STATUS_TEXT:
                    qres_wgt.view.clicked.emit(qtindex)
                    qres_wgt._on_click(qtindex)
                else:
                    #qres_wgt.view.doubleClicked.emit(qtindex)
                    qres_wgt._on_doubleclick(qtindex)
            qres_wgt.tt = utool.tic()
        _check_for_double_click(qres_wgt, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_activated(qres_wgt, qtindex):
        print('Activated: ' + str(qtype.qindexinfo(qtindex)))
        pass

    #@guitool.slot_()
    def on_alt_pressed(qres_wgt, view, event):
        selected_qtindex_list = view.selectedIndexes()
        if len(selected_qtindex_list) == 1:
            # popup context menu on alt
            qtindex = selected_qtindex_list[0]
            qrect = view.visualRect(qtindex)
            pos = qrect.center()
            qres_wgt.on_contextMenuRequested(qtindex, pos)

    #@guitool.slot_()
    def on_special_key_pressed(qres_wgt, view, event):
        selected_qtindex_list = view.selectedIndexes()

        if len(selected_qtindex_list) == 1:
            print('event = %r ' % (event,))
            print('event.key() = %r ' % (event.key(),))
            qtindex = selected_qtindex_list[0]
            ibs = qres_wgt.ibs
            aid1, aid2 = get_aidpair_from_qtindex(qres_wgt, qtindex)
            _tup = get_widget_review_vars(qres_wgt, aid1)
            ibs, cm, qreq_, update_callback, backend_callback = _tup
            options = get_aidpair_context_menu_options(
                ibs, aid1, aid2, cm, qreq_=qreq_,
                update_callback=update_callback,
                backend_callback=backend_callback)
            option_dict = {key[key.find('&') + 1]: val for key, val in options
                           if key.find('&') > -1}

            event_key = event.key()
            if event_key == QtCore.Qt.Key_R:
                option_dict['R']()
            elif event_key == QtCore.Qt.Key_T:
                option_dict['T']()
            elif event_key == QtCore.Qt.Key_F:
                option_dict['F']()
            elif event_key == QtCore.Qt.Key_S:
                option_dict['S']()
            elif event_key == QtCore.Qt.Key_P:
                option_dict['P']()
            print('emiting data changed')
            # This may not work with PyQt5
            # http://stackoverflow.com/questions/22560296/pyqt-list-view-not-responding-to-datachanged-signal
            model = qtindex.model()
            # This should work by itself
            model.dataChanged.emit(qtindex, qtindex)
            # but it doesnt seem to be, but this seems to solve the issue
            model.layoutChanged.emit()
            print('emited data changed')
            #model.select()

    @guitool.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(qres_wgt, qtindex, qpoint):
        """
        popup context menu
        """
        qwin = qres_wgt
        aid1, aid2 = get_aidpair_from_qtindex(qres_wgt, qtindex)
        tup = get_widget_review_vars(qres_wgt, aid1)
        ibs, cm, qreq_, update_callback, backend_callback = tup
        show_aidpair_context_menu(ibs, qwin, qpoint, aid1, aid2, cm,
                                  qreq_=qreq_, update_callback=update_callback,
                                  backend_callback=backend_callback)


def get_widget_review_vars(qres_wgt, qaid):
    ibs   = qres_wgt.ibs
    qreq_ = qres_wgt.qreq_
    cm  = qres_wgt.qaid2_cm[qaid]
    update_callback = None  # hack (checking if necessary)
    backend_callback = qres_wgt.callback
    return ibs, cm, qreq_, update_callback, backend_callback


def get_aidpair_from_qtindex(qres_wgt, qtindex):
    model = qtindex.model()
    qaid  = model.get_header_data('qaid', qtindex)
    daid  = model.get_header_data('aid', qtindex)
    return qaid, daid


def get_annotmatch_rowid_from_qtindex(qres_wgt, qtindex):
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    ibs = qres_wgt.ibs
    annotmatch_rowid_list = ibs.add_annotmatch([qaid], [daid])
    return annotmatch_rowid_list


def show_match_at_qtindex(qres_wgt, qtindex):
    print('interact')
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    cm = qres_wgt.qaid2_cm[qaid]
    match_interaction = cm.ishow_single_annotmatch(
        qres_wgt.qreq_, daid, mode=0)
    fig = match_interaction.fig
    fig_presenter.bring_to_front(fig)


def review_match_at_qtindex(qres_wgt, qtindex):
    print('review')
    #qres_callback = partial(show_match_at_qtindex, qres_wgt, qtindex)
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    #update_callback = model._update
    #ibs   = qres_wgt.ibs
    #qreq_ = qres_wgt.qreq_
    #cm  = qres_wgt.qaid2_cm[qaid]
    #update_callback = None  # hack (checking if necessary)
    #backend_callback = qres_wgt.callback
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    tup = get_widget_review_vars(qres_wgt, qaid)
    ibs, cm, qreq_, update_callback, backend_callback = tup
    review_match(ibs, qaid, daid, update_callback=update_callback,
                 backend_callback=backend_callback, cm=cm, qreq_=qreq_)


# ______


def show_aidpair_context_menu(ibs, qwin, qpoint, aid1, aid2, cm, qreq_=None,
                              **kwargs):
    """
    kwargs are used for callbacks like qres_callback and query_callback
    """
    options = get_aidpair_context_menu_options(ibs, aid1, aid2, cm,
                                               qreq_=qreq_, **kwargs)
    guitool.popup_menu(qwin, qpoint, options)


def set_annot_pair_as_positive_match_(ibs, aid1, aid2, cm, qreq_, **kwargs):
    def on_nontrivial_merge(ibs, aid1, aid2):
        MERGE_NEEDS_INTERACTION  = False
        MERGE_NEEDS_VERIFICATION = True
        if MERGE_NEEDS_INTERACTION:
            raise guiexcept.NeedsUserInput('confirm merge')
        elif MERGE_NEEDS_VERIFICATION:
            name1, name2 = ibs.get_annot_names([aid1, aid2])
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
            msgfmt = ut.codeblock('''
               Confirm merge of animal {name1} and {name2}
               {name1} has {num_gt1} annotations
               {name2} has {num_gt2} annotations
               ''')
            msg = msgfmt.format(name1=name1, name2=name2,
                                num_gt1=len(aid1_and_groundtruth),
                                num_gt2=len(aid2_and_groundtruth),)
            if not guitool.are_you_sure(parent=None, msg=msg, default='Yes'):
                raise guiexcept.UserCancel('canceled merge')
    try:
        status = ibs.set_annot_pair_as_positive_match(
            aid1, aid2, on_nontrivial_merge=on_nontrivial_merge)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match(ibs, aid1, aid2, qreq_=qreq_, cm=cm, **kwargs)
    except guiexcept.UserCancel:
        print('user canceled positive match')


def set_annot_pair_as_negative_match_(ibs, aid1, aid2, cm, qreq_, **kwargs):
    def on_nontrivial_split(ibs, aid1, aid2):
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        print('There are %d annots in this name. Need more sophisticated split'
              % (len(aid1_groundtruth)))
        raise guiexcept.NeedsUserInput('non-trivial split')
    try:
        status = ibs.set_annot_pair_as_negative_match(
            aid1, aid2, on_nontrivial_split=on_nontrivial_split)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match(ibs, aid1, aid2, qreq_=qreq_, cm=cm, **kwargs)
    except guiexcept.UserCancel:
        print('user canceled negative match')


def review_match(ibs, aid1, aid2, update_callback=None, backend_callback=None,
                 qreq_=None, cm=None, **kwargs):
    print('Review match: ' + ibsfuncs.vsstr(aid1, aid2))
    from ibeis.viz.interact import interact_name
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    mvinteract = interact_name.MatchVerificationInteraction(
        ibs, aid1, aid2, fnum=64, update_callback=update_callback,
        cm=cm,
        qreq_=qreq_,
        backend_callback=backend_callback, **kwargs)
    return mvinteract
    #ih.register_interaction(mvinteract)


def get_match_status(ibs, aid_pair):
    """ Data role for status column
    FIXME: no other function in this project takes a tuple of scalars as an
    argument. Everything else is written in the context of lists, This function
    should follow the same paradigm, but CustomAPI will have to change.
    """
    aid1, aid2 = aid_pair
    assert not utool.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not utool.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    #text  = ibsfuncs.vsstr(aid1, aid2)
    text = ibs.get_match_text(aid1, aid2)
    if text is None:
        raise AssertionError('impossible state inspect_gui')
    return text


def get_reviewed_status(ibs, aid_pair):
    """ Data role for status column
    FIXME: no other function in this project takes a tuple of scalars as an
    argument. Everything else is written in the context of lists, This function
    should follow the same paradigm, but CustomAPI will have to change.
    """
    aid1, aid2 = aid_pair
    assert not utool.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not utool.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    #text  = ibsfuncs.vsstr(aid1, aid2)
    annotmach_reviewed = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    return 'Yes' if annotmach_reviewed else 'No'
    #text = ibs.get_match_text(aid1, aid2)
    #if text is None:
    #    raise AssertionError('impossible state inspect_gui')
    #return 'No'


def get_match_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


def get_reviewed_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    annotmach_reviewed = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    #truth = ibs.get_annot_pair_truth([aid1], [aid2])[0]
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    lighten_amount = .35 if annotmach_reviewed else .9
    truth_color = vh.get_truth_color(truth, base255=True,
                                     lighten_amount=lighten_amount)
    #truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    #truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


def test_inspect_matches(ibs, qaid_list, daid_list):
    """

    Args:
        ibs       (IBEISController):
        qaid_list (list): query annotation id list
        daid_list (list): database annotation id list

    Returns:
        dict: locals_

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show --nodelete
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --cmd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> import guitool
        >>> ibs = ibeis.opendb(db='PZ_MTEST')
        >>> assert ibs.dbname == 'PZ_MTEST', 'do not use on a real database'
        >>> qaid_list = ibs.get_valid_aids()[0:5]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> if not ut.get_argflag('--nodelete'):
        >>>     ibs.delete_annotmatch(ibs._get_all_annotmatch_rowids())
        >>> main_locals = test_inspect_matches(ibs, qaid_list, daid_list)
        >>> main_execstr = ibeis.main_loop(main_locals)
        >>> ut.quit_if_noshow()
        >>> # TODO: add in qwin to main loop
        >>> guitool.qtapp_loop(qwin=main_locals['qres_wgt'])
        >>> print(main_execstr)
        >>> exec(main_execstr)
    """
    from ibeis.gui import inspect_gui
    qreq_ = ibs.new_query_request(qaid_list, daid_list, cfgdict={'augment_queryside_hack': True})
    cm_list = ibs.query_chips(qreq_=qreq_, return_cm=True)
    tblname = ''
    name_scoring = False
    ranks_lt = 5
    # This is where you create the result widigt
    guitool.ensure_qapp()
    print('[inspect_matches] make_qres_widget')
    #ut.view_directory(ibs.get_match_thumbdir())
    qres_wgt = inspect_gui.QueryResultsWidget(
        ibs, cm_list, ranks_lt=ranks_lt, qreq_=qreq_, filter_reviewed=False,
        filter_duplicate_namepair_matches=True)
    print('[inspect_matches] show')
    qres_wgt.show()
    print('[inspect_matches] raise')
    qres_wgt.raise_()
    print('</inspect_matches>')
    # simulate double click
    #qres_wgt._on_click(qres_wgt.model.index(2, 2))
    #qres_wgt._on_doubleclick(qres_wgt.model.index(2, 0))
    locals_ =  locals()
    return locals_


def get_match_thumb_fname(cm, daid, qreq_):
    """
    CommandLine:
        python -m ibeis.gui.inspect_gui --exec-get_match_thumb_fname

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> cm, qreq_ = ibeis.testdata_cm('PZ_MTEST')
        >>> thumbsize = (128, 128)
        >>> daid = cm.get_top_aids()[0]
        >>> match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
        >>> result = match_thumb_fname
        >>> print(result)
        match_aids=1,1_cfgstr=ubpzwu5k54h6xbnr.jpg
    """
    # Make thumbnail name
    config_hash = ut.hashstr27(qreq_.get_cfgstr())
    qaid = cm.qaid
    match_thumb_fname = 'match_aids=%d,%d_cfgstr=%s.jpg' % ((qaid, daid,
                                                             config_hash))
    return match_thumb_fname


def ensure_match_img(ibs, cm, daid, qreq_=None, match_thumbtup_cache={}):
    r"""
    CommandLine:
        python -m ibeis.gui.inspect_gui --test-ensure_match_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> cm, qreq_ = ibeis.testdata_cm()
        >>> daid = cm.get_top_aids()[0]
        >>> match_thumbtup_cache = {}
        >>> # execute function
        >>> match_thumb_fpath_ = ensure_match_img(qreq_.ibs, cm, daid, qreq_, match_thumbtup_cache)
        >>> # verify results
        >>> result = str(match_thumb_fpath_)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> ut.startfile(match_thumb_fpath_, quote=True)
    """
    #from os.path import exists
    match_thumbdir = ibs.get_match_thumbdir()
    match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
    match_thumb_fpath_ = ut.unixjoin(match_thumbdir, match_thumb_fname)
    #if exists(match_thumb_fpath_):
    #    return match_thumb_fpath_
    if match_thumb_fpath_ in match_thumbtup_cache:
        fpath = match_thumbtup_cache[match_thumb_fpath_]
    else:
        # TODO: just draw the image at the correct thumbnail size
        # TODO: draw without matplotlib?
        fpath = cm.imwrite_single_annotmatch(
            qreq_, daid, fpath=match_thumb_fpath_, saveax=True, fnum=32,
            notitle=True, verbose=False)
        match_thumbtup_cache[match_thumb_fpath_] = fpath
    return fpath


def make_qres_api(ibs, cm_list, ranks_lt=None, name_scoring=False,
                  filter_reviewed=False,
                  filter_duplicate_namepair_matches=False,
                  qreq_=None,
                  ):
    """
    Builds columns which are displayable in a ColumnListTableWidget

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show
        python -m ibeis.gui.inspect_gui --test-make_qres_api

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> import guitool
        >>> from ibeis.gui import inspect_gui
        >>> cm_list, qreq_ = ibeis.main_helpers.testdata_cmlist()
        >>> tblname = 'chipmatch'
        >>> name_scoring = False
        >>> ranks_lt = 5
        >>> qres_api = make_qres_api(qreq_.ibs, cm_list, ranks_lt, name_scoring, qreq_=qreq_)
        >>> print('qres_api = %r' % (qres_api,))
    """
    if ut.VERBOSE:
        print('[inspect] make_qres_api')
    #ibs.cfg.other_cfg.ranks_lt = 2
    # Overwrite
    #ranks_lt_ = ibs.cfg.other_cfg.ensure_attr('ranks_lt', 2)
    #filter_reviewed = ibs.cfg.other_cfg.ensure_attr('filter_reviewed', True)
    #if filter_reviewed is None:
    #    # only filter big queries if not specified
    #    filter_reviewed = len(cm_list) > 6
    #ranks_lt = ranks_lt if ranks_lt is not None else ranks_lt_

    candidate_matches = get_automatch_candidates(
        cm_list, ranks_lt=ranks_lt, name_scoring=name_scoring, ibs=ibs,
        directed=False, filter_reviewed=filter_reviewed,
        filter_duplicate_namepair_matches=filter_duplicate_namepair_matches
    )
    # Get extra info
    (qaids, daids, scores, ranks) = candidate_matches

    RES_THUMB_TEXT = 'ResThumb'
    MATCH_THUMB_TEXT = 'MatchThumb'

    col_name_list = [
        'result_index',
        'score',
        REVIEWED_STATUS_TEXT,
        MATCHED_STATUS_TEXT,
        'querythumb',
        RES_THUMB_TEXT,
        'qname',
        'name',
        'rank',
        'qaid',
        'aid',
    ]

    col_types_dict = dict([
        ('qaid',       int),
        ('aid',        int),
        #('d_nGt',      int),
        #('q_nGt',      int),
        #('review',     'BUTTON'),
        (MATCHED_STATUS_TEXT, str),
        (REVIEWED_STATUS_TEXT, str),
        ('querythumb', 'PIXMAP'),
        (RES_THUMB_TEXT,   'PIXMAP'),
        ('qname',      str),
        ('name',       str),
        ('score',      float),
        ('rank',       int),
        ('truth',      bool),
        ('opt',        int),
        ('result_index',  int),
    ])

    col_getter_dict = dict([
        ('qaid',       np.array(qaids)),
        ('aid',        np.array(daids)),
        #('d_nGt',      ibs.get_annot_num_groundtruth),
        #('q_nGt',      ibs.get_annot_num_groundtruth),
        #('review',     lambda rowid: get_buttontup),
        (MATCHED_STATUS_TEXT,  partial(get_match_status, ibs)),
        (REVIEWED_STATUS_TEXT,  partial(get_reviewed_status, ibs)),
        ('querythumb', ibs.get_annot_chip_thumbtup),
        (RES_THUMB_TEXT,   ibs.get_annot_chip_thumbtup),
        ('qname',      ibs.get_annot_names),
        ('name',       ibs.get_annot_names),
        ('score',      np.array(scores)),
        ('rank',       np.array(ranks)),
        ('result_index',       np.arange(len(ranks))),
        #('truth',     truths),
        #('opt',       opts),
    ])

    # default is 100
    col_width_dict = {
        'score': 75,
        REVIEWED_STATUS_TEXT: 75,
        MATCHED_STATUS_TEXT: 75,
        'rank': 42,
        'qaid': 42,
        'aid': 42,
        'result_index': 42,
        'qname': 60,
        'name': 60,
    }

    USE_MATCH_THUMBS = True
    if USE_MATCH_THUMBS:

        def get_match_thumbtup(ibs, qaid2_cm, qaids, daids, index, qreq_=None,
                               thumbsize=(128, 128), match_thumbtup_cache={}):
            daid = daids[index]
            qaid = qaids[index]
            cm = qaid2_cm[qaid]
            assert cm.qaid == qaid, 'aids do not aggree'
            #cm = cm_list[qaid]
            fpath = ensure_match_img(ibs, cm, daid, qreq_=qreq_,
                                     match_thumbtup_cache=match_thumbtup_cache)
            if isinstance(thumbsize, int):
                thumbsize = (thumbsize, thumbsize)
            thumbtup = (ut.augpath(fpath, 'thumb_%d,%d' % thumbsize), fpath, thumbsize,
                        [], [])
            return thumbtup

        col_name_list.insert(col_name_list.index(RES_THUMB_TEXT) + 1,
                             MATCH_THUMB_TEXT)
        col_types_dict[MATCH_THUMB_TEXT] = 'PIXMAP'
        qaid2_cm = {cm.qaid: cm for cm in cm_list}
        get_match_thumbtup_ = partial(get_match_thumbtup, ibs, qaid2_cm,
                                      qaids, daids, qreq_=qreq_,
                                      match_thumbtup_cache={})
        col_getter_dict[MATCH_THUMB_TEXT] = get_match_thumbtup_

    col_bgrole_dict = {
        MATCHED_STATUS_TEXT : partial(get_match_status_bgrole, ibs),
        REVIEWED_STATUS_TEXT: partial(get_reviewed_status_bgrole, ibs),
    }
    # TODO: remove ider dict.
    # it is massively unuseful
    col_ider_dict = {
        MATCHED_STATUS_TEXT     : ('qaid', 'aid'),
        REVIEWED_STATUS_TEXT    : ('qaid', 'aid'),
        #'d_nGt'      : ('aid'),
        #'q_nGt'      : ('qaid'),
        'querythumb' : ('qaid'),
        'ResThumb'   : ('aid'),
        'qname'      : ('qaid'),
        'name'       : ('aid'),
    }
    col_setter_dict = {
        'qname': ibs.set_annot_names,
        'name': ibs.set_annot_names
    }
    editable_colnames =  ['truth', 'notes', 'qname', 'name', 'opt']

    USE_BOOLS = False
    if USE_BOOLS:
        # DEPRICATED use tag funcs
        boolean_annotmatch_columns = [
            'is_hard',
            'is_nondistinct',
            'is_scenerymatch',
            'is_photobomb',
        ]

        def make_annotmatch_boolean_getter_wrapper(ibs, colname):
            colname_getter = getattr(ibs, 'get_annotmatch_' + colname)
            def getter_wrapper(aidpair):
                qaid, daid = aidpair
                annotmatch_rowid_list = ibs.add_annotmatch([qaid], [daid])
                value_list = colname_getter(annotmatch_rowid_list)
                value = value_list[0]
                return value if value is not None else False
            ut.set_funcname(getter_wrapper, 'getter_wrapper_' + colname)
            return getter_wrapper

        def make_annotmatch_boolean_setter_wrapper(ibs, colname):
            colname_setter = getattr(ibs, 'set_annotmatch_' + colname)
            def setter_wrapper(aidpair, value):
                qaid, daid = aidpair
                annotmatch_rowid_list = ibs.add_annotmatch([qaid], [daid])
                value_list = [value]
                return colname_setter(annotmatch_rowid_list, value_list)
            ut.set_funcname(setter_wrapper, 'setter_wrapper_' + colname)
            return setter_wrapper

        for colname in boolean_annotmatch_columns:
            #annotmatch_rowid_list = ibs.add_annotmatch(qaids, daids)
            #col_name_list.append(colname)
            col_name_list.insert(col_name_list.index('qname'), colname)
            #rank
            #col_ider_dict[colname] = annotmatch_rowid_list
            col_ider_dict[colname] = ('qaid', 'aid')
            col_types_dict[colname] = bool
            col_getter_dict[colname] = make_annotmatch_boolean_getter_wrapper(
                ibs, colname)
            col_setter_dict[colname] = make_annotmatch_boolean_setter_wrapper(
                ibs, colname)
            col_width_dict[colname] = 70
            editable_colnames.append(colname)

    sortby = 'score'

    def get_thumb_size():
        return ibs.cfg.other_cfg.thumb_size

    # Insert info into dict
    qres_api = guitool.CustomAPI(col_name_list, col_types_dict,
                                 col_getter_dict, col_bgrole_dict,
                                 col_ider_dict, col_setter_dict,
                                 editable_colnames, sortby, get_thumb_size,
                                 True, col_width_dict)
    return qres_api


def launch_review_matches_interface(ibs, cm_list, dodraw=False, filter_reviewed=False):
    """ TODO: move to a more general function """
    from ibeis.gui import inspect_gui
    import guitool
    guitool.ensure_qapp()
    #backend_callback = back.front.update_tables
    backend_callback = None
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, cm_list,
                                              callback=backend_callback,
                                              filter_reviewed=filter_reviewed)
    if dodraw:
        qres_wgt.show()
        qres_wgt.raise_()
    return qres_wgt


@profile
def get_automatch_candidates(cm_list, ranks_lt=5, directed=True,
                             name_scoring=False, ibs=None, filter_reviewed=False,
                             filter_duplicate_namepair_matches=False):
    """
    Needs to be moved to a better file. Maybe something to do with
    identification.

    Returns a list of matches that should be inspected
    This function is more lightweight than orgres or allres.
    Used in inspect_gui and interact_qres2

    Args:
        cm_list (list): list of chip match objects
        ranks_lt (int): put all ranks less than this number into the graph
        directed (bool):

    Returns:
        tuple: candidate_matches = (qaid_arr, daid_arr, score_arr, rank_arr)

    CommandLine:
        python -m ibeis.expt.results_organizer --test-get_automatch_candidates:2
        python -m ibeis.expt.results_organizer --test-get_automatch_candidates:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qreq_ = ibeis.main_helpers.testdata_qreq_()
        >>> cm_list = ibs.query_chips(qreq_=qreq_, return_cm=True)
        >>> ranks_lt = 5
        >>> directed = True
        >>> name_scoring = False
        >>> candidate_matches = get_automatch_candidates(cm_list, ranks_lt, directed, ibs=ibs)
        >>> print(candidate_matches)

    Example1:
        >>> # UNSTABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:5]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> cm_list = ibs.query_chips(qaid_list, daid_list, return_cm=True)
        >>> ranks_lt = 5
        >>> directed = False
        >>> name_scoring = False
        >>> filter_reviewed = False
        >>> filter_duplicate_namepair_matches = True
        >>> candidate_matches = get_automatch_candidates(
        ...    cm_list, ranks_lt, directed, name_scoring=name_scoring,
        ...    filter_reviewed=filter_reviewed,
        ...    filter_duplicate_namepair_matches=filter_duplicate_namepair_matches,
        ...    ibs=ibs)
        >>> print(candidate_matches)

    Example3:
        >>> # UNSTABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:1]
        >>> daid_list = ibs.get_valid_aids()[10:100]
        >>> qaid2_cm = ibs.query_chips(qaid_list, daid_list, return_cm=True)
        >>> ranks_lt = 1
        >>> directed = False
        >>> name_scoring = False
        >>> filter_reviewed = False
        >>> filter_duplicate_namepair_matches = True
        >>> candidate_matches = get_automatch_candidates(
        ...    cm_list, ranks_lt, directed, name_scoring=name_scoring,
        ...    filter_reviewed=filter_reviewed,
        ...    filter_duplicate_namepair_matches=filter_duplicate_namepair_matches,
        ...    ibs=ibs)
        >>> print(candidate_matches)

    Example4:
        >>> # UNSTABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:10]
        >>> daid_list = ibs.get_valid_aids()[0:10]
        >>> qres_list = ibs.query_chips(qaid_list, daid_list)
        >>> ranks_lt = 3
        >>> directed = False
        >>> name_scoring = False
        >>> filter_reviewed = False
        >>> filter_duplicate_namepair_matches = True
        >>> candidate_matches = get_automatch_candidates(
        ...    qaid2_cm, ranks_lt, directed, name_scoring=name_scoring,
        ...    filter_reviewed=filter_reviewed,
        ...    filter_duplicate_namepair_matches=filter_duplicate_namepair_matches,
        ...    ibs=ibs)
        >>> print(candidate_matches)
    """
    import vtool as vt
    from ibeis.algo.hots import chip_match
    print(('[resorg] get_automatch_candidates('
           'filter_reviewed={filter_reviewed},'
           'filter_duplicate_namepair_matches={filter_duplicate_namepair_matches},'
           'directed={directed},'
           'ranks_lt={ranks_lt},'
           ).format(**locals()))
    print('[resorg] len(cm_list) = %d' % (len(cm_list)))
    qaids_stack  = []
    daids_stack  = []
    ranks_stack  = []
    scores_stack = []

    # For each QueryResult, Extract inspectable candidate matches
    if isinstance(cm_list, dict):
        cm_list = list(cm_list.values())

    for cm in cm_list:
        if isinstance(cm, chip_match.ChipMatch):
            daids  = cm.get_top_aids(ntop=ranks_lt)
            scores = cm.get_top_scores(ntop=ranks_lt)
            ranks  = np.arange(len(daids))
            qaids  = np.full(daids.shape, cm.qaid, dtype=daids.dtype)
        else:
            (qaids, daids, scores, ranks) = cm.get_match_tbldata(
                ranks_lt=ranks_lt, name_scoring=name_scoring, ibs=ibs)
        qaids_stack.append(qaids)
        daids_stack.append(daids)
        scores_stack.append(scores)
        ranks_stack.append(ranks)

    # Stack them into a giant array
    # utool.embed()
    qaid_arr  = np.hstack(qaids_stack)
    daid_arr  = np.hstack(daids_stack)
    score_arr = np.hstack(scores_stack)
    rank_arr  = np.hstack(ranks_stack)

    # Sort by scores
    sortx = score_arr.argsort()[::-1]
    qaid_arr  = qaid_arr[sortx]
    daid_arr   = daid_arr[sortx]
    score_arr = score_arr[sortx]
    rank_arr  = rank_arr[sortx]

    if filter_reviewed:
        _is_reviewed = ibs.get_annot_pair_is_reviewed(qaid_arr.tolist(), daid_arr.tolist())
        is_unreviewed = ~np.array(_is_reviewed, dtype=np.bool)
        qaid_arr  = qaid_arr.compress(is_unreviewed)
        daid_arr   = daid_arr.compress(is_unreviewed)
        score_arr = score_arr.compress(is_unreviewed)
        rank_arr  = rank_arr.compress(is_unreviewed)

    # Remove directed edges
    if not directed:
        #nodes = np.unique(directed_edges.flatten())
        directed_edges = np.vstack((qaid_arr, daid_arr)).T
        #idx1, idx2 = vt.intersect2d_indices(directed_edges, directed_edges[:, ::-1])

        unique_rowx = vt.find_best_undirected_edge_indexes(directed_edges, score_arr)

        qaid_arr  = qaid_arr.take(unique_rowx)
        daid_arr  = daid_arr.take(unique_rowx)
        score_arr = score_arr.take(unique_rowx)
        rank_arr  = rank_arr.take(unique_rowx)

    # Filter Double Name Matches
    if filter_duplicate_namepair_matches:
        qnid_arr = ibs.get_annot_nids(qaid_arr)
        dnid_arr = ibs.get_annot_nids(daid_arr)
        if not directed:
            directed_name_edges = np.vstack((qnid_arr, dnid_arr)).T
            unique_rowx2 = vt.find_best_undirected_edge_indexes(directed_name_edges, score_arr)
        else:
            namepair_id_list = np.array(vt.compute_unique_data_ids_(list(zip(qnid_arr, dnid_arr))))
            unique_namepair_ids, namepair_groupxs = vt.group_indices(namepair_id_list)
            score_namepair_groups = vt.apply_grouping(score_arr, namepair_groupxs)
            unique_rowx2 = np.array(sorted([
                groupx[score_group.argmax()]
                for groupx, score_group in zip(namepair_groupxs, score_namepair_groups)
            ]), dtype=np.int32)
        qaid_arr  = qaid_arr.take(unique_rowx2)
        daid_arr  = daid_arr.take(unique_rowx2)
        score_arr = score_arr.take(unique_rowx2)
        rank_arr  = rank_arr.take(unique_rowx2)

    candidate_matches = (qaid_arr, daid_arr, score_arr, rank_arr)
    return candidate_matches


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show

        python -m ibeis.gui.inspect_gui
        python -m ibeis.gui.inspect_gui --allexamples
        python -m ibeis.gui.inspect_gui --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
