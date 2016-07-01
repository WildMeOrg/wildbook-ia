# -*- coding: utf-8 -*-
"""
This module was never really finished. It is used in some cases
to display the results from a query in a qt window.

TODO:
    Refresh name table on inspect gui close

CommandLine:
    python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show
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
#from ibeis import constants as const
import utool as ut
from ibeis.gui import guiexcept
(print, rrr, profile) = ut.inject2(__name__, '[inspect_gui]')


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
        >>> cm_list = qreq_.execute()
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
            ('Mark as &Reviewed',
             lambda: ibs.set_annot_pair_as_reviewed(aid1, aid2)),
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
            matches, metadata = vsone_pipeline.vsone_single(qaid, daid, qreq2_,
                                                            use_ibscache=True)
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
        annotmatch_rowid = ibs.get_annotmatch_rowid_from_undirected_superkey(
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
            am_rowid = ibs.add_annotmatch_undirected([aid1], [aid2])[0]
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
        assert qreq_ is not None, 'must specify qreq_'

        if ut.VERBOSE:
            print('[qres_wgt] Init QueryResultsWidget')
        # Uncomment below to turn on FilterProxyModel
        if USE_FILTER_PROXY:
            APIItemWidget.__init__(qres_wgt, parent=parent,
                                    model_class=CustomFilterModel)
        else:
            APIItemWidget.__init__(qres_wgt, parent=parent)

        qres_wgt.OLD_CLICK_BEHAVIOR = False

        #qres_wgt.altkey_shortcut =
        #QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.ALT), qres_wgt,
        #                qres_wgt.on_alt_pressed,
        #                context=QtCore..Qt.WidgetShortcut)
        qres_wgt.button_list = None
        qres_wgt.show_new = True
        qres_wgt.show_join = True
        qres_wgt.show_split = True
        qres_wgt.tt = ut.tic()
        # Set results data
        if USE_FILTER_PROXY:
            qres_wgt.add_checkboxes(qres_wgt.show_new, qres_wgt.show_join,
                                    qres_wgt.show_split)

        lbl = QtGui.QLabel('\'T\' marks as correct match. \'F\' marks as incorrect match. Alt brings up context menu. Double click a row to inspect matches.')
        from guitool.__PYQT__.QtCore import Qt
        qres_wgt.layout().setSpacing(0)
        bottom_bar = guitool.newFrame(qres_wgt, orientation=Qt.Horizontal)
        bottom_bar.layout().setSpacing(0)
        #import utool
        #utool.embed()
        qres_wgt.layout().addWidget(bottom_bar)
        bottom_bar.addWidget(lbl)
        bottom_bar.addNewButton('Mark all above as correct', clicked=qres_wgt.mark_unreviewed_above_score_as_correct)

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

        dbdir = qres_wgt.qreq_.ibs.get_dbdir()
        expt_dir = ut.ensuredir(ut.unixjoin(dbdir, 'SPECIAL_GGR_EXPT_LOGS'))
        review_log_dir = ut.ensuredir(ut.unixjoin(expt_dir, 'review_logs'))

        ts = ut.get_timestamp(isutc=True, timezone=True)
        log_fpath = ut.unixjoin(review_log_dir, 'review_log_%s_%s.json' % (qres_wgt.qreq_.ibs.dbname, ts))

        # LOG ALL CHANGES MADE TO NAMES
        import logging
        # create logger with 'spam_application'
        logger = logging.getLogger('query_review')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_fpath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        qres_wgt.logger = logger
        logger.info('START QUERY_RESULT_REVIEW')
        logger.info('NUM CHIP_MATCH OBJECTS (len(cm_list)=%d)' % (len(cm_list),))
        logger.info('NUM PAIRS TO REVIEW (nRows=%d)' % (qres_wgt.qres_api.nRows,))
        logger.info('PARENT QUERY REQUEST (cfgstr=%s)' % (qres_wgt.qreq_.get_cfgstr(with_input=True),))

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

    def sizeHint(qres_wgt):
        # should eventually improve this to use the widths of the header columns
        return QtCore.QSize(1100, 500)

    def connect_signals_and_slots(qres_wgt):
        qres_wgt.view.doubleClicked.connect(qres_wgt._on_doubleclick)
        qres_wgt.view.pressed.connect(qres_wgt._on_pressed)

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
        if not qres_wgt.OLD_CLICK_BEHAVIOR or colname != MATCHED_STATUS_TEXT:
            return qres_wgt.show_match_at_qtindex(qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(qres_wgt, qtindex):
        print('[qres_wgt] _on_pressed: ')
        def _check_for_double_click(qres_wgt, qtindex):
            threshold = 0.20  # seconds
            distance = ut.toc(qres_wgt.tt)
            if distance <= threshold:
                qres_wgt._on_doubleclick(qtindex)
            qres_wgt.tt = ut.tic()
        _check_for_double_click(qres_wgt, qtindex)

    def selectedRows(qres_wgt):
        selected_qtindex_list = qres_wgt.view.selectedIndexes()
        selected_qtindex_list2 = []
        seen_ = set([])
        for qindex in selected_qtindex_list:
            row = qindex.row()
            if row not in seen_:
                selected_qtindex_list2.append(qindex)
                seen_.add(row)
        return selected_qtindex_list2

    #@guitool.slot_()
    def on_alt_pressed(qres_wgt, view, event):
        selected_qtindex_list = qres_wgt.selectedRows()
        for qindex in selected_qtindex_list:
            pass
        if len(selected_qtindex_list) == 1:
            # popup context menu on alt
            qtindex = selected_qtindex_list[0]
            qrect = view.visualRect(qtindex)
            pos = qrect.center()
            qres_wgt.on_contextMenuRequested(qtindex, pos)
        else:
            print('[alt] Multiple %d selection' % (len(selected_qtindex_list),))

    #@guitool.slot_()
    def on_special_key_pressed(qres_wgt, view, event):
        #selected_qtindex_list = view.selectedIndexes()
        selected_qtindex_list = qres_wgt.selectedRows()

        if len(selected_qtindex_list) == 1:
            print('event = %r ' % (event,))
            print('event.key() = %r ' % (event.key(),))
            qtindex = selected_qtindex_list[0]
            ibs = qres_wgt.ibs
            aid1, aid2 = qres_wgt.get_aidpair_from_qtindex(qtindex)
            _tup = qres_wgt.get_widget_review_vars(aid1)
            ibs, cm, qreq_, update_callback, backend_callback = _tup

            options = get_aidpair_context_menu_options(
                ibs, aid1, aid2, cm, qreq_=qreq_,
                logger=qres_wgt.logger, update_callback=update_callback,
                backend_callback=backend_callback)

            option_dict = {key[key.find('&') + 1]: val for key, val in options
                           if '&' in key}
            #print('option_dict = %s' % (ut.repr3(option_dict, nl=2),))

            event_key = event.key()
            if event_key == QtCore.Qt.Key_R:
                # ibs.set_annot_pair_as_reviewed
                option_dict['R']()
            elif event_key == QtCore.Qt.Key_T:
                # Calls set_annot_pair_as_positive_match_
                option_dict['T']()
            elif event_key == QtCore.Qt.Key_F:
                # set_annot_pair_as_negative_match_
                option_dict['F']()
            ## BROKEN FOR NOW
            #elif event_key == QtCore.Qt.Key_S:
            #    option_dict['S']()
            #elif event_key == QtCore.Qt.Key_P:
            #    option_dict['P']()
            print('emiting data changed')
            # This may not work with PyQt5
            # http://stackoverflow.com/questions/22560296/view-not-resp-datachanged
            model = qtindex.model()
            # This should work by itself
            model.dataChanged.emit(qtindex, qtindex)
            # but it doesnt seem to be, but this seems to solve the issue
            model.layoutChanged.emit()
            print('emited data changed')
            #model.select()
        else:
            print('[key] Multiple %d selection' % (len(selected_qtindex_list),))

    @guitool.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(qres_wgt, qtindex, qpoint):
        """
        popup context menu
        """
        #selected_qtindex_list = qres_wgt.view.selectedIndexes()
        selected_qtindex_list = qres_wgt.selectedRows()
        if len(selected_qtindex_list) == 1:
            qwin = qres_wgt
            aid1, aid2 = qres_wgt.get_aidpair_from_qtindex(qtindex)
            tup = qres_wgt.get_widget_review_vars(aid1)
            ibs, cm, qreq_, update_callback, backend_callback = tup
            options = get_aidpair_context_menu_options(
                ibs, aid1, aid2, cm, qreq_=qreq_, logger=qres_wgt.logger,
                update_callback=update_callback,
                backend_callback=backend_callback)
            guitool.popup_menu(qwin, qpoint, options)
        else:
            print('[context] Multiple %d selection' % (len(selected_qtindex_list),))

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
        qaid, daid = qres_wgt.get_aidpair_from_qtindex(qtindex)
        ibs = qres_wgt.ibs
        annotmatch_rowid_list = ibs.add_annotmatch_undirected([qaid], [daid])
        return annotmatch_rowid_list

    def show_match_at_qtindex(qres_wgt, qtindex):
        print('interact')
        qaid, daid = qres_wgt.get_aidpair_from_qtindex(qtindex)
        cm = qres_wgt.qaid2_cm[qaid]
        match_interaction = cm.ishow_single_annotmatch(
            qres_wgt.qreq_, daid, mode=0)
        fig = match_interaction.fig
        fig_presenter.bring_to_front(fig)

    def mark_unreviewed_above_score_as_correct(qres_wgt):
        selected_qtindex_list = qres_wgt.selectedRows()
        if len(selected_qtindex_list) == 1:
            qtindex = selected_qtindex_list[0]
            #aid1, aid2 = qres_wgt.get_aidpair_from_qtindex(qtindex)
            thresh  = qtindex.model().get_header_data('score', qtindex)
            print('thresh = %r' % (thresh,))

            rows = qres_wgt.qres_api.ider()
            scores_ = qres_wgt.qres_api.get(qres_wgt.qres_api.col_name_list.index('score'), rows)
            valid_rows = ut.compress(rows, scores_ >= thresh)
            aids1 = qres_wgt.qres_api.get(qres_wgt.qres_api.col_name_list.index('qaid'), valid_rows)
            aids2 = qres_wgt.qres_api.get(qres_wgt.qres_api.col_name_list.index('aid'), valid_rows)
            #ibs = qres_wgt.ibs
            ibs = qres_wgt.ibs
            am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
            reviewed = ibs.get_annotmatch_reviewed(am_rowids)
            unreviewed = ut.not_list(reviewed)

            valid_rows = ut.compress(valid_rows, unreviewed)
            aids1 = ut.compress(aids1, unreviewed)
            aids2 = ut.compress(aids2, unreviewed)

            import networkx as nx
            graph = nx.Graph()
            graph.add_edges_from(list(zip(aids1, aids2)), {'user_thresh_match': True})
            review_groups = list(nx.connected_component_subgraphs(graph))

            changing_aids = list(graph.nodes())
            nids = ibs.get_annot_nids(changing_aids)
            nid2_aids = ut.group_items(changing_aids, nids)
            for nid, aids in nid2_aids.items():
                # Connect all original names in the database to denote merges
                for u, v in ut.itertwo(aids):
                    graph.add_edge(u, v)
            dbside_groups = list(nx.connected_component_subgraphs(graph))

            options = [
                'Accept',
                #'Review More'
            ]
            msg = ut.codeblock(
                '''
                There are %d names and %d annotations in this mass review set.
                Mass review has discovered %d internal groups.
                Accepting will induce a database grouping of %d names.
                ''') % (len(nid2_aids), len(changing_aids), len(review_groups), len(dbside_groups))

            reply = guitool.user_option(msg=msg, options=options)

            if reply == options[0]:
                # This is not the smartest way to group names.
                # Ideally what will happen here, is that reviewed edges will go into
                # the new graph name inference algorithm.
                # then the chosen point will be used as the threshold. Then
                # the graph cut algorithm will be applied.
                logger = qres_wgt.logger
                logger.debug(msg)
                logger.info('START MASS_THRESHOLD_MERGE')
                logger.info('num_groups=%d thresh=%r' % (
                    len(dbside_groups), thresh,))
                for count, subgraph in enumerate(dbside_groups):
                    thresh_aid_pairs = [
                        edge for edge, flag in
                        nx.get_edge_attributes(graph, 'user_thresh_match').items()
                        if flag]
                    thresh_uuid_pairs = ibs.unflat_map(ibs.get_annot_uuids, thresh_aid_pairs)
                    aids = list(subgraph.nodes())
                    nids = ibs.get_annot_name_rowids(aids)
                    flags = ut.not_list(ibs.is_aid_unknown(aids))
                    previous_names = ibs.get_name_texts(nids)
                    valid_nids = ut.compress(nids, flags)
                    if len(valid_nids) == 0:
                        merge_nid = ibs.make_next_nids(num=1)[0]
                        type_ = 'new'
                    else:
                        merge_nid = min(valid_nids)
                        type_ = 'existing'

                    # Need to find other non-exemplar / query names that may
                    # need merging
                    other_aids = ibs.get_name_aids(valid_nids)
                    other_aids = set(ut.flatten(other_aids)) - set(aids)
                    other_auuids = ibs.get_annot_uuids(other_aids)
                    other_previous_names = ibs.get_annot_names(other_aids)

                    merge_name = ibs.get_name_texts(merge_nid)
                    annot_uuids = ibs.get_annot_uuids(aids)
                    ###
                    # Set as reviewed (so we dont see them again), but mark it
                    # with a different code to denote that it was a MASS review
                    aid1_list = ut.take_column(thresh_aid_pairs, 0)
                    aid2_list = ut.take_column(thresh_aid_pairs, 1)
                    am_rowids = ibs.add_annotmatch_undirected(aid1_list, aid2_list)
                    MASS_REVIEW_CODE = 2
                    ibs.set_annotmatch_reviewed(am_rowids, [MASS_REVIEW_CODE] * len(am_rowids))

                    logger.info('START GROUP %d' % (count,))
                    logger.info('GROUP BASED ON %d ANNOT_PAIRS WITH SCORE ABOVE (thresh=%r)' % (len(thresh_uuid_pairs), thresh,))
                    logger.debug('(uuid_pairs=%r)' % (thresh_uuid_pairs))
                    logger.debug('(merge_name=%r)' % (merge_name))
                    logger.debug('CHANGE NAME OF %d (annot_uuids=%r) WITH (previous_names=%r) TO (%s) (merge_name=%r)' % (
                        len(annot_uuids), annot_uuids, previous_names, type_, merge_name))
                    logger.debug('ADDITIONAL CHANGE NAME OF %d (annot_uuids=%r) WITH (previous_names=%r) TO (%s) (merge_name=%r)' % (
                        len(other_auuids), other_auuids, other_previous_names, type_, merge_name))
                    logger.info('END GROUP %d' % (count,))
                    new_nids = [merge_nid] * len(aids)
                    ibs.set_annot_name_rowids(aids, new_nids)
                logger.info('END MASS_THRESHOLD_MERGE')

            #ibs.get_annotmatch_truth(am_rowids)
        else:
            print('[context] Multiple %d selection' % (len(selected_qtindex_list),))

# ______


def set_annot_pair_as_positive_match_(ibs, aid1, aid2, cm, qreq_, **kwargs):
    """
    MARK AS CORRECT
    """
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
            aid1, aid2, on_nontrivial_merge=on_nontrivial_merge,
            logger=kwargs.get('logger', None))
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match(ibs, aid1, aid2, qreq_=qreq_, cm=cm, **kwargs)
    except guiexcept.UserCancel:
        print('user canceled positive match')


def set_annot_pair_as_negative_match_(ibs, aid1, aid2, cm, qreq_, **kwargs):
    """
    MARK AS INCORRECT
    """
    def on_nontrivial_split(ibs, aid1, aid2):
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        print('There are %d annots in this name. Need more sophisticated split'
              % (len(aid1_groundtruth)))
        raise guiexcept.NeedsUserInput('non-trivial split')
    try:
        status = ibs.set_annot_pair_as_negative_match(
            aid1, aid2, on_nontrivial_split=on_nontrivial_split, logger=kwargs.get('logger', None)
        )
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        options = [
            'Flag for later',
            'Review now',
        ]
        reply = guitool.user_option(
            msg=ut.codeblock(
                '''
                Marking this as False induces a split case.
                Choose how to handle this.
                '''),
            options=options)
        if reply == options[0]:
            prop = 'SplitCase'
            if 'logger' in kwargs:
                log = kwargs['logger'].info
            else:
                log = print
            annot_uuid_pair = ibs.get_annot_uuids((aid1, aid2))
            log('FLAG SplitCase: (annot_uuid_pair=%r)' % annot_uuid_pair)
            am_rowid = ibs.add_annotmatch_undirected([aid1], [aid2])[0]
            ibs.set_annotmatch_prop(prop, [am_rowid], [True])
        elif reply == options[1]:
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
    assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
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
    assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
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
    if annotmach_reviewed == 0:
        lighten_amount = .9
    elif annotmach_reviewed == 2:
        lighten_amount = .7
    else:
        lighten_amount = .35
    truth_color = vh.get_truth_color(truth, base255=True,
                                     lighten_amount=lighten_amount)
    #truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    #truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


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
        #with ut.Timer('render-1'):
        fpath = cm.imwrite_single_annotmatch(
            qreq_, daid, fpath=match_thumb_fpath_, saveax=True, fnum=32,
            notitle=True, verbose=False)
        #with ut.Timer('render-2'):
        #    img = cm.render_single_annotmatch(qreq_, daid, fnum=32, notitle=True, dpi=30)
        #    cv2.imwrite(match_thumb_fpath_, img)
        #    fpath = match_thumb_fpath_
        #with ut.Timer('render-3'):
        #fpath = match_thumb_fpath_
        #render_config = {
        #    'dpi'              : 60,
        #    'draw_fmatches'    : True,
        #    #'vert'             : view_orientation == 'vertical',
        #    'show_aidstr'      : False,
        #    'show_name'        : False,
        #    'show_exemplar'    : False,
        #    'show_num_gt'      : False,
        #    'show_timedelta'   : False,
        #    'show_name_rank'   : False,
        #    'show_score'       : False,
        #    'show_annot_score' : False,
        #    'show_name_score'  : False,
        #    'draw_lbl'         : False,
        #    'draw_border'      : False,
        #}
        #cm.imwrite_single_annotmatch2(qreq_, daid, fpath, fnum=32, notitle=True, **render_config)
        #print('fpath = %r' % (fpath,))
        match_thumbtup_cache[match_thumb_fpath_] = fpath
    return fpath


def make_ensure_match_img_nosql_func(qreq_, cm, daid):
    r"""
    CommandLine:
        python -m ibeis.gui.inspect_gui --test-ensure_match_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> cm, qreq_ = ibeis.testdata_cm()
        >>> ibs = qreq_.ibs
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
    #import ibeis.viz
    from ibeis.viz import viz_matches
    import cv2
    import io
    import plottool as pt
    import vtool as vt
    import matplotlib as mpl
    aid1 = cm.qaid
    aid2 = daid

    ibs = qreq_.ibs
    resize_factor = .5

    match_thumbdir = ibs.get_match_thumbdir()
    match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
    fpath = ut.unixjoin(match_thumbdir, match_thumb_fname)

    def main_thread_load():
        # This gets executed in the main thread and collects data
        # from sql
        rchip1_fpath, rchip2_fpath, kpts1, kpts2 = viz_matches._get_annot_pair_info(
            ibs, aid1, aid2, qreq_, draw_fmatches=True, as_fpath=True)
        return rchip1_fpath, rchip2_fpath, kpts1, kpts2

    def nosql_draw(check_func, rchip1_fpath, rchip2_fpath, kpts1, kpts2):
        # This gets executed in the child thread and does drawing async style
        #from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
        #from matplotlib.backends.backend_pdf import Figure
        #from matplotlib.backends.backend_svg import FigureCanvas
        #from matplotlib.backends.backend_svg import Figure
        from matplotlib.backends.backend_agg import FigureCanvas
        from matplotlib.backends.backend_agg import Figure

        kpts1_ = vt.offset_kpts(kpts1, (0, 0), (resize_factor, resize_factor))
        kpts2_ = vt.offset_kpts(kpts2, (0, 0), (resize_factor, resize_factor))

        #from matplotlib.figure import Figure
        if check_func is not None and check_func():
            return

        rchip1 = vt.imread(rchip1_fpath)
        rchip1 = vt.resize_image_by_scale(rchip1, resize_factor)
        if check_func is not None and check_func():
            return
        rchip2 = vt.imread(rchip2_fpath)
        rchip2 = vt.resize_image_by_scale(rchip2, resize_factor)
        if check_func is not None and check_func():
            return

        idx = cm.daid2_idx[daid]
        fm   = cm.fm_list[idx]
        fsv  = None if cm.fsv_list is None else cm.fsv_list[idx]
        fs   = None if fsv is None else fsv.prod(axis=1)

        maxnum = 200
        if len(fs) > maxnum:
            # HACK TO ONLY SHOW TOP MATCHES
            sortx = fs.argsort()[::-1]
            fm = fm.take(sortx[:maxnum], axis=0)
            fs = fs.take(sortx[:maxnum], axis=0)

        was_interactive = mpl.is_interactive()
        if was_interactive:
            mpl.interactive(False)
        #fnum = 32
        fig = Figure()
        canvas = FigureCanvas(fig)  # NOQA
        #fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        if check_func is not None and check_func():
            return
        #fig = pt.plt.figure(fnum)
        #H1 = np.eye(3)
        #H2 = np.eye(3)
        #H1[0, 0] = .5
        #H1[1, 1] = .5
        #H2[0, 0] = .5
        #H2[1, 1] = .5
        ax, xywh1, xywh2 = pt.show_chipmatch2(rchip1, rchip2, kpts1_, kpts2_, fm,
                                              fs=fs, colorbar_=False, ax=ax)
        if check_func is not None and check_func():
            return
        savekw = {
            'dpi' : 60,
        }
        axes_extents = pt.extract_axes_extents(fig)
        #assert len(axes_extents) == 1, 'more than one axes'
        extent = axes_extents[0]
        with io.BytesIO() as stream:
            # This call takes 23% - 15% of the time depending on settings
            fig.savefig(stream, bbox_inches=extent, **savekw)
            stream.seek(0)
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        if check_func is not None and check_func():
            return
        pt.plt.close(fig)
        image = cv2.imdecode(data, 1)
        thumbsize = 221
        max_dsize = (thumbsize, thumbsize)
        dsize, sx, sy = vt.resized_clamped_thumb_dims(vt.get_size(image), max_dsize)
        if check_func is not None and check_func():
            return
        image = vt.resize(image, dsize)
        vt.imwrite(fpath, image)
        if check_func is not None and check_func():
            return
        #fig.savefig(fpath, bbox_inches=extent, **savekw)
    #match_thumbtup_cache[match_thumb_fpath_] = fpath
    return fpath, nosql_draw, main_thread_load


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
    # TODO: Add in timedelta to column info
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
        'qaid',
        'aid',
        'rank',
        'qname',
        'name',
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

    USE_MATCH_THUMBS = 1
    if USE_MATCH_THUMBS:

        def get_match_thumbtup(ibs, qaid2_cm, qaids, daids, index, qreq_=None,
                               thumbsize=(128, 128), match_thumbtup_cache={}):
            daid = daids[index]
            qaid = qaids[index]
            cm = qaid2_cm[qaid]
            assert cm.qaid == qaid, 'aids do not aggree'

            OLD = True
            OLD = False

            if OLD:
                fpath = ensure_match_img(ibs, cm, daid, qreq_=qreq_,
                                         match_thumbtup_cache=match_thumbtup_cache)
                if isinstance(thumbsize, int):
                    thumbsize = (thumbsize, thumbsize)
                thumbtup = (ut.augpath(fpath, 'thumb_%d,%d' % thumbsize), fpath, thumbsize,
                            [], [])
                return thumbtup
            else:
                # Hacky new way of drawing
                fpath, func, func2 = make_ensure_match_img_nosql_func(qreq_, cm, daid)
                #match_thumbdir = ibs.get_match_thumbdir()
                #match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
                #fpath = ut.unixjoin(match_thumbdir, match_thumb_fname)
                thumbdat = {
                    'fpath': fpath,
                    'thread_func': func,
                    'main_func': func2,
                    #'args': (ibs, cm, daid),
                    #'kwargs': dict(qreq_=qreq_,
                    #               match_thumbtup_cache=match_thumbtup_cache)
                }
                return thumbdat

        col_name_list.insert(col_name_list.index(RES_THUMB_TEXT) + 1,
                             MATCH_THUMB_TEXT)
        col_types_dict[MATCH_THUMB_TEXT] = 'PIXMAP'
        #col_types_dict[MATCH_THUMB_TEXT] = CustomMatchThumbDelegate
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
                annotmatch_rowid_list = ibs.add_annotmatch_undirected([qaid], [daid])
                value_list = colname_getter(annotmatch_rowid_list)
                value = value_list[0]
                return value if value is not None else False
            ut.set_funcname(getter_wrapper, 'getter_wrapper_' + colname)
            return getter_wrapper

        def make_annotmatch_boolean_setter_wrapper(ibs, colname):
            colname_setter = getattr(ibs, 'set_annotmatch_' + colname)
            def setter_wrapper(aidpair, value):
                qaid, daid = aidpair
                annotmatch_rowid_list = ibs.add_annotmatch_undirected([qaid], [daid])
                value_list = [value]
                return colname_setter(annotmatch_rowid_list, value_list)
            ut.set_funcname(setter_wrapper, 'setter_wrapper_' + colname)
            return setter_wrapper

        for colname in boolean_annotmatch_columns:
            #annotmatch_rowid_list = ibs.add_annotmatch_undirected(qaids, daids)
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
    #qres_api.candidate_matches = candidate_matches
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
                             name_scoring=False, ibs=None,
                             filter_reviewed=False,
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
        >>> cm_list = qreq_.execute()
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
        >>> cm_list = ibs.query_chips(qaid_list, daid_list)
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
        >>> qaid2_cm = ibs.query_chips(qaid_list, daid_list)
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

    if len(cm_list) == 0:
        return ([], [], [], [])

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
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> assert ibs.dbname in ['PZ_MTEST', 'testdb1'], 'do not use on a real database'
        >>> qaid_list = ibs.get_valid_aids()  #[0:5]
        >>> daid_list = ibs.get_valid_aids()  # [0:20]
        >>> if ut.get_argflag('--fresh-inspect'):
        >>>     #ut.remove_files_in_dir(ibs.get_match_thumbdir())
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
    qreq_ = ibs.new_query_request(qaid_list, daid_list, cfgdict={
        'sv_on': False, 'augment_queryside_hack': True})
    cm_list = qreq_.execute()
    tblname = ''
    name_scoring = False
    ranks_lt = 10000
    # This is where you create the result widigt
    guitool.ensure_qapp()
    print('[inspect_matches] make_qres_widget')
    #ut.view_directory(ibs.get_match_thumbdir())
    qres_wgt = inspect_gui.QueryResultsWidget(
        ibs, cm_list, ranks_lt=ranks_lt, qreq_=qreq_, filter_reviewed=False,
        filter_duplicate_namepair_matches=False)
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
