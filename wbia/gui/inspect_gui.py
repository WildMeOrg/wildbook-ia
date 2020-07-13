# -*- coding: utf-8 -*-
"""
This module was never really finished. It is used in some cases
to display the results from a query in a qt window.


CommandLine:
    python -m wbia.gui.inspect_gui --test-QueryResultsWidget --show


TODO:
    Refresh name table on inspect gui close
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from functools import partial
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import QtWidgets
from wbia.guitool.__PYQT__.QtCore import Qt
from wbia.plottool import fig_presenter
import wbia.guitool as gt

# import six
import utool as ut
from wbia.gui import id_review_api
from wbia.gui import guiexcept

(print, rrr, profile) = ut.inject2(__name__)


USE_FILTER_PROXY = False


def launch_review_matches_interface(ibs, cm_list, dodraw=False, filter_reviewed=False):
    """ TODO: move to a more general function """
    from wbia.gui import inspect_gui

    gt.ensure_qapp()
    # backend_callback = back.front.update_tables
    backend_callback = None
    review_cfg = dict(filter_reviewed=filter_reviewed)
    qres_wgt = inspect_gui.QueryResultsWidget(
        ibs, cm_list, callback=backend_callback, review_cfg=review_cfg
    )
    if dodraw:
        qres_wgt.show()
        qres_wgt.raise_()
    return qres_wgt


class CustomFilterModel(gt.FilterProxyModel):
    def __init__(model, headers=None, parent=None, *args):
        gt.FilterProxyModel.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.imgsetid = -1  # negative one is an invalid imgsetid  # seems unused
        model.original_ider = None
        model.sourcemodel = gt.APIItemModel(parent=parent)
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
        with gt.ChangeLayoutContext([model]):
            gt.FilterProxyModel._update_rows(model)


class QueryResultsWidget(gt.APIItemWidget):
    """ Window for gui inspection

    CommandLine:
        python -m wbia.gui.inspect_gui --test-QueryResultsWidget --show
        python -m wbia.gui.inspect_gui --test-QueryResultsWidget --show
        python -m wbia.gui.inspect_gui --test-QueryResultsWidget --show --fresh-inspect
        python -m wbia.gui.inspect_gui --test-QueryResultsWidget --cmd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.inspect_gui import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='PZ_MTEST', a='default:qindex=0:5,dindex=0:20', t='default:SV=False,AQH=True')
        >>> ibs = qreq_.ibs
        >>> assert qreq_.ibs.dbname in ['PZ_MTEST', 'testdb1'], 'do not use on a real database'
        >>> if ut.get_argflag('--fresh-inspect'):
        >>>     #ut.remove_files_in_dir(ibs.get_match_thumbdir())
        >>>     ibs.delete_annotmatch(ibs._get_all_annotmatch_rowids())
        >>> cm_list = qreq_.execute()
        >>> print('[inspect_matches] make_qres_widget')
        >>> review_cfg = dict(
        >>>     ranks_top=10000,
        >>>     #filter_reviewed=True,
        >>>     filter_reviewed=True,
        >>>     #filter_true_matches=False,
        >>> )
        >>> #ut.view_directory(ibs.get_match_thumbdir())
        >>> gt.ensure_qapp()
        >>> qres_wgt = QueryResultsWidget(qreq_.ibs, cm_list, qreq_=qreq_, review_cfg=review_cfg)
        >>> ut.quit_if_noshow()
        >>> qres_wgt.show()
        >>> qres_wgt.raise_()
        >>> print('</inspect_matches>')
        >>> # simulate double click
        >>> #qres_wgt._on_click(qres_wgt.model.index(2, 2))
        >>> #qres_wgt._on_doubleclick(qres_wgt.model.index(2, 0))
        >>> # TODO: add in qwin to main loop
        >>> gt.qtapp_loop(qwin=qres_wgt)
        >>> print(main_execstr)
        >>> exec(main_execstr)
    """

    def __init__(
        qres_wgt,
        ibs,
        cm_list,
        parent=None,
        callback=None,
        qreq_=None,
        query_title='',
        review_cfg={},
    ):
        if ut.VERBOSE:
            print('[qres_wgt] Init QueryResultsWidget')

        assert not isinstance(cm_list, dict)
        assert qreq_ is not None, 'must specify qreq_'

        if USE_FILTER_PROXY:
            super(QueryResultsWidget, qres_wgt).__init__(
                parent=parent, model_class=CustomFilterModel
            )
        else:
            super(QueryResultsWidget, qres_wgt).__init__(parent=parent)

        # if USE_FILTER_PROXY:
        #    APIItemWidget.__init__(qres_wgt, parent=parent,
        #                            model_class=CustomFilterModel)
        # else:
        #    APIItemWidget.__init__(qres_wgt, parent=parent)

        qres_wgt.cm_list = cm_list
        qres_wgt.ibs = ibs
        qres_wgt.qreq_ = qreq_
        qres_wgt.query_title = query_title
        qres_wgt.qaid2_cm = dict([(cm.qaid, cm) for cm in cm_list])

        qres_wgt.review_cfg = id_review_api.REVIEW_CFG_DEFAULTS.copy()
        qres_wgt.review_cfg = ut.update_existing(
            qres_wgt.review_cfg, review_cfg, assert_exists=True
        )

        # qres_wgt.altkey_shortcut =
        # QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.ALT), qres_wgt,
        #                qres_wgt.on_alt_pressed,
        #                context=QtCore..Qt.WidgetShortcut)
        qres_wgt.button_list = None
        qres_wgt.show_new = True
        qres_wgt.show_join = True
        qres_wgt.show_split = True
        qres_wgt.tt = ut.tic()
        # Set results data
        if USE_FILTER_PROXY:
            qres_wgt.add_checkboxes(
                qres_wgt.show_new, qres_wgt.show_join, qres_wgt.show_split
            )

        lbl = gt.newLineEdit(
            qres_wgt,
            text="'T' marks as correct match. 'F' marks as incorrect match. Alt brings up context menu. Double click a row to inspect matches.",
            editable=False,
            enabled=False,
        )
        qres_wgt.layout().setSpacing(0)
        qres_wgt_layout = qres_wgt.layout()
        if hasattr(qres_wgt_layout, 'setMargin'):
            qres_wgt_layout.setMargin(0)
        else:
            qres_wgt_layout.setContentsMargins(0, 0, 0, 0)
        bottom_bar = gt.newWidget(
            qres_wgt, orientation=Qt.Horizontal, spacing=0, margin=0
        )
        bottom_bar.layout().setSpacing(0)
        bottom_bar_layout = bottom_bar.layout()
        if hasattr(bottom_bar_layout, 'setMargin'):
            bottom_bar_layout.setMargin(0)
        else:
            bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        lbl.setMinimumSize(0, 0)
        lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        # lbl.setSizePolicy(gt.newSizePolicy())

        qres_wgt.layout().addWidget(bottom_bar)
        bottom_bar.addWidget(lbl)
        bottom_bar.addNewButton(
            'Mark unreviewed with higher scores as correct',
            pressed=qres_wgt.mark_unreviewed_above_score_as_correct,
        )
        bottom_bar.addNewButton('Repopulate', pressed=qres_wgt.repopulate)
        bottom_bar.addNewButton('Edit Filters', pressed=qres_wgt.edit_filters)

        qres_wgt.setSizePolicy(gt.newSizePolicy())
        qres_wgt.repopulate()
        qres_wgt.connect_signals_and_slots()
        if callback is None:
            callback = partial(ut.identity, None)
        qres_wgt.callback = callback
        qres_wgt.view.setColumnHidden(0, False)
        qres_wgt.view.setColumnHidden(1, False)
        qres_wgt.view.connect_single_key_to_slot(gt.ALT_KEY, qres_wgt.on_alt_pressed)
        qres_wgt.view.connect_keypress_to_slot(qres_wgt.on_special_key_pressed)
        if parent is None:
            # Register parentless QWidgets
            fig_presenter.register_qt4_win(qres_wgt)

        dbdir = qres_wgt.qreq_.ibs.get_dbdir()
        expt_dir = ut.ensuredir(ut.unixjoin(dbdir, 'SPECIAL_GGR_EXPT_LOGS'))
        review_log_dir = ut.ensuredir(ut.unixjoin(expt_dir, 'review_logs'))

        ts = ut.get_timestamp(isutc=True, timezone=True)
        log_fpath = ut.unixjoin(
            review_log_dir, 'review_log_%s_%s.json' % (qres_wgt.qreq_.ibs.dbname, ts)
        )

        # LOG ALL CHANGES MADE TO NAMES
        import logging

        # ut.vd(review_log_dir)
        # create logger with 'spam_application'
        logger = logging.getLogger('query_review')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
        logger.info(
            'NUM PAIRS TO EVIDENCE_DECISION (nRows=%d)' % (qres_wgt.review_api.nRows,)
        )
        logger.info(
            'PARENT QUERY REQUEST (cfgstr=%s)'
            % (qres_wgt.qreq_.get_cfgstr(with_input=True),)
        )

    def edit_filters(qres_wgt):
        from wbia import dtool

        config = dtool.Config.from_dict(qres_wgt.review_cfg)
        dlg = gt.ConfigConfirmWidget.as_dialog(
            qres_wgt,
            title='Edit Filters',
            msg='Edit Filters',
            with_spoiler=False,
            config=config,
        )
        dlg.resize(700, 500)
        self = dlg.widget
        dlg.exec_()
        print('config = %r' % (config,))
        updated_config = self.config  # NOQA
        print('updated_config = %r' % (updated_config,))
        qres_wgt.review_cfg = updated_config.asdict()
        qres_wgt.repopulate()

    def repopulate(qres_wgt):
        print('repopulate')
        # Really just reloads the widget
        qreq_ = qres_wgt.qreq_

        print('[qres_wgt] repopulate set_query_results()')
        tblnice = 'Query Results: ' + qres_wgt.query_title

        qres_wgt.qreq_ = qreq_
        qres_wgt.review_api = id_review_api.make_review_api(
            qres_wgt.ibs,
            qres_wgt.cm_list,
            qreq_=qres_wgt.qreq_,
            review_cfg=qres_wgt.review_cfg,
        )

        headers = qres_wgt.review_api.make_headers(tblname='review_api', tblnice=tblnice)

        # HACK IN ROW SIZE
        vertical_header = qres_wgt.view.verticalHeader()
        vertical_header.setDefaultSectionSize(qres_wgt.review_api.get_thumb_size())

        # super call
        qres_wgt.change_headers(headers)
        qres_wgt.setWindowTitle(
            headers.get('nice', '') + ' nRows=%d' % (qres_wgt.model.rowCount())
        )

        # HACK IN COL SIZE
        qres_wgt.resize_headers(api=qres_wgt.review_api)
        # horizontal_header = qres_wgt.view.horizontalHeader()
        # for col, width in six.iteritems(qres_wgt.review_api.col_width_dict):
        #    #horizontal_header.defaultSectionSize()
        #    try:
        #        index = qres_wgt.review_api.col_name_list.index(col)
        #    except ValueError:
        #        pass
        #    horizontal_header.resizeSection(index, width)

    @gt.slot_()
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
        # qres_wgt.view.pressed.connect(qres_wgt._on_pressed)

    @gt.slot_(QtCore.QModelIndex)
    def _on_doubleclick(qres_wgt, qtindex):
        print('[qres_wgt] _on_doubleclick: ')
        print('[qres_wgt] DoubleClicked: ' + str(gt.qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        if qres_wgt.review_api.col_edit_list[col]:
            print('do nothing special for editable columns')
            return
        return qres_wgt.show_match_at_qtindex(qtindex)

    # @gt.slot_(QtCore.QModelIndex)
    # def _on_pressed(qres_wgt, qtindex):
    #    print('[qres_wgt] _on_pressed: ')
    #    def _check_for_double_click(qres_wgt, qtindex):
    #        threshold = 0.20  # seconds
    #        distance = ut.toc(qres_wgt.tt)
    #        if distance <= threshold:
    #            qres_wgt._on_doubleclick(qtindex)
    #        qres_wgt.tt = ut.tic()
    #    _check_for_double_click(qres_wgt, qtindex)

    def selectedRows(qres_wgt):
        selected_qtindex_list2 = qres_wgt.view.selectedRows()
        # # selected_qtindex_list = qres_wgt.view.selectedIndexes()
        # selected_qtindex_list2 = []
        # seen_ = set([])
        # for qindex in selected_qtindex_list:
        #    row = qindex.row()
        #    if row not in seen_:
        #        selected_qtindex_list2.append(qindex)
        #        seen_.add(row)
        return selected_qtindex_list2

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

    def on_special_key_pressed(qres_wgt, view, event):
        # selected_qtindex_list = view.selectedIndexes()
        selected_qtindex_list = qres_wgt.selectedRows()

        # if len(selected_qtindex_list) == 1:
        for qtindex in selected_qtindex_list:
            print('event = %r ' % (event,))
            print('event.key() = %r ' % (event.key(),))
            qtindex = selected_qtindex_list[0]
            ibs = qres_wgt.ibs
            aid1, aid2 = qres_wgt.get_aidpair_from_qtindex(qtindex)
            _tup = qres_wgt.get_widget_review_vars(aid1)
            ibs, cm, qreq_, update_callback, backend_callback = _tup

            options = get_aidpair_context_menu_options(
                ibs,
                aid1,
                aid2,
                cm,
                qreq_=qreq_,
                logger=qres_wgt.logger,
                update_callback=update_callback,
                backend_callback=backend_callback,
            )

            def make_option_dict(options):
                option_dict = {
                    key[key.find('&') + 1]: val for key, val in options if '&' in key
                }
                return option_dict

            # TODO: use guitool options dict
            # print('option_dict = %s' % (ut.repr3(option_dict, nl=2),))
            option_dict = make_option_dict(options)

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
            elif event_key == QtCore.Qt.Key_P:
                # Mark as photobomb
                pair_options = option_dict['g']
                pair_option_dict = make_option_dict(pair_options)
                pair_option_dict['P']()
            elif event_key == QtCore.Qt.Key_K:
                annot1_options_dict = make_option_dict(option_dict['1'])
                qual1_options_dict = make_option_dict(annot1_options_dict['Q'])
                qual1_options_dict['4']()
            elif event_key == QtCore.Qt.Key_L:
                annot1_options_dict = make_option_dict(option_dict['2'])
                qual1_options_dict = make_option_dict(annot1_options_dict['Q'])
                qual1_options_dict['4']()
            # # BROKEN FOR NOW
            # elif event_key == QtCore.Qt.Key_S:
            #    option_dict['S']()
            # elif event_key == QtCore.Qt.Key_P:
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
            # model.select()
        # else:
        #    print('[key] Multiple %d selection' % (len(selected_qtindex_list),))

    @gt.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(qres_wgt, qtindex, qpoint):
        """
        popup context menu
        """
        # selected_qtindex_list = qres_wgt.view.selectedIndexes()
        selected_qtindex_list = qres_wgt.selectedRows()
        if len(selected_qtindex_list) == 1:
            qwin = qres_wgt
            aid1, aid2 = qres_wgt.get_aidpair_from_qtindex(qtindex)
            tup = qres_wgt.get_widget_review_vars(aid1)
            ibs, cm, qreq_, update_callback, backend_callback = tup
            options = get_aidpair_context_menu_options(
                ibs,
                aid1,
                aid2,
                cm,
                qreq_=qreq_,
                logger=qres_wgt.logger,
                update_callback=update_callback,
                backend_callback=backend_callback,
            )
            gt.popup_menu(qwin, qpoint, options)
        else:
            print('[context] Multiple %d selection' % (len(selected_qtindex_list),))

    def get_widget_review_vars(qres_wgt, qaid):
        ibs = qres_wgt.ibs
        qreq_ = qres_wgt.qreq_
        cm = qres_wgt.qaid2_cm[qaid]
        update_callback = None  # hack (checking if necessary)
        backend_callback = qres_wgt.callback
        return ibs, cm, qreq_, update_callback, backend_callback

    def get_aidpair_from_qtindex(qres_wgt, qtindex):
        model = qtindex.model()
        qaid = model.get_header_data('qaid', qtindex)
        daid = model.get_header_data('aid', qtindex)
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
        match_interaction = cm.ishow_single_annotmatch(qres_wgt.qreq_, daid, mode=0)
        fig = match_interaction.fig
        fig_presenter.bring_to_front(fig)

    def mark_unreviewed_above_score_as_correct(qres_wgt):
        selected_qtindex_list = qres_wgt.selectedRows()
        if len(selected_qtindex_list) == 1:
            qtindex = selected_qtindex_list[0]
            # aid1, aid2 = qres_wgt.get_aidpair_from_qtindex(qtindex)
            thresh = qtindex.model().get_header_data('score', qtindex)
            print('thresh = %r' % (thresh,))

            rows = qres_wgt.review_api.ider()
            scores_ = qres_wgt.review_api.get(
                qres_wgt.review_api.col_name_list.index('score'), rows
            )
            valid_rows = ut.compress(rows, scores_ >= thresh)
            aids1 = qres_wgt.review_api.get(
                qres_wgt.review_api.col_name_list.index('qaid'), valid_rows
            )
            aids2 = qres_wgt.review_api.get(
                qres_wgt.review_api.col_name_list.index('aid'), valid_rows
            )
            # ibs = qres_wgt.ibs
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
                # 'Review More'
            ]
            msg = (
                ut.codeblock(
                    """
                There are %d names and %d annotations in this mass review set.
                Mass review has discovered %d internal groups.
                Accepting will induce a database grouping of %d names.
                """
                )
                % (
                    len(nid2_aids),
                    len(changing_aids),
                    len(review_groups),
                    len(dbside_groups),
                )
            )

            reply = gt.user_option(msg=msg, options=options)

            if reply == options[0]:
                # This is not the smartest way to group names.
                # Ideally what will happen here, is that reviewed edges will go into
                # the new graph name inference algorithm.
                # then the chosen point will be used as the threshold. Then
                # the graph cut algorithm will be applied.
                logger = qres_wgt.logger
                logger.debug(msg)
                logger.info('START MASS_THRESHOLD_MERGE')
                logger.info('num_groups=%d thresh=%r' % (len(dbside_groups), thresh,))
                for count, subgraph in enumerate(dbside_groups):
                    thresh_aid_pairs = [
                        edge
                        for edge, flag in nx.get_edge_attributes(
                            graph, 'user_thresh_match'
                        ).items()
                        if flag
                    ]
                    thresh_uuid_pairs = ibs.unflat_map(
                        ibs.get_annot_uuids, thresh_aid_pairs
                    )
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
                    ibs.set_annotmatch_reviewer(
                        am_rowids, ['algo:lnbnn_thresh'] * len(am_rowids)
                    )

                    logger.info('START GROUP %d' % (count,))
                    logger.info(
                        'GROUP BASED ON %d ANNOT_PAIRS WITH SCORE ABOVE (thresh=%r)'
                        % (len(thresh_uuid_pairs), thresh,)
                    )
                    logger.debug('(uuid_pairs=%r)' % (thresh_uuid_pairs))
                    logger.debug('(merge_name=%r)' % (merge_name))
                    logger.debug(
                        'CHANGE NAME OF %d (annot_uuids=%r) WITH (previous_names=%r) TO (%s) (merge_name=%r)'
                        % (
                            len(annot_uuids),
                            annot_uuids,
                            previous_names,
                            type_,
                            merge_name,
                        )
                    )
                    logger.debug(
                        'ADDITIONAL CHANGE NAME OF %d (annot_uuids=%r) WITH (previous_names=%r) TO (%s) (merge_name=%r)'
                        % (
                            len(other_auuids),
                            other_auuids,
                            other_previous_names,
                            type_,
                            merge_name,
                        )
                    )
                    logger.info('END GROUP %d' % (count,))
                    new_nids = [merge_nid] * len(aids)
                    ibs.set_annot_name_rowids(aids, new_nids)
                logger.info('END MASS_THRESHOLD_MERGE')
        else:
            print('[context] Multiple %d selection' % (len(selected_qtindex_list),))


# ______


def set_annot_pair_as_positive_match_(ibs, aid1, aid2, cm, qreq_, **kwargs):
    """
    MARK AS CORRECT
    """

    def on_nontrivial_merge(ibs, aid1, aid2):
        MERGE_NEEDS_INTERACTION = False
        MERGE_NEEDS_VERIFICATION = True
        if MERGE_NEEDS_INTERACTION:
            raise guiexcept.NeedsUserInput('confirm merge')
        elif MERGE_NEEDS_VERIFICATION:
            name1, name2 = ibs.get_annot_names([aid1, aid2])
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
            msgfmt = ut.codeblock(
                """
               Confirm merge of animal {name1} and {name2}
               {name1} has {num_gt1} annotations
               {name2} has {num_gt2} annotations
               """
            )
            msg = msgfmt.format(
                name1=name1,
                name2=name2,
                num_gt1=len(aid1_and_groundtruth),
                num_gt2=len(aid2_and_groundtruth),
            )
            if not gt.are_you_sure(parent=None, msg=msg, default='Yes'):
                raise guiexcept.UserCancel('canceled merge')

    try:
        status = ibs.set_annot_pair_as_positive_match(
            aid1,
            aid2,
            on_nontrivial_merge=on_nontrivial_merge,
            logger=kwargs.get('logger', None),
        )
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
        print(
            'There are %d annots in this name. Need more sophisticated split'
            % (len(aid1_groundtruth))
        )
        raise guiexcept.NeedsUserInput('non-trivial split')

    try:
        status = ibs.set_annot_pair_as_negative_match(
            aid1,
            aid2,
            on_nontrivial_split=on_nontrivial_split,
            logger=kwargs.get('logger', None),
        )
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        options = ['Flag for later', 'Review now']
        reply = gt.user_option(
            msg=ut.codeblock(
                """
                Marking this as False induces a split case.
                Choose how to handle this.
                """
            ),
            options=options,
        )
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
            ibs.set_annotmatch_evidence_decision(
                [am_rowid], [ibs.const.EVIDENCE_DECISION.NEGATIVE]
            )
        elif reply == options[1]:
            review_match(ibs, aid1, aid2, qreq_=qreq_, cm=cm, **kwargs)
    except guiexcept.UserCancel:
        print('user canceled negative match')


def review_match(
    ibs,
    aid1,
    aid2,
    update_callback=None,
    backend_callback=None,
    qreq_=None,
    cm=None,
    **kwargs,
):
    print('Review match: {}-vs-{}'.format(aid1, aid2))
    from wbia.viz.interact import interact_name

    # ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    mvinteract = interact_name.MatchVerificationInteraction(
        ibs,
        aid1,
        aid2,
        fnum=64,
        update_callback=update_callback,
        cm=cm,
        qreq_=qreq_,
        backend_callback=backend_callback,
        **kwargs,
    )
    return mvinteract
    # ih.register_interaction(mvinteract)


def get_aidpair_context_menu_options(
    ibs, aid1, aid2, cm, qreq_=None, aid_list=None, **kwargs
):
    """ assert that the ampersand cannot have duplicate keys

    Args:
        ibs (wbia.IBEISController):  wbia controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        cm (wbia.ChipMatch):  object of feature correspondences and scores
        qreq_ (wbia.QueryRequest):  query request object with hyper-parameters(default = None)
        aid_list (list):  list of annotation rowids(default = None)

    Returns:
        list: options

    CommandLine:
        python -m wbia.gui.inspect_gui --exec-get_aidpair_context_menu_options
        python -m wbia.gui.inspect_gui --exec-get_aidpair_context_menu_options --verbose
        python -m wbia.gui.inspect_gui --exec-get_aidpair_context_menu_options --verbose -a timecontrolled -t invarbest --db PZ_Master1  --qaid 574

        # Other scripts that call this one;w
        python -m wbia.dev -e cases --db PZ_Master1  -a timectrl -t best --filt :sortdsc=gfscore,fail=True,min_gtscore=.0001 --show
        python -m wbia.dev -e cases --db PZ_MTEST  -a timectrl -t best --filt :sortdsc=gfscore,fail=True,min_gtscore=.0001 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.gui.inspect_gui import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(t=['default:fg_on=False'])
        >>> cm_list = qreq_.execute()
        >>> cm = cm_list[0]
        >>> ibs = qreq_.ibs
        >>> aid1 = cm.qaid
        >>> aid2 = cm.get_top_aids()[0]
        >>> aid_list = None
        >>> options = get_aidpair_context_menu_options(ibs, aid1, aid2, cm, qreq_, aid_list)
        >>> result = ('options = %s' % (ut.repr2(options),))
        >>> print(result)
    """
    if ut.VERBOSE:
        print('[inspect_gui] Building AID pair context menu options')
    options = []

    # assert qreq_ is not None, 'must specify qreq_'

    if cm is not None:
        # MAKE SURE THIS IS ALL CM
        show_chip_match_features_option = (
            'Show chip feature matches',
            partial(cm.ishow_single_annotmatch, qreq_, aid2, mode=0),
        )
        if aid_list is not None:
            # Give a subcontext menu for multiple options
            def partial_show_chip_matches_to(aid_):
                return lambda: cm.ishow_single_annotmatch(qreq_, aid_, mode=0)

            show_chip_match_features_option = (
                'Show chip feature matches',
                [
                    ('to aid=%r' % (aid_,), partial_show_chip_matches_to(aid_))
                    for aid_ in aid_list
                ],
            )

        def show_single_namematch():
            import wbia.plottool as pt

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
        chip_contex_options = make_annotpair_context_options(ibs, aid1, aid2, qreq_)
        if len(chip_contex_options) > 2:
            options += [
                ('Annot Conte&xt Options', chip_contex_options),
            ]
        else:
            options += chip_contex_options

    with_review_options = True

    from wbia.viz import viz_graph2

    if with_review_options:
        aid_list2 = [aid1, aid2]
        options += [
            ('Mark as &Reviewed', lambda: ibs.set_annot_pair_as_reviewed(aid1, aid2)),
            (
                'Mark as &True Match.',
                lambda: set_annot_pair_as_positive_match_(
                    ibs, aid1, aid2, cm, qreq_, **kwargs
                ),
            ),
            (
                'Mark as &False Match.',
                lambda: set_annot_pair_as_negative_match_(
                    ibs, aid1, aid2, cm, qreq_, **kwargs
                ),
            ),
            # ('Mark Disjoint Viewpoints.',
            # lambda:  set_annot_pair_as_unknown_match_(
            #     ibs, aid1, aid2, cm, qreq_, **kwargs)),
            (
                'Inspect Match Candidates',
                lambda: review_match(ibs, aid1, aid2, qreq_=qreq_, cm=cm, **kwargs),
            ),
            # FIXME, more than 2 aids
            (
                'New Graph Interaction (Names)',
                partial(
                    viz_graph2.make_qt_graph_interface,
                    ibs,
                    nids=ibs.get_annot_nids(aid_list2),
                ),
            ),
        ]

    with_vsone = True
    if with_vsone:
        options += [
            (
                'Tune Vsone(vt)',
                make_vsone_context_options(ibs, aid1, aid2, qreq_=qreq_)[0][1],
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
            cm = qreq2_.execute([aid1])[0]
            cm.ishow_single_annotmatch(qreq_, aid2, mode=0)

        options += [
            ('Load Vsmany', vsmany_load_and_show),
        ]
        pass

    with_match_tags = True
    if with_match_tags:
        pair_tag_options = make_aidpair_tag_context_options(ibs, aid1, aid2)
        options += [('Match Ta&gs', pair_tag_options)]

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


def make_vsone_tuner(
    ibs, edge=None, qreq_=None, autoupdate=True, info_text=None, cfgdict=None
):
    """
    Makes a qt widget for inspecting one-vs-one matches

    CommandLine:
        python -m wbia.gui.inspect_gui make_vsone_tuner --show

    Example:
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.gui.inspect_gui import *  # NOQA
        >>> import wbia
        >>> gt.ensure_qapp()
        >>> ut.qtensure()
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> edge = ut.get_argval('--aids', default=[1, 2], type_=list)
        >>> self = make_vsone_tuner(ibs, edge, autoupdate=False)
        >>> ut.quit_if_noshow()
        >>> self.show()
        >>> gt.qtapp_loop(qwin=self, freq=10)

    """
    from vtool import inspect_matches
    import vtool as vt

    if cfgdict is not None:
        assert qreq_ is None, 'specify only one cfg or qreq_'
    else:
        cfgdict = {}

    def set_edge(self, edge, info_text=None):
        aid1, aid2 = edge
        if qreq_ is None:
            qreq2_ = ibs.new_query_request([aid1], [aid2], cfgdict=cfgdict, verbose=False)
        else:
            qreq2_ = ibs.new_query_request(
                [aid1], [aid2], cfgdict=qreq_.qparams, verbose=False
            )
        qconfig2_ = qreq2_.extern_query_config2
        dconfig2_ = qreq2_.extern_data_config2
        annot1 = ibs.annots([aid1], config=qconfig2_)[0]._make_lazy_dict()
        annot2 = ibs.annots([aid2], config=dconfig2_)[0]._make_lazy_dict()
        match = vt.PairwiseMatch(annot1, annot2)

        def on_context():
            from wbia.gui import inspect_gui

            return inspect_gui.make_annotpair_context_options(ibs, aid1, aid2, None)

        self.set_match(match, on_context, info_text)

    self = inspect_matches.MatchInspector(autoupdate=autoupdate, cfgdict=cfgdict)
    ut.inject_func_as_method(self, set_edge)
    if edge is not None:
        self.set_edge(edge, info_text)
    return self


def show_vsone_tuner(ibs, qaid, daid, qreq_=None):
    edge = (qaid, daid)
    print('[inspect_gui] show_vsone_tuner edge={}'.format(edge))
    self = make_vsone_tuner(ibs, edge, qreq_=qreq_)
    self.show()


def make_vsone_context_options(ibs, aid1, aid2, qreq_):
    r"""
    CommandLine:
        python -m wbia.gui.inspect_gui make_vsone_context_options --db PZ_MTEST
        python -m wbia.gui.inspect_gui make_vsone_context_options \
            --dbdir ~/lev/media/hdd/work/WWF_Lynx/  --aids=2587,2398
        python -m wbia.gui.inspect_gui make_vsone_context_options \
                --db PZ_Master1 --aids=8,242

    Example:
        >>> # SCRIPT
        >>> from wbia.gui.inspect_gui import *  # NOQA
        >>> import wbia
        >>> gt.ensure_qapp()
        >>> ut.qtensure()
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aids = ut.get_argval('--aids', default=[1, 2], type_=list)
        >>> print('aids = %r' % (aids,))
        >>> aid1, aid2 = aids
        >>> options = make_vsone_context_options(ibs, 1, 2, None)
        >>> dict(options)['Tune Vsone(vt)']()
        >>> gt.qtapp_loop(freq=10)
    """
    options = [
        ('Tune Vsone(vt)', partial(show_vsone_tuner, ibs, aid1, aid2, qreq_=qreq_)),
    ]
    return options


def make_annotpair_context_options(ibs, aid1, aid2, qreq_):
    from wbia.viz.interact import interact_chip

    aid_list2 = [aid1, aid2]
    if qreq_ is None:
        config2_list_ = [None, None]
    else:
        config2_list_ = [qreq_.extern_query_config2, qreq_.extern_data_config2]

    chip_contex_options = []
    print('config2_list_ = %r' % (config2_list_,))
    print('aid_list2 = %r' % (aid_list2,))
    for count, (aid, config2_) in enumerate(zip(aid_list2, config2_list_), start=1):
        chip_contex_options += [
            (
                'Annot&%d Options (aid=%r)' % (count, aid,),
                interact_chip.build_annot_context_options(
                    ibs, aid, refresh_func=None, config2_=config2_
                ),
            )
        ]

    # interact_chip_options = []
    # for count, (aid, config2_) in enumerate(zip(aid_list2,
    #                                            config2_list_),
    #                                        start=1):
    #    interact_chip_options += [
    #        ('Interact Annot&%d' % (count,),
    #         partial(interact_chip.ishow_chip, ibs, aid, config2_=config2_,
    #                 fnum=None, **kwargs)),
    #    ]
    # interact_chip_actions = ut.get_list_column(interact_chip_options, 1)
    # interact_chip_options.append(
    #    ('Interact &All Annots', lambda: [func() for func in
    #                                      interact_chip_actions]),
    # )

    # options += [
    #    #('Interact Annots', interact_chip_options),
    #    #('Annot Conte&xt Options', chip_contex_options),
    # ]
    # if len(chip_contex_options) > 2:
    #    return [
    #        ('Annot Conte&xt Options', chip_contex_options),
    #    ]
    # else:
    return chip_contex_options


def make_aidpair_tag_context_options(ibs, aid1, aid2):
    from wbia import tag_funcs

    annotmatch_rowid = ibs.get_annotmatch_rowid_from_undirected_superkey([aid1], [aid2])[
        0
    ]

    if annotmatch_rowid is None:
        tags = []
    else:
        tags = ibs.get_annotmatch_case_tags([annotmatch_rowid])[0]
        tags = [_.lower() for _ in tags]
    standard, other = tag_funcs.get_cate_categories()
    case_list = standard + other

    # used_chars = gt.find_used_chars(ut.get_list_column(options, 0))
    used_chars = []
    case_hotlink_list = gt.make_word_hotlinks(case_list, used_chars)
    pair_tag_options = []
    if True or ut.VERBOSE:
        print('[inspect_gui] aid1, aid2 = %r, %r' % (aid1, aid2,))
        print('[inspect_gui] annotmatch_rowid = %r' % (annotmatch_rowid,))
        print('[inspect_gui] tags = %r' % (tags,))
    if ut.VERBOSE:
        print('[inspect_gui] Making case hotlist: ' + ut.repr2(case_hotlink_list))

    def _wrap_set_annotmatch_prop(prop, toggle_val):
        if ut.VERBOSE:
            print('[SETTING] Clicked set prop=%r to val=%r' % (prop, toggle_val,))
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
            print('[inspect_gui] annotmatch_rowid = %r' % (annotmatch_rowid,))
            print('[inspect_gui] tags = %r' % (tags,))

    for case, case_hotlink in zip(case_list, case_hotlink_list):
        toggle_val = case.lower() not in tags
        fmtstr = 'Flag %s case' if toggle_val else 'Unflag %s case'
        pair_tag_options += [
            # (fmtstr % (case_hotlink,), lambda:
            # ibs.set_annotmatch_prop(case, _get_annotmatch_rowid(),
            #                        [toggle_val])),
            # (fmtstr % (case_hotlink,), partial(ibs.set_annotmatch_prop,
            # case, [annotmatch_rowid], [toggle_val])),
            (
                fmtstr % (case_hotlink,),
                partial(_wrap_set_annotmatch_prop, case, toggle_val),
            ),
        ]
    if ut.VERBOSE:
        print(
            'Partial tag funcs:'
            + ut.repr2(
                [
                    ut.func_str(func, func.args, func.keywords)
                    for func in ut.get_list_column(pair_tag_options, 1)
                ]
            )
        )
    return pair_tag_options


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.gui.inspect_gui
        python -m wbia.gui.inspect_gui --allexamples
        python -m wbia.gui.inspect_gui --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
