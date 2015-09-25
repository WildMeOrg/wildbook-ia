"""
This module was never really finished. It is used in some cases
to display the results from a query in a qt window. It needs
some work if its to be re-integrated.

TODO:
    Refresh name table on inspect gui close
"""
from __future__ import absolute_import, division, print_function
from functools import partial
from guitool import qtype, APIItemWidget, APIItemModel, FilterProxyModel, ChangeLayoutContext
from guitool.__PYQT__ import QtGui, QtCore
from ibeis import ibsfuncs
from ibeis.experiments import results_organizer
#from ibeis.viz import interact
from ibeis.viz import viz_helpers as vh
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
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[inspect_gui]')


MATCHED_STATUS_TEXT  = 'Matched'
REVIEWED_STATUS_TEXT = 'Reviewed'
USE_FILTER_PROXY = False


class CustomFilterModel(FilterProxyModel):
    def __init__(model, headers=None, parent=None, *args):
        FilterProxyModel.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.eid = -1  # negative one is an invalid eid  # seems unused
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
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_iders[0]()

    def _change_enc(model, eid):
        model.eid = eid
        # seems unused
        with ChangeLayoutContext([model]):
            FilterProxyModel._update_rows(model)


class QueryResultsWidget(APIItemWidget):
    """ Window for gui inspection

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show

    """

    def __init__(qres_wgt, ibs, qaid2_qres, parent=None, callback=None,
                 name_scoring=False, qreq_=None, **kwargs):
        if ut.VERBOSE:
            print('[qres_wgt] Init QueryResultsWidget')
        # Uncomment below to turn on FilterProxyModel
        if USE_FILTER_PROXY:
            APIItemWidget.__init__(qres_wgt, parent=parent,
                                    model_class=CustomFilterModel)
        else:
            APIItemWidget.__init__(qres_wgt, parent=parent)

        #qres_wgt.altkey_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.ALT), qres_wgt, qres_wgt.on_alt_pressed, context=QtCore..Qt.WidgetShortcut)
        qres_wgt.button_list = None
        qres_wgt.show_new = True
        qres_wgt.show_join = True
        qres_wgt.show_split = True
        qres_wgt.tt = utool.tic()
        # Set results data
        if USE_FILTER_PROXY:
            qres_wgt.add_checkboxes(qres_wgt.show_new, qres_wgt.show_join, qres_wgt.show_split)
        qres_wgt.set_query_results(ibs, qaid2_qres, name_scoring=name_scoring, qreq_=qreq_, **kwargs)
        qres_wgt.connect_signals_and_slots()
        if callback is None:
            callback = lambda: None
        qres_wgt.callback = callback
        qres_wgt.view.setColumnHidden(0, False)
        qres_wgt.view.setColumnHidden(1, False)
        #qres_wgt.view.connect_single_key_to_slot(QtCore.Qt.ALT, qres_wgt.on_alt_pressed)
        ALT_KEY = 16777251
        qres_wgt.view.connect_single_key_to_slot(ALT_KEY, qres_wgt.on_alt_pressed)
        qres_wgt.view.connect_keypress_to_slot(qres_wgt.on_special_key_pressed)
        if parent is None:
            # Register parentless QWidgets
            fig_presenter.register_qt4_win(qres_wgt)

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

    def set_query_results(qres_wgt, ibs, qaid2_qres, name_scoring=False, qreq_=None, **kwargs):
        print('[qres_wgt] Change QueryResultsWidget data')
        tblnice = 'Query Results: ' + kwargs.get('query_title', '')
        ut.util_dict.delete_dict_keys(kwargs, ['query_title'])

        qres_wgt.ibs = ibs
        qres_wgt.qaid2_qres = qaid2_qres
        qres_wgt.qreq_ = qreq_
        qres_wgt.qres_api = make_qres_api(ibs, qaid2_qres, name_scoring=name_scoring, qreq_=qreq_, **kwargs)
        qres_wgt.update_checkboxes()

        headers = qres_wgt.qres_api.make_headers(tblname='qres_api', tblnice=tblnice)

        # HACK IN ROW SIZE
        vertical_header = qres_wgt.view.verticalHeader()
        vertical_header.setDefaultSectionSize(qres_wgt.qres_api.get_thumb_size())

        # super call
        APIItemWidget.change_headers(qres_wgt, headers)
        #qres_wgt.change_headers(headers)

        # HACK IN COL SIZE
        horizontal_header = qres_wgt.view.horizontalHeader()
        for col, width in six.iteritems(qres_wgt.qres_api.col_width_dict):
            #horizontal_header.defaultSectionSize()
            index = qres_wgt.qres_api.col_name_list.index(col)
            horizontal_header.resizeSection(index, width)

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
            #review_match_at_qtindex(qres_wgt, qtindex, qres_callback=qres_callback)
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
            ibs, qres, qreq_, update_callback, backend_callback = get_widget_review_vars(qres_wgt, aid1)
            options = get_aidpair_context_menu_options(ibs, aid1, aid2, qres, qreq_=qreq_, update_callback=update_callback, backend_callback=backend_callback)
            option_dict = {key[key.find('&') + 1]: val for key, val in options if key.find('&') > -1}

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
        ibs, qres, qreq_, update_callback, backend_callback = get_widget_review_vars(qres_wgt, aid1)
        show_aidpair_context_menu(ibs, qwin, qpoint, aid1, aid2, qres,
                                     qreq_=qreq_,
                                     update_callback=update_callback,
                                     backend_callback=backend_callback)


def get_widget_review_vars(qres_wgt, qaid):
    ibs   = qres_wgt.ibs
    qreq_ = qres_wgt.qreq_
    qres  = qres_wgt.qaid2_qres[qaid]
    update_callback = None  # hack (checking if necessary)
    backend_callback = qres_wgt.callback
    return ibs, qres, qreq_, update_callback, backend_callback


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


def mark_pair_as_reviewed(qres_wgt, qtindex):
    """
    Sets the reviewed flag to whatever the current truth status is
    """
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    ibs = qres_wgt.ibs
    ibs.mark_annot_pair_as_reviewed(qaid, daid)


def mark_pair_as_positive_match(qres_wgt, qtindex):
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    ibs = qres_wgt.ibs
    try:
        status = mark_annot_pair_as_positive_match_(ibs, qaid, daid)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match_at_qtindex(qres_wgt, qtindex)
    except guiexcept.UserCancel:
        print('user canceled positive match')


def mark_pair_as_negative_match(qres_wgt, qtindex):
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    ibs = qres_wgt.ibs
    return mark_annot_pair_as_negative_match_(ibs, qaid, daid)


def show_match_at_qtindex(qres_wgt, qtindex):
    print('interact')
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    qreq_ = qres_wgt.qreq_
    #fig = interact.ishow_matches(qres_wgt.ibs, qres_wgt.qaid2_qres[qaid], aid, mode=1)
    #match_interaction = qres_wgt.qaid2_qres[qaid].ishow_matches(qres_wgt.ibs, aid, mode=1, qreq_=qreq_)
    match_interaction = qres_wgt.qaid2_qres[qaid].ishow_matches(qres_wgt.ibs, daid, mode=0, qreq_=qreq_)
    fig = match_interaction.fig
    fig_presenter.bring_to_front(fig)


def review_match_at_qtindex(qres_wgt, qtindex):
    print('review')
    #qres_callback = partial(show_match_at_qtindex, qres_wgt, qtindex)
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    #update_callback = model._update
    #ibs   = qres_wgt.ibs
    #qreq_ = qres_wgt.qreq_
    #qres  = qres_wgt.qaid2_qres[qaid]
    #update_callback = None  # hack (checking if necessary)
    #backend_callback = qres_wgt.callback
    qaid, daid = get_aidpair_from_qtindex(qres_wgt, qtindex)
    ibs, qres, qreq_, update_callback, backend_callback = get_widget_review_vars(qres_wgt, qaid)
    review_match(ibs, qaid, daid, update_callback=update_callback,
                 backend_callback=backend_callback, qres=qres, qreq_=qreq_)


# ______


def get_aidpair_context_menu_options(ibs, aid1, aid2, qres, qreq_=None, aid_list=None, **kwargs):
    """ assert that the ampersand cannot have duplicate keys

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        qres (QueryResult):  object of feature correspondences and scores
        qreq_ (QueryRequest):  query request object with hyper-parameters(default = None)
        aid_list (list):  list of annotation rowids(default = None)

    Returns:
        list: options

    CommandLine:
        python -m ibeis.gui.inspect_gui --exec-get_aidpair_context_menu_options
        python -m ibeis.gui.inspect_gui --exec-get_aidpair_context_menu_options --verbose
        python -m ibeis.gui.inspect_gui --exec-get_aidpair_context_menu_options --verbose -a timecontrolled -t invarbest --db PZ_Master1  --qaid 574

        # Other scripts that call this one;w
        python -m ibeis.dev -e cases -a timecontrolled -t invarbest --db PZ_Master1 --qaid 574 --show
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres -a timecontrolled -t invarbest --db PZ_Master1 --qaid 574 --show --verbadd --verbaset --verbose
        python -m ibeis.viz.interact.interact_qres --test-ishow_qres -a timecontrolled -t invarbest --db testdb1 --show --qaid 1
        python -m ibeis.dev -e scores -t invarbest -a timecontrolled:require_quality=True --db PZ_Master1 --filt :onlygood=False,smallfptime=False --show


        python -m ibeis.dev -e cases -t invarbest -a timecontrolled:require_quality=True --db PZ_Master1 --show --vh

    Example:
        >>> # SCRIPT
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs, qreq_, qres = ibeis.testdata_qres()
        >>> aid1 = qres.qaid
        >>> aid2 = qres.get_top_aids()[0]
        >>> aid_list = None
        >>> options = get_aidpair_context_menu_options(ibs, aid1, aid2, qres, qreq_, aid_list)
        >>> result = ('options = %s' % (ut.list_str(options),))
        >>> print(result)

    """
    if ut.VERBOSE:
        print('[inspect_gui] Building AID pair context menu options')
    assert qreq_ is not None, 'must specify qreq_'
    show_chip_match_features_option = ('Show chip feature matches', lambda: qres.ishow_matches(ibs, aid2, mode=0, qreq_=qreq_))
    if aid_list is not None:
        # Give a subcontext menu for multiple options
        def partial_show_chip_matches_to(aid_):
            return lambda: qres.ishow_matches(ibs, aid_, mode=0, qreq_=qreq_)
        show_chip_match_features_option = (
            'Show chip feature matches',
            [
                ('to aid=%r' % (aid_,), partial_show_chip_matches_to(aid_))
                for aid_ in aid_list
            ]
        )

    options = [
        show_chip_match_features_option,
        ('Show name feature matches', lambda: qres.show_name_matches(ibs, aid2, mode=0, qreq_=qreq_)),
        ('Inspect Match Candidates', lambda: review_match(ibs, aid1, aid2, qreq_=qreq_, qres=qres, **kwargs)),
        ('Mark as &Reviewed', lambda: ibs.mark_annot_pair_as_reviewed(aid1, aid2)),
        ('Mark as &True Match.', lambda: mark_annot_pair_as_positive_match_(ibs, aid1, aid2, qres, qreq_, **kwargs)),
        ('Mark as &False Match.', lambda:  mark_annot_pair_as_negative_match_(ibs, aid1, aid2, qres, qreq_, **kwargs)),
    ]

    from ibeis.viz.interact import interact_chip

    interact_chip_options = [
        ('Interact Chip&1', lambda: interact_chip.ishow_chip(ibs, aid1, config2_=qreq_.get_external_query_config2(), fnum=None, **kwargs)),
        ('Interact Chip&2', lambda: interact_chip.ishow_chip(ibs, aid2, config2_=qreq_.get_external_data_config2(), fnum=None, **kwargs)),
    ]
    interact_chip_actions = ut.get_list_column(interact_chip_options, 1)
    interact_chip_options.append(
        ('Interact &All Chips', lambda: [func() for func in interact_chip_actions]),
    )

    options += [
        ('Interact Chips', interact_chip_options),
    ]

    annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])[0]

    OLD = False
    if not OLD:
        if annotmatch_rowid is None:
            tags = []
        else:
            tags = ibs.get_annotmatch_case_tags([annotmatch_rowid])[0]
            tags = [_.lower() for _ in tags]
        standard, other = ibsfuncs.get_cate_categories()
        case_list = standard + other

        used_chars = find_used_chars(ut.get_list_column(options, 0))
        case_hotlink_list = make_word_hotlinks(case_list, used_chars)
        case_options = []
        if True or ut.VERBOSE:
            print('[inspect_gui] aid1, aid2 = %r, %r' % (aid1, aid2,))
            print('[inspect_gui] annotmatch_rowid = %r' % (annotmatch_rowid,))
            print('[inspect_gui] tags = %r' % (tags,))
        if ut.VERBOSE:
            print('[inspect_gui] Making case hotlist: ' + ut.list_str(case_hotlink_list))

        def _wrap_set_annotmatch_prop(prop, toggle_val):
            if ut.VERBOSE:
                print('[SETTING] Clicked set prop=%r to val=%r' % (prop, toggle_val,))
            am_rowid = ibs.add_annotmatch([aid1], [aid2])[0]
            if ut.VERBOSE:
                print('[SETTING] aid1, aid2 = %r, %r' % (aid1, aid2,))
                print('[SETTING] annotmatch_rowid = %r' % (am_rowid,))
            ibs.set_annotmatch_prop(prop, [am_rowid], [toggle_val])
            if ut.VERBOSE:
                print('[SETTING] done')

        for case, case_hotlink in zip(case_list, case_hotlink_list):
            toggle_val = case.lower() not in tags
            fmtstr = 'Flag %s case' if toggle_val else 'Unflag %s case'
            case_options += [
                #(fmtstr % (case_hotlink,), lambda: ibs.set_annotmatch_prop(case, _get_annotmatch_rowid(), [toggle_val])),
                #(fmtstr % (case_hotlink,), partial(ibs.set_annotmatch_prop, case, [annotmatch_rowid], [toggle_val])),
                (fmtstr % (case_hotlink,), partial(_wrap_set_annotmatch_prop, case, toggle_val)),
            ]
        if ut.VERBOSE:
            print('Partial tag funcs:' + ut.list_str([ut.func_str(func, func.args, func.keywords) for func in ut.get_list_column(case_options, 1)]))
        options += case_options
    elif OLD:
        pass
        #if annotmatch_rowid is None:
        #    if ut.VERBOSE:
        #        print('[inspect_gui] Explicit annotmatch options')
        #    options += [
        #        ('Flag &Scenery Case', lambda:  ibs.set_annotmatch_is_scenerymatch(ibs.add_annotmatch([aid1], [aid2]), [True])),
        #        ('Flag &Photobomb Case', lambda:  ibs.set_annotmatch_is_photobomb(ibs.add_annotmatch([aid1], [aid2]), [True])),
        #        ('Flag &Hard Case', lambda:  ibs.set_annotmatch_is_hard(ibs.add_annotmatch([aid1], [aid2]), [True])),
        #        ('Flag &NonDistinct Case', lambda:  ibs.set_annotmatch_is_nondistinct(ibs.add_annotmatch([aid1], [aid2]), [True])),
        #    ]
        #else:
        #    # If the match already exists allow untoggleing
        #    key_list = [
        #        'SceneryMatch',
        #        'Photobomb',
        #        'Hard',
        #        'NonDistinct',
        #    ]
        #    # VERY HACKY
        #    if ut.VERBOSE:
        #        print('[inspect_gui] Hacking the annotmatch options')
        #    for key in key_list:
        #        getter = getattr(ibs, 'get_annotmatch_is_' + key.lower())
        #        setter = getattr(ibs, 'set_annotmatch_is_' + key.lower())
        #        flag = getter([annotmatch_rowid])[0]
        #        if ut.VERBOSE:
        #            print('[inspect_gui] * getter = %r' % (getter,))
        #            print('[inspect_gui] * setter = %r' % (setter,))
        #            print('[inspect_gui] * flag = %r' % (flag,))
        #        if flag:
        #            options += [
        #                ('Remove &' + key + ' Flag', lambda:  setter([annotmatch_rowid], [False])),
        #            ]
        #        else:
        #            options += [
        #                ('Flag &' + key + ' Case', lambda:  setter([annotmatch_rowid], [True])),
        #            ]
        #    if ut.VERBOSE:
        #        print('[inspect_gui] options = %s' % (ut.list_str(options),))
    return options


def find_used_chars(name_list):
    """ Move to guitool """
    used_chars = []
    for name in name_list:
        index = name.find('&')
        if index == -1 or index + 1 >= len(name):
            continue
        char = name[index + 1]
        used_chars.append(char)
    return used_chars


def make_word_hotlinks(name_list, used_chars):
    """ Move to guitool """
    seen_ = set(used_chars)
    hotlinked_name_list = []
    for name in name_list:
        added = False
        for count, char in enumerate(name):
            char = char.upper()
            if char not in seen_:
                added = True
                seen_.add(char)
                linked_name = name[:count] + '&' + name[count:]
                hotlinked_name_list.append(linked_name)
                break
        if not added:
            # Cannot hotlink this name
            hotlinked_name_list.append(name)
    return hotlinked_name_list


def show_aidpair_context_menu(ibs, qwin, qpoint, aid1, aid2, qres, qreq_=None, **kwargs):
    """
    kwargs are used for callbacks like qres_callback and query_callback
    """
    options = get_aidpair_context_menu_options(ibs, aid1, aid2, qres, qreq_=qreq_, **kwargs)
    guitool.popup_menu(qwin, qpoint, options)


def mark_annot_pair_as_positive_match_(ibs, aid1, aid2, qres, qreq_, **kwargs):
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
        status = ibs.mark_annot_pair_as_positive_match(aid1, aid2, on_nontrivial_merge=on_nontrivial_merge)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match(ibs, aid1, aid2, qreq_=qreq_, qres=qres, **kwargs)
    except guiexcept.UserCancel:
        print('user canceled positive match')


def mark_annot_pair_as_negative_match_(ibs, aid1, aid2, qres, qreq_, **kwargs):
    def on_nontrivial_split(ibs, aid1, aid2):
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        print('There are %d annots in this name. Need more sophisticated split' % (len(aid1_groundtruth)))
        raise guiexcept.NeedsUserInput('non-trivial split')
    try:
        status = ibs.mark_annot_pair_as_negative_match(aid1, aid2, on_nontrivial_split=on_nontrivial_split)
        print('status = %r' % (status,))
    except guiexcept.NeedsUserInput:
        review_match(ibs, aid1, aid2, qreq_=qreq_, qres=qres, **kwargs)
    except guiexcept.UserCancel:
        print('user canceled negative match')


def review_match(ibs, aid1, aid2, update_callback=None, backend_callback=None, qreq_=None, qres=None, **kwargs):
    print('Review match: ' + ibsfuncs.vsstr(aid1, aid2))
    from ibeis.viz.interact import interact_name
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    mvinteract = interact_name.MatchVerificationInteraction(
        ibs, aid1, aid2, fnum=64, update_callback=update_callback,
        qres=qres,
        qreq_=qreq_,
        backend_callback=backend_callback, **kwargs)
    return mvinteract
    #ih.register_interaction(mvinteract)


#class CustomAPI(object):
#    """
#    Allows list of lists to be represented as an abstract api table

#    # TODO: Rename CustomAPI
#    API wrapper around a list of lists, each containing column data
#    Defines a single table
#    """
#    def __init__(self, col_name_list, col_types_dict, col_getter_dict,
#                 col_bgrole_dict, col_ider_dict, col_setter_dict,
#                 editable_colnames, sortby, get_thumb_size=None,
#                 sort_reverse=True, col_width_dict={}):
#        if ut.VERBOSE:
#            print('[CustomAPI] <__init__>')
#        self.col_width_dict = col_width_dict
#        self.col_name_list = []
#        self.col_type_list = []
#        self.col_getter_list = []
#        self.col_setter_list = []
#        self.nCols = 0
#        self.nRows = 0
#        if get_thumb_size is None:
#            self.get_thumb_size = lambda: 128
#        else:
#            self.get_thumb_size = get_thumb_size

#        self.parse_column_tuples(col_name_list, col_types_dict, col_getter_dict,
#                                 col_bgrole_dict, col_ider_dict, col_setter_dict,
#                                 editable_colnames, sortby, sort_reverse)
#        if ut.VERBOSE:
#            print('[CustomAPI] </__init__>')

#    def parse_column_tuples(self,
#                            col_name_list,
#                            col_types_dict,
#                            col_getter_dict,
#                            col_bgrole_dict,
#                            col_ider_dict,
#                            col_setter_dict,
#                            editable_colnames,
#                            sortby,
#                            sort_reverse=True):
#        """
#        parses simple lists into information suitable for making guitool headers
#        """
#        # Unpack the column tuples into names, getters, and types
#        self.col_name_list = col_name_list
#        self.col_type_list = [col_types_dict.get(colname, str) for colname in col_name_list]
#        self.col_getter_list = [col_getter_dict.get(colname, str) for colname in col_name_list]  # First col is always a getter
#        # Get number of rows / columns
#        self.nCols = len(self.col_getter_list)
#        self.nRows = 0 if self.nCols == 0 else len(self.col_getter_list[0])  # FIXME
#        # Init iders to default and then overwite based on dict inputs
#        self.col_ider_list = utool.alloc_nones(self.nCols)
#        for colname, ider_colnames in six.iteritems(col_ider_dict):
#            try:
#                col = self.col_name_list.index(colname)
#                # Col iders might have tuple input
#                ider_cols = utool.uinput_1to1(self.col_name_list.index, ider_colnames)
#                col_ider  = utool.uinput_1to1(lambda c: partial(self.get, c), ider_cols)
#                self.col_ider_list[col] = col_ider
#                del col_ider
#                del ider_cols
#                del col
#                del colname
#            except Exception as ex:
#                ut.printex(ex, keys=['colname', 'ider_colnames', 'col', 'col_ider', 'ider_cols'])
#                raise
#        # Init setters to data, and then overwrite based on dict inputs
#        self.col_setter_list = list(self.col_getter_list)
#        for colname, col_setter in six.iteritems(col_setter_dict):
#            col = self.col_name_list.index(colname)
#            self.col_setter_list[col] = col_setter
#        # Init bgrole_getters to None, and then overwrite based on dict inputs
#        self.col_bgrole_getter_list = [col_bgrole_dict.get(colname, None) for colname in self.col_name_list]
#        # Mark edtiable columns
#        self.col_edit_list = [name in editable_colnames for name in col_name_list]
#        # Mark the sort column index
#        if utool.is_str(sortby):
#            self.col_sort_index = self.col_name_list.index(sortby)
#        else:
#            self.col_sort_index = sortby
#        self.col_sort_reverse = sort_reverse

#    def _infer_index(self, column, row):
#        """
#        returns the row based on the columns iders.
#        This is the identity for the default ider
#        """
#        ider_ = self.col_ider_list[column]
#        if ider_ is None:
#            return row
#        iderfunc = lambda func_: func_(row)
#        return utool.uinput_1to1(iderfunc, ider_)

#    def get(self, column, row, **kwargs):
#        """
#        getters always receive primary rowids, rectify if col_ider is
#        specified (row might be a row_pair)
#        """
#        index = self._infer_index(column, row)
#        column_getter = self.col_getter_list[column]
#        # Columns might be getter funcs indexable read/write arrays
#        try:
#            return utool.general_get(column_getter, index, **kwargs)
#        except Exception:
#            # FIXME: There may be an issue on tuple-key getters when row input is
#            # vectorized. Hack it away
#            if utool.isiterable(row):
#                row_list = row
#                return [self.get(column, row_, **kwargs) for row_ in row_list]
#            else:
#                raise

#    def set(self, column, row, val):
#        index = self._infer_index(column, row)
#        column_setter = self.col_setter_list[column]
#        # Columns might be setter funcs or indexable read/write arrays
#        utool.general_set(column_setter, index, val)

#    def get_bgrole(self, column, row):
#        bgrole_getter = self.col_bgrole_getter_list[column]
#        if bgrole_getter is None:
#            return None
#        index = self._infer_index(column, row)
#        return utool.general_get(bgrole_getter, index)

#    def ider(self):
#        return list(range(self.nRows))

#    def make_headers(self, tblname='qres_api', tblnice='Query Results'):
#        """
#        Builds headers for APIItemModel
#        """
#        headers = {
#            'name': tblname,
#            'nice': tblname if tblnice is None else tblnice,
#            'iders': [self.ider],
#            'col_name_list'    : self.col_name_list,
#            'col_type_list'    : self.col_type_list,
#            'col_nice_list'    : self.col_name_list,
#            'col_edit_list'    : self.col_edit_list,
#            'col_sort_index'   : self.col_sort_index,
#            'col_sort_reverse' : self.col_sort_reverse,
#            'col_getter_list'  : self._make_getter_list(),
#            'col_setter_list'  : self._make_setter_list(),
#            'col_setter_list'  : self._make_setter_list(),
#            'col_bgrole_getter_list' : self._make_bgrole_getter_list(),
#            'get_thumb_size'   : self.get_thumb_size,
#        }
#        return headers

#    def _make_bgrole_getter_list(self):
#        return [partial(self.get_bgrole, column) for column in range(self.nCols)]

#    def _make_getter_list(self):
#        return [partial(self.get, column) for column in range(self.nCols)]

#    def _make_setter_list(self):
#        return [partial(self.set, column) for column in range(self.nCols)]


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
    truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=lighten_amount)
    #truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    #truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


#def get_buttontup(ibs, qtindex):
#    """
#    helper for make_qres_api
#    """
#    model = qtindex.model()
#    aid1 = model.get_header_data('qaid', qtindex)
#    aid2 = model.get_header_data('aid', qtindex)
#    truth = ibs.get_match_truth(aid1, aid2)
#    truth_color = vh.get_truth_color(truth, base255=True,
#                                        lighten_amount=0.35)
#    truth_text = ibs.get_match_text(aid1, aid2)
#    callback = partial(review_match, ibs, aid1, aid2)
#    #print('get_button, aid1=%r, aid2=%r, row=%r, truth=%r' % (aid1, aid2, row, truth))
#    buttontup = (truth_text, callback, truth_color)
#    return buttontup


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
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:5]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> if not ut.get_argflag('--nodelete'):
        >>>     ibs.delete_annotmatch(ibs._get_all_annotmatch_rowids())
        >>> main_locals = test_inspect_matches(ibs, qaid_list, daid_list)
        >>> main_execstr = ibeis.main_loop(main_locals)
        >>> ut.quit_if_noshow()
        >>> # TODO: add in qwin to main loop
        >>> guitool.qtapp_loop()
        >>> print(main_execstr)
        >>> exec(main_execstr)
    """
    from ibeis.viz.interact import interact_qres2  # NOQA
    from ibeis.gui import inspect_gui
    from ibeis.experiments import results_all
    allres = results_all.get_allres(ibs, qaid_list, cfgdict={'augment_queryside_hack': True})
    tblname = 'qres'
    qreq_ = allres.qreq_
    qaid2_qres = allres.qaid2_qres
    name_scoring = False
    ranks_lt = 5
    # This object is created inside QresResultsWidget
    #qres_api = inspect_gui.make_qres_api(ibs, qaid2_qres)  # NOQA
    # This is where you create the result widigt
    guitool.ensure_qapp()
    print('[inspect_matches] make_qres_widget')
    #qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, ranks_lt=ranks_lt, qreq_=qreq_)
    #ut.view_directory(ibs.get_match_thumbdir())
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres,
                                              ranks_lt=ranks_lt, qreq_=qreq_,
                                              filter_reviewed=False,
                                              filter_duplicate_namepair_matches=True)
    print('[inspect_matches] show')
    qres_wgt.show()
    print('[inspect_matches] raise')
    qres_wgt.raise_()
    #query_review = interact_qres2.Interact_QueryResult(ibs, qaid2_qres)
    #self = interact_qres2.Interact_QueryResult(ibs, qaid2_qres, ranks_lt=ranks_lt)
    print('</inspect_matches>')
    # simulate double click
    #qres_wgt._on_click(qres_wgt.model.index(2, 2))
    #qres_wgt._on_doubleclick(qres_wgt.model.index(2, 0))
    locals_ =  locals()
    return locals_


def get_match_thumb_fname(qres, daid):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import results_all
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:2]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> allres = results_all.get_allres(ibs, qaid_list)
        >>> thumbsize = (128, 128)
        >>> qreq_ = None
        >>> qres = allres.qaid2_qres[qaid_list[0]]
        >>> daid = daid_list[0]
        >>> match_thumb_fname = get_match_thumb_fname(qres, daid)
        >>> result = match_thumb_fname
        >>> print(result)
        match_aids=1,1_cfgstr=ubpzwu5k54h6xbnr.jpg
    """
    # Make thumbnail name
    config_hash = ut.hashstr(qres.cfgstr)
    qaid = qres.qaid
    match_thumb_fname = 'match_aids=%d,%d_cfgstr=%s.jpg' % ((qaid, daid, config_hash))
    return match_thumb_fname


def ensure_match_img(ibs, qres, daid, qreq_=None, match_thumbtup_cache={}):
    r"""
    CommandLine:
        python -m ibeis.gui.inspect_gui --test-ensure_match_img

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> species = ibeis.const.Species.ZEB_PLAIN
        >>> daids = ibs.get_valid_aids(species=species)
        >>> qaids = ibs.get_valid_aids(species=species)
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
        >>> daid = qaids[0]
        >>> qreq_ = None
        >>> match_thumbtup_cache = {}
        >>> # execute function
        >>> match_thumb_fpath_ = ensure_match_img(ibs, qres, daid, qreq_, match_thumbtup_cache)
        >>> # verify results
        >>> result = str(match_thumb_fpath_)
        >>> print(result)
        >>> ut.quit_if_noshow():
        >>> ut.startfile(match_thumb_fpath_, quote=True)
    """
    #from os.path import exists
    match_thumbdir = ibs.get_match_thumbdir()
    match_thumb_fname = get_match_thumb_fname(qres, daid)
    match_thumb_fpath_ = ut.unixjoin(match_thumbdir, match_thumb_fname)
    #if exists(match_thumb_fpath_):
    #    return match_thumb_fpath_
    if match_thumb_fpath_ in match_thumbtup_cache:
        fpath = match_thumbtup_cache[match_thumb_fpath_]
    else:
        # TODO: just draw the image at the correct thumbnail size
        # TODO: draw without matplotlib?
        fpath = qres.dump_match_img(
            ibs, daid, fpath=match_thumb_fpath_, saveax=True, fnum=32,
            notitle=True, verbose=False, qreq_=qreq_)
        match_thumbtup_cache[match_thumb_fpath_] = fpath
    return fpath


def get_match_thumbtup(ibs, qaid2_qres, qaids, daids, index, qreq_=None, thumbsize=(128, 128), match_thumbtup_cache={}):
    qaid, daid = qaids[index], daids[index]
    qres = qaid2_qres[qaid]
    fpath = ensure_match_img(ibs, qres, daid, qreq_=qreq_, match_thumbtup_cache=match_thumbtup_cache)
    if isinstance(thumbsize, int):
        thumbsize = (thumbsize, thumbsize)
    thumbtup = (ut.augpath(fpath, 'thumb_%d,%d' % thumbsize), fpath, thumbsize, [], [])
    return thumbtup


def make_qres_api(ibs, qaid2_qres, ranks_lt=None, name_scoring=False,
                  filter_reviewed=None,
                  filter_duplicate_namepair_matches=False,
                  qreq_=None,
                  ):
    """
    Builds columns which are displayable in a ColumnListTableWidget

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_inspect_matches --show
        python -m ibeis.gui.inspect_gui --test-make_qres_api

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> import guitool
        >>> from ibeis.viz.interact import interact_qres2  # NOQA
        >>> from ibeis.gui import inspect_gui
        >>> from ibeis.experiments import results_all
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:2]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> allres = results_all.get_allres(ibs, qaid_list)
        >>> tblname = 'qres'
        >>> qaid2_qres = allres.qaid2_qres
        >>> name_scoring = False
        >>> ranks_lt = 5
        >>> make_qres_api(ibs, qaid2_qres, ranks_lt, name_scoring)

    """
    if ut.VERBOSE:
        print('[inspect] make_qres_api')
    ibs.cfg.other_cfg.ranks_lt = 2
    if filter_reviewed is None:
        # only filter big queries if not specified
        filter_reviewed = len(qaid2_qres) > 6
    ranks_lt = ranks_lt if ranks_lt is not None else ibs.cfg.other_cfg.ranks_lt
    candidate_matches = results_organizer.get_automatch_candidates(
        qaid2_qres, ranks_lt=ranks_lt, name_scoring=name_scoring, ibs=ibs,
        directed=False, filter_reviewed=filter_reviewed,
        filter_duplicate_namepair_matches=filter_duplicate_namepair_matches
    )
    # Get extra info
    (qaids, daids, scores, ranks) = candidate_matches

    #opts = np.zeros(len(qaids))
    # Define column information

    # TODO: MAKE A PAIR IDER AND JUST USE EXISTING API_ITEM_MODEL FUNCTIONALITY
    # TO GET THOSE PAIRWISE INDEXES

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
    #if ut.is_developer():
    #    pass
    #    col_name_list.insert(2, 'd_nGt')
    #    col_name_list.insert(2, 'q_nGt')
    #    #col_name_list.insert(2, 'q_nGt')

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
        col_name_list.insert(col_name_list.index(RES_THUMB_TEXT) + 1, MATCH_THUMB_TEXT)
        col_types_dict[MATCH_THUMB_TEXT] = 'PIXMAP'
        get_match_thumbtup_ = partial(get_match_thumbtup, ibs, qaid2_qres,
                                      qaids, daids, qreq_=qreq_,
                                      match_thumbtup_cache={})
        col_getter_dict[MATCH_THUMB_TEXT] = get_match_thumbtup_

    #get_status_bgrole_func = partial(get_match_status_bgrole, ibs)
    col_bgrole_dict = {
        MATCHED_STATUS_TEXT : partial(get_match_status_bgrole, ibs),
        REVIEWED_STATUS_TEXT: partial(get_reviewed_status_bgrole, ibs),
        #'aid'    : get_status_bgrole_func,
        #'qaid'   : get_status_bgrole_func,
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

    USE_BOOLS = True
    if USE_BOOLS:
        boolean_annotmatch_columns = [
            'is_hard',
            'is_nondistinct',
            'is_scenerymatch',
            'is_photobomb',
        ]

        boolean_annot_columns = [
            'is_occluded',
            'is_shadowed',
            'is_washedout',
            'is_blury',
            'is_novelpose',
            'is_commonpose',
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
            col_getter_dict[colname] = make_annotmatch_boolean_getter_wrapper(ibs, colname)
            col_setter_dict[colname] = make_annotmatch_boolean_setter_wrapper(ibs, colname)
            col_width_dict[colname] = 70
            editable_colnames.append(colname)

        for colname_ in boolean_annot_columns:
            # TODO
            pass

    sortby = 'score'
    get_thumb_size = lambda: ibs.cfg.other_cfg.thumb_size
    # Insert info into dict
    qres_api = guitool.CustomAPI(col_name_list, col_types_dict, col_getter_dict,
                                 col_bgrole_dict, col_ider_dict, col_setter_dict,
                                 editable_colnames, sortby, get_thumb_size, True, col_width_dict)
    return qres_api


def launch_review_matches_interface(ibs, qres_list, dodraw=False):
    """ TODO: move to a more general function """
    from ibeis.gui import inspect_gui
    import guitool
    guitool.ensure_qapp()
    #backend_callback = back.front.update_tables
    backend_callback = None
    qaid2_qres = {qres.qaid: qres for qres in qres_list}
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, callback=backend_callback)
    if dodraw:
        qres_wgt.show()
        qres_wgt.raise_()
    return qres_wgt


def inspect_orphaned_qres_bigcache(ibs, bc_fpath, cfgdict={}):
    """
    Hack to try and grab the last big query

    import ibeis
    ibs = ibeis.opendb('PZ_Master0')
    fname = 'PZ_Master0_QRESMAP_QSUUIDS((187)85k!tqcpgtb8k%rj)_DSUUIDS((200)4%t0tenxktstb676)2m8fp@nto!s@@0f+_a@2duauqcb4r18g7.cPkl'
    bc_dpath = ibs.get_big_cachedir()
    from os.path import join
    bc_fpath = join(bc_dpath, fname)

    import os
    bc_dpath = ibs.get_big_cachedir()
    fpath_list = ut.ls(bc_dpath)
    ctime_list = list(map(os.path.getctime, fpath_list))
    sorted_fpath_list = ut.sortedby(fpath_list, ctime_list, reverse=True)
    bc_fpath = sorted_fpath_list[0]

    cfgdict = dict(
        can_match_samename=False, use_k_padding=False, affine_invariance=False,
        scale_max=150, augment_queryside_hack=True)

    """
    qaid2_qres = ut.load_cPkl(bc_fpath)
    qaid_list = list(qaid2_qres.keys())
    qres = qaid2_qres[qaid_list[0]]
    daid_list = qres.daids
    #for qres in six.itervalues(qaid2_qres):
    #    assert np.all(daid_list == qres.daids)
    qreq_ = ibs.new_query_request(qaid_list, daid_list, cfgdict=cfgdict)

    true_cfgstr = qres.cfgstr
    guess_cfgstr = qreq_.get_cfgstr()

    true_cfgstr_ = '\n'.join(true_cfgstr.split('_'))
    guess_cfgstr_ = '\n'.join(guess_cfgstr.split('_'))
    textdiff = (ut.get_textdiff(true_cfgstr_, guess_cfgstr_))
    print(textdiff)
    if len(textdiff) > 0:
        raise Exception('you may need to fix the configstr')

    from ibeis.viz.interact import interact_qres2  # NOQA
    from ibeis.gui import inspect_gui
    guitool.ensure_qapp()
    ranks_lt = 1
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres,
                                              ranks_lt=ranks_lt, qreq_=qreq_,
                                              filter_reviewed=True,
                                              filter_duplicate_namepair_matches=True,
                                              query_title='Recovery Hack')
    qres_wgt.show()
    qres_wgt.raise_()


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
