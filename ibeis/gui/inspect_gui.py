from __future__ import absolute_import, division, print_function
import utool
import functools  # NOQA
from functools import partial
from ibeis.viz import interact
from plottool import interact_helpers as ih
from ibeis.dev import results_organizer
from plottool import fig_presenter
from guitool import qtype, APIItemWidget, APIItemModel, FilterProxyModel, ChangeLayoutContext
from PyQt4 import QtCore, QtGui
import guitool
from ibeis.dev import ibsfuncs
import numpy as np
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[inspect_gui]', DEBUG=False)


USE_FILTER_PROXY = True


class QueryResultsWidget(APIItemWidget):
    """ Window for gui inspection """

    def __init__(qres_wgt, ibs, qaid2_qres, parent=None, callback=None, **kwargs):
        print('[qres_wgt] Init QueryResultsWidget')
        # Uncomment below to turn on FilterProxyModel
        if USE_FILTER_PROXY:
            APIItemWidget.__init__(qres_wgt, parent=parent,
                                    model_class=CustomFilterModel)
        else:
            APIItemWidget.__init__(qres_wgt, parent=parent)
        qres_wgt.show_new = True
        qres_wgt.show_join = True
        qres_wgt.show_split = True
        qres_wgt.tt = utool.tic()
        # Set results data
        qres_wgt.add_checkboxes(qres_wgt.show_new, qres_wgt.show_join, qres_wgt.show_split)
        qres_wgt.set_query_results(ibs, qaid2_qres, **kwargs)
        qres_wgt.connect_signals_and_slots()
        if callback is None:
            callback = lambda: None
        qres_wgt.callback = callback
        qres_wgt.view.setColumnHidden(0, False)
        qres_wgt.view.setColumnHidden(1, False)
        if parent is None:
            # Register parentless QWidgets
            fig_presenter.register_qt4_win(qres_wgt)

    def add_checkboxes(qres_wgt, show_new, show_join, show_split):
        _CHECK  = functools.partial(guitool.newCheckBox, qres_wgt)
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

    def set_query_results(qres_wgt, ibs, qaid2_qres, **kwargs):
        print('[qres_wgt] Change QueryResultsWidget data')
        qres_wgt.ibs = ibs
        qres_wgt.qaid2_qres = qaid2_qres
        qres_wgt.qres_api = make_qres_api(ibs, qaid2_qres, **kwargs)
        qres_wgt.update_checkboxes()
        headers = qres_wgt.qres_api.make_headers()
        APIItemWidget.change_headers(qres_wgt, headers)

    def connect_signals_and_slots(qres_wgt):
        qres_wgt.view.clicked.connect(qres_wgt._on_click)
        qres_wgt.view.doubleClicked.connect(qres_wgt._on_doubleclick)
        qres_wgt.view.pressed.connect(qres_wgt._on_pressed)
        qres_wgt.view.activated.connect(qres_wgt._on_activated)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_click(iqrw, qtindex):
        print('[qres_wgt] _on_click: ')
        print('[qres_wgt] _on_click: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)
        if colname == 'status':
            qres_callback = partial(show_match_at, iqrw, qtindex)
            review_match_at(iqrw, qtindex, quickmerge=False, qres_callback=qres_callback)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(iqrw, qtindex):
        print('[qres_wgt] _on_doubleclick: ')
        print('[qres_wgt] DoubleClicked: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)
        if colname != 'status':
            return show_match_at(iqrw, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(iqrw, qtindex):
        print('[qres_wgt] _on_pressed: ')
        def _check_for_double_click(iqrw, qtindex):
            threshold = 0.50  # seconds
            distance = utool.toc(iqrw.tt)
            print('Pressed %r' % (distance,))
            col = qtindex.column()
            model = qtindex.model()
            colname = model.get_header_name(col)
            if distance <= threshold:
                if colname == 'status':
                    iqrw.view.clicked.emit(qtindex)
                    iqrw._on_click(qtindex)
                else:
                    #iqrw.view.doubleClicked.emit(qtindex)
                    iqrw._on_doubleclick(qtindex)
            iqrw.tt = utool.tic()
        _check_for_double_click(iqrw, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_activated(iqrw, qtindex):
        print('Activated: ' + str(qtype.qindexinfo(qtindex)))
        pass

    @guitool.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(iqrw, qtindex, qpos):
        printDBG('[newgui] contextmenu')
        guitool.popup_menu(iqrw, qpos, [
            ('view match annotations', lambda: show_match_at(iqrw, qtindex)),
            ('review match', lambda: review_match_at(iqrw, qtindex)),
        ])


def show_match_at(qres_wgt, qtindex):
    print('interact')
    model = qtindex.model()
    aid  = model.get_header_data('aid', qtindex)
    qaid = model.get_header_data('qaid', qtindex)
    fig = interact.ishow_matches(qres_wgt.ibs, qres_wgt.qaid2_qres[qaid], aid, mode=1)
    fig_presenter.bring_to_front(fig)


def review_match_at(qres_wgt, qtindex, quickmerge=False, **kwargs):
    print('review')
    ibs = qres_wgt.ibs
    model = qtindex.model()
    aid1 = model.get_header_data('qaid', qtindex)
    aid2 = model.get_header_data('aid', qtindex)
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    model = qtindex.model()
    #update_callback = model._update
    update_callback = None  # hack (checking if necessary)
    backend_callback = qres_wgt.callback
    if quickmerge:
        is_unknown = ibs.is_aid_unknown((aid1, aid2))
        if all(is_unknown):
            ibs.set_annot_names_to_next_name((aid1, aid2))
            update_callback()
            backend_callback()
            return
        elif is_unknown[0]:
            ibs.set_annot_nids(aid1, ibs.get_annot_nids(aid2))
            update_callback()
            backend_callback()
            return
        elif is_unknown[1]:
            ibs.set_annot_nids(aid2, ibs.get_annot_nids(aid1))
            update_callback()
            backend_callback()
            return
    review_match(ibs, aid1, aid2, update_callback=update_callback,
                 backend_callback=backend_callback, **kwargs)


def review_match(ibs, aid1, aid2, update_callback=None, backend_callback=None, **kwargs):
    print('Review match: ' + ibsfuncs.vsstr(aid1, aid2))
    from ibeis.viz.interact.interact_name import MatchVerificationInteraction
    #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
    mvinteract = MatchVerificationInteraction(ibs, aid1, aid2, fnum=64,
                                              update_callback=update_callback,
                                              backend_callback=backend_callback, **kwargs)
    ih.register_interaction(mvinteract)


class CustomFilterModel(FilterProxyModel):
    def __init__(model, headers=None, parent=None, *args):
        FilterProxyModel.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.eid = -1  # negative one is an invalid eid
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
        with ChangeLayoutContext([model]):
            FilterProxyModel._update_rows(model)


class CustomAPI(object):
    """ # TODO: Rename CustomAPI
    API wrapper around a list of lists, each containing column data
    Defines a single table """
    def __init__(self, col_name_list, col_types_dict, col_getters_dict,
                 col_bgrole_dict, col_ider_dict, col_setter_dict,
                 editable_colnames, sortby, sort_reverse=True):
        print('[CustomAPI] <__init__>')
        self.col_name_list = []
        self.col_type_list = []
        self.col_getter_list = []
        self.col_setter_list = []
        self.nCols = 0
        self.nRows = 0
        self.parse_column_tuples(col_name_list, col_types_dict, col_getters_dict,
                                 col_bgrole_dict, col_ider_dict, col_setter_dict,
                                 editable_colnames, sortby, sort_reverse)
        print('[CustomAPI] </__init__>')

    def parse_column_tuples(self, col_name_list, col_types_dict, col_getters_dict,
                            col_bgrole_dict, col_ider_dict, col_setter_dict,
                            editable_colnames, sortby, sort_reverse=True):
        # Unpack the column tuples into names, getters, and types
        self.col_name_list = col_name_list
        self.col_type_list = [col_types_dict.get(colname, str) for colname in col_name_list]
        self.col_getter_list = [col_getters_dict.get(colname, str) for colname in col_name_list]  # First col is always a getter
        # Get number of rows / columns
        self.nCols = len(self.col_getter_list)
        self.nRows = 0 if self.nCols == 0 else len(self.col_getter_list[0])  # FIXME
        # Init iders to default and then overwite based on dict inputs
        self.col_ider_list = utool.alloc_nones(self.nCols)
        for colname, ider_colnames in col_ider_dict.iteritems():
            col = self.col_name_list.index(colname)
            # Col iders might have tuple input
            ider_cols = utool.uinput_1to1(self.col_name_list.index, ider_colnames)
            col_ider  = utool.uinput_1to1(lambda c: partial(self.get, c), ider_cols)
            self.col_ider_list[col] = col_ider
        # Init setters to data, and then overwrite based on dict inputs
        self.col_setter_list = list(self.col_getter_list)
        for colname, col_setter in col_setter_dict.iteritems():
            col = self.col_name_list.index(colname)
            self.col_setter_list[col] = col_setter
        # Init bgrole_getters to None, and then overwrite based on dict inputs
        self.col_bgrole_getter_list = [col_bgrole_dict.get(colname, None) for colname in self.col_name_list]
        # Mark edtiable columns
        self.col_edit_list = [name in editable_colnames for name in col_name_list]
        # Mark the sort column index
        if utool.is_str(sortby):
            self.col_sort_index = self.col_name_list.index(sortby)
        else:
            self.col_sort_index = sortby
        self.col_sort_reverse = sort_reverse

    def _infer_index(self, column, row):
        """ returns the row based on the columns iders.
        This is the identity for the default ider """
        ider_ = self.col_ider_list[column]
        if ider_ is None:
            return row
        return utool.uinput_1to1(lambda func: func(row), ider_)

    def get(self, column, row):
        """ getters always receive primary rowids, rectify if col_ider is
        specified (row might be a row_pair) """
        index = self._infer_index(column, row)
        column_getter = self.col_getter_list[column]
        # Columns might be getter funcs indexable read/write arrays
        try:
            return utool.general_get(column_getter, index)
        except Exception:
            # FIXME: There may be an issue on tuple-key getters when row input is
            # vectorized. Hack it away
            if utool.isiterable(row):
                row_list = row
                return [self.get(column, row_) for row_ in row_list]
            else:
                raise

    def set(self, column, row, val):
        index = self._infer_index(column, row)
        column_setter = self.col_setter_list[column]
        # Columns might be setter funcs or indexable read/write arrays
        utool.general_set(column_setter, index, val)

    def get_bgrole(self, column, row):
        bgrole_getter = self.col_bgrole_getter_list[column]
        if bgrole_getter is None:
            return None
        index = self._infer_index(column, row)
        return utool.general_get(bgrole_getter, index)

    def ider(self):
        return range(self.nRows)

    def make_headers(self, tblname='qres_api', tblnice='Query Results'):
        """ Builds headers for APIItemModel """
        headers = {
            'name': tblname,
            'nice': tblname if tblnice is None else tblnice,
            'iders': [self.ider],
            'col_name_list'    : self.col_name_list,
            'col_type_list'    : self.col_type_list,
            'col_nice_list'    : self.col_name_list,
            'col_edit_list'    : self.col_edit_list,
            'col_sort_index'   : self.col_sort_index,
            'col_sort_reverse' : self.col_sort_reverse,
            'col_getter_list'  : self._make_getter_list(),
            'col_setter_list'  : self._make_setter_list(),
            'col_setter_list'  : self._make_setter_list(),
            'col_bgrole_getter_list' : self._make_bgrole_getter_list()
        }
        return headers

    def _make_bgrole_getter_list(self):
        return [partial(self.get_bgrole, column) for column in xrange(self.nCols)]

    def _make_getter_list(self):
        return [partial(self.get, column) for column in xrange(self.nCols)]

    def _make_setter_list(self):
        return [partial(self.set, column) for column in xrange(self.nCols)]


def get_status(ibs, aid_pair):
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


def get_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    truth_color = vh.get_truth_color(truth, base255=True,
                                        lighten_amount=0.35)
    return truth_color


def get_buttontup(ibs, qtindex):
    model = qtindex.model()
    aid1 = model.get_header_data('qaid', qtindex)
    aid2 = model.get_header_data('aid', qtindex)
    truth = ibs.get_match_truth(aid1, aid2)
    truth_color = vh.get_truth_color(truth, base255=True,
                                        lighten_amount=0.35)
    truth_text = ibs.get_match_text(aid1, aid2)
    callback = partial(review_match, ibs, aid1, aid2)
    #print('get_button, aid1=%r, aid2=%r, row=%r, truth=%r' % (aid1, aid2, row, truth))
    buttontup = (truth_text, callback, truth_color)
    return buttontup


def make_qres_api(ibs, qaid2_qres, ranks_lt=None):
    """
    Builds columns which are displayable in a ColumnListTableWidget
    """
    print('[inspect] make_qres_api')
    if not hasattr(ibs.cfg.other_cfg, 'ranks_lt'):
        ibs.cfg.other_cfg.ranks_lt = 2
    ranks_lt = ranks_lt if ranks_lt is not None else ibs.cfg.other_cfg.ranks_lt
    candidate_matches = results_organizer.get_automatch_candidates(
        qaid2_qres, ranks_lt=ranks_lt)
    # Get extra info
    (qaids, aids, scores, ranks) = candidate_matches
    # assert all([isinstance(qaid, np.int32) for qaid in qaids]), "ERROR not all qaids are ints: %r" % (qaids)
    # assert all([isinstance(aid, np.int32)  for aid  in aids]), "ERROR not all aids are ints: %r" % (aids)
    #qnames = ibs.get_annot_names(qaids)
    #names = ibs.get_annot_names(aids)
    #truths = np.array((ibs.get_annot_lblannot_rowids_oftype(qaids) - ibs.get_annot_lblannot_rowids_oftype(aids)) == 0)
    #buttons = [get_review_match_buttontup(aid1, aid2) for (aid1, aid2) in izip(qaids, aids)]

    #def get_review_match_buttontup(aid1, aid2):
    #    """ A buttontup is a string and a callback """
    #    return get_button  # ('Merge', partial(review_match, aid1, aid2))

    def get_rowid_button(rowid):
        return get_buttontup
    #opts = np.zeros(len(qaids))
    # Define column information

    # TODO: MAKE A PAIR IDER AND JUST USE EXISTING API_ITEM_MODEL FUNCTIONALITY
    # TO GET THOSE PAIRWISE INDEXES

    col_name_list = ['qaid', 'aid', 'status', 'querythumb', 'resthumb', 'qname',
                     'name', 'score', 'rank', ]

    col_types_dict = dict([
        ('qaid',       int),
        ('aid',        int),
        ('review',     'BUTTON'),
        ('status',     str),
        ('querythumb', 'PIXMAP'),
        ('resthumb',   'PIXMAP'),
        ('qname',      str),
        ('name',       str),
        ('score',      float),
        ('rank',       int),
        ('truth',      bool),
        ('opt',        int),
    ])

    col_getters_dict = dict([
        ('qaid',       np.array(qaids)),
        ('aid',        np.array(aids)),
        ('review',     get_rowid_button),
        ('status',     partial(get_status, ibs)),
        ('querythumb', ibs.get_annot_chip_thumbtup),
        ('resthumb',   ibs.get_annot_chip_thumbtup),
        ('qname',      ibs.get_annot_names),
        ('name',       ibs.get_annot_names),
        ('score',      np.array(scores)),
        ('rank',       np.array(ranks)),
        #('truth',     truths),
        #('opt',       opts),
    ])

    col_bgrole_dict = {
        'status': partial(get_status_bgrole, ibs),
    }
    col_ider_dict = {
        'status'     : ('qaid', 'aid'),
        'querythumb' : ('qaid'),
        'resthumb'   : ('aid'),
        'qname'      : ('qaid'),
        'name'       : ('aid'),
    }
    col_setter_dict = {
        'qname': ibs.set_annot_names,
        'name': ibs.set_annot_names
    }
    editable_colnames =  ['truth', 'notes', 'qname', 'name', 'opt']
    sortby = 'score'
    # Insert info into dict
    qres_api = CustomAPI(col_name_list, col_types_dict, col_getters_dict,
                         col_bgrole_dict, col_ider_dict, col_setter_dict,
                         editable_colnames, sortby)
    return qres_api
