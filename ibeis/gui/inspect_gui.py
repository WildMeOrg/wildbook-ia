from __future__ import absolute_import, division, print_function
import utool
from functools import partial
#from itertools import izip
#utool.rrrr()
from ibeis.viz import interact
from plottool import interact_helpers as ih
from ibeis.dev import results_organizer
from plottool import fig_presenter
from guitool import qtype, APITableWidget
from PyQt4 import QtCore
import guitool
from ibeis.dev import ibsfuncs
import numpy as np
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[inspect_gui]', DEBUG=False)


RANKS_LT = 2


class QueryResultsWidget(APITableWidget):
    """ Window for gui inspection """

    def __init__(qres_wgt, ibs, qrid2_qres, parent=None, callback=None, **kwargs):
        print('[qres_wgt] Init QueryResultsWidget')
        APITableWidget.__init__(qres_wgt, parent=parent)
        # Set results data
        qres_wgt.set_query_results(ibs, qrid2_qres, **kwargs)
        qres_wgt.connect_signals_and_slots()
        if callback is None:
            callback = lambda: None
        qres_wgt.callback = callback
        print("callback=%r" % callback)
        if parent is None:
            # Register parentless QWidgets
            fig_presenter.register_qt4_win(qres_wgt)

    def set_query_results(qres_wgt, ibs, qrid2_qres, **kwargs):
        print('[qres_wgt] Change QueryResultsWidget data')
        qres_wgt.ibs = ibs
        qres_wgt.qrid2_qres = qrid2_qres
        qres_wgt.qres_api = make_qres_api(ibs, qrid2_qres, **kwargs)
        headers = qres_wgt.qres_api.make_headers()
        APITableWidget.change_headers(qres_wgt, headers)

    def connect_signals_and_slots(qres_wgt):
        qres_wgt.view.clicked.connect(qres_wgt._on_click)
        qres_wgt.view.doubleClicked.connect(qres_wgt._on_doubleclick)
        qres_wgt.view.pressed.connect(qres_wgt._on_pressed)
        qres_wgt.view.activated.connect(qres_wgt._on_activated)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_click(iqrw, qtindex):
        print('Clicked: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)
        if colname == 'status':
            review_match_at(iqrw, qtindex, quickmerge=False)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(iqrw, qtindex):
        print('DoubleClicked: ' + str(qtype.qindexinfo(qtindex)))
        col = qtindex.column()
        model = qtindex.model()
        colname = model.get_header_name(col)
        if colname != 'status':
            return show_match_at(iqrw, qtindex)
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(iqrw, qtindex):
        #print('Pressed: ' + str(qtype.qindexinfo(qtindex)))
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_activated(iqrw, qtindex):
        print('Activated: ' + str(qtype.qindexinfo(qtindex)))
        pass

    @guitool.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuRequested(iqrw, qtindex, qpos):
        printDBG('[newgui] contextmenu')
        guitool.popup_menu(iqrw, qpos, [
            ('view match rois', lambda: show_match_at(iqrw, qtindex)),
            ('review match', lambda: review_match_at(iqrw, qtindex)),
        ])


def show_match_at(qres_wgt, qtindex):
    print('interact')
    model = qtindex.model()
    row = qtindex.row()
    rid  = model.get_header_data('rid', row)
    qrid = model.get_header_data('qrid', row)
    fig = interact.ishow_matches(qres_wgt.ibs, qres_wgt.qrid2_qres[qrid], rid)
    fig_presenter.bring_to_front(fig)


def review_match_at(qres_wgt, qtindex, quickmerge=False):
    print('review')
    ibs = qres_wgt.ibs
    model = qtindex.model()
    row = qtindex.row()
    rid1 = model.get_header_data('qrid', row)
    rid2 = model.get_header_data('rid', row)
    model = qtindex.model()
    update_callback = model._update
    backend_callback = qres_wgt.callback
    print("review_match_at backend_callback=%r" % qres_wgt.callback)
    if quickmerge:
        is_unknown = ibs.is_rid_unknown((rid1, rid2))
        if all(is_unknown):
            ibs.set_roi_names_to_next_name((rid1, rid2))
            update_callback()
            backend_callback()
            return
        elif is_unknown[0]:
            ibs.set_roi_nids(rid1, ibs.get_roi_nids(rid2))
            update_callback()
            backend_callback()
            return
        elif is_unknown[1]:
            ibs.set_roi_nids(rid2, ibs.get_roi_nids(rid1))
            update_callback()
            backend_callback()
            return
    review_match(ibs, rid1, rid2, update_callback=update_callback, 
                 backend_callback=backend_callback)


def review_match(ibs, rid1, rid2, update_callback=None, backend_callback=None):
    print('Review match: ' + ibsfuncs.vsstr(rid1, rid2))
    from ibeis.viz.interact.interact_name import MatchVerificationInteraction
    mvinteract = MatchVerificationInteraction(ibs, rid1, rid2, fnum=64,
                                              update_callback=update_callback,
                                              backend_callback=backend_callback)
    ih.register_interaction(mvinteract)


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


def get_status(ibs, rid_pair):
    """ Data role for status column
    FIXME: no other function in this project takes a tuple of scalars as an
    argument. Everything else is written in the context of lists, This function
    should follow the same paradigm, but CustomAPI will have to change.
    """
    rid1, rid2 = rid_pair
    assert not utool.isiterable(rid1), 'rid1=%r, rid2=%r' % (rid1, rid2)
    assert not utool.isiterable(rid2), 'rid1=%r, rid2=%r' % (rid1, rid2)
    #text  = ibsfuncs.vsstr(rid1, rid2)
    text = ibs.get_match_text(rid1, rid2)
    if text is None:
        raise AssertionError('impossible state inspect_gui')
    return text


def get_status_bgrole(ibs, rid_pair):
    """ Background role for status column """
    rid1, rid2 = rid_pair
    truth = ibs.get_match_truth(rid1, rid2)
    #print('get status bgrole: %r truth=%r' % (rid_pair, truth))
    truth_color = vh.get_truth_color(truth, base255=True,
                                        lighten_amount=0.35)
    return truth_color


def get_buttontup(ibs, qtindex):
    model = qtindex.model()
    row = qtindex.row()
    rid1 = model.get_header_data('qrid', row)
    rid2 = model.get_header_data('rid', row)
    truth = ibs.get_match_truth(rid1, rid2)
    truth_color = vh.get_truth_color(truth, base255=True,
                                        lighten_amount=0.35)
    truth_text = ibs.get_match_text(rid1, rid2)
    callback = partial(review_match, ibs, rid1, rid2)
    #print('get_button, rid1=%r, rid2=%r, row=%r, truth=%r' % (rid1, rid2, row, truth))
    buttontup = (truth_text, callback, truth_color)
    return buttontup


def make_qres_api(ibs, qrid2_qres, ranks_lt=None):
    """
    Builds columns which are displayable in a ColumnListTableWidget
    """
    print('[inspect] make_qres_api')
    ranks_lt = ranks_lt if ranks_lt is not None else RANKS_LT
    candidate_matches = results_organizer.get_automatch_candidates(
        qrid2_qres, ranks_lt=ranks_lt)
    # Get extra info
    (qrids, rids, scores, ranks) = candidate_matches
    #qnames = ibs.get_roi_names(qrids)
    #names = ibs.get_roi_names(rids)
    #truths = np.array((ibs.get_roi_nids(qrids) - ibs.get_roi_nids(rids)) == 0)
    #buttons = [get_review_match_buttontup(rid1, rid2) for (rid1, rid2) in izip(qrids, rids)]

    #def get_review_match_buttontup(rid1, rid2):
    #    """ A buttontup is a string and a callback """
    #    return get_button  # ('Merge', partial(review_match, rid1, rid2))

    def get_rowid_button(rowid):
        return get_buttontup
    #opts = np.zeros(len(qrids))
    # Define column information

    # TODO: MAKE A PAIR IDER AND JUST USE EXISTING API_ITEM_MODEL FUNCTIONALITY
    # TO GET THOSE PAIRWISE INDEXES

    col_name_list = ['qrid', 'rid', 'status', 'querythumb', 'resthumb', 'qname',
                     'name', 'score', 'rank', ]

    col_types_dict = dict([
        ('qrid',       int),
        ('rid',        int),
        ('review',    'BUTTON'),
        ('status',     str),
        ('querythumb', 'PIXMAP'),
        ('resthumb',   'PIXMAP'),
        ('qname',      str),
        ('name',       str),
        ('score',      float),
        ('rank',       int),
        ('truth',     bool),
        ('opt',       int),
    ])

    col_getters_dict = dict([
        ('qrid',       np.array(qrids)),
        ('rid',        np.array(rids)),
        ('review',     get_rowid_button),
        ('status',     partial(get_status, ibs)),
        ('querythumb', ibs.get_roi_chip_thumbtup),
        ('resthumb',   ibs.get_roi_chip_thumbtup),
        ('qname',      ibs.get_roi_names),
        ('name',       ibs.get_roi_names),
        ('score',      np.array(scores)),
        ('rank',       np.array(ranks)),
        #('truth',     truths),
        #('opt',       opts),
    ])

    col_bgrole_dict = {
        'status': partial(get_status_bgrole, ibs),
    }
    col_ider_dict = {
        'status'     : ('qrid', 'rid'),
        'querythumb' : ('qrid'),
        'resthumb'   : ('rid'),
        'qname'      : ('qrid'),
        'name'       : ('rid'),
    }
    col_setter_dict = {
        'qname': ibs.set_roi_names,
        'name': ibs.set_roi_names
    }
    editable_colnames =  ['truth', 'notes', 'qname', 'name', 'opt']
    sortby = 'score'
    # Insert info into dict
    qres_api = CustomAPI(col_name_list, col_types_dict, col_getters_dict,
                         col_bgrole_dict, col_ider_dict, col_setter_dict,
                         editable_colnames, sortby)
    return qres_api
