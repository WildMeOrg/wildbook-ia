from __future__ import absolute_import, division, print_function
import utool
from functools import partial
from itertools import izip
#utool.rrrr()
from ibeis.viz import interact
from ibeis.dev import results_organizer
from plottool import fig_presenter
from guitool import qtype, APITableWidget
from PyQt4 import QtCore
import guitool
from ibeis.dev import ibsfuncs
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[inspect_gui]', DEBUG=False)


RANKS_LT = 5


class QueryResultsWidget(APITableWidget):
    """ Window for gui inspection """

    def __init__(qres_wgt, ibs, qrid2_qres, parent=None, **kwargs):
        print('[qres_wgt] Init QueryResultsWidget')
        APITableWidget.__init__(qres_wgt, parent=parent)
        # Set results data
        qres_wgt.set_query_results(ibs, qrid2_qres, **kwargs)
        qres_wgt.connect_signals_and_slots()
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
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(iqrw, qtindex):
        print('Clicked: ' + str(qtype.qindexinfo(qtindex)))
        return show_match_at(iqrw, qtindex)
        # This is actually a release
        print('DoubleClicked: ' + str(qtype.qindexinfo(qtindex)))
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(iqrw, qtindex):
        print('Pressed: ' + str(qtype.qindexinfo(qtindex)))
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_activated(iqrw, qtindex):
        print('Activated: ' + str(qtype.qindexinfo(qtindex)))
        pass


def show_match_at(qres_wgt, qtindex):
    print('interact')
    model = qtindex.model()
    row = qtindex.row()
    rid  = model.get_header_data('rid', row)
    qrid = model.get_header_data('qrid', row)
    fig = interact.ishow_matches(qres_wgt.ibs, qres_wgt.qrid2_qres[qrid], rid)
    fig_presenter.bring_to_front(fig)


class CustomAPI(object):
    # TODO: Rename CustomAPI
    """
    API wrapper around a list of lists, each containing column data
    Defines a single table
    """
    def __init__(self, column_tuples, editable_colnames, sortby=0,
                 sort_reverse=False):
        print('[CustomAPI] <__init__>')
        col_name_list, col_data_list, col_type_list = list(izip(*column_tuples))
        # Unpack the tuple into flat lists
        self.col_name_list = col_name_list
        self.col_type_list = col_type_list
        # First data column is always at least a getter
        self.col_getter_list = col_data_list
        self.nCols = len(self.col_getter_list)
        if self.nCols == 0:
            self.nRows = 0
        else:
            self.nRows = len(col_data_list[0])
        # Initially the setters are set to the getters :O
        self.col_setter_list = list(col_data_list)
        # Initially set all column iders to universal ider
        self.col_ider_list = [None for _ in xrange(self.nCols)]
        # But then we overwrite the setter if specified
        # Column tuples are specified as a minimum of 3 tuples
        # (col_name, col_getter, col_type)
        # (col_name, col_getter, col_type, col_setter, colname_ider)
        # if setter is not specified the getter is used, which only works
        # if the getter is a list/array
        for column, tup in enumerate(column_tuples):
            if len(tup) > 3:
                # specify a setter explicitly
                print('[CustomAPI] Augment Column Setter')
                tup = column_tuples[column]
                _col_setter = tup[3]
                self.col_setter_list[column] = _col_setter
                if len(tup) > 4:
                    # Use another columns values as this column's ids
                    print('[CustomAPI] Augment Column Ider')
                    _colname = tup[4]
                    _column = self.col_name_list.index(_colname)
                    print(' * (_colname: '  + _colname + ') = _column: (' + str(_column) + ')')
                    _col_ider = partial(self.get, _column)
                    self.col_ider_list[column] = _col_ider

        self.col_edit_list = [name in editable_colnames
                              for name in col_name_list]
        self.col_sort_index = (self.col_name_list.index(sortby)
                               if isinstance(sortby, (str, unicode))
                               else sortby)
        self.col_sort_reverse = sort_reverse
        print('[CustomAPI] </__init__>')

    def _rectify_row(self, column, row):
        """ if this columns values are not indexed by the ider """
        ider_ = self.col_ider_list[column]
        if ider_ is None:
            return row
        else:
            row_ = ider_(row)
            #print(ider_)
            #print('Rectify: col=%r, row=%r, row_=%r' % (column, row, row_))
            return row_

    #@getter
    def get(self, column, row):
        # getters always receive primary rowids, rectify if
        # col_ider is specified
        row = self._rectify_row(column, row)
        column_getter = self.col_getter_list[column]
        # Columns might be indexable read/write arrays
        # or read only getters
        if hasattr(column_getter, '__getitem__'):
            val = column_getter[row]
        else:
            val = column_getter(row)
        return val

    #@setter
    def set(self, column, row, val):
        row = self._rectify_row(column, row)
        column_setter = self.col_setter_list[column]
        # Columns might be indexable read/write arrays
        # or write only setters
        if hasattr(column_setter, '__setitem__'):
            column_setter[row] = val
        else:
            column_setter(row, val)

    def ider(self):
        return range(self.nRows)

    def make_headers(self, tblname='qres_api', tblnice=None):
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
        }
        return headers

    def _make_getter_list(self):
        return [partial(self.get, column) for column in xrange(self.nCols)]

    def _make_setter_list(self):
        return [partial(self.set, column) for column in xrange(self.nCols)]


def review_match(rid1, rid2):
    print('Review match: ' + ibsfuncs.vsstr(rid1, rid2))
    #if text.startswith(BREAK_MATCH_PREF):
    #    ibs.set_roi_names([rid1, rid2], ['____', '____'])
    #elif text.startswith(NEW_MATCH_PREF):
    #    new_name = ibsfuncs.make_new_name(ibs)
    #    ibs.set_roi_names([rid1, rid2], [new_name, new_name])
    #elif text.startswith(RENAME1_PREF):
    #    name2 = ibs.get_roi_names(rid2)
    #    ibs.set_roi_names([rid1], [name2])
    #elif text.startswith(RENAME2_PREF):
    #    name1 = ibs.get_roi_names(rid1)
    #    ibs.set_roi_names([rid2], [name1])
    ## Emit that something has changed
    #self.on_change_callback()
    #self.show_page()


def make_qres_api(ibs, qrid2_qres, ranks_lt=None, tblname='qres'):
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
    from ibeis.viz import viz_helpers as vh

    def get_buttontup(qtindex):
        model = qtindex.model()
        row = qtindex.row()
        rid1 = model.get_header_data('qrid', row)
        rid2 = model.get_header_data('rid', row)
        truth = ibs.get_match_truth(rid1, rid2)
        truth_color = vh.get_truth_color(truth, base255=True,
                                         lighten_amount=0.35)
        callback = partial(review_match, rid1, rid2)
        #print('get_button, rid1=%r, rid2=%r, row=%r, truth=%r' % (rid1, rid2, row, truth))
        if truth == 2:
            buttontup = ('NEW Match', callback, truth_color)
        elif truth == 0:
            buttontup = ('JOIN Match', callback, truth_color)
        elif truth == 1:
            buttontup = ('SPLIT Match', callback, truth_color)
        else:
            raise AssertionError('impossible match state')
        return buttontup

    def get_rowid_button(rowid):
        return get_buttontup
    #opts = np.zeros(len(qrids))
    # Define column information
    column_tuples = [
        ('qrid',       np.array(qrids),           int),
        ('rid',        np.array(rids),            int),
        ('review',     get_rowid_button,          'BUTTON'),
        ('querythumb', ibs.get_roi_chip_thumbtup, 'PIXMAP', None,              'qrid'),
        ('resthumb',   ibs.get_roi_chip_thumbtup, 'PIXMAP', None,              'rid'),
        ('qname',      ibs.get_roi_names,         str,      ibs.set_roi_names, 'qrid'),
        ('name',       ibs.get_roi_names,         str,      ibs.set_roi_names, 'rid'),
        ('score',      np.array(scores),          float),
        ('rank',       np.array(ranks),           int),
        #('truth',     truths,                    bool),
        #('opt',       opts,   ('COMBO',          int)),
    ]
    editable_colnames =  ['truth', 'notes', 'qname', 'name', 'opt']
    sortby = 'score'
    # Insert info into dict
    qres_api = CustomAPI(column_tuples, editable_colnames, sortby, True)
    return qres_api
