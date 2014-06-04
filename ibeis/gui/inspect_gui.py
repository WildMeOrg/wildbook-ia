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
#from ibeis.dev import ibsfuncs
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
        # Register parentless QWidget
        if parent is None:
            fig_presenter.register_qt4_win(qres_wgt)
        qres_wgt.connect_signals_and_slots()

    def set_query_results(qres_wgt, ibs, qrid2_qres, **kwargs):
        print('[qres_wgt] Change QueryResultsWidget data')
        qres_wgt.ibs = ibs
        qres_wgt.qrid2_qres = qrid2_qres
        qres_wgt.lists_api = make_qres_lists_api(ibs, qrid2_qres, **kwargs)
        headers = qres_wgt.lists_api.make_headers()
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


class ListsAPI(object):
    """
    API wrapper around a list of lists, each containing column data
    Defines a single table
    """
    def __init__(self, column_tuples, editable_colnames, sortby=0,
                 sort_reverse=False):
        col_name_list, col_data_list, col_type_list = list(izip(*column_tuples))
        # Unpack the tuple into flat lists
        self.col_name_list = col_name_list
        self.col_data_list = col_data_list
        self.col_type_list = col_type_list
        self.col_edit_list = [name in editable_colnames
                              for name in col_name_list]
        self.col_sort_index = (self.col_name_list.index(sortby)
                               if isinstance(sortby, (str, unicode))
                               else sortby)
        self.col_sort_reverse = sort_reverse
        self.nCols = len(self.col_data_list)
        if len(col_data_list) == 0:
            self.nRows = 0
        else:
            self.nRows = len(col_data_list[0])

    def get(self, column, row):
        column_getter = self.col_data_list[column]
        if hasattr(column_getter, '__getitem__'):
            return column_getter[row]
        else:
            return column_getter(row)

    def set(self, column, row, val):
        column_setter = self.col_data_list[column]
        column_setter[row] = val
        if hasattr(column_setter, '__setitem__'):
            return column_setter[row]
        else:
            return column_setter(row)

    def make_getter_list(self):
        return [partial(self.get, column) for column in xrange(self.nCols)]

    def make_setter_list(self):
        return [partial(self.set, column) for column in xrange(self.nCols)]

    def ider(self):
        return range(self.nRows)

    def make_headers(self, tblname='lists_api_table', tblnice=None):
        """ Builds headers for APIItemModel """
        headers = {
            'name': tblname,
            'nice': tblname if tblnice is None else tblnice,
            'ider': self.ider,
            'col_name_list'    : self.col_name_list,
            'col_type_list'    : self.col_type_list,
            'col_nice_list'    : self.col_name_list,
            'col_edit_list'    : self.col_edit_list,
            'col_sort_index'   : self.col_sort_index,
            'col_sort_reverse' : self.col_sort_reverse,
            'col_getter_list'  : self.make_getter_list(),
            'col_setter_list'  : self.make_setter_list(),
        }
        return headers


def make_qres_lists_api(ibs, qrid2_qres, ranks_lt=None, tblname='qres'):
    """
    Builds columns which are displayable in a ColumnListTableWidget
    """
    ranks_lt = ranks_lt if ranks_lt is not None else RANKS_LT
    candidate_matches = results_organizer.get_automatch_candidates(
        qrid2_qres, ranks_lt=ranks_lt)
    # Get extra info
    (qrids, rids, scores, ranks) = candidate_matches
    qnames = ibs.get_roi_names(qrids)
    names = ibs.get_roi_names(rids)

    truths = (ibs.get_roi_nids(qrids) - ibs.get_roi_nids(rids)) == 0
    #views  = ['view ' + ibsfuncs.vsstr(qrid, rid, lite=True)
    #          for qrid, rid in izip(qrids, rids)]
    #opts = np.zeros(len(qrids))
    # Define column information
    column_tuples = [
        ('qrid',  np.array(qrids),  int),
        ('query',  lambda ids: ibs.get_roi_chip_thumbs(qrids[ids]), 'PIXMAP'),
        ('qname', np.array(qnames), str),
        ('rid',   np.array(rids),   int),
        ('name',  np.array(names),  str),
        ('result',  lambda ids: ibs.get_roi_chip_thumbs(rids[ids]), 'PIXMAP'),
        ('score', np.array(scores), float),
        ('rank',  np.array(ranks),  int),
        ('truth', np.array(truths), bool),
        #('opt',   opts,   ('COMBO', int)),
        #('view',  views, ('BUTTON', str)),
    ]
    editable_colnames =  ['truth', 'notes', 'qname', 'name', 'Opt']
    sortby = 'score'
    # Insert info into dict
    lists_api = ListsAPI(column_tuples, editable_colnames, sortby, True)
    return lists_api
