from __future__ import absolute_import, division, print_function
import utool
from itertools import izip
#utool.rrrr()
from ibeis.viz import interact
from ibeis.dev import results_organizer
from plottool import fig_presenter
from guitool import guitool_tables, qtype
from PyQt4 import QtCore
import guitool
#from ibeis.dev import ibsfuncs
#import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[inspect_gui]', DEBUG=False)


class QueryResultsWidget(guitool_tables.ColumnListTableWidget):
    """ Window for gui inspection """
    def __init__(qrw, ibs, qrid2_qres, ranks_lt=5, parent=None):
        super(QueryResultsWidget, qrw).__init__(parent=parent)
        qrw.change_data(ibs, qrid2_qres, ranks_lt)
        fig_presenter.register_qt4_win(qrw)
        qrw.connect_signals_and_slots()
        qrw.setWindowTitle('QueryResultView')

    def change_data(qrw, ibs, qrid2_qres, ranks_lt=5):
        qrw.ibs = ibs
        qrw.qrid2_qres = qrid2_qres
        column_dict = make_query_result_column_dict(ibs, qrid2_qres, ranks_lt)
        super(QueryResultsWidget, qrw).change_data(**column_dict)

    def connect_signals_and_slots(qrw):
        qrw.view.clicked.connect(qrw._on_click)
        qrw.view.doubleClicked.connect(qrw._on_doubleclick)
        qrw.view.pressed.connect(qrw._on_doubleclick)
        qrw.view.activated.connect(qrw._on_activated)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_click(iqrw, index):
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(iqrw, index):
        print('Clicked: ' + str(qtype.qindexinfo(index)))
        return show_match_at(iqrw, index)
        # This is actually a release
        #print('DoubleClicked: ' + str(qtype.qindexinfo(index)))
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_pressed(iqrw, index):
        #print('Pressed: ' + str(qtype.qindexinfo(index)))
        pass

    @guitool.slot_(QtCore.QModelIndex)
    def _on_activated(iqrw, index):
        #print('Activated: ' + str(qtype.qindexinfo(index)))
        pass


class ColumnListsAPI(object):
    def __init__(self, col_data_list, column_headers, column_editable):
        self.col_data_list = col_data_list
        if len(col_data_list) == 0:
            self.nRows = 0
        else:
            self.nRows = len(col_data_list[0])

    def get(self, column, row):
        return self.col_data_list[column][row]

    def set(self, column, row, val):
        self.col_data_list[column][row] = val

    def ider(self):
        return range(self.nRows)


def show_match_at(qrw, index):
    print('interact')
    if qrw.is_index_clickable(index):
        return
    rid  = qrw.get_index_header_data('rid', index)
    qrid = qrw.get_index_header_data('qrid', index)
    fig = interact.ishow_matches(qrw.ibs, qrw.qrid2_qres[qrid], rid)
    fig_presenter.bring_to_front(fig)


def make_query_result_column_dict(ibs, qrid2_qres, ranks_lt=5):
    """
    Builds columns which are displayable in a ColumnListTableWidget
    """
    candidate_matches = results_organizer.get_automatch_candidates(qrid2_qres, ranks_lt=ranks_lt)
    # Get extra info
    (qrids, rids, scores, ranks) = candidate_matches
    truths = (ibs.get_roi_nids(qrids) - ibs.get_roi_nids(rids)) == 0
    #views  = ['view ' + ibsfuncs.vsstr(qrid, rid, lite=True)
    #          for qrid, rid in izip(qrids, rids)]
    #opts = np.zeros(len(qrids))
    # Define column information
    column_tuples = [
        ('qrid',  qrids,  int),
        ('rid',   rids,   int),
        ('score', scores, float),
        ('rank',  ranks,  int),
        #('opt',   opts,   ('COMBO', int)),
        ('truth', truths, bool),
        #('view',  views, ('BUTTON', str)),
    ]
    # Unpack the tuple into flat lists
    col_name_list, col_data_list, col_type_list = list(izip(*column_tuples))
    col_edit_list = ['truth', 'notes', 'Opt']
    # Insert info into dict
    column_dict = {
        'col_edit_list':  col_edit_list,
        'col_name_list':  col_name_list,
        'col_data_list':    col_data_list,
        'col_type_list':  col_type_list,
        'col_sort_index': col_name_list.index('score'),
    }
    return column_dict
