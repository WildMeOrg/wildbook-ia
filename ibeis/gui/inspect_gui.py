from __future__ import absolute_import, division, print_function
import utool
#utool.rrrr()
from ibeis.viz import interact
from PyQt4 import QtCore
#from ibeis.dev import ibsfuncs
import guitool
from itertools import izip
from guitool import guitool_tables, qtype
from plottool import fig_presenter
#import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[back]', DEBUG=False)


def make_query_result_column_dict(ibs, qrid2_qres, maxrank=5):
    """
    Builds columns which are displayable in a ColumnListTableWidget
    """
    from ibeis.dev import results_organizer
    candidate_matches = results_organizer.get_automatch_candidates(qrid2_qres, maxrank=maxrank)
    # Get extra info
    (qrids, rids, scores, ranks) = candidate_matches
    truths = (ibs.get_roi_nids(qrids) - ibs.get_roi_nids(rids)) == 0
    #views  = ['view ' + ibsfuncs.vsstr(qrid, rid, lite=True)
              #for qrid, rid in izip(qrids, rids)]
    #opts  = np.zeros(len(qrids))
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
    header_list, column_list, coltype_list = list(izip(*column_tuples))
    editable_headers = ['truth', 'notes', 'Opt']
    # Insert info into dict
    column_dict = {
        'editable_headers':  editable_headers,
        'header_list':       header_list,
        'column_list':       column_list,
        'coltype_list':      coltype_list,
    }
    return column_dict


class QueryResultsWidget(guitool_tables.ColumnListTableWidget):
    """ Window for gui inspection """
    def __init__(qrw, ibs, qrid2_qres, maxrank=5, parent=None):
        super(QueryResultsWidget, qrw).__init__(parent=parent)
        qrw.change_data(ibs, qrid2_qres, maxrank)
        fig_presenter.register_qt4_win(qrw)
        qrw.connect_signals_and_slots()
        qrw.setWindowTitle('QueryResultView')

    def change_data(qrw, ibs, qrid2_qres, maxrank=5):
        qrw.ibs = ibs
        qrw.qrid2_qres = qrid2_qres
        column_dict = make_query_result_column_dict(ibs, qrid2_qres, maxrank)
        super(QueryResultsWidget, qrw).change_data(**column_dict)

    def connect_signals_and_slots(qrw):
        qrw.view.clicked.connect(qrw._on_click)
        qrw.view.doubleClicked.connect(qrw._on_doubleclick)
        qrw.view.pressed.connect(qrw._on_doubleclick)
        qrw.view.activated.connect(qrw._on_activated)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_click(iqrw, index):
        print('Clicked: ' + str(qtype.qindexinfo(index)))
        return on_click(iqrw, index)

    @guitool.slot_(QtCore.QModelIndex)
    def _on_doubleclick(iqrw, index):
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


def on_click(qrw, index):
    print('interact')
    if qrw.is_index_clickable(index):
        return
    rid  = qrw.get_index_header_data('rid', index)
    qrid = qrw.get_index_header_data('qrid', index)
    interact.ishow_matches(qrw.ibs, qrw.qrid2_qres[qrid], rid)
