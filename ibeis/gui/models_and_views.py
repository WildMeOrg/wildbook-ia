from __future__ import absolute_import, division, print_function
import utool
from PyQt4 import QtGui, QtCore
from guitool import APITableModel, updater, signal_, slot_
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui_models]')

#--------------------
# --- Data Models ---
#--------------------


def _null_ider(**kwargs):
    return []


class IBEISTableModel(APITableModel):
    def __init__(model, headers=None, parent=None, *args):
        model.ibswin = parent
        model.eid = None
        model.original_ider = None
        APITableModel.__init__(model, headers=headers, parent=parent)

    def _update_headers(model, **headers):
        model.original_ider = headers.get('ider', _null_ider)
        headers['ider'] = model._ider
        return APITableModel._update_headers(model, **headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_ider(eid=model.eid)

    @updater
    def _change_enc(model, eid):
        model.eid = eid
        model._update_rows()


class EncTableModel(APITableModel):
    def __init__(model, headers=None, parent=None):
        model.ibswin = parent
        model.headers = headers
        APITableModel.__init__(model, headers=headers, parent=parent)


#############################
######### Data Views ########
#############################


class APITableView(QtGui.QTableView):
    contextMenuClicked = signal_(QtCore.QModelIndex, QtCore.QPoint)

    def __init__(tblview, parent=None):
        QtGui.QTableView.__init__(tblview, parent)
        # Allow sorting by column
        tblview.setSortingEnabled(True)
        # No vertical header
        verticalHeader = tblview.verticalHeader()
        verticalHeader.setVisible(False)
        # Stretchy column widths
        horizontalHeader = tblview.horizontalHeader()
        horizontalHeader.setStretchLastSection(True)
        horizontalHeader.setSortIndicatorShown(True)
        horizontalHeader.setHighlightSections(True)
        horizontalHeader.setCascadingSectionResizes(True)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        horizontalHeader.setMovable(True)
        # Selection behavior
        tblview.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        tblview.resizeColumnsToContents()
        tblview.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Context menu
        tblview.customContextMenuRequested.connect(tblview.on_customMenuRequested)

    @slot_(QtCore.QPoint)
    def on_customMenuRequested(tblview, pos):
        index = tblview.indexAt(pos)
        tblview.contextMenuClicked.emit(index, pos)


class IBEISTableView(APITableView):
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent

    def _change_enc(tblview, eid):
        tblview.model()._change_enc(eid)


class EncTableView(APITableView):
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent
        tblview.setMaximumSize(500, 9999)
