from __future__ import absolute_import, division, print_function
import utool
from PyQt4 import QtGui
from guitool import APITableModel, updater
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


def default_tableview_layout(view):
    # Allow sorting by column
    view.setSortingEnabled(True)
    # No vertical header
    verticalHeader = view.verticalHeader()
    verticalHeader.setVisible(False)
    # Stretchy column widths
    horizontalHeader = view.horizontalHeader()
    horizontalHeader.setStretchLastSection(True)
    horizontalHeader.setSortIndicatorShown(True)
    horizontalHeader.setHighlightSections(True)
    horizontalHeader.setCascadingSectionResizes(True)
    #horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
    horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
    #horizontalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
    horizontalHeader.setMovable(True)
    # Selection behavior
    view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
    view.resizeColumnsToContents()


class IBEISTableView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.ibswin = parent
        default_tableview_layout(view)

    def _change_enc(view, eid):
        view.model()._change_enc(eid)


class EncTableView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.ibswin = parent
        default_tableview_layout(view)
        view.setMaximumSize(500, 9999)
