from __future__ import absolute_import, division, print_function
import utool
from PyQt4 import QtCore, QtGui
from guitool import APITableModel, APITableView, APITreeView, APITableWidget, ChangeLayoutContext
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui_models]')

#---------------------
# --- IBEIS Tables ---
#---------------------


class IBEISTableWidget(APITableWidget):
    def __init__(widget, headers=None, parent=None, *args):
        widget.ibswin = parent
        widget.eid = None
        APITableWidget.__init__(widget, headers=headers, parent=parent,
                                model_class=IBEISTableModel,
                                view_class=IBEISTableView)


class IBEISTableModel(APITableModel):
    def __init__(model, headers=None, parent=None, *args):
        model.ibswin = parent
        model.eid = None
        model.original_ider = None
        APITableModel.__init__(model, headers=headers, parent=parent)

    def _update_headers(model, **headers):
        def _null_ider(**kwargs):
            return []
        model.original_ider = headers.get('ider', _null_ider)
        headers['ider'] = model._ider
        return APITableModel._update_headers(model, **headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_ider(eid=model.eid)

    def _change_enc(model, eid):
        model.eid = eid
        with ChangeLayoutContext([model]):
            model._update_rows()


class IBEISTableView(APITableView):
    """
    View for ROI / NAME / IMAGE Tables
    """
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent

    def _change_enc(tblview, eid):
        tblview.verticalScrollBar().setSliderPosition(0)
        model = tblview.model()
        if model is not None:
            model._change_enc(eid)
class IBEISTreeView(APITreeView):
    """
    View for NAME / ROI Tree
    """
    def __init__(treeview, parent=None):
        APITreeView.__init__(treeview, parent)
        treeview.ibswin = parent

    def _change_enc(treeview, eid):
        treeview.verticalScrollBar().setSliderPosition(0)
        model = treeview.model()
        if model is not None:
            model._change_enc(eid)



#-------------------------
# --- ENCOUNTER TABLES ---
#-------------------------


class EncTableWidget(APITableWidget):
    def __init__(widget, headers=None, parent=None, *args):
        widget.ibswin = parent
        APITableWidget.__init__(widget, headers=headers, parent=parent,
                                model_class=EncTableModel,
                                view_class=EncTableView)


class EncTableModel(APITableModel):
    def __init__(model, headers=None, parent=None):
        model.ibswin = parent
        model.headers = headers
        APITableModel.__init__(model, headers=headers, parent=parent)


class EncTableView(APITableView):
    """
    View for Encounter Table
    """
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent
        tblview.setMaximumSize(500, 9999)
