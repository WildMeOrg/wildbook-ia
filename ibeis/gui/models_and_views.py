from __future__ import absolute_import, division, print_function
import utool
#from PyQt4 import QtCore, QtGui
from guitool import (APIItemModel, APITableView, APITreeView, APIItemWidget,
                     StripeProxyModel, ChangeLayoutContext)
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui_models]')

#---------------------
# --- IBEIS Tables ---
#---------------------


class IBEISTableWidget(APIItemWidget):
    def __init__(widget, headers=None, parent=None, *args):
        widget.ibswin = parent
        widget.eid = None
        APIItemWidget.__init__(widget, headers=headers, parent=parent,
                                model_class=IBEISTableModel,
                                view_class=IBEISTableView)


class IBEISTreeWidget(APIItemWidget):
    def __init__(widget, headers=None, parent=None, *args):
        widget.ibswin = parent
        widget.eid = None
        APIItemWidget.__init__(widget, headers=headers, parent=parent,
                                model_class=IBEISTreeModel,
                                view_class=IBEISTreeView)


IBEISTABLEMODEL_BASE = StripeProxyModel
#IBEISTABLEMODEL_BASE = APIItemModel
IBEISTREEMODEL_BASE = APIItemModel


class IBEISTableModel(IBEISTABLEMODEL_BASE):
    def __init__(model, headers=None, parent=None, *args):
        IBEISTABLEMODEL_BASE.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.eid = None
        model.original_ider = None
        if IBEISTABLEMODEL_BASE == StripeProxyModel:
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
        model._nd = headers.get('num_duplicates', 1)
        model.sourcemodel._update_headers(**headers)
        #return IBEISTABLEMODEL_BASE._update_headers(model, **headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_iders[0](eid=model.eid)

    def _change_enc(model, eid):
        model.eid = eid
        with ChangeLayoutContext([model]):
            IBEISTABLEMODEL_BASE._update_rows(model)


class IBEISTableView(APITableView):
    """
    View for ANNOTATION / NAME / IMAGE Tables
    """
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent

    def _change_enc(tblview, eid):
        tblview.verticalScrollBar().setSliderPosition(0)
        model = tblview.model()
        if model is not None:
            model._change_enc(eid)


class IBEISTreeModel(IBEISTREEMODEL_BASE):
    def __init__(model, headers=None, parent=None, *args):
        IBEISTREEMODEL_BASE.__init__(model, parent=parent, *args)
        model.ibswin = parent
        model.eid = None
        model.original_ider = None

    def _update_headers(model, **headers):
        def _null_ider(**kwargs):
            return []
        model.original_iders = headers.get('iders', [_null_ider])
        if len(model.original_iders) > 0:
            model.new_iders = model.original_iders[:]
            model.new_iders[0] = model._ider
        headers['iders'] = model.new_iders
        return IBEISTREEMODEL_BASE._update_headers(model, **headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_iders[0](eid=model.eid)

    def _change_enc(model, eid):
        model.eid = eid
        with ChangeLayoutContext([model]):
            IBEISTREEMODEL_BASE._update_rows(model)


class IBEISTreeView(APITreeView):
    """
    View for NAME / ANNOTATION Tree
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


class EncTableWidget(APIItemWidget):
    def __init__(widget, headers=None, parent=None, *args):
        widget.ibswin = parent
        APIItemWidget.__init__(widget, headers=headers, parent=parent,
                                model_class=EncTableModel,
                                view_class=EncTableView)


class EncTableModel(APIItemModel):
    def __init__(model, headers=None, parent=None):
        model.ibswin = parent
        model.headers = headers
        APIItemModel.__init__(model, headers=headers, parent=parent)


class EncTableView(APITableView):
    """
    View for Encounter Table
    """
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent
        tblview.setMaximumSize(500, 9999)
