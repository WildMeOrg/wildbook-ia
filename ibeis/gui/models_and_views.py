from __future__ import absolute_import, division, print_function
import utool
from PyQt4 import QtCore, QtGui
from guitool import APITableModel, APITableView, APITreeView, APITableWidget, StripeProxyModel, ChangeLayoutContext
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


class IBEISTreeWidget(APITableWidget):
    def __init__(widget, headers=None, parent=None, *args):
        widget.ibswin = parent
        widget.eid = None
        APITableWidget.__init__(widget, headers=headers, parent=parent,
                                model_class=IBEISTableModel,
                                view_class=IBEISTreeView)


_DID_IBEISTABLEMODEL_METACLASS_HACK = False
IBEISTABLEMODEL_BASE = StripeProxyModel


class IBEISTableModel(IBEISTABLEMODEL_BASE):
    def __init__(model, headers=None, parent=None, *args):
        global _DID_IBEISTABLEMODEL_METACLASS_HACK
        model.sourcemodel = APITableModel(headers=headers, parent=parent)
        model.sourcemodel.ibswin = parent
        model.sourcemodel.eid = None
        model.sourcemodel.original_ider = None
        IBEISTABLEMODEL_BASE.__init__(model, parent=parent, *args)
        model.setSourceModel(model.sourcemodel)
        if not _DID_IBEISTABLEMODEL_METACLASS_HACK:
            _DID_IBEISTABLEMODEL_METACLASS_HACK = True
            exclude_list = ["_update_headers", "_ider", "_change_enc", "sourcemodel", "__class__", "__setattr__", "__getattr__", "_nd"]
            old_getattr = model.__class__.__getattr__
            #print('old_getattr outside: %r' % old_getattr)
            def new_getattr(obj, item):
                #print('old_getattr is %r' % old_getattr)
                #print('new_getattr(%r, %r)' % (obj, item))
                if item not in exclude_list:
                    print('sourcemodel.dict %r' % model.sourcemodel.__dict__)
                    try:
                        val = old_getattr(model.sourcemodel, item)
                    except AttributeError:
                        val = getattr(model.sourcemodel, item)
                else:
                    val = old_getattr(obj, item)
                #print('new_getattr returning %r' % val)
                return val
            model.__class__.__getattr__ = new_getattr

            old_setattr = model.__class__.__setattr__
            def new_setattr(obj, name, val):
                print('new_setattr(%r, %r, %r)' % (obj, name, val))
                if name not in exclude_list:
                    old_setattr(model.sourcemodel, name, val)
                else:
                    old_setattr(obj, name, val)
            model.__class__.__setattr__ = new_setattr

    def _update_headers(model, **headers):
        def _null_ider(**kwargs):
            return []
        model.sourcemodel.original_iders = headers.get('iders', [_null_ider])
        if len(model.sourcemodel.original_iders) > 0:
            model.sourcemodel.new_iders = model.sourcemodel.original_iders[:]
            model.sourcemodel.new_iders[0] = model._ider
        headers['iders'] = model.sourcemodel.new_iders
        return APITableModel._update_headers(model.sourcemodel, **headers)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.sourcemodel.original_iders[0](eid=model.eid)

    def _change_enc(model, eid):
        model.sourcemodel.eid = eid
        with ChangeLayoutContext([model.sourcemodel]):
            model.sourcemodel._update_rows()


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
