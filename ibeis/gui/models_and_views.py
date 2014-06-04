from __future__ import absolute_import, division, print_function
import utool
from guitool import APITableModel, APITableView, ChangeLayoutContext
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

    def _change_enc(model, eid):
        model.eid = eid
        with ChangeLayoutContext([model]):
            model._update_rows()


class EncTableModel(APITableModel):
    def __init__(model, headers=None, parent=None):
        model.ibswin = parent
        model.headers = headers
        APITableModel.__init__(model, headers=headers, parent=parent)


#-------------------
# --- Data Views ---
#-------------------

class IBEISTableView(APITableView):
    """
    View for ROI / NAME / IMAGE Tables
    """
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent

    def _change_enc(tblview, eid):
        tblview.model()._change_enc(eid)


class EncTableView(APITableView):
    """
    View for Encounter Table
    """
    def __init__(tblview, parent=None):
        APITableView.__init__(tblview, parent)
        tblview.ibswin = parent
        tblview.setMaximumSize(500, 9999)
