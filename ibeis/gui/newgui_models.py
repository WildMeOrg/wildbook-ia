from __future__ import absolute_import, division, print_function
import utool
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


class EncModel(APITableModel):
    def __init__(model, headers=None, parent=None):
        model.ibswin = parent
        model.headers = headers
        APITableModel.__init__(model, headers=headers, parent=parent)
