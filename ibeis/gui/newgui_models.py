from __future__ import absolute_import, division, print_function
import utool
from guitool import APITableModel
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui_models]')

#--------------------
# --- Data Models ---
#--------------------


class IBEISTableModel(APITableModel.APITableModel):
    def __init__(model, headers, parent=None, *args):
        model.eid = None
        model.window = parent
        model.original_ider = headers['ider']
        APITableModel.APITableModel.__init__(model, parent=parent, **headers)
        model._set_ider(model._ider)

    def _ider(model):
        """ Overrides the API model ider to give only selected encounter ids """
        return model.original_ider(eid=model.eid)

    @APITableModel.updater
    def _change_enc(model, eid):
        model.eid = eid
        model._update_rows()


class EncModel(APITableModel.APITableModel):
    def __init__(model, headers, parent=None, *args):
        model.window = parent
        model.headers = headers
        APITableModel.APITableModel.__init__(model, parent=parent, **headers)
