from __future__ import absolute_import, division, print_function
import utool
from guitool import APITableModel
from ibeis.gui import guiheaders as gh
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui_models]')

#############################
######## Data Models ########
#############################


class IBEISTableModel(APITableModel.APITableModel):
    def __init__(model, headers, parent=None, *args):
        model.encounter_id = None
        model.window = parent
        model.headers = headers
        kwargs = {
            'col_name_list'      : gh.header_names(headers),
            'col_type_list'      : gh.header_types(headers),
            'col_nice_list'      : gh.header_nices(headers),
            'col_edit_list'      : gh.header_edits(headers),
            'col_getter_list'    : model._getter,
            'col_setter_list'    : model._setter,
            'row_index_callback' : model._row_index_callback,
        }
        APITableModel.APITableModel.__init__(model, parent=parent, **kwargs)

    def _change_enc(model, encounter_id):
        model.encounter_id = encounter_id
        model._update_rows()

    def _row_index_callback(model, col_sort_name):
        # Get subset of ids (depending on selected tab)
        ids_ = gh.header_ids(model.headers)(eid=model.encounter_id)
        values = gh.getter_from_name(model.headers, col_sort_name)(ids_)
        values = zip(values, ids_)
        return [ tup[1] for tup in sorted(values) ]

    def _getter(model, column_name, row_id):
        result_list = gh.getter_from_name(model.headers, column_name)([row_id])
        if result_list[0] is None:
            model._update_rows()
        else:
            return result_list[0]

    def _setter(model, column_name, row_id, value):
        if value != '':
            gh.setter_from_name(model.headers, column_name)([row_id], [value])
            model.window._update_data()
        return True


class EncModel(APITableModel.APITableModel):
    def __init__(model, headers, parent=None, *args):
        model.window = parent
        model.headers = headers
        super(EncModel, model).__init__(col_name_list=gh.header_names(headers),
                                             col_type_list=gh.header_types(headers),
                                             col_nice_list=gh.header_nices(headers),
                                             col_edit_list=gh.header_edits(headers),
                                             col_getter_list=model._getter,
                                             col_setter_list=model._setter,
                                             row_index_callback=model._row_index_callback,
                                             paernt=parent)

    def _get_enc_id_name(model, qtindex):
        row, col = model._row_col(qtindex)
        tab_display_name_column_index = 0
        column_name = gh.header_names(model.headers)[tab_display_name_column_index]
        encounter_id = model._get_row_id(row)
        encounter_name = gh.getter_from_name(model.headers, column_name)([encounter_id])[0]
        return encounter_id, encounter_name

    def _row_index_callback(model, col_sort_name):
        gids = gh.header_ids(model.headers)()
        values = gh.getter_from_name(model.headers, col_sort_name)(gids)
        values = zip(values, gids)
        return [ tup[1] for tup in sorted(values) ]

    def _getter(model, column_name, row_id):
        result_list = gh.getter_from_name(model.headers, column_name)([row_id])
        if result_list[0] is None:
            model._update_rows()
        else:
            return result_list[0]

    def _setter(model, column_name, row_id, value):
        if value != '':
            gh.setter_from_name(model.headers, column_name)([row_id], [value])
            model.window._update_enc_tab_name(row_id, value)
        return True
