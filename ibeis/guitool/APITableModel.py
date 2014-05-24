from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from . import qtype
import utool
import types
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APITableModel]', DEBUG=False)


class APITableModel(QtCore.QAbstractTableModel):
    """ Item model for displaying a list of columns """

    #
    # Non-Qt Init Functions

    def __init__(model, col_name_list=None, col_type_list=None, col_nice_list=None,
                 col_edit_list=None, col_setter_list=None, col_getter_list=None,
                 col_sort_name=None, col_sort_reverse=None,
                 row_index_callback=None, parent=None, **kwargs):
        """
            col_name_list -> list of keys or SQL-like name for column to reference
                             abstracted data storage using getters and setters
            col_type_list -> list of column value (Python) types
            col_nice_list -> list of well-formatted names of the columns
            col_edit_list -> list of booleans for if column should be editable
            ----
            col_setter_list -> list of setter functions
            col_getter_list -> list of getter functions
            ----
            col_sort_name -> key or SQL-like name to sort the column
            col_sort_reverse -> boolean of if to reverse the sort ordering
            ----
            row_index_callback -> a function that expects a call back for sort order,
                                  it should return a list of row indices.

            REQUIRED:
                col_name_list
                col_type_list
                col_setter_list
                col_getter_list
                row_index_callback
        """
        super(APITableModel, model).__init__()
        model.last_row = None
        model.last_role = None

        model.layoutAboutToBeChanged.emit()

        model._set_col_name_type(col_name_list, col_type_list)
        model._set_col_nice(col_nice_list)
        model._set_col_edit(col_edit_list)
        model._set_col_setter(col_setter_list)
        model._set_col_getter(col_getter_list)
        model._set_row_index_callback(row_index_callback)

        model._set_sort(col_sort_name, col_sort_reverse)  # calls model._update_rows()

        model.layoutChanged.emit()

    def _update_data(model):
        model.layoutAboutToBeChanged.emit()
        model.layoutChanged.emit()
        
    def _update_rows(model):
        '''
            The row_index_callback function should be of the following format:

            def callback(sort_col_name, sort_reverse)
                # sort_col_name is the key or SQL-like name for the column that the
                #    data will need to be sorted on
                # sort_reverse is a boolean to denote the order of the sorting

            The callback function should return the ordered list of keys or SQL-like
            ids that correspond to the rows in the abstracted data storage
        '''
        model.layoutAboutToBeChanged.emit()
        row_indices = model.row_index_callback(col_sort_name=model.col_sort_name)
        assert row_indices is not None, 'no indices'
        if model.col_sort_reverse:
            row_indices = row_indices[::-1]
        model.row_index_list = row_indices
        model.layoutChanged.emit()

    def _qtindex(model, row, column, parent=QtCore.QModelIndex()):
        return model.createIndex(row, column)

    def _row_col(model, qtindex):
        return qtindex.row(), qtindex.column()

    #############################
    ###### Setter Functions #####
    #############################

    def _set_col_name_type(model, col_name_list=None, col_type_list=None):
        model.layoutAboutToBeChanged.emit()

        if col_name_list is None:
            col_name_list = []
        if col_type_list is None:
            col_type_list = []
        assert len(col_name_list) == len(col_type_list)
        model.col_name_list = col_name_list
        model.col_type_list = col_type_list

        model.layoutChanged.emit()

    def _set_col_nice(model, col_nice_list=None):
        model.layoutAboutToBeChanged.emit()
        if col_nice_list is None:
            col_nice_list = model.col_name_list[:]
        assert len(model.col_name_list) == len(col_nice_list)
        model.col_nice_list = col_nice_list
        model.layoutChanged.emit()

    def _set_col_edit(model, col_edit_list=None):
        if col_edit_list is None:
            col_edit_list = [False] * len(model.col_name_list)
        assert len(model.col_name_list) == len(col_edit_list)
        model.col_edit_list = col_edit_list

    def _set_col_setter(model, col_setter_list=None):
        if isinstance(col_setter_list, types.FunctionType) or \
           isinstance(col_setter_list, types.MethodType):
            col_setter_list = [col_setter_list] * len(model.col_name_list)
        if col_setter_list is None:
            col_setter_list = []
        assert len(model.col_name_list) == len(col_setter_list)
        model.col_setter_list = col_setter_list

    def _set_col_getter(model, col_getter_list=None):
        if isinstance(col_getter_list, types.FunctionType) or \
           isinstance(col_getter_list, types.MethodType):
            col_getter_list = [col_getter_list] * len(model.col_name_list)
        if col_getter_list is None:
            col_getter_list = []
        assert len(model.col_name_list) == len(col_getter_list)
        model.col_getter_list = col_getter_list

    def _set_sort(model, col_sort_name=None, col_sort_reverse=None):
        if col_sort_name is None:
            col_sort_name = model.col_name_list[0]
        if not isinstance(col_sort_reverse, bool):
            col_sort_reverse = False
        assert col_sort_name in model.col_name_list, 'cannot sort by: %r' % col_sort_name
        model.col_sort_name = col_sort_name
        model.col_sort_reverse = col_sort_reverse
        model._update_rows()

    def _set_row_index_callback(model, row_index_callback):
        assert isinstance(row_index_callback, types.FunctionType) or \
            isinstance(row_index_callback, types.MethodType)
        model.row_index_callback = row_index_callback

    def _set_cell_qt(model, qtindex, value):
        row, col = model._row_col(qtindex)
        return model._set_cell(row, col, value)

    def _set_cell(model, row, col, value):
        '''
            The setter function should be of the following format:

            def setter(column_name, row_id, value)
                # column_name is the key or SQL-like name for the column
                # row_id is the corresponding row key or SQL-like id that the
                #    row call back returned
                # value is the value that needs to be stored

            The setter function should return a boolean, if setting the value
            was successfull or not
        '''
        setter = model._get_col_setter(column=col)
        col_name = model._get_col_name(column=col)
        row_id = model._get_row_id(row)
        return setter(col_name, row_id, value)

    #############################
    ###### Getter Functions #####
    #############################

    def _get_col_index(model, name):
        for index, col_name in enumerate(model.col_name_list):
            if col_name == name:
                return index
        return None

    def _get_col_general(model, _list, column=None, name=None):
        if name is not None:
            _column = model._get_col_index(name)
            if _column is not None:
                column = _column
        assert column is not None
        assert -1 * len(model.col_name_list) <= column and column < len(model.col_name_list)
        return _list[column]

    def _get_col_align(model, column=None, name=None):
        if name is not None:
            _column = model._get_col_index(name)
            if _column is not None:
                column = _column
        assert column is not None
        if model._get_col_type(column) in utool.VALID_FLOAT_TYPES:
            return Qt.AlignRight
        else:
            return Qt.AlignHCenter

    def _get_col_name(model, column):
        return model._get_col_general(model.col_name_list, column)

    def _get_col_type(model, column=None, name=None):
        return model._get_col_general(model.col_type_list, column, name)

    def _get_col_nice(model, column=None, name=None):
        return model._get_col_general(model.col_nice_list, column, name)

    def _get_col_edit(model, column=None, name=None):
        return model._get_col_general(model.col_edit_list, column, name)

    def _get_col_setter(model, column=None, name=None):
        return model._get_col_general(model.col_setter_list, column, name)

    def _get_col_getter(model, column=None, name=None):
        return model._get_col_general(model.col_getter_list, column, name)

    def _get_sort_name(model):
        return model.col_sort_name

    def _get_sort_reverse(model):
        return model.col_sort_reverse

    def _get_row_id(model, row):
        return model.row_index_list[row]

    def _get_cell_qt(model, qtindex):
        row, col = model._row_col(qtindex)
        return model._get_cell(row, col)

    def _get_cell(model, row, col):
        '''
            The getter function should be of the following format:

            def getter(column_name, row_id)
                # column_name is the key or SQL-like name for the column
                # row_id is the corresponding row key or SQL-like id that the
                #    row call back returned

            The getter function should return the corresponding value that belongs
            to row_id for column_name
        '''
        getter = model._get_col_getter(column=col)
        col_name = model._get_col_name(column=col)
        row_id = model._get_row_id(row)
        return getter(col_name, row_id)

    #############################
    ###### Qt Gui Functions #####
    #############################

    def rowCount(model, parent=QtCore.QModelIndex()):
        return len(model.row_index_list)

    def columnCount(model, parent=QtCore.QModelIndex()):
        return len(model.col_name_list)

    def data(model, qtindex, role=Qt.DisplayRole):
        """ Returns the data to display """
        if not qtindex.isValid():
            return None
        flags = model.flags(qtindex)
        row, col = model._row_col(qtindex)
        if role == Qt.TextAlignmentRole:
            return model._get_col_align(column=col)
        if role == Qt.BackgroundRole and (flags & Qt.ItemIsEditable or
                                          flags & Qt.ItemIsUserCheckable):
            return QtCore.QVariant(QtGui.QColor(250, 240, 240))
        if role == Qt.DisplayRole or (role == Qt.CheckStateRole and \
                (model.last_row != row or model.last_role != Qt.DisplayRole)):
            data = model._get_cell_qt(qtindex)
            value = qtype.cast_into_qt(data, role, flags)
            model.last_row = row
            model.last_role = role
            return value
        else:
            return QtCore.QVariant()

    def setData(model, qtindex, value, role=Qt.EditRole):
        """ Sets the role data for the item at qtindex to value.
        value is a QVariant (called data in documentation)
        """
        try:
            if not qtindex.isValid():
                return None
            flags = model.flags(qtindex)
            row, col = model._row_col(qtindex)
            if not (flags & Qt.ItemIsEditable or flags & Qt.ItemIsUserCheckable):
                return None
            if role == Qt.CheckStateRole:
                type_ = 'QtCheckState'
                data = value == Qt.Checked
            elif role != Qt.EditRole:
                return False
            else:
                # Cast value into datatype
                type_ = model._get_col_type(column=col)
                data = qtype.cast_from_qt(value, type_)
            # Do actual setting of data
            model._set_cell_qt(qtindex, data)
            # Emit that data was changed and return succcess
            model.dataChanged.emit(qtindex, qtindex)
            return True
        except Exception as ex:
            value = str(value.toString())  # NOQA
            utool.printex(ex, 'ignoring setData', '[model]',
                          key_list=['value'])
            return False

    def headerData(model, column, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return model._get_col_nice(column=column)
        else:
            return QtCore.QVariant()

    def sort(model, column, order):
        model.layoutAboutToBeChanged.emit()
        name = model._get_col_name(column)
        reverse = order == QtCore.Qt.DescendingOrder
        model._set_sort(name, reverse)
        model.layoutChanged.emit()

    def flags(model, qtindex):
        row, col = model._row_col(qtindex)
        if not model._get_col_edit(col):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if model._get_col_type(column=col) in utool.VALID_BOOL_TYPES:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
