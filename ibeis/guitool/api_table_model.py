from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from . import qtype
from .guitool_decorators import checks_qt_error, signal_
from itertools import izip
import functools
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APITableModel]', DEBUG=False)


class ChangingModelLayout(object):
    """
    Context manager emitting layoutChanged before body,
    not updating durring body, and then updating after body.
    """
    @utool.accepts_scalar_input
    def __init__(self, model_list, *args):
        #print('Changing: %r' % (model_list,))
        self.model_list = list(model_list) + list(args)
        self.selfid = id(self)

    def __enter__(self):
        for model in self.model_list:
            if model._context_id is not None:
                continue
            model._context_id = self.selfid
            model._about_to_change()
            model._changeblocked = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for model in self.model_list:
            if model._context_id == self.selfid:
                model._context_id = None
                model._changeblocked = False
                model._change()


def default_method_decorator(func):
    """ Dummy decorator """
    return checks_qt_error(profile(func))


def updater(func):
    """
    Decorates a function by executing layoutChanged signals if not already in
    the middle of a layout changed
    """
    func = default_method_decorator(func)
    #@checks_qt_error
    @functools.wraps(func)
    def upd_wrapper(model, *args, **kwargs):
        #model._about_to_change()
        with ChangingModelLayout([model]):
            ret = func(model, *args, **kwargs)
        #model._change()
        return ret
    return upd_wrapper


class APITableModel(QtCore.QAbstractTableModel):
    """ Item model for displaying a list of columns """
    _rows_updated = signal_(str, int)

    #
    # Non-Qt Init Functions
    def __init__(model, headers=None, parent=None):
        """
        ider          : function which returns ids for setters and getters
        col_name_list : list of keys or SQL-like name for column to reference
                        abstracted data storage using getters and setters
        col_type_list : list of column value (Python) types
        col_nice_list : list of well-formatted names of the columns
        col_edit_list : list of booleans for if column should be editable
        ----
        col_setter_list : list of setter functions
        col_getter_list : list of getter functions
        ----
        col_sort_index : index into col_name_list for sorting
        col_sort_reverse : boolean of if to reverse the sort ordering
        ----
        """
        super(APITableModel, model).__init__()
        # Class member variables
        model._abouttochange   = False
        model._context_id      = None
        model._haschanged      = True
        model._changeblocked   = False
        model.name             = None
        model.nice             = None
        model.ider             = None
        model.col_name_list    = None
        model.col_type_list    = None
        model.col_nice_list    = None
        model.col_edit_list    = None
        model.col_setter_list  = None
        model.col_getter_list  = None
        model.col_sort_index   = None
        model.col_sort_reverse = None
        model.cache = None  # FIXME: This is not sustainable
        model.row_index_list = None
        # Initialize member variables
        model._about_to_change()
        if headers is None:
            headers = {}
        with ChangingModelLayout([model]):
            model._update_headers(**headers)

    @updater
    def _update_headers(model,
                        getdirty=None,
                        setdirty=None,
                        ider=None,
                        name=None,
                        nice=None,
                        col_name_list=None,
                        col_type_list=None,
                        col_nice_list=None,
                        col_edit_list=None,
                        col_setter_list=None,
                        col_getter_list=None,
                        col_sort_index=None,
                        col_sort_reverse=None):
        model.cache = {}  # FIXME: This is not sustainable
        model.name = str(name)
        model.nice = str(nice)
        model.getdirty = getdirty if getdirty is not None else lambda: False
        model.setdirty = setdirty if setdirty is not None else lambda flag: None
        # Initialize class
        model._set_ider(ider)
        model._set_col_name_type(col_name_list, col_type_list)
        model._set_col_nice(col_nice_list)
        model._set_col_edit(col_edit_list)
        model._set_col_setter(col_setter_list)
        model._set_col_getter(col_getter_list)
        # calls model._update_rows()
        model._set_sort(col_sort_index, col_sort_reverse)

    @default_method_decorator
    def _about_to_change(model, force=False):
        N = range(0, 15)  # NOQA
        if force or (model._haschanged and not model._abouttochange):
            print('ABOUT TO CHANGE: %r, caller=%r' % (model.name, utool.get_caller_name(N=N)))
            model._abouttochange = True
            #model._haschanged = False
            model.layoutAboutToBeChanged.emit()
            return True
        else:
            #print('NOT ABOUT TO CHANGE: %r, caller=%r' % (model.name, utool.get_caller_name(N=N)))
            return False

    @default_method_decorator
    def _change(model, force=False):
        N = range(0, 15)  # NOQA
        if force or (not model._haschanged or model._abouttochange and not model._changeblocked):
            print('LAYOUT CHANGED:  %r, caller=%r' % (model.name, utool.get_caller_name(N=N)))
            model._abouttochange = False
            model._haschanged = True
            model.layoutChanged.emit()
            return True
        else:
            #print('NOT LAYOU CHANGED: %r, caller=%r' % (model.name, utool.get_caller_name(N=N)))
            return False

    @default_method_decorator
    def _check_if_dirty(model):
        """ check the api has changed without us knowing """
        if model.getdirty():
            model._force_update()

    @default_method_decorator
    def _force_update(model):
        model._about_to_change(force=True)
        model._change(force=True)

    @default_method_decorator
    def _update(model, newrows=False):
        if newrows:
            model._update_rows()
        model.cache = {}
        #model.setdirty(False)

    @updater
    def _update_rows(model):
        """
        Uses the current ider and col_sort_index to create
        row_indicies
        """
        ids_ = model.ider()
        if len(ids_) == 0:
            model.row_index_list = []
        else:
            # start sort
            values = model.col_getter_list[model.col_sort_index](ids_)
            row_indices = [id_ for (value, id_) in sorted(list(izip(values, ids_)))]
            # end sort
            assert row_indices is not None, 'no indices'
            if model.col_sort_reverse:
                row_indices = row_indices[::-1]
            model.row_index_list = row_indices
        model._rows_updated.emit(model.name, len(model.row_index_list))

    @updater
    def _set_ider(model, ider=None):
        if ider is None:
            ider = lambda: []
        assert utool.is_funclike(ider), 'bad type: %r' % type(ider)
        model.ider = ider

    @updater
    def _set_col_name_type(model, col_name_list=None, col_type_list=None):
        if col_name_list is None:
            col_name_list = []
        if col_type_list is None:
            col_type_list = []
        assert len(col_name_list) == len(col_type_list), \
            'inconsistent colnametype'
        model.col_name_list = col_name_list
        model.col_type_list = col_type_list

    @updater
    def _set_col_nice(model, col_nice_list=None):
        if col_nice_list is None:
            col_nice_list = model.col_name_list[:]
        assert len(model.col_name_list) == len(col_nice_list), \
            'inconsistent colnice'
        model.col_nice_list = col_nice_list

    @default_method_decorator
    def _set_col_edit(model, col_edit_list=None):
        if col_edit_list is None:
            col_edit_list = [False] * len(model.col_name_list)
        assert len(model.col_name_list) == len(col_edit_list), \
            'inconsistent coledit'
        model.col_edit_list = col_edit_list

    @default_method_decorator
    def _set_col_setter(model, col_setter_list=None):
        if col_setter_list is None:
            col_setter_list = []
        assert len(model.col_name_list) == len(col_setter_list), \
            'inconsistent colsetter'
        model.col_setter_list = col_setter_list

    @default_method_decorator
    def _set_col_getter(model, col_getter_list=None):
        if col_getter_list is None:
            col_getter_list = []
        assert len(model.col_name_list) == len(col_getter_list), \
            'inconsistent colgetter'
        model.col_getter_list = col_getter_list

    @updater
    def _set_sort(model, col_sort_index=None, col_sort_reverse=None):
        if col_sort_index is None:
            col_sort_index = 0
        else:
            assert col_sort_index < len(model.col_name_list), \
                'sort index out of bounds by: %r' % col_sort_index
        if not utool.is_bool(col_sort_reverse):
            col_sort_reverse = False
        model.col_sort_index = col_sort_index
        model.col_sort_reverse = col_sort_reverse
        model._update_rows()

    #--------------------------------
    # --- API Interface Functions ---
    #--------------------------------

    @default_method_decorator
    def _get_col_align(model, column):
        assert column is not None
        if model.col_type_list[column] in utool.VALID_FLOAT_TYPES:
            return Qt.AlignRight
        else:
            return Qt.AlignHCenter

    @default_method_decorator
    def _get_row_id(model, row):
        try:
            id_ = model.row_index_list[row]
            return id_
        except IndexError as ex:
            msg = '\n'.join([
                'Error in _get_row_id',
                'name=%r\n' % model.name,
                'row=%r\n' % row,
                'len(model.row_index_list) = %r' % len(model.row_index_list),
            ])
            utool.printex(ex, msg)
            raise

    @default_method_decorator
    def _get_data(model, row, col):
        # Get general getter for this column
        getter = model.col_getter_list[col]
        # Get row_id accoring to sorting
        row_id = model._get_row_id(row)
        cachekey = (row_id, col)
        try:
            data = model.cache[cachekey]
        except KeyError:
            data = getter(row_id)
            model.cache[cachekey] = data
        return data

    @default_method_decorator
    def _set_data(model, row, col, value):
        """
            The setter function should be of the following format:

            def setter(column_name, row_id, value)
                # column_name is the key or SQL-like name for the column
                # row_id is the corresponding row key or SQL-like id that the
                #    row call back returned
                # value is the value that needs to be stored

            The setter function should return a boolean, if setting the value
            was successfull or not
        """
        row_id = model._get_row_id(row)
        cachekey = (row_id, col)
        try:
            del model.cache[cachekey]
        except KeyError:
            pass
        setter = model.col_setter_list[col]
        print('Setting data: row_id=%r, setter=%r' % (row_id, setter))
        return setter(row_id, value)

    #------------------------
    # --- QtGui Functions ---
    #------------------------

    @default_method_decorator
    def index(model, row, column, parent=QtCore.QModelIndex()):
        return model.createIndex(row, column)

    @default_method_decorator
    def rowCount(model, parent=QtCore.QModelIndex()):
        return len(model.row_index_list)

    @default_method_decorator
    def columnCount(model, parent=QtCore.QModelIndex()):
        return len(model.col_name_list)

    @default_method_decorator
    def data(model, qtindex, role=Qt.DisplayRole):
        """
        Depending on the role, returns either data or how to display data
        """
        if not qtindex.isValid():
            return None
        flags = model.flags(qtindex)
        row = qtindex.row()
        col = qtindex.column()
        #
        # Specify alignment
        if role == Qt.TextAlignmentRole:
            return model._get_col_align(col)
        #
        # Editable fields are colored
        if role == Qt.BackgroundRole and (flags & Qt.ItemIsEditable or
                                          flags & Qt.ItemIsUserCheckable):
            return QtCore.QVariant(QtGui.QColor(250, 240, 240))
        #
        # Editable fields are colored
        if role == Qt.DisplayRole or role == Qt.CheckStateRole:
            data = model._get_data(row, col)
            value = qtype.cast_into_qt(data, role, flags)
            return value
        #
        # else return an empty QVariant
        else:
            return QtCore.QVariant()

    @default_method_decorator
    def setData(model, qtindex, value, role=Qt.EditRole):
        """
        Sets the role data for the item at qtindex to value.
        value is a QVariant (called data in documentation)
        """
        try:
            if not qtindex.isValid():
                return None
            flags = model.flags(qtindex)
            row, col = qtindex.row(), qtindex.column()
            if not (flags & Qt.ItemIsEditable or flags & Qt.ItemIsUserCheckable):
                return None
            if role == Qt.CheckStateRole:
                type_ = 'QtCheckState'
                data = value == Qt.Checked
            elif role != Qt.EditRole:
                return False
            else:
                # Cast value into datatype
                type_ = model.col_type_list[col]
                data = qtype.cast_from_qt(value, type_)
            # Do actual setting of data
            model._set_data(row, col, data)
            # Emit that data was changed and return succcess
            model.dataChanged.emit(qtindex, qtindex)
            return True
        except Exception as ex:
            value = str(value.toString())  # NOQA
            utool.printex(ex, 'ignoring setData', '[model]',
                          key_list=['value'], iswarning=True)
            return False

    @default_method_decorator
    def headerData(model, column, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if column >= len(model.col_nice_list):
                return []
            return model.col_nice_list[column]
        else:
            return QtCore.QVariant()

    @default_method_decorator
    def sort(model, column, order):
        reverse = order == QtCore.Qt.DescendingOrder
        model._set_sort(column, reverse)

    @default_method_decorator
    def flags(model, qtindex):
        col = qtindex.column()
        if not model.col_edit_list[col]:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif model.col_type_list[col] in utool.VALID_BOOL_TYPES:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
