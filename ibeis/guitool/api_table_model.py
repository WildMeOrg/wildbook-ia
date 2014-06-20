# TODO: Rename api_item_model
from __future__ import absolute_import, division, print_function
#import __builtin__
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from . import qtype
from .guitool_decorators import checks_qt_error, signal_
#from .api_thumb_delegate import APIThumbDelegate
from itertools import izip
import functools
import utool
import numpy as np
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APITableModel]', DEBUG=False)

#API_MODEL_BASE = QtCore.QAbstractTableModel
API_MODEL_BASE = QtCore.QAbstractItemModel


class ChangeLayoutContext(object):
    """
    Context manager emitting layoutChanged before body,
    not updating durring body, and then updating after body.
    """
    @utool.accepts_scalar_input
    def __init__(self, model_list, *args):
        #print('Changing: %r' % (model_list,))
        self.model_list = list(model_list) + list(args)

    def __enter__(self):
        for model in self.model_list:
            if model._context_id is not None:
                continue
            model._context_id = id(self)
            model._about_to_change()
            model._changeblocked = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for model in self.model_list:
            if model._context_id == id(self):
                model._context_id = None
                model._changeblocked = False
                model._change()


def default_method_decorator(func):
    """ Dummy decorator """
    return checks_qt_error(profile(func))
    #func_name = func.func_name
    ##func_ = checks_qt_error(profile(func))
    #func_ = func
    #@functools.wraps(func)
    #def wrapper(*args, **kwargs):
    #    __builtin__.print('func_name = ' + func_name)
    #    return func_(*args, **kwargs)
    #return wrapper


def updater(func):
    """
    Decorates a function by executing layoutChanged signals if not already in
    the middle of a layout changed
    """
    func = default_method_decorator(func)
    #@checks_qt_error
    @functools.wraps(func)
    def upd_wrapper(model, *args, **kwargs):
        with ChangeLayoutContext([model]):
            return func(model, *args, **kwargs)
    return upd_wrapper


class APITableModel(API_MODEL_BASE):
    """ Item model for displaying a list of columns """
    _rows_updated = signal_(str, int)
    EditableItemColor = QtGui.QColor(220, 220, 255)
    TrueItemColor     = QtGui.QColor(230, 250, 230)
    FalseItemColor    = QtGui.QColor(250, 230, 230)

    #
    # Non-Qt Init Functions
    def __init__(model, headers=None, parent=None):
        """
        iders          : list of functions that return ids for setters and getters
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
        model.view = parent
        API_MODEL_BASE.__init__(model, parent=parent)
        # Internal Flags
        model._abouttochange   = False
        model._context_id      = None
        model._haschanged      = True
        model._changeblocked   = False
        # Model Data And Accessors
        model.name             = 'None'
        model.nice             = 'None'
        model.iders            = [lambda: []]
        model.col_name_list    = []
        model.col_name_list_counts = {}  # HACKY
        model.col_type_list    = []
        model.col_nice_list    = []
        model.col_edit_list    = []
        model.col_setter_list  = []
        model.col_getter_list  = []
        model.col_level_list   = []
        model.col_sort_index   = None
        model.col_sort_reverse = False
        model.level_index_list = []
        model.cache = None  # FIXME: This is not sustainable
        # Initialize member variables
        #model._about_to_change()
        model.headers = headers  # save the headers
        if headers is not None:
            model._update_headers(**headers)

    @default_method_decorator
    def _about_to_change(model, force=False):
        #N = range(0, 10)  # NOQA
        if force or (not model._abouttochange and not model._changeblocked):
            #printDBG('ABOUT TO CHANGE: %r' % (model.name,))
            #printDBG('caller=%r' % (utool.get_caller_name(N=N)))
            model._abouttochange = True
            model.layoutAboutToBeChanged.emit()
            return True
        else:
            #printDBG('NOT ABOUT TO CHANGE')
            return False

    @default_method_decorator
    def _change(model, force=False):
        #N = range(0, 10)  # NOQA
        if force or (model._abouttochange and not model._changeblocked):
            #printDBG('LAYOUT CHANGED:  %r' % (model.name,))
            #printDBG('caller=%r' % (utool.get_caller_name(N=N)))
            #model._abouttochange = False
            model._abouttochange = False
            printDBG('CHANGE: CACHE INVALIDATED!')
            model.cache = {}
            model.layoutChanged.emit()
            return True
        else:
            #printDBG('NOT CHANGING')
            #print('NOT LAYOU CHANGED: %r, caller=%r' % (model.name, utool.get_caller_name(N=N)))
            return False

    @updater
    def _update_headers(model, **headers):
        iders            = headers.get('iders', None)
        name             = headers.get('name', None)
        nice             = headers.get('nice', None)
        col_name_list    = headers.get('col_name_list', None)
        col_type_list    = headers.get('col_type_list', None)
        col_nice_list    = headers.get('col_nice_list', None)
        col_edit_list    = headers.get('col_edit_list', None)
        col_setter_list  = headers.get('col_setter_list', None)
        col_getter_list  = headers.get('col_getter_list', None)
        col_level_list   = headers.get('col_level_list', None)
        col_sort_index   = headers.get('col_sort_index', None)
        col_sort_reverse = headers.get('col_sort_reverse', False)
        # New for dynamically getting non-data roles for each row
        col_rolegetter_list  = headers.get('col_rolegetter_list', None)
        model.cache = {}  # FIXME: This is not sustainable
        model.name = str(name)
        model.nice = str(nice)
        # Initialize class
        model._set_iders(iders)
        model._set_col_name_type(col_name_list, col_type_list)
        model._set_col_nice(col_nice_list)
        model._set_col_edit(col_edit_list)
        model._set_col_setter(col_setter_list)
        model._set_col_getter(col_getter_list)
        model._set_col_level(col_level_list)
        # calls model._update_rows()
        model._set_sort(col_sort_index, col_sort_reverse)

    @default_method_decorator
    def _update(model, newrows=False):
        #if newrows:
        model._update_rows()
        #printDBG('UPDATE: CACHE INVALIDATED!')
        model.cache = {}

    def _use_ider(model, level):
        if level == 0:
            return model.iders[level]()
        else:
            return model.iders[level](model._use_ider(level - 1))

    @updater
    def _update_rows(model):
        """
        Uses the current ider and col_sort_index to create
        row_indicies
        """
        #printDBG('UPDATE ROWS!')
        #print('UPDATE model(%s) rows' % model.name)
        if len(model.col_level_list) > 0:
            #print('-----')
            model.level_index_list = []
            highest_level = max(model.col_level_list)
            sort_index = 0 if model.col_sort_index is None else model.col_sort_index
            for i in range(0, highest_level + 1):
                ids_ = model._use_ider(i)
                #print('ids_: %r' % ids_)
                row_indices = []
                if len(ids_) != 0:
                    # start sort
                    if model.col_sort_index is not None:
                        level = model.col_level_list[sort_index]
                        getter = model.col_getter_list[sort_index]
                        values = getter(model._use_ider(level))
                        #print('values: %r' % values)
                    else:
                        values = ids_
                    reverse = model.col_sort_reverse
                    sorted_pairs = sorted(izip(values, ids_), reverse=reverse)
                    row_indices = [id_ for (value, id_) in sorted_pairs]
                    #print('row_indicies: %r' % row_indices)
                    # end sort
                assert row_indices is not None, 'no indices'
                model.level_index_list.append(row_indices)
            model._rows_updated.emit(model.name, len(model.level_index_list[0]))
            #print(model.level_index_list)

    @updater
    def _set_iders(model, iders=None):
        #printDBG('NEW IDER')
        if iders is None:
            iders = [lambda: []]
        assert utool.is_list(iders), 'bad type: %r' % type(iders)
        for index, ider in enumerate(iders):
            assert utool.is_funclike(ider), 'bad type at index %r: %r' % (index, type(ider))
        model.iders = iders

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
        # HACK: Column striping
        model.col_name_list_counts = {name: model.col_name_list.count(name) for name in set(col_name_list)}
        # Check if any of the column types are specified as delegates
        # PSA: Don't do this in the model.
        #for colx in xrange(len(model.col_type_list)):
        #    coltype_ = col_type_list[colx]
        #    if coltype_ == 'PIXMAP':
        #        try:
        #            model.view.setItemDelegateForColumn(colx, APIThumbDelegate(model.view))
        #        except:
        #            print("COLUMN INDEXING VIEW %r" % model.view)
        #            #print("IGNORING")

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

    @default_method_decorator
    def _set_col_rolegetter(model, col_rolegetter_list=None):
        """ rolegetter will be used for metadata like column color """
        if col_rolegetter_list is None:
            col_rolegetter_list = []
        assert len(model.col_name_list) == len(col_rolegetter_list), \
            'inconsistent col_rolegetter_list'
        model.col_rolegetter_list = col_rolegetter_list

    @default_method_decorator
    def _set_col_level(model, col_level_list=None):
        if col_level_list is None:
            col_level_list = map(lambda x: 0, model.col_name_list)
        assert len(model.col_name_list) == len(col_level_list), \
            'inconsistent collevel'
        model.col_level_list = col_level_list

    @updater
    def _set_sort(model, col_sort_index, col_sort_reverse=False):
        #printDBG('SET SORT')
        #print('_set_sort: model.col_name_list: %r' % model.col_name_list)
        if len(model.col_name_list) > 0:
            assert col_sort_index < len(model.col_name_list), \
                'sort index out of bounds by: %r' % col_sort_index
            model.col_sort_index = col_sort_index
            model.col_sort_reverse = col_sort_reverse
            # Update the row-id order
            model._update_rows()

    #----------------------------------
    # --- API Convineince Functions ---
    #----------------------------------

    @default_method_decorator
    def get_header_data(model, colname, row):
        """ Use _get_data if the column number is known """
        col = model.col_name_list.index(colname)
        return model._get_data(row, col)

    #--------------------------------
    # --- API Interface Functions ---
    #--------------------------------

    @default_method_decorator
    def _get_col_align(model, col):
        assert col is not None, 'bad column'

    @default_method_decorator
    def _get_row_id(model, row, col=None):
        if col is not None:
            num = model.col_name_list_counts[model.col_name_list[col]]
            # FOR NOW, NO MIXED TYPES: STRIPPED AND VERTICAL
            #assert num == len(model.col_name_list), "mixing stripped with non stripped"
            row = row * num + col
            col = 0

        try:
            id_ = model.level_index_list[0][row]
            return id_
        except IndexError as ex:  # NOQA
            # msg = '\n'.join([
            #     'Error in _get_row_id',
            #     'name=%r\n' % model.name,
            #     'row=%r\n' % row,
            #     'len(model.row_index_list) = %r' % len(model.row_index_list),
            # ])
            # utool.printex(ex, msg)
            # raise
            return None

    @default_method_decorator
    def _get_type(model, col):
        return model.col_type_list[col]

    @default_method_decorator
    def _get_data(model, row, col):
        #
        # <HACK: COLUMN_STRIPING>
        num = model.col_name_list_counts[model.col_name_list[col]]
        if num > 1:
            # FOR NOW, NO MIXED TYPES: STRIPPED AND VERTICAL
            assert num == len(model.col_name_list), "mixing stripped with non stripped"
            row = row * num + col
            col = 0
        # </HACK: COLUMN_STRIPING>
        #
        if row < len(model.level_index_list[0]):
            getter = model.col_getter_list[col]  # getter for this column
            row_id = model._get_row_id(row)  # row_id w.r.t. to sorting
            # <HACK: MODEL CACHE>
            cachekey = (row_id, col)
            try:
                if True:  # Cache is disabled
                    raise KeyError('')
                data = model.cache[cachekey]
            except KeyError:
                data = getter(row_id)
                model.cache[cachekey] = data
            # </HACK: MODEL CACHE>
            return data
        else:
            #raise AssertionError('row=%r, < len(model.row_index_list)=%r' % (row, len(model.level_index_list[0])))
            return None
            #return ('','',())
            #return "!!!<EMPTY FOR STRIPE>!!!"

    @default_method_decorator
    def _set_data(model, row, col, value):
        """ The setter function should be of the following format def
        setter(column_name, row_id, value) column_name is the key or SQL-like
        name for the column row_id is the corresponding row key or SQL-like id
        that the row call back returned value is the value that needs to be
        stored The setter function should return a boolean, if setting the value
        was successfull or not """
        row_id = model._get_row_id(row)
        #if row_id is None:
        #    return "__NONE__"
        cachekey = (row_id, col)
        try:
            del model.cache[cachekey]
        except KeyError:
            pass
        setter = model.col_setter_list[col]
        print('Setting data: row_id=%r, setter=%r' % (row_id, setter))
        return setter(row_id, value)

    def _get_columns_of_level(model, level):
        acc = []
        for i, lev in enumerate(model.col_level_list):
            if lev == level:
                acc.append(i)
        return acc

    #------------------------
    # --- QtGui Functions ---
    #------------------------
    @default_method_decorator
    def parent(model, qindex):
        """
        Returns the parent of the model item with the given index. If the item
        has no parent, an invalid QModelIndex is returned.

        A common convention used in models that expose tree data structures is
        that only items in the first column have children. For that case, when
        reimplementing this function in a subclass the column of the returned
        QModelIndex would be 0.

        When reimplementing this function in a subclass, be careful to avoid
        calling QModelIndex member functions, such as QModelIndex.parent(),
        since indexes belonging to your model will simply call your
        implementation, leading to infinite recursion.  """
        if qindex.isValid():
            indexlevel = model.col_level_list[qindex.column()]
            parentlevel_cols = model._get_columns_of_level(indexlevel-1)
            if len(parentlevel_cols) > 0:
                return model.createIndex(qindex.row(), parentlevel_cols[0])
        return QtCore.QModelIndex()

    @default_method_decorator
    def index(model, row, column, parent=QtCore.QModelIndex()):
        """ Qt Override
        Returns the index of the item in the model specified by the given row,
        column and parent index.  When reimplementing this function in a
        subclass, call createIndex() to generate model indexes that other
        components can use to refer to items in your model.
        NOTE: Object must be specified to sort delegates.
        """
        row_id = model._get_row_id(row)
        #data = model._get_data(row, column)
        return model.createIndex(row, column, object=row_id)

    @default_method_decorator
    def rowCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        if len(model.level_index_list) > 0:
            parent_level = model.col_level_list[parent.column()] if parent.isValid() else -1
            try:
                length = len(model.level_index_list[parent_level+1])
                # <HACK>
                counts = [np.ceil(length / count) for name, count in model.col_name_list_counts.items()]
                # </HACK>
                return max(counts)
            except:
                return len(model.level_index_list[parent_level+1])
        else:
            return 0

    @default_method_decorator
    def columnCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        if len(model.level_index_list) > 0:
            parent_level = model.col_level_list[parent.column()] if parent.isValid() else -1
            return len(model._get_columns_of_level(parent_level+1))
        else:
            return 0

    @default_method_decorator
    def data(model, qtindex, role=Qt.DisplayRole):
        """ Depending on the role, returns either data or how to display data
        Returns the data stored under the given role for the item referred to by
        the index.  Note: If you do not have a value to return, return an
        invalid QVariant instead of returning 0.  """
        if not qtindex.isValid():
            return None
        flags = model.flags(qtindex)
        row = qtindex.row()
        col = qtindex.column()
        type_ = model._get_type(col)

        if row >= model.rowCount():
                return QtCore.QVariant()

        #if role == Qt.SizeHintRole:
        #    #printDBG('REQUEST QSIZE FOR: ' + qtype.ItemDataRoles[role])
        #    return QtCore.QSize(64, 64)
        #
        # Specify Text Alignment Role
        if role == Qt.TextAlignmentRole:
            if type_ in qtype.QT_IMAGE_TYPES:
                value = Qt.AlignRight
            elif type_ in qtype.QT_BUTTON_TYPES:
                value = Qt.AlignRight
            elif type_ in utool.VALID_FLOAT_TYPES:
                value = Qt.AlignRight
            else:
                value = Qt.AlignHCenter
            return value
        #
        # Specify Background Rule
        elif role == Qt.BackgroundRole:
            if flags & Qt.ItemIsEditable:
                # Editable fields are colored
                return QtCore.QVariant(model.EditableItemColor)
            elif flags & Qt.ItemIsUserCheckable:
                # Checkable color depends on the truth value
                data = model._get_data(row, col)
                if data:
                    return QtCore.QVariant(model.TrueItemColor)
                else:
                    return QtCore.QVariant(model.FalseItemColor)
            else:
                pass
        #
        # Specify Foreground Role
        elif role == Qt.ForegroundRole:
            if flags & Qt.ItemIsEditable:
                return QtGui.QBrush(QtGui.QColor(0, 0, 0))
        #
        # Specify Decoration Role
        # elif role == Qt.DecorationRole and type_ in qtype.QT_IMAGE_TYPES:
        #     # The type is a pixelmap
        #     npimg = model._get_data(row, col)
        #     if npimg is not None:
        #         if type_ in qtype.QT_PIXMAP_TYPES:
        #             return qtype.numpy_to_qicon(npimg)
        #         elif type_ in qtype.QT_ICON_TYPES:
        #             return qtype.numpy_to_qpixmap(npimg)
        # Specify CheckState Role:
        if role == Qt.CheckStateRole:
            if flags & Qt.ItemIsUserCheckable:
                data = model._get_data(row, col)
                return Qt.Checked if data else Qt.Unchecked
        #
        # Return the data to edit or display
        elif role in (Qt.DisplayRole, Qt.EditRole):
            # For types displayed with custom delegates do not cast data into a
            # qvariant. This includes PIXMAP, BUTTON, and COMBO
            if type_ in qtype.QT_DELEGATE_TYPES:
                data = model._get_data(row, col)
                #print(data)
                return data
            else:
                # Display data with default delegate by casting to a qvariant
                data = model._get_data(row, col)
                value = qtype.cast_into_qt(data)
                return value
        else:
            #import __builtin__
            #role_name = qtype.ItemDataRoles[role]
            #__builtin__.print('UNHANDLED ROLE=%r' % role_name)
            pass
        # else return an empty QVariant
        value = QtCore.QVariant()
        return value

    @default_method_decorator
    def setData(model, qtindex, value, role=Qt.EditRole):
        """ Sets the role data for the item at qtindex to value.  value is a
        QVariant (called data in documentation) Returns a map with values for
        all predefined roles in the model for the item at the given index.
        Reimplement this function if you want to extend the default behavior of
        this function to include custom roles in the map. """
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
            old_data = model._get_data(row, col)
            if old_data != data:
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
    def headerData(model, section, orientation, role=Qt.DisplayRole):
        """ Qt Override
        Returns the data for the given role and section in the header with the
        specified orientation.  For horizontal headers, the section number
        corresponds to the column number. Similarly, for vertical headers, the
        section number corresponds to the row number. """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            column = section
            if column >= len(model.col_nice_list):
                return []
            return model.col_nice_list[column]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            row = section
            rowid = model._get_row_id(row)
            return rowid
        return QtCore.QVariant()

    @updater
    def sort(model, column, order):
        """ Qt Override """
        reverse = (order == QtCore.Qt.DescendingOrder)
        model._set_sort(column, reverse)

    @default_method_decorator
    def flags(model, qtindex):
        """ Qt Override
        returns Qt::ItemFlag
             0: 'NoItemFlags'          # It does not have any properties set.
             1: 'ItemIsSelectable'     # It can be selected.
             2: 'ItemIsEditable'       # It can be edited.
             4: 'ItemIsDragEnabled'    # It can be dragged.
             8: 'ItemIsDropEnabled'    # It can be used as a drop target.
            16: 'ItemIsUserCheckable'  # It can be checked or unchecked by the user.
            32: 'ItemIsEnabled'        # The user can interact with the item.
            64: 'ItemIsTristate'       # The item is checkable with three separate states. """
        # Return flags based on column properties (like type, and editable)
        col      = qtindex.column()
        type_    = model._get_type(col)
        editable = model.col_edit_list[col]
        if type_ in qtype.QT_IMAGE_TYPES:
            #return Qt.NoItemFlags
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif not editable:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif type_ in utool.VALID_BOOL_TYPES:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable

    # http://qt-project.org/doc/qt-4.8/qabstractitemmodel.html

    # QModelIndexList match ( const QModelIndex & start, int role, const
    # QVariant & value, int hits = 1, Qt::MatchFlags flags = Qt::MatchFlags(
    # Qt::MatchStartsWith | Qt::MatchWrap ) ) const [virtual]

    # def mimeData(QModelIndexList indexes): return QMimeData

    # def mimeTypes (QModelIndexList indexes) return QStringList

    # QModelIndexList persistentIndexList () const [protected] Returns the list
    # of indexes stored as persistent indexes in the model.

    # bool removeColumns ( int column, int count, const QModelIndex & parent =
    # QModelIndex() ) [virtual]

    # bool removeRow ( int row, const QModelIndex & parent = QModelIndex() )

    # bool removeRows ( int row, int count, const QModelIndex & parent =
    # QModelIndex() ) [virtual]

    # const QHash<int, QByteArray> & roleNames () const

    # void rowsAboutToBeInserted ( const QModelIndex & parent, int start, int
    # end ) [signal]

    # void rowsAboutToBeMoved ( const QModelIndex & sourceParent, int
    # sourceStart, int sourceEnd, const QModelIndex & destinationParent, int
    # destinationRow ) [signal]

    # void rowsAboutToBeRemoved ( const QModelIndex & parent, int start, int end
    # ) [signal]

    # void rowsInserted ( const QModelIndex & parent, int start, int end )
    # [signal]

    # void rowsMoved ( const QModelIndex & sourceParent, int sourceStart, int
    # sourceEnd, const QModelIndex & destinationParent, int destinationRow )
    # [signal]

    # void rowsRemoved ( const QModelIndex & parent, int start, int end )
    # [signal]

    # bool setData ( const QModelIndex & index, const QVariant & value, int role
    # = Qt::EditRole ) [virtual]

    # bool setHeaderData ( int section, Qt::Orientation orientation, const
    # QVariant & value, int role = Qt::EditRole ) [virtual] d
    # headerDataChanged() signals must be emitted explicitly

    # bool setItemData ( const QModelIndex & index, const QMap<int, QVariant> &
    # roles ) [virtual] Sets the role data for the item at index to the
    # associated value in roles, for every Qt::ItemDataRole.  Returns true if
    # successful; otherwise returns false.

    # void setRoleNames ( const QHash<int, QByteArray> & roleNames ) [protected]

    # void setSupportedDragActions ( Qt::DropActions actions)

    # QModelIndex sibling ( int row, int column, const QModelIndex & index )
    # const

    # QSize span ( const QModelIndex & index ) const [virtual]

    # bool submit () [virtual slot]

    # Qt::DropActions supportedDragActions () const

    # Qt::DropActions supportedDropActions () const [virtual]
