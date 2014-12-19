# TODO: Rename api_item_model
from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtCore, QtGui, QVariantHack
from guitool.__PYQT__.QtCore import Qt
from guitool import qtype
from guitool.guitool_decorators import checks_qt_error, signal_  # NOQA
from six.moves import zip  # builtins
#from utool._internal.meta_util_six import get_funcname
import functools
import utool
import utool as ut
#from .api_thumb_delegate import APIThumbDelegate
#import numpy as np
#profile = lambda func: func
#printDBG = lambda *args: None
# UTOOL PRINT STATEMENTS CAUSE RACE CONDITIONS IN QT THAT CAN LEAD TO SEGFAULTS
# DO NOT INJECT THEM IN GUITOOL
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APIItemModel]', DEBUG=False)
ut.noinject(__name__, '[APIItemModel]', DEBUG=False)

profile = ut.profile

API_MODEL_BASE = QtCore.QAbstractItemModel

VERBOSE = utool.VERBOSE or ut.get_argflag(('--verbose-qt', '--verbqt'))


try:
    # TODO Cyth should take care of this stuff
    # also, it should be a function level import not module?
    #if not utool.get_argflag('--nocyth'):
    if utool.get_argflag('--cyth'):
        from guitool import api_tree_node_cython as _atn
    else:
        raise ImportError('')
    #print('[guitool] cython ON')
except ImportError:
    #print('[guitool] cython OFF')
    # TODO: Cython should be wrapped in parent module
    from guitool import api_tree_node as _atn


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
            if model._get_context_id() is not None:
                continue
            model._set_context_id(id(self))
            model._about_to_change()
            model._set_changeblocked(True)
        return self

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('[api_model] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        for model in self.model_list:
            if model._get_context_id() == id(self):
                model._set_context_id(None)
                model._set_changeblocked(False)
                model._change()


def default_method_decorator(func):
    """ Dummy decorator """
    #return profile(func)
    #return checks_qt_error(profile(func))
    return func


def updater(func):
    """
    Decorates a function by executing layoutChanged signals if not already in
    the middle of a layout changed
    """
    func_ = default_method_decorator(func)
    #@checks_qt_error
    @functools.wraps(func)
    def upd_wrapper(model, *args, **kwargs):
        with ChangeLayoutContext([model]):
            return func_(model, *args, **kwargs)
    return upd_wrapper


class APIItemModel(API_MODEL_BASE):
    """
    Item model for displaying a list of columns

    Attributes:
        iders         : list of functions that return ids for setters and getters
        col_name_list : list of keys or SQL-like name for column to reference
                        abstracted data storage using getters and setters
        col_type_list : list of column value (Python) types
        col_nice_list : list of well-formatted names of the columns
        col_edit_list : list of booleans for if column should be editable

        col_setter_list : list of setter functions
        col_getter_list : list of getter functions

        col_sort_index : index into col_name_list for sorting
        col_sort_reverse : boolean of if to reverse the sort ordering
    """
    _rows_updated = signal_(str, int)
    EditableItemColor = QtGui.QColor(220, 220, 255)
    TrueItemColor     = QtGui.QColor(230, 250, 230)
    FalseItemColor    = QtGui.QColor(250, 230, 230)

    def _set_context_id(self, id_):
        self._context_id = id_

    def _get_context_id(self):
        return self._context_id

    def _set_changeblocked(self, changeblocked_):
        self._changeblocked = changeblocked_

    def _get_changeblocked(self):
        return self._changeblocked
    #
    # Non-Qt Init Functions
    def __init__(model, headers=None, parent=None):
        if VERBOSE:
            print('[APIItemModel] __init__')
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
        model.col_visible_list = []
        model.col_name_list    = []
        model.col_type_list    = []
        model.col_nice_list    = []
        model.col_edit_list    = []
        model.col_setter_list  = []
        model.col_getter_list  = []
        model.col_level_list   = []
        model.col_bgrole_getter_list = None
        model.col_sort_index   = None
        model.col_sort_reverse = False
        model.level_index_list = []
        model.cache = None  # FIXME: This is not sustainable
        model.scope_hack_list = []
        model.root_node = _atn.TreeNode(-1, None, -1)
        # Initialize member variables
        #model._about_to_change()
        model.headers = headers  # save the headers

        model.lazy_updater = None
        if headers is not None:
            model._update_headers(**headers)

    #@profile
    @updater
    def _update_headers(model, **headers):
        if VERBOSE:
            print('[APIItemModel] _update_headers')
        iders            = headers.get('iders', None)
        name             = headers.get('name', None)
        nice             = headers.get('nice', None)
        #print('[api_model] UPDATE HEADERS: %r' % (name,))
        col_name_list    = headers.get('col_name_list', None)
        col_type_list    = headers.get('col_type_list', None)
        col_nice_list    = headers.get('col_nice_list', None)
        col_edit_list    = headers.get('col_edit_list', None)
        col_setter_list  = headers.get('col_setter_list', None)
        col_getter_list  = headers.get('col_getter_list', None)
        col_level_list   = headers.get('col_level_list', None)
        col_sort_index   = headers.get('col_sort_index', -1)
        col_sort_reverse = headers.get('col_sort_reverse', False)
        # New for dynamically getting non-data roles for each row
        col_bgrole_getter_list  = headers.get('col_bgrole_getter_list', None)
        col_visible_list = headers.get('col_visible_list', None)
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
        model._set_col_bgrole_getter(col_bgrole_getter_list)
        model._set_col_visible_list(col_visible_list)

        model._set_col_level(col_level_list)
        # calls model._update_rows()
        model._set_sort(col_sort_index, col_sort_reverse, rebuild_structure=True)

    #@profile
    @updater
    def _update_rows(model, rebuild_structure=True):
        """
        Uses the current ider and col_sort_index to create
        row_indicies
        """
        if VERBOSE:
            print('[APIItemModel] +-----------')
            print('[APIItemModel] _update_rows')
        # this is not slow
        #with utool.Timer('update_rows'):
        #printDBG('UPDATE ROWS!')
        #print('UPDATE ROWS!')
        #print('num_rows=%r' % len(model.col_level_list))
        #print('UPDATE model(%s) rows' % model.name)
        #print('[api_model] UPDATE ROWS: %r' % (model.name,))
        #print(utool.get_caller_name(range(4, 12)))
        if len(model.col_level_list) == 0:
            return
        #old_root = model.root_node  # NOQA
        if rebuild_structure:
            with utool.Timer('[%s] _update_rows: %r' %
                             ('cyth' if _atn.CYTHONIZED else 'pyth',
                              model.name,), newline=False):
                model.root_node = _atn.build_internal_structure(model)
        #print('-----')
        #def lazy_update_rows():
        #    with utool.Timer('lazy updater: %r' % (model.name,)):
        #        printDBG('[model] calling lazy updater: %r' % (model.name,))
        # REMOVING LAZY FUNCTION BECAUSE IT MIGHT HAVE CAUSED PROBLEMS
        with utool.Timer('[%s] _update_rows2: %r' %
                         ('cyth' if _atn.CYTHONIZED else 'pyth',
                          model.name,), newline=False):
            if VERBOSE:
                print('[APIItemModel] lazy_update_rows')
            model.level_index_list = []
            sort_index = 0 if model.col_sort_index is None else model.col_sort_index
            print('sort_index=%r' % (sort_index,))
            children = model.root_node.get_children()  # THIS IS THE LINE THAT TAKES FOREVER
            id_list = [child.get_id() for child in children]
            #print('ids_ generated')
            nodes = []
            if len(id_list) != 0:
                if VERBOSE:
                    print('[APIItemModel] lazy_update_rows len(id_list) = %r' % (len(id_list)))
                # start sort
                if model.col_sort_index is not None:
                    getter = model.col_getter_list[sort_index]
                    values = getter(id_list)
                    #print('values got')
                else:
                    values = id_list
                reverse = model.col_sort_reverse
                sorted_pairs = sorted(zip(values, id_list, children), reverse=reverse)
                nodes = [child for (value, id_, child) in sorted_pairs]
                level = model.col_level_list[sort_index]
                #print("row_indices sorted")
                if level == 0:
                    model.root_node.set_children(nodes)
                # end sort
            if utool.USE_ASSERT:
                assert nodes is not None, 'no indices'
            model.level_index_list = nodes
            #if VERBOSE:
            #    print('[APIItemModel] lazy_update_rows emmiting _rows_updated')

            # EMIT THE NUMERR OF ROWS AND THE NAME OF FOR THE VIEW TO DISPLAY
            model._rows_updated.emit(model.name, len(model.level_index_list))

            # lazy method didn't work. Eagerly evaluate
            #lazy_update_rows()
            # HACK TO MAKE SURE TREE NODES DONT DELETE THEMSELVES
            #if VERBOSE:
            #    print('[APIItemModel] build_scope_hack_list')
            # SCOPE HACK SEEMS TO HAVE NOT HALPED
            #model.scope_hack_list = []
            #_atn.build_scope_hack_list(model.root_node, model.scope_hack_list)
            #model.lazy_updater = lazy_update_rows
            #print("Rows updated")
            if VERBOSE:
                print('[APIItemModel] finished _update_rows')
                print('[APIItemModel] L__________')
        #del old_root

    #@profile
    def lazy_checks(model):
        if model.lazy_updater is not None:
            print('[model] lazy update %r caller %r: ' %
                  (model.name, utool.get_caller_name(N=range(4))))
            model.lazy_updater()
            model.lazy_updater = None

    @updater
    def _set_iders(model, iders=None):
        """ sets iders """
        if VERBOSE:
            print('[APIItemModel] _set_iders')
        if iders is None:
            iders = []
        if utool.USE_ASSERT:
            assert utool.is_list(iders), 'bad type: %r' % type(iders)
            for index, ider in enumerate(iders):
                assert utool.is_funclike(ider), 'bad type at index %r: %r' % (index, type(ider))
        #printDBG('NEW IDER')
        model.iders = iders

    @updater
    def _set_col_name_type(model, col_name_list=None, col_type_list=None):
        if VERBOSE:
            print('[APIItemModel] _set_col_name_type')
        if col_name_list is None:
            col_name_list = []
        if col_type_list is None:
            col_type_list = []
        if utool.USE_ASSERT:
            assert len(col_name_list) == len(col_type_list), \
                'inconsistent colnametype'
        model.col_name_list = col_name_list
        model.col_type_list = col_type_list

    @updater
    def _set_col_nice(model, col_nice_list=None):
        if col_nice_list is None:
            col_nice_list = model.col_name_list[:]
        if utool.USE_ASSERT:
            assert len(model.col_name_list) == len(col_nice_list), \
                'inconsistent colnice'
        model.col_nice_list = col_nice_list

    @default_method_decorator
    def _set_col_edit(model, col_edit_list=None):
        if col_edit_list is None:
            col_edit_list = [False] * len(model.col_name_list)
        if utool.USE_ASSERT:
            assert len(model.col_name_list) == len(col_edit_list), \
                'inconsistent coledit'
        model.col_edit_list = col_edit_list

    @default_method_decorator
    def _set_col_setter(model, col_setter_list=None):
        if VERBOSE:
            print('[APIItemModel] _set_col_setter')
        if col_setter_list is None:
            col_setter_list = []
        if utool.USE_ASSERT:
            assert len(model.col_name_list) == len(col_setter_list), \
                'inconsistent colsetter'
        model.col_setter_list = col_setter_list

    @default_method_decorator
    def _set_col_getter(model, col_getter_list=None):
        if VERBOSE:
            print('[APIItemModel] _set_col_getter')
        if col_getter_list is None:
            col_getter_list = []
        if utool.USE_ASSERT:
            assert len(model.col_name_list) == len(col_getter_list), \
                'inconsistent colgetter'
        model.col_getter_list = col_getter_list

    @default_method_decorator
    def _set_col_bgrole_getter(model, col_bgrole_getter_list=None):
        """ background rolegetter will be used for metadata like column color """
        if col_bgrole_getter_list is None:
            model.col_bgrole_getter_list = [None] * len(model.col_name_list)
        else:
            if utool.USE_ASSERT:
                assert len(col_bgrole_getter_list) == len(model.col_name_list), \
                    'inconsistent col_bgrole_getter_list'
            model.col_bgrole_getter_list = col_bgrole_getter_list

    @default_method_decorator
    def _set_col_visible_list(model, col_visible_list=None):
        """ used to turn columns off dynamically """
        if col_visible_list is None:
            model.col_visible_list = [True] * len(model.col_name_list)
        else:
            if utool.USE_ASSERT:
                assert len(col_visible_list) == len(model.col_name_list), \
                    'inconsistent col_visible_list'
            model.col_visible_list = col_visible_list

    @default_method_decorator
    def _set_col_level(model, col_level_list=None):
        if VERBOSE:
            print('[APIItemModel] _set_col_level')
        if col_level_list is None:
            col_level_list = [0] * len(model.col_name_list)
        if utool.USE_ASSERT:
            assert len(model.col_name_list) == len(col_level_list), \
                'inconsistent collevel'
        model.col_level_list = col_level_list

    @updater
    def _set_sort(model, col_sort_index, col_sort_reverse=False, rebuild_structure=False):
        if VERBOSE:
            print('[APIItemModel] _set_sort')
        #with utool.Timer('set_sort'):
        #printDBG('SET SORT')
        if len(model.col_name_list) > 0:
            if utool.USE_ASSERT:
                assert isinstance(col_sort_index, int) and col_sort_index < len(model.col_name_list), \
                    'sort index out of bounds by: %r' % col_sort_index
            model.col_sort_index = col_sort_index
            model.col_sort_reverse = col_sort_reverse
            # Update the row-id order
            model._update_rows(rebuild_structure=rebuild_structure)

    #------------------------------------
    # --- Data maintainence functions ---
    #------------------------------------

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
            #printDBG('CHANGE: CACHE INVALIDATED!')
            model.cache = {}
            model.layoutChanged.emit()
            return True
        else:
            #printDBG('NOT CHANGING')
            #print('NOT LAYOU CHANGED: %r, caller=%r' % (model.name, utool.get_caller_name(N=N)))
            return False

    @default_method_decorator
    def _update(model, newrows=False):
        #if newrows:
        model._update_rows()
        #printDBG('UPDATE: CACHE INVALIDATED!')
        model.cache = {}

    #def _use_ider(model, level):
    #    if level == 0:
    #        return model.iders[level]()
    #    else:
    #        return model.iders[level](model._use_ider(level - 1))

    def _use_ider(model, level=0):
        if level == 0:
            return model.iders[level]()
        else:
            parent_ids = model._use_ider(level - 1)
            level_ider = model.iders[level]
            return level_ider(parent_ids)

    def get_row_from_id(model, _id):
        r"""
        returns the row if an _id from the iders list

        Args:
            _id (?):

        Returns:
            int: row
        """
        row = model.root_node.find_row_from_id(_id)
        return row

    #----------------------------------
    # --- API Convineince Functions ---
    #----------------------------------

    @default_method_decorator
    def get_header_data(model, colname, qtindex):
        """ Use _get_data if the column number is known """
        # <HACK>
        # Hacked to only work on tables. Should be in terms of qtindex
        row = qtindex.row()
        if utool.USE_ASSERT:
            assert max(model.col_level_list) == 0, "Must be a table. Input is a tree"
        col = model.col_name_list.index(colname)
        id_ = model.root_node[row].get_id()
        getter = model.col_getter_list[col]
        value = getter(id_)
        return value
        # </HACK>

    @default_method_decorator
    def get_header_name(model, column):
        # TODO: use qtindex?
        colname = model.col_name_list[column]
        return colname

    @default_method_decorator
    def _get_level(model, qtindex):
        node = qtindex.internalPointer()
        level = node.get_level()
        #level = model.col_level_list[column]
        return level

    #--------------------------------
    # --- API Interface Functions ---
    #--------------------------------

    @default_method_decorator
    def _get_col_align(model, col):
        if utool.USE_ASSERT:
            assert col is not None, 'bad column'
        raise NotImplementedError('_get_col_align')

    @default_method_decorator
    def _get_row_id(model, qtindex=QtCore.QModelIndex()):
        if qtindex.isValid():
            node = qtindex.internalPointer()
            if utool.USE_ASSERT:
                try:
                    assert isinstance(node, _atn.TreeNode), 'type(node)=%r, node=%r' % (type(node), node)
                except AssertionError as ex:
                    utool.printex(ex, 'error in _get_row_id', keys=['model', 'qtindex', 'node'])
                    raise
            try:
                id_ = node.get_id()
            except AttributeError as ex:
                utool.printex(ex, key_list=['node', 'model', 'qtindex'])
                raise
            return id_

    @default_method_decorator
    def _get_adjacent_qtindex(model, qtindex=QtCore.QModelIndex(), offset=1):
        # check qtindex
        if not qtindex.isValid():
            return None
        node = qtindex.internalPointer()
        # check node
        try:
            if utool.USE_ASSERT:
                assert isinstance(node, _atn.TreeNode), type(node)
        except AssertionError as ex:
            utool.printex(ex, key_list=['node'], separate=True)
            raise
        # get node parent
        try:
            node_parent = node.get_parent()
        except Exception as ex:
            utool.printex(ex, key_list=['node'], reraise=False, separate=True)
            raise
        # parent_node check
        if node_parent is None:
            print('[model._get_adjacent_qtindex] node_parent is None!')
            return None
        # Offset to find the next qtindex
        next_index = node_parent.child_index(node) + offset
        nChildren = node_parent.get_num_children()
        # check next index validitiy
        if next_index >= 0 and next_index < nChildren:
            next_node = node_parent.get_child(next_index)
            next_level = next_node.get_level()
            col = model.col_level_list.index(next_level)
            row = next_node.get_row()
            # Create qtindex for the adjacent note
            parent_qtindex = model.parent(qtindex)
            next_qtindex = model.index(row, col, parent_qtindex)
            return next_qtindex
        else:
            # There is no adjacent node
            return None

    @default_method_decorator
    def _get_type(model, col):
        return model.col_type_list[col]

    @default_method_decorator
    def _get_bgrole_value(model, qtindex):
        """ Gets the background role if specified """
        col = qtindex.column()
        bgrole_getter = model.col_bgrole_getter_list[col]
        if bgrole_getter is None:
            return None
        row_id = model._get_row_id(qtindex)  # row_id w.r.t. to sorting
        color = bgrole_getter(row_id)
        if color is None:
            return None
        val = qtype.to_qcolor(color)
        return val

    @default_method_decorator
    def _get_data(model, qtindex, **kwargs):
        #row = qtindex.row()
        col = qtindex.column()
        row_id = model._get_row_id(qtindex)  # row_id w.r.t. to sorting
        getter = model.col_getter_list[col]  # getter for this column
        # Using this getter may not be thread safe
        try:
            # Should this work around decorators?
            #data = getter((row_id,), **kwargs)[0]
            data = getter(row_id, **kwargs)
        except Exception as ex:
            utool.printex(ex, 'problem getting in column %r' % (col,))
            #getting from: %r' % utool.util_str.get_callable_name(getter))
            raise
        # <HACK: MODEL_CACHE>
        #cachekey = (row_id, col)
        #try:
        #    if True:  # Cache is disabled
        #        raise KeyError('')
        #    #data = model.cache[cachekey]
        #except KeyError:
        #    data = getter(row_id)
        #    #model.cache[cachekey] = data
        # </HACK: MODEL_CACHE>
        return data

    @default_method_decorator
    def _set_data(model, qtindex, value):
        """ The setter function should be of the following format def
        setter(column_name, row_id, value) column_name is the key or SQL-like
        name for the column row_id is the corresponding row key or SQL-like id
        that the row call back returned value is the value that needs to be
        stored The setter function should return a boolean, if setting the value
        was successfull or not """
        col = qtindex.column()
        row_id = model._get_row_id(qtindex)
        # <HACK: MODEL_CACHE>
        #cachekey = (row_id, col)
        #try:
        #    del model.cache[cachekey]
        #except KeyError:
        #    pass
        # </HACK: MODEL_CACHE>
        setter = model.col_setter_list[col]
        if VERBOSE:
            print('[model] Setting data: row_id=%r, setter=%r' % (row_id, setter))
        try:
            return setter(row_id, value)
        except Exception as ex:
            ut.printex(ex, 'ERROR: setting data: row_id=%r, setter=%r' % (row_id, setter))
            raise

    #------------------------
    # --- QtGui Functions ---
    #------------------------
    @default_method_decorator
    def parent(model, qindex):
        """
        A common convention used in models that expose tree data structures is
        that only items in the first column have children. For that case, when
        reimplementing this function in a subclass the column of the returned
        QModelIndex would be 0.

        When reimplementing this function in a subclass, be careful to avoid
        calling QModelIndex member functions, such as QModelIndex.parent(),
        since indexes belonging to your model will simply call your
        implementation, leading to infinite recursion.

        Returns:
            the parent of the model item with the given index. If the item has
            no parent, an invalid QModelIndex is returned.
        """

        model.lazy_checks()
        if qindex.isValid():
            node = qindex.internalPointer()
            #<HACK>
            if not isinstance(node, _atn.TreeNode):
                print("WARNING: tried to access parent of %r type object" % type(node))
                return QtCore.QModelIndex()
            #assert node.__dict__, "node.__dict__=%r" % node.__dict__
            #</HACK>
            parent_node = node.get_parent()
            parent_id = parent_node.get_id()
            if parent_id == -1 or parent_id is None:
                return QtCore.QModelIndex()
            row = parent_node.get_row()
            col = model.col_level_list.index(parent_node.get_level())
            return model.createIndex(row, col, parent_node)
        return QtCore.QModelIndex()

    @default_method_decorator
    def index(model, row, column, parent=QtCore.QModelIndex()):
        """
        Qt Override

        Returns:
            the index of the item in the model specified by the given row,
            column and parent index.  When reimplementing this function in a
            subclass, call createIndex() to generate model indexes that other
            components can use to refer to items in your model.

        NOTE:
            Object must be specified to sort delegates.
        """
        model.lazy_checks()
        if not parent.isValid():
            # This is a top level == 0 index
            #print('[model.index] ROOT: row=%r, col=%r' % (row, column))
            if row >= model.root_node.get_num_children():
                return QtCore.QModelIndex()
                #import traceback
                #traceback.print_stack()
            node = model.root_node[row]
            if model.col_level_list[column] != node.get_level():
                return QtCore.QModelIndex()
            qtindex = model.createIndex(row, column, object=node)
            return qtindex
        else:
            # This is a child level > 0 index
            parent_node = parent.internalPointer()
            node = parent_node[row]
            if utool.USE_ASSERT:
                assert isinstance(parent_node, _atn.TreeNode), type(parent_node)
                assert isinstance(node, _atn.TreeNode), type(node)
            return model.createIndex(row, column, object=node)

    @default_method_decorator
    def rowCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        #model.lazy_checks()
        if not parent.isValid():
            # Root row count
            if len(model.level_index_list) == 0:
                return 0
            nRows = len(model.level_index_list)
            #print('* nRows=%r' % nRows)
            return nRows
        else:
            node = parent.internalPointer()
            nRows = node.get_num_children()
            #print('+ nRows=%r' % nRows)
            return nRows

    @default_method_decorator
    def columnCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        # FOR NOW THE COLUMN COUNT IS CONSTANT
        model.lazy_checks()
        return len(model.col_name_list)

    @default_method_decorator
    def data(model, qtindex, role=Qt.DisplayRole, **kwargs):
        """
        Depending on the role, returns either data or how to display data
        Returns the data stored under the given role for the item referred to by
        the index.

        Note:
            If you do not have a value to return, return None
        """
        if not qtindex.isValid():
            return None
        flags = model.flags(qtindex)
        #row = qtindex.row()
        col = qtindex.column()
        node = qtindex.internalPointer()
        if model.col_level_list[col] != node.get_level():
            return QVariantHack()
        type_ = model._get_type(col)

        #if row >= model.rowCount():
        #    # Yuck.
        #    print('[item_model] Yuck. row=%r excedes rowCount=%r' %
        #          (row, model.rowCount()))
        #    return QVariantHack()

        #if role == Qt.SizeHintRole:
        #    #printDBG('REQUEST QSIZE FOR: ' + qtype.ItemDataRoles[role])
        #    return QtCore.QSize(64, 64)
        #
        # Specify Text Alignment Role
        if role == Qt.TextAlignmentRole:
            if type_ in qtype.QT_IMAGE_TYPES:
                value = Qt.AlignRight | Qt.AlignVCenter
            elif type_ in qtype.QT_BUTTON_TYPES:
                value = Qt.AlignRight | Qt.AlignVCenter
            elif type_ in utool.VALID_FLOAT_TYPES:
                value = Qt.AlignRight | Qt.AlignVCenter
            else:
                value = Qt.AlignHCenter | Qt.AlignVCenter
            return value
        #
        # Specify Background Rule
        elif role == Qt.BackgroundRole:
            value = model._get_bgrole_value(qtindex)
            if value is not None:
                return value
            if flags & Qt.ItemIsEditable:
                # Editable fields are colored
                return QVariantHack(model.EditableItemColor)
            elif flags & Qt.ItemIsUserCheckable:
                # Checkable color depends on the truth value
                data = model._get_data(qtindex, **kwargs)
                if data:
                    return QVariantHack(model.TrueItemColor)
                else:
                    return QVariantHack(model.FalseItemColor)
            else:
                pass
        #
        # Specify Foreground Role
        elif role == Qt.ForegroundRole:
            if flags & Qt.ItemIsEditable:
                return QtGui.QBrush(QtGui.QColor(0, 0, 0))

        # Specify Decoration Role (superceded by thumbdelegate)
        # elif role == Qt.DecorationRole and type_ in qtype.QT_IMAGE_TYPES:

        # Specify CheckState Role:
        if role == Qt.CheckStateRole:
            if flags & Qt.ItemIsUserCheckable:
                data = model._get_data(qtindex, **kwargs)
                return Qt.Checked if data else Qt.Unchecked
        #
        # Return the data to edit or display
        elif role in (Qt.DisplayRole, Qt.EditRole):
            # For types displayed with custom delegates do not cast data into a
            # qvariant. This includes PIXMAP, BUTTON, and COMBO
            if type_ in qtype.QT_DELEGATE_TYPES:
                data = model._get_data(qtindex, **kwargs)
                #print(data)
                return data
            else:
                # Display data with default delegate by casting to a qvariant
                data = model._get_data(qtindex, **kwargs)
                value = qtype.cast_into_qt(data)
                return value
        else:
            #import builtins
            #role_name = qtype.ItemDataRoles[role]
            #builtins.print('UNHANDLED ROLE=%r' % role_name)
            pass
        # else return None
        return QVariantHack()

    @default_method_decorator
    def setData(model, qtindex, value, role=Qt.EditRole):
        """
        Sets the role data for the item at qtindex to value.  value is a
        QVariant (called data in documentation) Returns a map with values for
        all predefined roles in the model for the item at the given index.
        Reimplement this function if you want to extend the default behavior of
        this function to include custom roles in the map.
        """
        try:
            if not qtindex.isValid():
                return None
            flags = model.flags(qtindex)
            #row = qtindex.row()
            col = qtindex.column()
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
            old_data = model._get_data(qtindex)
            if old_data != data:
                model._set_data(qtindex, data)
                # Emit that data was changed and return succcess
                model.dataChanged.emit(qtindex, qtindex)
            return True
        except Exception as ex:
            #value = str(value.toString())  # NOQA
            utool.printex(ex, 'ignoring setData', '[model]', tb=True,
                          key_list=['value'], iswarning=True)
            return False

    @default_method_decorator
    def headerData(model, section, orientation, role=Qt.DisplayRole):
        """
        Qt Override

        Returns:
            the data for the given role and section in the header with the
            specified orientation.  For horizontal headers, the section number
            corresponds to the column number. Similarly, for vertical headers,
            the section number corresponds to the row number.
        """
        model.lazy_checks()
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            column = section
            if column >= len(model.col_nice_list):
                return []
            return model.col_nice_list[column]
        #if orientation == Qt.Vertical and role == Qt.DisplayRole:
        #    row = section
        #    rowid = model._get_row_id(row)
        #    return rowid
        return QVariantHack()

    @updater
    def sort(model, column, order):
        """ Qt Override """
        model.lazy_checks()
        reverse = (order == QtCore.Qt.DescendingOrder)
        model._set_sort(column, reverse)

    @default_method_decorator
    def flags(model, qtindex):
        """
        Qt Override

        Returns:
            Qt::ItemFlag::
                 0: 'NoItemFlags'          # It does not have any properties set.
                 1: 'ItemIsSelectable'     # It can be selected.
                 2: 'ItemIsEditable'       # It can be edited.
                 4: 'ItemIsDragEnabled'    # It can be dragged.
                 8: 'ItemIsDropEnabled'    # It can be used as a drop target.
                16: 'ItemIsUserCheckable'  # It can be checked or unchecked by the user.
                32: 'ItemIsEnabled'        # The user can interact with the item.
                64: 'ItemIsTristate'       # The item is checkable with three separate states.
        """
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
