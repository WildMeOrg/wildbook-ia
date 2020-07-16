# -*- coding: utf-8 -*-
"""
TODO:
    Fix slowness
    Fix sorting so columns are initially sorted in ascending order


"""
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtCore, QtGui, QVariantHack
from wbia.guitool.__PYQT__.QtCore import Qt
from wbia.guitool import qtype
from wbia.guitool.guitool_decorators import checks_qt_error, signal_  # NOQA
from six.moves import zip  # builtins  # NOQA

# from utool._internal.meta_util_six import get_funcname
import functools
import utool as ut

# from .api_thumb_delegate import APIThumbDelegate
import numpy as np
from wbia.guitool import api_tree_node as _atn
import cachetools

# UTOOL PRINT STATEMENTS CAUSE RACE CONDITIONS IN QT THAT CAN LEAD TO SEGFAULTS
# DO NOT INJECT THEM IN GUITOOL
# print, rrr, profile = ut.inject2(__name__)
ut.noinject(__name__, '[APIItemModel]')

# raise ImportError('refused to import wbia.guitool')
profile = ut.profile

API_MODEL_BASE = QtCore.QAbstractItemModel

VERBOSE_MODEL = ut.VERBOSE or ut.get_argflag(('--verbose-qt', '--verbqt'))
VERBOSE_MODEL = VERBOSE_MODEL or ut.get_argflag(('--verbose-qt-api', '--verbqt-api'))


class ChangeLayoutContext(object):
    """
    Context manager emitting layoutChanged before body,
    not updating durring body, and then updating after body.
    """

    @ut.accepts_scalar_input
    def __init__(self, model_list, *args):
        # print('Changing: %r' % (model_list,))
        self.model_list = list(model_list) + list(args)

    def __enter__(self):
        for model in self.model_list:
            if model._get_context_id() is not None:
                # print("[ChangeLayoutContext] WARNING: ENTERING CONTEXT TWICE")
                continue
            model._set_context_id(id(self))
            # print("[ChangeLayoutContext] ENTERING CONTEXT, context_id: %r" % (model._get_context_id(), ))
            model._about_to_change()
            # isabouttochange = model._about_to_change()
            # print("... isabouttochange = %r" % (isabouttochange,))
            model._set_changeblocked(True)
        return self

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('[api_model] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        for model in self.model_list:
            if model._get_context_id() == id(self):
                # print("[ChangeLayoutContext] EXITING CONTEXT, context_id: %r" % (id(self), ))
                model._set_context_id(None)
                model._set_changeblocked(False)
                model._change()
                # didchange = model._change()
                # print("... didchange = %r" % (didchange,))


def default_method_decorator(func):
    """ Dummy decorator """
    # return profile(func)
    # return checks_qt_error(profile(func))
    return func


def updater(func):
    """
    Decorates a function by executing layoutChanged signals if not already in
    the middle of a layout changed
    """
    func_ = default_method_decorator(func)

    # @checks_qt_error
    @functools.wraps(func)
    def upd_wrapper(model, *args, **kwargs):
        with ChangeLayoutContext([model]):
            return func_(model, *args, **kwargs)

    return upd_wrapper


class APIItemModel(API_MODEL_BASE):
    """
    Item model for displaying a list of columns

    Attributes:
        iders         (list) : functions that return ids for setters and getters
        col_name_list (list) : keys or SQL-like name for column to reference
                        abstracted data storage using getters and setters
        col_type_list (list) : column value (Python) types
        col_nice_list (list) : well-formatted names of the columns
        col_edit_list (list) : booleans for if column should be editable

        col_setter_list (list) : setter functions
        col_getter_list (list) : getter functions

        col_sort_index (int) : index into col_name_list for sorting
        col_sort_reverse (bool) : flag to reverse the sort ordering
    """

    _rows_updated = signal_(str, int)
    EditableItemColor = QtGui.QColor(242, 242, 255)
    # EditableItemColor = QtGui.QColor(200, 200, 255)
    TrueItemColor = QtGui.QColor(230, 250, 230)
    FalseItemColor = QtGui.QColor(250, 230, 230)

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
        if VERBOSE_MODEL:
            print('[APIItemModel] __init__')
        # FIXME: don't let the model point to the view
        model.view = parent
        API_MODEL_BASE.__init__(model, parent=parent)
        # Internal Flags
        model._abouttochange = False
        model._context_id = None
        model._haschanged = True
        model._changeblocked = False
        # Model Data And Accessors
        model.name = 'None'
        model.nice = 'None'
        model.iders = [lambda: []]
        model.col_visible_list = []
        model.col_name_list = []
        model.col_type_list = []
        model.col_nice_list = []
        model.col_edit_list = []
        model.col_setter_list = []
        model.col_getter_list = []
        model.col_level_list = []
        model.col_bgrole_getter_list = None
        model.col_sort_index = None
        model.col_sort_reverse = False
        model.level_index_list = []
        model.cache = None  # FIXME: This is not sustainable
        model.cache_timeout_sec = 2.5
        model.cache_size = 512
        model.batch_size = None  # Small batch sizes give good response time
        model.scope_hack_list = []
        model.root_node = _atn.TreeNode(-1, None, -1)
        # Initialize member variables
        # model._about_to_change()
        model.headers = headers  # save the headers
        model.ider_filters = None
        model.num_rows_loaded = 0
        model.num_rows_total = None
        # len(model.level_index_list)
        # model.lazy_updater = None
        if headers is not None:
            model._update_headers(**headers)

    def set_ider_filters(model, ider_filters):
        """  Used to induce a filter on the rows, needs call of udpate rows after """
        model.ider_filters = ider_filters

    def get_iders(model):
        # def filtfun_test(x_list):
        #    return [x for x in x_list if x % 2 == 0]
        # model.name == 'annotations'
        # if len(model.iders) == 1:
        #    model.ider_filters = [filtfun_test]
        if model.ider_filters is None:
            ider_list = model.iders
        else:
            assert len(model.ider_filters) == len(model.iders), 'bad filters'
            # ider_list =  [lambda: filtfn(ider()) for filtfn, ider in zip(model.ider_filters, model.iders)]
            # with ut.embed_on_exception_context:

            def wrap_ider(ider, filtfn):
                def wrapped_ider(*args, **kwargs):
                    return filtfn(ider(*args, **kwargs))

                return wrapped_ider

            ider_list = [
                # ider
                wrap_ider(ider, filtfn)
                # lambda *args: filtfn(ider(*args))
                for filtfn, ider in zip(model.ider_filters, model.iders)
            ]
        return ider_list

    @updater
    def _update_headers(model, **headers):
        if VERBOSE_MODEL:
            print('[APIItemModel] _update_headers')
        iders = headers.get('iders', None)
        name = headers.get('name', None)
        nice = headers.get('nice', None)
        col_name_list = headers.get('col_name_list', None)
        col_type_list = headers.get('col_type_list', None)
        col_nice_list = headers.get('col_nice_list', None)
        col_edit_list = headers.get('col_edit_list', None)
        col_setter_list = headers.get('col_setter_list', None)
        col_getter_list = headers.get('col_getter_list', None)
        col_level_list = headers.get('col_level_list', None)
        col_sort_index = headers.get('col_sort_index', 0)
        col_sort_reverse = headers.get('col_sort_reverse', False)
        # New for dynamically getting non-data roles for each row
        col_bgrole_getter_list = headers.get('col_bgrole_getter_list', None)
        col_visible_list = headers.get('col_visible_list', None)
        #
        if iders is None:
            iders = []
        if ut.USE_ASSERT:
            assert ut.is_list(iders), 'bad type: %r' % type(iders)
            for index, ider in enumerate(iders):
                assert ut.is_funclike(ider), 'bad type at index %r: %r' % (
                    index,
                    type(ider),
                )

        if col_name_list is None:
            col_name_list = []
        if col_type_list is None:
            col_type_list = []
        if col_nice_list is None:
            col_nice_list = col_name_list[:]
        if col_edit_list is None:
            col_edit_list = [False] * len(col_name_list)
        if col_setter_list is None:
            col_setter_list = []
        if col_getter_list is None:
            col_getter_list = []
        if col_bgrole_getter_list is None:
            col_bgrole_getter_list = [None] * len(col_name_list)
        if col_visible_list is None:
            col_visible_list = [True] * len(col_name_list)
        if col_level_list is None:
            col_level_list = [0] * len(col_name_list)

        if True or ut.USE_ASSERT:
            assert len(col_name_list) == len(col_type_list), 'inconsistent colnametype'
            assert len(col_name_list) == len(col_nice_list), 'inconsistent colnice'
            assert len(col_name_list) == len(col_edit_list), 'inconsistent coledit'
            assert len(col_name_list) == len(col_setter_list), 'inconsistent colsetter'
            assert len(col_bgrole_getter_list) == len(
                col_name_list
            ), 'inconsistent col_bgrole_getter_list'
            assert len(col_name_list) == len(col_getter_list), 'inconsistent colgetter'
            assert len(col_visible_list) == len(
                col_name_list
            ), 'inconsistent col_visible_list'
            assert len(col_name_list) == len(col_level_list), 'inconsistent collevel'
            for colname, flag, func in zip(col_name_list, col_edit_list, col_setter_list):
                if flag:
                    assert func is not None, 'column=%r is editable but func is None' % (
                        colname,
                    )

        model.clear_cache()

        model.name = str(name)
        model.nice = str(nice)
        model.iders = iders
        model.col_name_list = col_name_list
        model.col_type_list = col_type_list
        model.col_nice_list = col_nice_list
        model.col_edit_list = col_edit_list
        model.col_setter_list = col_setter_list
        model.col_getter_list = col_getter_list
        model.col_visible_list = col_visible_list
        model.col_level_list = col_level_list
        model.col_bgrole_getter_list = col_bgrole_getter_list
        model.col_display_role_func_dict = headers.get('col_display_role_func_dict', None)

        model.num_rows_loaded = 0
        # model.num_cols_loaded = 0
        model.num_rows_total = None
        model.lazy_rows = True

        # calls model._update_rows()
        model._set_sort(col_sort_index, col_sort_reverse, rebuild_structure=True)

    def clear_cache(model):
        model.cache = cachetools.TTLCache(
            maxsize=model.cache_size, ttl=model.cache_timeout_sec
        )

    @updater
    def _set_sort(model, col_sort_index, col_sort_reverse=False, rebuild_structure=False):
        if VERBOSE_MODEL:
            print(
                '[APIItemModel] _set_sort, index=%r reverse=%r, rebuild=%r'
                % (col_sort_index, col_sort_reverse, rebuild_structure,)
            )
        if len(model.col_name_list) > 0:
            if ut.USE_ASSERT:
                assert isinstance(col_sort_index, int) and col_sort_index < len(
                    model.col_name_list
                ), ('sort index out of bounds by: %r' % col_sort_index)
            model.col_sort_index = col_sort_index
            model.col_sort_reverse = col_sort_reverse
            # Update the row-id order
            model._update_rows(rebuild_structure=rebuild_structure)

    @updater
    def _update_rows(model, rebuild_structure=True):
        """
        Uses the current ider and col_sort_index to create
        row_indices
        """
        # with ut.Timer('[gt] update_rows (%s)' % (model.name,)):
        if True:
            # flag = model.blockSignals(True)
            if VERBOSE_MODEL:
                print('[APIItemModel] +-----------')
                print('[APIItemModel] _update_rows')
            # this is not slow
            # print('UPDATE ROWS!')
            if len(model.col_level_list) == 0:
                return
            # old_root = model.root_node  # NOQA
            if rebuild_structure:
                # print('Rebuilging api_item_model internal structure')
                model.beginResetModel()  # I think this is preventing a segfault
                model.root_node = _atn.build_internal_structure(model)
                model.endResetModel()
            if VERBOSE_MODEL:
                print('[APIItemModel] lazy_update_rows')
            model.level_index_list = []
            sort_index = 0 if model.col_sort_index is None else model.col_sort_index
            children = (
                model.root_node.get_children()
            )  # THIS IS THE LINE THAT TAKES FOREVER
            id_list = [child.get_id() for child in children]
            # print('ids_ generated')
            nodes = []
            if len(id_list) != 0:
                if VERBOSE_MODEL:
                    print(
                        '[APIItemModel] lazy_update_rows len(id_list) = %r'
                        % (len(id_list))
                    )
                # start sort
                if model.col_sort_index is not None:
                    type_ = model.col_type_list[sort_index]
                    getter = model.col_getter_list[sort_index]
                    values = getter(id_list)
                    if type_ == 'PIXMAP':
                        # TODO: find a better sorting metric for pixmaps
                        values = ut.get_list_column(values, 0)
                else:
                    type_ = int
                    values = id_list
                reverse = model.col_sort_reverse

                # <NUMPY MULTIARRAY SORT>
                if True:
                    if values is None:
                        print('SORTING VALUES IS NONE. VERY WEIRD')
                    if type_ is float:
                        values = np.array(ut.replace_nones(values, np.nan))
                        # Force nan to be the smallest number
                        values[np.isnan(values)] = -np.inf
                    elif type_ is str:
                        values = ut.replace_nones(values, '')
                    import vtool as vt

                    sortx = vt.argsort_records([values, id_list], reverse=reverse)
                    # </NUMPY MULTIARRAY SORT>
                    nodes = ut.take(children, sortx)
                    level = model.col_level_list[sort_index]
                    if level == 0:
                        model.root_node.set_children(nodes)
                    # end sort
            if ut.USE_ASSERT:
                assert nodes is not None, 'no indices'
            model.level_index_list = nodes

            # Book keeping for lazy loading rows
            model.num_rows_total = len(model.level_index_list)
            # model.num_cols_total = len(model.col_name_list)
            model.num_cols_loaded = 0

            if model.lazy_rows:
                model.num_rows_loaded = 0
            else:
                model.num_rows_loaded = model.num_rows_total
            # emit the numerr of rows and the name of for the view to display
            # model.blockSignals(flag)
            model._rows_updated.emit(model.name, model.num_rows_total)
            if VERBOSE_MODEL:
                print('[APIItemModel] finished _update_rows')
                print('[APIItemModel] L__________')

    # ------------------------------------
    # --- Data maintainence functions ---
    # ------------------------------------

    @default_method_decorator
    def _about_to_change(model, force=False):
        if force or (not model._abouttochange and not model._changeblocked):
            if VERBOSE_MODEL:
                print('ABOUT TO CHANGE: %r' % (model.name,))
            model._abouttochange = True
            model.layoutAboutToBeChanged.emit()
            return True
        else:
            if VERBOSE_MODEL:
                print('NOT ABOUT TO CHANGE')
            return False

    @default_method_decorator
    def _change(model, force=False):
        if force or (model._abouttochange and not model._changeblocked):
            if VERBOSE_MODEL:
                print('LAYOUT CHANGED:  %r' % (model.name,))
            model._abouttochange = False
            model.clear_cache()
            model.layoutChanged.emit()
            return True
        else:
            if VERBOSE_MODEL:
                print('NOT LAYOUT CHANGING')
            return False

    @default_method_decorator
    def _update(model, newrows=False):
        model.cache = {}
        model._update_rows()

    def _use_ider(model, level=0):
        if level == 0:
            return model.iders[level]()
        else:
            parent_ids = model._use_ider(level - 1)
            level_ider = model.iders[level]
            return level_ider(parent_ids)

    def get_row_and_qtindex_from_id(model, _id):
        """ uses an sqlrowid (from iders) to get a qtindex """
        row = model.root_node.find_row_from_id(_id)
        qtindex = model.index(row, 0) if row is not None else None
        return qtindex, row

    # ----------------------------------
    # --- API Convineince Functions ---
    # ----------------------------------

    @default_method_decorator
    def get_header_data(model, colname, qtindex):
        """ Use _get_data if the column number is known """
        if not qtindex.isValid():
            return None
        # row = qtindex.row()
        node = qtindex.internalPointer()
        col = model.col_name_list.index(colname)
        getter = model.col_getter_list[col]
        id_ = node.id_
        # id_ = model.root_node[row].get_id()
        value = getter(id_)
        return value

    @default_method_decorator
    def get_header_name(model, column):
        # TODO: use qtindex?
        colname = model.col_name_list[column]
        return colname

    @default_method_decorator
    def _get_level(model, qtindex):
        node = qtindex.internalPointer()
        if node is None:
            return -1
        level = node.get_level()
        # level = model.col_level_list[column]
        return level

    # --------------------------------
    # --- API Interface Functions ---
    # --------------------------------

    @default_method_decorator
    def _get_col_align(model, col):
        if ut.USE_ASSERT:
            assert col is not None, 'bad column'
        raise NotImplementedError('_get_col_align')

    @default_method_decorator
    def _get_row_id(model, qtindex=QtCore.QModelIndex()):
        """
        returns the id (specified by iders i.e. an wbia rowid) from qtindex
        """
        if qtindex is not None and qtindex.isValid():
            node = qtindex.internalPointer()
            if ut.USE_ASSERT:
                try:
                    assert isinstance(node, _atn.TreeNode), 'type(node)=%r, node=%r' % (
                        type(node),
                        node,
                    )
                except AssertionError as ex:
                    ut.printex(
                        ex, 'error in _get_row_id', keys=['model', 'qtindex', 'node']
                    )
                    raise
            try:
                id_ = node.get_id()
            except AttributeError as ex:
                ut.printex(ex, key_list=['node', 'model', 'qtindex'])
                raise
            return id_

    @default_method_decorator
    def _get_adjacent_qtindex(model, qtindex=QtCore.QModelIndex(), offset=1):
        # check qtindex
        if qtindex is None or not qtindex.isValid():
            return None
        node = qtindex.internalPointer()
        # check node
        try:
            if ut.USE_ASSERT:
                assert isinstance(node, _atn.TreeNode), type(node)
        except AssertionError as ex:
            ut.printex(ex, key_list=['node'], pad_stdout=True)
            raise
        # get node parent
        try:
            node_parent = node.get_parent()
        except Exception as ex:
            ut.printex(ex, key_list=['node'], reraise=False, pad_stdout=True)
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
        col = qtindex.column()
        # row_id wrt. to sorting
        row_id = model._get_row_id(qtindex)
        cachekey = (row_id, col)
        try:
            data = model.cache[cachekey]
        except KeyError:
            # getter function for this column
            getter = model.col_getter_list[col]
            try:
                # Using this getter may not be thread safe
                # Should this work around decorators?
                # data = getter((row_id,), **kwargs)[0]
                data = getter(row_id, **kwargs)
            except Exception as ex:
                qtindex_rc = (qtindex.row(), qtindex.column())  # NOQA
                ut.printex(
                    ex,
                    '[api_item_model] problem getting in column %r' % (col,),
                    keys=[
                        'model.name',
                        'getter',
                        'row_id',
                        'col',
                        'qtindex',
                        'qtindex_rc',
                    ],
                    iswarning=True,
                )
                # getting from: %r' % ut.util_str.get_callable_name(getter))
                raise
            model.cache[cachekey] = data
        # </MODEL_CACHE>
        return data

    @default_method_decorator
    def _set_data(model, qtindex, value):
        """
        The setter function should be of the following format def
        setter(column_name, row_id, value) column_name is the key or SQL-like
        name for the column row_id is the corresponding row key or SQL-like id
        that the row call back returned value is the value that needs to be
        stored The setter function should return a boolean, if setting the
        value was successfull or not
        """
        col = qtindex.column()
        row_id = model._get_row_id(qtindex)
        # <HACK: MODEL_CACHE>
        cachekey = (row_id, col)
        try:
            del model.cache[cachekey]
        except KeyError:
            pass
        # </HACK: MODEL_CACHE>
        setter = model.col_setter_list[col]
        if VERBOSE_MODEL:
            print('[model] Setting data: row_id=%r, setter=%r' % (row_id, setter))
        try:
            return setter(row_id, value)
        except Exception as ex:
            ut.printex(
                ex,
                'ERROR: setting data: row_id=%r, setter=%r, col=%r'
                % (row_id, setter, col),
            )
            raise

    # ------------------------
    # --- QtGui Functions ---
    # ------------------------
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

        FIXME:
            seems to segfault in here
            https://riverbankcomputing.com/pipermail/pyqt/2016-February/036977.html
            https://gist.github.com/estan/c051d1f798c4c46caa7d

        Returns:
            the parent of the model item with the given index. If the item has
            no parent, an invalid QModelIndex is returned.
        """
        # model.lazy_checks()
        if qindex.isValid():
            try:
                node = qindex.internalPointer()
                # <HACK>
                # A segfault happens in isinstance when updating rows?
                if not isinstance(node, _atn.TreeNode):
                    print(
                        'WARNING: tried to access parent of %r type object' % type(node)
                    )
                    return QtCore.QModelIndex()
                # assert node.__dict__, "node.__dict__=%r" % node.__dict__
                # </HACK>
                parent_node = node.get_parent()
                parent_id = parent_node.get_id()
                if parent_id == -1 or parent_id is None:
                    return QtCore.QModelIndex()
                row = parent_node.get_row()
                col = model.col_level_list.index(parent_node.get_level())
                return model.createIndex(row, col, parent_node)
            except Exception as ex:
                import utool

                with utool.embed_on_exception_context:
                    qindex_rc = (qindex.row(), qindex.column())  # NOQA
                    ut.printex(
                        ex,
                        'failed to do parenty things',
                        keys=['qindex_rc', 'model.name'],
                        tb=True,
                    )
                import utool

                utool.embed()
                raise
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
        # model.lazy_checks()
        if not parent.isValid():
            # This is a top level == 0 index
            # print('[model.index] ROOT: row=%r, col=%r' % (row, column))
            if row >= model.root_node.get_num_children():
                return QtCore.QModelIndex()
                # import traceback
                # traceback.print_stack()
            node = model.root_node[row]
            if model.col_level_list[column] != node.get_level():
                return QtCore.QModelIndex()
            qtindex = model.createIndex(row, column, object=node)
            return qtindex
        else:
            # This is a child level > 0 index
            parent_node = parent.internalPointer()
            node = parent_node[row]
            if ut.USE_ASSERT:
                assert isinstance(parent_node, _atn.TreeNode), type(parent_node)
                assert isinstance(node, _atn.TreeNode), type(node)
            return model.createIndex(row, column, object=node)

    def _get_level_row_count(model, qtindex):
        return model.rowCount(qtindex.parent())

    def _get_level_row_index(model, qtindex):
        node = qtindex.internalPointer()
        return node.get_row()

    @default_method_decorator
    def rowCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        # model.lazy_checks()
        if not parent.isValid():
            # Root row count
            if len(model.level_index_list) == 0:
                return 0
            return model.num_rows_loaded
            # nRows = len(model.level_index_list)
            # # print('* nRows=%r' % nRows)
            # return nRows
        else:
            node = parent.internalPointer()
            nRows = node.get_num_children()
            # print('+ nRows=%r' % nRows)
            return nRows

    @default_method_decorator
    def columnCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        # FOR NOW THE COLUMN COUNT IS CONSTANT
        # model.lazy_checks()
        return len(model.col_name_list)

    @default_method_decorator
    def canFetchMore(model, parent=QtCore.QModelIndex()):
        """
        Returns true if there is more data available for parent; otherwise
        returns false.  The default implementation always returns false.  If
        canFetchMore() returns true, the fetchMore() function should be called.
        This is the behavior of QAbstractItemView, for example.


        References:
            http://doc.qt.io/qt-5/qtwidgets-itemviews-fetchmore-example.html
            # Extend this to work well with QTreeViews
            http://blog.tjwakeham.com/lazy-loading-pyqt-data-models/
            http://stackoverflow.com/questions/38506808/pyqt4-force-view-to-fetchmore-from
        """
        if parent is None:
            return
        if parent.isValid():
            # Check if we are at a leaf node
            node = parent.internalPointer()
            if node.get_num_children() == 0:
                return
            # if node.get_level() == len(model.col_level_list):
            #     return
        # print('model.num_rows_total = %r' % (model.num_rows_total,))
        # print('model.num_rows_loaded = %r' % (model.num_rows_loaded,))
        if model.num_rows_total is not None:
            if model.num_rows_loaded < model.num_rows_total:
                if VERBOSE_MODEL:
                    print('canFetchMore %s? -- Yes' % (model.name,))
                return True
        if VERBOSE_MODEL:
            print('canFetchMore %s? -- No' % (model.name,))
        return False
        # if not parent.isValid():
        #    return False
        # flags = model.flags(qtindex)
        # # row = qtindex.row()
        # col = qtindex.column()
        # node = qtindex.internalPointer()
        # return False

    @default_method_decorator
    def fetchMore(model, parent=QtCore.QModelIndex()):
        """
        Fetches any available data for the items with the parent specified by
        the parent index.

        Reimplement this if you are populating your model incrementally.
        The default implementation does nothing.
        """
        if parent is None:
            return
        if parent.isValid():
            # Check if we are at a leaf node
            node = parent.internalPointer()
            if node.get_num_children() == 0:
                return

        remainder = model.num_rows_total - model.num_rows_loaded
        if model.batch_size is None:
            num_fetching = remainder
        else:
            num_fetching = min(model.batch_size, remainder)
        if VERBOSE_MODEL:
            print('Fetching %r more %s' % (num_fetching, model.name))
        idx1 = model.num_rows_total
        idx2 = model.num_rows_total + num_fetching - 1
        # model.beginInsertRows(QtCore.QModelIndex(), idx1, idx2)
        model.beginInsertRows(parent, idx1, idx2)
        model.num_rows_loaded += num_fetching
        # print('model.num_rows_total = %r' % (model.num_rows_total,))
        # print('model.num_rows_loaded = %r' % (model.num_rows_loaded,))
        model.endInsertRows()
        if VERBOSE_MODEL:
            print('Fetched %r/%r rows' % (model.num_rows_loaded, model.num_rows_total))
        # model.numberPopulated.emit(num_loading)

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
        # row = qtindex.row()
        col = qtindex.column()
        node = qtindex.internalPointer()
        if model.col_level_list[col] != node.get_level():
            return QVariantHack()
        type_ = model._get_type(col)
        #
        # Specify Text Alignment Role
        if role == Qt.TextAlignmentRole:
            if type_ in qtype.QT_IMAGE_TYPES:
                value = Qt.AlignRight | Qt.AlignVCenter
            elif type_ in qtype.QT_BUTTON_TYPES:
                value = Qt.AlignRight | Qt.AlignVCenter
            elif type_ in ut.VALID_FLOAT_TYPES:
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
                # print(data)
                return data
            else:
                # Display data with default delegate by casting to a qvariant
                data = model._get_data(qtindex, **kwargs)
                if model.col_display_role_func_dict is not None:
                    col_name = model.col_name_list[col]
                    display_role_func = model.col_display_role_func_dict.get(
                        col_name, None
                    )
                    if display_role_func is not None:
                        value = display_role_func(data)
                        return value
                value = qtype.cast_into_qt(data)
                return value
        else:
            # import builtins
            # role_name = qtype.ItemDataRoles[role]
            # builtins.print('UNHANDLED ROLE=%r' % role_name)
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
            # row = qtindex.row()
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
                # This may not work with PyQt5
                # http://stackoverflow.com/questions/22560296/not-responding-datachanged
                # Emit that data was changed and return succcess
                model.dataChanged.emit(qtindex, qtindex)
            return True
        except Exception as ex:
            # value = str(value.toString())  # NOQA
            ut.printex(
                ex,
                'ignoring setData',
                '[model]',
                tb=True,
                key_list=['value'],
                iswarning=True,
            )
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
        # model.lazy_checks()
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            column = section
            if column >= len(model.col_nice_list):
                return []
            return model.col_nice_list[column]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            # row = section
            # rowid = model._get_row_id(row)
            # return rowid
            return section
        return QVariantHack()

    @updater
    def sort(model, column, order):
        """ Qt Override """
        # model.lazy_checks()
        reverse = order == QtCore.Qt.DescendingOrder
        model._set_sort(column, reverse)

    @default_method_decorator
    def flags(model, qtindex):
        """
        Qt Override

        Returns:
            Qt.ItemFlag:
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
        col = qtindex.column()
        type_ = model._get_type(col)
        editable = (
            model.col_edit_list[col]
            and model._get_level(qtindex) == model.col_level_list[col]
        )
        if type_ in qtype.QT_IMAGE_TYPES:
            # return Qt.NoItemFlags
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif not editable:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif type_ in ut.VALID_BOOL_TYPES:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable


def simple_thumbnail_widget():
    r"""
    Very simple example to test thumbnails

    CommandLine:
        python -m wbia.guitool.api_item_model --test-simple_thumbnail_widget  --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> import wbia.guitool as guitool
        >>> from wbia.guitool.api_item_model import *  # NOQA
        >>> guitool.ensure_qapp()  # must be ensured before any embeding
        >>> wgt = simple_thumbnail_widget()
        >>> ut.quit_if_noshow()
        >>> wgt.show()
        >>> guitool.qtapp_loop(wgt, frequency=100, init_signals=True)
    """
    import wbia.guitool as guitool

    guitool.ensure_qapp()
    col_name_list = ['rowid', 'image_name', 'thumb']
    col_types_dict = {
        'thumb': 'PIXMAP',
    }

    def thumb_getter(id_, thumbsize=128):
        """ Thumb getters must conform to thumbtup structure """
        # print(id_)
        return ut.grab_test_imgpath(id_)
        # return None

    col_getter_dict = {
        'rowid': [1, 2, 3],
        'image_name': ['lena.png', 'carl.jpg', 'patsy.jpg'],
        'thumb': thumb_getter,
    }
    col_ider_dict = {
        'thumb': 'image_name',
    }
    col_setter_dict = {}
    editable_colnames = []
    sortby = 'rowid'

    def get_thumb_size():
        return 128

    col_width_dict = {}
    col_bgrole_dict = {}

    api = guitool.CustomAPI(
        col_name_list,
        col_types_dict,
        col_getter_dict,
        col_bgrole_dict,
        col_ider_dict,
        col_setter_dict,
        editable_colnames,
        sortby,
        get_thumb_size,
        True,
        col_width_dict,
    )
    headers = api.make_headers(tblnice='Simple Example')

    wgt = guitool.APIItemWidget()
    wgt.change_headers(headers)
    # guitool.qtapp_loop(qwin=wgt, ipy=ipy, frequency=loop_freq)
    return wgt


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.guitool.api_item_model
        python -m wbia.guitool.api_item_model --allexamples
        python -m wbia.guitool.api_item_model --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
