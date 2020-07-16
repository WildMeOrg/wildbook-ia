# -*- coding: utf-8 -*-
"""
provides common methods for api_tree_view and api_table_view
"""
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtGui  # NOQA
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import QtWidgets
from wbia.guitool.__PYQT__.QtCore import Qt
import functools
from wbia.guitool import qtype
from wbia.guitool import api_button_delegate
from wbia.guitool import api_thumb_delegate
from wbia.guitool import guitool_main
from wbia.guitool import guitool_misc
from six.moves import range, reduce  # NOQA
import utool
import utool as ut
import operator

# Valid API Models
from wbia.guitool.stripe_proxy_model import StripeProxyModel
from wbia.guitool.filter_proxy_model import FilterProxyModel
from wbia.guitool.api_item_model import APIItemModel

(print, rrr, profile) = utool.inject2(__name__, '[APIItemView]')

VERBOSE_QT = ut.get_argflag(('--verbose-qt', '--verbqt'))
VERBOSE_ITEM_VIEW = ut.get_argflag(('--verbose-item-view'))
VERBOSE = utool.VERBOSE or VERBOSE_QT or VERBOSE_ITEM_VIEW

API_VIEW_BASE = QtWidgets.QAbstractItemView
ABSTRACT_VIEW_INJECT_KEY = ('QtWidgets.QAbstractItemView', 'guitool')
register_view_method = utool.make_class_method_decorator(
    ABSTRACT_VIEW_INJECT_KEY, __name__
)

injectviewinstance = functools.partial(
    utool.inject_instance, classkey=ABSTRACT_VIEW_INJECT_KEY
)


VALID_API_MODELS = (FilterProxyModel, StripeProxyModel, APIItemModel)


class APIItemView(API_VIEW_BASE):
    """
    Trees and Tables implicitly inherit from this class.
    Abstractish class.

    other function in this file will be injected into the concrete
    implementations of either a table or tree view. The code is only written
    once but duplicated in each of the psuedo-children. It is done this way to
    avoid explicit multiple inheritance.
    """

    def __init__(view, parent=None):
        API_VIEW_BASE.__init__(view, parent)


@register_view_method
def _init_api_item_view(view):
    view.registered_single_keys = []
    view.registered_keypress_funcs = []


# ---------------
# Data Manipulation
# ---------------


@register_view_method
def _init_itemview_behavior(view):
    """

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> # TODO figure out how to test these
        >>> from wbia.guitool.api_item_view import *  # NOQA
        >>> from wbia.guitool import api_table_view
        >>> from wbia.guitool import api_tree_view
        >>> view = api_table_view.APITableView()
        >>> view = api_tree_view.APITreeView()

    References:
        http://qt-project.org/doc/qt-4.8/qabstractitemview.html
    """
    # http://stackoverflow.com/questions/28680150/qtgui-qtableview-shows-data-in-background-while-a-cell-being-edited-pyqt4
    view.setAutoFillBackground(True)

    view.setWordWrap(True)

    # Selection behavior
    # view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    # view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
    view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)

    # Selection behavior
    # view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    # view.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
    # view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
    # view.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    # uniformRowHeights
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerItem)
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

    # Allow sorting by column
    view.setSortingEnabled(True)

    # Edit Triggers
    # view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)  # No Editing
    # view.setEditTriggers(QtWidgets.QAbstractItemView.SelectedClicked)
    # QtWidgets.QAbstractItemView.NoEditTriggers  |  # 0
    # QtWidgets.QAbstractItemView.CurrentChanged  |  # 1
    # QtWidgets.QAbstractItemView.DoubleClicked   |  # 2
    # QtWidgets.QAbstractItemView.SelectedClicked |  # 4
    # QtWidgets.QAbstractItemView.EditKeyPressed  |  # 8
    # QtWidgets.QAbstractItemView.AnyKeyPressed      # 16
    # view._defaultEditTriggers = QtWidgets.QAbstractItemView.AllEditTriggers

    bitwise_or = operator.__or__
    chosen_triggers = [
        # QtWidgets.QAbstractItemView.NoEditTriggers,
        QtWidgets.QAbstractItemView.CurrentChanged,
        QtWidgets.QAbstractItemView.DoubleClicked,
        QtWidgets.QAbstractItemView.SelectedClicked,
        QtWidgets.QAbstractItemView.EditKeyPressed,
        QtWidgets.QAbstractItemView.AnyKeyPressed,
    ]
    view._defaultEditTriggers = reduce(bitwise_or, chosen_triggers)
    # view._defaultEditTriggers = QtWidgets.QAbstractItemView.NoEditTriggers
    view.setEditTriggers(view._defaultEditTriggers)
    # TODO: Figure out how to not edit when you are selecting
    # view.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)


@register_view_method
def infer_delegates(view, **headers):
    """ Infers which columns should be given item delegates """
    get_thumb_size = headers.get('get_thumb_size', None)
    col_type_list = headers.get('col_type_list', [])
    num_cols = view.model().columnCount()
    num_duplicates = int(num_cols / len(col_type_list))
    col_type_list = col_type_list * num_duplicates
    view.has_thumbs = False
    for colx, coltype in enumerate(col_type_list):
        if coltype in qtype.QT_PIXMAP_TYPES:
            if VERBOSE:
                print('[view] colx=%r is a PIXMAP' % colx)
            thumb_delegate = api_thumb_delegate.APIThumbDelegate(view, get_thumb_size)
            view.setItemDelegateForColumn(colx, thumb_delegate)
            view.has_thumbs = True
            # HACK
            # verticalHeader = view.verticalHeader()
            # verticalHeader.setDefaultSectionSize(256)
        elif coltype in qtype.QT_BUTTON_TYPES:
            if VERBOSE:
                print('[view] colx=%r is a BUTTON' % colx)
            button_delegate = api_button_delegate.APIButtonDelegate(view)
            view.setItemDelegateForColumn(colx, button_delegate)
        elif isinstance(coltype, QtWidgets.QAbstractItemDelegate):
            if VERBOSE:
                print('[view] colx=%r is a CUSTOM DELEGATE' % colx)
            view.setItemDelegateForColumn(colx, coltype)
        else:
            if VERBOSE:
                print('[view] colx=%r does not have a delgate' % colx)
            # Effectively unsets any existing delegates
            default_delegate = QtWidgets.QStyledItemDelegate(view)
            view.setItemDelegateForColumn(colx, default_delegate)


@register_view_method
def set_column_persistant_editor(view, column):
    """ Set each row in a column as persistant """
    num_rows = view.model.rowCount()
    print('view.set_persistant: %r rows' % num_rows)
    for row in range(num_rows):
        index = view.model.index(row, column)
        view.view.openPersistentEditor(index)


@register_view_method
def _update_headers(view, **headers):
    """ Mirrors _update_headers in api_item_model """
    # Use headers from model #model = view.model #headers = model.headers
    # Get header info
    col_sort_index = headers.get('col_sort_index', None)
    col_sort_reverse = headers.get('col_sort_reverse', False)
    view.col_hidden_list = headers.get('col_hidden_list', [])
    view.col_name_list = headers.get('col_name_list', [])
    # Call updates
    # FIXME: is this the right thing to do here?
    view._set_sort(col_sort_index, col_sort_reverse)
    view.infer_delegates(**headers)
    if ut.VERBOSE:
        print('[view] updating headers')
    col_width_list = headers.get('col_width_list', None)
    if col_width_list is not None:
        if isinstance(view, QtWidgets.QTreeView):
            horizontal_header = view.header()
        else:
            horizontal_header = view.horizontalHeader()
        for index, width in enumerate(col_width_list):
            # Check for functionally sepcified widths
            if hasattr(width, '__call__'):
                width = width()
            horizontal_header.resizeSection(index, width)
        # except AttributeError as ex:
        #    ut.embed()
        #    ut.printex(ex, 'tree view?')
        # view.infer_delegates_from_model(model=model) #view.resizeColumnsToContents()


@register_view_method
def _set_sort(view, col_sort_index, col_sort_reverse=True):
    if col_sort_index is not None:
        order = [Qt.AscendingOrder, Qt.DescendingOrder][col_sort_reverse]
        view.sortByColumn(col_sort_index, order)


@register_view_method
def hide_cols(view):
    total_num_cols = view.model().columnCount()
    num_cols = len(view.col_hidden_list)
    num_duplicates = int(total_num_cols / num_cols)
    duplicated_hidden_list = view.col_hidden_list * num_duplicates
    for col, hidden in enumerate(duplicated_hidden_list):
        view.setColumnHidden(col, hidden)


# @register_view_method
# def clear_selection(view):
#    #print('[api_item_view] clear_selection()')
#    selection_model = view.selectionModel()
#    selection_model.clearSelection()


@register_view_method
def get_row_and_qtindex_from_id(view, _id):
    """ uses an sqlrowid (from iders) to get a qtindex """
    model = view.model()
    qtindex, row = model.get_row_and_qtindex_from_id(_id)
    return qtindex, row


@register_view_method
def select_row_from_id(view, _id, scroll=False, collapse=True):
    """
        _id is from the iders function (i.e. an wbia rowid)
        selects the row in that view if it exists
    """
    with ut.Timer(
        '[api_item_view] select_row_from_id(id=%r, scroll=%r, collapse=%r)'
        % (_id, scroll, collapse)
    ):
        qtindex, row = view.get_row_and_qtindex_from_id(_id)
        if row is not None:
            if isinstance(view, QtWidgets.QTreeView):
                if collapse:
                    view.collapseAll()
                select_model = view.selectionModel()
                select_flag = QtCore.QItemSelectionModel.ClearAndSelect
                # select_flag = QtCore.QItemSelectionModel.Select
                # select_flag = QtCore.QItemSelectionModel.NoUpdate
                with ut.Timer('[api_item_view] selecting name. qtindex=%r' % (qtindex,)):
                    select_model.select(qtindex, select_flag)
                with ut.Timer('[api_item_view] expanding'):
                    view.setExpanded(qtindex, True)
            else:
                # For Table Views
                view.selectRow(row)
            # Scroll to selection
            if scroll:
                with ut.Timer('scrolling'):
                    view.scrollTo(qtindex)
            return row
    return None


@register_view_method
def connect_single_key_to_slot(view, key, func):
    """
    hacky way to simulate slots for generic key press events
    """
    view.registered_single_keys.append((key, func))


@register_view_method
def connect_keypress_to_slot(view, func):
    """
    hacky way to simulate slots for single key press events
    """
    view.registered_keypress_funcs.append(func)


@register_view_method
def selectedRows(view):
    selected_qtindex_list = view.selectedIndexes()
    selected_qtindex_list2 = []
    seen_ = set([])
    for qindex in selected_qtindex_list:
        row = qindex.row()
        if row not in seen_:
            selected_qtindex_list2.append(qindex)
            seen_.add(row)
    return selected_qtindex_list2


# ---------------
# Qt Overrides
# ---------------


def keyPressEvent(view, event):
    """
    Handles simple key press events. There is probably a better way to do this
    using real signals / slots, but maybe you need to always overwrite to set the
    handled flag correctly.

    CommandLine:
        xdoctest -m ~/code/guitool/guitool/api_item_view.py keyPressEvent
        --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.api_item_view import *  # NOQA
        >>> import wbia.guitool as gt
        >>> app = gt.ensure_qapp()[0]
        >>> wgt = gt.simple_api_item_widget()
        >>> view = wgt.view
        >>> c_pressed = [0]
        >>> def foo(view, event):
        >>>     key = event.key()
        >>>     print('[foo] Pressed key = %r' % (key,))
        >>>     if event.key() == Qt.Key_C:
        >>>         print('Pressed C')
        >>>         c_pressed[0] = 1
        >>>         return True
        >>> view.connect_keypress_to_slot(foo)
        >>> view._init_header_behavior()
        >>> # Try to simulate an event for testing
        >>> wgt.show()
        >>> from wbia.guitool.__PYQT__ import QtTest, GUITOOL_PYQT_VERSION
        >>> QTest = QtTest.QTest
        >>> if GUITOOL_PYQT_VERSION == 4:
        >>>     QTest.qWaitForWindowShown(wgt)
        >>> else:
        >>>     QTest.qWaitForWindowActive(wgt)
        >>> qtindex = view.model().index(1, 2)
        >>> point = view.visualRect(qtindex).center()
        >>> #point = wgt.visibleRegion().boundingRect().center()
        >>> #QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        >>> QTest.mouseClick(wgt, Qt.LeftButton, Qt.NoModifier, point)
        >>> selected_indices = view.selectedIndexes()
        >>> print('selected_indices = %r' % (selected_indices,))
        >>> # Why does this not work?
        >>> def check_selection():
        >>>     selected_indices = view.selectedIndexes()
        >>>     if len(selected_indices) > 0:
        >>>         return selected_indices[0].data()
        >>> # Hack because I cant figure out how to get a click to simulate
        >>> # a selection
        >>> QTest.keyPress(view, Qt.Key_Right)
        >>> QTest.keyPress(view, Qt.Key_B)
        >>> ut.assert_eq(check_selection(), 'b')
        >>> QTest.keyPress(view, Qt.Key_C)
        >>> ut.assert_eq(check_selection(), 'b')
        >>> assert c_pressed[0] == 1
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(wgt, frequency=100)
    """
    # TODO: can this be in api_item_view?
    assert isinstance(event, QtGui.QKeyEvent)
    if event.matches(QtGui.QKeySequence.Copy):
        # print('Received Ctrl+C in View')
        view.copy_selection_to_clipboard()
    # print ('[view] keyPressEvent: %s' % event.key())
    flag = False
    for func in view.registered_keypress_funcs:
        flag |= bool(func(view, event))
    for key, func in view.registered_single_keys:
        # print(key)
        if event.key() == key:
            flag = True
            func(view, event)
    if not flag:
        view.API_VIEW_BASE.keyPressEvent(view, event)


# @register_view_method
def itemDelegate(view, qindex):
    """ QtOverride: Returns item delegate for this index """
    # Does this even work? TODO: testme
    return API_VIEW_BASE.itemDelegate(view, qindex)


def setModel(view, model):
    """ QtOverride: Returns item delegate for this index """
    assert isinstance(model, VALID_API_MODELS), (
        'APIItemViews only accepts APIItemModels (or one of its proxys),'
        'received a %r' % type(model)
    )
    # Learn some things about the model before you fully connect it.
    if VERBOSE:
        print('[view] setting model')
    model._rows_updated.connect(view.on_rows_updated)
    # view.infer_delegates_from_model(model=model)
    # TODO: Update headers
    return view.API_VIEW_BASE.setModel(view, model)


# ---------------
# Slots
# ---------------


@register_view_method
def copy_selection_to_clipboard(view):
    """ Copys selected grid to clipboard """
    if VERBOSE:
        print('[guitool] Copying selection to clipboard')
    copy_str = guitool_misc.get_view_selection_as_str(view)
    # copy_qstr = QtCore.Q__String(copy_str)
    copy_qstr = str(copy_str)
    clipboard = guitool_main.get_qtapp().clipboard()
    if VERBOSE:
        print(copy_str)
    clipboard.setText(copy_qstr)
    if VERBOSE:
        print('[guitool] finished copy')


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.guitool.api_item_view
        python -m wbia.guitool.api_item_view --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
