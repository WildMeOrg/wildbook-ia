from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
#import guitool
from guitool import qtype
from guitool.api_thumb_delegate import APIThumbDelegate
from guitool.api_button_delegate import APIButtonDelegate
from guitool.guitool_decorators import signal_, slot_
from guitool.api_table_model import APITableModel
from guitool.guitool_main import get_qtapp
from guitool.guitool_misc import get_view_selection_as_str
import utool

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APITableView]', DEBUG=False)


# If you need to set the selected index try:
# AbstractItemView::setCurrentIndex
# AbstractItemView::scrollTo
# AbstractItemView::keyboardSearch

API_VIEW_BASE = QtGui.QTableView
#API_VIEW_BASE = QtGui.QAbstractItemView


class APITableView(API_VIEW_BASE):
    """
    Base class for all IBEIS Tables
    """
    rows_updated = signal_(str, int)
    contextMenuClicked = signal_(QtCore.QModelIndex, QtCore.QPoint)

    def __init__(view, parent=None):
        API_VIEW_BASE.__init__(view, parent)
        # Allow sorting by column
        view._init_table_behavior()
        view._init_header_behavior()
        # Context menu
        view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(view.on_customMenuRequested)

    #---------------
    # Initialization
    #---------------

    def _init_table_behavior(view):
        """ Table behavior """
        view.setCornerButtonEnabled(False)
        view.setWordWrap(True)
        view.setSortingEnabled(True)
        view.setShowGrid(True)

        # Selection behavior
        #view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        #view.setSelectionBehavior(QtGui.QAbstractItemView.SelectColumns)
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)

        # Selection behavior
        #view.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        view.setSelectionMode(QtGui.QAbstractItemView.ContiguousSelection)
        #view.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        #view.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        #view.setSelectionMode(QtGui.QAbstractItemView.NoSelection)

        # Edit Triggers
        #view.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)  # No Editing
        #view.setEditTriggers(QtGui.QAbstractItemView.SelectedClicked)
        #QtGui.QAbstractItemView.NoEditTriggers  |  # 0
        #QtGui.QAbstractItemView.CurrentChanged  |  # 1
        #QtGui.QAbstractItemView.DoubleClicked   |  # 2
        #QtGui.QtGui.QAbstractItemView.SelectedClicked |  # 4
        #QtGui.QAbstractItemView.EditKeyPressed  |  # 8
        #QtGui.QAbstractItemView.AnyKeyPressed      # 16
        view._defaultEditTriggers = QtGui.QAbstractItemView.AllEditTriggers
        view.setEditTriggers(view._defaultEditTriggers)
        # TODO: Figure out how to not edit when you are selecting
        #view.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)

        view.setIconSize(QtCore.QSize(64, 64))

    def _init_header_behavior(view):
        """ Header behavior """
        # Row Headers
        verticalHeader = view.verticalHeader()
        verticalHeader.setVisible(True)
        #verticalHeader.setSortIndicatorShown(True)
        verticalHeader.setHighlightSections(True)
        verticalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        verticalHeader.setMovable(True)

        # Column headers
        horizontalHeader = view.horizontalHeader()
        horizontalHeader.setVisible(True)
        horizontalHeader.setStretchLastSection(True)
        horizontalHeader.setSortIndicatorShown(True)
        horizontalHeader.setHighlightSections(True)
        # Column Sizes
        # DO NOT USE RESIZETOCONTENTS. IT MAKES THINGS VERY SLOW
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        #horizontalHeader.setCascadingSectionResizes(True)
        # Columns moveable
        horizontalHeader.setMovable(True)

    #---------------
    # Data Manipulation
    #---------------

    def infer_delegates(view, **headers):
        """ Infers which columns should be given item delegates """
        col_type_list = headers.get('col_type_list', [])
        for colx, coltype in enumerate(col_type_list):
            if coltype in  qtype.QT_PIXMAP_TYPES:
                print('[view] colx=%r is a PIXMAP' % colx)
                view.setItemDelegateForColumn(colx, APIThumbDelegate(view))
            elif coltype in qtype.QT_BUTTON_TYPES:
                print('[view] colx=%r is a BUTTON' % colx)
                view.setItemDelegateForColumn(colx, APIButtonDelegate(view))

    def set_column_persistant_editor(view, column):
        """ Set each row in a column as persistant """
        num_rows = view.model.rowCount()
        print('view.set_persistant: %r rows' % num_rows)
        for row in xrange(num_rows):
            index  = view.model.index(row, column)
            view.view.openPersistentEditor(index)

    def _update_headers(view, **headers):
        """ Mirrors _update_headers in api_table_model """
        # Use headers from model #model = view.model #headers = model.headers
        # Get header info
        col_sort_index = headers.get('col_sort_index', None)
        col_sort_reverse = headers.get('col_sort_reverse', False)
        # Call updates
        view._set_sort(col_sort_index, col_sort_reverse)
        view.infer_delegates(**headers)
        #view.infer_delegates_from_model(model=model) #view.resizeColumnsToContents()

    def _set_sort(view, col_sort_index, col_sort_reverse=False):
        if col_sort_index is not None:
            order = [Qt.AscendingOrder, Qt.DescendingOrder][col_sort_reverse]
            view.sortByColumn(col_sort_index, order)

    #---------------
    # Qt Overrides
    #---------------

    def setModel(view, model):
        """ QtOverride: Returns item delegate for this index """
        assert isinstance(model, APITableModel), 'apitblview only accepts apitblemodels'
        # Learn some things about the model before you fully connect it.
        print('[view] setting model')
        model._rows_updated.connect(view.on_rows_updated)
        #view.infer_delegates_from_model(model=model)
        # TODO: Update headers
        return API_VIEW_BASE.setModel(view, model)

    def itemDelegate(view, qindex):
        """ QtOverride: Returns item delegate for this index """
        return API_VIEW_BASE.itemDelegate(view, qindex)

    def keyPressEvent(view, event):
        assert isinstance(event, QtGui.QKeyEvent)
        API_VIEW_BASE.keyPressEvent(view, event)
        if event.matches(QtGui.QKeySequence.Copy):
            #print('Received Ctrl+C in View')
            view.copy_selection_to_clipboard()
        #print ('[view] keyPressEvent: %s' % event.key())

    def mouseMoveEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        API_VIEW_BASE.mouseMoveEvent(view, event)

    def mousePressEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        API_VIEW_BASE.mousePressEvent(view, event)
        #print('no editing')
        view.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

    def mouseReleaseEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        #print('editing ok')
        view.setEditTriggers(view._defaultEditTriggers)
        API_VIEW_BASE.mouseReleaseEvent(view, event)

    #---------------
    # Slots
    #---------------

    @slot_(str, int)
    def on_rows_updated(view, tblname, num):
        # re-emit the model signal
        view.rows_updated.emit(tblname, num)

    @slot_(QtCore.QPoint)
    def on_customMenuRequested(view, pos):
        index = view.indexAt(pos)
        view.contextMenuClicked.emit(index, pos)

    def copy_selection_to_clipboard(view):
        """ Copys selected grid to clipboard """
        print('[guitool] Copying selection to clipboard')
        copy_str = get_view_selection_as_str(view)
        copy_qstr = QtCore.QString(copy_str)
        clipboard = get_qtapp().clipboard()
        print(copy_str)
        clipboard.setText(copy_qstr)
        print('[guitool] finished copy')
