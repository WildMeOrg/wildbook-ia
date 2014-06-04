from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
import guitool
from .guitool_decorators import signal_, slot_


# If you need to set the selected index try:
# AbstractItemView::setCurrentIndex
# AbstractItemView::scrollTo
# AbstractItemView::keyboardSearch
class APITableView(QtGui.QTableView):
    """
    Base class for all IBEIS Tables
    """
    contextMenuClicked = signal_(QtCore.QModelIndex, QtCore.QPoint)

    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        # Allow sorting by column
        view.setSortingEnabled(True)

        view._init_table_behavior()
        view._init_header_behavior()

        #view.resizeColumnsToContents()
        # Context menu
        view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(view.on_customMenuRequested)

    def _init_table_behavior(view):
        """ Table behavior """
        # Selection behavior
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        # Edit Triggers
        #view.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)  # No Editing
        #view.setEditTriggers(QtGui.QAbstractItemView.SelectedClicked)
        #view.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked)
        view.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)

    def _init_header_behavior(view):
        """ Header behavior """
        # No vertical header
        verticalHeader = view.verticalHeader()
        verticalHeader.setVisible(True)
        #verticalHeader.setSortIndicatorShown(True)
        #verticalHeader.setHighlightSections(True)
        verticalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        #verticalHeader.setMovable(True)
        # Stretchy column widths
        horizontalHeader = view.horizontalHeader()
        horizontalHeader.setStretchLastSection(True)
        horizontalHeader.setSortIndicatorShown(True)
        horizontalHeader.setHighlightSections(True)
        # Column Sizes
        # DO NOT USE RESIZETOCONTENTS. IT MAKES THINGS VERY SLOW FOR SOME REASON.
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        #horizontalHeader.setCascadingSectionResizes(True)
        # Columns moveable
        horizontalHeader.setMovable(True)

    def _update_headers(view, **headers):
        """ Mirrors _update_headers in api_table_model """
        # Get header info
        col_sort_index = headers.get('col_sort_index', None)
        col_sort_reverse = headers.get('col_sort_reverse', False)
        # Call updates
        view._set_sort(col_sort_index, col_sort_reverse)

    def _set_sort(view, col_sort_index, col_sort_reverse=False):
        if col_sort_index is not None:
            order = [Qt.AscendingOrder, Qt.DescendingOrder][col_sort_reverse]
            view.sortByColumn(col_sort_index, order)

    @slot_(QtCore.QPoint)
    def on_customMenuRequested(view, pos):
        index = view.indexAt(pos)
        view.contextMenuClicked.emit(index, pos)

    def set_column_as_delegate(view, column, delegate_type):
        """ Checks delegate type from tuple"""
        DelegateClass = guitool.DELEGATE_MAP[delegate_type]
        print('view.setItemDelegateForColumn(%r, %r)' % (column, delegate_type))
        view.setItemDelegateForColumn(column, DelegateClass(view))
        return DelegateClass.is_persistant_editable

    def set_column_persistant_editor(cltw, column):
        """
        Set each row in a column as persistant
        """
        num_rows = cltw.model.rowCount()
        print('cltw.set_persistant: %r rows' % num_rows)
        for row in xrange(num_rows):
            index  = cltw.model.index(row, column)
            cltw.view.openPersistentEditor(index)
