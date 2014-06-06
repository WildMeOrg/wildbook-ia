from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
import guitool
from guitool.guitool_decorators import signal_, slot_
from guitool.guitool_main import get_qtapp
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
    # Qt Overrides
    #---------------

    def itemDelegate(qindex):
        """ QtOverride: Returns item delegate for this index """
        return API_VIEW_BASE.itemDelegate(qindex)

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
        print('no editing')
        view.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

    def mouseReleaseEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        print('editing ok')
        view.setEditTriggers(view._defaultEditTriggers)
        API_VIEW_BASE.mouseReleaseEvent(view, event)

    #---------------
    # Data Manipulation
    #---------------

    def _update_headers(view, **headers):
        """ Mirrors _update_headers in api_table_model """
        # Get header info
        col_sort_index = headers.get('col_sort_index', None)
        col_sort_reverse = headers.get('col_sort_reverse', False)
        # Call updates
        view._set_sort(col_sort_index, col_sort_reverse)
        #view.resizeColumnsToContents()

    def _set_sort(view, col_sort_index, col_sort_reverse=False):
        if col_sort_index is not None:
            order = [Qt.AscendingOrder, Qt.DescendingOrder][col_sort_reverse]
            view.sortByColumn(col_sort_index, order)

    #---------------
    # Slots
    #---------------

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

    def copy_selection_to_clipboard(view):
        """
        Taken from here http://stackoverflow.com/questions/3135737/
            copying-part-of-qtableview
        TODO: Make this pythonic
        """
        print('[guitool] Copying selection to clipboard')
        model = view.model()
        selection_model = view.selectionModel()
        qindex_list = selection_model.selectedIndexes()
        qindex_list = sorted(qindex_list)
        print('[guitool] %d cells selected' % len(qindex_list))
        if len(qindex_list) == 0:
            return
        copy_table = []
        previous = qindex_list[0]

        def astext(data):
            """ Helper which casts model data to a string """
            try:
                if isinstance(data, QtCore.QVariant):
                    text = str(data.toString())
                elif isinstance(data, QtCore.QString):
                    text = str(data)
                else:
                    text = str(data)
            except Exception as ex:
                text = repr(ex)
            return text.replace('\n', '<NEWLINE>').replace(',', '<COMMA>')

        #
        for ix in xrange(1, len(qindex_list)):
            text = astext(model.data(previous))
            copy_table.append(text)
            qindex = qindex_list[ix]

            if qindex.row() != previous.row():
                copy_table.append('\n')
            else:
                copy_table.append(', ')
            previous = qindex

        # Do last element in list
        text = astext(model.data(qindex_list[-1]))
        copy_table.append(text)
        #copy_table.append('\n')
        copy_str = str(''.join(copy_table))
        copy_qstr = QtCore.QString(copy_str)
        QAPP = get_qtapp()
        clipboard = QAPP.clipboard()
        print(copy_str)
        clipboard.setText(copy_qstr)
        print('[guitool] finished copy')
