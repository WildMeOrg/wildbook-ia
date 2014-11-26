from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtCore, QtGui
from guitool import api_item_view
from guitool.guitool_decorators import signal_, slot_
import utool

(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableView]', DEBUG=False)


# If you need to set the selected index try:
# AbstractItemView::setCurrentIndex
# AbstractItemView::scrollTo
# AbstractItemView::keyboardSearch

API_VIEW_BASE = QtGui.QTableView
#API_VIEW_BASE = QtGui.QAbstractItemView


class APITableView(API_VIEW_BASE):
    """
    Table view of API data.
    Implicitly inherits from APIItemView
    """
    rows_updated = signal_(str, int)
    contextMenuClicked = signal_(QtCore.QModelIndex, QtCore.QPoint)
    API_VIEW_BASE = API_VIEW_BASE

    def __init__(view, parent=None):
        # Qt Inheritance
        API_VIEW_BASE.__init__(view, parent)
        # Implicitly inject common APIItemView functions
        api_item_view.injectviewinstance(view)
        #utool.inject_instance(view, API_VIEW_BASE)
        # Allow sorting by column
        view._init_table_behavior()
        view._init_header_behavior()
        view.col_hidden_list = []
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
        # view.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        # view.setSelectionMode(QtGui.QAbstractItemView.ContiguousSelection)
        view.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        # view.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
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

    def setModel(view, model):
        """ QtOverride: Returns item delegate for this index """
        api_item_view.setModel(view, model)

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

    def clearSelection(view, *args, **kwargs):
        print('[table_view] clear selection')
        API_VIEW_BASE.clearSelection(view, *args, **kwargs)

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
