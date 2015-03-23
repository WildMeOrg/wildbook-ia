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
        view._init_itemview_behavior()
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
        """ Table behavior

        SeeAlso:
            api_item_view._init_itemview_behavior
        """
        # Allow sorting by column
        view.setCornerButtonEnabled(False)
        view.setShowGrid(True)

        view.setIconSize(QtCore.QSize(64, 64))

    def _init_header_behavior(view):
        """ Header behavior

        Example:
            >>> # ENABLE_DOCTEST
            >>> from guitool.api_table_view import *  # NOQA
            >>> view = APITableView()

        """
        # Row Headers
        verticalHeader = view.verticalHeader()
        verticalHeader.setVisible(True)
        #verticalHeader.setSortIndicatorShown(True)
        verticalHeader.setHighlightSections(True)
        verticalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        verticalHeader.setMovable(True)
        # TODO: get good estimate if there are thumbnails
        #verticalHeader.setDefaultSectionSize(256)

        # Column headers
        horizontalHeader = view.horizontalHeader()
        horizontalHeader.setVisible(True)
        horizontalHeader.setStretchLastSection(True)
        horizontalHeader.setSortIndicatorShown(True)
        horizontalHeader.setHighlightSections(True)
        # Column Sizes
        # DO NOT USE ResizeToContents. IT MAKES THINGS VERY SLOW
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        #horizontalHeader.setCascadingSectionResizes(True)
        # Columns moveable
        horizontalHeader.setMovable(True)
        view.registered_single_keys = []
        view.registered_keypress_funcs = []

    def connect_single_key_to_slot(view, key, func):
        # TODO: move to api_item_view
        view.registered_single_keys.append((key, func))

    def connect_keypress_to_slot(view, func):
        # TODO: move to api_item_view
        view.registered_keypress_funcs.append(func)

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
        for func in view.registered_keypress_funcs:
            func(view, event)
        for key, func in view.registered_single_keys:
            #print(key)
            if event.key() == key:
                func(view, event)

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
