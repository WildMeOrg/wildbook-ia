# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtCore, QtGui
from wbia.guitool.__PYQT__ import QtWidgets
from wbia.guitool import api_item_view
from wbia.guitool.guitool_decorators import signal_, slot_
import utool

(print, rrr, profile) = utool.inject2(__name__, '[APITableView]', DEBUG=False)


# If you need to set the selected index try:
# AbstractItemView::setCurrentIndex
# AbstractItemView::scrollTo
# AbstractItemView::keyboardSearch

API_VIEW_BASE = QtWidgets.QTableView
# API_VIEW_BASE = QtWidgets.QAbstractItemView


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
        view._init_api_item_view()

    # ---------------
    # Initialization
    # ---------------

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

        CommandLine:
            python -m wbia.guitool.api_item_widget --test-simple_api_item_widget --show
            python -m wbia.guitool.api_table_view --test-_init_header_behavior --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> # GUI_DOCTEST
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.guitool.api_table_view import *  # NOQA
            >>> from wbia import guitool
            >>> guitool.ensure_qapp()
            >>> view = APITableView()
            >>> view._init_header_behavior()
        """
        # Row Headers
        verticalHeader = view.verticalHeader()
        verticalHeader.setVisible(False)
        # verticalHeader.setSortIndicatorShown(True)
        verticalHeader.setHighlightSections(True)
        try:
            verticalHeader.setResizeMode(QtWidgets.QHeaderView.Interactive)
            verticalHeader.setMovable(False)
        except AttributeError:
            verticalHeader.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
            verticalHeader.setSectionsMovable(False)
        # TODO: get good estimate if there are thumbnails
        # verticalHeader.setDefaultSectionSize(256)

        # Column headers
        horizontalHeader = view.horizontalHeader()
        horizontalHeader.setVisible(True)
        horizontalHeader.setStretchLastSection(True)
        horizontalHeader.setSortIndicatorShown(True)
        horizontalHeader.setHighlightSections(True)
        # Column Sizes
        # DO NOT USE ResizeToContents. IT MAKES THINGS VERY SLOW
        # horizontalHeader.setResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        # horizontalHeader.setResizeMode(QtWidgets.QHeaderView.Stretch)
        try:
            horizontalHeader.setResizeMode(QtWidgets.QHeaderView.Interactive)
            horizontalHeader.setMovable(True)
        except AttributeError:
            horizontalHeader.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
            horizontalHeader.setSectionsMovable(True)
        # horizontalHeader.setCascadingSectionResizes(True)
        # Columns moveable

    # ---------------
    # Qt Overrides
    # ---------------

    def setModel(view, model):
        """ QtOverride: Returns item delegate for this index """
        api_item_view.setModel(view, model)

    def keyPressEvent(view, event):
        """
        CommandLine:
            python -m wbia.guitool.api_item_widget --test-simple_api_item_widget --show
            python -m wbia.guitool.api_table_view --test-keyPressEvent --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> # GUI_DOCTEST
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.guitool.api_table_view import *  # NOQA
            >>> from wbia import guitool
            >>> guitool.ensure_qapp()
            >>> view = APITableView()
            >>> view._init_header_behavior()
        """
        return api_item_view.keyPressEvent(view, event)

        # # TODO: can this be in api_item_view?
        # assert isinstance(event, QtGui.QKeyEvent)
        # view.API_VIEW_BASE.keyPressEvent(view, event)
        # if event.matches(QtGui.QKeySequence.Copy):
        #     #print('Received Ctrl+C in View')
        #     view.copy_selection_to_clipboard()
        # #print ('[view] keyPressEvent: %s' % event.key())
        # for func in view.registered_keypress_funcs:
        #     func(view, event)
        # for key, func in view.registered_single_keys:
        #     #print(key)
        #     if event.key() == key:
        #         func(view, event)

    def mouseMoveEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        API_VIEW_BASE.mouseMoveEvent(view, event)

    def mousePressEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        API_VIEW_BASE.mousePressEvent(view, event)
        # print('no editing')
        view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def mouseReleaseEvent(view, event):
        assert isinstance(event, QtGui.QMouseEvent)
        # print('editing ok')
        view.setEditTriggers(view._defaultEditTriggers)
        API_VIEW_BASE.mouseReleaseEvent(view, event)

    # ---------------
    # Slots
    # ---------------

    @slot_(str, int)
    def on_rows_updated(view, tblname, num):
        # re-emit the model signal
        view.rows_updated.emit(tblname, num)

    @slot_(QtCore.QPoint)
    def on_customMenuRequested(view, pos):
        index = view.indexAt(pos)
        view.contextMenuClicked.emit(index, pos)
