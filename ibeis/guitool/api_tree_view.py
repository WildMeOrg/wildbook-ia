from __future__ import absolute_import, division, print_function
from guitool import qtype
from guitool.api_thumb_delegate import APIThumbDelegate
from guitool.api_button_delegate import APIButtonDelegate
from guitool.api_item_view import injectviewinstance
from guitool.stripe_proxy_model import StripeProxyModel
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
import guitool
from guitool.guitool_decorators import signal_, slot_
from guitool.api_item_model import APIItemModel
from guitool.guitool_main import get_qtapp
from guitool.guitool_misc import get_view_selection_as_str
import utool

(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[APITreeView]', DEBUG=False)


# If you need to set the selected index try:
# AbstractItemView::setCurrentIndex
# AbstractItemView::scrollTo
# AbstractItemView::keyboardSearch

API_VIEW_BASE = QtGui.QTreeView
#API_VIEW_BASE = QtGui.QAbstractItemView


class APITreeView(API_VIEW_BASE):
    """
    Base class for all IBEIS Tables
    """
    rows_updated = signal_(str, int)
    contextMenuClicked = signal_(QtCore.QModelIndex, QtCore.QPoint)

    def __init__(view, parent=None):
        API_VIEW_BASE.__init__(view, parent)
        injectviewinstance(view)
        # Allow sorting by column
        view._init_tree_behavior()
        view.col_hidden_list = []
        ##view._init_header_behavior()
        # Context menu
        view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(view.on_customMenuRequested)

    #---------------
    # Initialization
    #---------------

    def _init_tree_behavior(view):
        """ Tree behavior """
        view.setWordWrap(True)
        view.setSortingEnabled(True)

    def _init_header_behavior(view):
        """ Header behavior """
        # Row Headers
        # Column headers
        header = view.header()
        header.setVisible(True)
        header.setStretchLastSection(True)
        header.setSortIndicatorShown(True)
        header.setHighlightSections(True)
        # Column Sizes
        # DO NOT USE RESIZETOCONTENTS. IT MAKES THINGS VERY SLOW
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
        header.setResizeMode(QtGui.QHeaderView.Interactive)
        #horizontalHeader.setCascadingSectionResizes(True)
        # Columns moveable
        header.setMovable(True)

    #---------------
    # Qt Overrides
    #---------------

    def setModel(view, model):
        """ QtOverride: Returns item delegate for this index """
        assert isinstance(model, (StripeProxyModel, APIItemModel)), 'apitblview only accepts apitblemodels, received a %r' % type(model)
        # Learn some things about the model before you fully connect it.
        print('[view] setting model')
        model._rows_updated.connect(view.on_rows_updated)
        #view.infer_delegates_from_model(model=model)
        # TODO: Update headers
        return API_VIEW_BASE.setModel(view, model)

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
