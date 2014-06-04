from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
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

    def __init__(tblview, parent=None):
        QtGui.QTableView.__init__(tblview, parent)
        # Allow sorting by column
        tblview.setSortingEnabled(True)
        # No vertical header
        verticalHeader = tblview.verticalHeader()
        verticalHeader.setVisible(False)
        # Stretchy column widths
        horizontalHeader = tblview.horizontalHeader()
        horizontalHeader.setStretchLastSection(True)
        horizontalHeader.setSortIndicatorShown(True)
        horizontalHeader.setHighlightSections(True)
        horizontalHeader.setCascadingSectionResizes(True)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Stretch)
        # DO NOT USE RESIZETOCONTENTS. IT MAKES THINGS VERY SLOW FOR SOME REASON.
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        #horizontalHeader.setResizeMode(QtGui.QHeaderView.Interactive)
        #horizontalHeader.setMovable(True)

        # Selection behavior
        tblview.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        # Edit Triggers
        #tblview.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)  # No Editing
        #tblview.setEditTriggers(QtGui.QAbstractItemView.SelectedClicked)
        #tblview.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked)
        tblview.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)

        #tblview.resizeColumnsToContents()
        # Context menu
        tblview.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        tblview.customContextMenuRequested.connect(tblview.on_customMenuRequested)

    @slot_(QtCore.QPoint)
    def on_customMenuRequested(tblview, pos):
        index = tblview.indexAt(pos)
        tblview.contextMenuClicked.emit(index, pos)
