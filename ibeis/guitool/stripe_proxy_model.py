from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from . import qtype
import math

BASE_CLASS = QtGui.QIdentityProxyModel

class StripeProxyModel(BASE_CLASS):
    def __init__(self, parent=None, numduplicates=1):
        BASE_CLASS.__init__(self, parent=parent)
        self._nd = numduplicates

    def rowCount(self, parent=QtCore.QModelIndex()):
        source_rows = self.sourceModel().rowCount(parent=parent)
        rows = math.ceil(source_rows / self._nd)
        #print('StripeProxyModel.rowCount(): %r %r' % (source_rows, rows))
        return int(rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        source_cols = self.sourceModel().columnCount(parent=parent)
        cols = self._nd * source_cols
        #print('StripeProxyModel.columnCount(): %r %r' % (source_cols, cols))
        return int(cols)

    def mapToSource(self, proxyIndex):
        """
        returns index into original model
        """
        if proxyIndex.isValid():
            parent = proxyIndex.parent()
            source_model = self.sourceModel()
            source_rows = source_model.rowCount(parent=parent)
            source_cols = source_model.columnCount(parent=parent)
            r, c, p = proxyIndex.row(), proxyIndex.column(), parent
            r2 = int(math.floor(c / source_cols)) + (r * self._nd)
            c2 = c % source_cols
            p2 = p
            #print('StripeProxyModel.mapToSource(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            return self.sourceModel().index(r2, c2, p2)
        else:
            return QtCore.QModelIndex()

    def mapFromSource(self, sourceIndex):
        """
        returns index into proxy model
        """
        if sourceIndex.isValid():
            parent = sourceIndex.parent()
            source_model = self.sourceModel()
            source_rows = source_model.rowCount(parent=parent)
            source_cols = source_model.columnCount(parent=parent)
            r, c, p = sourceIndex.row(), sourceIndex.column(), parent
            r2 = int(math.floor(r/self._nd))
            c2 = ((r%self._nd)*source_cols)+c
            p2 = p
            #print('StripeProxyModel.mapFromSource(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            return self.sourceModel().index(r2, c2, p2)
        else:
            return QtCore.QModelIndex()

    def index(self, row, col, parent=QtCore.QModelIndex()):
        if (row, col) != (-1, -1):
            source_rows = self.sourceModel().rowCount(parent=parent)
            source_cols = self.sourceModel().columnCount(parent=parent)
            r, c, p = row, col, parent
            r2 = int(math.floor(c / source_cols)) + (r * self._nd)
            c2 = c % source_cols
            p2 = p
            #print('StripeProxyModel.index(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            return self.sourceModel().index(r2, c2, parent=p2)
        else:
            return QtCore.QModelIndex()

    def data(self, proxyIndex, role=Qt.DisplayRole):
        return self.sourceModel().data(self.mapToSource(proxyIndex), role)

    def sort(self, column, order):
        source_model = self.sourceModel()
        source_cols = source_model.columnCount()
        source_model.sort(column % source_cols, order)
