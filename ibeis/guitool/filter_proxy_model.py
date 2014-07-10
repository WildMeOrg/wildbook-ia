from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
import math
import utool

#BASE_CLASS = QtGui.QAbstractProxyModel
BASE_CLASS = QtGui.QSortFilterProxyModel
# BASE_CLASS = QtGui.QIdentityProxyModel


class FilterProxyModel(BASE_CLASS):
    __metaclass__ = utool.makeForwardingMetaclass(lambda self: self.sourceModel(),
                                            ['_set_context_id',
                                             '_get_context_id',
                                             '_set_changeblocked',
                                            '_get_changeblocked',
                                             '_about_to_change',
                                             '_change',
                                             '_update',
                                            '_rows_updated',
                                             'name',],
                                            base_class=BASE_CLASS)

    def __init__(self, parent=None, numduplicates=1):
        BASE_CLASS.__init__(self, parent=parent)
        self._nd = numduplicates

    def rowCount(self, parent=QtCore.QModelIndex()):
        idx = self.mapToSource(parent)
        source_rows = self.sourceModel().rowCount(parent=idx)
        rows = math.ceil(source_rows / self._nd)
        #print('StripeProxyModel.rowCount(): %r %r' % (source_rows, rows))
        return int(rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        source_cols = self.sourceModel().columnCount(parent=parent)
        cols = self._nd * source_cols
        #print('StripeProxyModel.columnCount(): %r %r' % (source_cols, cols))
        return int(cols)

    def proxy_to_source(self, row, col, parent=QtCore.QModelIndex()):
        source_model = self.sourceModel()
        source_cols = source_model.columnCount(parent=parent)
        r, c, p = row, col, parent
        r2 = int(math.floor(c / source_cols)) + (r * self._nd)
        c2 = c % source_cols
        p2 = p
        return r2, c2, p2

    def source_to_proxy(self, row, col, parent=QtCore.QModelIndex()):
        source_model = self.sourceModel()
        source_cols = source_model.columnCount(parent=parent)
        r, c, p = row, col, parent
        r2 = int(math.floor(r / self._nd))
        c2 = ((r % self._nd) * source_cols) + c
        p2 = p
        return r2, c2, p2

    def mapToSource(self, proxyIndex):
        """
        returns index into original model
        """
        if proxyIndex.isValid():
            r2, c2, p2 = self.proxy_to_source(proxyIndex.row(), proxyIndex.column())
            #print('StripeProxyModel.mapToSource(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            idx = self.sourceModel().index(r2, c2, parent=p2)  # self.sourceModel().root_node[r2]
        else:
            idx = QtCore.QModelIndex()
        return idx

    def mapFromSource(self, sourceIndex):
        """
        returns index into proxy model
        """
        if sourceIndex.isValid():
            r2, c2, p2 = self.source_to_proxy(sourceIndex.row(), sourceIndex.column(), sourceIndex.parent())
            #print('StripeProxyModel.mapFromSource(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            idx = self.index(r2, c2, p2)
        else:
            idx = QtCore.QModelIndex()
        return idx

    def filterAcceptsColumn(self, source_column, source_parent):
        print(source_column, source_parent)
        return False

    def index(self, row, col, parent=QtCore.QModelIndex()):
        if (row, col) != (-1, -1):
            idx = self.createIndex(row, col, parent)
            #sidx = self.mapToSource(idx)
            #print('stripeproxymodel.index: (%r, %r) -> (%r, %r)' % (idx.row(), idx.column(), sidx.row(), sidx.column()))
        else:
            idx = QtCore.QModelIndex()
        return idx

    def data(self, proxyIndex, role=Qt.DisplayRole):
        idx = self.mapToSource(proxyIndex)
        return self.sourceModel().data(idx, role)

    def setData(self, proxyIndex, value, role=Qt.EditRole):
        idx = self.mapToSource(proxyIndex)
        return self.sourceModel().setData(idx, value, role)

    def sort(self, column, order):
        source_model = self.sourceModel()
        source_cols = source_model.columnCount()
        if source_cols > 0:
            source_model.sort(column % source_cols, order)

    def parent(self, index):
        return self.sourceModel().parent(self.mapToSource(index))

    def _update_rows(self):
        return self.sourceModel()._update_rows()

    def _get_row_id(self, proxyIndex):
        return self.sourceModel()._get_row_id(self.mapToSource(proxyIndex))
