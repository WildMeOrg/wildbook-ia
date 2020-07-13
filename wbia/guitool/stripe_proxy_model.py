# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtCore, QtGui
from wbia.guitool.__PYQT__ import QtWidgets  # NOQA
from wbia.guitool.__PYQT__.QtCore import Qt
import math
import utool

import six

utool.noinject(__name__, '[StripProxyModel]', DEBUG=False)

# STRIPE_PROXY_BASE = QtGui.QAbstractProxyModel
# STRIPE_PROXY_BASE = QtGui.QSortFilterProxyModel
try:
    STRIPE_PROXY_BASE = QtGui.QIdentityProxyModel
except Exception:
    STRIPE_PROXY_BASE = QtCore.QIdentityProxyModel

STRIP_PROXY_META_CLASS = utool.makeForwardingMetaclass(
    lambda self: self.sourceModel(),
    [
        '_set_context_id',
        '_get_context_id',
        '_set_changeblocked',
        '_get_changeblocked',
        '_about_to_change',
        '_change',
        '_update',
        '_rows_updated',
        'name',
    ],
    base_class=STRIPE_PROXY_BASE,
)

STRIP_PROXY_SIX_BASE = six.with_metaclass(STRIP_PROXY_META_CLASS, STRIPE_PROXY_BASE)


class StripeProxyModel(
    STRIP_PROXY_SIX_BASE
):  # (STRIPE_PROXY_BASE, metaclass=STRIP_PROXY_META_CLASS):
    # __metaclass__ = STRIP_PROXY_META_CLASS

    def __init__(self, parent=None, numduplicates=1):
        STRIPE_PROXY_BASE.__init__(self, parent=parent)
        self._nd = numduplicates

    def rowCount(self, parent=QtCore.QModelIndex()):
        sourceParent = self.mapToSource(parent)
        source_rows = self.sourceModel().rowCount(parent=sourceParent)
        rows = math.ceil(source_rows / self._nd)
        # print('StripeProxyModel.rowCount(): %r %r' % (source_rows, rows))
        return int(rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        source_cols = self.sourceModel().columnCount(parent=parent)
        cols = self._nd * source_cols
        # print('StripeProxyModel.columnCount(): %r %r' % (source_cols, cols))
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
        """ returns index into original model """
        if proxyIndex is None:
            return None
        if proxyIndex.isValid():
            r2, c2, p2 = self.proxy_to_source(proxyIndex.row(), proxyIndex.column())
            # print('StripeProxyModel.mapToSource(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            sourceIndex = self.sourceModel().index(
                r2, c2, parent=p2
            )  # self.sourceModel().root_node[r2]
        else:
            sourceIndex = QtCore.QModelIndex()
        return sourceIndex

    def mapFromSource(self, sourceIndex):
        """ returns index into proxy model """
        if sourceIndex is None:
            return None
        if sourceIndex.isValid():
            r2, c2, p2 = self.source_to_proxy(
                sourceIndex.row(), sourceIndex.column(), sourceIndex.parent()
            )
            proxyIndex = self.index(r2, c2, p2)
        else:
            proxyIndex = QtCore.QModelIndex()
        return proxyIndex

    def index(self, row, col, parent=QtCore.QModelIndex()):
        if (row, col) != (-1, -1):
            proxyIndex = self.createIndex(row, col, parent)
        else:
            proxyIndex = QtCore.QModelIndex()
        return proxyIndex

    def data(self, proxyIndex, role=Qt.DisplayRole, **kwargs):
        sourceIndex = self.mapToSource(proxyIndex)
        return self.sourceModel().data(sourceIndex, role, **kwargs)

    def setData(self, proxyIndex, value, role=Qt.EditRole):
        sourceIndex = self.mapToSource(proxyIndex)
        return self.sourceModel().setData(sourceIndex, value, role)

    def sort(self, column, order):
        source_model = self.sourceModel()
        source_cols = source_model.columnCount()
        if source_cols > 0:
            source_model.sort(column % source_cols, order)

    def parent(self, index):
        return self.sourceModel().parent(self.mapToSource(index))

    #    def mapSelectionToSource(self, sel):

    #    def flags(self, *args, **kwargs):
    #        return self.sourceModel().flags(*args, **kwargs)

    #    def headerData(self, *args, **kwargs):
    #        return self.sourceModel().headerData(*args, **kwargs)
    #
    #    def hasChildren(self, *args, **kwargs):
    #        return self.sourceModel().hasChildren(*args, **kwargs)
    #
    #    def itemData(self, *args, **kwargs):
    #        return self.sourceModel().itemData(*args, **kwargs)

    def _update_rows(self):
        return self.sourceModel()._update_rows()

    def _get_row_id(self, proxyIndex):
        return self.sourceModel()._get_row_id(self.mapToSource(proxyIndex))

    def _get_level(self, proxyIndex):
        return self.sourceModel()._get_level(self.mapToSource(proxyIndex))

    def _get_adjacent_qtindex(self, proxyIndex, *args, **kwargs):
        qtindex = self.mapToSource(proxyIndex)
        next_qtindex = self.sourceModel()._get_adjacent_qtindex(qtindex, *args, **kwargs)
        next_proxyindex = self.mapFromSource(next_qtindex)
        return next_proxyindex
