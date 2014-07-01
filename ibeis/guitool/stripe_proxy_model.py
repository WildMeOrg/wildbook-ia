from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
import math

#BASE_CLASS = QtGui.QAbstractProxyModel
#BASE_CLASS = QtGui.QSortFilterProxyModel
BASE_CLASS = QtGui.QIdentityProxyModel


# makes a metaclass that overrides __getattr__ and __setattr__ to forward some specific attribute references to a specified instance variable
def makeForwardingMetaclass(forwarding_dest_getter, whitelist):
    class ForwardingMetaclass(BASE_CLASS.__class__):
        def __init__(cls, name, bases, dct):
            print('ForwardingMetaclass.__init__(): {forwarding_dest_getter: %r; whitelist: %r}' % (forwarding_dest_getter, whitelist))
            super(ForwardingMetaclass, cls).__init__(name, bases, dict)
            old_getattr = cls.__getattribute__
            def new_getattr(obj, item):
                if item in whitelist:
                    #dest = old_getattr(obj, forwarding_dest_name)
                    dest = forwarding_dest_getter(obj)
                    try:
                        val = dest.__class__.__getattribute__(dest, item)
                    except AttributeError:
                        val = getattr(dest, item)
                else:
                    val = old_getattr(obj, item)
                return val
            cls.__getattribute__ = new_getattr
            old_setattr = cls.__setattr__
            def new_setattr(obj, name, val):
                if name in whitelist:
                    #dest = old_getattr(obj, forwarding_dest_name)
                    dest = forwarding_dest_getter(obj)
                    dest.__class__.__setattr__(dest, name, val)
                else:
                    old_setattr(obj, name, val)
            cls.__setattr__ = new_setattr
    return ForwardingMetaclass


class StripeProxyModel(BASE_CLASS):
    __metaclass__ = makeForwardingMetaclass(lambda self: self.sourceModel(), ['_set_context_id', '_get_context_id', '_set_changeblocked', '_get_changeblocked', '_about_to_change', '_change', '_update', '_rows_updated', 'name'])

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

#    def mapSelectionToSource(self, sel):

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

#    def flags(self, *args, **kwargs):
#        return self.sourceModel().flags(*args, **kwargs)

    def parent(self, index):
        return self.sourceModel().parent(self.mapToSource(index))

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
