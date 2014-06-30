from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from .guitool_decorators import checks_qt_error, signal_, slot_
from . import qtype
import math

#BASE_CLASS = QtGui.QAbstractProxyModel
#BASE_CLASS = QtGui.QSortFilterProxyModel
BASE_CLASS = QtGui.QIdentityProxyModel

# ["__init__", "rowCount", "columnCount", "mapToSource", "mapFromSource", "index", "data", "setData", "sort"]

#_DID_IBEISTABLEMODEL_METACLASS_HACK = False
    #global _DID_IBEISTABLEMODEL_METACLASS_HACK
#        if not _DID_IBEISTABLEMODEL_METACLASS_HACK:
#            _DID_IBEISTABLEMODEL_METACLASS_HACK = True
#            exclude_list = ["_update_headers", "_ider", "_change_enc", "sourcemodel", "__class__", "__setattr__", "__getattr__", "_nd", "sourceModel", "__init__", "rowCount", "columnCount", "mapToSource", "mapFromSource", "index", "data", "setData", "sort"]
#            old_getattr = model.__class__.__getattr__
#            #print('old_getattr outside: %r' % old_getattr)
#            def new_getattr(obj, item):
#                #print('old_getattr is %r' % old_getattr)
#                #print('new_getattr(%r, %r)' % (obj, item))
#                if item not in exclude_list:
#                    #print('sourcemodel.dict %r' % model.sourcemodel.__dict__)
#                    try:
#                        val = old_getattr(model.sourcemodel, item)
#                    except AttributeError:
#                        val = getattr(model.sourcemodel, item)
#                else:
#                    val = old_getattr(obj, item)
#                #print('new_getattr returning %r' % val)
#                return val
#            model.__class__.__getattr__ = new_getattr
#
#            old_setattr = model.__class__.__setattr__
#            def new_setattr(obj, name, val):
#                #print('new_setattr(%r, %r, %r)' % (obj, name, val))
#                if name not in exclude_list:
#                    old_setattr(model.sourcemodel, name, val)
#                else:
#                    old_setattr(obj, name, val)
#            model.__class__.__setattr__ = new_setattr

class StripeProxyModel(BASE_CLASS):
    _rows_updated = signal_(str, int)

    @slot_(str, int)
    def _on_rows_updated(self, *args, **kwargs):
        self._rows_updated.emit(*args, **kwargs)

    def setSourceModel(self, source):
        source._rows_updated.connect(self._on_rows_updated)
        return BASE_CLASS.setSourceModel(self, source)

    def __init__(self, parent=None, numduplicates=1):
        BASE_CLASS.__init__(self, parent=parent)
        self._nd = numduplicates

    def rowCount(self, parent=QtCore.QModelIndex()):
        source_rows = self.sourceModel().rowCount(parent=parent)
        rows = math.ceil(source_rows / self._nd)
        print('StripeProxyModel.rowCount(): %r %r' % (source_rows, rows))
        return int(rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        source_cols = self.sourceModel().columnCount(parent=parent)
        cols = self._nd * source_cols
        print('StripeProxyModel.columnCount(): %r %r' % (source_cols, cols))
        return int(cols)

    def proxy_to_source(self, row, col, parent=QtCore.QModelIndex()):
        source_model = self.sourceModel()
        source_rows = source_model.rowCount(parent=parent)
        source_cols = source_model.columnCount(parent=parent)
        r, c, p = row, col, parent
        r2 = int(math.floor(c / source_cols)) + (r * self._nd)
        c2 = c % source_cols
        p2 = p
        return r2, c2, p2

    def source_to_proxy(self, row, col, parent=QtCore.QModelIndex()):
        source_model = self.sourceModel()
        source_rows = source_model.rowCount(parent=parent)
        source_cols = source_model.columnCount(parent=parent)
        r, c, p = row, col, parent
        r2 = int(math.floor(r/self._nd))
        c2 = ((r%self._nd)*source_cols)+c
        p2 = p
        return r2, c2, p2

    def mapToSource(self, proxyIndex):
        """
        returns index into original model
        """
        if proxyIndex.isValid():
            r2, c2, p2 = self.proxy_to_source(proxyIndex.row(), proxyIndex.column(), proxyIndex.parent())
            #print('StripeProxyModel.mapToSource(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            #idx = self.sourceModel().index(r2, c2, p2)
            idx = self.index(r2, c2, p2)
            idx.__dict__['actual'] = self.sourceModel().index(r2, c2, parent=p2)
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
            #idx = self.sourceModel().index(r2, c2, p2)
            idx = self.index(r2, c2, p2)
            idx.__dict__['actual'] = self.sourceModel().index(r2, c2, parent=p2)
        else:
            idx = QtCore.QModelIndex()
        return idx

#    def mapSelectionToSource(self, sel):
#        

    def index(self, row, col, parent=QtCore.QModelIndex()):
        if (row, col) != (-1, -1):
            r2, c2, p2 = self.proxy_to_source(row, col, parent)
            #print('StripeProxyModel.index(): %r %r %r; %r %r %r' % (r, c, p, r2, c2, p2))
            #idx = self.sourceModel().index(r2, c2, parent=p2)
            idx = self.createIndex(r2, c2, parent)
            idx.__dict__['actual'] = self.sourceModel().index(r2, c2, parent=p2)
#            def f():
#                print('in inline parent %r' % parent)
#                return parent
#            idx.parent = f
        else:
            idx = QtCore.QModelIndex()
        #print('stripeproxymodel.index: (%r, %r) -> (%r, %r)' % (row, col, idx.row(), idx.column()))
        return idx

    def data(self, proxyIndex, role=Qt.DisplayRole):
        #print('stripeproxymodel.data %r; %r' % (proxyIndex, type(proxyIndex)))
        #print(proxyIndex.parent, proxyIndex.parent())
        #idx = self.sourceModel().index(proxyIndex.row(), proxyIndex.column(), proxyIndex.parent())
        idx = self.sourceModel().index(proxyIndex.row(), proxyIndex.column())
#        assert proxyIndex.isValid()
#        idx = proxyIndex.__dict__['actual']
        #return self.sourceModel().data(self.mapToSource(proxyIndex), role)
        return self.sourceModel().data(idx, role)
        #return self.sourceModel().data(proxyIndex, role)

#    def setData(self, proxyIndex, value, role=Qt.EditRole):
#        print('stripeproxymodel.setdata')
#        #return self.sourceModel().setData(self.mapToSource(proxyIndex), value, role)
#        return self.sourceModel().setData(proxyIndex, value, role)

    def sort(self, column, order):
        source_model = self.sourceModel()
        source_cols = source_model.columnCount()
        if source_cols > 0:
            source_model.sort(column % source_cols, order)

#    def flags(self, *args, **kwargs):
#        return self.sourceModel().flags(*args, **kwargs)

    def parent(self, index):
        #print('hello')
        #return self.sourceModel().parent(*args, **kwargs)
        return QtCore.QModelIndex()

#    def headerData(self, *args, **kwargs):
#        return self.sourceModel().headerData(*args, **kwargs)
#
#    def hasChildren(self, *args, **kwargs):
#        return self.sourceModel().hasChildren(*args, **kwargs)
#
#    def itemData(self, *args, **kwargs):
#        return self.sourceModel().itemData(*args, **kwargs)

    def _update_headers(self, **kwargs):
        print('forwarding update headers')
        return self.sourceModel()._update_headers(**kwargs)

    def _update_rows(self):
        return self.sourceModel()._update_rows()

    def _set_context_id(self, id_):
        self.sourceModel()._set_context_id(id_)

    def _get_context_id(self):
        return self.sourceModel()._get_context_id()

    def _set_changeblocked(self, changeblocked_):
        self.sourceModel()._set_changeblocked(changeblocked_)

    def _get_changeblocked(self):
        return self.sourceModel()._get_changeblocked()

    def _about_to_change(self, *args, **kwargs):
        return self.sourceModel()._about_to_change(*args, **kwargs)

    def _change(self, *args, **kwargs):
        return self.sourceModel()._change(*args, **kwargs)

    def _update(self, *args, **kwargs):
        return self.sourceModel()._update(*args, **kwargs)

    def _get_row_id(self, *args, **kwargs):
        return self.sourceModel()._get_row_id(*args, **kwargs)
