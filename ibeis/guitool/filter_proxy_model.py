from __future__ import absolute_import, division, print_function
from guitool_ibeis.__PYQT__ import QtGui, QtCore  # NOQA
from guitool_ibeis.__PYQT__.QtCore import Qt
import utool

utool.noinject(__name__, '[APIItemView]', DEBUG=False)

#BASE_CLASS = QtGui.QAbstractProxyModel
try:
    BASE_CLASS = QtGui.QSortFilterProxyModel
except Exception:
    BASE_CLASS = QtCore.QIdentityProxyModel
# BASE_CLASS = QtGui.QIdentityProxyModel


class FilterProxyModel(BASE_CLASS):
    __metaclass__ = utool.makeForwardingMetaclass(
        lambda self: self.sourceModel(),
        ['_set_context_id', '_get_context_id', '_set_changeblocked',
         '_get_changeblocked', '_about_to_change', '_change', '_update',
         '_rows_updated', 'name', 'get_header_name'],
        base_class=BASE_CLASS)

    def __init__(self, parent=None):
        BASE_CLASS.__init__(self, parent=parent)
        self.filter_dict = {}

    def proxy_to_source(self, row, col, parent=QtCore.QModelIndex()):
        r2, c2, p2 = row, col, parent
        return r2, c2, p2

    def source_to_proxy(self, row, col, parent=QtCore.QModelIndex()):
        r2, c2, p2 = row, col, parent
        return r2, c2, p2

    def mapToSource(self, proxyIndex):
        """ returns index into original model """
        if proxyIndex is None:
            return None
        if proxyIndex.isValid():
            r2, c2, p2 = self.proxy_to_source(proxyIndex.row(), proxyIndex.column())
            sourceIndex = self.sourceModel().index(r2, c2, parent=p2)  # self.sourceModel().root_node[r2]
        else:
            sourceIndex = QtCore.QModelIndex()
        return sourceIndex

    def mapFromSource(self, sourceIndex):
        """ returns index into proxy model """
        if sourceIndex is None:
            return None
        if sourceIndex.isValid():
            r2, c2, p2 = self.source_to_proxy(sourceIndex.row(), sourceIndex.column(), sourceIndex.parent())
            proxyIndex = self.index(r2, c2, p2)
        else:
            proxyIndex = QtCore.QModelIndex()
        return proxyIndex

    def filterAcceptsRow(self, source_row, source_parent):
        source = self.sourceModel()
        row_type = str(source.data(source.index(source_row, 2, parent=source_parent)))
        #print('%r \'%r\'' % (source_row, row_type))
        #print(self.filter_dict)
        rv = self.filter_dict.get(row_type, True)
        #print('return value %r' % rv)
        return rv

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
        self.sourceModel().sort(column, order)

    def parent(self, index):
        return self.sourceModel().parent(self.mapToSource(index))

    def get_header_data(self, colname, proxyIndex):
        #print('[guitool_ibeis] calling default map to source')
        #print('[guitool_ibeis] proxyIndex=%r' % proxyIndex)
        #proxy_keys = dir(proxyIndex)
        #proxy_vals = [getattr(proxyIndex, key) for key in proxy_keys]
        #proxy_dict = dict(zip(proxy_keys, proxy_vals))
        #print('[guitool_ibeis] proxyIndex.__dict__=%s' % utool.repr2(proxy_dict))
        #utool.embed()
        #sourceIndex = BASE_CLASS.mapToSource(self, proxyIndex)
        sourceIndex = self.mapToSource(proxyIndex)
        #print('[guitool_ibeis] calling set header')
        ret = self.sourceModel().get_header_data(colname, sourceIndex)
        #print('[guitool_ibeis] finished')
        return ret

    def update_filterdict(self, new_dict):
        self.filter_dict = new_dict

    def _update_rows(self):
        return self.sourceModel()._update_rows()

    def _get_row_id(self, proxyIndex):
        return self.sourceModel()._get_row_id(self.mapToSource(proxyIndex))
