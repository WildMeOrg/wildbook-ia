from __future__ import absolute_import, division, print_function
from PyQt4 import QtGui
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
                                                        'name',
                                                        'get_header_name'],
                                                    base_class=BASE_CLASS)

    def __init__(self, parent=None):
        BASE_CLASS.__init__(self, parent=parent)
        self.filter_dict = {}

    def filterAcceptsRow(self, source_row, source_parent):
        source = self.sourceModel()
        row_type = str(source.data(source.index(source_row, 2, parent=source_parent)))
        #print('%r \'%r\'' % (source_row, row_type))
        #print(self.filter_dict)
        rv = self.filter_dict.get(row_type, True)
        #print('return value %r' % rv)
        return rv

    def update_filterdict(self, new_dict):
        self.filter_dict = new_dict

    def _update_rows(self):
        return self.sourceModel()._update_rows()

    def _get_row_id(self, proxyIndex):
        return self.sourceModel()._get_row_id(self.mapToSource(proxyIndex))

    def get_header_data(self, colname, qtidx):
        sidx = self.mapToSource(qtidx)
        return self.sourceModel().get_header_data(colname, sidx)

    def sort(self, column, order):
        self.sourceModel().sort(column, order)
