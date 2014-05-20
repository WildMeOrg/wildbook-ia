from __future__ import absolute_import, division, print_function
import numpy as np
from PyQt4 import QtCore, QtGui


def make_header_lists(tbl_headers, editable_list, prop_keys=[]):
    col_headers = tbl_headers[:] + prop_keys
    col_editable = [False] * len(tbl_headers) + [True] * len(prop_keys)
    for header in editable_list:
        col_editable[col_headers.index(header)] = True
        return col_headers, col_editable


class ListTableModel(QtCore.QAbstractTableModel):
    """ Does the lazy loading magic
    http://qt-project.org/doc/qt-5/QAbstractItemModel.html
    """
    def __init__(self, column_list, headers_name, headers_nice=None, parent=None, *args):
        super(ListTableModel, self).__init__()
        self.headers_name = headers_name
        self.headers_nice = headers_name if headers_nice is None else headers_nice

        self.column_list = column_list  # the numpy array or list of lists
        assert len(self.headers_name) == len(self.column_list)
        self.db_sort_index = 0
        self.db_sort_reversed = False
        self.row_indices = []

        self._refresh_row_indicies()

    def _refresh_row_indicies(self):
        """ NonQT """
        self.row_indices = np.arange(len(self.column_list[self.db_sort_index]))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.row_indices)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers_name)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            column = index.column()
            row_data = self.column_list[column]
            return str(row_data[row])
        else:
            return QtCore.QVariant()

    def setData(self, index, data, role=QtCore.Qt.EditRole):
        """ Sets the role data for the item at index to value. """
        if role != QtCore.Qt.EditRole:
            return False
        print('Setting Data: %r, %r' % (index, data))
        #if result is True:
        #    self.dataChanged.emit(index, index)
        #return result

    def headerData(self, index, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.headers_nice[index]
        else:
            return QtCore.QVariant()

    def sort(self, index, order):
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self.db_sort_index = index
        self.db_sort_reversed = order == QtCore.Qt.DescendingOrder
        self._refresh_row_indicies()
        self.emit(QtCore.SIGNAL("layoutChanged()"))

    def flags(self, index):
        #return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        #return Qt.ItemFlag(0)


class TableView(QtGui.QTableView):
    """ The table view houses the AbstractItemModel

    Public Signals:
        activated(QModelIndex index)
        clicked(QModelIndex index)
        doubleClicked( QModelIndex index )
        entered(QModelIndex index)
        pressed(QModelIndex index)
        viewportEntered()
        customContextMenuRequested(QPoint pos)

    Public Slots:
        clearSelection ()
        edit(QModelIndex index)
        reset()
        scrollToBottom()
        scrollToTop()
        selectAll()
        setCurrentIndex(QModelIndex index)
        setRootIndex(QModelIndex index)
        update(QModelIndex index)

    """
    def __init__(self, *args, **kwargs):
        QtGui.QTableView.__init__(self, *args, **kwargs)
        self.setSortingEnabled(True)

        vh = self.verticalHeader()
        vh.setVisible(False)

        self.resizeColumnsToContents()


class DummyListTableWidget(QtGui.QWidget):
    """ Test Main Window """
    def __init__(self, column_values, column_headers, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.vlayout = QtGui.QVBoxLayout(self)
        self._tm = ListTableModel(column_values, column_headers, parent=self)
        self._tv = TableView(self)
        self._tv.setModel(self._tm)
        self.vlayout.addWidget(self._tv)


def make_dummy_table_widget(column_values, column_headers,
                            show=True, raise_=True):
    widget = DummyListTableWidget(column_values, column_headers)
    if show:
        widget.show()
    if raise_:
        widget.raise_()
    return widget

#if __name__ == '__main__':
    #import sys
    #app = QtGui.QApplication(sys.argv)
    #widget = DummyWidget()
    #widget.show()
    #widget.raise_()
    #sys.exit(app.exec_())
