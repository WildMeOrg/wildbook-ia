from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from . import qtype
import utool


def make_header_lists(tbl_headers, editable_list, prop_keys=[]):
    col_headers = tbl_headers[:] + prop_keys
    col_editable = [False] * len(tbl_headers) + [True] * len(prop_keys)
    for header in editable_list:
        col_editable[col_headers.index(header)] = True
        return col_headers, col_editable


class ListTableModel(QtCore.QAbstractTableModel):
    """
    Does the lazy loading magic
    http://qt-project.org/doc/qt-5/QAbstractItemModel.html

    Public Signals:
        columnsAboutToBeInserted(QModelIndex parent, int start, int end )
        columnsAboutToBeMoved(QModelIndex sourceParent, int sourceStart, int sourceEnd, QModelIndex destinationParent, int destinationColumn )
        columnsAboutToBeRemoved(QModelIndex parent, int start, int end )
        columnsInserted(QModelIndex parent, int start, int end )
        columnsMoved(QModelIndex sourceParent, int sourceStart, int sourceEnd, QModelIndex destinationParent, int destinationColumn )
        columnsRemoved(QModelIndex parent, int start, int end )
        dataChanged(QModelIndex topLeft, QModelIndex bottomRight )
        headerDataChanged(Qt::Orientation orientation, int first, int last )
        layoutAboutToBeChanged()
        layoutChanged()
        modelAboutToBeReset()
        modelReset()
        rowsAboutToBeInserted(QModelIndex parent, int start, int end )
        rowsAboutToBeMoved(QModelIndex sourceParent, int sourceStart, int sourceEnd, QModelIndex destinationParent, int destinationRow )
        rowsAboutToBeRemoved(QModelIndex parent, int start, int end )
        rowsInserted(QModelIndex parent, int start, int end )
        rowsMoved(QModelIndex sourceParent, int sourceStart, int sourceEnd, QModelIndex destinationParent, int destinationRow )
        rowsRemoved(QModelIndex parent, int start, int end )
    """
    debug_qindex = QtCore.pyqtSignal(QtCore.QModelIndex)

    def __init__(self, column_list, headers_name, headers_nice=None,
                 column_types=None, parent=None, *args):
        super(ListTableModel, self).__init__()
        # Internal header names
        self.headers_name = headers_name
        # Set the data to display (list of lists)
        self.column_list = column_list
        # Set user readable "nice" headers, default to internal ones
        if headers_nice is None:
            self.headers_nice = headers_name
        else:
            self.headers_nice = headers_nice
        # Column datatypes
        if column_types is None:
            try:
                self.column_types = [type(column_data[0]) for column_data in self.column_list]
            except Exception:
                self.column_types = [str] * len(self.column_list)
        else:
            self.column_types = column_types
        assert len(self.headers_name) == len(self.column_list)
        self.db_sort_index = 0
        self.db_sort_reversed = False
        self.row_sortx = []
        # Refresh state
        self._refresh_row_indicies()

    def _refresh_row_indicies(self):
        """  Non-Qt Helper """
        column_values = self.column_list[self.db_sort_index]
        indicies = range(len(column_values))
        #if self.db_sort_reversed:
        #    self.row_sortx = indicies[::-1]
        #else:
        self.row_sortx = indicies
        self.row_sortx = utool.sortedby(indicies, column_values, reverse=self.db_sort_reversed)

    def get_data(self, index):
        """ Non-Qt Helper """
        row = index.row()
        column = index.column()
        row_data = self.column_list[column]
        data = row_data[self.row_sortx[row]]
        return data

    def get_header(self, column):
        """ Non-Qt Helper """
        return self.headers_name[column]

    def get_header_data(self, header, row):
        """ Non-Qt Helper """
        column = self.headers_name.index(header)
        index  = self.index(row, column)
        data   = self.get_data(index)
        return data

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.row_sortx)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers_name)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        return self.createIndex(row, column)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """ Returns the data to display """
        if role == QtCore.Qt.DisplayRole:
            data = self.get_data(index)
            var = qtype.cast_into_qt(data)
            return var
        else:
            return QtCore.QVariant()

    def setData(self, index, var, role=QtCore.Qt.EditRole):
        """ Sets the role data for the item at index to value.
        var is a QVariant (called data in documentation)
        """
        self.debug_qindex.emit(index)
        if role != QtCore.Qt.EditRole:
            return False
        row    = index.row()
        column = index.column()
        type_ = self.column_types[column]
        try:
            # Cast var into datatype
            data = utool.smart_cast(str(var.toString()), type_)
            # Do actual setting of data
            self.column_list[column][row] = data
            print('Setting Data: %r, %r' % (str(qtype.qindexinfo(index)), data))
            # Emit that data was changed and return succcess
            self.dataChanged.emit(index, index)
            return True
        except Exception as ex:
            print('<!!! ERROR !!!>')
            print(ex)
            print('row = %r' % (row,))
            print('column = %r' % (column,))
            print('type_ = %r' % (type_,))
            print('var.toString() = %r' % (str(var.toString()),))
            print('</!!! ERROR !!!>')
            raise
        #if result is True:
        #    self.dataChanged.emit(index, index)
        #return result

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.headers_nice[section]
        else:
            return QtCore.QVariant()

    def sort(self, index, order):
        self.layoutAboutToBeChanged.emit()
        self.db_sort_index = index
        self.db_sort_reversed = order == QtCore.Qt.DescendingOrder
        self._refresh_row_indicies()
        self.layoutChanged.emit()

    def flags(self, index):
        #return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        #return Qt.ItemFlag(0)


class TableView(QtGui.QTableView):
    """
    The table view houses the AbstractItemModel

    Public Signals:
        activated(QModelIndex index)
        clicked(QModelIndex index)
        doubleClicked(QModelIndex index)
        entered(QModelIndex index)
        pressed(QModelIndex index)
        viewportEntered()
        customContextMenuRequested(QPoint pos)

    Public Slots:
        clearSelection()
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
        super(TableView, self).__init__(*args, **kwargs)
        self.setSortingEnabled(True)

        vh = self.verticalHeader()
        vh.setVisible(False)

        self.resizeColumnsToContents()
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)


class DummyListTableWidget(QtGui.QWidget):
    """ Test Main Window """
    def __init__(self, column_values, column_headers, parent=None):
        super(DummyListTableWidget, self).__init__(parent)
        # Create vertical layout for the table to go into
        self.vlayout = QtGui.QVBoxLayout(self)
        # Instansiate the AbstractItemModel
        self.list_table_model = ListTableModel(column_values, column_headers, parent=self)
        # Create a TableView for the AbstractItemModel
        self.table_view = TableView(self)
        self.table_view.setModel(self.list_table_model)
        self.vlayout.addWidget(self.table_view)


global_index  = None


def _receive_global_index(index):
    global global_index
    global_index = index
    #print('receiving global index: %r' % (index,))
    """
    From getting this QModelIndex I learned
    that its methods are used like this:

    **data, -> QVariant
    **model, -> ListTableModel
    **column, -> int
    **row, -> int
    flags, -> QtCore.ItemFlags
    isValid, -> bool
    child, -> QModelIndex
    parent, -> QModelIndex
    sibling -> QModelIndex
    internalId, -> int
    internalPointer, -> None
    """


def dummy_list_table(column_values, column_headers, show=True, raise_=True,
                     on_click=None):
    widget = DummyListTableWidget(column_values, column_headers)

    def on_doubleclick(index):
        # This is actually a release
        #print('DoubleClicked: ' + str(qtype.qindexinfo(index)))
        pass

    def on_pressed(index):
        #print('Pressed: ' + str(qtype.qindexinfo(index)))
        pass

    def on_activated(index):
        #print('Activated: ' + str(qtype.qindexinfo(index)))
        pass
    if on_click is not None:
        widget.table_view.clicked.connect(on_click)
    widget.table_view.doubleClicked.connect(on_doubleclick)
    widget.table_view.pressed.connect(on_doubleclick)
    widget.table_view.activated.connect(on_activated)

    widget.list_table_model.debug_qindex.connect(_receive_global_index)

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
