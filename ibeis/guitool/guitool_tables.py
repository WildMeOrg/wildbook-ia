from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
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

    def __init__(self, column_list, header_names, header_nicenames=None,
                 column_types=None, editable_headers=None, parent=None, *args):
        super(ListTableModel, self).__init__()
        # Internal header names
        self.header_names = header_names
        # Set the data to display (list of lists)
        self.column_list = column_list
        # Set user readable "nice" headers, default to internal ones
        if header_nicenames is None:
            self.header_nicenames = header_names
        else:
            self.header_nicenames = header_nicenames
        # Column datatypes
        if column_types is None:
            try:
                self.column_types = [type(column_data[0]) for column_data in self.column_list]
            except Exception:
                self.column_types = [str] * len(self.column_list)
        else:
            self.column_types = column_types
        # Is editable
        if editable_headers is None:
            self.column_editable = [False] * len(self.column_list)
        else:
            self.column_editable = [header in editable_headers for header in self.header_names]
        assert len(self.header_names) == len(self.column_list)
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

    def set_data(self, index, data):
        """ Non-Qt Helper """
        row = index.row()
        column = index.column()
        row_data = self.column_list[column]
        row_data[self.row_sortx[row]] = data

    def get_header(self, column):
        """ Non-Qt Helper """
        return self.header_names[column]

    def get_header_data(self, header, row):
        """ Non-Qt Helper """
        column = self.header_names.index(header)
        index  = self.index(row, column)
        data   = self.get_data(index)
        return data

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.row_sortx)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.header_names)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        return self.createIndex(row, column)

    def data(self, index, role=Qt.DisplayRole):
        """ Returns the data to display """
        flags = self.flags(index)
        if index.isValid() and role == Qt.BackgroundRole and (flags & Qt.ItemIsEditable):
            return QtCore.QVariant(QtGui.QColor(Qt.red))
        if role == Qt.DisplayRole or (role == Qt.CheckStateRole ):
            data = self.get_data(index)
            value = qtype.cast_into_qt(data, role, flags)
            return value
        #    data = self.get_data(index)
        #    return QtCore.QVariant(Qt.CheckState(data))
        else:
            return QtCore.QVariant()

    def setData(self, index, value, role=Qt.EditRole):
        """ Sets the role data for the item at index to value.
        value is a QVariant (called data in documentation)
        """
        print('About to set data: %r' % (str(qtype.qindexinfo(index))))
        self.debug_qindex.emit(index)
        try:
            row    = index.row()
            column = index.column()
            if role == Qt.CheckStateRole:
                type_ = 'QtCheckState'
                data = value == Qt.Checked
            elif role != Qt.EditRole:
                return False
            else:
                # Cast value into datatype
                type_ = self.column_types[column]
                data = utool.smart_cast(str(value.toString()), type_)
            # Do actual setting of data
            print(' * %s new_data = %r' % (utool.type_str(type_), data,))
            self.set_data(index, data)
            # Emit that data was changed and return succcess
            self.dataChanged.emit(index, index)
            return True
        except Exception as ex:
            print('<!!! ERROR !!!>')
            print(ex)
            try:
                print('row = %r' % (row,))
                print('column = %r' % (column,))
                print('type_ = %r' % (type_,))
                print('value.toString() = %r' % (str(value.toString()),))
            except Exception:
                pass
            print('</!!! ERROR !!!>')
            raise

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.header_nicenames[section]
        else:
            return QtCore.QVariant()

    def sort(self, index, order):
        self.layoutAboutToBeChanged.emit()
        self.db_sort_index = index
        self.db_sort_reversed = order == Qt.DescendingOrder
        self._refresh_row_indicies()
        self.layoutChanged.emit()

    def flags(self, index):
        #return Qt.ItemIsEnabled
        col = index.column()
        if self.column_editable[col]:
            if self.column_types[col] in utool.VALID_BOOL_TYPES:
                #print(col)
                return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
            else:
                return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
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
        #self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)


class DummyListTableWidget(QtGui.QWidget):
    """ Test Main Window """
    def __init__(self, column_values, column_headers, editable_headers=None, parent=None):
        super(DummyListTableWidget, self).__init__(parent)
        # Create vertical layout for the table to go into
        self.vlayout = QtGui.QVBoxLayout(self)
        # Instansiate the AbstractItemModel
        self.list_table_model = ListTableModel(column_values, column_headers,
                                               editable_headers=editable_headers, parent=self)
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


def dummy_list_table(column_values, column_headers, editable_headers=None, show=True, raise_=True,
                     on_click=None):
    widget = DummyListTableWidget(column_values, column_headers,
                                  editable_headers=editable_headers)

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
    widget.setGeometry(20, 50, 600, 800)

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
