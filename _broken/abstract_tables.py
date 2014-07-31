from __future__ import absolute_import, division, print_function
import sys
import traceback
from PyQt4.QtCore import (QAbstractItemModel, QModelIndex, QVariant, QString,
                          Qt, QObject)


# Decorator to help catch errors that QT wont report
def report_thread_error(fn):
    def report_thread_error_wrapper(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
            return ret
        except Exception as ex:
            print('\n\n *!!* Thread Raised Exception: ' + str(ex))
            print('\n\n *!!* Thread Exception Traceback: \n\n' + traceback.format_exc())
            sys.stdout.flush()
            et, ei, tb = sys.exc_info()
            raise
    return report_thread_error_wrapper


class IBEIS_QTable(QAbstractItemModel):
    """ Convention states only items with column index 0 can have children """
    @report_thread_error
    def __init__(self, ibs,
                 tblname='gids',
                 tblcols=['gid', 'gname'],
                 fancycols_dict={},
                 tbleditable=[],
                 parent=None):
        super(IBEIS_QTable, self).__init__(parent)
        self.ibs  = ibs
        self.tblname = tblname
        self.fancycols_dict = fancycols_dict

    @report_thread_error
    def index2_tableitem(self, index=QModelIndex()):
        """ Internal helper method """
        if index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return None

    #-----------
    # Overloaded ItemModel Read Functions
    @report_thread_error
    def rowCount(self, parent=QModelIndex()):
        parentPref = self.index2_tableitem(parent)
        return parentPref.qt_row_count()

    @report_thread_error
    def columnCount(self, parent=QModelIndex()):
        parentPref = self.index2_tableitem(parent)
        return parentPref.qt_col_count()

    @report_thread_error
    def data(self, index, role=Qt.DisplayRole):
        """ Returns the data stored under the given role
        for the item referred to by the index. """
        if not index.isValid():
            return QVariant()
        if role != Qt.DisplayRole and role != Qt.EditRole:
            return QVariant()
        nodePref = self.index2_tableitem(index)
        data = nodePref.qt_get_data(index.column())
        var = QVariant(data)
        #print('--- data() ---')
        #print('role = %r' % role)
        #print('data = %r' % data)
        #print('type(data) = %r' % type(data))
        if isinstance(data, float):
            var = QVariant(QString.number(data, format='g', precision=6))
        if isinstance(data, bool):
            var = QVariant(data).toString()
        if isinstance(data, int):
            var = QVariant(data).toString()
        #print('var= %r' % var)
        #print('type(var)= %r' % type(var))
        return var

    @report_thread_error
    def index(self, row, col, parent=QModelIndex()):
        """ Returns the index of the item in the model specified
        by the given row, column and parent index. """
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()
        parentPref = self.index2_tableitem(parent)
        childPref  = parentPref.qt_get_child(row)
        if childPref:
            return self.createIndex(row, col, childPref)
        else:
            return QModelIndex()

    @report_thread_error
    def parent(self, index=None):
        """ Returns the parent of the model item with the given index.
        If the item has no parent, an invalid QModelIndex is returned. """
        if index is None:  # Overload with QObject.parent()
            return QObject.parent(self)
        if not index.isValid():
            return QModelIndex()
        nodePref = self.index2_tableitem(index)
        parentPref = nodePref.qt_get_parent()
        if parentPref == self.rootPref:
            return QModelIndex()
        return self.createIndex(parentPref.qt_parents_index_of_me(), 0, parentPref)

    #-----------
    # Overloaded ItemModel Write Functions
    @report_thread_error
    def flags(self, index):
        """ Returns the item flags for the given index. """
        if index.column() == 0:
            # The First Column is just a label and unchangable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag(0)
        item_col, item_rowid = self.index2_itemdata(index)
        if item_rowid:
            if item_col in self.col_editable:
                return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemFlag(0)

    @report_thread_error
    def setData(self, index, data, role=Qt.EditRole):
        """ Sets the role data for the item at index to value. """
        if role != Qt.EditRole:
            return False
        #print('--- setData() ---')
        #print('role = %r' % role)
        #print('data = %r' % data)
        #print('type(data) = %r' % type(data))
        leafPref = self.index2_tableitem(index)
        result = leafPref.qt_set_leaf_data(data)
        if result is True:
            self.dataChanged.emit(index, index)
        return result

    @report_thread_error
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            column_key = self.table_headers[section]
            column_name = self.fancycols_dict.get(column_key, column_key)
            return QVariant(column_name)
        return QVariant()
