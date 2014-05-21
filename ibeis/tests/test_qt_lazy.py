#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join

from PyQt4 import QtCore, QtGui
import string
import random

print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_LAZY] ')


def create_databse():
    def _randstr(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    sqldb_fname = 'data_test_qt.sqlite3'
    sqldb_dpath = utool.util_cplat.get_app_resource_dir('ibeis', 'testfiles')
    utool.ensuredir(sqldb_dpath)
    utool.util_path.remove_file(join(sqldb_dpath, sqldb_fname), dryrun=False)
    db = SQLDatabaseControl.SQLDatabaseController(sqldb_dpath=sqldb_dpath,
                                                  sqldb_fname=sqldb_fname)

    encounters = [
        ('encounter_id',      'INTEGER PRIMARY KEY'),
        ('encounter_name',    'TEXT'),
    ]
    db.schema('encounters', encounters)

    rows = 1 * (10 ** 3)
    feats_iter = ( (_randstr(), ) for i in xrange(rows) )

    print('[TEST] insert encounters')
    tt = utool.tic()
    db.executemany(operation='''
        INSERT
        INTO encounters
        (
            encounter_name
        )
        VALUES (?)
        ''', params_iter=feats_iter)
    print(' * execute insert time=%r sec' % utool.toc(tt))

    ##############################################
    headers = [
        ('data_id',      'INTEGER PRIMARY KEY'),
        ('encounter_id', 'INT'),
        ('data_float',   'FLOAT'),
        ('data_int',     'INT'),
        ('data_text',    'TEXT'),
        ('data_text2',   'TEXT'),
    ]
    headers_name = [ column[0] for column in headers ]
    headers_nice = [
        'ID',
        'Encounter ID',
        'TEST Float',
        'TEST Int',
        'TEST String 1',
        'TEST String 2',
    ]
    db.schema('data', headers)

    rows = 1 * (10 ** 5)
    feats_iter = ((random.randint(0, 1000), random.uniform(0.0, 1.0), random.randint(0, 100), _randstr(), _randstr())
                        for i in xrange(rows) )

    print('[TEST] insert data')
    tt = utool.tic()
    db.executemany(operation='''
        INSERT
        INTO data
        (
            encounter_id,
            data_float,
            data_int,
            data_text,
            data_text2
        )
        VALUES (?,?,?,?,?)
        ''', params_iter=feats_iter)
    print(' * execute insert time=%r sec' % utool.toc(tt))

    return headers_name, headers_nice, db


class TableModel_SQL(QtCore.QAbstractTableModel):
    """ Does the lazy loading magic
    http://qt-project.org/doc/qt-5/QAbstractItemModel.html
    """
    def __init__(self, headers_name, headers_nice, db, parent=None, *args):
        super(TableModel_SQL, self).__init__()
        self.headers_name = headers_name
        self.headers_nice = headers_nice
        self.encounter = -1

        self.db = db
        self.db_sort_index = 0
        self.db_sort_reversed = False
        self.row_indices = []

        self._refresh_row_indicies()

    def _refresh_row_indicies(self):
        """ NonQT """
        column = self.headers_name[self.db_sort_index]
        order = (' DESC' if self.db_sort_reversed else ' ASC')
        query = 'SELECT data_id FROM data WHERE (? IS -1 OR encounter_id=?) ORDER BY ' + column + order
        self.db.execute(query, [self.encounter, self.encounter])
        self.row_indices = [result for result in self.db.result_iter()]

    def _change_encounter(self, encounter):
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self.encounter = encounter
        self._refresh_row_indicies()
        self.emit(QtCore.SIGNAL("layoutChanged()"))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.row_indices)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers_name)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            column = index.column()

            self.db.execute('SELECT * FROM data WHERE data_id=?', [self.row_indices[row]])
            row_data = list(self.db.result())
            return str(row_data[column])
        else:
            return QtCore.QVariant()

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        """ Sets the role data for the item at index to value. """
        if role == QtCore.Qt.EditRole:
            value = str(value.toString())
            if value != "":
                row = index.row()
                column = index.column()

                query = 'UPDATE data SET ' + self.headers_name[column] + '=? WHERE data_id=?'
                self.db.execute(query, [value, self.row_indices[row]])

                self.emit(QtCore.SIGNAL("dataChanged()"))
                return True

        return False

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
        if index.column() > 0:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

class ListModel_SQL(QtCore.QAbstractListModel):
    """ Does the lazy loading magic
    http://qt-project.org/doc/qt-5/QAbstractItemModel.html
    """
    def __init__(self, db, tm, parent=None, *args):
        super(ListModel_SQL, self).__init__()
        self.db = db
        self.tm = tm
        self.row_indices = []

        self._refresh_row_indicies()

    def _refresh_row_indicies(self):
        """ NonQT """
        self.db.execute('SELECT encounter_id FROM encounters ORDER BY encounter_id ASC', [])
        self.row_indices = [result for result in self.db.result_iter()]

    def _change_encounter(self, index):
        row = index.row()
        self.db.execute('SELECT * FROM encounters WHERE encounter_id=?', [self.row_indices[row]])
        row_data = list(self.db.result())
        encounter_id = str(row_data[0])
        encounter_name = str(row_data[1])
        self.tm._change_encounter(encounter_id)
        return encounter_id, encounter_name

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.row_indices)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()

            self.db.execute('SELECT * FROM encounters WHERE encounter_id=?', [self.row_indices[row]])
            row_data = list(self.db.result())
            return str(row_data[1])
        else:
            return QtCore.QVariant()
    
    def setData(self, index, value, role=QtCore.Qt.EditRole):
        """ Sets the role data for the item at index to value. """
        if role == QtCore.Qt.EditRole:
            value = str(value.toString())
            if value != "":
                row = index.row()

                query = 'UPDATE encounters SET encounter_name=? WHERE encounter_id=?'
                self.db.execute(query, [value, self.row_indices[row]])

                self.emit(QtCore.SIGNAL("dataChanged()"))
                return True

        return False

    def flags(self, index):
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        

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

        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.resizeColumnsToContents()

class ListView(QtGui.QListView):
    
    def __init__(self, parent=None):
        QtGui.QListView.__init__(self, parent)

    def mouseDoubleClickEvent(self, event):
        index = self.selectedIndexes()[0]
        enc_id, enc_name = self.model()._change_encounter(index)
        self.parent()._addWidget(enc_id, enc_name)


class TabWidget(QtGui.QTabWidget):
    def __init__(self, tm, parent=None):
        QtGui.QTabWidget.__init__(self, parent)
        self.setTabsClosable(True)
        self.setMaximumSize (9999, 21)
        self._tb = self.tabBar()
        self._tb.setMovable(True)
        self.tabCloseRequested.connect(self.onClose)
        self.currentChanged.connect(self.onChange)
        self.setStyleSheet("border: none;");

        self.id_list = []
        self.tm = tm

    def onChange(self, index):
        self.tm._change_encounter(self.id_list[index])

    def onClose(self, index):
        if len(self.id_list) > 0:
            self.id_list.pop(index)
            self.removeTab(index)

    def addID(self, _id):
        self.id_list.append(_id)
        self.setCurrentIndex(len(self.id_list) - 1)


class DummyWidget(QtGui.QWidget):
    """ Test Main Window """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.vlayout = QtGui.QVBoxLayout(self)
        
        headers_name, headers_nice, db = create_databse()

        self._tm = TableModel_SQL(headers_name, headers_nice, db, parent=self)
        self._tv = TableView(self)
        self._tv.setModel(self._tm)

        self._lm = ListModel_SQL(db, self._tm, parent=self)
        self._lv = ListView(self)
        self._lv.setModel(self._lm)

        self._tw = TabWidget(self._tm) 
        self._addWidget(-1, "Database")

        self.vlayout.addWidget(self._tw) 
        self.vlayout.addWidget(self._tv)
        self.vlayout.addWidget(self._lv)

    def _addWidget(self, enc_id, enc_name):
        if enc_id not in self._tw.id_list:
            temp = QtGui.QWidget() 
            self._tw.addTab(temp, str(enc_id) + " - " + str(enc_name))
            self._tw.addID(enc_id)
        else:
            for i in range(len(self._tw.id_list)):
                if enc_id == self._tw.id_list[i]:
                    self._tw.setCurrentIndex(i)
                    break



if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    widget = DummyWidget()
    widget.show()
    widget.raise_()
    sys.exit(app.exec_())
