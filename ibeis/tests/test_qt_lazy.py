
#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import numpy as np
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

    sqldb_fname = 'temp_test_sql_numpy.sqlite3'
    sqldb_dpath = utool.util_cplat.get_app_resource_dir('ibeis', 'testfiles')
    utool.ensuredir(sqldb_dpath)
    utool.util_path.remove_file(join(sqldb_dpath, sqldb_fname), dryrun=False)
    db = SQLDatabaseControl.SQLDatabaseController(sqldb_dpath=sqldb_dpath,
                                                  sqldb_fname=sqldb_fname)
    
    headers = [
        ('temp_id',      'INTEGER PRIMARY KEY'),
        ('temp_float',   'FLOAT'),
        ('temp_int',     'INT'),
        ('temp_text',    'TEXT'),
        ('temp_text2',   'TEXT'),
    ]
    headers_name = [ column[0] for column in headers ]
    headers_nice = [
        'ID',
        'TEST Float',
        'TEST Int',
        'TEST String 1',
        'TEST String 2',
    ]
    db.schema('temp', headers)

    rows = 1*(10**5)
    feats_iter = ( (random.uniform(0.0,1.0), random.randint(0,100), _randstr(), _randstr()) 
                        for i in xrange(rows) )
    
    print('[TEST] insert numpy arrays')
    tt = utool.tic()
    db.executemany(operation='''
        INSERT
        INTO temp
        (
            temp_float,
            temp_int,
            temp_text,
            temp_text2
        )
        VALUES (?,?,?,?)
        ''', params_iter=feats_iter)
    print(' * execute insert time=%r sec' % utool.toc(tt))

    return headers_name, headers_nice, db


class TableModel(QtCore.QAbstractTableModel): 
    def __init__(self, headers_name, headers_nice, db, parent=None, *args): 
        super(TableModel, self).__init__()
        self.headers_name = headers_name
        self.headers_nice = headers_nice

        self.db = db
        self.db_sort_index = 0
        self.db_sort_reversed = False

        self._get_row_indices()

    def _get_row_indices(self):
        column = self.headers_name[self.db_sort_index]
        order = (' DESC' if self.db_sort_reversed else ' ASC')
        self.db.execute('SELECT temp_id FROM temp ORDER BY ' + column + order, [])
        self.row_indices = [result for result in self.db.result_iter()]

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.row_indices)
        
    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers_name) 
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            column = index.column()

            self.db.execute('SELECT * FROM temp WHERE temp_id=?', [self.row_indices[row]])
            row_data = list(self.db.result())
            self.cache_row_data = row_data[:]
            self.cache_row = row

            return str(row_data[column])
        else:
            return QtCore.QVariant()

    def headerData(self, index, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.headers_nice[index]
        else:
            return QtCore.QVariant()
    
    def sort(self, index, order):
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self.db_sort_index = index
        self.db_sort_reversed = order == QtCore.Qt.DescendingOrder
        self._get_row_indices()
        self.emit(QtCore.SIGNAL("layoutChanged()"))

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled
            
 
class TableView(QtGui.QTableView):
    
    def __init__(self, *args, **kwargs):
        QtGui.QTableView.__init__(self, *args, **kwargs)
        self.setSortingEnabled(True)

        vh = self.verticalHeader()
        vh.setVisible(False)

        self.resizeColumnsToContents()



if __name__=='__main__':
    from sys import argv, exit
    
    class Widget(QtGui.QWidget):
        
        def __init__(self, parent=None):
            QtGui.QWidget.__init__(self, parent)
            
            headers_name, headers_nice, db = create_databse()

            l=QtGui.QVBoxLayout(self)
            self._tm=TableModel(headers_name, headers_nice, db, parent=self)
            self._tv=TableView(self)
            self._tv.setModel(self._tm)
            l.addWidget(self._tv)
            
 
    a=QtGui.QApplication(argv)
    w=Widget()
    w.show()
    w.raise_()
    exit(a.exec_())