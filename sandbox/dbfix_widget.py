# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import guitool as gt
from guitool.__PYQT__ import QtCore, QtGui, QtWidgets  # NOQA
from guitool.__PYQT__.QtCore import Qt
(print, rrr, profile) = ut.inject2(__name__, '[back]')


def populate_table(table, headers, data):
    table.setColumnCount(len(headers))
    print('Populating table')
    for n, key in ut.ProgIter(enumerate(headers), lbl='pop table'):
        if n == 0:
            # Set row count on first iteration
            table.setRowCount(len(data[key]))
        for m, item in enumerate(data[key]):
            newitem = QtWidgets.QTableWidgetItem(str(item))
            table.setItem(m, n, newitem)
            newitem.setFlags(newitem.flags() ^ Qt.ItemIsEditable)
    table.setHorizontalHeaderLabels(headers)
    table.resizeColumnsToContents()
    table.resizeRowsToContents()


class FixDatabaseWidget(gt.GuitoolWidget):
    """
    CommandLine:
        python -m ibeis.gui.dbfix_widget FixDatabaseWidget --show
        python -m ibeis.gui.dbfix_widget FixDatabaseWidget --show --db PZ_MTEST
        python -m ibeis.gui.dbfix_widget FixDatabaseWidget --show --debugwidget

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.dbfix_widget import *  # NOQA
        >>> import ibeis
        >>> gt.ensure_qtapp()
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> self = FixDatabaseWidget(ibs)
        >>> rect = gt.QtWidgets.QDesktopWidget().availableGeometry(screen=0)
        >>> self.move(rect.x(), rect.y())
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """
    def __init__(self, ibs):
        #from ibeis.expt import annotation_configs
        #from guitool import PrefWidget2
        #from guitool.__PYQT__.QtCore import Qt
        super(FixDatabaseWidget, self).__init__()
        self.ibs = ibs
        self.setWindowTitle('Database Fixer')

        #cfg_size_policy = (QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        tabwgt = self.addNewTabWidget(verticalStretch=1)
        #tab1 = tabwgt.addNewTab('Tab1')
        tab2 = tabwgt.addNewTab('Tab2')

        self.dirsize_table = QtWidgets.QTableWidget()
        self.dirsize_table.doubleClicked.connect(self.on_table_doubleclick)
        tab2.addWidget(self.dirsize_table)

        self.populate_dirsize_table()

        button_bar = self.addNewHWidget()
        #button_bar.addNewButton('TestLog', pressed=self.log_query)
        button_bar.addNewButton('Embed', pressed=self.embed)

    def populate_dirsize_table(self):
        #data = {'col1': ['1','2','3'], 'col2':['4','5','6'], 'col3':['7','8','9']}
        print('Updating saved query table')
        headers = ['dpath', 'size']
        table = self.dirsize_table

        ibs = self.ibs

        from os.path import isdir, dirname, islink
        if False:
            path_list = ut.glob(ibs.dbdir, '*', recursive=True)

            dirs = ut.defaultdict(list)
            fpath_list = []
            for path in path_list:
                if isdir(path):
                    dirs[path] = []
                if islink(path):
                    pass
                else:
                    dirs[dirname(path)].append(path)
                    fpath_list.append(path)

            nbytes_list = [ut.get_file_nBytes(fpath) for fpath in ut.ProgIter(fpath_list)]
            db_byte_str = ut.byte_str2(sum(nbytes_list))
            print('db_byte_str = %r' % (db_byte_str,))

        interest_dirs = [
            ibs.imgdir,
            ibs.chipdir,
            ibs.get_neuralnet_dir(),
            ibs.backupdir,
            ibs.cachedir,
            ibs.dbdir
        ]
        data = {'dpath': [], 'size': []}

        for dpath in interest_dirs:
            fpath_list = ut.glob(dpath, '*', recursive=True, with_dirs=False)
            nbytes_list = [ut.get_file_nBytes(fpath) for fpath in ut.ProgIter(fpath_list)
                           if not islink(fpath)]
            db_byte_str = ut.byte_str2(sum(nbytes_list))
            print('db_byte_str = %r' % (db_byte_str,))
            data['dpath'].append(dpath)
            data['size'].append(db_byte_str)

        self.data = data

        populate_table(table, headers, data)
        print('Finished populating table')

    def on_table_doubleclick(self, index):
        row = index.row()
        print('row = %r' % (row,))
        dpath = self.data['dpath'][row]
        print('dpath = %r' % (dpath,))

        ut.vd(dpath)

        # For linux
        if ut.is_developer():
            disk_usage_analyzer = 'baobab'
            ut.cmd(disk_usage_analyzer + ' ' + dpath, detach=True)
        pass

    def sizeHint(self):
        return QtCore.QSize(900, 960)

    def embed(self):
        import utool
        utool.embed()

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.gui.dbfix_widget
        python -m ibeis.gui.dbfix_widget --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
