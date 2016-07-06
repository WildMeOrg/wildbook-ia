#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from six.moves import range
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join
from guitool.api_item_model import APIItemModel
from guitool.__PYQT__ import QtGui
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

    imagesets = [
        ('imageset_id',      'INTEGER PRIMARY KEY'),
        ('imageset_name',    'TEXT'),
    ]
    db.add_table('imagesets', imagesets)

    rows = 1 * (10 ** 3)
    feats_iter = ( (_randstr(), ) for i in range(rows) )

    print('[TEST] insert imagesets')
    tt = utool.tic()
    db.executemany(operation='''
        INSERT
        INTO imagesets
        (
            imageset_name
        )
        VALUES (?)
        ''', params_iter=feats_iter)
    print(' * execute insert time=%r sec' % utool.toc(tt))

    ##############################################
    headers = [
        ('data_id',      'INTEGER PRIMARY KEY'),
        ('imageset_id', 'INT'),
        ('data_float',   'FLOAT'),
        ('data_int',     'INT'),
        ('data_text',    'TEXT'),
        ('data_text2',   'TEXT'),
    ]
    db.add_table('data', headers)

    col_name_list = [ column[0] for column in headers ]
    col_type_list = [ str ] * len(col_name_list)
    col_edit_list = [ False, True, True, True, True, True ]
    col_nice_list = [
        'ID',
        'ImageSet ID',
        'TEST Float',
        'TEST Int',
        'TEST String 1',
        'TEST String 2',
    ]

    rows = 1 * (10 ** 4)
    feats_iter = ((random.randint(0, 1000), random.uniform(0.0, 1.0), random.randint(0, 100), _randstr(), _randstr())
                        for i in range(rows) )

    print('[TEST] insert data')
    tt = utool.tic()
    db.executemany(operation='''
        INSERT
        INTO data
        (
            imageset_id,
            data_float,
            data_int,
            data_text,
            data_text2
        )
        VALUES (?,?,?,?,?)
        ''', params_iter=feats_iter)
    print(' * execute insert time=%r sec' % utool.toc(tt))

    return col_name_list, col_type_list, col_edit_list, col_nice_list, db


class ImageModelSQL(APIItemModel):
    def __init__(model, col_name_list, col_type_list, col_edit_list, col_nice_list, db, parent=None, *args):
        model.db = db
        model.imageset_id = '-1'
        #row_index_callback=model._row_index_callback
        headers = dict(col_name_list=col_name_list,
                       col_type_list=col_type_list,
                       col_nice_list=col_nice_list,
                       col_edit_list=col_edit_list,
                       #col_getter_list=model._getter,
                       #col_setter_list=model._setter
                       )
        super(ImageModelSQL, model).__init__(headers, parent)

    def _change_imageset(model, imageset_id):
        model.imageset_id = imageset_id
        model._update_rows()

    def _row_index_callback(model, col_sort_name):
        query = 'SELECT data_id FROM data WHERE (? IS "-1" OR imageset_id=?) ORDER BY ' + col_sort_name + ' ASC'
        model.db.execute(query, [model.imageset_id, model.imageset_id])
        return [result for result in model.db.result_iter()]

    def _setter(model, column_name, row_id, value):
        if value != '':
            query = 'UPDATE data SET ' + column_name + '=? WHERE data_id=?'
            model.db.execute(query, [value, row_id])
        return True

    def _getter(model, column_name, row_id):
        query = 'SELECT ' + column_name + ' FROM data WHERE data_id=?'
        model.db.execute(query, [row_id])
        result_list = list(model.db.result())
        return str(result_list[0])


class ImageSetModelSQL(APIItemModel):
    def __init__(model, col_name_list, col_type_list, col_edit_list, db, parent=None, *args):
        model.db = db
        super(ImageSetModelSQL, model).__init__(col_name_list=col_name_list,
                                                 col_type_list=col_type_list,
                                                 col_getter_list=model._getter,
                                                 col_edit_list=col_edit_list,
                                                 col_setter_list=model._setter,
                                                 row_index_callback=model._row_index_callback,
                                                 parent=parent)

    def _get_imageset_id_name(model, qtindex):
        row, col = model._row_col(qtindex)
        imageset_id = model._get_row_id(row)
        imageset_name = model._get_cell(row, 0)
        return imageset_id, imageset_name

    def _row_index_callback(model, col_sort_name):
        if col_sort_name == 'num_images':
            col_sort_name = 'imageset_id'
        model.db.execute('SELECT imageset_id FROM imagesets ORDER BY ' + col_sort_name + ' ASC', [])
        return [result for result in model.db.result_iter()]

    def _setter(model, column_name, row_id, value):
        if value != '':
            query = 'UPDATE imagesets SET ' + column_name + '=? WHERE imageset_id=?'
            model.db.execute(query, [value, row_id])
            # model.parent()._update_imageset_tab_name(row_id, value)
        return True

    def _getter(model, column_name, row_id):
        if column_name == 'num_images':
            return 0
        query = 'SELECT ' + column_name + ' FROM imagesets WHERE imageset_id=?'
        model.db.execute(query, [row_id])
        result_list = list(model.db.result())
        return str(result_list[0])


class ImageView(QtWidgets.QTableView):
    def __init__(view, parent=None):
        QtWidgets.QTableView.__init__(view, parent)
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    def _change_imageset(view, imageset_id):
        view.model()._change_imageset(imageset_id)


class ImageSetView(QtWidgets.QTableView):
    def __init__(view, parent=None):
        QtWidgets.QTableView.__init__(view, parent)
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        #hh = view.horizontalHeader()
        #hh.setVisible(False)

    def mouseDoubleClickEvent(view, event):
        index = view.selectedIndexes()[0]
        imageset_id, imageset_name = view.model()._get_imageset_id_name(index)
        view.parent()._add_imageset_tab(imageset_id, imageset_name)


class TabWidget(QtWidgets.QTabWidget):
    def __init__(widget, parent=None):
        QtWidgets.QTabWidget.__init__(widget, parent)
        widget.setTabsClosable(True)
        if sys.platform.startswith('darwin'):
            tab_height = 21
        else:
            tab_height = 30
        widget.setMaximumSize(9999, tab_height)
        widget._tb = widget.tabBar()
        widget._tb.setMovable(True)
        widget.setStyleSheet('border: none;')
        widget._tb.setStyleSheet('border: none;')

        widget.tabCloseRequested.connect(widget._close_tab)
        widget.currentChanged.connect(widget._on_change)

        widget.imageset_id_list = []
        widget._add_imageset_tab('-1', 'Database')

    def _on_change(widget, index):
        if 0 <= index and index < len(widget.imageset_id_list):
            widget.parent()._change_imageset(widget.imageset_id_list[index])

    def _close_tab(widget, index):
        if widget.imageset_id_list[index] != '-1':
            widget.imageset_id_list.pop(index)
            widget.removeTab(index)

    def _add_imageset_tab(widget, imageset_id, imageset_name):
        if imageset_id not in widget.imageset_id_list:
            tab_name = str(imageset_id) + ' - ' + str(imageset_name)
            widget.addTab(QtWidgets.QWidget(), tab_name)

            widget.imageset_id_list.append(imageset_id)
            index = len(widget.imageset_id_list) - 1
        else:
            index = widget.imageset_id_list.index(imageset_id)

        widget.setCurrentIndex(index)
        widget._on_change(index)

    def _update_imageset_tab_name(widget, imageset_id, imageset_name):
        for index, _id in enumerate(widget.imageset_id_list):
            if imageset_id == _id:
                widget.setTabText(index, imageset_name)


class DummyWidget(QtWidgets.QWidget):
    ''' Test Main Window '''
    def __init__(widget, parent=None):
        QtWidgets.QWidget.__init__(widget, parent)
        widget.vlayout = QtWidgets.QVBoxLayout(widget)

        col_name_list, col_type_list, col_edit_list, col_nice_list, db = create_databse()
        widget._image_model = ImageModelSQL(col_name_list, col_type_list, col_edit_list, col_nice_list, db, parent=widget)
        widget._image_view = ImageView(parent=widget)
        widget._image_view.setModel(widget._image_model)

        col_name_list = ['imageset_name', 'num_images']
        col_type_list = [str, int]
        col_edit_list = [True, False]

        #splitter = QtWidgets.QSplitter(centralwidget)
        #splitter.setOrientation(QtCore.Qt.Vertical)

        widget._imageset_model = ImageSetModelSQL(col_name_list, col_type_list, col_edit_list, db, parent=widget)
        widget._imageset_view = ImageSetView(parent=widget)
        widget._imageset_view.setModel(widget._imageset_model)

        widget._tab_widget = TabWidget(parent=widget)

        widget.vlayout.addWidget(widget._tab_widget)
        widget.vlayout.addWidget(widget._image_view)
        widget.vlayout.addWidget(widget._imageset_view)

    def _change_imageset(widget, imageset_id):
        widget._image_view._change_imageset(imageset_id)

    def _add_imageset_tab(widget, imageset_id, imageset_name):
        widget._tab_widget._add_imageset_tab(imageset_id, imageset_name)

    def _update_imageset_tab_name(widget, imageset_id, imageset_name):
        widget._tab_widget._update_imageset_tab_name(imageset_id, imageset_name)


if __name__ == '__main__':
    import sys
    import signal
    import guitool
    def _on_ctrl_c(signal, frame):
        print('Caught ctrl+c')
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_ctrl_c)
    app = QtWidgets.QApplication(sys.argv)
    widget = DummyWidget()
    widget.show()
    widget.timer = guitool.ping_python_interpreter()
    widget.raise_()
    sys.exit(app.exec_())
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior
