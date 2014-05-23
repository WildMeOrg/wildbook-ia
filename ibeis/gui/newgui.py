#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join
from guitool import APITableModel
from PyQt4 import  QtGui
from PyQt4.QtCore import Qt
import string
import random

from ibeis.control import IBEISControl
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
    db.schema('data', headers)

    col_name_list = [ column[0] for column in headers ]
    col_type_list = [ str ] * len(col_name_list)
    col_edit_list = [ False, True, True, True, True, True ]
    col_nice_list = [
        'ID',
        'Encounter ID',
        'TEST Float',
        'TEST Int',
        'TEST String 1',
        'TEST String 2',
    ]

    rows = 1 * (10 ** 4)
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

    return col_name_list, col_type_list, col_edit_list, col_nice_list, db


#############################
######## Data Models ########
#############################


class ImageModelSQL(APITableModel.APITableModel):
    def __init__(model, ibs, parent=None, *args):
        model.ibs = ibs
        model.encounter_id = None
        col_name_list = ['image_uid', 'image_original_name', 'image_width', 'image_height']
        col_type_list = [str, str, int, int]
        col_nice_list = ['Image ID', 'Original Name', 'Image Width', 'Image Height']
        col_edit_list = [ True, False, False, False]
        super(ImageModelSQL, model).__init__(col_name_list=col_name_list,
                                             col_type_list=col_type_list, col_nice_list=col_nice_list,
                                             col_edit_list=col_edit_list,
                                             col_getter_list=model._getter, col_setter_list=model._setter,
                                             row_index_callback=model._row_index_callback,
                                             paernt=parent)

    def _change_encounter(model, encounter_id):
        model.encounter_id = encounter_id
        model._update_rows()

    def _row_index_callback(model, col_sort_name):
        return model.ibs.get_valid_gids(eid=model.encounter_id, sort=col_sort_name)

    def _setter(model, column_name, row_id, value):
        if value != '':
            model.ibs.set_image_props([row_id], column_name, [value])
        return True

    def _getter(model, column_name, row_id):
        result_list = model.ibs.get_image_props([column_name], [row_id])
        if result_list[0] is None:
            model._update_rows()
        else:
            return result_list[0]


class ROIModelSQL(APITableModel.APITableModel):
    def __init__(model, ibs, parent=None, *args):
        model.ibs = ibs
        model.encounter_id = None
        col_name_list = ['roi_uid', 'image_uid', 'roi_xtl', 'roi_ytl', 'roi_width', 'roi_height']
        col_type_list = [str, str, int, int, int, int]
        col_nice_list = ['ROI ID', 'Image ID', 'X', 'Y', 'Width', 'Height']
        col_edit_list = [ True, False, False, False, False, False]
        super(ROIModelSQL, model).__init__(col_name_list=col_name_list,
                                             col_type_list=col_type_list, col_nice_list=col_nice_list,
                                             col_edit_list=col_edit_list,
                                             col_getter_list=model._getter, col_setter_list=model._setter,
                                             row_index_callback=model._row_index_callback,
                                             paernt=parent)

    def _change_encounter(model, encounter_id):
        model.encounter_id = encounter_id
        model._update_rows()

    def _row_index_callback(model, col_sort_name):
        return model.ibs.get_valid_rids(eid=model.encounter_id, sort=col_sort_name)

    def _setter(model, column_name, row_id, value):
        if value != '':
            model.ibs.set_roi_props([row_id], column_name, [value])
        return True

    def _getter(model, column_name, row_id):
        result_list = model.ibs.get_roi_props([column_name], [row_id])
        if result_list[0] is None:
            model._update_rows()
        else:
            return result_list[0]

class NameModelSQL(APITableModel.APITableModel):
    def __init__(model, col_name_list, col_type_list, col_edit_list, col_nice_list, db, parent=None, *args):
        model.db = db
        model.encounter_id = None
        super(NameModelSQL, model).__init__(col_name_list=col_name_list,
                                             col_type_list=col_type_list, col_nice_list=col_nice_list,
                                             col_edit_list=col_edit_list,
                                             col_getter_list=model._getter, col_setter_list=model._setter,
                                             row_index_callback=model._row_index_callback,
                                             paernt=parent)

    def _change_encounter(model, encounter_id):
        model.encounter_id = encounter_id
        model._update_rows()

    def _row_index_callback(model, col_sort_name):
        query = 'SELECT data_id FROM data WHERE (? IS "-1" OR encounter_id=?) ORDER BY ' + col_sort_name + ' ASC'
        model.db.execute(query, [model.encounter_id, model.encounter_id])
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


class EncounterModelSQL(APITableModel.APITableModel):
    def __init__(model, window, col_name_list, col_type_list, col_edit_list, db, parent=None, *args):
        model.db = db
        model.window = window
        super(EncounterModelSQL, model).__init__(col_name_list=col_name_list,
                                                 col_type_list=col_type_list,
                                                 col_getter_list=model._getter,
                                                 col_edit_list=col_edit_list,
                                                 col_setter_list=model._setter,
                                                 row_index_callback=model._row_index_callback,
                                                 parent=parent)

    def _get_encounter_id_name(model, qtindex):
        row, col = model._row_col(qtindex)
        encounter_id = model._get_row_id(row)
        encounter_name = model._get_cell(row, 0)
        return encounter_id, encounter_name

    def _row_index_callback(model, col_sort_name):
        if col_sort_name == 'num_images':
            col_sort_name = 'encounter_id'
        model.db.execute('SELECT encounter_id FROM encounters ORDER BY ' + col_sort_name + ' ASC', [])
        return [result for result in model.db.result_iter()]

    def _setter(model, column_name, row_id, value):
        if value != '':
            query = 'UPDATE encounters SET ' + column_name + '=? WHERE encounter_id=?'
            model.db.execute(query, [value, row_id])
            model.window._update_encounter_tab_name(row_id, value)
        return True

    def _getter(model, column_name, row_id):
        if column_name == 'num_images':
            return 0
        query = 'SELECT ' + column_name + ' FROM encounters WHERE encounter_id=?'
        model.db.execute(query, [row_id])
        result_list = list(model.db.result())
        return str(result_list[0])


#############################
######### Data Views ########
#############################


class ImageView(QtGui.QTableView):
    def __init__(view, window, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = window
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    def _change_encounter(view, encounter_id):
        view.model()._change_encounter(encounter_id)


class ROIView(QtGui.QTableView):
    def __init__(view, window, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = window
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    def _change_encounter(view, encounter_id):
        view.model()._change_encounter(encounter_id)


class NameView(QtGui.QTableView):
    def __init__(view, window, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = window
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    def _change_encounter(view, encounter_id):
        view.model()._change_encounter(encounter_id)


class EncounterView(QtGui.QTableView):
    def __init__(view, window, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = window
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setMaximumSize(250, 9999)
        #hh = view.horizontalHeader()
        #hh.setVisible(False)

    def mouseDoubleClickEvent(view, event):
        index = view.selectedIndexes()[0]
        encounter_id, encounter_name = view.model()._get_encounter_id_name(index)
        view.window._add_encounter_tab(encounter_id, encounter_name)


#############################
###### Window Widgets #######
#############################


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(widget, window, parent=None):
        QtGui.QTabWidget.__init__(widget, parent)
        widget.window = window
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

        widget.encounter_id_list = []
        widget._add_encounter_tab(None, 'Recognition Database')

    def _on_change(widget, index):
        if 0 <= index and index < len(widget.encounter_id_list):
            widget.window._change_encounter(widget.encounter_id_list[index])

    def _close_tab(widget, index):
        if widget.encounter_id_list[index] != None:
            widget.encounter_id_list.pop(index)
            widget.removeTab(index)

    def _add_encounter_tab(widget, encounter_id, encounter_name):
        if encounter_id not in widget.encounter_id_list:
            # tab_name = str(encounter_id) + ' - ' + str(encounter_name)
            tab_name = str(encounter_name)
            widget.addTab(QtGui.QWidget(), tab_name)

            widget.encounter_id_list.append(encounter_id)
            index = len(widget.encounter_id_list) - 1
        else:
            index = widget.encounter_id_list.index(encounter_id)

        widget.setCurrentIndex(index)
        widget._on_change(index)

    def _update_encounter_tab_name(widget, encounter_id, encounter_name):
        for index, _id in enumerate(widget.encounter_id_list):
            if encounter_id == _id:
                widget.setTabText(index, encounter_name)


class IBEISGuiWidget(QtGui.QWidget):
    def __init__(widget, parent=None):
        QtGui.QWidget.__init__(widget, parent)
        widget.vlayout = QtGui.QVBoxLayout(widget)
        # widget.hsplitter = QtGui.QSplitter(Qt.Horizontal, widget)
        widget.hsplitter = QtGui.QHBoxLayout(widget)
        
        # Open DB

        widget.ibs = IBEISControl.IBEISController(dbdir='~/Desktop/testdb1')
        print(widget.ibs.get_valid_gids())
        col_name_list, col_type_list, col_edit_list, col_nice_list, db = create_databse()

        # Tabes Tab
        widget._tab_table_widget = QtGui.QTabWidget(widget)
        
        # Images Table
        widget._image_model = ImageModelSQL(widget.ibs, parent=widget)
        widget._image_view = ImageView(widget)
        widget._image_view.setModel(widget._image_model)

        widget._roi_model = ROIModelSQL(widget.ibs, parent=widget)
        widget._roi_view = ROIView(widget)
        widget._roi_view.setModel(widget._roi_model)

        widget._name_model = NameModelSQL(col_name_list, col_type_list, col_edit_list, col_nice_list, db, parent=widget)
        widget._name_view = NameView(widget)
        widget._name_view.setModel(widget._name_model)

        # Add Tabes to Tables Tab
        widget._tab_table_widget.addTab(widget._image_view, 'Images')
        widget._tab_table_widget.addTab(widget._roi_view, 'ROIs')
        widget._tab_table_widget.addTab(widget._name_view, 'Names')
        
        # Encounter List
        col_name_list = ['encounter_name', 'num_images']
        col_type_list = [str, int]
        col_edit_list = [True, False]
        widget._encounter_model = EncounterModelSQL(widget, col_name_list, col_type_list, col_edit_list, db, parent=widget)
        widget._encounter_view = EncounterView(widget)
        widget._encounter_view.setModel(widget._encounter_model)
        
        # Encounters Tabs
        widget._tab_encounter_widget = EncoutnerTabWidget(widget)
        
        # Add Other elements to the view
        widget.vlayout.addWidget(widget._tab_encounter_widget)
        widget.hsplitter.addWidget(widget._encounter_view)
        widget.hsplitter.addWidget(widget._tab_table_widget)
        # widget.vlayout.addWidget(widget.hsplitter)
        widget.vlayout.addLayout(widget.hsplitter)

    def _change_encounter(widget, encounter_id):
        widget._image_view._change_encounter(encounter_id)
        widget._roi_view._change_encounter(encounter_id)
        widget._name_view._change_encounter(encounter_id)

    def _add_encounter_tab(widget, encounter_id, encounter_name):
        widget._tab_encounter_widget._add_encounter_tab(encounter_id, encounter_name)

    def _update_encounter_tab_name(widget, encounter_id, encounter_name):
        widget._tab_encounter_widget._update_encounter_tab_name(encounter_id, encounter_name)


if __name__ == '__main__':
    import sys
    import signal
    import guitool
    def _on_ctrl_c(signal, frame):
        print('Caught ctrl+c')
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_ctrl_c)
    app = QtGui.QApplication(sys.argv)
    widget = IBEISGuiWidget()
    widget.show()
    widget.timer = guitool.ping_python_interpreter()
    widget.raise_()
    sys.exit(app.exec_())
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior
