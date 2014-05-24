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

import guiheaders as gh
from ibeis.control import IBEISControl
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui] ')


#############################
######## Data Models ########
#############################


class DataTablesModel(APITableModel.APITableModel):
    def __init__(model, headers, parent=None, *args):
        model.encounter_id = None
        model.headers=headers
        super(DataTablesModel, model).__init__(col_name_list=gh.header_names(headers),
                                             col_type_list=gh.header_types(headers), 
                                             col_nice_list=gh.header_nices(headers),
                                             col_edit_list=gh.header_edits(headers),
                                             col_getter_list=model._getter, 
                                             col_setter_list=model._setter,
                                             row_index_callback=model._row_index_callback,
                                             paernt=parent)

    def _change_enc(model, encounter_id):
        model.encounter_id = encounter_id
        model._update_rows()

    def _row_index_callback(model, col_sort_name):
        gids = gh.header_ids(model.headers)(eid=model.encounter_id)
        values = gh.getter_from_name(model.headers, col_sort_name)(gids)
        values = zip(values, gids)
        return [ tup[1] for tup in sorted(values) ]

    def _getter(model, column_name, row_id):
        result_list = gh.getter_from_name(model.headers, column_name)([row_id])
        if result_list[0] is None:
            model._update_rows()
        else:
            return result_list[0]

    def _setter(model, column_name, row_id, value):
        if value != '':
            gh.setter_from_name(model.headers, column_name)([row_id], [value])
        return True


class EncModel(APITableModel.APITableModel):
    def __init__(model, headers, parent=None, *args):
        model.window = parent
        model.headers=headers
        super(EncModel, model).__init__(col_name_list=gh.header_names(headers),
                                             col_type_list=gh.header_types(headers), 
                                             col_nice_list=gh.header_nices(headers),
                                             col_edit_list=gh.header_edits(headers),
                                             col_getter_list=model._getter, 
                                             col_setter_list=model._setter,
                                             row_index_callback=model._row_index_callback,
                                             paernt=parent)
    
    def _get_enc_id_name(model, qtindex):
        row, col = model._row_col(qtindex)
        tab_display_name_column_index = 0
        column_name = gh.header_names(model.headers)[tab_display_name_column_index]
        encounter_id = model._get_row_id(row)
        encounter_name = gh.getter_from_name(model.headers, column_name)([encounter_id])[0]
        return encounter_id, encounter_name

    def _row_index_callback(model, col_sort_name):
        gids = gh.header_ids(model.headers)()
        values = gh.getter_from_name(model.headers, col_sort_name)(gids)
        values = zip(values, gids)
        return [ tup[1] for tup in sorted(values) ]

    def _getter(model, column_name, row_id):
        result_list = gh.getter_from_name(model.headers, column_name)([row_id])
        if result_list[0] is None:
            model._update_rows()
        else:
            return result_list[0]

    def _setter(model, column_name, row_id, value):
        if value != '':
            gh.setter_from_name(model.headers, column_name)([row_id], [value])
            model.window._update_enc_tab_name(row_id, value)
        return True


#############################
######### Data Views ########
#############################


class DataTablesView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    def _change_enc(view, encounter_id):
        view.model()._change_enc(encounter_id)


class EncView(QtGui.QTableView):
    def __init__(view, parent=None):
        QtGui.QTableView.__init__(view, parent)
        view.window = parent
        view.setSortingEnabled(True)
        vh = view.verticalHeader()
        vh.setVisible(False)
        view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        view.setMaximumSize(200, 9999)
        #hh = view.horizontalHeader()
        #hh.setVisible(False)

    def mouseDoubleClickEvent(view, event):
        qtindex = view.selectedIndexes()[0]
        encounter_id, encounter_name = view.model()._get_enc_id_name(qtindex)
        view.window._add_enc_tab(encounter_id, encounter_name)


#############################
###### Window Widgets #######
#############################


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(wgt, parent=None):
        QtGui.QTabWidget.__init__(wgt, parent)
        wgt.window = parent
        wgt.setTabsClosable(True)
        if sys.platform.startswith('darwin'):
            tab_height = 21
        else:
            tab_height = 30
        wgt.setMaximumSize(9999, tab_height)
        wgt._tb = wgt.tabBar()
        wgt._tb.setMovable(True)
        wgt.setStyleSheet('border: none;')
        wgt._tb.setStyleSheet('border: none;')

        wgt.tabCloseRequested.connect(wgt._close_tab)
        wgt.currentChanged.connect(wgt._on_change)

        wgt.encounter_id_list = []
        wgt._add_enc_tab(None, 'Recognition Database')

    def _on_change(wgt, index):
        if 0 <= index and index < len(wgt.encounter_id_list):
            wgt.window._change_enc(wgt.encounter_id_list[index])

    def _close_tab(wgt, index):
        if wgt.encounter_id_list[index] != None:
            wgt.encounter_id_list.pop(index)
            wgt.removeTab(index)

    def _add_enc_tab(wgt, encounter_id, encounter_name):
        if encounter_id not in wgt.encounter_id_list:
            # tab_name = str(encounter_id) + ' - ' + str(encounter_name)
            tab_name = str(encounter_name)
            wgt.addTab(QtGui.QWidget(), tab_name)

            wgt.encounter_id_list.append(encounter_id)
            index = len(wgt.encounter_id_list) - 1
        else:
            index = wgt.encounter_id_list.index(encounter_id)

        wgt.setCurrentIndex(index)
        wgt._on_change(index)

    def _update_enc_tab_name(wgt, encounter_id, encounter_name):
        for index, _id in enumerate(wgt.encounter_id_list):
            if encounter_id == _id:
                wgt.setTabText(index, encounter_name)


class IBEISGuiWidget(QtGui.QWidget):
    def __init__(wgt, parent=None):
        QtGui.QWidget.__init__(wgt, parent)
        wgt.vlayout = QtGui.QVBoxLayout(wgt)
        # wgt.hsplitter = QtGui.QSplitter(Qt.Horizontal, wgt)
        wgt.hsplitter = QtGui.QHBoxLayout(wgt)
        
        # Open DB
        # wgt.ibs = IBEISControl.IBEISController(dbdir='~/Desktop/testdb1')
        wgt.ibs = IBEISControl.IBEISController(dbdir='~/Desktop/GZ_Siva')
        wgt.headers = gh.gui_headers(wgt.ibs)
        # Tabes Tab
        wgt._tab_table_wgt = QtGui.QTabWidget(wgt)
        
        # Images Table
        wgt._image_model = DataTablesModel(wgt.headers['images'], parent=wgt)
        wgt._image_view = DataTablesView(parent=wgt)
        wgt._image_view.setModel(wgt._image_model)

        wgt._roi_model = DataTablesModel(wgt.headers['rois'], parent=wgt)
        wgt._roi_view = DataTablesView(parent=wgt)
        wgt._roi_view.setModel(wgt._roi_model)

        wgt._name_model = DataTablesModel(wgt.headers['names'], parent=wgt)
        wgt._name_view = DataTablesView(parent=wgt)
        wgt._name_view.setModel(wgt._name_model)

        # Add Tabes to Tables Tab
        wgt._tab_table_wgt.addTab(wgt._image_view, 'Images')
        wgt._tab_table_wgt.addTab(wgt._roi_view, 'ROIs')
        wgt._tab_table_wgt.addTab(wgt._name_view, 'Names')
        
        # Enc List
        wgt._enc_model = EncModel(wgt.headers['encounters'], parent=wgt)
        wgt._enc_view = EncView(parent=wgt)
        wgt._enc_view.setModel(wgt._enc_model)
        
        # Encs Tabs
        wgt._tab_enc_wgt = EncoutnerTabWidget(parent=wgt)
        
        # Add Other elements to the view
        wgt.vlayout.addWidget(wgt._tab_enc_wgt)
        wgt.hsplitter.addWidget(wgt._enc_view)
        wgt.hsplitter.addWidget(wgt._tab_table_wgt)
        # wgt.vlayout.addWidget(wgt.hsplitter)
        wgt.vlayout.addLayout(wgt.hsplitter)

    def _change_enc(wgt, encounter_id):
        wgt._image_view._change_enc(encounter_id)
        wgt._roi_view._change_enc(encounter_id)
        wgt._name_view._change_enc(encounter_id)

    def _add_enc_tab(wgt, encounter_id, encounter_name):
        wgt._tab_enc_wgt._add_enc_tab(encounter_id, encounter_name)

    def _update_enc_tab_name(wgt, encounter_id, encounter_name):
        wgt._tab_enc_wgt._update_enc_tab_name(encounter_id, encounter_name)


if __name__ == '__main__':
    import sys
    import signal
    import guitool
    def _on_ctrl_c(signal, frame):
        print('Caught ctrl+c')
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_ctrl_c)
    app = QtGui.QApplication(sys.argv)
    wgt = IBEISGuiWidget()
    wgt.show()
    wgt.timer = guitool.ping_python_interpreter()
    wgt.raise_()
    sys.exit(app.exec_())
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior
