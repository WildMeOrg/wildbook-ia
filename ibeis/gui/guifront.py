#TODO: Licence
from __future__ import absolute_import, division, print_function
# Python
import functools
# Qt
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
# IBEIS
import utool
from guitool import slot_, signal_
from ibeis.gui import gui_item_tables as item_table
import ibeis.gui.frontend_helpers as fh
from ibeis.gui.Skeleton import Ui_mainSkel

QUIET   = utool.get_flag('--quiet')
VERBOSE = utool.get_flag('--verbose')

#=================
# Decorators / Helpers
#=================


def clicked(func):
    """ Decorator which adds the decorated function as a slot and passes
    the row and column of the item clicked instead of the item itself
    Also provies debugging prints
    """
    @slot_(QtGui.QTableWidgetItem)
    @functools.wraps(func)
    def clicked_wrapper(front, item):
        if front.isItemEditable(item):
            front.print('[front] does not select when clicking editable column')
            return
        if item == front.prev_tbl_item:
            return
        front.prev_tbl_item = item
        row, col = (item.row(), item.column())
        front.print('%s(%r, %r)' % (func.func_name, row, col))
        return func(front, row, col)
    # Hacky decorator
    return clicked_wrapper


def _tee_logging(front):
    if VERBOSE:
        print('[front] teeing log output')
    # Connect a StreamStealer object to the GUI output window
    #if __common__.__LOGGING__:
        #front.logging_handler = guitool.GUILoggingHandler(front.gui_write)
        #__common__.add_logging_handler(front.logging_handler)
    #else:
        #front.ostream = StreamStealer(front, share=not noshare)


#=================
# Initialization
#=================


def update_tabwidget_text(front, tblname, text):
    tablename2_tabwidget = {
        item_table.IMAGE_TABLE: front.ui.image_view,
        item_table.ROI_TABLE: front.ui.roi_view,
        item_table.NAME_TABLE: front.ui.name_view,
        item_table.RES_TABLE: front.ui.qres_view,
    }
    ui = front.ui
    tab_widget = tablename2_tabwidget[tblname]
    tab_index = ui.tablesTabWidget.indexOf(tab_widget)
    tab_text = QtGui.QApplication.translate(front.objectName(), text, None,
                                            QtGui.QApplication.UnicodeUTF8)
    ui.tablesTabWidget.setTabText(tab_index, tab_text)


QT_UUID_TYPE = item_table.QT_UUID_TYPE


class MainWindowFrontend(QtGui.QMainWindow):
    printSignal      = signal_(str)
    quitSignal       = signal_()
    selectGidSignal  = signal_(QT_UUID_TYPE)
    selectRidSignal  = signal_(QT_UUID_TYPE)
    selectNidSignal  = signal_(QT_UUID_TYPE)
    selectQResSignal = signal_(QT_UUID_TYPE)
    setRoiPropSignal = signal_(QT_UUID_TYPE, str, str)
    aliasNidSignal   = signal_(QT_UUID_TYPE, str, str)
    setGidPropSignal = signal_(QT_UUID_TYPE, str, QtCore.QVariant)
    querySignal      = signal_()

    def __init__(front, back):
        super(MainWindowFrontend, front).__init__()
        #print('[*front] creating frontend')
        front.prev_tbl_item = None
        front.logging_handler = None
        front.back = back
        front.ui = Ui_mainSkel()
        front.ui.setupUi(front)
        # Programatially Defined Actions
        #newMenuAction(front, 'menuHelp', 'actionDetect_Duplicate_Images',
        #text='Detect Duplicate Images', slot_fn=back.detect_dupimg)
        fh.newMenuAction(front, 'menuActions', 'Query2', slot_fn=back.query)
        fh.newMenuAction(front, 'menuBatch', 'Detect Grevys', slot_fn=back.detect_grevys)
        # Progress bar is not hooked up yet
        front.ui.progressBar.setVisible(False)
        front.connect_signals()
        _tee_logging(front)

    @slot_()
    def closeEvent(front, event):
        event.accept()
        front.quitSignal.emit()

    def connect_signals(front):
        # Connect signals to slots
        back = front.back
        ui = front.ui
        # Frontend Signals
        front.printSignal.connect(back.backend_print)
        front.quitSignal.connect(back.quit)
        front.selectGidSignal.connect(back.select_gid)
        front.selectRidSignal.connect(back.select_rid)
        front.selectNidSignal.connect(back.select_nid)
        front.selectQResSignal.connect(back.select_qres_rid)
        front.setRoiPropSignal.connect(back.set_roi_prop)
        front.aliasNidSignal.connect(back.set_name_prop)
        front.setGidPropSignal.connect(back.set_image_prop)
        front.querySignal.connect(back.query)

        # Connect all signals from GUI
        for func in front.ui.connect_fns:
            func()
        #
        # Gui Components
        # Tables Widgets
        ui.rids_TBL.itemClicked.connect(front.roi_tbl_clicked)
        ui.rids_TBL.itemChanged.connect(front.roi_tbl_changed)
        ui.gids_TBL.itemClicked.connect(front.img_tbl_clicked)
        ui.gids_TBL.itemChanged.connect(front.img_tbl_changed)
        ui.qres_TBL.itemClicked.connect(front.qres_tbl_clicked)
        ui.qres_TBL.itemChanged.connect(front.qres_tbl_changed)
        ui.nids_TBL.itemClicked.connect(front.name_tbl_clicked)
        ui.nids_TBL.itemChanged.connect(front.name_tbl_changed)
        # Tab Widget
        ui.tablesTabWidget.currentChanged.connect(front.change_view)
        ui.rids_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.qres_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.gids_TBL.sortByColumn(0, Qt.AscendingOrder)

    def print(front, msg):
        if VERBOSE:
            print('[*front*] ' + msg)

    @slot_(bool)
    def setEnabled(front, flag):
        ui = front.ui
        # Enable or disable all actions
        for uikey in ui.__dict__.keys():
            if uikey.find('action') == 0:
                ui.__dict__[uikey].setEnabled(flag)

        # The following options are always enabled
        ui.actionOpen_Database.setEnabled(True)
        ui.actionNew_Database.setEnabled(True)
        ui.actionQuit.setEnabled(True)
        ui.actionAbout.setEnabled(True)
        ui.actionView_Docs.setEnabled(True)
        ui.actionDelete_global_preferences.setEnabled(True)

        # The following options are no implemented. Disable them
        ui.actionConvert_all_images_into_chips.setEnabled(False)
        ui.actionBatch_Change_Name.setEnabled(False)
        ui.actionScale_all_ROIS.setEnabled(False)
        ui.actionWriteLogs.setEnabled(False)
        ui.actionAbout.setEnabled(False)

    @slot_(str, list, list, list, list)
    def populate_tbl(front, tblname, col_fancyheaders, col_editable,
                     row_list, datatup_list):
        tblname = str(tblname)
        front.print('populate_tbl(%s)' % tblname)
        tbl = {
            item_table.IMAGE_TABLE: front.ui.gids_TBL,
            item_table.ROI_TABLE:   front.ui.rids_TBL,
            item_table.NAME_TABLE:  front.ui.nids_TBL,
            item_table.RES_TABLE:   front.ui.qres_TBL,
        }[tblname]
        item_table.populate_item_table(tbl, col_fancyheaders, col_editable, row_list, datatup_list)
        # Set the tab text to show the number of items listed
        fancy_tablename = item_table.fancy_tablenames[tblname]
        text = fancy_tablename + ' : %d' % len(row_list)
        update_tabwidget_text(front, tblname, text)

    def isItemEditable(self, item):
        return int(Qt.ItemIsEditable & item.flags()) == int(Qt.ItemIsEditable)

    #=======================
    # General Table Getters
    #=======================

    def get_tbl_header(front, tbl, col):
        # Map the fancy header back to the internal one.
        fancy_header = str(tbl.horizontalHeaderItem(col).text())
        header = (item_table.reverse_fancy[fancy_header]
                  if fancy_header in item_table.reverse_fancy else fancy_header)
        return header

    def get_tbl_int(front, tbl, row, col):
        return int(tbl.item(row, col).text())

    def get_tbl_str(front, tbl, row, col):
        return str(tbl.item(row, col).text())

    def get_header_val(front, tbl, header, row):
        # RCOS TODO: This is hacky. These just need to be
        # in dicts to begin with.
        tblname = str(tbl.objectName()).replace('_TBL', '')
        tblname = item_table.sqltable_names[tblname]
        col = item_table.table_headers[tblname].index(header)
        return tbl.item(row, col).text()

    #=======================
    # Specific Item Getters
    #=======================

    def get_roitbl_header(front, col):
        return front.get_tbl_header(front.ui.rids_TBL, col)

    def get_imgtbl_header(front, col):
        return front.get_tbl_header(front.ui.gids_TBL, col)

    def get_qrestbl_header(front, col):
        return front.get_tbl_header(front.ui.qres_TBL, col)

    def get_nametbl_header(front, col):
        return front.get_tbl_header(front.ui.nids_TBL, col)

    def get_qrestbl_rid(front, row):
        return QT_UUID_TYPE(front.get_header_val(front.ui.qres_TBL, 'rid', row))

    def get_roitbl_rid(front, row):
        return QT_UUID_TYPE(front.get_header_val(front.ui.rids_TBL, 'rid', row))

    def get_nametbl_name(front, row):
        return str(front.get_header_val(front.ui.nids_TBL, 'name', row))

    def get_nametbl_nid(front, row):
        return QT_UUID_TYPE(front.get_header_val(front.ui.nids_TBL, 'nid', row))

    def get_imgtbl_gid(front, row):
        return QT_UUID_TYPE(front.get_header_val(front.ui.gids_TBL, 'gid', row))

    #=======================
    # Table Changed Functions
    #=======================

    @slot_(QtGui.QTableWidgetItem)
    def img_tbl_changed(front, item):
        front.print('img_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_gid = front.get_imgtbl_gid(row)
        header_lbl = front.get_imgtbl_header(col)
        if header_lbl == 'aif':
            new_val = item.checkState() == Qt.Checked
        else:
            new_val = item.text()
        front.setGidPropSignal.emit(sel_gid, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def roi_tbl_changed(front, item):
        front.print('roi_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_rid = front.get_roitbl_rid(row)  # Get selected roiid
        new_val = item.text()
        header_lbl = front.get_roitbl_header(col)  # Get changed column
        front.setRoiPropSignal.emit(sel_rid, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def qres_tbl_changed(front, item):
        front.print('qres_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_rid  = front.get_qrestbl_rid(row)  # The changed row's roi id
        new_val  = item.text()
        header_lbl = front.get_qrestbl_header(col)  # Get changed column
        front.setRoiPropSignal.emit(sel_rid, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def name_tbl_changed(front, item):
        front.print('name_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_nid = front.get_nametbl_nid(row)    # The changed row's name index
        new_val  = item.text()
        header_lbl = front.get_nametbl_header(col)  # Get changed column
        front.aliasNidSignal.emit(sel_nid, header_lbl, new_val)

    #=======================
    # Table Clicked Functions
    #=======================
    @clicked
    def img_tbl_clicked(front, row, col):
        sel_gid = front.get_imgtbl_gid(row)
        front.selectGidSignal.emit(sel_gid)

    @clicked
    def roi_tbl_clicked(front, row, col):
        sel_rid = front.get_roitbl_rid(row)
        front.selectRidSignal.emit(sel_rid)

    @clicked
    def qres_tbl_clicked(front, row, col):
        sel_rid = front.get_qrestbl_rid(row)
        front.selectQResSignal.emit(sel_rid)

    @clicked
    def name_tbl_clicked(front, row, col):
        sel_nid = front.get_nametbl_nid(row)
        front.selectNidSignal.emit(sel_nid)

    #=======================
    # Other
    #=======================

    @slot_(int)
    def change_view(front, new_state):
        tab_name = str(front.ui.tablesTabWidget.tabText(new_state))
        front.print('change_view(%r)' % new_state)
        prevBlock = front.ui.tablesTabWidget.blockSignals(True)
        front.ui.tablesTabWidget.blockSignals(prevBlock)
        if tab_name.startswith('Query Results Table'):
            print(front.back.cfg)

    @slot_(str)
    def gui_write(front, msg_):
        # Slot for teed log output
        app = front.back.app
        outputEdit = front.ui.outputEdit
        # Write msg to text area
        outputEdit.moveCursor(QtGui.QTextCursor.End)
        # TODO: Find out how to do backspaces in textEdit
        msg = str(msg_)
        if msg.find('\b') != -1:
            msg = msg.replace('\b', '') + '\n'
        outputEdit.insertPlainText(msg)
        if app is not None:
            app.processEvents()

    @slot_()
    def gui_flush(front):
        app = front.back.app
        if app is not None:
            app.processEvents()
