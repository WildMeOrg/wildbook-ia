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
from ibeis.gui import uidtables
from ibeis.gui.Skeleton import Ui_mainSkel

QUIET   = utool.get_flag('--quiet')
VERBOSE = utool.get_flag('--verbose')


UUID_TYPE = uidtables.UUID_TYPE
QUTF8      = QtGui.QApplication.UnicodeUTF8
QTRANSLATE = QtGui.QApplication.translate

#=================
# Decorators / Helpers
#=================


def uid_tbl_clicked(func):
    """
    Wrapper around item_clicked slot, which takes only the item.
    This extract the row, column, and table and passes that instead.
    Also provies debugging prints
    """
    @slot_(QtGui.QTableWidgetItem)
    @functools.wraps(func)
    def clicked_wrapper(front, item):
        if front.isItemEditable(item):
            front.print('[front] does not select when clicking editable column')
            return
        if item == front._prev_tbl_item:
            return
        front._prev_tbl_item = item
        tbl = item.tableWidget()
        row, col = (item.row(), item.column())
        """
        #front.print('Clicked ' + tbl.objectName() + ' at ' + str((row, col)) +
                    #'--> ' + func.func_name)
        """
        return func(front, row, col, tbl)
    # Hacky decorator
    return clicked_wrapper


def uid_tbl_changed(func):
    """
    Wrapper around item_changed slot, which takes only the item.
    This extract the row, column, and table and passes that instead.
    Also provies debugging prints
    """
    @slot_(QtGui.QTableWidgetItem)
    @functools.wraps(func)
    def changed_wrapper(front, item):
        front._prev_tbl_item = item
        tbl = item.tableWidget()
        row, col = (item.row(), item.column())
        """
        #front.print('Changed ' + tbl.objectName() + ' at ' + str((row, col)) +
                    #'--> ' + func.func_name)
        """
        state = item.checkState()
        text  = str(item.text())
        #_pressed_state = front._pressed_item_state
        _pressed_text  = str(front._pressed_item_text)
        print(_pressed_text)
        isCheckableItem = text in ['true', 'false']  # HUGE HACK!
        if isCheckableItem:
            new_val = state == Qt.Checked
            """
            #try:
                #assert text == _pressed_text, 'cannot change text and state at the same time'
            #except AssertionError as ex:
                #utool.printex(ex, key_list=['state', 'text', '_pressed_state',
                                            #'_pressed_text'])
                #raise
            """
        else:
            new_val = text
        return func(front, row, col, tbl, new_val)
    # Hacky decorator
    return changed_wrapper


def _tee_logging(front):
    if VERBOSE:
        front.print('teeing log output')
    # Connect a StreamStealer object to the GUI output window
    #if __common__.__LOGGING__:
    #    front.logging_handler = guitool.GUILoggingHandler(front.gui_write)
    #    __common__.add_logging_handler(front.logging_handler)
    #else:
    #    front.ostream = StreamStealer(front, share=not noshare)


#=================
# Initialization
#=================


def update_tabwidget_text(front, tblname, text, enctext=''):
    tabTable  = front.get_tabWidget(enctext)
    tabWidget = front.get_tabView(tblname, enctext)
    index     = tabTable.indexOf(tabWidget)
    tab_text  = QTRANSLATE(front.objectName(), text, None, QUTF8)
    tabTable.setTabText(index, tab_text)


class MainWindowFrontend(QtGui.QMainWindow):
    printSignal      = signal_(str)
    raiseExceptionSignal = signal_(Exception)
    quitSignal       = signal_()
    selectGidSignal  = signal_(UUID_TYPE)
    selectRidSignal  = signal_(UUID_TYPE)
    selectNidSignal  = signal_(UUID_TYPE)
    selectQResSignal = signal_(UUID_TYPE)
    setRoiPropSignal = signal_(UUID_TYPE, str, str)
    aliasNidSignal   = signal_(UUID_TYPE, str, str)
    setGidPropSignal = signal_(UUID_TYPE, str, QtCore.QVariant)

    def __init__(front, back):
        super(MainWindowFrontend, front).__init__()
        #print('[*front] creating frontend')
        front._prev_tbl_item = None
        front._pressed_item_state = None
        front._pressed_item_text = None
        front.logging_handler = None
        front.back = back
        front.ui = Ui_mainSkel()
        front.ui.setupUi(front)
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
        front.raiseExceptionSignal.connect(back.backend_exception)
        front.quitSignal.connect(back.quit)
        front.selectGidSignal.connect(back.select_gid)
        front.selectRidSignal.connect(back.select_rid)
        front.selectNidSignal.connect(back.select_nid)
        front.selectQResSignal.connect(back.select_qres_rid)
        front.setRoiPropSignal.connect(back.set_roi_prop)
        front.aliasNidSignal.connect(back.set_name_prop)
        front.setGidPropSignal.connect(back.set_image_prop)

        # Connect all signals from GUI
        for func in front.ui.connect_fns:
            #printDBG('Connecting: %r' % func.func_name)
            func()
        #
        # Gui Components
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

    @slot_(str, list, list, list, list, list, str)
    def populate_tbl(front, tblname,
                     col_fancyheaders,
                     col_editable,
                     col_types,
                     row_list,
                     datatup_list,
                     enctext=''):
        tblname = str(tblname)
        enctext = str(enctext)
        #print('populate_tbl(%r, %r)' % (tblname, enctext,))
        try:
            tbl = front.get_tabTable(tblname, enctext)
        except KeyError:
            print('tbl not found')
            return

        try:
            uidtables.populate_item_table(tbl, col_fancyheaders, col_editable,
                                                col_types, row_list, datatup_list)
        except Exception as ex:
            front.raiseExceptionSignal.emit(ex)
            raise
        # Set the tab text to show the number of items listed
        fancy_tablename = uidtables.fancy_tablenames[tblname]
        text = fancy_tablename + ' : %d' % len(row_list)
        update_tabwidget_text(front, tblname, text, enctext)

    def isItemEditable(self, item):
        return int(Qt.ItemIsEditable & item.flags()) == int(Qt.ItemIsEditable)

    #====================
    # Encounter Widget / View / Table getters
    #====================

    def get_tabWidget(front, enctext):
        """ Returns the widget of the <enctext> encounter tab """
        tabWidget  = front.ui.__dict__[str('tablesTabWidget' + enctext)]
        return tabWidget

    def get_tabView(front, tblname, enctext):
        """ Returns the view containting the uid table """
        view = front.ui.__dict__[str(tblname + '_view' + enctext)]
        return view

    def get_tabTable(front, tblname, enctext):
        """ tblname in ['gids', 'rids', 'nids', 'qres'] """
        try:
            tableWidget = front.ui.__dict__[str(tblname + '_TBL' + enctext)]
        except KeyError:
            raise
        return tableWidget

    #=======================
    # General Table Getters
    #=======================

    def get_tbl_header(front, tbl, col):
        # Map the fancy header back to the internal one.
        fancy_header = str(tbl.horizontalHeaderItem(col).text())
        header = (uidtables.reverse_fancy[fancy_header]
                  if fancy_header in uidtables.reverse_fancy else fancy_header)
        return header

    def get_tbl_int(front, tbl, row, col):
        return int(tbl.item(row, col).text())

    def get_tbl_str(front, tbl, row, col):
        return str(tbl.item(row, col).text())

    def get_header_val(front, tbl, header, row):
        # RCOS TODO: This is hacky. These just need to be
        # in dicts to begin with.
        objectName = str(tbl.objectName())
        tblpos  = objectName.find('_TBL')
        tblname = objectName[0:tblpos]
        #enctext = tblname[tblpos + 4, :]
        #print(enctext)
        #tblname, enctext = str(tbl.objectName()).split('_TBL')
        col = uidtables.table_headers[tblname].index(header)
        return tbl.item(row, col).text()

    @slot_(QtGui.QTableWidgetItem)
    def uid_tbl_pressed(front, item):
        """ Keeps track of item state: i.e. if text or check value is changed """
        #front.print('')
        #front.print('>>>>>>>>>>>>>>>>>> PRESSED')
        #front.print('!!!!-Pressed')
        #front.print('Pressed _uid_tbl_pressed, tbl=%r' % str(item.tableWidget().objectName()))
        _pressed_state = item.checkState()
        _pressed_text = item.text()
        front._pressed_item_state = _pressed_state
        front._pressed_item_text = _pressed_text

    #=======================
    # Table Changed Functions
    #=======================

    @uid_tbl_changed
    def gids_tbl_changed(front, row, col, tbl, new_val):
        front.print('gids_tbl_changed()')
        sel_gid = UUID_TYPE(front.get_header_val(tbl, 'gid', row))
        header_lbl = front.get_tbl_header(tbl, col)
        front.setGidPropSignal.emit(sel_gid, header_lbl, new_val)

    @uid_tbl_changed
    def rids_tbl_changed(front, row, col, tbl, new_val):
        front.print('rids_tbl_changed()')
        sel_rid = UUID_TYPE(front.get_header_val(tbl, 'rid', row))  # Get selected roiid
        header_lbl = front.get_tbl_header(tbl, col)  # Get changed column
        front.setRoiPropSignal.emit(sel_rid, header_lbl, new_val)

    @uid_tbl_changed
    def qres_tbl_changed(front, row, col, tbl, new_val):
        front.print('qres_tbl_changed()')
        sel_rid = UUID_TYPE(front.get_header_val(tbl, 'rid', row))  # The changed row's roi id
        header_lbl = front.get_tbl_header(tbl, col)  # Get changed column
        front.setRoiPropSignal.emit(sel_rid, header_lbl, new_val)

    @uid_tbl_changed
    def nids_tbl_changed(front, row, col, tbl, new_val):
        front.print('nids_tbl_changed()')
        sel_nid = UUID_TYPE(front.get_header_val(tbl, 'nid', row))    # The changed row's name index
        header_lbl = front.get_tbl_header(tbl, col)  # Get changed column
        front.aliasNidSignal.emit(sel_nid, header_lbl, new_val)

    #=======================
    # Table Clicked Functions
    #=======================
    @uid_tbl_clicked
    def gids_tbl_clicked(front, row, col, tbl):
        sel_gid = UUID_TYPE(front.get_header_val(tbl, 'gid', row))
        front.selectGidSignal.emit(sel_gid)

    @uid_tbl_clicked
    def rids_tbl_clicked(front, row, col, tbl):
        sel_rid = UUID_TYPE(front.get_header_val(tbl, 'rid', row))
        front.selectRidSignal.emit(sel_rid)

    @uid_tbl_clicked
    def qres_tbl_clicked(front, row, col, tbl):
        sel_rid = UUID_TYPE(front.get_header_val(tbl, 'rid', row))
        front.selectQResSignal.emit(sel_rid)

    @uid_tbl_clicked
    def nids_tbl_clicked(front, row, col, tbl):
        sel_nid = UUID_TYPE(front.get_header_val(tbl, 'nid', row))
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
            front.print(front.back.cfg)

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
