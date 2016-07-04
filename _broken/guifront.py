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
from ibeis.gui import rowidtables
from ibeis.gui.Skeleton import Ui_mainSkel

print, print_, printDBG, profile, rrr = utool.inject(
    __name__, '[front*]', DEBUG=False)

QUIET   = utool.get_flag('--quiet')
VERBOSE = utool.get_flag(('--verbose', '--verbose-front', '--vf'))


UID_TYPE = rowidtables.UID_TYPE
ENC_TYPE = str  # imagesets are ided with imagesettext right now
QUTF8      = QtWidgets.QApplication.UnicodeUTF8
QTRANSLATE = QtWidgets.QApplication.translate

#=================
# Decorators / Helpers
#=================


def rowid_tbl_clicked(func):
    """
    Wrapper around item_clicked slot, which takes only the item.
    This extract the row, column, and table and passes that instead.
    Also provies debugging prints
    """
    @slot_(QtWidgets.QTableWidgetItem)
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
        front.print('Clicked ' + tbl.objectName() + ' at ' + str((row, col)) +
                    '--> ' + func.func_name)
        return func(front, row, col, tbl)
    # Hacky decorator
    return clicked_wrapper


def rowid_tbl_changed(func):
    """
    Wrapper around item_changed slot, which takes only the item.
    This extract the row, column, and table and passes that instead.
    Also provies debugging prints
    """
    @slot_(QtWidgets.QTableWidgetItem)
    @functools.wraps(func)
    def changed_wrapper(front, item):
        front._prev_tbl_item = item
        tbl = item.tableWidget()
        row, col = (item.row(), item.column())
        front.print('Changed ' + tbl.objectName() + ' at ' + str((row, col)) +
                    '--> ' + func.func_name)
        state = item.checkState()
        text  = str(item.text())
        #_pressed_state = front._pressed_item_state
        _pressed_text  = str(front._pressed_item_text)
        print(_pressed_text)
        isCheckableItem = text in ['true', 'false']  # HUGE HACK!
        if isCheckableItem:
            new_val = state == Qt.Checked
            #try:
            #    assert text == _pressed_text, \
            #        'cannot change text and state at the same time'
            #except AssertionError as ex:
            #    utool.printex(ex, key_list=[
            #        'state', 'text', '_pressed_state', '_pressed_text'])
            #    raise
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


def update_tabwidget_text(front, tblname, text, imagesettext=''):
    tabTable  = front.get_tabWidget(imagesettext)
    tabWidget = front.get_tabView(tblname, imagesettext)
    index     = tabTable.indexOf(tabWidget)
    tab_text  = QTRANSLATE(front.objectName(), text, None, QUTF8)
    tabTable.setTabText(index, tab_text)


class MainWindowFrontend(QtWidgets.QMainWindow):
    printSignal      = signal_(str)
    raiseExceptionSignal = signal_(Exception)
    quitSignal       = signal_()
    selectGidSignal  = signal_(UID_TYPE, ENC_TYPE)
    selectRidSignal  = signal_(UID_TYPE, ENC_TYPE)
    selectNidSignal  = signal_(UID_TYPE, ENC_TYPE)
    selectQResSignal = signal_(UID_TYPE, ENC_TYPE)
    setRoiPropSignal = signal_(UID_TYPE, str, str)
    aliasNidSignal   = signal_(UID_TYPE, str, str)
    setGidPropSignal = signal_(UID_TYPE, str, QtCore.QVariant)

    def __init__(front, back):
        super(MainWindowFrontend, front).__init__()
        print('[*front] creating frontend')
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

        #
        # Gui Components
        # Tab Widget
        ui.tablesTabWidget.currentChanged.connect(front.change_view)
        #ui.rids_TBL.sortByColumn(0, Qt.AscendingOrder)
        #ui.qres_TBL.sortByColumn(0, Qt.AscendingOrder)
        #ui.gids_TBL.sortByColumn(0, Qt.AscendingOrder)

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

    @slot_(str)
    def updateWindowTitle(front, title):
        #front.print('front.setWindowTitle(title=%r)' % (str(title),))
        front.setWindowTitle(title)
        #front.ui.retranslateUi(front)

    @slot_(str, list, list, list, list, list, str)
    def populate_tbl(front, tblname,
                     col_fancyheaders,
                     col_editable,
                     col_types,
                     row_list,
                     datatup_list,
                     imagesettext=''):
        tblname = str(tblname)
        imagesettext = str(imagesettext)
        #print('populate_tbl(%r, %r)' % (tblname, imagesettext,))
        try:
            tbl = front.get_tabTable(tblname, imagesettext)
        except KeyError:
            print('tbl not found')
            return

        try:
            rowidtables.populate_item_table(tbl,
                                          col_fancyheaders,
                                          col_editable,
                                          col_types,
                                          row_list,
                                          datatup_list)
        except Exception as ex:
            front.raiseExceptionSignal.emit(ex)
            raise
        # Set the tab text to show the number of items listed
        fancy_tablename = rowidtables.fancy_tablenames[tblname]
        text = fancy_tablename + ' : %d' % len(row_list)
        update_tabwidget_text(front, tblname, text, imagesettext)

    def isItemEditable(self, item):
        return int(Qt.ItemIsEditable & item.flags()) == int(Qt.ItemIsEditable)

    #====================
    # ImageSet Widget / View / Table getters
    #====================

    def get_tabWidget(front, imagesettext):
        """ Returns the widget of the <imagesettext> imageset tab """
        tabWidget  = front.ui.__dict__[str('tablesTabWidget' + imagesettext)]
        return tabWidget

    def get_tabView(front, tblname, imagesettext):
        """ Returns the view containting the rowid table """
        view = front.ui.__dict__[str(tblname + '_view' + imagesettext)]
        return view

    def get_tabTable(front, tblname, imagesettext):
        """ tblname in ['gids', 'rids', 'nids', 'qres'] """
        try:
            tableWidget = front.ui.__dict__[str(tblname + '_TBL' + imagesettext)]
        except KeyError:
            raise
        return tableWidget

    #=======================
    # General Table Getters
    #=======================

    def get_tbl_header(front, tbl, col):
        # Map the fancy header back to the internal one.
        fancy_header = str(tbl.horizontalHeaderItem(col).text())
        header = (rowidtables.reverse_fancy[fancy_header]
                  if fancy_header in rowidtables.reverse_fancy else fancy_header)
        return header

    def get_tbl_int(front, tbl, row, col):
        return int(tbl.item(row, col).text())

    def get_tbl_str(front, tbl, row, col):
        return str(tbl.item(row, col).text())

    def get_tbl_imagesettext(front, tbl):
        objectName = str(tbl.objectName())
        tblpos  = objectName.find('_TBL')
        #tblname = objectName[0:tblpos]
        imagesettext = objectName[tblpos + 4:]
        return imagesettext

    def get_header_val(front, tbl, header, row):
        # RCOS TODO: This is hacky. These just need to be
        # in dicts to begin with.
        objectName = str(tbl.objectName())
        tblpos  = objectName.find('_TBL')
        tblname = objectName[0:tblpos]
        #imagesettext = tblname[tblpos + 4, :]
        #print(imagesettext)
        #tblname, imagesettext = str(tbl.objectName()).split('_TBL')
        col = rowidtables.table_headers[tblname].index(header)
        return tbl.item(row, col).text()

    @slot_(QtWidgets.QTableWidgetItem)
    def rowid_tbl_pressed(front, item):
        """ Keeps track of item state: if text or check value is changed """
        #front.print('')
        #front.print('>>>>>>>>>>>>>>>>>> PRESSED')
        #front.print('!!!!-Pressed')
        #front.print('Pressed _rowid_tbl_pressed, tbl=%r' %
        #            str(item.tableWidget().objectName()))
        _pressed_state = item.checkState()
        _pressed_text = item.text()
        front._pressed_item_state = _pressed_state
        front._pressed_item_text = _pressed_text

    #=======================
    # Table Changed Functions
    #=======================

    @rowid_tbl_changed
    def gids_tbl_changed(front, row, col, tbl, new_val):
        front.print('gids_tbl_changed()')
        sel_gid = UID_TYPE(front.get_header_val(tbl, 'gid', row))
        header_lbl = front.get_tbl_header(tbl, col)
        front.setGidPropSignal.emit(sel_gid, header_lbl, new_val)

    @rowid_tbl_changed
    def rids_tbl_changed(front, row, col, tbl, new_val):
        front.print('rids_tbl_changed()')
        # Get selected roid
        sel_rid = UID_TYPE(front.get_header_val(tbl, 'rid', row))
        # Get changed column
        header_lbl = front.get_tbl_header(tbl, col)
        front.setRoiPropSignal.emit(sel_rid, header_lbl, new_val)

    @rowid_tbl_changed
    def qres_tbl_changed(front, row, col, tbl, new_val):
        front.print('qres_tbl_changed()')
        # The changed row's roi id
        sel_rid = UID_TYPE(front.get_header_val(tbl, 'rid', row))
        # Get changed column
        header_lbl = front.get_tbl_header(tbl, col)
        front.setRoiPropSignal.emit(sel_rid, header_lbl, new_val)

    @rowid_tbl_changed
    def nids_tbl_changed(front, row, col, tbl, new_val):
        front.print('nids_tbl_changed()')
        # The changed row's name index
        sel_nid = UID_TYPE(front.get_header_val(tbl, 'nid', row))
        # Get changed column
        header_lbl = front.get_tbl_header(tbl, col)
        front.aliasNidSignal.emit(sel_nid, header_lbl, new_val)

    #=======================
    # Table Clicked Functions
    #=======================
    @rowid_tbl_clicked
    def gids_tbl_clicked(front, row, col, tbl):
        sel_gid = UID_TYPE(front.get_header_val(tbl, 'gid', row))
        imagesettext = front.get_tbl_imagesettext(tbl)
        front.selectGidSignal.emit(sel_gid, imagesettext)

    @rowid_tbl_clicked
    def rids_tbl_clicked(front, row, col, tbl):
        sel_rid = UID_TYPE(front.get_header_val(tbl, 'rid', row))
        imagesettext = front.get_tbl_imagesettext(tbl)
        front.selectRidSignal.emit(sel_rid, imagesettext)

    @rowid_tbl_clicked
    def qres_tbl_clicked(front, row, col, tbl):
        sel_rid = UID_TYPE(front.get_header_val(tbl, 'rid', row))
        imagesettext = front.get_tbl_imagesettext(tbl)
        front.selectQResSignal.emit(sel_rid, imagesettext)

    @rowid_tbl_clicked
    def nids_tbl_clicked(front, row, col, tbl):
        sel_nid = UID_TYPE(front.get_header_val(tbl, 'nid', row))
        imagesettext = front.get_tbl_imagesettext(tbl)
        front.selectNidSignal.emit(sel_nid, imagesettext)

    #=======================
    # Other
    #=======================

    @slot_(int)
    def change_view(front, new_state):
        """ Function called whenever the veiw is changed """
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
