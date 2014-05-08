#TODO: Licence
from __future__ import absolute_import, division, print_function
from PyQt4 import QtGui, QtCore

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


def _new_size_policy(widget, vspace=0):
    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(vspace)
    sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def _new_gridLayout(parent, _numtrick=[0]):
    gridLayout = QtGui.QGridLayout(parent)
    _numtrick[0] += 1  # NOQA
    next_gridname = _fromUtf8('gridLayout_' + str(_numtrick[0]))
    gridLayout.setObjectName(next_gridname)
    return gridLayout


def _new_verticalLayout(_numtrick=[0]):
    _numtrick[0] += 1  # NOQA
    verticalLayout = QtGui.QVBoxLayout()
    next_vlname = _fromUtf8('verticalLayout' + str(_numtrick[0]))
    verticalLayout.setObjectName(next_vlname)
    return verticalLayout


def newTabWidget(ui, parent, name, **kwargs):
    tabWidget = QtGui.QTabWidget(parent)
    sizePolicy = _new_size_policy(tabWidget, vspace=10)
    tabWidget.setSizePolicy(sizePolicy)
    tabWidget.setMinimumSize(QtCore.QSize(0, 0))
    tabWidget.setObjectName(_fromUtf8(name))
    return tabWidget


def newTabView(tabWidget, viewname, tblname, layout='default', **kwargs):
    """ Builds view, uidTBL, gridLayout, verticalLayout """
    # IMG / ROI/ NAME / RES VIEW
    view = QtGui.QWidget()
    view.setObjectName(_fromUtf8(viewname))
    # NEXT LAYOUT
    gridLayout     = _new_gridLayout(view)
    verticalLayout = _new_verticalLayout()
    # G/R/N/Q-ID TABLE
    uid_TBL = QtGui.QTableWidget(view)
    uid_TBL.setDragEnabled(False)
    uid_TBL.setObjectName(_fromUtf8(tblname))
    uid_TBL.setColumnCount(0)
    uid_TBL.setRowCount(0)
    verticalLayout.addWidget(uid_TBL)
    gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)
    tabWidget.addTab(view, _fromUtf8(''))
    return view, gridLayout, verticalLayout, uid_TBL


def new_output_edit(ui):
    ui.outputEdit = QtGui.QTextEdit(ui.splitter)
    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(1)
    sizePolicy.setHeightForWidth(ui.outputEdit.sizePolicy().hasHeightForWidth())
    ui.outputEdit.setSizePolicy(sizePolicy)
    ui.outputEdit.setAcceptRichText(False)
    ui.outputEdit.setObjectName(_fromUtf8("outputEdit"))


def new_progress_bar(ui):
    ui.progressBar = QtGui.QProgressBar(ui.splitter)
    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(1)
    sizePolicy.setHeightForWidth(ui.progressBar.sizePolicy().hasHeightForWidth())
    ui.progressBar.setSizePolicy(sizePolicy)
    ui.progressBar.setProperty("value", 24)
    ui.progressBar.setObjectName(_fromUtf8("progressBar"))


def new_tab_tables(ui):
    ui.tablesTabWidget = newTabWidget(
        ui, ui.splitter, 'tablesTabWidget', vspace=10)
    tabWidget = ui.tablesTabWidget

    # IMG TAB
    (ui.image_view, ui.gids_TBL,
     ui.gridLayout_0, ui.verticalLayout_0,) = \
        newTabView(tabWidget, 'img_view', 'gids_TBL')

    # ROI TAB
    (ui.roi_view, ui.rids_TBL,
     ui.gridLayout_1, ui.verticalLayout_1,) = \
        newTabView(tabWidget, 'roi_view', 'rids_TBL')

    # NAME TAB
    (ui.name_view, ui.nids_TBL,
     ui.gridLayout_2, ui.verticalLayout_2,) = \
        newTabView(tabWidget, 'name_view', 'nids_TBL')

    # RECOGNITION TAB
    (ui.res_view, ui.res_TBL,
     ui.gridLayout_3, ui.verticalLayout_3,) = \
        newTabView(tabWidget, 'res_view', 'qres_TBL')
    return ui


def new_menu_action(front, menu_name, name, text=None, shortcut=None, slot_fn=None):
    ui = front.ui
    # Dynamically add new menu actions programatically
    action_name = name
    action_text = text
    action_shortcut = shortcut
    ui = front.ui
    if hasattr(ui, action_name):
        raise Exception('menu action already defined')
    # Create new action
    action = QtGui.QAction(front)
    setattr(ui, action_name, action)
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    action.setObjectName(_fromUtf8(action_name))
    menu = getattr(ui, menu_name)
    menu.addAction(action)
    if action_text is None:
        action_text = action_name
    # TODO: Have ui.retranslateUi call this
    qtranslate = QtGui.QApplication.translate
    qutf8 = QtGui.QApplication.UnicodeUTF8
    def retranslate_fn():
        qtext = qtranslate('mainSkel', action_text, None, qutf8)
        action.setText(qtext)
        if action_shortcut is not None:
            qshortcut = qtranslate('mainSkel', action_shortcut, None, qutf8)
            action.setShortcut(qshortcut)
    def connect_fn():
        action.triggered.connect(slot_fn)
    connect_fn.func_name = name + '_' + connect_fn.func_name
    retranslate_fn.func_name = name + '_' + retranslate_fn.func_name
    front.connect_fns.append(connect_fn)
    front.retranslatable_fns.append(retranslate_fn)
    retranslate_fn()
