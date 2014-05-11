#TODO: Licence
from __future__ import absolute_import, division, print_function
import functools
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
import guitool

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


QTRANSLATE = QtGui.QApplication.translate
QUTF8      = QtGui.QApplication.UnicodeUTF8


# ____________
# HELPERS

def _new_size_policy(widget,
                     hpolicy=QSizePolicy.Expanding,
                     vpolicy=QSizePolicy.Expanding,
                     hstretch=0,
                     vstretch=0):
    """
    The QSizePolicy class is a layout attribute describing horizontal and
    vertical resizing policy.

    The size policy of a widget is an expression of its willingness to be
    resized in various ways, and affects how the widget is treated by the layout
    engine. Each widget returns a QSizePolicy that describes the horizontal and
    vertical resizing policy it prefers when being laid out. You can change this
    for a specific widget by changing its QWidget::sizePolicy property.

    QSizePolicy contains two independent QSizePolicy::Policy values and two
    stretch factors; one describes the widgets's horizontal size policy, and the
    other describes its vertical size policy. It also contains a flag to
    indicate whether the height and width of its preferred size are related.
    """
    sizePolicy = QtGui.QSizePolicy(hpolicy, vpolicy)
    sizePolicy.setHorizontalStretch(hstretch)
    sizePolicy.setVerticalStretch(vstretch)
    sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def _new_gridLayout(parent, _numtrick=[0]):
    gridLayout = QtGui.QGridLayout(parent)
    _numtrick[0] += 1  # NOQA
    next_gridname = _fromUtf8('gridLayout_' + str(_numtrick[0]))
    gridLayout.setObjectName(next_gridname)
    return gridLayout


def _new_verticalLayout(parent=None, _numtrick=[0]):
    _numtrick[0] += 1  # NOQA
    verticalLayout = QtGui.QVBoxLayout() if parent is None else QtGui.QVBoxLayout(parent)
    next_vlname = _fromUtf8('verticalLayout' + str(_numtrick[0]))
    verticalLayout.setObjectName(next_vlname)
    return verticalLayout

# ____________
# COMPONENTS


def func_name_aug(name):
    """ Decorator which augments a function name """
    def augment_wrapper(func):
        func.func_name = name + '_' + func.func_name
        return func
    return augment_wrapper


def initMainWidget(front, name, size=(500, 300), title=''):
    front.setObjectName(_fromUtf8(name))
    (w, h) = size
    front.resize(w, h)
    front.setUnifiedTitleAndToolBarOnMac(False)
    @func_name_aug(name)
    def retranslate_fn():
        front.setWindowTitle(QTRANSLATE(name, title, None, QUTF8))
    front.ui.retranslatable_fns.append(retranslate_fn)
    return front


def newCentralLayout(front):
    centralwidget = QtGui.QWidget(front)
    centralwidget.setObjectName(_fromUtf8('centralwidget'))
    verticalLayout = _new_verticalLayout(centralwidget)
    front.setCentralWidget(centralwidget)
    return centralwidget, verticalLayout


def newVerticalSplitter(centralwidget, verticalLayout, name='splitter'):
    splitter = QtGui.QSplitter(centralwidget)
    splitter.setOrientation(QtCore.Qt.Vertical)
    splitter.setObjectName(_fromUtf8(name))
    verticalLayout.addWidget(splitter)
    return splitter


def newMenubar(front, name):
    """ Defines the menubar on top of the main widget """
    menubar = QtGui.QMenuBar(front)
    menubar.setGeometry(QtCore.QRect(0, 0, 1013, 23))
    menubar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
    menubar.setDefaultUp(False)
    menubar.setNativeMenuBar(False)
    menubar.setObjectName(_fromUtf8(name))
    front.setMenuBar(menubar)
    return menubar


def newMenu(front, menubar, name, text):
    """ Defines each menu category in the menubar """
    menu = QtGui.QMenu(menubar)
    menu.setObjectName(_fromUtf8(name))
    @func_name_aug(name)
    def retranslate_fn():
        menu.setTitle(QTRANSLATE(front.objectName(), text, None, QUTF8))
    front.ui.retranslatable_fns.append(retranslate_fn)
    @func_name_aug(name)
    def postsetup_fn():
        menubar.addAction(menu.menuAction())
    front.ui.postsetup_fns.append(postsetup_fn)
    # Define a custom newAction function for the menu
    # The QT function is called addAction
    newAction = functools.partial(newMenuAction, front, name)
    setattr(menu, 'newAction', newAction)
    return menu


def newTable(front, parent, tblname):
    table = QtGui.QTableWidget(parent)
    table.setDragEnabled(False)
    table.setObjectName(_fromUtf8(tblname))
    table.setColumnCount(0)
    table.setRowCount(0)
    @func_name_aug(tblname)
    def retranslate_fn():
        table.setSortingEnabled(True)
    front.ui.retranslatable_fns.append(retranslate_fn)
    return table


# TABS TABS TABS

def newTabWidget(front, parent, name, vstretch=10):
    tabWidget = QtGui.QTabWidget(parent)
    sizePolicy = _new_size_policy(tabWidget, vstretch=vstretch)
    tabWidget.setSizePolicy(sizePolicy)
    tabWidget.setMinimumSize(QtCore.QSize(0, 0))
    tabWidget.setObjectName(_fromUtf8(name))
    # Custom new tab function
    # Partial function to create a new tab in this widget
    newTabbedTable_ = functools.partial(newTabbedTable, front, tabWidget)
    setattr(tabWidget, 'newTabbedTable', newTabbedTable_)
    setattr(front.ui, name, tabWidget)
    return tabWidget


def newTabbedView(front, tabWidget, viewname, text):
    """ ANY TAB IS A VIEW WITH AN OBJECT IN IT """
    view = QtGui.QWidget()
    view.setObjectName(_fromUtf8(viewname))
    gridLayout     = _new_gridLayout(view)
    verticalLayout = _new_verticalLayout()
    # Keep references to layout
    setattr(front.ui, str(verticalLayout.objectName()), verticalLayout)
    setattr(front.ui, str(gridLayout.objectName()), gridLayout)
    gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)
    tabWidget.addTab(view, _fromUtf8(''))
    def retranslate_fn():
        qtext = QTRANSLATE(front.objectName(), text, None, QUTF8)
        tabWidget.setTabText(tabWidget.indexOf(view), qtext)
    front.ui.retranslatable_fns.append(retranslate_fn)
    setattr(front.ui, viewname, view)
    return view, verticalLayout


def newTabbedTable(front, tabWidget, viewname, tblname, text=''):
    """ Builds view, uidTBL, gridLayout, verticalLayout """
    # IMG / ROI/ NAME / RES VIEW
    view, verticalLayout = newTabbedView(front, tabWidget, viewname, text)
    # G/R/N/Q-ID TABLE
    table = newTable(front, view, tblname)
    setattr(front.ui, tblname, table)
    verticalLayout.addWidget(table)
    return view, table


def newTabbedTabWidget(front, tabWidget, viewname, name, text='', **kwargs):
    """ Builds view, uidTBL, gridLayout, verticalLayout """
    # IMG / ROI/ NAME / RES VIEW
    view, verticalLayout = newTabbedView(front, tabWidget, viewname, text)
    # G/R/N/Q-ID TABLE
    tabWidget2 = newTabWidget(front, view, name, **kwargs)
    verticalLayout.addWidget(tabWidget2)
    return tabWidget2


def newOutputEdit(parent, name='outputEdit', visible=True):
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = _new_size_policy(outputEdit, vstretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    outputEdit.setObjectName(_fromUtf8(name))
    outputEdit.setVisible(visible)
    return outputEdit


def newProgressBar(parent, name='progressBar', visible=True):
    progressBar = QtGui.QProgressBar(parent)
    sizePolicy = _new_size_policy(progressBar, vpolicy=QSizePolicy.Fixed, vstretch=1)
    progressBar.setSizePolicy(sizePolicy)
    progressBar.setProperty('value', 24)
    progressBar.setObjectName(_fromUtf8(name))
    progressBar.setVisible(visible)
    return progressBar


def msg_event(title, msg):
    """ Returns a message event slot """
    return lambda: guitool.msgbox(title, msg)


def newMenuAction(front, menu_name, name=None, text=None, shortcut=None,
                  tooltip=None, slot_fn=None, enabled=True):
    assert name is not None
    ui = front.ui
    # Dynamically add new menu actions programatically
    action_name = name
    action_text = text
    action_shortcut = shortcut
    action_tooltip  = tooltip
    ui = front.ui
    if hasattr(ui, action_name):
        raise Exception('menu action already defined')
    # Create new action
    action = QtGui.QAction(front)
    setattr(ui, action_name, action)
    action.setEnabled(enabled)
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    action.setObjectName(_fromUtf8(action_name))
    menu = getattr(ui, menu_name)
    menu.addAction(action)
    if action_text is None:
        action_text = action_name
    # TODO: Have ui.retranslateUi call this
    @func_name_aug(name)
    def retranslate_fn():
        if action_text is not None:
            qtext    = QTRANSLATE(front.objectName(), action_text, None, QUTF8)
            action.setText(qtext)
        if action_tooltip is not None:
            qtooltip = QTRANSLATE(front.objectName(), action_tooltip, None, QUTF8)
            action.setToolTip(qtooltip)
        if action_shortcut is not None:
            qshortcut = QTRANSLATE(front.objectName(), action_shortcut, None, QUTF8)
            action.setShortcut(qshortcut)
    front.ui.retranslatable_fns.append(retranslate_fn)
    if slot_fn is not None:
        @func_name_aug(name)
        def connect_fn():
            action.triggered.connect(slot_fn)
        front.ui.connect_fns.append(connect_fn)
    return action
    #retranslate_fn()
