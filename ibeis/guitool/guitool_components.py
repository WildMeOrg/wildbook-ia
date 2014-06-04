from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt
import functools
import utool
from . import guitool_dialogs
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool_components]')


def newSizePolicy(widget,
                  verticalSizePolicy=QSizePolicy.Expanding,
                  horizontalSizePolicy=QSizePolicy.Expanding,
                  horizontalStretch=0,
                  verticalStretch=0):
    """
    input: widget - the central widget
    """
    sizePolicy = QSizePolicy(horizontalSizePolicy, verticalSizePolicy)
    sizePolicy.setHorizontalStretch(horizontalStretch)
    sizePolicy.setVerticalStretch(verticalStretch)
    #sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newSplitter(widget, orientation=Qt.Horizontal, verticalStretch=1):
    """
    input: widget - the central widget
    """
    hsplitter = QtGui.QSplitter(orientation, widget)
    # This line makes the hsplitter resize with the widget
    sizePolicy = newSizePolicy(hsplitter, verticalStretch=verticalStretch)
    hsplitter.setSizePolicy(sizePolicy)
    return hsplitter


def newTabWidget(parent, horizontalStretch=1):
    tabwgt = QtGui.QTabWidget(parent)
    sizePolicy = newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
    tabwgt.setSizePolicy(sizePolicy)
    return tabwgt


def newMenubar(widget):
    """ Defines the menubar on top of the main widget """
    menubar = QtGui.QMenuBar(widget)
    menubar.setGeometry(QtCore.QRect(0, 0, 1013, 23))
    menubar.setContextMenuPolicy(Qt.DefaultContextMenu)
    menubar.setDefaultUp(False)
    menubar.setNativeMenuBar(False)
    widget.setMenuBar(menubar)
    return menubar


def newMenu(widget, menubar, name, text):
    """ Defines each menu category in the menubar """
    menu = QtGui.QMenu(menubar)
    menu.setObjectName(name)
    menu.setTitle(text)
    # Define a custom newAction function for the menu
    # The QT function is called addAction
    newAction = functools.partial(newMenuAction, widget, name)
    setattr(menu, 'newAction', newAction)
    # Add the menu to the menubar
    menubar.addAction(menu.menuAction())
    return menu


def newMenuAction(front, menu_name, name=None, text=None, shortcut=None,
                  tooltip=None, slot_fn=None, enabled=True):
    assert name is not None, 'menuAction name cannot be None'
    # Dynamically add new menu actions programatically
    action_name = name
    action_text = text
    action_shortcut = shortcut
    action_tooltip  = tooltip
    if hasattr(front, action_name):
        raise Exception('menu action already defined')
    # Create new action
    action = QtGui.QAction(front)
    setattr(front, action_name, action)
    action.setEnabled(enabled)
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    menu = getattr(front, menu_name)
    menu.addAction(action)
    if action_text is None:
        action_text = action_name
    if action_text is not None:
        action.setText(action_text)
    if action_tooltip is not None:
        action.setToolTip(action_tooltip)
    if action_shortcut is not None:
        action.setShortcut(action_shortcut)
    if slot_fn is not None:
        action.triggered.connect(slot_fn)
    return action


def newProgressBar(parent, visible=True, verticalStretch=1):
    progressBar = QtGui.QProgressBar(parent)
    sizePolicy = newSizePolicy(progressBar,
                               verticalSizePolicy=QSizePolicy.Maximum,
                               verticalStretch=verticalStretch)
    progressBar.setSizePolicy(sizePolicy)
    progressBar.setProperty('value', 42)
    progressBar.setTextVisible(False)
    progressBar.setVisible(visible)
    return progressBar


def newOutputLog(parent, pointSize=6, visible=True, verticalStretch=1):
    from .guitool_misc import QLoggedOutput
    outputLog = QLoggedOutput(parent)
    sizePolicy = newSizePolicy(outputLog,
                               #verticalSizePolicy=QSizePolicy.Preferred,
                               verticalStretch=verticalStretch)
    outputLog.setSizePolicy(sizePolicy)
    outputLog.setAcceptRichText(False)
    outputLog.setVisible(visible)
    #outputLog.setFontPointSize(8)
    outputLog.setFont(newFont('Courier New', pointSize))
    return outputLog


def newTextEdit(parent, visible=True):
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = newSizePolicy(outputEdit, verticalStretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    outputEdit.setVisible(visible)
    return outputEdit


def newWidget(parent, orientation=Qt.Vertical,
              verticalSizePolicy=QSizePolicy.Expanding,
              horizontalSizePolicy=QSizePolicy.Expanding,
              verticalStretch=1):
    widget = QtGui.QWidget(parent)

    sizePolicy = newSizePolicy(widget,
                               horizontalSizePolicy=horizontalSizePolicy,
                               verticalSizePolicy=verticalSizePolicy,
                               verticalStretch=1)
    widget.setSizePolicy(sizePolicy)
    if orientation == Qt.Vertical:
        layout = QtGui.QVBoxLayout(widget)
    elif orientation == Qt.Horizontal:
        layout = QtGui.QHBoxLayout(widget)
    else:
        raise NotImplementedError('orientation')
    # Black magic
    widget._guitool_layout = layout
    widget.addWidget = widget._guitool_layout.addWidget
    widget.addLayout = widget._guitool_layout.addLayout
    return widget


def newFont(fontname='Courier New', pointSize=-1, weight=-1, italic=False):
    #fontname = 'Courier New'
    #pointSize = 8
    #weight = -1
    #italic = False
    font = QtGui.QFont(fontname, pointSize=pointSize, weight=weight, italic=italic)
    return font


def newButton(parent, text, clicked=None):
    button = QtGui.QPushButton(text, parent=parent, clicked=clicked)
    return button


def newLabel(parent, text):
    label = QtGui.QLabel(text, parent=parent)
    label.setAlignment(Qt.AlignCenter)
    return label


def getAvailableFonts():
    fontdb = QtGui.QFontDatabase()
    available_fonts = map(str, list(fontdb.families()))
    return available_fonts


def layoutSplitter(splitter):
    old_sizes = splitter.sizes()
    print(old_sizes)
    phi = utool.get_phi()
    total = sum(old_sizes)
    ratio = 1 / phi
    sizes = []
    for count, size in enumerate(old_sizes[:-1]):
        new_size = int(round(total * ratio))
        total -= new_size
        sizes.append(new_size)
    sizes.append(total)
    splitter.setSizes(sizes)
    print(sizes)
    print('===')


def msg_event(title, msg):
    """ Returns a message event slot """
    return lambda: guitool_dialogs.msgbox(title, msg)
