from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt
import functools
import utool
from . import guitool_dialogs
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool-components]')


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
    sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newHorizontalSplitter(widget):
    """
    input: widget - the central widget
    """
    hsplitter = QtGui.QSplitter(Qt.Horizontal, widget)
    # This line makes the hsplitter resize with the widget
    sizePolicy = newSizePolicy(widget)
    hsplitter.setSizePolicy(sizePolicy)
    return hsplitter


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


def newProgressBar(parent, visible=True):
    progressBar = QtGui.QProgressBar(parent)
    sizePolicy = newSizePolicy(progressBar,
                               verticalSizePolicy=QSizePolicy.Fixed,
                               verticalStretch=1)
    progressBar.setSizePolicy(sizePolicy)
    progressBar.setProperty('value', 24)
    progressBar.setVisible(visible)
    return progressBar


def newOutputEdit(parent, visible=True):
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = newSizePolicy(outputEdit, verticalStretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    outputEdit.setVisible(visible)
    return outputEdit


def msg_event(title, msg):
    """ Returns a message event slot """
    return lambda: guitool_dialogs.msgbox(title, msg)
