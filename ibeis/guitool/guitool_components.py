from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt      # NOQA
import functools
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool-components]')


def newSizePolicy(widget):
    """
    input: widget - the central widget
    """
    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newHorizontalSplitter(widget):
    """
    input: widget - the central widget
    """
    hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal, widget)
    # This line makes the hsplitter resize with the widget
    sizePolicy = newSizePolicy(widget)
    hsplitter.setSizePolicy(sizePolicy)
    return hsplitter


def newMenubar(widget):
    """ Defines the menubar on top of the main widget """
    menubar = QtGui.QMenuBar(widget)
    menubar.setGeometry(QtCore.QRect(0, 0, 1013, 23))
    menubar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
    menubar.setDefaultUp(False)
    menubar.setNativeMenuBar(False)
    widget.setMenuBar(menubar)
    return menubar


def newMenu(widget, menubar, name, text):
    """ Defines each menu category in the menubar """
    menu = QtGui.QMenu(menubar)
    menu.setObjectName(name)
    menu.setTitle(text)
    #@__define_postsetup(name, widget)
    #def postsetup_fn():
    #    menubar.addAction(menu.menuAction())
    # Define a custom newAction function for the menu
    # The QT function is called addAction
    newAction = functools.partial(newMenuAction, widget, name)
    setattr(menu, 'newAction', newAction)
    return menu


def newMenuAction(front, menu_name, name=None, text=None, shortcut=None,
                  tooltip=None, slot_fn=None, enabled=True):
    assert name is not None
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
    #retranslate_fn()
