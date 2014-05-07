#TODO: Licence
from __future__ import absolute_import, division, print_function
from PyQt4 import QtGui, QtCore

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


def new_tab_tables(front):
    ui = front.ui
    return ui


def new_menu_action(front, menu_name, name, text=None, shortcut=None, slot_fn=None):
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
