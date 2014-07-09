# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/joncrall/code/hotspotter/hsgui/_frontend/EditPrefSkel.ui'
#
# Created: Mon Feb 10 13:40:41 2014
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_editPrefSkel(object):
    def setupUi(self, editPrefSkel):
        editPrefSkel.setObjectName(_fromUtf8("editPrefSkel"))
        editPrefSkel.resize(668, 530)
        self.verticalLayout = QtGui.QVBoxLayout(editPrefSkel)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.prefTreeView = QtGui.QTreeView(editPrefSkel)
        self.prefTreeView.setObjectName(_fromUtf8("prefTreeView"))
        self.verticalLayout.addWidget(self.prefTreeView)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.redrawBUT = QtGui.QPushButton(editPrefSkel)
        self.redrawBUT.setObjectName(_fromUtf8("redrawBUT"))
        self.horizontalLayout.addWidget(self.redrawBUT)
        self.unloadFeaturesAndModelsBUT = QtGui.QPushButton(editPrefSkel)
        self.unloadFeaturesAndModelsBUT.setObjectName(_fromUtf8("unloadFeaturesAndModelsBUT"))
        self.horizontalLayout.addWidget(self.unloadFeaturesAndModelsBUT)
        self.defaultPrefsBUT = QtGui.QPushButton(editPrefSkel)
        self.defaultPrefsBUT.setObjectName(_fromUtf8("defaultPrefsBUT"))
        self.horizontalLayout.addWidget(self.defaultPrefsBUT)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(editPrefSkel)
        QtCore.QMetaObject.connectSlotsByName(editPrefSkel)

    def retranslateUi(self, editPrefSkel):
        editPrefSkel.setWindowTitle(QtGui.QApplication.translate("editPrefSkel", "Edit Preferences", None, QtGui.QApplication.UnicodeUTF8))
        self.redrawBUT.setText(QtGui.QApplication.translate("editPrefSkel", "Redraw", None, QtGui.QApplication.UnicodeUTF8))
        self.unloadFeaturesAndModelsBUT.setText(QtGui.QApplication.translate("editPrefSkel", "Unload Features and Models", None, QtGui.QApplication.UnicodeUTF8))
        self.defaultPrefsBUT.setText(QtGui.QApplication.translate("editPrefSkel", "Defaults", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    editPrefSkel = QtGui.QWidget()
    ui = Ui_editPrefSkel()
    ui.setupUi(editPrefSkel)
    editPrefSkel.show()
    sys.exit(app.exec_())

