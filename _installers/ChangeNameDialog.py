# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/joncrall/code/hotspotter/hsgui/_frontend/ChangeNameDialog.ui'
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

class Ui_changeNameDialog(object):
    def setupUi(self, changeNameDialog):
        changeNameDialog.setObjectName(_fromUtf8("changeNameDialog"))
        changeNameDialog.resize(441, 109)
        self.verticalLayout = QtGui.QVBoxLayout(changeNameDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(changeNameDialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.label_2 = QtGui.QLabel(changeNameDialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.newNameEdit = QtGui.QLineEdit(changeNameDialog)
        self.newNameEdit.setObjectName(_fromUtf8("newNameEdit"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.newNameEdit)
        self.oldNameEdit = QtGui.QLineEdit(changeNameDialog)
        self.oldNameEdit.setObjectName(_fromUtf8("oldNameEdit"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.oldNameEdit)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtGui.QDialogButtonBox(changeNameDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(changeNameDialog)
        QtCore.QMetaObject.connectSlotsByName(changeNameDialog)

    def retranslateUi(self, changeNameDialog):
        changeNameDialog.setWindowTitle(QtGui.QApplication.translate("changeNameDialog", "Change Name Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("changeNameDialog", "Change all names matching:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("changeNameDialog", "To the new name:", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    changeNameDialog = QtGui.QDialog()
    ui = Ui_changeNameDialog()
    ui.setupUi(changeNameDialog)
    changeNameDialog.show()
    sys.exit(app.exec_())

