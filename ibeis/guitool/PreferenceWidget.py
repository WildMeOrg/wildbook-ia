"""
old code ported from utool
"""
from __future__ import absolute_import, division, print_function
import sys
import traceback
from utool.Preferences import Pref
# Qt
from .__PYQT__ import QtCore, QtGui
from .__PYQT__.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, pyqtSlot
from .__PYQT__.QtGui import QWidget
from .__PYQT__.QtCore import QString, QVariant


# Decorator to help catch errors that QT wont report
def report_thread_error(fn):
    def report_thread_error_wrapper(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
            return ret
        except Exception as ex:
            print('\n\n *!!* Thread Raised Exception: ' + str(ex))
            print('\n\n *!!* Thread Exception Traceback: \n\n' + traceback.format_exc())
            sys.stdout.flush()
            et, ei, tb = sys.exc_info()
            raise
    return report_thread_error_wrapper

# ---
# Functions
# ---
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s
try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


# ---
# THE ABSTRACT ITEM MODEL
# ---
#QComboBox
class QPreferenceModel(QAbstractItemModel):
    """ Convention states only items with column index 0 can have children """
    @report_thread_error
    def __init__(self, pref_struct, parent=None):
        super(QPreferenceModel, self).__init__(parent)
        self.rootPref  = pref_struct

    @report_thread_error
    def index2Pref(self, index=QModelIndex()):
        """ Internal helper method """
        if index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return self.rootPref

    #-----------
    # Overloaded ItemModel Read Functions
    @report_thread_error
    def rowCount(self, parent=QModelIndex()):
        parentPref = self.index2Pref(parent)
        return parentPref.qt_row_count()

    @report_thread_error
    def columnCount(self, parent=QModelIndex()):
        parentPref = self.index2Pref(parent)
        return parentPref.qt_col_count()

    @report_thread_error
    def data(self, index, role=Qt.DisplayRole):
        """ Returns the data stored under the given role
        for the item referred to by the index. """
        if not index.isValid():
            return QVariant()
        if role != Qt.DisplayRole and role != Qt.EditRole:
            return QVariant()
        nodePref = self.index2Pref(index)
        data = nodePref.qt_get_data(index.column())
        var = QVariant(data)
        #print('--- data() ---')
        #print('role = %r' % role)
        #print('data = %r' % data)
        #print('type(data) = %r' % type(data))
        if isinstance(data, float):
            var = QVariant(QString.number(data, format='g', precision=6))
        if isinstance(data, bool):
            var = QVariant(data).toString()
        if isinstance(data, int):
            var = QVariant(data).toString()
        #print('var= %r' % var)
        #print('type(var)= %r' % type(var))
        return var

    @report_thread_error
    def index(self, row, col, parent=QModelIndex()):
        """Returns the index of the item in the model specified
        by the given row, column and parent index."""
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()
        parentPref = self.index2Pref(parent)
        childPref  = parentPref.qt_get_child(row)
        if childPref:
            return self.createIndex(row, col, childPref)
        else:
            return QModelIndex()

    @report_thread_error
    def parent(self, index=None):
        """Returns the parent of the model item with the given index.
        If the item has no parent, an invalid QModelIndex is returned."""
        if index is None:  # Overload with QObject.parent()
            return QObject.parent(self)
        if not index.isValid():
            return QModelIndex()
        nodePref = self.index2Pref(index)
        parentPref = nodePref.qt_get_parent()
        if parentPref == self.rootPref:
            return QModelIndex()
        return self.createIndex(parentPref.qt_parents_index_of_me(), 0, parentPref)

    #-----------
    # Overloaded ItemModel Write Functions
    @report_thread_error
    def flags(self, index):
        """Returns the item flags for the given index."""
        if index.column() == 0:
            # The First Column is just a label and unchangable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag(0)
        childPref = self.index2Pref(index)
        if childPref:
            if childPref.qt_is_editable():
                return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemFlag(0)

    @report_thread_error
    def setData(self, index, data, role=Qt.EditRole):
        """Sets the role data for the item at index to value."""
        if role != Qt.EditRole:
            return False
        #print('--- setData() ---')
        #print('role = %r' % role)
        #print('data = %r' % data)
        #print('type(data) = %r' % type(data))
        leafPref = self.index2Pref(index)
        result = leafPref.qt_set_leaf_data(data)
        if result is True:
            self.dataChanged.emit(index, index)
        return result

    @report_thread_error
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return QVariant('Config Key')
            if section == 1:
                return QVariant('Config Value')
        return QVariant()


# ---
# THE PREFERENCE SKELETON
# ---
class Ui_editPrefSkel(object):
    def setupUi(self, editPrefSkel):
        editPrefSkel.setObjectName(_fromUtf8('editPrefSkel'))
        editPrefSkel.resize(668, 530)
        # Add Pane for TreeView
        self.verticalLayout = QtGui.QVBoxLayout(editPrefSkel)
        self.verticalLayout.setObjectName(_fromUtf8('verticalLayout'))
        # The TreeView for QAbstractItemModel to attach to
        self.prefTreeView = QtGui.QTreeView(editPrefSkel)
        self.prefTreeView.setObjectName(_fromUtf8('prefTreeView'))
        self.verticalLayout.addWidget(self.prefTreeView)
        # Add Pane for buttons
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8('horizontalLayout'))
        #
        #self.redrawBUT = QtGui.QPushButton(editPrefSkel)
        #self.redrawBUT.setObjectName(_fromUtf8('redrawBUT'))
        #self.horizontalLayout.addWidget(self.redrawBUT)
        ##
        #self.unloadFeaturesAndModelsBUT = QtGui.QPushButton(editPrefSkel)
        #self.unloadFeaturesAndModelsBUT.setObjectName(_fromUtf8('unloadFeaturesAndModelsBUT'))
        #self.horizontalLayout.addWidget(self.unloadFeaturesAndModelsBUT)
        #
        self.defaultPrefsBUT = QtGui.QPushButton(editPrefSkel)
        self.defaultPrefsBUT.setObjectName(_fromUtf8('defaultPrefsBUT'))
        self.horizontalLayout.addWidget(self.defaultPrefsBUT)
        # Buttons are a child of the View
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.retranslateUi(editPrefSkel)
        QtCore.QMetaObject.connectSlotsByName(editPrefSkel)

    def retranslateUi(self, editPrefSkel):
        # UTF-8 Support
        editPrefSkel.setWindowTitle(_translate('editPrefSkel', 'Edit Preferences', None))
        #self.redrawBUT.setText(_translate('editPrefSkel', 'Redraw', None))
        #self.unloadFeaturesAndModelsBUT.setText(_translate('editPrefSkel', 'Unload Features and Models', None))
        self.defaultPrefsBUT.setText(_translate('editPrefSkel', 'Defaults', None))


# ---
# THE PREFERENCE WIDGET
# ---
class EditPrefWidget(QWidget):
    """The Settings Pane; Subclass of Main Windows."""
    def __init__(self, pref_struct):
        super(EditPrefWidget, self).__init__()
        self.ui = Ui_editPrefSkel()
        self.ui.setupUi(self)
        self.pref_model = None
        self.populatePrefTreeSlot(pref_struct)
        #self.ui.redrawBUT.clicked.connect(fac.redraw)
        #self.ui.defaultPrefsBUT.clicked.connect(fac.default_prefs)
        #self.ui.unloadFeaturesAndModelsBUT.clicked.connect(fac.unload_features_and_models)

    @pyqtSlot(Pref, name='populatePrefTreeSlot')
    def populatePrefTreeSlot(self, pref_struct):
        """Populates the Preference Tree Model"""
        #printDBG('Bulding Preference Model of: ' + repr(pref_struct))
        #Creates a QStandardItemModel that you can connect to a QTreeView
        self.pref_model = QPreferenceModel(pref_struct)
        #printDBG('Built: ' + repr(self.pref_model))
        self.ui.prefTreeView.setModel(self.pref_model)
        self.ui.prefTreeView.header().resizeSection(0, 250)

    def refresh_layout(self):
        self.pref_model.layoutAboutToBeChanged.emit()
        self.pref_model.layoutChanged.emit()
