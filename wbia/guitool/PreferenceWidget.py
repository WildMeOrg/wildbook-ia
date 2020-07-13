# -*- coding: utf-8 -*-
"""
old code ported from utool
"""
from __future__ import absolute_import, division, print_function
import sys
import six
import traceback
from utool.Preferences import Pref, PrefNode, PrefChoice
from wbia.guitool.__PYQT__ import _fromUtf8, _translate, QVariantHack
from wbia.guitool.__PYQT__ import QtWidgets, QtCore
from wbia.guitool.__PYQT__.QtCore import Qt
from wbia.guitool import qtype
import utool as ut
from utool import util_type

ut.noinject(__name__, '[PreferenceWidget]', DEBUG=False)

VERBOSE_PREF = ut.get_argflag('--verbpref')


def report_thread_error(fn):
    """ Decorator to help catch errors that QT wont report """

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


def _qt_set_leaf_data(self, qvar):
    """ Sets backend data using QVariants """
    if VERBOSE_PREF:
        print('')
        print('+--- [pref.qt_set_leaf_data]')
        print('[pref.qt_set_leaf_data] qvar = %r' % qvar)
        print('[pref.qt_set_leaf_data] _intern.name=%r' % self._intern.name)
        print('[pref.qt_set_leaf_data] _intern.type_=%r' % self._intern.get_type())
        print('[pref.qt_set_leaf_data] type(_intern.value)=%r' % type(self._intern.value))
        print('[pref.qt_set_leaf_data] _intern.value=%r' % self._intern.value)
        # print('[pref.qt_set_leaf_data] qvar.toString()=%s' % str(qvar.toString()))
    if self._tree.parent is None:
        raise Exception('[Pref.qtleaf] Cannot set root preference')
    if self.qt_is_editable():
        new_val = '[Pref.qtleaf] BadThingsHappenedInPref'
        if self._intern.value == PrefNode:
            raise Exception('[Pref.qtleaf] Qt can only change leafs')
        elif self._intern.value is None:
            # None could be a number of types
            def cast_order(var, order=[bool, int, float, str]):
                for type_ in order:
                    try:
                        ret = type_(var)
                        return ret
                    except Exception:
                        continue

            new_val = cast_order(str(qvar))
        self._intern.get_type()
        if isinstance(self._intern.value, bool):
            # new_val = bool(qvar.toBool())
            print('qvar = %r' % (qvar,))
            new_val = util_type.smart_cast(qvar, bool)
            # new_val = bool(eval(qvar, {}, {}))
            print('new_val = %r' % (new_val,))
        elif isinstance(self._intern.value, int):
            # new_val = int(qvar.toInt()[0])
            new_val = int(qvar)
        # elif isinstance(self._intern.value, float):
        elif self._intern.get_type() in util_type.VALID_FLOAT_TYPES:
            # new_val = float(qvar.toDouble()[0])
            new_val = float(qvar)
        elif isinstance(self._intern.value, six.string_types):
            # new_val = str(qvar.toString())
            new_val = str(qvar)
        elif isinstance(self._intern.value, PrefChoice):
            # new_val = qvar.toString()
            new_val = str(qvar)
            if new_val.upper() == 'NONE':
                new_val = None
        else:
            try:
                # new_val = str(qvar.toString())
                type_ = self._intern.get_type()
                if type_ is not None:
                    new_val = type_(str(qvar))
                else:
                    new_val = str(qvar)
            except Exception:
                raise NotImplementedError(
                    (
                        '[Pref.qtleaf] Unknown internal type. '
                        'type(_intern.value) = %r, '
                        '_intern.get_type() = %r, '
                    )
                    % type(self._intern.value),
                    self._intern.get_type(),
                )
        # Check for a set of None
        if isinstance(new_val, six.string_types):
            if new_val.lower() == 'none':
                new_val = None
            elif new_val.lower() == 'true':
                new_val = True
            elif new_val.lower() == 'false':
                new_val = False
        # save to disk after modifying data
        if VERBOSE_PREF:
            print('---')
            print('[pref.qt_set_leaf_data] new_val=%r' % new_val)
            print('[pref.qt_set_leaf_data] type(new_val)=%r' % type(new_val))
            print('L____ [pref.qt_set_leaf_data]')
        # TODO Add ability to set a callback function when certain
        # preferences are changed.
        return self._tree.parent.pref_update(self._intern.name, new_val)
    return 'PrefNotEditable'


class QPreferenceModel(QtCore.QAbstractItemModel):
    """ Convention states only items with column index 0 can have children """

    @report_thread_error
    def __init__(self, pref_struct, parent=None):
        super(QPreferenceModel, self).__init__(parent)
        self.rootPref = pref_struct

    @report_thread_error
    def index2Pref(self, index=QtCore.QModelIndex()):
        """ Internal helper method """
        if index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return self.rootPref

    # -----------
    # Overloaded ItemModel Read Functions
    @report_thread_error
    def rowCount(self, parent=QtCore.QModelIndex()):
        parentPref = self.index2Pref(parent)
        return parentPref.qt_row_count()

    @report_thread_error
    def columnCount(self, parent=QtCore.QModelIndex()):
        parentPref = self.index2Pref(parent)
        return parentPref.qt_col_count()

    @report_thread_error
    def data(self, qtindex, role=Qt.DisplayRole):
        """ Returns the data stored under the given role
        for the item referred to by the qtindex. """
        if not qtindex.isValid():
            return QVariantHack()
        # Specify CheckState Role:
        flags = self.flags(qtindex)
        if role == Qt.CheckStateRole:
            if flags & Qt.ItemIsUserCheckable:
                data = self.index2Pref(qtindex).qt_get_data(qtindex.column())
                return Qt.Checked if data else Qt.Unchecked
        if role != Qt.DisplayRole and role != Qt.EditRole:
            return QVariantHack()
        nodePref = self.index2Pref(qtindex)
        data = nodePref.qt_get_data(qtindex.column())
        # print('--- data() ---')
        # print('role = %r' % role)
        # print('data = %r' % data)
        # print('type(data) = %r' % type(data))
        # <SIP.API_MODE(1)>
        # var = QVariantHack(data)
        # if isinstance(data, float):
        #    var = QVariantHack(QString.number(data, format='g', precision=6))
        # if isinstance(data, bool):
        #    var = QVariantHack(data).toString()
        # if isinstance(data, int):
        #    var = QVariantHack(data).toString()
        # </SIP.API_MODE(1)>
        # <SIP.API_MODE(2)>
        if isinstance(data, float):
            var = qtype.locale_float(data, 6)
        else:
            var = data
        # </SIP.API_MODE(2)>
        # print('var= %r' % var)
        # print('type(var)= %r' % type(var))
        return str(var)

    @report_thread_error
    def setData(self, qtindex, value, role=Qt.EditRole):
        """Sets the role data for the item at qtindex to value."""
        if role == Qt.EditRole:
            data = value
        elif role == Qt.CheckStateRole:
            data = value == Qt.Checked
        else:
            return False

        if VERBOSE_PREF:
            print('[qt] --- setData() ---')
            print('[qt] role = %r' % role)
            print('[qt] value = %r' % value)
            print('[qt] type(data) = %r' % type(data))
            print('[qt] type(value) = %r' % type(value))

        leafPref = self.index2Pref(qtindex)
        old_data = leafPref.qt_get_data(qtindex.column())
        if VERBOSE_PREF:
            print('[qt] old_data = %r' % (old_data,))
            print('[qt] old_data != data = %r' % (old_data != data,))
        if old_data != data:
            result = _qt_set_leaf_data(leafPref, data)
            if VERBOSE_PREF:
                print('[qt] result = %r' % (result,))
            # if result is True:
            # return result
        self.dataChanged.emit(qtindex, qtindex)
        return True

    @report_thread_error
    def index(self, row, col, parent=QtCore.QModelIndex()):
        """Returns the index of the item in the model specified
        by the given row, column and parent index."""
        if parent.isValid() and parent.column() != 0:
            return QtCore.QModelIndex()
        parentPref = self.index2Pref(parent)
        childPref = parentPref.qt_get_child(row)
        if childPref:
            return self.createIndex(row, col, childPref)
        else:
            return QtCore.QModelIndex()

    @report_thread_error
    def parent(self, index=None):
        """Returns the parent of the model item with the given index.
        If the item has no parent, an invalid QModelIndex is returned."""
        if index is None:  # Overload with QtCore.QObject.parent()
            return QtCore.QObject.parent(self)
        if not index.isValid():
            return QtCore.QModelIndex()
        nodePref = self.index2Pref(index)
        parentPref = nodePref.qt_get_parent()
        if parentPref == self.rootPref:
            return QtCore.QModelIndex()
        return self.createIndex(parentPref.qt_parents_index_of_me(), 0, parentPref)

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
                if childPref._intern.get_type() is bool:
                    # print(childPref)
                    # print(childPref._intern.get_type())
                    # print(childPref._intern.get_type() is bool)
                    return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
                else:
                    return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemFlag(0)

    @report_thread_error
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return QVariantHack('Config Key')
            if section == 1:
                return QVariantHack('Config Value')
        return QVariantHack()


class Ui_editPrefSkel(object):
    """
    THE PREFERENCE SKELETON
    """

    def setupUi(self, editPrefSkel):
        editPrefSkel.setObjectName(_fromUtf8('editPrefSkel'))
        editPrefSkel.resize(668, 530)
        # Add Pane for TreeView
        self.verticalLayout = QtWidgets.QVBoxLayout(editPrefSkel)
        self.verticalLayout.setObjectName(_fromUtf8('verticalLayout'))
        # The TreeView for QtCore.QAbstractItemModel to attach to
        self.prefTreeView = QtWidgets.QTreeView(editPrefSkel)
        self.prefTreeView.setObjectName(_fromUtf8('prefTreeView'))
        self.verticalLayout.addWidget(self.prefTreeView)
        # Add Pane for buttons
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8('horizontalLayout'))
        #
        # self.redrawBUT = QtWidgets.QPushButton(editPrefSkel)
        # self.redrawBUT.setObjectName(_fromUtf8('redrawBUT'))
        # self.horizontalLayout.addWidget(self.redrawBUT)
        ##
        # self.unloadFeaturesAndModelsBUT = QtWidgets.QPushButton(editPrefSkel)
        # self.unloadFeaturesAndModelsBUT.setObjectName(_fromUtf8('unloadFeaturesAndModelsBUT'))
        # self.horizontalLayout.addWidget(self.unloadFeaturesAndModelsBUT)
        #
        self.defaultPrefsBUT = QtWidgets.QPushButton(editPrefSkel)
        self.defaultPrefsBUT.setObjectName(_fromUtf8('defaultPrefsBUT'))
        self.horizontalLayout.addWidget(self.defaultPrefsBUT)
        # Buttons are a child of the View
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.retranslateUi(editPrefSkel)
        QtCore.QMetaObject.connectSlotsByName(editPrefSkel)

    def retranslateUi(self, editPrefSkel):
        # UTF-8 Support
        editPrefSkel.setWindowTitle(_translate('editPrefSkel', 'Edit Preferences', None))
        # self.redrawBUT.setText(_translate('editPrefSkel', 'Redraw', None))
        # self.unloadFeaturesAndModelsBUT.setText(_translate('editPrefSkel', 'Unload Features and Models', None))
        self.defaultPrefsBUT.setText(_translate('editPrefSkel', 'Defaults', None))


# ---
# THE PREFERENCE WIDGET
# ---
class EditPrefWidget(QtWidgets.QWidget):
    """The Settings Pane; Subclass of Main Windows."""

    def __init__(self, pref_struct):
        super(EditPrefWidget, self).__init__()
        self.ui = Ui_editPrefSkel()
        self.ui.setupUi(self)
        self.pref_model = None
        self.populatePrefTreeSlot(pref_struct)
        # self.ui.redrawBUT.clicked.connect(fac.redraw)
        # self.ui.defaultPrefsBUT.clicked.connect(fac.default_prefs)
        # self.ui.unloadFeaturesAndModelsBUT.clicked.connect(fac.unload_features_and_models)

    @QtCore.pyqtSlot(Pref, name='populatePrefTreeSlot')
    def populatePrefTreeSlot(self, pref_struct):
        """Populates the Preference Tree Model"""
        # printDBG('Bulding Preference Model of: ' + repr(pref_struct))
        # Creates a QStandardItemModel that you can connect to a QTreeView
        self.pref_model = QPreferenceModel(pref_struct)
        # printDBG('Built: ' + repr(self.pref_model))
        self.ui.prefTreeView.setModel(self.pref_model)
        self.ui.prefTreeView.header().resizeSection(0, 250)

    def refresh_layout(self):
        self.pref_model.layoutAboutToBeChanged.emit()
        self.pref_model.layoutChanged.emit()


def test_preference_gui():
    r"""
    CommandLine:
        python -m wbia.guitool.PreferenceWidget --exec-test_preference_gui --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.guitool.PreferenceWidget import *  # NOQA
        >>> import wbia.guitool
        >>> guitool.ensure_qtapp()
        >>> scope = test_preference_gui()
        >>> ut.quit_if_noshow()
        >>> guitool.qtapp_loop(freq=10)
    """
    from wbia import dtool

    class DtoolConfig(dtool.Config):
        _param_info_list = [
            ut.ParamInfo('str_option2', 'hello'),
            ut.ParamInfo('int_option2', 456),
            ut.ParamInfo('bool_option2', True),
        ]

    class OldPrefConfig(ut.Pref):
        def __init__(self):
            super(OldPrefConfig, self).__init__()
            self.str_option = 'goodbye'
            self.int_option = 123
            self.bool_option = False
            self.float_option = 3.2
            self.listvar = ['one', 'two', 'three']

    old = OldPrefConfig()
    new = DtoolConfig()
    new_wrap = ut.Pref()
    for k, v in new.items():
        setattr(new_wrap, k, v)
    old.new = new_wrap
    epw = old.createQWidget()
    from wbia.plottool import fig_presenter

    fig_presenter.register_qt4_win(epw)
    # epw.ui.defaultPrefsBUT.clicked.connect(back.default_config)
    epw.show()
    return old, epw


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.guitool.PreferenceWidget
        python -m wbia.guitool.PreferenceWidget --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
