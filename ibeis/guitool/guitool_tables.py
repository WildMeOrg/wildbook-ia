# DEPRICATE?
from __future__ import absolute_import, division, print_function
from guitool_ibeis.__PYQT__ import QtCore, QtGui
from guitool_ibeis.__PYQT__ import QtWidgets
from guitool_ibeis.__PYQT__.QtCore import Qt
from guitool_ibeis.guitool_delegates import ComboDelegate, ButtonDelegate
from guitool_ibeis import qtype
from six.moves import range, map
import utool
(print, rrr, profile) = utool.inject2(__name__)


class ColumnListTableView(QtWidgets.QTableView):
    """ Table View for an AbstractItemModel """
    def __init__(view, *args, **kwargs):
        super(ColumnListTableView, view).__init__(*args, **kwargs)
        view.setSortingEnabled(True)
        view.vertical_header = view.verticalHeader()
        view.vertical_header.setVisible(True)
        #view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    @QtCore.pyqtSlot()
    def cellButtonClicked(self):
        print(self.sender())
        print(self.sender().text())


class ColumnListItemModel(QtCore.QAbstractTableModel):
    """ Item model for displaying a list of columns """

    #
    # Non-Qt Init Functions

    def __init__(model, col_data_list=None, col_name_list=None, niceheader_list=None,
                 col_type_list=None, col_edit_list=None,
                 display_indices=False, col_sort_index=None,
                 parent=None, *args):
        super(ColumnListItemModel, model).__init__()
        model.sortcolumn = None
        model.sortreverse = False
        model.display_indices = False  # FIXME: Broken
        model._change_data(col_data_list, col_name_list, niceheader_list,
                           col_type_list, col_edit_list, display_indices,
                           col_sort_index)

    def _change_data(model, col_data_list=None, col_name_list=None,
                     niceheader_list=None, col_type_list=None,
                     col_edit_list=None, display_indices=False,
                     col_sort_index=None):
        model.layoutAboutToBeChanged.emit()
        if col_name_list is None:
            col_name_list = []
        if col_data_list is None:
            col_data_list = []
        if len(col_data_list) > 0:
            model.sortcolumn = 0
        #print('[model] Changing data')
        # Set the data to display (list of lists)
        model._change_columns(col_data_list, col_type_list)
        # Set the headers
        model._change_headers(col_name_list, niceheader_list)
        # Is editable
        model._change_editable(col_edit_list)
        # Make internal indices
        model.set_sorting(col_sort_index)
        model.display_indices = display_indices
        model._change_row_indices()
        # Make sure the user didn't do anything bad
        model._assert_feasibility()
        model.layoutChanged.emit()

    def _change_editable(model, col_edit_list=None):
        if col_edit_list is None:
            model.column_editable = [False] * len(model.col_data_list)
        else:
            model.column_editable = [header in col_edit_list for header in
                                     model.col_name_list]
        #print('[model] new column_editable = %r' % (model.column_editable,))

    def _change_headers(model, col_name_list, niceheader_list=None):
        """ Internal header names """
        model.col_name_list = col_name_list
        # Set user readable "nice" headers, default to internal ones
        if niceheader_list is None:
            model.niceheader_list = col_name_list
        else:
            model.niceheader_list = niceheader_list
        #print('[model] new col_name_list = %r' % (model.col_name_list,))

    def _change_columns(model, col_data_list, col_type_list=None):
        model.col_data_list = col_data_list
        if col_type_list is not None:
            model.col_type_list = col_type_list
        else:
            model.col_type_list = qtype.infer_coltype(model.col_data_list)
        #print('[model] new col_type_list = %r' % (model.col_type_list,))

    def _change_row_indices(model):
        """  Non-Qt Helper """
        if model.sortcolumn is not None:
            print('using: sortcolumn=%r' % model.sortcolumn)
            column_data = model.col_data_list[model.sortcolumn]
            indices = list(range(len(column_data)))
            model.row_sortx = utool.sortedby(indices, column_data,
                                             reverse=model.sortreverse)
        elif len(model.col_data_list) > 0:
            model.row_sortx = list(range(len(model.col_data_list[0])))
        else:
            model.row_sortx = []
        #print('[model] new len(row_sortx) = %r' % (len(model.row_sortx),))

    def _assert_feasibility(model):
        assert len(model.col_name_list) == len(model.col_data_list)
        nrows = list(map(len, model.col_data_list))
        assert all([nrows[0] == num for num in nrows]), 'inconsistent data'
        #print('[model] is feasible')

    #
    #
    # Non-Qt Helper function

    def get_data(model, index):
        """ Non-Qt Helper """
        row = index.row()
        column = index.column()
        row_data = model.col_data_list[column]
        data = row_data[model.row_sortx[row]]
        return data

    def set_sorting(model, col_sort_index=None, order=Qt.DescendingOrder):
        model.sortreverse = (order == Qt.DescendingOrder)
        if col_sort_index is not None:
            if utool.is_str(col_sort_index):
                col_sort_index = model.col_name_list.index(col_sort_index)
            assert utool.is_int(col_sort_index), 'sort by an index not %r' % type(col_sort_index)
            model.sortcolumn = col_sort_index
            assert model.sortcolumn < len(model.col_name_list), 'outofbounds'
            print('sortcolumn: %r' % model.sortcolumn)

    def set_data(model, index, data):
        """ Non-Qt Helper """
        row = index.row()
        column = index.column()
        row_data = model.col_data_list[column]
        row_data[model.row_sortx[row]] = data

    def get_header(model, column):
        """ Non-Qt Helper """
        return model.col_name_list[column]

    def get_header_data(model, header, row):
        """ Non-Qt Helper """
        column = model.col_name_list.index(header)
        index  = model.index(row, column)
        data   = model.get_data(index)
        return data

    def get_coltype(model, column):
        type_ = model.col_type_list[column]
        return type_

    def get_editable(model, column):
        return model.column_editable[column]

    def get_niceheader(model, column):
        """ Non-Qt Helper """
        return model.niceheader_list[column]

    def get_column_alignment(model, column):
        coltype = model.get_coltype(column)
        if coltype in utool.VALID_FLOAT_TYPES:
            return Qt.AlignRight
        else:
            return Qt.AlignHCenter

    #
    #
    # Qt AbstractItemTable Overrides

    def rowCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        return len(model.row_sortx)

    def columnCount(model, parent=QtCore.QModelIndex()):
        """ Qt Override """
        return len(model.col_name_list)

    def index(model, row, column, parent=QtCore.QModelIndex()):
        """ Qt Override """
        return model.createIndex(row, column)

    def data(model, index, role=Qt.DisplayRole):
        """ Returns the data to display """
        if not index.isValid():
            return None
        flags = model.flags(index)
        if role == Qt.TextAlignmentRole:
            return model.get_column_alignment(index.column())
        if role == Qt.BackgroundRole and (flags & Qt.ItemIsEditable or
                                          flags & Qt.ItemIsUserCheckable):
            return QtCore.QVariant(QtGui.QColor(250, 240, 240))
        if role == Qt.DisplayRole or role == Qt.CheckStateRole:
            data = model.get_data(index)
            var = qtype.cast_into_qt(data, role, flags)
            return var
        else:
            return QtCore.QVariant()

    def setData(model, index, var, role=Qt.EditRole):
        """ Sets the role data for the item at index to var.
        var is a QVariant (called data in documentation)
        """
        print('[model] setData: %r' % (str(qtype.qindexinfo(index))))
        try:
            if not index.isValid():
                return None
            flags = model.flags(index)
            if not (flags & Qt.ItemIsEditable or flags & Qt.ItemIsUserCheckable):
                return None
            if role == Qt.CheckStateRole:
                type_ = 'QtCheckState'
                data = var == Qt.Checked
            elif role != Qt.EditRole:
                return False
            else:
                # Cast var into datatype
                type_ = model.get_coltype(index.column())
                data = qtype.cast_from_qt(var, type_)
            # Do actual setting of data
            print(' * new_data = %s(%r)' % (utool.type_str(type_), data,))
            model.set_data(index, data)
            # Emit that data was changed and return succcess
            model.dataChanged.emit(index, index)
            return True
        except Exception as ex:
            var_ = str(var.toString())  # NOQA
            utool.printex(ex, 'ignoring setData', '[model]',
                          key_list=['var_'])
            #raise
            #print(' * ignoring setData: %r' % locals().get('var', None))
            return False

    def headerData(model, section, orientation, role=Qt.DisplayRole):
        """ Qt Override """
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return model.get_niceheader(section)
        else:
            return QtCore.QVariant()

    def sort(model, column, order):
        """ Qt Override """
        model.layoutAboutToBeChanged.emit()
        model.set_sorting(column, order)
        model._change_row_indices()
        model.layoutChanged.emit()

    def flags(model, index):
        """ Qt Override """
        #return Qt.ItemFlag(0)
        column = index.column()
        if not model.get_editable(column):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if model.get_coltype(column) in utool.VALID_BOOL_TYPES:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable


class ColumnListTableWidget(QtWidgets.QWidget):
    """ ColumnList Table Main Widget """
    def __init__(cltw, col_data_list=None, col_name_list=None,
                 niceheader_list=None, col_type_list=None,
                 col_edit_list=None, display_indices=False,
                 col_sort_index=None, parent=None):
        super(ColumnListTableWidget, cltw).__init__(parent)
        # QtWidgets.QWidget.__init__(cltw, parent)
        # Create vertical layout for the table to go into
        cltw.vert_layout = QtWidgets.QVBoxLayout(cltw)
        # Instansiate the AbstractItemModel
        cltw.model = ColumnListItemModel(parent=cltw)
        # Create a ColumnListTableView for the AbstractItemModel
        cltw.view = ColumnListTableView(cltw)
        cltw.view.setModel(cltw.model)
        cltw.vert_layout.addWidget(cltw.view)
        # Make sure we don't call a childs method
        ColumnListTableWidget.change_data(cltw, col_data_list, col_name_list,
                                          niceheader_list, col_type_list,
                                          col_edit_list, display_indices,
                                          col_sort_index)

    def change_data(cltw, col_data_list=None, col_name_list=None,
                    niceheader_list=None, col_type_list=None,
                    col_edit_list=None, display_indices=False,
                    col_sort_index=None):
        """
        Checks for deligates
        """
        marked_columns = []  # these will be persistantly editable
        if col_type_list is not None:
            print('cltw.change_data: %r' % len(col_type_list))
            col_type_list = list(col_type_list)
            for column in range(len(col_type_list)):
                if isinstance(col_type_list[column], tuple):
                    delegate_type, coltype = col_type_list[column]
                    col_type_list[column] = coltype
                    if cltw.set_column_as_delegate(column, delegate_type):
                        marked_columns.append(column)
        else:
            print('cltw.change_data: None')
        cltw.model._change_data(col_data_list, col_name_list, niceheader_list,
                                col_type_list, col_edit_list,
                                display_indices, col_sort_index)
        # Set persistant editability after data is changed
        for column in marked_columns:
            cltw.set_column_persistant_editor(column)

    def set_column_as_delegate(cltw, column, delegate_type):
        if delegate_type == 'COMBO':
            print('cltw.set_col_del %r %r' % (column, delegate_type))
            cltw.view.setItemDelegateForColumn(column, ComboDelegate(cltw.view))
            return True
        elif delegate_type == 'BUTTON':
            print('cltw.set_col_del %r %r' % (column, delegate_type))
            cltw.view.setItemDelegateForColumn(column, ButtonDelegate(cltw.view))
            return False

    def set_column_persistant_editor(cltw, column):
        """
        Set each row in a column as persistant
        """
        num_rows = cltw.model.rowCount()
        print('cltw.set_persistant: %r rows' % num_rows)
        for row in range(num_rows):
            index  = cltw.model.index(row, column)
            cltw.view.openPersistentEditor(index)

    def is_index_clickable(cltw, index):
        model = index.model()
        clickable = not (model.flags(index) & QtCore.Qt.ItemIsSelectable)
        return clickable

    def get_index_header_data(cltw, header, index):
        model = index.model()
        return model.get_header_data(header, index.row())


def make_listtable_widget(col_data_list, col_name_list, col_edit_list=None,
                          show=True, raise_=True, on_click=None):
    widget = ColumnListTableWidget(col_data_list, col_name_list,
                                   col_edit_list=col_edit_list)

    def on_doubleclick(index):
        # This is actually a release
        #print('DoubleClicked: ' + str(qtype.qindexinfo(index)))
        pass

    def on_pressed(index):
        #print('Pressed: ' + str(qtype.qindexinfo(index)))
        pass

    def on_activated(index):
        #print('Activated: ' + str(qtype.qindexinfo(index)))
        pass

    if on_click is not None:
        widget.view.clicked.connect(on_click)
    widget.view.doubleClicked.connect(on_doubleclick)
    widget.view.pressed.connect(on_doubleclick)
    widget.view.activated.connect(on_activated)
    widget.setGeometry(20, 50, 600, 800)

    if show:
        widget.show()
    if raise_:
        widget.raise_()
    return widget

#if __name__ == '__main__':
    #import sys
    #app = guitoo.ensure_qtapp()
    #widget = DummyWidget()
    #widget.show()
    #widget.raise_()
    #sys.exit(app.exec_())


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m guitool_ibeis.guitool_tables
        python -m guitool_ibeis.guitool_tables --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
