from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from . import qtype
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[guitbls]', DEBUG=False)


class ColumnListTableView(QtGui.QTableView):
    """ Table View for an AbstractItemModel """
    def __init__(view, *args, **kwargs):
        super(ColumnListTableView, view).__init__(*args, **kwargs)
        view.setSortingEnabled(True)
        view.vertical_header = view.verticalHeader()
        view.vertical_header.setVisible(True)
        #view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        view.resizeColumnsToContents()

    @QtCore.pyqtSlot()
    def cellButtonClicked(self):
        print(self.sender())
        print(self.sender().text())


class ComboDelegate(QtGui.QItemDelegate):
    """
    A delegate that places a fully functioning QComboBox in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        QtGui.QItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        combo = QtGui.QComboBox(parent)
        combo.addItems(['option1', 'option2', 'option3'])
        #self.connect(combo.currentIndexChanged, self.currentIndexChanged)
        # FIXME: Change to newstyle signal slot
        self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
                     self, QtCore.SLOT("currentIndexChanged()"))
        return combo

    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setCurrentIndex(int(index.model().data(index).toString()))
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentIndex())

    @QtCore.pyqtSlot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())


class ButtonDelegate(QtGui.QItemDelegate):
    """
    A delegate that places a fully functioning QPushButton in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        # The parent is not an optional argument for the delegate as
        # we need to reference it in the paint method (see below)
        QtGui.QItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):
        # This method will be called every time a particular cell is
        # in view and that view is changed in some way. We ask the
        # delegates parent (in this case a table view) if the index
        # in question (the table cell) already has a widget associated
        # with it. If not, create one with the text for this index and
        # connect its clicked signal to a slot in the parent view so
        # we are notified when its used and can do something.
        if not self.parent().indexWidget(index):
            self.parent().setIndexWidget(
                index,
                QtGui.QPushButton(
                    index.data().toString(),
                    self.parent(),
                    clicked=self.parent().cellButtonClicked
                )
            )


class ColumnListItemModel(QtCore.QAbstractTableModel):
    """ Item model for displaying a list of columns """

    #
    # Non-Qt Init Functions

    def __init__(model, column_list=None, header_list=None, niceheader_list=None,
                 coltype_list=None, editable_headers=None,
                 display_indicies=False, sortby=None,
                 parent=None, *args):
        super(ColumnListItemModel, model).__init__()
        model.sortcolumn = None
        model.sortreverse = False
        model.display_indicies = False  # FIXME: Broken
        model._change_data(column_list, header_list, niceheader_list,
                           coltype_list, editable_headers, display_indicies,
                           sortby)

    def _change_data(model, column_list=None, header_list=None,
                     niceheader_list=None, coltype_list=None,
                     editable_headers=None, display_indicies=False,
                     sortby=None):
        model.layoutAboutToBeChanged.emit()
        if header_list is None:
            header_list = []
        if column_list is None:
            column_list = []
        if len(column_list) > 0:
            model.sortcolumn = 0
        #print('[model] Changing data')
        # Set the data to display (list of lists)
        model._change_columns(column_list, coltype_list)
        # Set the headers
        model._change_headers(header_list, niceheader_list)
        # Is editable
        model._change_editable(editable_headers)
        # Make internal indicies
        model.set_sorting(sortby)
        model.display_indicies = display_indicies
        model._change_row_indicies()
        # Make sure the user didn't do anything bad
        model._assert_feasibility()
        model.layoutChanged.emit()

    def _change_editable(model, editable_headers=None):
        if editable_headers is None:
            model.column_editable = [False] * len(model.column_list)
        else:
            model.column_editable = [header in editable_headers for header in
                                     model.header_list]
        #print('[model] new column_editable = %r' % (model.column_editable,))

    def _change_headers(model, header_list, niceheader_list=None):
        """ Internal header names """
        model.header_list = header_list
        # Set user readable "nice" headers, default to internal ones
        if niceheader_list is None:
            model.niceheader_list = header_list
        else:
            model.niceheader_list = niceheader_list
        #print('[model] new header_list = %r' % (model.header_list,))

    def _change_columns(model, column_list, coltype_list=None):
        model.column_list = column_list
        if coltype_list is not None:
            model.coltype_list = coltype_list
        else:
            model.coltype_list = qtype.infer_coltype(model.column_list)
        #print('[model] new coltype_list = %r' % (model.coltype_list,))

    def _change_row_indicies(model):
        """  Non-Qt Helper """
        if model.sortcolumn is not None:
            print('using: sortcolumn=%r' % model.sortcolumn)
            column_data = model.column_list[model.sortcolumn]
            indicies = range(len(column_data))
            model.row_sortx = utool.sortedby(indicies, column_data,
                                             reverse=model.sortreverse)
        elif len(model.column_list) > 0:
            model.row_sortx = range(len(model.column_list[0]))
        else:
            model.row_sortx = []
        #print('[model] new len(row_sortx) = %r' % (len(model.row_sortx),))

    def _assert_feasibility(model):
        assert len(model.header_list) == len(model.column_list)
        nrows = map(len, model.column_list)
        assert all([nrows[0] == num for num in nrows]), 'inconsistent data'
        #print('[model] is feasible')

    #
    #
    # Non-Qt Helper function

    def get_data(model, index):
        """ Non-Qt Helper """
        row = index.row()
        column = index.column()
        row_data = model.column_list[column]
        data = row_data[model.row_sortx[row]]
        return data

    def set_sorting(model, sortby=None, order=Qt.DescendingOrder):
        model.sortreverse = (order == Qt.DescendingOrder)
        if sortby is not None:
            if utool.is_str(sortby):
                sortby = model.header_list.index(sortby)
            assert utool.is_int(sortby), 'sort by an index not %r' % type(sortby)
            model.sortcolumn = sortby
            assert model.sortcolumn < len(model.header_list), 'outofbounds'
            print('sortcolumn: %r' % model.sortcolumn)

    def set_data(model, index, data):
        """ Non-Qt Helper """
        row = index.row()
        column = index.column()
        row_data = model.column_list[column]
        row_data[model.row_sortx[row]] = data

    def get_header(model, column):
        """ Non-Qt Helper """
        return model.header_list[column]

    def get_header_data(model, header, row):
        """ Non-Qt Helper """
        column = model.header_list.index(header)
        index  = model.index(row, column)
        data   = model.get_data(index)
        return data

    def get_coltype(model, column):
        type_ = model.coltype_list[column]
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
        return len(model.row_sortx)

    def columnCount(model, parent=QtCore.QModelIndex()):
        return len(model.header_list)

    def index(model, row, column, parent=QtCore.QModelIndex()):
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
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return model.get_niceheader(section)
        else:
            return QtCore.QVariant()

    def sort(model, column, order):
        model.layoutAboutToBeChanged.emit()
        model.set_sorting(column, order)
        model._change_row_indicies()
        model.layoutChanged.emit()

    def flags(model, index):
        #return Qt.ItemFlag(0)
        column = index.column()
        if not model.get_editable(column):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if model.get_coltype(column) in utool.VALID_BOOL_TYPES:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable


class ColumnListTableWidget(QtGui.QWidget):
    """ ColumnList Table Main Widget """
    def __init__(cltw, column_list=None, header_list=None,
                 niceheader_list=None, coltype_list=None,
                 editable_headers=None, display_indicies=False,
                 sortby=None, parent=None):
        QtGui.QWidget.__init__(cltw, parent)
        # Create vertical layout for the table to go into
        cltw.vert_layout = QtGui.QVBoxLayout(cltw)
        # Instansiate the AbstractItemModel
        cltw.model = ColumnListItemModel(parent=cltw)
        # Create a ColumnListTableView for the AbstractItemModel
        cltw.view = ColumnListTableView(cltw)
        cltw.view.setModel(cltw.model)
        cltw.vert_layout.addWidget(cltw.view)
        # Make sure we don't call a childs method
        ColumnListTableWidget.change_data(cltw, column_list, header_list,
                                          niceheader_list, coltype_list,
                                          editable_headers, display_indicies,
                                          sortby)

    def change_data(cltw, column_list=None, header_list=None,
                    niceheader_list=None, coltype_list=None,
                    editable_headers=None, display_indicies=False,
                    sortby=None):
        """
        Checks for deligates
        """
        marked_columns = []  # these will be persistantly editable
        if coltype_list is not None:
            print('cltw.change_data: %r' % len(coltype_list))
            coltype_list = list(coltype_list)
            for column in xrange(len(coltype_list)):
                if isinstance(coltype_list[column], tuple):
                    delegate_type, coltype = coltype_list[column]
                    coltype_list[column] = coltype
                    if cltw.set_column_as_delegate(column, delegate_type):
                        marked_columns.append(column)
        else:
            print('cltw.change_data: None')
        cltw.model._change_data(column_list, header_list, niceheader_list,
                                coltype_list, editable_headers,
                                display_indicies, sortby)
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
        for row in xrange(num_rows):
            index  = cltw.model.index(row, column)
            cltw.view.openPersistentEditor(index)

    def is_index_clickable(cltw, index):
        model = index.model()
        clickable = not (model.flags(index) & QtCore.Qt.ItemIsSelectable)
        return clickable

    def get_index_header_data(cltw, header, index):
        model = index.model()
        return model.get_header_data(header, index.row())


def make_listtable_widget(column_list, header_list, editable_headers=None,
                          show=True, raise_=True, on_click=None):
    widget = ColumnListTableWidget(column_list, header_list,
                                   editable_headers=editable_headers)
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
    #app = QtGui.QApplication(sys.argv)
    #widget = DummyWidget()
    #widget.show()
    #widget.raise_()
    #sys.exit(app.exec_())
