from guitool.api_table_model import APITableModel
from guitool.api_table_view import APITableView
#from guitool import guitool_components as comp
from PyQt4 import QtGui


class APITableWidget(QtGui.QWidget):
    def __init__(widget, headers=None, parent=None):
        QtGui.QWidget.__init__(widget, parent)
        # Create vertical layout for the table to go into
        widget.vert_layout = QtGui.QVBoxLayout(widget)
        # Instansiate the AbstractItemModel
        widget.model = APITableModel(parent=widget)
        # Create a ColumnListTableView for the AbstractItemModel
        widget.view = APITableView(widget)
        widget.view.setModel(widget.model)
        widget.vert_layout.addWidget(widget.view)
        # Make sure we don't call a childs method
        APITableWidget.change_headers(widget, headers)

    def change_headers(widget, headers=None):
        if headers is not None:
            parent = widget.parent()
            widget.model._update_headers(**headers)
            widget.view._update_headers(**headers)
            if parent is None:
                nice = headers.get('nice', 'NO NICE NAME')
                widget.setWindowTitle(nice)

    def connect_signals(widget):
        widget.model._rows_updated.connect(widget.on_rows_updated)

    def on_rows_updated(widget, name, num):
        pass
