from __future__ import absolute_import, division, print_function
from guitool.api_table_model import APITableModel
from guitool.api_table_view import APITableView
#from guitool import guitool_components as comp
from PyQt4 import QtGui
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableWidget]', DEBUG=False)


WIDGET_BASE = QtGui.QWidget


class APITableWidget(WIDGET_BASE):
    """ SIMPLE WIDGET WHICH AUTOCREATES MODEL AND VIEW FOR YOU.
    TODO: Depricate this except for as an example.
    """

    def __init__(widget, headers=None, parent=None,
                 model_class=APITableModel,
                 view_class=APITableView):
        WIDGET_BASE.__init__(widget, parent)
        # Create vertical layout for the table to go into
        widget.vert_layout = QtGui.QVBoxLayout(widget)
        # Create a ColumnListTableView for the AbstractItemModel
        widget.view = view_class(parent=widget)
        # Instansiate the AbstractItemModel
        # FIXME: It is very bad to give the model a view. Only the view should have a model
        widget.model = model_class(parent=widget.view)
        widget.view.setModel(widget.model)
        widget.vert_layout.addWidget(widget.view)
        if headers is not None:
            # Make sure we don't call a subclass method
            APITableWidget.change_headers(widget, headers)

    def change_headers(widget, headers):
        parent = widget.parent()
        # Update headers of both model and view
        widget.model._update_headers(**headers)
        widget.view._update_headers(**headers)
        if parent is None:
            nice = headers.get('nice', 'NO NICE NAME')
            widget.setWindowTitle(nice)

    def connect_signals(widget):
        widget.model._rows_updated.connect(widget.on_rows_updated)

    def on_rows_updated(widget, name, num):
        print('rows updated')
        pass
