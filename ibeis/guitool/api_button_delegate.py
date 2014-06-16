from __future__ import absolute_import, division, print_function
from PyQt4 import QtGui, QtCore
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableWidget]', DEBUG=False)


DELEGATE_BASE = QtGui.QItemDelegate


class APIButtonDelegate(DELEGATE_BASE):
    def __init__(dgt, parent=None):
        DELEGATE_BASE.__init__(dgt, parent)

    def get_model_data(dgt, index):
        data = index.model().data(index, QtCore.Qt.DisplayRole)
        # The data should be specified as a thumbtup
        assert isinstance(data, tuple), 'data should be a thumbtup'
        thumbtup = data
        #(thumb_path, img_path, bbox_list) = thumbtup
        return thumbtup

    def paint(dgt, painter, option, index):
        view = dgt.parent()
        text = index.data().toString()
        button = QtGui.QPushButton(text, view, clicked=view.cellButtonClicked)
        view.setIndexWidget(index, button)
