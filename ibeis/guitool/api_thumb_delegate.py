from __future__ import absolute_import, division, print_function
from multiprocessing import Process
import numpy as np
#from guitool import guitool_components as comp
from PyQt4 import QtGui, QtCore
import utool
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableWidget]', DEBUG=False)


class APIThumbDelegate(QtGui.QItemDelegate):
    loadSignal = QtCore.pyqtSignal(
        QtGui.QPainter, 
        QtGui.QStyleOptionViewItemV4, 
        QtCore.QModelIndex
    )

    def __init__(dgt, parent=None):
        super(APIThumbDelegate, dgt).__init__(parent)
        dgt.loadSignal.connect(dgt.loadSlot)
       
    def paint(dgt, painter, option, index):
        painter.save()
        painter.translate(option.rect.x(), option.rect.y())
        painter.drawLine(QtCore.QLineF(0.0, 0.0, 50.0, 50.0))
        painter.restore()
        # print('emit signal')
        dgt.loadSignal.emit(painter, option, index)
        # print("signal")

    @QtCore.pyqtSlot(
        QtGui.QPainter, 
        QtGui.QStyleOptionViewItemV4, 
        QtCore.QModelIndex
    )
    def loadSlot(dgt, painter, option, index):
        # print('startload')
        npimg   = index.model().data(index, QtCore.Qt.DisplayRole)
        data    = npimg.astype(np.uint8)
        (height, width) = npimg.shape[0:2]
        format_ = QtGui.QImage.Format_RGB888
        qimg    = QtGui.QImage(data, width, height, format_)

        painter.save()
        painter.translate(option.rect.x(), option.rect.y())
        painter.drawImage(QtCore.QRectF(0,0,width,height), qimg)
        painter.restore()

        # temp = 0
        # for ix in xrange(int(1E8)):
        #     temp += ix / 2
        # print("loaded")
