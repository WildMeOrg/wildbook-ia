from __future__ import absolute_import, division, print_function
from multiprocessing import Process
import cv2
import numpy as np
from os.path import exists
from vtool import image as gtool
#from guitool import guitool_components as comp
from PyQt4 import QtGui, QtCore
import utool
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableWidget]', DEBUG=False)


class APIThumbDelegate(QtGui.QItemDelegate):
    def __init__(dgt, parent=None):
        super(APIThumbDelegate, dgt).__init__(parent)
        dgt.pool = QtCore.QThreadPool()
        dgt.pool.setMaxThreadCount(8)
        
    def paint(dgt, painter, option, index):
        try:
            dgt.thumb_path, dgt.image_path, dgt.bboxes = index.model().data(index, QtCore.Qt.DisplayRole)
            #print('%d, %d' % (index.column(),index.row()))
            #print(dgt.thumb_path)
            if exists(dgt.image_path):
                if not utool.checkpath(dgt.thumb_path):
                    index.model()._update()
                    dgt.bboxes = index.model().data(index, QtCore.Qt.DisplayRole)[2]
                    print('%d, %d' % (index.column(),index.row()))
                    print(dgt.bboxes)
                    print("Should be remaking thumbnail")
                #if True:
                    offset = dgt.parent().verticalOffset()
                    dgt.pool.start(
                        ThumbnailCreationThread(
                            dgt.thumb_path, 
                            dgt.image_path, 
                            index, 
                            dgt.parent(),
                            offset + option.rect.y(),
                            dgt.bboxes
                        ) 
                    )
                else:
                    npimg   = gtool.imread(dgt.thumb_path, )
                    npimg   = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
                    data    = npimg.astype(np.uint8)
                    (height, width, nDims) = npimg.shape[0:3]
                    npimg   = np.dstack((npimg[:, :, 3], npimg[:, :, 0:2]))
                    format_ = QtGui.QImage.Format_ARGB32
                    qimg    = QtGui.QImage(data, width, height, format_)

                    painter.save()
                    painter.setClipRect(option.rect)
                    painter.translate(option.rect.x(), option.rect.y())
                    painter.drawImage(QtCore.QRectF(0,0,width, height), qimg)
                    painter.restore()
            else:
                print("SOURCE IMAGE NOT COMPUTED")
        except:
            painter.save()
            painter.restore()
    

class ThumbnailCreationThread(QtCore.QRunnable):
    def __init__(thread, thumb_path, image_path, index, view, offset, bboxes):
        QtCore.QRunnable.__init__(thread)
        thread.thumb_path = thumb_path
        thread.image_path = image_path
        thread.index = index
        thread.offset = offset
        thread.thumb_size = 200
        thread.view = view
        thread.bboxes = bboxes

    def run(thread):
        # size = thread.view.viewport().size().height()
        # if( abs(thread.view.verticalOffset() + int(size / 2) - thread.offset) < size ):
        image = gtool.imread(thread.image_path)
        max_dsize = (thread.thumb_size, thread.thumb_size)
        for bbox in thread.bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            (r, g, b) = (255,128,0)
            cv2.rectangle(image, (x1,y1), (x2, y2), (b, g, r), 3)
        thumb_image = gtool.resize_thumb(image, max_dsize)
        gtool.imwrite(thread.thumb_path, thumb_image)
        print("Thumb Written: %s" % thread.thumb_path)
        thread.index.model().dataChanged.emit(thread.index, thread.index)
