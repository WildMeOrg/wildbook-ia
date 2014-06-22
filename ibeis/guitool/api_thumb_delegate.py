from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import utool
from os.path import exists
from PyQt4 import QtGui, QtCore
from vtool import image as gtool
#from multiprocessing import Process
#from guitool import guitool_components as comp
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APITableWidget]', DEBUG=False)


DELEGATE_BASE = QtGui.QItemDelegate
RUNNABLE_BASE = QtCore.QRunnable
MAX_NUM_THUMB_THREADS = 1
VERBOSE = utool.VERBOSE


def read_thumb_as_qimg(thumb_path):
    # Read thumbnail image and convert to 32bit aligned for Qt
    npimg   = gtool.imread(thumb_path)
    npimg   = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
    data    = npimg.astype(np.uint8)
    (height, width, nDims) = npimg.shape[0:3]
    npimg   = np.dstack((npimg[:, :, 3], npimg[:, :, 0:2]))
    format_ = QtGui.QImage.Format_ARGB32
    qimg    = QtGui.QImage(data, width, height, format_)
    return qimg, width, height


class APIThumbDelegate(DELEGATE_BASE):
    """ TODO: The delegate can have a reference to the view, and it is allowed
    to resize the rows to fit the images.  It probably should not resize columns
    but it can get the column width and resize the image to that size.  """
    def __init__(dgt, parent=None):
        """ PSA: calling super is unsafe in pyqt4 code super(APIThumbDelegate,
        dgt).__init__(parent) Instead call the parent classes init directly """
        DELEGATE_BASE.__init__(dgt, parent)
        dgt.pool = QtCore.QThreadPool()
        dgt.thumb_size = 128
        # Initialize threadcount
        dgt.pool.setMaxThreadCount(MAX_NUM_THUMB_THREADS)

    def get_model_data(dgt, qtindex):
        """ The model data for a thumb should be a (thumb_path, img_path, bbox_list) tuple """
        data = qtindex.model().data(qtindex, QtCore.Qt.DisplayRole)
        if data is None:
            return (None, None, None)
        # The data should be specified as a thumbtup
        if isinstance(data, QtCore.QVariant):
            data = data.toPyObject()
        if data is None:
            return (None, None, None)
        assert isinstance(data, tuple), 'data=%r is %r. should be a thumbtup' % (data, type(data))
        thumbtup = data
        #(thumb_path, img_path, bbox_list) = thumbtup
        return thumbtup

    def try_get_thumb_path(dgt, option, qtindex):
        """ Checks if the thumbnail is ready to paint
        Returns thumb_path if computed. Otherwise returns None """
        # Get data from the models display role
        try:
            thumb_path, img_path, bbox_list = dgt.get_model_data(qtindex)
            if thumb_path is None or img_path is None or bbox_list is None:
                return
        except AssertionError as ex:
            utool.printex(ex)
            return
        if not exists(img_path):
            if VERBOSE:
                print('[ThumbDelegate] SOURCE IMAGE NOT COMPUTED')
            return None
        if not exists(thumb_path):
            # Start computation of thumb if needed
            qtindex.model()._update()
            view = dgt.parent()
            thumb_size = dgt.thumb_size
            offset = view.verticalOffset()
            dgt.pool.start(
                ThumbnailCreationThread(
                    thumb_path,
                    img_path,
                    thumb_size,
                    qtindex,
                    view,
                    offset + option.rect.y(),
                    bbox_list
                )
            )
            #print('[ThumbDelegate] Waiting to compute')
            return None
        else:
            # thumb is computed return the path
            return thumb_path

    def paint(dgt, painter, option, qtindex):
        try:
            thumb_path = dgt.try_get_thumb_path(option, qtindex)
            if thumb_path is not None:
                # Read the precomputed thumbnail
                qimg, width, height = read_thumb_as_qimg(thumb_path)
                view = dgt.parent()
                col_width = view.columnWidth(qtindex.column())
                col_height = view.rowHeight(qtindex.row())
                # Let columns shrink
                if dgt.thumb_size != col_width:
                    view.setColumnWidth(qtindex.column(), dgt.thumb_size)
                # Let rows grow
                if height > col_height:
                    view.setRowHeight(qtindex.row(), height)
                # Paint image on an item in some view
                painter.save()
                painter.setClipRect(option.rect)
                painter.translate(option.rect.x(), option.rect.y())
                painter.drawImage(QtCore.QRectF(0, 0, width, height), qimg)
                painter.restore()
        except Exception as ex:
            # PSA: Always report errors on Exceptions!
            print('Error in APIThumbDelegate')
            utool.printex(ex, 'Error in APIThumbDelegate')
            painter.save()
            painter.restore()


class ThumbnailCreationThread(RUNNABLE_BASE):
    """ Helper to compute thumbnails concurrently """

    def __init__(thread, thumb_path, img_path, thumb_size, qtindex, view, offset, bbox_list):
        RUNNABLE_BASE.__init__(thread)
        thread.thumb_path = thumb_path
        thread.img_path = img_path
        thread.qtindex = qtindex
        thread.offset = offset
        thread.thumb_size = thumb_size
        thread.view = view
        thread.bbox_list = bbox_list

    def run(thread):
        # size = thread.view.viewport().size().height()
        # if( abs(thread.view.verticalOffset() + int(size / 2) - thread.offset) < size ):
        image = gtool.imread(thread.img_path)
        max_dsize = (thread.thumb_size, thread.thumb_size)
        # Resize image to thumb
        thumb = gtool.resize_thumb(image, max_dsize)
        # Get scale factor
        sx, sy = gtool.get_scale_factor(image, thumb)
        # Draw bboxes on thumb (not image)
        for bbox in thread.bbox_list:
            pt1, pt2 = gtool.cvt_bbox_xywh_to_pt1pt2(bbox, sx=sx, sy=sy, round_=True)
            orange_bgr = (0, 128, 255)
            color = orange_bgr
            thickness = 2
            cv2.rectangle(thumb, pt1, pt2, color, thickness)
        gtool.imwrite(thread.thumb_path, thumb)
        #print('[ThumbCreationThread] Thumb Written: %s' % thread.thumb_path)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)


# GRAVE:
#print('[APIItemDelegate] Request Thumb: rc=(%d, %d), nBboxes=%r' %
#      (qtindex.row(), qtindex.column(), len(bbox_list)))
#print('[APIItemDelegate] bbox_list = %r' % (bbox_list,))
