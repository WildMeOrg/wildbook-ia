from __future__ import absolute_import, division, print_function
from .__PYQT__ import QtGui, QtCore
import cv2
import numpy as np
import utool
#from six.moves import zip
from os.path import exists
from vtool import image as gtool
#from vtool import linalg, geometry
from vtool import geometry
#from multiprocessing import Process
#from guitool import guitool_components as comp
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APIItemWidget]', DEBUG=False)


DELEGATE_BASE = QtGui.QItemDelegate
RUNNABLE_BASE = QtCore.QRunnable
MAX_NUM_THUMB_THREADS = 1
VERBOSE = utool.VERBOSE


def read_thumb_as_qimg(thumb_path):
    # Read thumbnail image and convert to 32bit aligned for Qt
    npimg   = gtool.imread(thumb_path, delete_if_corrupted=True)
    npimg   = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
    data    = npimg.astype(np.uint8)
    (height, width, nDims) = npimg.shape[0:3]
    npimg   = np.dstack((npimg[:, :, 3], npimg[:, :, 0:2]))
    format_ = QtGui.QImage.Format_ARGB32
    qimg    = QtGui.QImage(data, width, height, format_)
    return qimg, width, height


RUNNING_CREATION_THREADS = {}


def register_thread(key, val):
    global RUNNING_CREATION_THREADS
    RUNNING_CREATION_THREADS[key] = val


def unregister_thread(key):
    global RUNNING_CREATION_THREADS
    del RUNNING_CREATION_THREADS[key]


class APIThumbDelegate(DELEGATE_BASE):
    """ TODO: The delegate can have a reference to the view, and it is allowed
    to resize the rows to fit the images.  It probably should not resize columns
    but it can get the column width and resize the image to that size.  """
    def __init__(dgt, parent=None):
        DELEGATE_BASE.__init__(dgt, parent)
        dgt.pool = None
        dgt.thumbsize = 128

    def get_model_data(dgt, qtindex):
        """ The model data for a thumb should be a (thumb_path, img_path, bbox_list) tuple """
        model = qtindex.model()
        data = model.data(qtindex, QtCore.Qt.DisplayRole, thumbsize=dgt.thumbsize)
        if data is None:
            return None
        # The data should be specified as a thumbtup
        #if isinstance(data, QtCore.QVariant):
        if hasattr(data, 'toPyObject'):
            data = data.toPyObject()
        if data is None:
            return None
        assert isinstance(data, tuple), 'data=%r is %r. should be a thumbtup' % (data, type(data))
        thumbtup = data
        #(thumb_path, img_path, bbox_list) = thumbtup
        return thumbtup

    def try_get_thumb_path(dgt, option, qtindex):
        """ Checks if the thumbnail is ready to paint
        Returns thumb_path if computed. Otherwise returns None """
        # Get data from the models display role
        try:
            data = dgt.get_model_data(qtindex)
            if data is None:
                return
            thumb_path, img_path, bbox_list, theta_list = data
            if thumb_path is None or img_path is None or bbox_list is None:
                print('something is wrong')
                return
        except AssertionError as ex:
            utool.printex(ex)
            return

        if not exists(thumb_path):
            if not exists(img_path):
                #if VERBOSE:
                #print('[ThumbDelegate] SOURCE IMAGE NOT COMPUTED: %r' % (img_path,))
                return None
            # Start computation of thumb if needed
            #qtindex.model()._update()  # should probably be deleted
            view = dgt.parent()
            thumbsize = dgt.thumbsize
            # where you are when you request the run
            offset = view.verticalOffset() + option.rect.y()
            thumb_creation_thread = ThumbnailCreationThread(
                thumb_path,
                img_path,
                thumbsize,
                qtindex,
                view,
                offset,
                bbox_list,
                theta_list
            )
            #register_thread(thumb_path, thumb_creation_thread)
            # Initialize threadcount
            dgt.pool = QtCore.QThreadPool()
            dgt.pool.setMaxThreadCount(MAX_NUM_THUMB_THREADS)
            dgt.pool.start(thumb_creation_thread)
            # print('[ThumbDelegate] Waiting to compute')
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
                if isinstance(view, QtGui.QTreeView):
                    col_width = view.columnWidth(qtindex.column())
                    col_height = view.rowHeight(qtindex)
                elif isinstance(view, QtGui.QTableView):
                    col_width = view.columnWidth(qtindex.column())
                    col_height = view.rowHeight(qtindex.row())
                    # Let columns shrink
                    if dgt.thumbsize != col_width:
                        view.setColumnWidth(qtindex.column(), dgt.thumbsize)
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

    def sizeHint(dgt, option, index):
        try:
            thumb_path = dgt.try_get_thumb_path(option, index)
            if thumb_path is not None:
                # Read the precomputed thumbnail
                qimg, width, height = read_thumb_as_qimg(thumb_path)
                return QtCore.QSize(width, height)
            else:
                #print("[APIThumbDelegate] Name not found")
                return QtCore.QSize()
        except Exception as ex:
            print("Error in APIThumbDelegate")
            utool.printex(ex, 'Error in APIThumbDelegate')
            return QtCore.QSize()


class ThumbnailCreationThread(RUNNABLE_BASE):
    """ Helper to compute thumbnails concurrently """

    def __init__(thread, thumb_path, img_path, thumbsize, qtindex, view, offset, bbox_list, theta_list):
        RUNNABLE_BASE.__init__(thread)
        thread.thumb_path = thumb_path
        thread.img_path = img_path
        thread.qtindex = qtindex
        thread.offset = offset
        thread.thumbsize = thumbsize
        thread.view = view
        thread.bbox_list = bbox_list
        thread.theta_list = theta_list

    #def __del__(self):
    #    print('About to delete creation thread')

    def thumb_would_be_visible(thread):
        viewport = thread.view.viewport()
        height = viewport.size().height()
        height_offset = thread.view.verticalOffset()
        current_offset = height_offset + height // 2
        # Check if the current scroll position is far beyond the
        # scroll position when this was initially requested.
        return abs(current_offset - thread.offset) < height

    def _run(thread):
        """ Compute thumbnail in a different thread """
        # TODO 6-Aug-2014: Vtool has functions very similar to these.
        # they should be called instead to reduce duplicate code.
        # Some duplicate code also exists in ibsfuncs
        # print(thread.img_path)
        if not thread.thumb_would_be_visible():
            #unregister_thread(thread.thumb_path)
            return
        image = gtool.imread(thread.img_path)
        max_dsize = (thread.thumbsize, thread.thumbsize)
        # Resize image to thumb
        thumb = gtool.resize_thumb(image, max_dsize)
        if not utool.is_listlike(thread.theta_list):
            theta_list = [thread.theta_list]
        else:
            theta_list = thread.theta_list
        # Get scale factor
        sx, sy = gtool.get_scale_factor(image, thumb)
        orange_bgr = (0, 128, 255)
        for new_verts in gtool.scale_bbox_to_verts_gen(thread.bbox_list,
                                                       theta_list, sx, sy):
            if not thread.thumb_would_be_visible():
                #unregister_thread(thread.thumb_path)
                return
            # -----------------
            thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
        # Draw bboxes on thumb (not image)
        #for bbox, theta in zip(thread.bbox_list, theta_list):
        #    if not thread.thumb_would_be_visible():
        #        #unregister_thread(thread.thumb_path)
        #        return
        #    # Transformation matrixes
        #    R = linalg.rotation_around_bbox_mat3x3(theta, bbox)
        #    S = linalg.scale_mat3x3(sx, sy)
        #    # Get verticies of the annotation polygon
        #    verts = geometry.verts_from_bbox(bbox, close=True)
        #    # Rotate and transform to thumbnail space
        #    xyz_pts = geometry.homogonize(np.array(verts).T)
        #    trans_pts = geometry.unhomogonize(S.dot(R).dot(xyz_pts))
        #    new_verts = np.round(trans_pts).astype(np.int).T.tolist()
        #    # -----------------
        #    orange_bgr = (0, 128, 255)
        #    thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
        gtool.imwrite(thread.thumb_path, thumb)
        #print('[ThumbCreationThread] Thumb Written: %s' % thread.thumb_path)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)
        #unregister_thread(thread.thumb_path)

    def run(thread):
        try:
            thread._run()
        except Exception as ex:
            utool.printex(ex, 'thread failed', traceback=True)
            #raise


# GRAVE:
#print('[APIItemDelegate] Request Thumb: rc=(%d, %d), nBboxes=%r' %
#      (qtindex.row(), qtindex.column(), len(bbox_list)))
#print('[APIItemDelegate] bbox_list = %r' % (bbox_list,))
