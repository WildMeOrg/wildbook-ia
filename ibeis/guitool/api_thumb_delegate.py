from __future__ import absolute_import, division, print_function
from .__PYQT__ import QtGui, QtCore
import cv2
import numpy as np
import utool
#import time
#from six.moves import zip
from os.path import exists
from vtool import image as gtool
#from vtool import linalg, geometry
from vtool import geometry
#from multiprocessing import Process
#from guitool import guitool_components as comp
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APIItemWidget]', DEBUG=False)
import utool as ut

VERBOSE = utool.VERBOSE or ut.get_argflag(('--verbose-qt', '--verbqt'))


DELEGATE_BASE = QtGui.QItemDelegate
RUNNABLE_BASE = QtCore.QRunnable
MAX_NUM_THUMB_THREADS = 1


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
        if VERBOSE:
            print('[ThumbDelegate] __init__')
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

    def try_get_thumb_path(dgt, view, offset, qtindex):
        """ Checks if the thumbnail is ready to paint
        Returns thumb_path if computed. Otherwise returns None """

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        # Get data from the models display role
        try:
            data = dgt.get_model_data(qtindex)
            if data is None:
                return
            thumb_path, img_path, img_size, bbox_list, theta_list = data
            if thumb_path is None or img_path is None or bbox_list is None or img_size is None:
                print('something is wrong')
                return
        except AssertionError as ex:
            utool.printex(ex)
            return

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        if not exists(thumb_path):
            if not exists(img_path):
                if VERBOSE:
                    print('[ThumbDelegate] SOURCE IMAGE NOT COMPUTED: %r' % (img_path,))
                return None
            # Start computation of thumb if needed
            #qtindex.model()._update()  # should probably be deleted
            thumbsize = dgt.thumbsize
            # where you are when you request the run
            thumb_creation_thread = ThumbnailCreationThread(
                thumb_path,
                img_path,
                img_size,
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
        view = dgt.parent()
        offset = view.verticalOffset() + option.rect.y()
        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None
        try:
            thumb_path = dgt.try_get_thumb_path(view, offset, qtindex)
            if thumb_path is not None:
                # Check if still in viewport
                if view_would_not_be_visible(view, offset):
                    return None
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
                # Check if still in viewport
                if view_would_not_be_visible(view, offset):
                    return None
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

    def sizeHint(dgt, option, qtindex):
        view = dgt.parent()
        offset = view.verticalOffset() + option.rect.y()
        try:
            thumb_path = dgt.try_get_thumb_path(view, offset, qtindex)
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


def view_would_not_be_visible(view, offset):
    viewport = view.viewport()
    height = viewport.size().height()
    height_offset = view.verticalOffset()
    current_offset = height_offset + height // 2
    # Check if the current scroll position is far beyond the
    # scroll position when this was initially requested.
    return abs(current_offset - offset) >= height


class ThumbnailCreationThread(RUNNABLE_BASE):
    """ Helper to compute thumbnails concurrently """

    def __init__(thread, thumb_path, img_path, img_size, thumbsize, qtindex, view, offset, bbox_list, theta_list):
        RUNNABLE_BASE.__init__(thread)
        thread.thumb_path = thumb_path
        thread.img_path = img_path
        thread.img_size = img_size
        thread.qtindex = qtindex
        thread.offset = offset
        thread.thumbsize = thumbsize
        thread.view = view
        thread.bbox_list = bbox_list
        thread.theta_list = theta_list

    #def __del__(self):
    #    print('About to delete creation thread')

    def thumb_would_not_be_visible(thread):
        return view_would_not_be_visible(thread.view, thread.offset)

    def _run(thread):
        """ Compute thumbnail in a different thread """
        #time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # Precompute info BEFORE reading the image (.0002s)
        bbox_list = thread.bbox_list
        theta_list = [thread.theta_list] if not utool.is_listlike(thread.theta_list) else thread.theta_list
        max_dsize = (thread.thumbsize, thread.thumbsize)
        dsize, sx, sy = gtool.resized_clamped_thumb_dims(thread.img_size, max_dsize)
        orange_bgr = (0, 128, 255)
        # Compute new verts list
        new_verts_list = []
        for new_verts in gtool.scale_bbox_to_verts_gen(bbox_list, theta_list, sx, sy):
            new_verts_list.append(new_verts)
        #time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # -----------------
        # This part takes time, hopefully the user actually wants to see this
        # thumbnail.
        img = gtool.imread(thread.img_path)  # Read Image (.0424s) <- Takes most time!
        thumb = gtool.resize(img, dsize)  # Resize to thumb dims (.0015s)
        for new_verts in new_verts_list:
            # Draw bboxes on thumb (not image)
            thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
        gtool.imwrite(thread.thumb_path, thumb)
        #print('[ThumbCreationThread] Thumb Written: %s' % thread.thumb_path)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)
        #unregister_thread(thread.thumb_path)

    def run(thread):
        try:
            thread._run()
        except Exception as ex:
            utool.printex(ex, 'thread failed', tb=True)
            #raise


# GRAVE:
#print('[APIItemDelegate] Request Thumb: rc=(%d, %d), nBboxes=%r' %
#      (qtindex.row(), qtindex.column(), len(bbox_list)))
#print('[APIItemDelegate] bbox_list = %r' % (bbox_list,))
