"""
CommandLine:
    main.py --db PZ_MUGU_19 --eid 13 --verbthumb

Notes:
    http://stackoverflow.com/questions/8312725/how-to-create-executable-file-for-a-qt-application
    http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
    For windows need at least these dlls:
        mingwm10.dll
        libgcc_s_dw2-1.dll
        QtCore4.dll
        QtGui4.dll
"""
from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtGui, QtCore
import cv2  # NOQA
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
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APIThumbDelegate]', DEBUG=False)
import utool as ut
ut.noinject(__name__, '[APIThumbDelegate]', DEBUG=False)


VERBOSE_QT = ut.get_argflag(('--verbose-qt', '--verbqt'))
VERBOSE_THUMB = utool.VERBOSE or ut.get_argflag(('--verbose-thumb', '--verbthumb')) or VERBOSE_QT


MAX_NUM_THUMB_THREADS = 1


def read_thumb_size(thumb_path):
    if VERBOSE_THUMB:
        print('[ThumbDelegate] Reading thumb size')
    npimg = gtool.imread(thumb_path, delete_if_corrupted=True)
    (height, width) = npimg.shape[0:2]
    del npimg
    return width, height


def test_show_qimg(qimg):
    qpixmap = QtGui.QPixmap(qimg)
    lbl = QtGui.QLabel()
    lbl.setPixmap(qpixmap)
    lbl.show()   # show label with qim image
    return lbl


#@ut.memprof
def read_thumb_as_qimg(thumb_path):
    r"""
    Args:
        thumb_path (?):

    Returns:
        tuple: (qimg, width, height)

    CommandLine:
        python -m guitool.api_thumb_delegate --test-read_thumb_as_qimg --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.api_thumb_delegate import *  # NOQA
        >>> import guitool
        >>> # build test data
        >>> thumb_path = ut.grab_test_imgpath('carl.jpg')
        >>> # execute function
        >>> guitool.ensure_qtapp()
        >>> (qimg) = ut.memprof(read_thumb_as_qimg)(thumb_path)
        >>> if ut.show_was_requested():
        >>>    lbl = test_show_qimg(qimg)
        >>>    guitool.qtapp_loop()
        >>> # verify results
        >>> print(qimg)

    Timeit::
        %timeit np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
        %timeit cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
        npimg1 = np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
        # seems to be memory leak in cvtColor?
        npimg2 = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)

    """
    if VERBOSE_THUMB:
        print('[ThumbDelegate] Reading thumb as qimg')
    # Read thumbnail image and convert to 32bit aligned for Qt
    #if False:
    #    data  = np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
    if False:
        # Reading the npimage and then handing it off to Qt causes a memory
        # leak. The numpy array probably is never unallocated because qt doesn't
        # own it and it never loses its reference count
        npimg = gtool.imread(thumb_path, delete_if_corrupted=True)
        print('npimg.dtype = %r, %r' % (npimg.shape, npimg.dtype))
        npimg   = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
        format_ = QtGui.QImage.Format_ARGB32
        #    #data    = npimg.astype(np.uint8)
        #    #npimg   = np.dstack((npimg[:, :, 3], npimg[:, :, 0:2]))
        #    #data    = npimg.astype(np.uint8)
        #else:
        # Memory seems to be no freed by the QImage?
        #data = np.ascontiguousarray(npimg[:, :, ::-1].astype(np.uint8), dtype=np.uint8)
        #data = np.ascontiguousarray(npimg[:, :, :].astype(np.uint8), dtype=np.uint8)
        data = npimg
        #format_ = QtGui.QImage.Format_RGB888
        (height, width) = data.shape[0:2]
        qimg    = QtGui.QImage(data, width, height, format_)
        del npimg
        del data
    else:
        format_ = QtGui.QImage.Format_ARGB32
        #qimg    = QtGui.QImage(thumb_path, format_)
        qimg    = QtGui.QImage(thumb_path)
    return qimg


RUNNING_CREATION_THREADS = {}


def register_thread(key, val):
    global RUNNING_CREATION_THREADS
    RUNNING_CREATION_THREADS[key] = val


def unregister_thread(key):
    global RUNNING_CREATION_THREADS
    del RUNNING_CREATION_THREADS[key]


DELEGATE_BASE = QtGui.QItemDelegate


class APIThumbDelegate(DELEGATE_BASE):
    """
    TODO: The delegate can have a reference to the view, and it is allowed
    to resize the rows to fit the images.  It probably should not resize columns
    but it can get the column width and resize the image to that size.

    get_thumb_size is a callback function which should return whatever the
    requested thumbnail size is
    """
    def __init__(dgt, parent=None, get_thumb_size=None):
        if VERBOSE_THUMB:
            print('[ThumbDelegate] __init__ parent=%r, get_thumb_size=%r' % (parent, get_thumb_size))
        DELEGATE_BASE.__init__(dgt, parent)
        dgt.pool = None
        if get_thumb_size is None:
            dgt.get_thumb_size = lambda: 128  # 256
        else:
            dgt.get_thumb_size = get_thumb_size  # 256
        dgt.last_thumbsize = None

    def get_model_data(dgt, qtindex):
        """
        The model data for a thumb should be a tuple:
        (thumb_path, img_path, imgsize, bboxes, thetas)
        """
        model = qtindex.model()
        datakw = dict(thumbsize=dgt.get_thumb_size())
        data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
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
        Returns thumb_path if computed. Otherwise returns None
        """

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        # Get data from the models display role
        try:
            data = dgt.get_model_data(qtindex)
            if data is None:
                print('[thumb_delegate] no data')
                return
            (thumb_path, img_path, img_size, bbox_list, theta_list) = data
            invalid = (thumb_path is None or img_path is None or bbox_list is None
                       or img_size is None)
            if invalid:
                print('[thumb_delegate] something is wrong')
                return
        except AssertionError as ex:
            utool.printex(ex, 'error getting thumbnail data')
            return

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        if not exists(thumb_path):
            if not exists(img_path):
                if VERBOSE_THUMB:
                    print('[ThumbDelegate] SOURCE IMAGE NOT COMPUTED: %r' % (img_path,))
                return None
            # Start computation of thumb if needed
            #qtindex.model()._update()  # should probably be deleted
            # where you are when you request the run
            if VERBOSE_THUMB:
                print('[ThumbDelegate] Spawning thumbnail creation thread')
            thumbsize = dgt.get_thumb_size()
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
                qimg = read_thumb_as_qimg(thumb_path)
                width, height = qimg.width(), qimg.height()
                view = dgt.parent()
                if isinstance(view, QtGui.QTreeView):
                    col_width = view.columnWidth(qtindex.column())
                    col_height = view.rowHeight(qtindex)
                elif isinstance(view, QtGui.QTableView):
                    # dimensions of the table cells
                    col_width = view.columnWidth(qtindex.column())
                    col_height = view.rowHeight(qtindex.row())
                    thumbsize = dgt.get_thumb_size()
                    if thumbsize != dgt.last_thumbsize:
                        # has thumbsize changed?
                        if thumbsize != col_width:
                            view.setColumnWidth(qtindex.column(), thumbsize)
                        if height != col_height:
                            view.setRowHeight(qtindex.row(), height)
                        dgt.last_thumbsize = thumbsize
                    # Let columns shrink
                    if thumbsize != col_width:
                        view.setColumnWidth(qtindex.column(), thumbsize)
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
                width, height = read_thumb_size(thumb_path)
                return QtCore.QSize(width, height)
            else:
                #print("[APIThumbDelegate] Name not found")
                return QtCore.QSize()
        except Exception as ex:
            print("Error in APIThumbDelegate")
            utool.printex(ex, 'Error in APIThumbDelegate', tb=True)
            return QtCore.QSize()


def view_would_not_be_visible(view, offset):
    viewport = view.viewport()
    height = viewport.size().height()
    height_offset = view.verticalOffset()
    current_offset = height_offset + height // 2
    # Check if the current scroll position is far beyond the
    # scroll position when this was initially requested.
    return abs(current_offset - offset) >= height


RUNNABLE_BASE = QtCore.QRunnable


def get_thread_thumb_info(bbox_list, theta_list, thumbsize, img_size):
    theta_list = [theta_list] if not utool.is_listlike(theta_list) else theta_list
    max_dsize = (thumbsize, thumbsize)
    dsize, sx, sy = gtool.resized_clamped_thumb_dims(img_size, max_dsize)
    # Compute new verts list
    new_verts_list = [new_verts for new_verts in gtool.scale_bbox_to_verts_gen(bbox_list, theta_list, sx, sy)]
    return dsize, new_verts_list


def make_thread_thumb(img_path, dsize, new_verts_list):
    orange_bgr = (0, 128, 255)
    img = gtool.imread(img_path)  # Read Image (.0424s) <- Takes most time!
    thumb = gtool.resize(img, dsize)  # Resize to thumb dims (.0015s)
    del img
    # Draw bboxes on thumb (not image)
    for new_verts in new_verts_list:
        geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2, out=thumb)
        #thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
    return thumb


class ThumbnailCreationThread(RUNNABLE_BASE):
    """
    Helper to compute thumbnails concurrently
    """

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

    def thumb_would_not_be_visible(thread):
        return view_would_not_be_visible(thread.view, thread.offset)

    def _run(thread):
        """ Compute thumbnail in a different thread """
        #time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # Precompute info BEFORE reading the image (.0002s)
        dsize, new_verts_list = get_thread_thumb_info(thread.bbox_list, thread.theta_list, thread.thumbsize, thread.img_size)
        #time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # -----------------
        # This part takes time, hopefully the user actually wants to see this
        # thumbnail.
        thumb = make_thread_thumb(thread.img_path, dsize, new_verts_list)
        if thread.thumb_would_not_be_visible():
            return
        gtool.imwrite(thread.thumb_path, thumb)
        del thumb
        if thread.thumb_would_not_be_visible():
            return
        #print('[ThumbCreationThread] Thumb Written: %s' % thread.thumb_path)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)
        #unregister_thread(thread.thumb_path)

    def run(thread):
        try:
            thread._run()
        except Exception as ex:
            utool.printex(ex, 'thread failed', tb=True)
            #raise

    #def __del__(self):
    #    print('About to delete creation thread')


# GRAVE:
#print('[APIItemDelegate] Request Thumb: rc=(%d, %d), nBboxes=%r' %
#      (qtindex.row(), qtindex.column(), len(bbox_list)))
#print('[APIItemDelegate] bbox_list = %r' % (bbox_list,))


if __name__ == '__main__':
    """
    CommandLine:
        python -m guitool.api_thumb_delegate
        python -m guitool.api_thumb_delegate --allexamples
        python -m guitool.api_thumb_delegate --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
