# -*- coding: utf-8 -*-
"""
CommandLine:
    rm -rf /media/raid/work/PZ_MTEST/_ibsdb/_wbia_cache/match_thumbs/
    python -m wbia.gui.inspect_gui --test-test_review_widget --show --verbose-thumb
"""
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtGui, QtCore
from wbia.guitool.__PYQT__ import QtWidgets  # NOQA
import six
from os.path import exists
import utool as ut

ut.noinject(__name__, '[APIThumbDelegate]')


VERBOSE_QT = ut.get_argflag(('--verbose-qt', '--verbqt'))
VERBOSE_THUMB = (
    ut.VERBOSE or ut.get_argflag(('--verbose-thumb', '--verbthumb')) or VERBOSE_QT
)


MAX_NUM_THUMB_THREADS = 1


def read_thumb_size(thumb_path):
    import vtool as vt

    if VERBOSE_THUMB:
        print('[ThumbDelegate] Reading thumb size')
    # npimg = vt.imread(thumb_path, delete_if_corrupted=True)
    # (height, width) = npimg.shape[0:2]
    # del npimg
    try:
        width, height = vt.open_image_size(thumb_path)
    except IOError as ex:
        if ut.checkpath(thumb_path, verbose=True):
            ut.printex(
                ex,
                'image=%r seems corrupted. Needs deletion' % (thumb_path,),
                iswarning=True,
            )
            ut.delete(thumb_path)
        else:
            ut.printex(ex, 'image=%r does not exist', (thumb_path,), iswarning=True)
        raise
    return width, height


def test_show_qimg(qimg):
    qpixmap = QtGui.QPixmap(qimg)
    lbl = QtWidgets.QLabel()
    lbl.setPixmap(qpixmap)
    lbl.show()  # show label with qim image
    return lbl


# @ut.memprof
def read_thumb_as_qimg(thumb_path):
    r"""
    Args:
        thumb_path (?):

    Returns:
        tuple: (qimg, width, height)

    CommandLine:
        python -m wbia.guitool.api_thumb_delegate --test-read_thumb_as_qimg --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.api_thumb_delegate import *  # NOQA
        >>> from wbia import guitool
        >>> # build test data
        >>> thumb_path = ut.grab_test_imgpath('carl.jpg')
        >>> # execute function
        >>> guitool.ensure_qtapp()
        >>> qimg = read_thumb_as_qimg(thumb_path)
        >>> print(qimg)
        >>> # xdoctest: +REQUIRES(--show)
        >>> lbl = test_show_qimg(qimg)
        >>> #guitool.qtapp_loop()
        >>> # verify results

    Timeit::
        %timeit np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
        %timeit cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
        npimg1 = np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
        # seems to be memory leak in cvtColor?
        npimg2 = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)

    """
    if VERBOSE_THUMB:
        print('[ThumbDelegate] Reading thumb as qimg. thumb_path=%r' % (thumb_path,))
    # Read thumbnail image and convert to 32bit aligned for Qt
    # if False:
    #    data  = np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
    # if False:
    #    # Reading the npimage and then handing it off to Qt causes a memory
    #    # leak. The numpy array probably is never unallocated because qt doesn't
    #    # own it and it never loses its reference count
    #    #npimg = vt.imread(thumb_path, delete_if_corrupted=True)
    #    #print('npimg.dtype = %r, %r' % (npimg.shape, npimg.dtype))
    #    #npimg   = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
    #    #format_ = QtGui.QImage.Format_ARGB32
    #    ##    #data    = npimg.astype(np.uint8)
    #    ##    #npimg   = np.dstack((npimg[:, :, 3], npimg[:, :, 0:2]))
    #    ##    #data    = npimg.astype(np.uint8)
    #    ##else:
    #    ## Memory seems to be no freed by the QImage?
    #    ##data = np.ascontiguousarray(npimg[:, :, ::-1].astype(np.uint8), dtype=np.uint8)
    #    ##data = np.ascontiguousarray(npimg[:, :, :].astype(np.uint8), dtype=np.uint8)
    #    #data = npimg
    #    ##format_ = QtGui.QImage.Format_RGB888
    #    #(height, width) = data.shape[0:2]
    #    #qimg    = QtGui.QImage(data, width, height, format_)
    #    #del npimg
    #    #del data
    # else:
    # format_ = QtGui.QImage.Format_ARGB32
    # qimg    = QtGui.QImage(thumb_path, format_)
    qimg = QtGui.QImage(thumb_path)
    return qimg


RUNNING_CREATION_THREADS = {}


def register_thread(key, val):
    global RUNNING_CREATION_THREADS
    RUNNING_CREATION_THREADS[key] = val


def unregister_thread(key):
    global RUNNING_CREATION_THREADS
    del RUNNING_CREATION_THREADS[key]


DELEGATE_BASE = QtWidgets.QItemDelegate


class APIThumbDelegate(DELEGATE_BASE):
    """
    There is one Thumb Delegate per column. Keep that in mind when writing for
    this class.

    TODO: The delegate can have a reference to the view, and it is allowed
    to resize the rows to fit the images.  It probably should not resize columns
    but it can get the column width and resize the image to that size.

    get_thumb_size is a callback function which should return whatever the
    requested thumbnail size is

    SeeAlso:
         api_item_view.infer_delegates
    """

    def __init__(dgt, parent=None, get_thumb_size=None):
        if VERBOSE_THUMB:
            print(
                '[ThumbDelegate] __init__ parent=%r, get_thumb_size=%r'
                % (parent, get_thumb_size)
            )
        DELEGATE_BASE.__init__(dgt, parent)
        dgt.pool = None
        # TODO: get from the view
        if get_thumb_size is None:
            dgt.get_thumb_size = lambda: 128  # 256
        else:
            dgt.get_thumb_size = get_thumb_size  # 256
        dgt.last_thumbsize = None
        dgt.row_rezised_flags = {}  # SUPER HACK FOR RESIZE SHRINK
        try:
            import cachetools

            dgt.thumb_cache = cachetools.TTLCache(256, ttl=2)
        except ImportError:
            dgt.thumb_cache = ut.LRUDict(256)
        # import utool
        # utool.embed()

    def paint(dgt, painter, option, qtindex):
        """
        TODO: prevent recursive paint
        """
        view = dgt.parent()
        offset = view.verticalOffset() + option.rect.y()
        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None
        try:
            thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
            if thumb_path is not None:
                # Check if still in viewport
                if view_would_not_be_visible(view, offset):
                    return None
                # Read the precomputed thumbnail
                if thumb_path in dgt.thumb_cache:
                    qimg = dgt.thumb_cache[thumb_path]
                else:
                    qimg = read_thumb_as_qimg(thumb_path)
                    dgt.thumb_cache[thumb_path] = qimg
                width, height = qimg.width(), qimg.height()
                # Adjust the cell size to fit the image
                dgt.adjust_thumb_cell_size(qtindex, width, height)
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
            print('Error in APIThumbDelegate')
            ut.printex(ex, 'Error in APIThumbDelegate', tb=True)
            painter.save()
            painter.restore()

    def sizeHint(dgt, option, qtindex):
        view = dgt.parent()
        offset = view.verticalOffset() + option.rect.y()
        try:
            thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
            if thumb_path is not None:
                # Read the precomputed thumbnail
                width, height = read_thumb_size(thumb_path)
                return QtCore.QSize(width, height)
            else:
                # print("[APIThumbDelegate] Name not found")
                return QtCore.QSize()
        except Exception as ex:
            print('Error in APIThumbDelegate')
            ut.printex(ex, 'Error in APIThumbDelegate', tb=True, iswarning=True)
            return QtCore.QSize()

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
        # if isinstance(data, QtCore.QVariant):
        if hasattr(data, 'toPyObject'):
            data = data.toPyObject()
        if data is None:
            return None
        if isinstance(data, six.string_types):
            # data = (data, None, None, None, None)
            return data
        if isinstance(data, dict):
            # HACK FOR DIFFERENT TYPE OF THUMB DATA
            return data
        assert isinstance(data, tuple), 'data=%r is %r. should be a thumbtup' % (
            data,
            type(data),
        )
        thumbtup = data
        # (thumb_path, img_path, bbox_list) = thumbtup
        return thumbtup

    def spawn_thumb_creation_thread(
        dgt,
        thumb_path,
        img_path,
        img_size,
        qtindex,
        view,
        offset,
        bbox_list,
        theta_list,
        interest_list,
    ):
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
            theta_list,
            interest_list,
        )
        # register_thread(thumb_path, thumb_creation_thread)
        # Initialize threadcount
        if dgt.pool is None:
            # dgt.pool = QtCore.QThreadPool()
            # dgt.pool.setMaxThreadCount(MAX_NUM_THUMB_THREADS)
            dgt.pool = QtCore.QThreadPool.globalInstance()
        dgt.pool.start(thumb_creation_thread)
        # print('[ThumbDelegate] Waiting to compute')

    def get_thumb_path_if_exists(dgt, view, offset, qtindex):
        """
        Checks if the thumbnail is ready to paint

        Returns:
            thumb_path if computed otherwise returns None
        """
        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        # Get data from the models display role
        try:
            data = dgt.get_model_data(qtindex)
            if data is None:
                if VERBOSE_THUMB:
                    print('[thumb_delegate] no data')
                return
            thumbtup_mode = isinstance(data, tuple)
            thumbdat_mode = isinstance(data, dict)
            if isinstance(data, six.string_types):
                thumb_path = data
                assert exists(thumb_path), 'must exist'
                return thumb_path
            if thumbtup_mode:
                if len(data) == 5:
                    (thumb_path, img_path, img_size, bbox_list, theta_list) = data
                    interest_list = []
                else:
                    (
                        thumb_path,
                        img_path,
                        img_size,
                        bbox_list,
                        theta_list,
                        interest_list,
                    ) = data
                invalid = (
                    thumb_path is None
                    or img_path is None
                    or bbox_list is None
                    or img_size is None
                )
                if invalid:
                    print('[thumb_delegate] something is wrong')
                    return
            elif thumbdat_mode:
                thumb_path = data['fpath']
            else:
                print('[thumb_delegate] something is wrong')
                return
        except AssertionError as ex:
            ut.printex(ex, 'error getting thumbnail data')
            return

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        if not exists(thumb_path):
            if thumbtup_mode:
                if not exists(img_path):
                    if VERBOSE_THUMB:
                        print(
                            '[ThumbDelegate] SOURCE IMAGE NOT COMPUTED: %r' % (img_path,)
                        )
                    return None
                dgt.spawn_thumb_creation_thread(
                    thumb_path,
                    img_path,
                    img_size,
                    qtindex,
                    view,
                    offset,
                    bbox_list,
                    theta_list,
                    interest_list,
                )
                return None
            elif thumbdat_mode:
                thumbdat = data
                thread_func = thumbdat['thread_func']
                main_func = thumbdat['main_func']
                # kwargs = data['kwargs']
                # func(*args, **kwargs)
                # print('data = %r' % (data,))
                # print('newdata not computed')
                # SPAWN
                if VERBOSE_THUMB:
                    print('[ThumbDelegate] Spawning thumbnail creation thread')
                args = main_func()
                thumb_creation_thread = ThumbnailCreationThread2(
                    thread_func, args, qtindex, view, offset
                )
                # register_thread(thumb_path, thumb_creation_thread)
                # Initialize threadcount
                if dgt.pool is None:
                    # dgt.pool = QtCore.QThreadPool()
                    # dgt.pool.setMaxThreadCount(MAX_NUM_THUMB_THREADS)
                    dgt.pool = QtCore.QThreadPool.globalInstance()
                dgt.pool.start(thumb_creation_thread)
                # print('[ThumbDelegate] Waiting to compute')
                return None
        else:
            # thumb is computed return the path
            return thumb_path

    def adjust_thumb_cell_size(dgt, qtindex, width, height):
        """
        called during paint to ensure that the cell is large enough for the
        image.
        """
        view = dgt.parent()
        if isinstance(view, QtWidgets.QTableView):
            # dimensions of the table cells
            row = qtindex.row()
            col_width = view.columnWidth(qtindex.column())
            col_height = view.rowHeight(row)
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
            if dgt.row_rezised_flags.get(row):
                # HACK TO ONLY SHRINK ONCE WONT WORK WITH RESORT
                return
            else:
                dgt.row_rezised_flags[row] = True
                # Let rows shrink
                # IF THERE IS MORE THAN ONE COLUMN WITH THUMBS THEN THIS WILL CAUSE
                # COLS TO BE RESIZED MANY TIMES UNDER THE HOOD. THAT CAUSES
                # MULTIPLE READS OF THE THUMBS WHICH CAUSES MAJOR SLOWDOWNS.
                if height < col_height:
                    view.setRowHeight(qtindex.row(), height)
        elif isinstance(view, QtWidgets.QTreeView):
            col_width = view.columnWidth(qtindex.column())
            col_height = view.rowHeight(qtindex)
            # TODO: finishme


def view_would_not_be_visible(view, offset):
    """
    Check if the current scroll position is far beyond the
    scroll position when this was initially requested.
    """
    viewport = view.viewport()
    height = viewport.size().height()
    height_offset = view.verticalOffset()
    current_offset = height_offset + height // 2
    return abs(current_offset - offset) >= height


def get_thread_thumb_info(bbox_list, theta_list, thumbsize, img_size):
    r"""
    CommandLine:
        python -m wbia.guitool.api_thumb_delegate --test-get_thread_thumb_info

    Example:
        >>> # ENABLE_DOCTEST
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.api_thumb_delegate import *  # NOQA
        >>> # build test data
        >>> bbox_list = [(100, 50, 400, 200)]
        >>> theta_list = [0]
        >>> thumbsize = 128
        >>> img_size = 600, 300
        >>> # execute function
        >>> result = get_thread_thumb_info(bbox_list, theta_list, thumbsize, img_size)
        >>> # verify results
        >>> print(result)
        ((128, 64), [[[21, 11], [107, 11], [107, 53], [21, 53], [21, 11]]])

    """
    import vtool as vt

    theta_list = [theta_list] if not ut.is_listlike(theta_list) else theta_list
    max_dsize = (thumbsize, thumbsize)
    dsize, sx, sy = vt.resized_clamped_thumb_dims(img_size, max_dsize)
    # Compute new verts list
    new_verts_list = list(vt.scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy))
    return dsize, new_verts_list


def make_thread_thumb(img_path, dsize, new_verts_list, interest_list):
    r"""
    Makes thumbnail with overlay. Called in thead

    CommandLine:
        python -m wbia.guitool.api_thumb_delegate --test-make_thread_thumb --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.api_thumb_delegate import *  # NOQA
        >>> import wbia.plottool as pt
        >>> # build test data
        >>> img_path = ut.grab_test_imgpath('carl.jpg')
        >>> dsize = (32, 32)
        >>> new_verts_list = []
        >>> # execute function
        >>> thumb = make_thread_thumb(img_path, dsize, new_verts_list)
        >>> ut.quit_if_noshow()
        >>> pt.imshow(thumb)
        >>> pt.show_if_requested()
    """
    import vtool as vt
    from vtool import geometry

    orange_bgr = (0, 128, 255)
    blue_bgr = (255, 128, 0)
    # imread causes a MEMORY LEAK most likely!
    img = vt.imread(img_path)  # Read Image (.0424s) <- Takes most time!
    # if False:
    #    #http://stackoverflow.com/questions/9794019/convert-numpy-array-to-pyside-qpixmap
    #    # http://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    #    #import numpy as np
    #    #qimg = QtGui.QImage(img_path, str(QtGui.QImage.Format_RGB32))
    #    #temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth(), 4)
    #    #result_shape = (qimg.height(), qimg.width())
    #    #buf = qimg.bits().asstring(qimg.numBytes())
    #    #result  = np.frombuffer(buf, np.uint8).reshape(temp_shape)
    #    #result = result[:, :result_shape[1]]
    #    #result = result[..., :3]
    #    #img = result
    thumb = vt.image.resize(img, dsize)  # Resize to thumb dims (.0015s)
    del img
    # Draw bboxes on thumb (not image)
    color_bgr_list = [blue_bgr if interest else orange_bgr for interest in interest_list]
    for new_verts, color_bgr in zip(new_verts_list, color_bgr_list):
        if new_verts is not None:
            geometry.draw_verts(thumb, new_verts, color=color_bgr, thickness=2, out=thumb)
        # thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
    return thumb


RUNNABLE_BASE = QtCore.QRunnable


class ThumbnailCreationThread2(RUNNABLE_BASE):
    """
    HACK
    TODO: http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
    """

    def __init__(thread, thread_func, args, qtindex, view, offset):
        RUNNABLE_BASE.__init__(thread)
        thread.thread_func = thread_func
        thread.args = args
        thread.qtindex = qtindex
        thread.offset = offset
        thread.view = view

    def thumb_would_not_be_visible(thread):
        return view_would_not_be_visible(thread.view, thread.offset)

    def _run(thread):
        """ Compute thumbnail in a different thread """
        if thread.thumb_would_not_be_visible():
            return
        # func = thread.thumbdat['func']
        thread.thread_func(thread.thumb_would_not_be_visible, *thread.args)
        # func(check_func=thread.thumb_would_not_be_visible)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)

    def run(thread):
        try:
            thread._run()
        except Exception as ex:
            ut.printex(ex, 'thread failed', tb=True)
            # raise


class ThumbnailCreationThread(RUNNABLE_BASE):
    """
    Helper to compute thumbnails concurrently

    References:
        TODO:
        http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
    """

    def __init__(
        thread,
        thumb_path,
        img_path,
        img_size,
        thumbsize,
        qtindex,
        view,
        offset,
        bbox_list,
        theta_list,
        interest_list,
    ):
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
        thread.interest_list = interest_list

    def thumb_would_not_be_visible(thread):
        return view_would_not_be_visible(thread.view, thread.offset)

    def _run(thread):
        """ Compute thumbnail in a different thread """
        import vtool as vt

        # time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # Precompute info BEFORE reading the image (.0002s)
        dsize, new_verts_list = get_thread_thumb_info(
            thread.bbox_list, thread.theta_list, thread.thumbsize, thread.img_size
        )
        # time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # -----------------
        # This part takes time, hopefully the user actually wants to see this
        # thumbnail.
        thumb = make_thread_thumb(
            thread.img_path, dsize, new_verts_list, thread.interest_list
        )
        if thread.thumb_would_not_be_visible():
            return
        vt.image.imwrite(thread.thumb_path, thumb)
        del thumb
        if thread.thumb_would_not_be_visible():
            return
        # print('[ThumbCreationThread] Thumb Written: %s' % thread.thumb_path)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)
        # unregister_thread(thread.thumb_path)

    def run(thread):
        try:
            thread._run()
        except Exception as ex:
            ut.printex(ex, 'thread failed', tb=True)
            # raise

    # def __del__(self):
    #    print('About to delete creation thread')


# GRAVE:
# print('[APIItemDelegate] Request Thumb: rc=(%d, %d), nBboxes=%r' %
#      (qtindex.row(), qtindex.column(), len(bbox_list)))
# print('[APIItemDelegate] bbox_list = %r' % (bbox_list,))


def simple_thumbnail_widget():
    r"""
    Very simple example to test thumbnails

    CommandLine:
        python -m wbia.guitool.api_thumb_delegate --test-simple_thumbnail_widget  --show --verbthumb
        python -m wbia.guitool.api_thumb_delegate --test-simple_thumbnail_widget  --show --tb

    Example:
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.api_thumb_delegate import *  # NOQA
        >>> import wbia.guitool
        >>> guitool.ensure_qapp()  # must be ensured before any embeding
        >>> wgt = simple_thumbnail_widget()
        >>> ut.quit_if_noshow()
        >>> wgt.show()
        >>> guitool.qtapp_loop(wgt, frequency=100)
    """
    from wbia import guitool

    guitool.ensure_qapp()
    col_name_list = ['rowid', 'image_name', 'thumb']
    col_types_dict = {
        'thumb': 'PIXMAP',
    }

    guitool_test_thumbdir = ut.ensure_app_resource_dir('guitool', 'thumbs')
    ut.delete(guitool_test_thumbdir)
    ut.ensuredir(guitool_test_thumbdir)
    import vtool as vt
    from os.path import join

    # imgname_list = sorted(ut.TESTIMG_URL_DICT.keys())
    imgname_list = ['carl.jpg', 'lena.png', 'patsy.jpg']
    imgname_list += ['doesnotexist.jpg']

    num_imgs = list(range(len(imgname_list)))
    # num_imgs = list(range(500))

    def thread_func(would_be, id_):
        from vtool.fontdemo import get_text_test_img

        get_text_test_img(id_)

    def thumb_getter(id_, thumbsize=128):
        """ Thumb getters must conform to thumbtup structure """
        if id_ not in imgname_list:
            return {
                'fpath': id_ + '.jpg',
                'thread_func': thread_func,
                'main_func': lambda: (id_,),
            }
        # print(id_)
        if id_ == 'doesnotexist.jpg':
            return None
            img_path = None
            img_size = (100, 100)
        else:
            img_path = ut.grab_test_imgpath(id_, verbose=False)
            img_size = vt.open_image_size(img_path)
        thumb_path = join(guitool_test_thumbdir, ut.hashstr(str(img_path)) + '.jpg')
        if id_ == 'carl.jpg':
            bbox_list = [(10, 10, 200, 200)]
            theta_list = [0]
        elif id_ == 'lena.png':
            # bbox_list = [(10, 10, 200, 200)]
            bbox_list = [None]
            theta_list = [None]
        else:
            bbox_list = []
            theta_list = []
        interest_list = [False]
        thumbtup = (thumb_path, img_path, img_size, bbox_list, theta_list, interest_list)
        # print('thumbtup = %r' % (thumbtup,))
        return thumbtup
        # return None

    def imgname_getter(rowid):
        if rowid < len(imgname_list):
            return imgname_list[rowid]
        else:
            return str(rowid)

    col_getter_dict = {
        'rowid': num_imgs,
        'image_name': imgname_getter,
        'thumb': thumb_getter,
    }
    col_ider_dict = {
        'thumb': 'image_name',
    }
    col_setter_dict = {}
    editable_colnames = []
    sortby = 'rowid'

    def get_thumb_size():
        return 128

    # get_thumb_size = lambda: 128  # NOQA
    col_width_dict = {}
    col_bgrole_dict = {}

    api = guitool.CustomAPI(
        col_name_list,
        col_types_dict,
        col_getter_dict,
        col_bgrole_dict,
        col_ider_dict,
        col_setter_dict,
        editable_colnames,
        sortby,
        get_thumb_size,
        True,
        col_width_dict,
    )
    headers = api.make_headers(tblnice='Utool Test Images')

    wgt = guitool.APIItemWidget()
    wgt.change_headers(headers)
    wgt.resize(600, 400)
    # guitool.qtapp_loop(qwin=wgt, ipy=ipy, frequency=loop_freq)
    return wgt


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.guitool.api_thumb_delegate
        python -m wbia.guitool.api_thumb_delegate --allexamples
        python -m wbia.guitool.api_thumb_delegate --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
