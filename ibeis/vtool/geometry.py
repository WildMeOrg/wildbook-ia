# -*- coding: utf-8 -*-
# LICENCE
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
import numpy as np
import utool as ut
try:
    import cv2
except ImportError as ex:
    print('WARNING: import cv2 is failing!')
(print, rrr, profile) = ut.inject2(__name__, '[geom]', DEBUG=False)


def bboxes_from_vert_list(verts_list, castint=False):
    """ Fit the bounding polygon inside a rectangle """
    return [bbox_from_verts(verts, castint=castint) for verts in verts_list]


def verts_list_from_bboxes_list(bboxes_list):
    """ Create a four-vertex polygon from the bounding rectangle """
    return [verts_from_bbox(bbox) for bbox in bboxes_list]


def bbox_center(bbox):
    (x, y, w, h) = bbox
    centerx = x + (w / 2)
    centery = y + (h / 2)
    return centerx, centery


def verts_from_bbox(bbox, close=False):
    r"""
    Args:
        bbox (tuple):  bounding box in the format (x, y, w, h)
        close (bool): (default = False)

    Returns:
        list: verts

    CommandLine:
        python -m vtool.geometry --test-verts_from_bbox

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.geometry import *  # NOQA
        >>> bbox = (10, 10, 50, 50)
        >>> close = False
        >>> verts = verts_from_bbox(bbox, close)
        >>> result = ('verts = %s' % (str(verts),))
        >>> print(result)
        verts = ((10, 10), (60, 10), (60, 60), (10, 60))
    """
    x1, y1, w, h = bbox
    x2 = (x1 + w)
    y2 = (y1 + h)
    if close:
        # Close the verticies list (for drawing lines)
        verts = ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))
    else:
        verts = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return verts


def bbox_from_verts(verts, castint=False):
    x = min(x[0] for x in verts)
    y = min(y[1] for y in verts)
    w = max(x[0] for x in verts) - x
    h = max(y[1] for y in verts) - y
    if castint:
        return (int(x), int(y), int(w), int(h))
    else:
        return (x, y, w, h)


def draw_border(img_in, color=(0, 128, 255), thickness=2, out=None):
    r"""
    Args:
        img_in (ndarray[uint8_t, ndim=2]):  image data
        color (tuple): in bgr
        thickness (int):
        out (None):

    CommandLine:
        python -m vtool.geometry --test-draw_border --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.geometry import *  # NOQA
        >>> import vtool as vt
        >>> img_in = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> color = (0, 128, 255)
        >>> thickness = 20
        >>> out = None
        >>> img = draw_border(img_in, color, thickness, out)
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img)
        >>> pt.show_if_requested()
    """
    h, w = img_in.shape[0:2]
    #verts = verts_from_bbox((0, 0, w, h))
    #verts = verts_from_bbox((0, 0, w - 1, h - 1))
    half_thickness = thickness // 2
    verts = verts_from_bbox((half_thickness, half_thickness,
                             w - thickness, h - thickness))
    # FIXME: adjust verts and draw lines here to fill in the corners correctly
    img = draw_verts(img_in, verts, color=color, thickness=thickness, out=out)
    return img


def draw_verts(img_in, verts, color=(0, 128, 255), thickness=2, out=None):
    r"""
    Args:
        img_in (?):
        verts (?):
        color (tuple):
        thickness (int):

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    CommandLine:
        python -m vtool.geometry --test-draw_verts --show
        python -m vtool.geometry --test-draw_verts:0 --show
        python -m vtool.geometry --test-draw_verts:1 --show

    References:
        http://docs.opencv.org/modules/core/doc/drawing_functions.html#line

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.geometry import *  # NOQA
        >>> import plottool as pt
        >>> import vtool as vt
        >>> # build test data
        >>> img_in = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> verts = ((10, 10), (10, 100), (100, 100), (100, 10))
        >>> color = (0, 128, 255)
        >>> thickness = 2
        >>> # execute function
        >>> out = None
        >>> img = draw_verts(img_in, verts, color, thickness, out)
        >>> assert img_in is not img
        >>> assert out is not img
        >>> assert out is not img_in
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> pt.imshow(img)
        >>> pt.show_if_requested()

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.geometry import *  # NOQA
        >>> import plottool as pt
        >>> import vtool as vt
        >>> # build test data
        >>> img_in = vt.imread(ut.grab_test_imgpath('carl.jpg'))
        >>> verts = ((10, 10), (10, 100), (100, 100), (100, 10))
        >>> color = (0, 128, 255)
        >>> thickness = 2
        >>> out = img_in
        >>> # execute function
        >>> img = draw_verts(img_in, verts, color, thickness, out)
        >>> assert img_in is img, 'should be in place'
        >>> assert out is img, 'should be in place'
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> pt.imshow(img)
        >>> pt.show_if_requested()


    out = img_in = np.zeros((500, 500, 3), dtype=np.uint8)
    """
    if out is None:
        out = np.copy(img_in)
    if isinstance(verts, np.ndarray):
        verts = verts.tolist()
    connect = True
    if connect:
        line_list_sequence = zip(verts[:-1], verts[1:])
        line_tuple_sequence = ((tuple(p1_), tuple(p2_)) for (p1_, p2_) in line_list_sequence)
        cv2.line(out, tuple(verts[0]), tuple(verts[-1]), color, thickness)
        for (p1, p2) in line_tuple_sequence:
            cv2.line(out, p1, p2, color, thickness)
            #print('p1, p2: (%r, %r)' % (p1, p2))
    else:
        for count, p in enumerate(verts, start=1):
            cv2.circle(out, tuple(p), count, color, thickness=1)
    return out


def closest_point_on_line_segment(p, e1, e2):
    """
    >>> import plottool as pt
    >>> pt.ensure_pylab_qt4()
    >>> p_list = np.array([[19, 7], [7, 14], [14, 11], [8, 7], [23, 21]], dtype=np.float)
    >>> pt.plt.plot(p_list.T[0], p_list.T[1], 'ro')
    >>> bbox = np.array([10, 10, 10, 10], dtype=np.float)
    >>> close_pts = np.array([closest_point_on_bbox(p, bbox) for p in p_list])
    >>> pt.plt.plot(close_pts.T[0], close_pts.T[1], 'rx')
    >>> for x, y in list(zip(p_list, close_pts)):
    >>>     pt.plt.plot(x, y, 'r--')
    >>> bbox_verts = np.array(vt.verts_from_bbox(bbox, close=True))
    >>> pt.plt.plot(bbox_verts.T[0], bbox_verts.T[1], 'b-')
    >>> pt.plt.xlim(0, 30)
    >>> pt.plt.ylim(0, 30)
    """
    de = (dx, dy) = e2 - e1  # shift e1 to origin
    pv = p - e1  # make point vector wrt orgin
    mag = np.linalg.norm(de)
    t = pv.dot(de) / mag
    # line segment bounds
    if t < 0:
        new_pt = e1
    elif t > 1:
        new_pt = e2
    else:
        new_pt = e1 + t * de
    return new_pt


def closest_point_on_bbox(p, bbox):
    """

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from vtool.geometry import *  # NOQA
        >>> p_list = np.array([[19, 7], [7, 14], [14, 11], [8, 7], [23, 21]], dtype=np.float)
        >>> bbox = np.array([10, 10, 10, 10], dtype=np.float)
        >>> [closest_point_on_bbox(p, bbox) for p in p_list]
    """
    import vtool as vt
    bbox_verts = np.array(vt.verts_from_bbox(bbox, close=True))
    candidates = [closest_point_on_line_segment(p, e1, e2) for e1, e2 in ut.itertwo(bbox_verts)]
    dists = np.array([vt.L2_sqrd(p, new_pt) for new_pt in candidates])
    new_pts = candidates[dists.argmin()]
    return new_pts


def bbox_from_xywh(xy, wh, xy_rel_pos=[0, 0]):
    """ need to specify xy_rel_pos if xy is not in tl already """
    to_tlx = xy_rel_pos[0] * wh[0]
    to_tly = xy_rel_pos[1] * wh[1]
    tl_x = xy[0] - to_tlx
    tl_y = xy[1] - to_tly
    bbox = [tl_x, tl_y, wh[0], wh[1]]
    return bbox


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.geometry
        python -m vtool.geometry --allexamples
        python -m vtool.geometry --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
