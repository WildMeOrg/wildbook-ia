# LICENCE
from __future__ import absolute_import, division, print_function
from six.moves import zip
import numpy as np
import utool as ut
import cv2
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[geom]', DEBUG=False)


def bboxes_from_vert_list(verts_list, castint=False):
    """ Fit the bounding polygon inside a rectangle """
    return [bbox_from_verts(verts, castint=castint) for verts in verts_list]


def verts_list_from_bboxes_list(bboxes_list):
    """ Create a four-vertex polygon from the bounding rectangle """
    return [verts_from_bbox(bbox) for bbox in bboxes_list]


def verts_from_bbox(bbox, close=False):
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
    """
    if out is None:
        out = np.copy(img_in)
    if isinstance(verts, np.ndarray):
        verts = verts.tolist()
    line_list_sequence = zip(verts[:-1], verts[1:])
    line_tuple_sequence = ((tuple(p1_), tuple(p2_)) for (p1_, p2_) in line_list_sequence)
    cv2.line(out, tuple(verts[0]), tuple(verts[-1]), color, thickness)
    for (p1, p2) in line_tuple_sequence:
        cv2.line(out, p1, p2, color, thickness)
        #print('p1, p2: (%r, %r)' % (p1, p2))
    return out


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
