# LICENCE
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[geom]', DEBUG=False)


""" Fit the bounding polygon inside a rectangle """
def bboxes_from_vert_list(verts_list):
    def _to_bbox(verts):
        x = min(x[0] for x in verts)
        y = min(y[1] for y in verts)
        w = max(x[0] for x in verts) - x
        h = max(y[1] for y in verts) - y
        return (x, y, w, h)
    return [_to_bbox(verts) for verts in verts_list]


""" Create a four-vertex polygon from the bounding rectangle """
def verts_list_from_bboxes_list(bboxes_list):
    return [((x,     y),
             (x + w, y),
             (x + w, y + h),
             (x,     y + h)) for (x, y, w, h) in bboxes_list]
