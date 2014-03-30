from __future__ import division, print_function
import utool
import drawtool.draw_func2 as df2
import numpy as np
from viz_helpers import get_ibsdat, set_ibsdat  # NOQA
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz-image]',
                                                       DEBUG=False)


@utool.indent_decor('[annote_roi]')
def annotate_roi(ibs, ax, rid, sel_rids, draw_lbls, annote):
    # Draw an roi around a chip in the image
    bbox  = ibs.get_roi_bboxes(rid)
    theta = ibs.get_roi_thetas(rid)
    if annote:
        is_sel = rid in sel_rids
        if draw_lbls:
            label  = ibs.get_roi_names(rid)
            label = rid if label == '____' else label
        else:
            label = None
        lbl_alpha  = .75 if is_sel else .6
        bbox_alpha = .95 if is_sel else .6
        lbl_color  = df2.BLACK * lbl_alpha
        bbox_color = (df2.ORANGE if is_sel else df2.DARK_ORANGE) * bbox_alpha
        df2.draw_roi(bbox, label, bbox_color, lbl_color, theta=theta, ax=ax)
    # Index the roi centers (for interaction)
    (x, y, w, h) = bbox
    xy_center = np.array([x + (w / 2), y + (h / 2)])
    return xy_center


@utool.indent_decor('[annote_image]')
def annotate_image(ibs, ax, gid, sel_rids, draw_lbls, annote):
    # draw chips in the image
    rid_list = ibs.get_rids_in_gids(gid)
    roi_centers = []
    # Draw all chip indexes in the image
    for rid in rid_list:
        xy_center = annotate_roi(ibs, ax, rid, sel_rids, draw_lbls, annote)
        roi_centers.append(xy_center)
    # Put roi centers in the axis
    roi_centers = np.array(roi_centers)
    set_ibsdat(ax, 'roi_centers', roi_centers)
    set_ibsdat(ax, 'rid_list', rid_list)


@utool.indent_decor('[show_image]')
def show_image(ibs, gid, sel_rids=[],
               fnum=1,
               figtitle='Image View',
               annote=True,
               draw_lbls=True,
               **kwargs):
    # Shows an image with annotations
    gname = ibs.get_image_gnames((gid,))
    title = 'gid=%r gname=%r' % (gid, gname)
    img = ibs.get_images(gid)
    fig = df2.figure(fnum=fnum, docla=True)
    fig, ax = df2.imshow(img, title=title, fnum=fnum, **kwargs)
    set_ibsdat(ax, 'viztype', 'image')
    annotate_image(ibs, ax, gid, sel_rids, draw_lbls, annote)
    df2.set_figtitle(figtitle)
