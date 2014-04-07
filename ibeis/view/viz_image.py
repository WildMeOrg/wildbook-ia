from __future__ import division, print_function
from itertools import izip
import utool
import drawtool.draw_func2 as df2
import numpy as np
import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_img]',
                                                       DEBUG=False)


def annotate_roi(ax, bbox, theta, label, is_sel):
    # Draw an roi around a chip in the image
    lbl_alpha  = .75 if is_sel else .6
    bbox_alpha = .95 if is_sel else .6
    lbl_color  = df2.BLACK * lbl_alpha
    bbox_color = (df2.ORANGE if is_sel else df2.DARK_ORANGE) * bbox_alpha
    df2.draw_roi(bbox, label, bbox_color, lbl_color, theta=theta, ax=ax)


def annotate_image(ibs, ax, gid, sel_rids, draw_lbls, annote):
    # draw chips in the image
    rid_list    = ibs.get_rids_in_gids(gid)
    bbox_list   = ibs.get_roi_bboxes(rid_list)
    theta_list  = ibs.get_roi_thetas(rid_list)
    label_list  = vh.get_roi_labels(ibs, rid_list, draw_lbls)
    roi_centers = vh.get_bbox_centers(bbox_list)
    sel_list  = [rid in sel_rids for rid in rid_list]
    # Draw all chip indexes in the image
    if annote:
        for bbox, theta, label, is_sel in izip(bbox_list, theta_list, label_list, sel_list):
            annotate_roi(ax, bbox, theta, label, is_sel)
    # Put roi centers in the axis
    vh.set_ibsdat(ax, 'roi_centers', np.array(roi_centers))
    vh.set_ibsdat(ax, 'rid_list', rid_list)


@utool.indent_decor('[show_image]')
def show_image(ibs, gid, sel_rids=[],
               fnum=1,
               figtitle='Image View',
               annote=True,
               draw_lbls=True,
               **kwargs):
    # Shows an image with annotations
    title = vh.get_image_titles(ibs, gid)
    img = ibs.get_images(gid)
    fig, ax = df2.imshow(img, title=title, fnum=fnum, docla=True, **kwargs)
    vh.set_ibsdat(ax, 'viztype', 'image')
    annotate_image(ibs, ax, gid, sel_rids, draw_lbls, annote)
    df2.set_figtitle(figtitle)
