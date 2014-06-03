from __future__ import absolute_import, division, print_function
from itertools import izip
import utool
import plottool.draw_func2 as df2
from plottool import viz_image2
import numpy as np
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_img]', DEBUG=False)


def annotate_image(ibs, ax, gid, sel_rids, draw_lbls=True, annote=True):
    try:
        # draw chips in the image
        rid_list    = ibs.get_image_rids(gid)
        bbox_list   = ibs.get_roi_bboxes(rid_list)
        theta_list  = ibs.get_roi_thetas(rid_list)
        label_list  = vh.get_roi_labels(ibs, rid_list, draw_lbls)
        roi_centers = vh.get_bbox_centers(bbox_list)
        sel_list    = [rid in sel_rids for rid in rid_list]

        viz_image2.annotate_image(ax, bbox_list, theta_list, label_list, sel_list, draw_lbls, annote)
        # Draw all chip indexes in the image
        if annote:
            roi_iter = izip(bbox_list, theta_list, label_list, sel_list)
            for bbox, theta, label, is_sel in roi_iter:
                viz_image2.annotate_roi(ax, bbox, theta, label, is_sel)
        # Put roi centers in the axis
        vh.set_ibsdat(ax, 'roi_centers', np.array(roi_centers))
        vh.set_ibsdat(ax, 'rid_list', rid_list)
    except Exception as ex:
        utool.printex(ex, key_list=['ibs', 'ax', 'gid', 'sel_rids'])
        raise


def get_roi_annotations(ibs, rid_list, sel_rids=[], draw_lbls=True):
    annotekw = {
        'bbox_list'  : ibs.get_roi_bboxes(rid_list),
        'theta_list' : ibs.get_roi_thetas(rid_list),
        'label_list' : vh.get_roi_labels(ibs, rid_list, draw_lbls),
        'sel_list'   : [rid in sel_rids for rid in rid_list],
    }
    return annotekw


@utool.indent_func
def show_image(ibs, gid, sel_rids=[], fnum=None,
               annote=True, draw_lbls=True, **kwargs):
    """ Driver function to show images """
    if fnum is None:
        fnum = df2.next_fnum()
    # Read Image
    img = ibs.get_images(gid)
    rid_list    = ibs.get_image_rids(gid)
    annotekw = get_roi_annotations(ibs, rid_list, sel_rids, draw_lbls)
    roi_centers = vh.get_bbox_centers(annotekw['bbox_list'])
    showkw = {
        'title'      : vh.get_image_titles(ibs, gid),
        'annote'     : annote,
        'fnum'       : fnum,
    }
    showkw.update(annotekw)
    fig, ax = viz_image2.show_image(img, **showkw)
    # Label the axis with data
    vh.set_ibsdat(ax, 'roi_centers', roi_centers)
    vh.set_ibsdat(ax, 'rid_list', rid_list)
    vh.set_ibsdat(ax, 'viztype', 'image')
    vh.set_ibsdat(ax, 'gid', gid)
    return fig, ax
