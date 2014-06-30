from __future__ import absolute_import, division, print_function
from itertools import izip
import utool
import plottool.draw_func2 as df2
from plottool import viz_image2
import numpy as np
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_img]', DEBUG=False)


def annotate_image(ibs, ax, gid, sel_aids, draw_lbls=True, annote=True):
    try:
        # draw chips in the image
        aid_list    = ibs.get_image_aids(gid)
        bbox_list   = ibs.get_annotion_bboxes(aid_list)
        theta_list  = ibs.get_annotion_thetas(aid_list)
        label_list  = vh.get_annotion_labels(ibs, aid_list, draw_lbls)
        annotion_centers = vh.get_bbox_centers(bbox_list)
        sel_list    = [aid in sel_aids for aid in aid_list]

        viz_image2.annotate_image(ax, bbox_list, theta_list, label_list, sel_list, draw_lbls, annote)
        # Draw all chip indexes in the image
        if annote:
            annotion_iter = izip(bbox_list, theta_list, label_list, sel_list)
            for bbox, theta, label, is_sel in annotion_iter:
                viz_image2.annotate_annotion(ax, bbox, theta, label, is_sel)
        # Put annotion centers in the axis
        vh.set_ibsdat(ax, 'annotion_centers', np.array(annotion_centers))
        vh.set_ibsdat(ax, 'aid_list', aid_list)
    except Exception as ex:
        utool.printex(ex, key_list=['ibs', 'ax', 'gid', 'sel_aids'])
        raise


def get_annotion_annotations(ibs, aid_list, sel_aids=[], draw_lbls=True):
    annotekw = {
        'bbox_list'  : ibs.get_annotion_bboxes(aid_list),
        'theta_list' : ibs.get_annotion_thetas(aid_list),
        'label_list' : vh.get_annotion_labels(ibs, aid_list, draw_lbls),
        'sel_list'   : [aid in sel_aids for aid in aid_list],
    }
    return annotekw


@utool.indent_func
def show_image(ibs, gid, sel_aids=[], fnum=None,
               annote=True, draw_lbls=True, **kwargs):
    """ Driver function to show images """
    if fnum is None:
        fnum = df2.next_fnum()
    # Read Image
    img = ibs.get_images(gid)
    aid_list    = ibs.get_image_aids(gid)
    annotekw = get_annotion_annotations(ibs, aid_list, sel_aids, draw_lbls)
    annotion_centers = vh.get_bbox_centers(annotekw['bbox_list'])
    showkw = {
        'title'      : vh.get_image_titles(ibs, gid),
        'annote'     : annote,
        'fnum'       : fnum,
    }
    showkw.update(annotekw)
    fig, ax = viz_image2.show_image(img, **showkw)
    # Label the axis with data
    vh.set_ibsdat(ax, 'annotion_centers', annotion_centers)
    vh.set_ibsdat(ax, 'aid_list', aid_list)
    vh.set_ibsdat(ax, 'viztype', 'image')
    vh.set_ibsdat(ax, 'gid', gid)
    return fig, ax
