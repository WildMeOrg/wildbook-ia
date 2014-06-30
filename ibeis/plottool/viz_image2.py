from __future__ import absolute_import, division, print_function
from itertools import izip
import utool
import plottool.draw_func2 as df2
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_img2]', DEBUG=False)


def annotate_annotation(ax, bbox, theta, label, is_sel):
    """ Draw an annotation around a chip in the image """
    lbl_alpha  = .75 if is_sel else .6
    bbox_alpha = .95 if is_sel else .6
    lbl_color  = df2.BLACK * lbl_alpha
    bbox_color = (df2.ORANGE if is_sel else df2.DARK_ORANGE) * bbox_alpha
    df2.draw_annotation(bbox, label, bbox_color, lbl_color, theta=theta, ax=ax)


def annotate_image(ax, bbox_list=[], theta_list=None, label_list=None,
                   sel_list=None, draw_lbls=True):
    if not draw_lbls:
        label_list = [''] * len(bbox_list)
    if theta_list is None:
        theta_list = [0] * len(bbox_list)
    if label_list is None:
        label_list = map(str, range(len(bbox_list)))
    if sel_list is None:
        sel_list = [False] * len(bbox_list)
    # Draw all bboxes on top on image
    annotation_iter = izip(bbox_list, theta_list, label_list, sel_list)
    for bbox, theta, label, is_sel in annotation_iter:
        annotate_annotation(ax, bbox, theta, label, is_sel)


@utool.indent_func
def show_image(img, bbox_list=[],  title='', theta_list=None,
               label_list=None, sel_list=None, draw_lbls=True,
               fnum=None, annote=True, **kwargs):
    """ Driver function to show images """
    # Shows an image with annotations
    if fnum is None:
        fnum = df2.next_fnum()
    fig, ax = df2.imshow(img, title=title, fnum=fnum, docla=True, **kwargs)
    df2.remove_patches(ax)
    if annote:
        annotate_image(ax, bbox_list, theta_list, label_list, sel_list,
                       draw_lbls)
    return fig, ax
