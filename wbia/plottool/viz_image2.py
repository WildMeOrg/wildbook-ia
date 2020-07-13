# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import zip, map
import utool
from . import draw_func2 as df2
from wbia.plottool import custom_constants

# (print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_img2]', DEBUG=False)
utool.noinject(__name__, '[viz_img2]')


def draw_chip_overlay(ax, bbox, theta, text, is_sel):
    """ Draw an annotation around a chip in the image """
    lbl_alpha = 0.75 if is_sel else 0.6
    bbox_alpha = 0.95 if is_sel else 0.6
    lbl_color = custom_constants.BLACK * lbl_alpha
    bbox_color = (
        custom_constants.ORANGE if is_sel else custom_constants.DARK_ORANGE
    ) * bbox_alpha
    df2.draw_bbox(bbox, text, bbox_color, lbl_color, theta=theta, ax=ax)


def draw_image_overlay(
    ax, bbox_list=[], theta_list=None, text_list=None, sel_list=None, draw_lbls=True
):
    if not draw_lbls:
        text_list = [''] * len(bbox_list)
    if theta_list is None:
        theta_list = [0] * len(bbox_list)
    if text_list is None:
        text_list = list(map(str, range(len(bbox_list))))
    if sel_list is None:
        sel_list = [False] * len(bbox_list)
    # Draw all bboxes on top on image
    annotation_iter = zip(bbox_list, theta_list, text_list, sel_list)
    for bbox, theta, text, is_sel in annotation_iter:
        draw_chip_overlay(ax, bbox, theta, text, is_sel)


# @utool.indent_func
def show_image(
    img,
    bbox_list=[],
    title='',
    theta_list=None,
    text_list=None,
    sel_list=None,
    draw_lbls=True,
    fnum=None,
    annote=True,
    **kwargs,
):
    """ Driver function to show images """
    # Shows an image with annotations
    if fnum is None:
        fnum = df2.next_fnum()
    fig, ax = df2.imshow(img, title=title, fnum=fnum, docla=True, **kwargs)
    df2.remove_patches(ax)
    if annote:
        draw_image_overlay(ax, bbox_list, theta_list, text_list, sel_list, draw_lbls)
    return fig, ax
