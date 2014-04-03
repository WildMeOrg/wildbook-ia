from __future__ import division, print_function
import utool
import drawtool.draw_func2 as df2
import numpy as np
import viz_helpers
from viz_helpers import get_ibsdat, set_ibsdat  # NOQA
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


@utool.indent_decor('[annote_kpts]')
def _annotate_kpts(kpts, color=None, sel_fx=None, **kwargs):
    color = kwargs.get('color', 'distinct' if sel_fx is None else df2.ORANGE)
    # Keypoint drawing kwargs
    drawkpts_kw = kwargs.copy()
    drawkpts_kw.update({
        'ell': True,
        'pts': False,
        'ell_alpha': .4,
        'ell_linewidth': 2,
        'ell_color': color,
    })
    # draw all keypoints
    if sel_fx is None:
        df2.draw_kpts2(kpts, **drawkpts_kw)
    else:
        # dont draw the selected keypoint in this batch
        nonsel_kpts_ = np.vstack((kpts[0:sel_fx], kpts[sel_fx + 1:]))
        # Draw selected keypoint
        sel_kpts = kpts[sel_fx:sel_fx + 1]
        drawkpts2_kw = drawkpts_kw.copy()
        drawkpts2_kw.update({
            'ell_color': df2.BLUE,
            'eig': True,
            'rect': True,
            'ori': True,
        })
        df2.draw_kpts2(nonsel_kpts_, **drawkpts_kw)
        df2.draw_kpts2(sel_kpts, **drawkpts_kw)


def _annote_chip(ibs, rid, in_image, **kwargs):
    # FIXME
    if in_image:
        kpts = viz_helpers.get_imgspace_chip_kpts(ibs, [rid])[0]
    else:
        kpts = ibs.get_chip_kpts(rid)
    # Draw keypoints on chip
    _annotate_kpts(kpts, **kwargs)


@utool.indent_decor('[show_chip]')
def show_chip(ibs, rid, in_image=False, **kwargs):
    printDBG('[viz] show_chip()')
    chip = kwargs.get('chip', ibs.get_roi_images(rid)
                      if in_image else ibs.get_roi_chips(rid))
    # Create chip title
    title_list = []
    title_list += [str(rid)]
    title_list += ['gname=%r' % ibs.get_roi_gnames(rid)]
    title_list += ['name=%r'  % ibs.get_roi_names(rid)]
    title_str = ', '.join(title_list)
    # Draw chip
    fig, ax = df2.imshow(chip, title=title_str, **kwargs)
    # Populate axis user data
    set_ibsdat(ax, 'viztype', 'chip')
    set_ibsdat(ax, 'rid', 'rid')
    # Annotate chip
    _annote_chip(ibs, rid, in_image, **kwargs)


@profile
def show_keypoints(rchip, kpts, fnum=0, pnum=None, **kwargs):
    #printDBG('[df2.show_kpts] %r' % (kwargs.keys(),))
    df2.imshow(rchip, fnum=fnum, pnum=pnum, **kwargs)
    _annotate_kpts(kpts, **kwargs)
    ax = df2.gca()
    set_ibsdat(ax, 'viztype', 'keypoints')
    set_ibsdat(ax, 'kpts', 'kpts')
