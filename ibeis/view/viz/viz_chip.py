from __future__ import absolute_import, division, print_function
import utool
import drawtool.draw_func2 as df2
import numpy as np
from . import viz_helpers as vh
from . import viz_image
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


def show_keypoints(chip, kpts, fnum=0, pnum=None, **kwargs):
    #printDBG('[df2.show_kpts] %r' % (kwargs.keys(),))
    fig, ax = df2.imshow(chip, fnum=fnum, pnum=pnum, **kwargs)
    _annotate_kpts(kpts, **kwargs)
    vh.set_ibsdat(ax, 'viztype', 'keypoints')
    vh.set_ibsdat(ax, 'kpts', kpts)
    if kwargs.get('ddd', False):
        vh.draw()


show_kpts = show_keypoints


@utool.indent_func
def _annotate_kpts(kpts, sel_fx=None, **kwargs):
    color = kwargs.get('color', 'distinct' if sel_fx is None else df2.ORANGE)
    # Keypoint drawing kwargs
    drawkpts_kw = {
        'ell': True,
        'pts': False,
        'ell_alpha': .4,
        'ell_linewidth': 2,
        'ell_color': color,
    }
    drawkpts_kw.update(kwargs)

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
            'eig':  True,
            'rect': True,
            'ori':  True,
        })
        df2.draw_kpts2(nonsel_kpts_, **drawkpts_kw)
        df2.draw_kpts2(sel_kpts, **drawkpts_kw)


@utool.indent_func
def show_chip(ibs, rid, in_image=False, **kwargs):
    """ Driver function to show chips """
    printDBG('[viz] show_chip()')
    # Get chip
    chip = vh.get_chips(ibs, rid, in_image, **kwargs)
    # Get Keypoints
    kpts = vh.get_kpts(ibs, rid, in_image, **kwargs)
    # Create chip title
    title_str = vh.get_chip_labels(ibs, rid, **kwargs)
    # Draw chip
    fig, ax = df2.imshow(chip, title=title_str, **kwargs)
    # Populate axis user data
    vh.set_ibsdat(ax, 'viztype', 'chip')
    vh.set_ibsdat(ax, 'rid', rid)
    # Draw keypoints
    _annotate_kpts(kpts, **kwargs)
    if in_image:
        gid = ibs.get_roi_gids(rid)
        viz_image.annotate_image(ibs, ax, gid, [rid])
