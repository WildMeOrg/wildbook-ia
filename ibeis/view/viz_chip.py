from __future__ import division, print_function
import utool
import drawtool.draw_func2 as df2
import numpy as np
import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


@profile
def show_keypoints(rchip, kpts, fnum=0, pnum=None, **kwargs):
    #printDBG('[df2.show_kpts] %r' % (kwargs.keys(),))
    fig, ax = df2.imshow(rchip, fnum=fnum, pnum=pnum, **kwargs)
    _annotate_kpts(kpts, **kwargs)
    vh.set_ibsdat(ax, 'viztype', 'keypoints')
    vh.set_ibsdat(ax, 'kpts', kpts)


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


@utool.indent_decor('[show_chip]')
def show_chip(ibs, cid, in_image=False, fnum=2, **kwargs):
    """ Driver function to show chips """
    printDBG('[viz] show_chip()')
    # Get chip
    chip = vh.get_chips(ibs, cid, in_image, **kwargs)
    # Get Keypoints
    kpts = vh.get_kpts(ibs, cid, in_image, **kwargs)
    # Create chip title
    title_str = vh.get_chip_titles(ibs, cid)
    # Draw chip
    fig, ax = df2.imshow(chip, title=title_str, fnum=fnum, **kwargs)
    # Populate axis user data
    vh.set_ibsdat(ax, 'viztype', 'chip')
    vh.set_ibsdat(ax, 'cid', cid)
    # Draw keypoints
    _annotate_kpts(kpts, **kwargs)
