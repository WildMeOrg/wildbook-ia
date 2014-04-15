from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import utool
import drawtool.draw_func2 as df2
from vtool import image as gtool
from vtool import keypoint as ktool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_sv]', DEBUG=False)


def warp_affine(chip1, Aff, wh2):
    chip1_At = cv2.warpAffine(chip1, Aff[0:2, :], wh2)
    return chip1_At


def warp_homog(chip1, H, wh2):
    chip1_Ht = cv2.warpPerspective(chip1, H, wh2)
    return chip1_Ht


def blend_chips(chip1, chip2):
    assert chip1.shape == chip2.shape, 'chips must be same shape to blend'
    chip_blend = np.zeros(chip2.shape, dtype=chip2.dtype)
    chip_blend = chip1 / 2 + chip2 / 2
    return chip_blend


def show_sv_affine(chip1, chip2, kpts1, kpts2, fm, Aff, aff_inliers, fnum=1, mx=None, **kwargs):
    printDBG('[draw] show_sv_affine')
    printDBG('[draw] len(kpts1) = %r' % len(kpts1))
    printDBG('[draw] len(kpts2) = %r' % len(kpts2))
    printDBG('[draw] len(fm) = %r' % len(fm))
    printDBG('[draw] len(aff_inliers) = %r' % len(aff_inliers))
    printDBG('[draw] aff_inliers = %r' % (aff_inliers,))
    printDBG('[draw] mx = %r' % mx)

    assert len(kpts1) > 0, 'len(kpts1) <= 0'
    assert len(kpts2) > 0, 'len(kpts2) <= 0'
    assert len(fm) > 0, 'len(fm) <= 0'
    kpts1_At = ktool.transform_kpts(kpts1, Aff)
    chip1_At = warp_affine(chip1, Aff, gtool.get_size(chip2))
    chip2_blendA = blend_chips(chip1_At, chip2)
    df2.figure(fnum=fnum, doclf=True, docla=True, **kwargs)
    pnumTop_ = df2.get_pnum_func(2, 2)
    pnumBot_ = df2.get_pnum_func(2, 4)

    in_kwargs = dict(rect=True, ell_alpha=.7, eig=True, ori=True, pts=True)
    out_kwargs = dict(rect=True, ell_alpha=.3, eig=True)

    # INLIER KEYPOINTS
    def draw_inliers(kpts, fxs, color):
        if len(aff_inliers) == 0:
            print('WARNING: NO INLIERS!')
            print('aff_inliers=%r' % aff_inliers)
            return
        df2.draw_kpts2(kpts[fxs[aff_inliers]], color=color, **in_kwargs)
        if mx is not None:
            fx = fxs[mx]
            df2.draw_kpts2(kpts[fx:(fx + 1)], color=color, ell_linewidth=3, **in_kwargs)

    # ORDINARY LINES
    def draw_lines(kpts1, kpts2, fm, color):
        color_list = [df2.DARK_ORANGE for _ in xrange(len(fm))]
        df2.draw_lines2(kpts1, kpts2, fm, color_list=color_list)

    # INLIER LINES
    def draw_inlier_lines(kpts1, kpts2, fm, color):
        df2.draw_lines2(kpts1, kpts2,
                        fm[aff_inliers],
                        color_list=[color],
                        lw=2,
                        line_alpha=1)

    # CHIP WITH KEYPOINTS AND LINES
    def _draw_chip(title, chip, px, kpts1=None, kpts2=None, color=None):
        df2.imshow(chip, title=title, fnum=fnum, pnum=pnumBot_(4 + px), **kwargs)
        if kpts1 is not None:
            df2.draw_kpts2(kpts1, color=df2.DARK_BLUE, **out_kwargs)
            draw_inliers(kpts1, fm[:, 0], df2.BLUE)
        if kpts2 is not None:
            df2.draw_kpts2(kpts2, color=df2.DARK_RED, **out_kwargs)
            draw_inliers(kpts2, fm[:, 1], df2.RED)
        if kpts2 is not None and kpts1 is not None:
            #draw_lines(kpts1, kpts2, fm, df2.DARK_ORANGE)
            draw_inlier_lines(kpts1, kpts2, fm, df2.ORANGE)

    # Draw original matches, affine inliers, and homography inliers
    def _draw_matches(title, fm, px):
        # Helper with common arguments to df2.show_chipmatch2
        dmkwargs = dict(fs=None, title=title, all_kpts=False, draw_lines=True,
                        docla=True, draw_border=True, fnum=fnum, pnum=pnumTop_(px))
        df2.show_chipmatch2(chip1, chip2, kpts1, kpts2, fm, **dmkwargs)

    # Draw the Assigned -> Affine -> Homography matches
    assign_fm = fm
    affine_fm = fm[aff_inliers]
    _draw_matches('%d Assigned matches' % len(assign_fm), assign_fm, 0)
    _draw_matches('%d Affine inliers' % len(affine_fm), affine_fm,   1)

    _draw_chip('Source',      chip1,        0, kpts1=kpts1)
    _draw_chip('Destination', chip2,        1, kpts2=kpts2)
    _draw_chip('Transform',   chip1_At,     2, kpts1=kpts1_At)
    _draw_chip('Aff Blend',   chip2_blendA, 3, kpts1=kpts1_At, kpts2=kpts2)
    df2.adjust_subplots_safe(left=.01, right=.99, wspace=.01)


@utool.indent_func
def show_sv_homog_and_affine(chip1, chip2, kpts1, kpts2, fm, H, inliers, Aff, aff_inliers, fnum=1):
    wh2 = gtool.get_size(chip2)
    chip1_Ht = warp_homog(chip1,    H, wh2)
    chip1_At = warp_affine(chip1, Aff, wh2)
    chip2_blendA = blend_chips(chip1_At, chip2)
    chip2_blendH = blend_chips(chip1_Ht, chip2)

    df2.figure(fnum=fnum, pnum=(3, 4, 1), docla=True, doclf=True)

    def _draw_chip(title, chip, px, *args, **kwargs):
        df2.imshow(chip, *args, title=title, fnum=fnum, pnum=(3, 4, px), **kwargs)

    # Draw original matches, affine inliers, and homography inliers
    def _draw_matches(title, fm, px):
        # Helper with common arguments to df2.show_chipmatch2
        dmkwargs = dict(fs=None, title=title, all_kpts=False, draw_lines=True,
                        docla=True, fnum=fnum, pnum=(3, 3, px))
        df2.show_chipmatch2(chip1, chip2, kpts1, kpts2, fm, **dmkwargs)

    # Draw the Assigned -> Affine -> Homography matches
    assign_fm = fm
    affine_fm = fm[aff_inliers]
    homog_fm = fm[inliers]
    _draw_matches('%d Assigned matches' % len(assign_fm), assign_fm, 1)
    _draw_matches('%d Affine inliers' % len(affine_fm), affine_fm,   2)
    _draw_matches('%d Homography inliers' % len(homog_fm), homog_fm, 3)
    # Draw the Affine Transformations
    _draw_chip('Source', chip1, 5)
    _draw_chip('Destination', chip2, 6)
    _draw_chip('Affine', chip1_At, 7)
    _draw_chip('Aff Blend', chip2_blendA, 8)
    # Draw the Homography Transformation
    _draw_chip('Source', chip1, 9)
    _draw_chip('Destination', chip2, 10)
    _draw_chip('Homog', chip1_Ht, 11)
    _draw_chip('Homog Blend', chip2_blendH, 12)
