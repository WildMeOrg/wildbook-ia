from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import plottool.draw_func2 as df2
from plottool import custom_constants
from vtool import image as gtool
#from vtool import keypoint as ktool
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[viz_sv]', DEBUG=False)
ut.noinject(__name__, '[viz_sv]')


#@ut.indent_func
def show_sv(chip1, chip2, kpts1, kpts2, fm, homog_tup=None, aff_tup=None,
            mx=None, show_assign=True, show_lines=True, show_kpts=True, fnum=1, **kwargs):
    """ Visualizes spatial verification """
    #import plottool as pt
    # GEt Matching chips
    kpts1_m = kpts1[fm.T[0]]
    kpts2_m = kpts2[fm.T[1]]
    wh2 = gtool.get_size(chip2)
    #
    # Get Affine Chips, Keypoints, Inliers
    show_aff   = aff_tup is not None
    if show_aff:
        (aff_inliers, Aff) = aff_tup
        chip1_At = gtool.warpAffine(chip1, Aff, wh2)
        #kpts1_mAt = ktool.transform_kpts(kpts1_m, Aff)
        chip2_blendA = gtool.blend_images(chip1_At, chip2)
    #
    # Get Homog Chips, Keypoints, Inliers
    show_homog = homog_tup is not None
    if show_homog:
        (hom_inliers, H) = homog_tup
        #kpts1_mHt = ktool.transform_kpts(kpts1_m, H)
        chip1_Ht = gtool.warpHomog(chip1, H, wh2)
        chip2_blendH = gtool.blend_images(chip1_Ht, chip2)
    #
    # Drawing settings
    nRows  = (show_assign) + (show_aff) + (show_homog)
    nCols1 = (show_assign) + (show_aff) + (show_homog)
    nCols2 = 2
    pnum1_ = df2.get_pnum_func(nRows, nCols1)
    pnum2_ = df2.get_pnum_func(nRows, nCols2)
    #in_kwargs  = dict(rect=True,  ell_alpha=.7, eig=False, ori=True, pts=True)
    #out_kwargs = dict(rect=False, ell_alpha=.3, eig=False)
    in_kwargs  = dict(rect=False,  ell_alpha=.7, eig=False, ori=False, pts=False)
    out_kwargs = dict(rect=False, ell_alpha=.7, eig=False, ori=False)

    def _draw_kpts(*args, **kwargs):
        if not show_kpts:
            return
        df2.draw_kpts2(*args, **kwargs)

    def draw_inlier_kpts(kpts_m, inliers, color, H=None):
        _draw_kpts(kpts_m[inliers], color=color, H=H, **in_kwargs)
        if mx is not None:
            _draw_kpts(kpts_m[mx:(mx + 1)], color=color, ell_linewidth=3, H=H, **in_kwargs)

    def _draw_matches(px, title, inliers):
        dmkwargs = dict(fs=None, title=title, all_kpts=False, draw_lines=True,
                        docla=True, draw_border=True, fnum=fnum, pnum=pnum1_(px), colors=df2.ORANGE)
        __fm = np.vstack((inliers, inliers)).T
        df2.show_chipmatch2(chip1, chip2, kpts1_m, kpts2_m, __fm, **dmkwargs)
        return px + 1

    from plottool import color_funcs
    colors = df2.distinct_colors(2, brightness=.95)
    color1, color2 = colors[0], colors[1]
    color1_dark = color_funcs.darken_rgb(color1, .2)
    color2_dark = color_funcs.darken_rgb(color2, .2)

    def _draw_chip(px, title, chip, inliers, kpts1_m, kpts2_m, H1=None):
        if isinstance(px, tuple):
            pnum = px
            df2.imshow(chip, title=title, fnum=fnum, pnum=pnum)
            px = pnum[2]
        else:
            df2.imshow(chip, title=title, fnum=fnum, pnum=pnum2_(px))
        if kpts1_m is not None:
            _draw_kpts(kpts1_m, color=color1_dark, H=H1, **out_kwargs)
            draw_inlier_kpts(kpts1_m, inliers, color1, H=H1)
        if kpts2_m is not None:
            _draw_kpts(kpts2_m, color=color2_dark, **out_kwargs)
            draw_inlier_kpts(kpts2_m, inliers, color2)
        if kpts2_m is not None and kpts1_m is not None and show_lines:
            __fm = np.vstack((inliers, inliers)).T
            df2.draw_lines2(kpts1_m, kpts2_m, __fm, color_list=[custom_constants.ORANGE], lw=2, line_alpha=1,
                            H1=H1)
        return px + 1
    #
    # Begin the drawing
    df2.figure(fnum=fnum, pnum=(nRows, nCols1, 1), docla=True, doclf=True)
    px = 0
    if show_assign:
        # Draw the Assigned -> Affine -> Homography matches
        px = _draw_matches(px, '%d Assigned matches  ' % len(fm), np.arange(len(fm)))
        if show_aff:
            px = _draw_matches(px, '%d Affine inliers    ' % len(aff_inliers), aff_inliers)
        if show_homog:
            px = _draw_matches(px, '%d Homography inliers' % len(hom_inliers),  hom_inliers)
    #
    # Draw the Affine Transformations
    px = nCols2 * show_assign
    #if show_aff or show_homog:

    if show_aff:
        #px = _draw_chip(px, 'Source',    chip1,        aff_inliers,   kpts1_m, None)
        #px = _draw_chip(px, 'Dest',      chip2,        aff_inliers,   None, kpts2_m)
        px = _draw_chip(px, 'Affine Warped',  chip1_At,     aff_inliers, kpts1_m, None, H1=Aff)
        px = _draw_chip(px, 'Aff Blend', chip2_blendA, aff_inliers, kpts1_m, kpts2_m, H1=Aff)
        #px = _draw_chip(px, 'Affine',    chip1_At,     aff_inliers, kpts1_mAt, None)
        #px = _draw_chip(px, 'Aff Blend', chip2_blendA, aff_inliers, kpts1_mAt, kpts2_m)
        pass
    #
    # Draw the Homography Transformation
    if show_homog:
        #px = _draw_chip(px, 'Source',      chip1,        hom_inliers,   kpts1_m, None)
        #px = _draw_chip(px, 'Dest',        chip2,        hom_inliers,   None, kpts2_m)
        #px = _draw_chip(px, 'Homog',       chip1_Ht,     hom_inliers, kpts1_mHt, None)
        #px = _draw_chip(px, 'Homog Blend', chip2_blendH, hom_inliers, kpts1_mHt, kpts2_m)
        px = _draw_chip(px, 'Homog Warped',  chip1_Ht,     hom_inliers, kpts1_m, None, H1=H)
        px = _draw_chip(px, 'Homog Blend', chip2_blendH, hom_inliers, kpts1_m, kpts2_m, H1=H)
    #
    # Adjust subplots
    df2.adjust_subplots_safe(left=.01, right=.99, wspace=.01, hspace=.1, bottom=.01)


if __name__ == '__main__':
    """
    CommandLine:
        python -m plottool.draw_sv
        python -m plottool.draw_sv --allexamples
        python -m plottool.draw_sv --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
