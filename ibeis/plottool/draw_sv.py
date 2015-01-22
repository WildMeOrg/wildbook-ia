from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import plottool.draw_func2 as df2
from plottool import custom_constants
from vtool import image as gtool
from vtool import keypoint as ktool
#(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[viz_sv]', DEBUG=False)
ut.noinject(__name__, '[viz_sv]')


#@ut.indent_func
def show_sv(chip1, chip2, kpts1, kpts2, fm, homog_tup=None, aff_tup=None,
            mx=None, show_assign=True, show_lines=True, show_kpts=True, fnum=1, **kwargs):
    """ Visualizes spatial verification """
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
        kpts1_mAt = ktool.transform_kpts(kpts1_m, Aff)
        chip2_blendA = gtool.blend_images(chip1_At, chip2)
    #
    # Get Homog Chips, Keypoints, Inliers
    show_homog = homog_tup is not None
    if show_homog:
        (hom_inliers, Hom) = homog_tup
        kpts1_mHt = ktool.transform_kpts(kpts1_m, Hom)
        chip1_Ht = gtool.warpHomog(chip1, Hom, wh2)
        chip2_blendH = gtool.blend_images(chip1_Ht, chip2)
    #
    # Drawing settings
    nRows  = (show_assign) + (show_aff) + (show_homog)
    nCols1 = (show_assign) + (show_aff) + (show_homog)
    nCols2 = 4
    pnum1_ = df2.get_pnum_func(nRows, nCols1)
    pnum2_ = df2.get_pnum_func(nRows, nCols2)
    in_kwargs  = dict(rect=True,  ell_alpha=.7, eig=False, ori=True, pts=True)
    out_kwargs = dict(rect=False, ell_alpha=.3, eig=False)

    def _draw_kpts(*args, **kwargs):
        if not show_kpts:
            return
        df2.draw_kpts2(*args, **kwargs)

    def draw_inlier_kpts(kpts_m, inliers, color):
        _draw_kpts(kpts_m[inliers], color=color, **in_kwargs)
        if mx is not None:
            _draw_kpts(kpts_m[mx:(mx + 1)], color=color, ell_linewidth=3, **in_kwargs)

    def _draw_matches(px, title, inliers):
        dmkwargs = dict(fs=None, title=title, all_kpts=False, draw_lines=True,
                        docla=True, draw_border=True, fnum=fnum, pnum=pnum1_(px))
        __fm = np.vstack((inliers, inliers)).T
        df2.show_chipmatch2(chip1, chip2, kpts1_m, kpts2_m, __fm, **dmkwargs)
        return px + 1

    def _draw_chip(px, title, chip, inliers, kpts1_m, kpts2_m):
        df2.imshow(chip, title=title, fnum=fnum, pnum=pnum2_(px))
        if kpts1_m is not None:
            _draw_kpts(kpts1_m, color=custom_constants.DARK_BLUE, **out_kwargs)
            draw_inlier_kpts(kpts1_m, inliers, custom_constants.BLUE)
        if kpts2_m is not None:
            _draw_kpts(kpts2_m, color=custom_constants.DARK_RED, **out_kwargs)
            draw_inlier_kpts(kpts2_m, inliers, custom_constants.RED)
        if kpts2_m is not None and kpts1_m is not None and show_lines:
            __fm = np.vstack((inliers, inliers)).T
            df2.draw_lines2(kpts1_m, kpts2_m, __fm, color_list=[custom_constants.ORANGE], lw=2, line_alpha=1)
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
    if show_aff:
        px = _draw_chip(px, 'Source',    chip1,        aff_inliers,   kpts1_m, None)
        px = _draw_chip(px, 'Dest',      chip2,        aff_inliers,      None, kpts2_m)
        px = _draw_chip(px, 'Affine',    chip1_At,     aff_inliers, kpts1_mAt, None)
        px = _draw_chip(px, 'Aff Blend', chip2_blendA, aff_inliers, kpts1_mAt, kpts2_m)
    #
    # Draw the Homography Transformation
    if show_homog:
        px = _draw_chip(px, 'Source',      chip1,        hom_inliers,   kpts1_m, None)
        px = _draw_chip(px, 'Dest',        chip2,        hom_inliers,      None, kpts2_m)
        px = _draw_chip(px, 'Homog',       chip1_Ht,     hom_inliers, kpts1_mHt, None)
        px = _draw_chip(px, 'Homog Blend', chip2_blendH, hom_inliers, kpts1_mHt, kpts2_m)
    #
    # Adjust subplots
    df2.adjust_subplots_safe(left=.01, right=.99, wspace=.01, hspace=.03, bottom=.01)


def draw_svmatch(chip1, chip2, H, kpts1=None, kpts2=None, fm=None, fnum=None,
                 pnum=None, update=False):
    r"""
    Args:
        chip1 (ndarray[uint8_t, ndim=2]):  annotation image data
        chip2 (ndarray[uint8_t, ndim=2]):  annotation image data
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        kpts1 (ndarray[float32_t, ndim=2]):  keypoints
        kpts2 (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        python -m plottool.draw_sv --test-draw_svmatch --show

    Example:
        >>> # DISABLE_DOCTEST (TODO REMOVE IBEIS DOCTEST)
        >>> from ibeis.viz.viz_sver import *  # NOQA
        >>> import plottool as pt
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1 = 1
        >>> aid2 = 3
        >>> H = np.array([[ -4.68815126e-01,   7.80306795e-02,  -2.23674587e+01],
        ...               [  4.54394231e-02,  -7.67438835e-01,   5.92158624e+01],
        ...               [  2.12918867e-04,  -8.64851418e-05,  -6.21472492e-01]])
        >>> chip1, chip2 = ibs.get_annot_chips((aid1, aid2))
        >>> kpts1, kpts2 = None, None  # ibs.get_annot_kpts((aid1, aid2))
        >>> # execute function
        >>> result = draw_svmatch(chip1, chip2, H, kpts1, kpts2)
        >>> # verify results
        >>> print(result)
        >>> pt.show_if_requested()
    """
    import plottool as pt
    wh2 = gtool.get_size(chip2)
    chip1_t = gtool.warpHomog(chip1, H, wh2)
    if kpts1 is not None:
        kpts1_t = ktool.transform_kpts(kpts1, H)
    else:
        kpts1_t = None
    fnum = 1
    #next_pnum = pt.make_pnum_nextgen(1, 2)

    pt.show_chipmatch2(chip1_t, chip2, kpts1_t, kpts2, fm=fm, fnum=fnum,
                       pnum=pnum)

    #if fnum is None:
    #    fnum = pt.next_fnum()
    #pt.figure(fnum=fnum)
    #pt.imshow(chip1_t, fnum=fnum, pnum=next_pnum())
    #if kpts1_t  is not None:
    #    pt.draw_kpts2(kpts1_t)
    #pt.imshow(chip2, fnum=fnum, pnum=next_pnum())
    #if kpts2  is not None:
    #    pt.draw_kpts2(kpts2)
    ##pt.imshow(chip1, fnum=fnum, pnum=next_pnum())
    ##if kpts1  is not None:
    ##    pt.draw_kpts2(kpts1)
    if update:
        pt.update()


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
