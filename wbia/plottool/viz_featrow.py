# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import six

# import itertools
from . import draw_func2 as df2
from wbia.plottool import plot_helpers as ph
from wbia.plottool import custom_constants
from wbia.plottool import custom_figure

# (print, print_, printDBG, rrr, profile) = ut.inject(
#    __name__, '[viz_featrow]', DEBUG=False)
ut.noinject(__name__, '[viz_featrow]')


def precisionstr(c='E', pr=2):
    return '%.' + str(pr) + c


def formatdist(val):
    pr = 3
    if val > 1000:
        return precisionstr('E', pr) % val
    elif val > 0.01 or val == 0:
        return precisionstr('f', pr) % val
    else:
        return precisionstr('e', pr) % val


# @ut.indent_func
def draw_feat_row(
    chip,
    fx,
    kp,
    sift,
    fnum,
    nRows,
    nCols=None,
    px=None,
    prevsift=None,
    origsift=None,
    aid=None,
    info='',
    type_=None,
    shape_labels=False,
    vecfield=False,
    multicolored_arms=False,
    draw_chip=False,
    draw_warped=True,
    draw_unwarped=True,
    draw_desc=True,
    rect=True,
    ori=True,
    pts=False,
    **kwargs
):
    """
    draw_feat_row

    SeeAlso:
        wbia.viz.viz_nearest_descriptors
        ~/code/wbia/wbia/viz/viz_nearest_descriptors.py

    CommandLine:

        # Use this to find the fx you want to visualize
        python -m wbia.plottool.interact_keypoints --test-ishow_keypoints --show --fname zebra.png

        # Use this to visualize the featrow
        python -m wbia.plottool.viz_featrow --test-draw_feat_row --show
        python -m wbia.plottool.viz_featrow --test-draw_feat_row --show --fname zebra.png --fx=121 --feat-all --no-sift
        python -m wbia.plottool.viz_featrow --test-draw_feat_row --dpath figures --save ~/latex/crall-candidacy-2015/figures/viz_featrow.jpg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.viz_featrow import *  # NOQA
        >>> import wbia.plottool as pt
        >>> # build test data
        >>> kpts, vecs, imgBGR = pt.viz_keypoints.testdata_kpts()
        >>> chip = imgBGR
        >>> print('There are %d features' % (len(vecs)))
        >>> fx = ut.get_argval('--fx', type_=int, default=0)
        >>> kp = kpts[fx]
        >>> sift = vecs[fx]
        >>> fnum = 1
        >>> nRows = 1
        >>> nCols = 2
        >>> px = 0
        >>> if True:
        >>>     from wbia.scripts.thesis import TMP_RC
        >>>     import matplotlib as mpl
        >>>     mpl.rcParams.update(TMP_RC)
        >>> hack = ut.get_argflag('--feat-all')
        >>> sift = sift if not ut.get_argflag('--no-sift') else None
        >>> draw_desc = sift is not None
        >>> kw = dict(
        >>>     prevsift=None, origsift=None, aid=None, info='', type_=None,
        >>>     shape_labels=False, vecfield=False, multicolored_arms=True,
        >>>     draw_chip=hack, draw_unwarped=hack, draw_warped=True, draw_desc=draw_desc
        >>> )
        >>> # execute function
        >>> result = draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px,
        >>>                           rect=False, ori=False, pts=False, **kw)
        >>> # verify results
        >>> print(result)
        >>> pt.show_if_requested()
    """
    import numpy as np
    import vtool as vt

    # should not need ncols here

    if nCols is not None:
        if ut.VERBOSE:
            print('Warning nCols is no longer needed')
    # assert nCols_ == nCols
    nCols = draw_chip + draw_unwarped + draw_warped + draw_desc

    pnum_ = df2.make_pnum_nextgen(nRows, nCols, start=px)

    # pnum_ = df2.get_pnum_func(nRows, nCols, base=1)
    # countgen = itertools.count(1)

    # pnumgen_ = df2.make_pnum_nextgen(nRows, nCols, base=1)

    def _draw_patch(**kwargs):
        return df2.draw_keypoint_patch(
            chip,
            kp,
            sift,
            rect=rect,
            ori=ori,
            pts=pts,
            ori_color=custom_constants.DEEP_PINK,
            multicolored_arms=multicolored_arms,
            **kwargs
        )

    # Feature strings
    xy_str, shape_str, scale, ori_str = ph.kp_info(kp)

    if draw_chip:
        pnum = pnum_()
        df2.imshow(chip, fnum=fnum, pnum=pnum)
        kpts_kw = dict(ell_linewidth=1, ell_alpha=1.0)
        kpts_kw.update(kwargs)
        df2.draw_kpts2([kp], **kpts_kw)

    if draw_unwarped:
        # Draw the unwarped selected feature
        # ax = _draw_patch(fnum=fnum, pnum=pnum_(px + six.next(countgen)))
        # pnum = pnum_(px + six.next(countgen)
        pnum = pnum_()
        ax = _draw_patch(fnum=fnum, pnum=pnum)
        ph.set_plotdat(ax, 'viztype', 'unwarped')
        ph.set_plotdat(ax, 'aid', aid)
        ph.set_plotdat(ax, 'fx', fx)
        if shape_labels:
            unwarped_lbl = 'affine feature invV =\n' + shape_str + '\n' + ori_str
            custom_figure.set_xlabel(unwarped_lbl, ax)

    if draw_warped:
        # Draw the warped selected feature
        # ax = _draw_patch(fnum=fnum, pnum=pnum_(px + six.next(countgen)), warped=True)
        pnum = pnum_()
        ax = _draw_patch(fnum=fnum, pnum=pnum, warped=True, **kwargs)
        ph.set_plotdat(ax, 'viztype', 'warped')
        ph.set_plotdat(ax, 'aid', aid)
        ph.set_plotdat(ax, 'fx', fx)
        if shape_labels:
            warped_lbl = ('warped feature\n' + 'fx=%r scale=%.1f\n' + '%s') % (
                fx,
                scale,
                xy_str,
            )
        else:
            warped_lbl = ''
        warped_lbl += info
        custom_figure.set_xlabel(warped_lbl, ax)

    if draw_desc:
        border_color = {
            'None': None,
            'query': None,
            'match': custom_constants.BLUE,
            'norm': custom_constants.ORANGE,
        }.get(str(type_).lower(), None)
        if border_color is not None:
            df2.draw_border(ax, color=border_color)

        # Draw the SIFT representation
        # pnum = pnum_(px + six.next(countgen))
        pnum = pnum_()
        sift_as_vecfield = ph.SIFT_OR_VECFIELD or vecfield
        if sift_as_vecfield:
            custom_figure.figure(fnum=fnum, pnum=pnum)
            df2.draw_keypoint_gradient_orientations(chip, kp, sift=sift)
        else:
            if sift.dtype.type == np.uint8:
                # sigtitle =  'sift histogram' if (px % 3) == 0 else ''
                # ax = df2.plot_sift_signature(sift, sigtitle, fnum=fnum, pnum=pnum)
                ax = df2.plot_sift_signature(sift, fnum=fnum, pnum=pnum)
            else:
                # sigtitle =  'descriptor vector' if (px % 3) == 0 else ''
                ax = df2.plot_descriptor_signature(sift, fnum=fnum, pnum=pnum)
            ax._hs_viztype = 'histogram'
        # dist_list = ['L1', 'L2', 'hist_isect', 'emd']
        # dist_list = ['L2', 'hist_isect']
        # dist_list = ['L2']
        # dist_list = ['bar_L2_sift', 'cos_sift']
        # dist_list = ['L2_sift', 'bar_cos_sift']
        dist_list = ['L2_sift']
        dist_str_list = []
        if origsift is not None:
            distmap_orig = vt.compute_distances(sift, origsift, dist_list)
            dist_str_list.append(
                'query_dist: '
                + ', '.join(
                    [
                        '(%s, %s)' % (key, formatdist(val))
                        for key, val in six.iteritems(distmap_orig)
                    ]
                )
            )
        if prevsift is not None:
            distmap_prev = vt.compute_distances(sift, prevsift, dist_list)
            dist_str_list.append(
                'prev_dist: '
                + ', '.join(
                    [
                        '(%s, %s)' % (key, formatdist(val))
                        for key, val in six.iteritems(distmap_prev)
                    ]
                )
            )
        dist_str = '\n'.join(dist_str_list)
        custom_figure.set_xlabel(dist_str)
    return px + nCols


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.plottool.viz_featrow
        python -m wbia.plottool.viz_featrow --allexamples
        python -m wbia.plottool.viz_featrow --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
