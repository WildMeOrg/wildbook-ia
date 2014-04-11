from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import drawtool.draw_func2 as df2
# IBEIS
from . import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz-featrow]', DEBUG=False)


@utool.indent_decor('[viz.draw_feat_row]')
def draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, prevsift=None,
                  cid=None, info='', type_=None):
    pnum_ = lambda px: (nRows, nCols, px)

    def _draw_patch(**kwargs):
        return df2.draw_keypoint_patch(rchip, kp, sift, ori_color=df2.DEEP_PINK, **kwargs)

    # Feature strings
    xy_str, shape_str, scale, ori_str = vh.kp_info(kp)

    # Draw the unwarped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 1))
    ax._hs_viewtype = 'unwarped'
    ax._hs_cid = cid
    ax._hs_fx = fx
    unwarped_lbl = 'affine feature invV =\n' + shape_str + '\n' + ori_str
    df2.set_xlabel(unwarped_lbl, ax)

    # Draw the warped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 2), warped=True)
    ax._hs_viewtype = 'warped'
    ax._hs_cid = cid
    ax._hs_fx = fx
    warped_lbl = ('warped feature\n' +
                  'fx=%r scale=%.1f\n' +
                  '%s' + info) % (fx, scale, xy_str)
    df2.set_xlabel(warped_lbl, ax)

    border_color = {None: None,
                    'query': None,
                    'match': df2.BLUE,
                    'norm': df2.ORANGE}[type_]
    if border_color is not None:
        df2.draw_border(ax, color=border_color)

    # Draw the SIFT representation
    pnum = pnum_(px + 3)
    if vh.SIFT_OR_VECFIELD:
        df2.figure(fnum=fnum, pnum=pnum)
        df2.draw_keypoint_gradient_orientations(rchip, kp, sift=sift)
    else:
        sigtitle = '' if px != 3 else 'sift histogram'
        ax = df2.plot_sift_signature(sift, sigtitle, fnum=fnum, pnum=pnum)
        ax._hs_viewtype = 'histogram'
        if prevsift is not None:
            #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
            #dist_list = ['L2', 'hist_isect']
            dist_list = ['L2']
            distmap = utool.compute_distances(sift, prevsift, dist_list)
            dist_str = ', '.join(['(%s, %.2E)' % (key, val) for key, val in distmap.iteritems()])
            df2.set_xlabel(dist_str)
    return px + nCols
