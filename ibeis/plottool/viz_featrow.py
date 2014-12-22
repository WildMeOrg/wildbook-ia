from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import six
import plottool.draw_func2 as df2
from plottool import plot_helpers as ph
from plottool import custom_constants
from plottool import custom_figure
#(print, print_, printDBG, rrr, profile) = utool.inject(
#    __name__, '[viz_featrow]', DEBUG=False)
utool.noinject(__name__, '[viz_featrow]')


#@utool.indent_func
def draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px, prevsift=None,
                  aid=None, info='', type_=None):
    """
    draw_feat_row

    Args:
        chip (?):
        fx (?):
        kp (?):
        sift (?):
        fnum (?):
        nRows (?):
        nCols (?):
        px (?):
        prevsift (None):
        aid (None):
        info (str):
        type_ (None):

    Returns:
        int : subplot index

    Example:
        >>> from plottool.viz_featrow import *  # NOQA
        >>> chip = '?'
        >>> fx = '?'
        >>> kp = '?'
        >>> sift = '?'
        >>> fnum = '?'
        >>> nRows = '?'
        >>> nCols = '?'
        >>> px = '?'
        >>> prevsift = None
        >>> aid = None
        >>> info = ''
        >>> type_ = None
        >>> result = draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px, prevsift, aid, info, type_)
        >>> print(result)
    """

    pnum_ = df2.get_pnum_func(nRows, nCols, base=1)

    def _draw_patch(**kwargs):
        return df2.draw_keypoint_patch(chip, kp, sift, ori_color=custom_constants.DEEP_PINK, **kwargs)

    # Feature strings
    xy_str, shape_str, scale, ori_str = ph.kp_info(kp)

    # Draw the unwarped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 1))
    ph.set_plotdat(ax, 'viztype', 'unwarped')
    ph.set_plotdat(ax, 'aid', aid)
    ph.set_plotdat(ax, 'fx', fx)
    unwarped_lbl = 'affine feature invV =\n' + shape_str + '\n' + ori_str
    custom_figure.set_xlabel(unwarped_lbl, ax)

    # Draw the warped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 2), warped=True)
    ph.set_plotdat(ax, 'viztype', 'warped')
    ph.set_plotdat(ax, 'aid', aid)
    ph.set_plotdat(ax, 'fx', fx)
    warped_lbl = ('warped feature\n' +
                  'fx=%r scale=%.1f\n' +
                  '%s' + info) % (fx, scale, xy_str)
    custom_figure.set_xlabel(warped_lbl, ax)

    border_color = {None: None,
                    'query': None,
                    'match': custom_constants.BLUE,
                    'norm': custom_constants.ORANGE}[type_]
    if border_color is not None:
        df2.draw_border(ax, color=border_color)

    # Draw the SIFT representation
    pnum = pnum_(px + 3)
    if ph.SIFT_OR_VECFIELD:
        custom_figure.figure(fnum=fnum, pnum=pnum)
        df2.draw_keypoint_gradient_orientations(chip, kp, sift=sift)
    else:
        sigtitle = '' if px != 3 else 'sift histogram'
        ax = df2.plot_sift_signature(sift, sigtitle, fnum=fnum, pnum=pnum)
        ax._hs_viztype = 'histogram'
        if prevsift is not None:
            #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
            #dist_list = ['L2', 'hist_isect']
            dist_list = ['L2']
            distmap = utool.compute_distances(sift, prevsift, dist_list)
            dist_str = ', '.join(['(%s, %.2E)' % (key, val) for key, val in six.iteritems(distmap)])
            custom_figure.set_xlabel(dist_str)
    return px + nCols
