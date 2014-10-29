from __future__ import absolute_import, division, print_function
# UTool
import utool
# Drawtool
import plottool.draw_func2 as df2
import plottool.plot_helpers as ph
# IBEIS
from . import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_matches]', DEBUG=False)


@utool.indent_func
def show_matches(ibs, qres, aid2, sel_fm=[], **kwargs):
    """ shows single annotated match result. """
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.get('draw_fmatches', True)
    aid1 = qres.qaid
    fm = qres.aid2_fm.get(aid2, [])
    fs = qres.aid2_fs.get(aid2, [])
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [aid1, aid2], **kwargs)
    if draw_fmatches:
        kpts1, kpts2 = vh.get_kpts( ibs, [aid1, aid2], **kwargs)
    else:
        kpts1, kpts2 = None, None

    # Build annotation strings / colors
    lbl1 = 'q' + vh.get_aidstrs(aid1)
    lbl2 = vh.get_aidstrs(aid2)
    if in_image:  # HACK!
        lbl1 = None
        lbl2 = None
    # Draws the chips and keypoint matches
    try:
        ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2,
                                               fm, fs=fs, lbl1=lbl1, lbl2=lbl2,
                                               **kwargs)
    except Exception as ex:
        utool.printex(ex, 'consider qr.remove_corrupted_queries',
                      '[viz_matches]')
        print('')
        raise
    (x1, y1, w1, h1) = xywh1
    (x2, y2, w2, h2) = xywh2
    if len(sel_fm) > 0:
        # Draw any selected matches
        sm_kw = dict(rect=True, colors=df2.BLUE)
        df2.plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_matches(ibs, qres, aid2, xywh2=xywh2, xywh1=xywh1,
                     offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_func
def annotate_matches(ibs, qres, aid2,
                     offset1=(0, 0),
                     offset2=(0, 0),
                     xywh2=(0, 0, 0, 0),
                     xywh1=(0, 0, 0, 0),
                     **kwargs):
    # TODO Use this function when you clean show_matches
    in_image    = kwargs.get('in_image', False)
    show_query  = kwargs.get('show_query', True)
    draw_border = kwargs.get('draw_border', True)
    draw_lbl    = kwargs.get('draw_lbl', True)

    printDBG('[viz] annotate_matches()')
    aid1 = qres.qaid
    truth = ibs.get_match_truth(aid1, aid2)
    truth_color = vh.get_truth_color(truth)
    # Build title
    title = vh.get_query_text(ibs, qres, aid2, truth, **kwargs)
    # Build xlbl
    ax = df2.gca()
    ph.set_plotdat(ax, 'viztype', 'matches')
    ph.set_plotdat(ax, 'qaid', aid1)
    ph.set_plotdat(ax, 'aid1', aid1)
    ph.set_plotdat(ax, 'aid2', aid2)
    if draw_lbl:
        name1, name2 = ibs.get_annot_names([aid1, aid2])
        lbl1 = repr(name1)  + ' : ' + 'q' + vh.get_aidstrs(aid1)
        lbl2 = repr(name2)  + ' : ' +  vh.get_aidstrs(aid2)
    else:
        lbl1, lbl2 = None, None
    if vh.NO_LBL_OVERRIDE:
        title = ''
    df2.set_title(title, ax)
    # Plot annotations over images
    if in_image:
        bbox1, bbox2 = vh.get_bboxes(ibs, [aid1, aid2], [offset1, offset2])
        theta1, theta2 = ibs.get_annot_thetas([aid1, aid2])
        # HACK!
        if show_query:
            df2.draw_bbox(bbox1, bbox_color=df2.ORANGE, lbl=lbl1, theta=theta1)
        bbox_color2 = truth_color if draw_border else df2.ORANGE
        df2.draw_bbox(bbox2, bbox_color=bbox_color2, lbl=lbl2, theta=theta2)
    else:
        xy, w, h = df2.get_axis_xy_width_height(ax)
        bbox2 = (xy[0], xy[1], w, h)
        theta2 = 0
        if draw_border:
            df2.draw_border(ax, truth_color, 4, offset=offset2)
        if draw_lbl:
            # Custom user lbl for chips 1 and 2
            (x1, y1, w1, h1) = xywh1
            (x2, y2, w2, h2) = xywh2
            df2.absolute_lbl(x1 + w1, y1, lbl1)
            df2.absolute_lbl(x2 + w2, y2, lbl2)
        # No matches draw a red box
    if aid2 not in qres.aid2_fm or len(qres.aid2_fm[aid2]) == 0:
        if draw_border:
            df2.draw_boxedX(bbox2, theta=theta2)
