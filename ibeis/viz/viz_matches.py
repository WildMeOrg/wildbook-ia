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
def show_matches(ibs, qres, rid2, sel_fm=[], **kwargs):
    """ shows single annotated match result. """
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.get('draw_fmatches', True)
    rid1 = qres.qrid
    fm = qres.rid2_fm[rid2]
    fs = qres.rid2_fs[rid2]
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [rid1, rid2], **kwargs)
    if draw_fmatches:
        kpts1, kpts2 = vh.get_kpts( ibs, [rid1, rid2], **kwargs)
    else:
        kpts1, kpts2 = None, None

    # Build annotation strings / colors
    lbl1 = 'q' + vh.get_ridstrs(rid1)
    lbl2 = vh.get_ridstrs(rid2)
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
    annotate_matches(ibs, qres, rid2, xywh2=xywh2, xywh1=xywh1,
                     offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_func
def annotate_matches(ibs, qres, rid2,
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
    rid1 = qres.qrid
    truth = ibs.get_match_truth(rid1, rid2)
    truth_color = vh.get_truth_color(truth)
    # Build title
    title = vh.get_query_label(ibs, qres, rid2, truth, **kwargs)
    # Build xlabel
    ax = df2.gca()
    ph.set_plotdat(ax, 'viztype', 'matches')
    ph.set_plotdat(ax, 'qrid', rid1)
    ph.set_plotdat(ax, 'rid1', rid1)
    ph.set_plotdat(ax, 'rid2', rid2)
    if draw_lbl:
        name1, name2 = ibs.get_roi_names([rid1, rid2])
        lbl1 = repr(name1)  + ' : ' + 'q' + vh.get_ridstrs(rid1)
        lbl2 = repr(name2)  + ' : ' +  vh.get_ridstrs(rid2)
    else:
        lbl1, lbl2 = None, None
    if vh.NO_LABEL_OVERRIDE:
        title = ''
        #xlabel = ''
    df2.set_title(title, ax)
    #df2.set_xlabel(xlabel, ax)
    # Plot annotations over images
    if in_image:
        bbox1, bbox2 = vh.get_bboxes(ibs, [rid1, rid2], [offset1, offset2])
        theta1, theta2 = ibs.get_roi_thetas([rid1, rid2])
        # HACK!
        if show_query:
            df2.draw_roi(bbox1, bbox_color=df2.ORANGE, label=lbl1, theta=theta1)
        bbox_color2 = truth_color if draw_border else df2.ORANGE
        df2.draw_roi(bbox2, bbox_color=bbox_color2, label=lbl2, theta=theta2)
    else:
        xy, w, h = df2._axis_xy_width_height(ax)
        bbox2 = (xy[0], xy[1], w, h)
        theta2 = 0
        if draw_border:
            df2.draw_border(ax, truth_color, 4, offset=offset2)
        if draw_lbl:
            # Custom user label for chips 1 and 2
            (x1, y1, w1, h1) = xywh1
            (x2, y2, w2, h2) = xywh2
            df2.absolute_lbl(x1 + w1, y1, lbl1)
            df2.absolute_lbl(x2 + w2, y2, lbl2)
        # No matches draw a red box
    if rid2 not in qres.rid2_fm or len(qres.rid2_fm[rid2]) == 0:
        if draw_border:
            df2.draw_boxedX(bbox2, theta=theta2)
