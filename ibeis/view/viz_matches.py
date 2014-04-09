from __future__ import division, print_function
import viz_helpers
# Scientific
import numpy as np
# UTool
import utool
# Drawtool
import drawtool.draw_func2 as df2
# IBEIS
from ibeis.model.jon_recognition import QueryResult
#
from ibeis.view import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz-matches]', DEBUG=False)


@utool.indent_decor('[show_chipres]')
def show_chipres(ibs, qres, cid2, fnum=None, pnum=None, sel_fm=[], in_image=False, **kwargs):
    'shows single annotated match result.'
    qcid = qres.qcid
    fm = qres.cid2_fm[cid2]
    fs = qres.cid2_fs[cid2]
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [qcid, cid2], in_image=in_image)
    kpts1,  kpts2  = vh.get_kpts( ibs, [qcid, cid2], in_image=in_image)

    # Build annotation strings / colors
    lbl1 = 'q' + vh.get_cidstrs(qcid)
    lbl2 = vh.get_cidstrs(cid2)
    if in_image:  # HACK!
        lbl1 = None
        lbl2 = None
    # Draws the chips and keypoint matches
    kwargs_ = dict(fs=fs, lbl1=lbl1, lbl2=lbl2, fnum=fnum, pnum=pnum)
    kwargs_.update(kwargs)
    try:
        ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm, **kwargs_)
    except Exception as ex:
        print('<!!!>')
        utool.print_exception(ex, '[viz_matches]')
        QueryResult.dbg_check_query_result(ibs, qres)
        print('consider qr.remove_corrupted_queries(ibs, qres, dryrun=False)')
        print('</!!!>')
        raise
    (x1, y1, w1, h1) = xywh1
    (x2, y2, w2, h2) = xywh2
    if len(sel_fm) > 0:
        # Draw any selected matches
        sm_kw = dict(rect=True, colors=df2.BLUE)
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_chipres(ibs, qres, cid2, xywh2=xywh2, in_image=in_image, offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_decor('[annote_chipres]')
def annotate_chipres(ibs, qres, cid2, showTF=True, showScore=True,
                     showRank=True, title_pref='', title_suff='',
                     time_appart=False, in_image=False, offset1=(0, 0),
                     offset2=(0, 0), show_query=True, xywh2=None, **kwargs):
    printDBG('[viz] annotate_chipres()')
    #print('Did not expect args: %r' % (kwargs.keys(),))
    qcid = qres.qcid
    score = qres.cid2_score[cid2]
    # TODO Use this function when you clean show_chipres
    (truestr, falsestr, nonamestr) = ('TRUE', 'FALSE', '???')
    is_true, is_unknown = vh.is_true_match(ibs, qcid, cid2)
    isgt_str = nonamestr if is_unknown else (truestr if is_true else falsestr)
    match_color = {nonamestr: df2.UNKNOWN_PURP,
                   truestr:   df2.TRUE_GREEN,
                   falsestr:  df2.FALSE_RED}[isgt_str]
    # Build title
    title = '*%s*' % isgt_str if showTF else ''
    if showRank:
        rank_str = ' rank=' + str(qres.get_cid_ranks([cid2])[0] + 1)
        title += rank_str
    if showScore:
        score_str = (' score=' + utool.num_fmt(score)) % (score)
        title += score_str

    title = title_pref + str(title) + title_suff
    # Build xlabel
    xlabel = vh.get_chip_labels(ibs, cid2, **kwargs)
    if time_appart:
        xlabel += ('\n' + ibs.get_timedelta_str(qcid, cid2))
    ax = df2.gca()
    ax._hs_viewtype = 'chipres'
    ax._hs_qcid = qcid
    ax._hs_cid = cid2
    if viz_helpers.NO_LABEL_OVERRIDE:
        title = ''
        xlabel = ''
    df2.set_title(title, ax)
    df2.set_xlabel(xlabel, ax)
    if in_image:
        roi1 = ibs.cid2_roi(qcid) + np.array(list(offset1) + [0, 0])
        roi2 = ibs.cid2_roi(cid2) + np.array(list(offset2) + [0, 0])
        theta1 = ibs.cid2_theta(qcid)
        theta2 = ibs.cid2_theta(cid2)
        # HACK!
        lbl1 = 'q' + ibs.cidstr(qcid)
        lbl2 = ibs.cidstr(cid2)
        if show_query:
            df2.draw_roi(roi1, bbox_color=df2.ORANGE, label=lbl1, theta=theta1)
        df2.draw_roi(roi2, bbox_color=match_color, label=lbl2, theta=theta2)
        # No matches draw a red box
        if len(qres.cid2_fm[cid2]) == 0:
            df2.draw_boxedX(roi2, theta=theta2)
    else:
        if xywh2 is None:
            xy, w, h = df2._axis_xy_width_height(ax)
            xywh2 = (xy[0], xy[1], w, h)
        df2.draw_border(ax, match_color, 4, offset=offset2)
        # No matches draw a red box
        if len(qres.cid2_fm[cid2]) == 0:
            df2.draw_boxedX(xywh2)
