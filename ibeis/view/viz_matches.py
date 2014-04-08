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
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz-matches]', DEBUG=False)


@utool.indent_decor('[show_chipres]')
def show_chipres(ibs, qres, cid, fnum=None, pnum=None, sel_fm=[], in_image=False, **kwargs):
    'shows single annotated match result.'
    qcid = qres.qcid
    #cid2_score = qres.get_cid2_score()
    cid2_fm    = qres.get_cid2_fm()
    cid2_fs    = qres.get_cid2_fs()
    #cid2_fk   = qres.get_cid2_fk()
    #printDBG('[viz.show_chipres()] Showing matches from %s' % (vs_str))
    #printDBG('[viz.show_chipres()] fnum=%r, pnum=%r' % (fnum, pnum))
    # Test valid cid
    printDBG('[viz] show_chipres()')
    if np.isnan(cid):
        nan_img = np.zeros((32, 32), dtype=np.uint8)
        title = '(q%s v %r)' % (ibs.cidstr(qcid), cid)
        df2.imshow(nan_img, fnum=fnum, pnum=pnum, title=title)
        return
    fm = cid2_fm[cid]
    fs = cid2_fs[cid]
    #fk = cid2_fk[cid]
    #vs_str = ibs.vs_str(qcid, cid)
    # Read query and result info (chips, names, ...)
    if in_image:
        # TODO: rectify build_transform2 with cc2
        # clean up so its not abysmal
        rchip1, rchip2 = [ibs.cid2_image(_) for _ in [qcid, cid]]
        kpts1, kpts2   = viz_helpers.get_imgspace_chip_kpts(ibs, [qcid, cid])
    else:
        rchip1, rchip2 = ibs.get_chips([qcid, cid])
        kpts1, kpts2   = ibs.get_kpts([qcid, cid])

    # Build annotation strings / colors
    lbl1 = 'q' + ibs.cidstr(qcid)
    lbl2 = ibs.cidstr(cid)
    if in_image:
        # HACK!
        lbl1 = None
        lbl2 = None
    # Draws the chips and keypoint matches
    kwargs_ = dict(fs=fs, lbl1=lbl1, lbl2=lbl2, fnum=fnum,
                   pnum=pnum, vert=ibs.prefs.display_cfg.vert)
    kwargs_.update(kwargs)
    try:
        ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm, **kwargs_)
    except Exception as ex:
        print('!!!!!!!!!!!!!!!')
        print('[viz] %s: %s' % (type(ex), ex))
        print('[viz] vsstr = %s' % ibs.vs_str(qcid, cid))
        QueryResult.dbg_check_query_result(ibs, qres)
        print('consider qr.remove_corrupted_queries(ibs, qres, dryrun=False)')
        utool.qflag()
        raise
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    if len(sel_fm) > 0:
        # Draw any selected matches
        _smargs = dict(rect=True, colors=df2.BLUE)
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **_smargs)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_chipres(ibs, qres, cid, xywh2=xywh2, in_image=in_image, offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_decor('[annote_chipres]')
def annotate_chipres(ibs, qres, cid, showTF=True, showScore=True, showRank=True, title_pref='',
                     title_suff='', show_gname=False, show_name=True,
                     time_appart=True, in_image=False, offset1=(0, 0),
                     offset2=(0, 0), show_query=True, xywh2=None, **kwargs):
    printDBG('[viz] annotate_chipres()')
    #print('Did not expect args: %r' % (kwargs.keys(),))
    qcid = qres.qcid
    score = qres.cid2_score[cid]
    # TODO Use this function when you clean show_chipres
    (truestr, falsestr, nonamestr) = ('TRUE', 'FALSE', '???')
    is_true, is_unknown = ibs.is_true_match(qcid, cid)
    isgt_str = nonamestr if is_unknown else (truestr if is_true else falsestr)
    match_color = {nonamestr: df2.UNKNOWN_PURP,
                   truestr:   df2.TRUE_GREEN,
                   falsestr:  df2.FALSE_RED}[isgt_str]
    # Build title
    title = '*%s*' % isgt_str if showTF else ''
    if showRank:
        rank_str = ' rank=' + str(qres.get_cid_ranks([cid])[0] + 1)
        title += rank_str
    if showScore:
        score_str = (' score=' + utool.num_fmt(score)) % (score)
        title += score_str

    title = title_pref + str(title) + title_suff
    # Build xlabel
    xlabel_ = []
    if show_gname:
        xlabel_.append('gname=%r' % ibs.cid2_gname(cid))
    if show_name:
        xlabel_.append('name=%r' % ibs.cid2_name(cid))
    if time_appart:
        xlabel_.append('\n' + ibs.get_timedelta_str(qcid, cid))
    xlabel = ', '.join(xlabel_)
    ax = df2.gca()
    ax._hs_viewtype = 'chipres'
    ax._hs_qcid = qcid
    ax._hs_cid = cid
    if viz_helpers.NO_LABEL_OVERRIDE:
        title = ''
        xlabel = ''
    df2.set_title(title, ax)
    df2.set_xlabel(xlabel, ax)
    if in_image:
        roi1 = ibs.cid2_roi(qcid) + np.array(list(offset1) + [0, 0])
        roi2 = ibs.cid2_roi(cid) + np.array(list(offset2) + [0, 0])
        theta1 = ibs.cid2_theta(qcid)
        theta2 = ibs.cid2_theta(cid)
        # HACK!
        lbl1 = 'q' + ibs.cidstr(qcid)
        lbl2 = ibs.cidstr(cid)
        if show_query:
            df2.draw_roi(roi1, bbox_color=df2.ORANGE, label=lbl1, theta=theta1)
        df2.draw_roi(roi2, bbox_color=match_color, label=lbl2, theta=theta2)
        # No matches draw a red box
        if len(qres.cid2_fm[cid]) == 0:
            df2.draw_boxedX(roi2, theta=theta2)
    else:
        if xywh2 is None:
            xy, w, h = df2._axis_xy_width_height(ax)
            xywh2 = (xy[0], xy[1], w, h)
        df2.draw_border(ax, match_color, 4, offset=offset2)
        # No matches draw a red box
        if len(qres.cid2_fm[cid]) == 0:
            df2.draw_boxedX(xywh2)
