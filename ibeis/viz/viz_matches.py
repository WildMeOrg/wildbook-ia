from __future__ import absolute_import, division, print_function
import numpy as np
# UTool
import utool
# Drawtool
import plottool.draw_func2 as df2
# IBEIS
from . import viz_helpers as vh
from . import viz_chip
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_matches]', DEBUG=False)


@utool.indent_func
def show_chipres(ibs, qres, rid2, sel_fm=[], **kwargs):
    """ shows single annotated match result. """
    in_image = kwargs.get('in_image', False)
    qrid = qres.qrid
    fm = qres.rid2_fm[rid2]
    fs = qres.rid2_fs[rid2]
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [qrid, rid2], **kwargs)
    kpts1,  kpts2  = vh.get_kpts( ibs, [qrid, rid2], **kwargs)

    # Build annotation strings / colors
    lbl1 = 'q' + vh.get_ridstrs(qrid)
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
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_chipres(ibs, qres, rid2, xywh2=xywh2,
                     offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_func
def annotate_chipres(ibs, qres, rid2,
                     offset1=(0, 0),
                     offset2=(0, 0), **kwargs):
    # TODO Use this function when you clean show_chipres
    in_image = kwargs.get('in_image', False)
    show_query = kwargs.get('show_query', True)
    printDBG('[viz] annotate_chipres()')
    qrid = qres.qrid
    truth = vh.get_match_truth(ibs, qrid, rid2)
    truth_color = vh.get_truth_color(ibs, truth)
    # Build title
    title = vh.get_query_label(ibs, qres, rid2, truth, **kwargs)
    # Build xlabel
    xlabel = vh.get_chip_labels(ibs, rid2, **kwargs)
    ax = df2.gca()
    vh.set_ibsdat(ax, 'viztype', 'chipres')
    vh.set_ibsdat(ax, 'qrid', qrid)
    vh.set_ibsdat(ax, 'rid2', rid2)
    if vh.NO_LABEL_OVERRIDE:
        title = ''
        xlabel = ''
    df2.set_title(title, ax)
    df2.set_xlabel(xlabel, ax)
    # Plot annotations over images
    if in_image:
        bbox1, bbox2 = vh.get_bboxes(ibs, [qrid, rid2], [offset1, offset2])
        theta1, theta2 = vh.get_thetas(ibs, [qrid, rid2])
        # HACK!
        lbl1 = 'q' + vh.get_ridstrs(qrid)
        lbl2 = vh.get_ridstrs(rid2)
        if show_query:
            df2.draw_roi(bbox1, bbox_color=df2.ORANGE, label=lbl1, theta=theta1)
        df2.draw_roi(bbox2, bbox_color=truth_color, label=lbl2, theta=theta2)
    else:
        xy, w, h = df2._axis_xy_width_height(ax)
        bbox2 = (xy[0], xy[1], w, h)
        theta2 = 0
        df2.draw_border(ax, truth_color, 4, offset=offset2)
        # No matches draw a red box
    if not rid2 in qres.rid2_fm or len(qres.rid2_fm[rid2]) == 0:
        df2.draw_boxedX(bbox2, theta=theta2)


@utool.indent_func
@profile
def show_top(ibs, qres, **kwargs):
    top_rids = qres.get_top_rids(ibs)
    N = len(top_rids)
    ridstr = ibs.ridstr(qres.qrid)
    figtitle = kwargs.pop('figtitle', 'q%s -- TOP %r' % (ridstr, N))
    return show_qres(ibs, qres, top_rids=top_rids, figtitle=figtitle,
                     draw_kpts=False, draw_ell=False,
                     all_kpts=False, **kwargs)


@utool.indent_func
@profile
def show_qres_analysis(ibs, qres, **kwargs):
        print('[viz] qres.show_analysis()')
        # Parse arguments
        noshow_gt  = kwargs.pop('noshow_gt', True)
        show_query = kwargs.pop('show_query', False)
        rid_list   = kwargs.pop('rid_list', None)
        figtitle   = kwargs.pop('figtitle', None)

        # Debug printing
        #print('[viz.analysis] noshow_gt  = %r' % noshow_gt)
        #print('[viz.analysis] show_query = %r' % show_query)
        #print('[viz.analysis] rid_list    = %r' % rid_list)

        # Compare to rid_list instead of using top ranks
        if rid_list is None:
            print('[viz.analysis] showing top rids')
            top_rids = qres.get_top_rids(ibs)
            if figtitle is None:
                if len(top_rids) == 0:
                    figtitle = 'WARNING: no top scores!' + ibs.ridstr(qres.qrid)
                else:
                    topscore = qres.get_rid2_score()[top_rids][0]
                    figtitle = ('q%s -- topscore=%r' % (ibs.ridstr(qres.qrid), topscore))
        else:
            print('[viz.analysis] showing a given list of rids')
            top_rids = rid_list
            if figtitle is None:
                figtitle = 'comparing to ' + ibs.ridstr(top_rids) + figtitle

        # Do we show the ground truth?
        def missed_rids():
            showgt_rids = vh.get_groundtruth(ibs, qres.qrid)
            return np.setdiff1d(showgt_rids, top_rids)
        showgt_rids = [] if noshow_gt else missed_rids()

        return show_qres(ibs, qres, gt_rids=showgt_rids, top_rids=top_rids,
                         figtitle=figtitle, show_query=show_query, **kwargs)


@utool.indent_func
def show_qres(ibs, qres, **kwargs):
    """ Displays query chip, groundtruth matches, and top 5 matches """
    annote     = kwargs.get('annote', 1) % 3  # this is toggled
    fnum       = kwargs.get('fnum', 3)
    figtitle   = kwargs.get('figtitle', '')
    aug        = kwargs.get('aug', '')
    top_rids   = kwargs.get('top_rids', 6)
    gt_rids    = kwargs.get('gt_rids',   [])
    all_kpts   = kwargs.get('all_kpts', False)
    show_query = kwargs.get('show_query', False)
    dosquare   = kwargs.get('dosquare', False)
    in_image   = kwargs.get('in_image', False)

    if isinstance(top_rids, int):
        top_rids = qres.get_top_rids(num=top_rids)

    all_gts = vh.get_groundtruth(ibs, qres.qrid)
    nTop   = len(top_rids)
    nSelGt = len(gt_rids)
    nAllGt = len(all_gts)

    max_nCols = 5
    if nTop in [6, 7]:
        max_nCols = 3
    if nTop in [8]:
        max_nCols = 4

    printDBG('[viz]========================')
    printDBG('[viz.show_qres()]----------------')
    printDBG('[viz.show_qres()] #nTop=%r #missed_gts=%r/%r' % (nTop, nSelGt,
                                                               nAllGt))
    printDBG('[viz.show_qres()] * fnum=%r' % (fnum,))
    printDBG('[viz.show_qres()] * figtitle=%r' % (figtitle,))
    printDBG('[viz.show_qres()] * max_nCols=%r' % (max_nCols,))
    printDBG('[viz.show_qres()] * show_query=%r' % (show_query,))
    printDBG(qres.get_inspect_str())
    ranked_rids = qres.get_top_rids()
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts    = nQuerySubplts + (0 if gt_rids is None else len(gt_rids))
    nTopNSubplts  = nTop
    nTopNCols     = min(max_nCols, nTopNSubplts)
    nGTCols       = min(max_nCols, nGtSubplts)
    nGTCols       = max(nGTCols, nTopNCols)
    nTopNCols     = nGTCols
    nGtRows       = 0 if nGTCols   == 0 else int(np.ceil(nGtSubplts   / nGTCols))
    nTopNRows     = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells      = nGtRows * nGTCols
    nRows         = nTopNRows + nGtRows

    # HACK:
    _color_list = df2.distinct_colors(nTop)
    rid2_color = {rid: _color_list[ox] for ox, rid in enumerate(top_rids)}

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        """ helper for viz.show_qres """
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annote)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['pnum'] = pnum
        _kwshow['rid2_color'] = rid2_color
        _kwshow['draw_ell'] = annote >= 1
        viz_chip.show_chip(ibs, qres.qrid, **_kwshow)

    def _plot_matches_rids(rid_list, plotx_shift, rowcols):
        """ helper for viz.show_qres to draw many rids """
        def _show_matches_fn(rid, orank, pnum):
            """ Helper function for drawing matches to one rid """
            aug = 'rank=%r\n' % orank
            #printDBG('[viz.show_qres()] plotting: %r'  % (pnum,))
            _kwshow  = dict(draw_ell=annote, draw_pts=False, draw_lines=annote,
                            ell_alpha=.5, all_kpts=all_kpts)
            _kwshow.update(kwargs)
            _kwshow['fnum'] = fnum
            _kwshow['pnum'] = pnum
            _kwshow['title_aug'] = aug
            # If we already are showing the query dont show it here
            if not show_query:
                _kwshow['draw_ell'] = annote == 1
                _kwshow['draw_lines'] = annote >= 1
                show_chipres(ibs, qres, rid, in_image=in_image, **_kwshow)
            else:
                _kwshow['draw_ell'] = annote >= 1
                if annote == 2:
                    # TODO Find a better name
                    _kwshow['color'] = rid2_color[rid]
                    _kwshow['sel_fx2'] = qres.rid2_fm[rid][:, 1]
                viz_chip.show_chip(ibs, rid, in_image=in_image, **_kwshow)
                annotate_chipres(ibs, qres, rid, in_image=in_image, show_query=not show_query)

        printDBG('[viz.show_qres()] Plotting Chips %s:' % vh.get_ridstrs(rid_list))
        if rid_list is None:
            return
        # Do lazy load before show
        vh.get_chips(ibs, rid_list, **kwargs)
        vh.get_kpts(ibs, rid_list, **kwargs)
        for ox, rid in enumerate(rid_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_rids == rid)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(rid, orank, pnum)

    if dosquare:
        # HACK
        nSubplots = nGtSubplts + nTopNSubplts
        nRows, nCols = vh.get_square_row_cols(nSubplots, 3)
        nTopNCols = nGTCols = nCols
        shift_topN = 1
        printDBG('nRows, nCols = (%r, %r)' % (nRows, nCols))
    else:
        shift_topN = nGtCells

    if nGtSubplts == 1:
        nGTCols = 1

    fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), docla=True, doclf=True)
    #df2.disconnect_callback(fig, 'button_press_event')
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query:
        _show_query_fn(0, (nRows, nGTCols))
    # Plot Ground Truth
    _plot_matches_rids(gt_rids, nQuerySubplts, (nRows, nGTCols))
    _plot_matches_rids(top_rids, shift_topN, (nRows, nTopNCols))
    #figtitle += ' q%s name=%s' % (ibs.ridstr(qres.qrid), ibs.rid2_name(qres.qrid))
    figtitle += aug
    df2.set_figtitle(figtitle, incanvas=not vh.NO_LABEL_OVERRIDE)

    # Result Interaction
    df2.adjust_subplots_safe()
    printDBG('[viz.show_qres()] Finished')
    return fig
