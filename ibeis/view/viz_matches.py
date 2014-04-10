from __future__ import division, print_function
import numpy as np
# UTool
import utool
# Drawtool
import drawtool.draw_func2 as df2
# IBEIS
from ibeis.view import viz_helpers as vh
from ibeis.view import viz_chip
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz-matches]', DEBUG=False)


@utool.indent_func
def show_chipres(ibs, qres, cid2, sel_fm=[], **kwargs):
    'shows single annotated match result.'
    in_image = kwargs.get('in_image', False)
    qcid = qres.qcid
    fm = qres.cid2_fm[cid2]
    fs = qres.cid2_fs[cid2]
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [qcid, cid2], **kwargs)
    kpts1,  kpts2  = vh.get_kpts( ibs, [qcid, cid2], **kwargs)

    # Build annotation strings / colors
    lbl1 = 'q' + vh.get_cidstrs(qcid)
    lbl2 = vh.get_cidstrs(cid2)
    if in_image:  # HACK!
        lbl1 = None
        lbl2 = None
    # Draws the chips and keypoint matches
    try:
        ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2,
                                               fm, fs=fs, lbl1=lbl1, lbl2=lbl2,
                                               **kwargs)
    except Exception as ex:
        utool.print_exception(ex, 'consider qr.remove_corrupted_queries',
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
    annotate_chipres(ibs, qres, cid2, xywh2=xywh2,
                     offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_func
def annotate_chipres(ibs, qres, cid2,
                     offset1=(0, 0),
                     offset2=(0, 0), **kwargs):
    # TODO Use this function when you clean show_chipres
    in_image = kwargs.get('in_image', False)
    show_query = kwargs.get('show_query', True)
    printDBG('[viz] annotate_chipres()')
    qcid = qres.qcid
    truth = vh.get_match_truth(ibs, qcid, cid2)
    truth_color = vh.get_truth_color(ibs, truth)
    # Build title
    title = vh.get_query_label(ibs, qres, cid2, truth, **kwargs)
    # Build xlabel
    xlabel = vh.get_chip_labels(ibs, cid2, **kwargs)
    ax = df2.gca()
    vh.set_ibsdat(ax, 'viewtype', 'chipres')
    vh.set_ibsdat(ax, 'qcid', qcid)
    vh.set_ibsdat(ax, 'cid2', cid2)
    if vh.NO_LABEL_OVERRIDE:
        title = ''
        xlabel = ''
    df2.set_title(title, ax)
    df2.set_xlabel(xlabel, ax)
    # Plot annotations over images
    if in_image:
        bbox1, bbox2 = vh.get_bboxes(ibs, [qcid, cid2], [offset1, offset2])
        theta1, theta2 = vh.get_thetas(ibs, [qcid, cid2])
        # HACK!
        lbl1 = 'q' + vh.get_cidstrs(qcid)
        lbl2 = vh.get_cidstrs(cid2)
        if show_query:
            df2.draw_roi(bbox1, bbox_color=df2.ORANGE, label=lbl1, theta=theta1)
        df2.draw_roi(bbox2, bbox_color=truth_color, label=lbl2, theta=theta2)
    else:
        xy, w, h = df2._axis_xy_width_height(ax)
        bbox2 = (xy[0], xy[1], w, h)
        theta2 = 0
        df2.draw_border(ax, truth_color, 4, offset=offset2)
        # No matches draw a red box
    if not cid2 in qres.cid2_fm or len(qres.cid2_fm[cid2]) == 0:
        df2.draw_boxedX(bbox2, theta=theta2)


@utool.indent_func
def show_qres(ibs, qres, **kwargs):
    """ Displays query chip, groundtruth matches, and top 5 matches """
    annote     = kwargs.pop('annote', 2)  # this is toggled
    fnum       = kwargs.get('fnum', 3)
    figtitle   = kwargs.get('figtitle', '')
    aug        = kwargs.get('aug', '')
    top_cids   = kwargs.get('top_cids', 6)
    gt_cids    = kwargs.get('gt_cids',   [])
    all_kpts   = kwargs.get('all_kpts', False)
    show_query = kwargs.get('show_query', False)
    dosquare   = kwargs.get('dosquare', False)
    in_image   = kwargs.get('in_image', False)

    if isinstance(top_cids, int):
        top_cids = qres.get_top_cids(num=top_cids)

    max_nCols = 5
    if len(top_cids) in [6, 7]:
        max_nCols = 3
    if len(top_cids) in [8]:
        max_nCols = 4

    printDBG('[viz]========================')
    printDBG('[viz._show_res()]----------------')
    all_gts = vh.get_groundtruth(ibs, qres.qcid)
    _tup = tuple(map(len, (top_cids, gt_cids, all_gts)))
    printDBG('[viz._show_res()] #topN=%r #missed_gts=%r/%r' % _tup)
    printDBG('[viz._show_res()] * fnum=%r' % (fnum,))
    printDBG('[viz._show_res()] * figtitle=%r' % (figtitle,))
    printDBG('[viz._show_res()] * max_nCols=%r' % (max_nCols,))
    printDBG('[viz._show_res()] * show_query=%r' % (show_query,))
    printDBG(qres.get_inspect_str())
    ranked_cids = qres.get_top_cids()
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts    = nQuerySubplts + (0 if gt_cids is None else len(gt_cids))
    nTopNSubplts  = 0 if top_cids is None else len(top_cids)
    nTopNCols     = min(max_nCols, nTopNSubplts)
    nGTCols       = min(max_nCols, nGtSubplts)
    nGTCols       = max(nGTCols, nTopNCols)
    nTopNCols     = nGTCols
    nGtRows       = 0 if nGTCols == 0 else int(np.ceil(nGtSubplts / nGTCols))
    nTopNRows     = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells      = nGtRows * nGTCols
    nRows         = nTopNRows + nGtRows

    # HACK:
    _color_list = df2.distinct_colors(len(top_cids))
    cid2_color = {cid: _color_list[ox] for ox, cid in enumerate(top_cids)}

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        """ helper for viz._show_res """
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annote)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['pnum'] = pnum
        _kwshow['cid2_color'] = cid2_color
        _kwshow['draw_ell'] = annote >= 1
        viz_chip.show_chip(ibs, qres.qcid, **_kwshow)

    def _plot_matches_cids(cid_list, plotx_shift, rowcols):
        """ helper for viz._show_res to draw many cids """
        def _show_matches_fn(cid, orank, pnum):
            """ Helper function for drawing matches to one cid """
            aug = 'rank=%r\n' % orank
            #printDBG('[viz._show_res()] plotting: %r'  % (pnum,))
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
                show_chipres(ibs, qres, cid, in_image=in_image, **_kwshow)
            else:
                _kwshow['draw_ell'] = annote >= 1
                if annote == 2:
                    # TODO Find a better name
                    _kwshow['color'] = cid2_color[cid]
                    _kwshow['sel_fx2'] = qres.cid2_fm[cid][:, 1]
                viz_chip.show_chip(ibs, cid, in_image=in_image, **_kwshow)
                annotate_chipres(ibs, qres, cid, in_image=in_image, show_query=not show_query)

        printDBG('[viz._show_res()] Plotting Chips %s:' % vh.get_cidstrs(cid_list))
        if cid_list is None:
            return
        # Do lazy load before show
        vh.get_chips(ibs, cid_list, **kwargs)
        vh.get_kpts(ibs, cid_list, **kwargs)
        for ox, cid in enumerate(cid_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_cids == cid)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(cid, orank, pnum)

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
    _plot_matches_cids(gt_cids, nQuerySubplts, (nRows, nGTCols))
    _plot_matches_cids(top_cids, shift_topN, (nRows, nTopNCols))
    #figtitle += ' q%s name=%s' % (ibs.cidstr(qres.qcid), ibs.cid2_name(qres.qcid))
    figtitle += aug
    df2.set_figtitle(figtitle, incanvas=not vh.NO_LABEL_OVERRIDE)

    # Result Interaction
    df2.adjust_subplots_safe()
    printDBG('[viz._show_res()] Finished')
    return fig
