from __future__ import absolute_import, division, print_function
from plottool import draw_func2 as df2
import numpy as np
from ibeis.dev import ibsfuncs
from . import viz_helpers as vh
from . import viz_chip
from . import viz_matches
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_qres]', DEBUG=False)


@utool.indent_func
@profile
def show_qres_top(ibs, qres, **kwargs):
    """
    Wrapper around show_qres.
    """
    N = kwargs.get('N', 6)
    top_rids = qres.get_top_rids(num=N)
    ridstr = ibsfuncs.ridstr(qres.qrid)
    figtitle = kwargs.get('figtitle', '')
    if len(figtitle) > 0:
        figtitle = ' ' + figtitle
    kwargs['figtitle'] = ('q%s -- TOP %r' % (ridstr, N)) + figtitle
    return show_qres(ibs, qres, top_rids=top_rids,
                     draw_kpts=False, draw_ell=False,
                     all_kpts=False, **kwargs)


@utool.indent_func
@profile
def show_qres_analysis(ibs, qres, **kwargs):
    """
    Wrapper around show_qres.

    KWARGS:
        rid_list - show matches against rid_list (default top 5)
    """
    print('[show_qres] qres.show_analysis()')
    # Parse arguments
    N = kwargs.get('N', 3)
    show_gt  = kwargs.pop('show_gt', True)
    show_query = kwargs.pop('show_query', False)
    rid_list   = kwargs.pop('rid_list', None)
    figtitle   = kwargs.pop('figtitle', None)

    # Debug printing
    #print('[analysis] noshow_gt  = %r' % noshow_gt)
    #print('[analysis] show_query = %r' % show_query)
    #print('[analysis] rid_list    = %r' % rid_list)

    if rid_list is None:
        # Compare to rid_list instead of using top ranks
        print('[analysis] showing top rids')
        top_rids = qres.get_top_rids(num=N)
        if figtitle is None:
            if len(top_rids) == 0:
                figtitle = 'WARNING: no top scores!' + ibsfuncs.ridstr(qres.qrid)
            else:
                topscore = qres.get_rid_scores(top_rids)[0]
                figtitle = ('q%s -- topscore=%r' % (ibsfuncs.ridstr(qres.qrid), topscore))
    else:
        print('[analysis] showing a given list of rids')
        top_rids = rid_list
        if figtitle is None:
            figtitle = 'comparing to ' + ibsfuncs.ridstr(top_rids) + figtitle

    # Get any groundtruth if you are showing it
    showgt_rids = []
    if show_gt:
        showgt_rids = ibs.get_roi_groundtruth(qres.qrid)
        showgt_rids = np.setdiff1d(showgt_rids, top_rids)

    return show_qres(ibs, qres, gt_rids=showgt_rids, top_rids=top_rids,
                        figtitle=figtitle, show_query=show_query, **kwargs)


@utool.indent_func
def show_qres(ibs, qres, **kwargs):
    """
    Display Query Result Logic
    Defaults to: query chip, groundtruth matches, and top 5 matches
    """
    annote_mode = kwargs.get('annote_mode', 1) % 3  # this is toggled
    figtitle    = kwargs.get('figtitle', '')
    aug         = kwargs.get('aug', '')
    top_rids    = kwargs.get('top_rids', 6)
    gt_rids     = kwargs.get('gt_rids',   [])
    all_kpts    = kwargs.get('all_kpts', False)
    show_query  = kwargs.get('show_query', False)
    in_image    = kwargs.get('in_image', False)
    fnum = df2.kwargs_fnum(kwargs)

    fig = df2.figure(fnum=fnum, docla=True, doclf=True)

    if isinstance(top_rids, int):
        top_rids = qres.get_top_rids(num=top_rids)

    all_gts = ibs.get_roi_groundtruth(qres.qrid)
    nTop   = len(top_rids)
    nSelGt = len(gt_rids)
    nAllGt = len(all_gts)

    max_nCols = 5
    if nTop in [6, 7]:
        max_nCols = 3
    if nTop in [8]:
        max_nCols = 4

    printDBG('[show_qres]========================')
    printDBG('[show_qres]----------------')
    printDBG('[show_qres] #nTop=%r #missed_gts=%r/%r' % (nTop, nSelGt, nAllGt))
    printDBG('[show_qres] * fnum=%r' % (fnum,))
    printDBG('[show_qres] * figtitle=%r' % (figtitle,))
    printDBG('[show_qres] * max_nCols=%r' % (max_nCols,))
    printDBG('[show_qres] * show_query=%r' % (show_query,))
    printDBG('[show_qres] * kwargs=%s' % (utool.dict_str(kwargs),))
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
        """ helper for show_qres """
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annote_mode)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['pnum'] = pnum
        _kwshow['rid2_color'] = rid2_color
        _kwshow['draw_ell'] = annote_mode >= 1
        viz_chip.show_chip(ibs, qres.qrid, **_kwshow)

    def _plot_matches_rids(rid_list, plotx_shift, rowcols):
        """ helper for show_qres to draw many rids """
        def _show_matches_fn(rid, orank, pnum):
            """ Helper function for drawing matches to one rid """
            aug = 'rank=%r\n' % orank
            #printDBG('[show_qres()] plotting: %r'  % (pnum,))
            _kwshow  = dict(draw_ell=annote_mode, draw_pts=False, draw_lines=annote_mode,
                            ell_alpha=.5, all_kpts=all_kpts)
            _kwshow.update(kwargs)
            _kwshow['fnum'] = fnum
            _kwshow['pnum'] = pnum
            _kwshow['title_aug'] = aug
            # If we already are showing the query dont show it here
            if not show_query:
                _kwshow['draw_ell'] = annote_mode == 1
                _kwshow['draw_lines'] = annote_mode >= 1
                viz_matches.show_matches(ibs, qres, rid, in_image=in_image, **_kwshow)
            else:
                _kwshow['draw_ell'] = annote_mode >= 1
                if annote_mode == 2:
                    # TODO Find a better name
                    _kwshow['color'] = rid2_color[rid]
                    _kwshow['sel_fx2'] = qres.rid2_fm[rid][:, 1]
                viz_chip.show_chip(ibs, rid, in_image=in_image, **_kwshow)
                viz_matches.annotate_matches(ibs, qres, rid, in_image=in_image, show_query=not show_query)

        printDBG('[show_qres()] Plotting Chips %s:' % vh.get_ridstrs(rid_list))
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

    shift_topN = nGtCells

    if nGtSubplts == 1:
        nGTCols = 1

    if nRows == 0:
        df2.imshow_null(fnum=fnum)
    else:
        fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), docla=True, doclf=True)
        #df2.disconnect_callback(fig, 'button_press_event')
        df2.plt.subplot(nRows, nGTCols, 1)
        # Plot Query
        if show_query:
            _show_query_fn(0, (nRows, nGTCols))
        # Plot Ground Truth
        _plot_matches_rids(gt_rids, nQuerySubplts, (nRows, nGTCols))
        _plot_matches_rids(top_rids, shift_topN, (nRows, nTopNCols))
        #figtitle += ' q%s name=%s' % (ibsfuncs.ridstr(qres.qrid), ibs.rid2_name(qres.qrid))
        figtitle += aug
        df2.set_figtitle(figtitle, incanvas=not vh.NO_LABEL_OVERRIDE)

    # Result Interaction
    df2.adjust_subplots_safe()
    printDBG('[show_qres()] Finished')
    return fig
