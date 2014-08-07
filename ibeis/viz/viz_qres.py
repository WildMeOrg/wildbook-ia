from __future__ import absolute_import, division, print_function
from plottool import draw_func2 as df2
import numpy as np
from ibeis import ibsfuncs
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
    top_aids = qres.get_top_aids(num=N)
    aidstr = ibsfuncs.aidstr(qres.qaid)
    figtitle = kwargs.get('figtitle', '')
    if len(figtitle) > 0:
        figtitle = ' ' + figtitle
    kwargs['figtitle'] = ('q%s -- TOP %r' % (aidstr, N)) + figtitle
    return show_qres(ibs, qres, top_aids=top_aids,
                     draw_kpts=False, draw_ell=False,
                     all_kpts=False, **kwargs)


@utool.indent_func
@profile
def show_qres_analysis(ibs, qres, **kwargs):
    """
    Wrapper around show_qres.

    KWARGS:
        aid_list - show matches against aid_list (default top 5)
    """
    print('[show_qres] qres.show_analysis()')
    # Parse arguments
    N = kwargs.get('N', 3)
    show_gt  = kwargs.pop('show_gt', True)
    show_query = kwargs.pop('show_query', True)
    aid_list   = kwargs.pop('aid_list', None)
    figtitle   = kwargs.pop('figtitle', None)

    # Debug printing
    #print('[analysis] noshow_gt  = %r' % noshow_gt)
    #print('[analysis] show_query = %r' % show_query)
    #print('[analysis] aid_list    = %r' % aid_list)

    if aid_list is None:
        # Compare to aid_list instead of using top ranks
        print('[analysis] showing top aids')
        top_aids = qres.get_top_aids(num=N)
        if figtitle is None:
            if len(top_aids) == 0:
                figtitle = 'WARNING: no top scores!' + ibsfuncs.aidstr(qres.qaid)
            else:
                topscore = qres.get_aid_scores(top_aids)[0]
                figtitle = ('q%s -- topscore=%r' % (ibsfuncs.aidstr(qres.qaid), topscore))
    else:
        print('[analysis] showing a given list of aids')
        top_aids = aid_list
        if figtitle is None:
            figtitle = 'comparing to ' + ibsfuncs.aidstr(top_aids) + figtitle

    # Get any groundtruth if you are showing it
    showgt_aids = []
    if show_gt:
        showgt_aids = ibs.get_annot_groundtruth(qres.qaid)
        showgt_aids = np.setdiff1d(showgt_aids, top_aids)

    return show_qres(ibs, qres, gt_aids=showgt_aids, top_aids=top_aids,
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
    top_aids    = kwargs.get('top_aids', 6)
    gt_aids     = kwargs.get('gt_aids',   [])
    all_kpts    = kwargs.get('all_kpts', False)
    show_query  = kwargs.get('show_query', False)
    in_image    = kwargs.get('in_image', False)
    sidebyside  = kwargs.get('sidebyside', True)
    fnum = df2.kwargs_fnum(kwargs)

    fig = df2.figure(fnum=fnum, docla=True, doclf=True)

    if isinstance(top_aids, int):
        top_aids = qres.get_top_aids(num=top_aids)

    all_gts = ibs.get_annot_groundtruth(qres.qaid)
    nTop   = len(top_aids)
    nSelGt = len(gt_aids)
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
    ranked_aids = qres.get_top_aids()
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts    = nQuerySubplts + (0 if gt_aids is None else len(gt_aids))
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
    aid2_color = {aid: _color_list[ox] for ox, aid in enumerate(top_aids)}

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
        _kwshow['aid2_color'] = aid2_color
        _kwshow['draw_ell'] = annote_mode >= 1
        viz_chip.show_chip(ibs, qres.qaid, annote=False, **_kwshow)

    def _plot_matches_aids(aid_list, plotx_shift, rowcols):
        """ helper for show_qres to draw many aids """
        def _show_matches_fn(aid, orank, pnum):
            """ Helper function for drawing matches to one aid """
            aug = 'rank=%r\n' % orank
            #printDBG('[show_qres()] plotting: %r'  % (pnum,))
            _kwshow  = dict(draw_ell=annote_mode, draw_pts=False, draw_lines=annote_mode,
                            ell_alpha=.5, all_kpts=all_kpts)
            _kwshow.update(kwargs)
            _kwshow['fnum'] = fnum
            _kwshow['pnum'] = pnum
            _kwshow['title_aug'] = aug
            _kwshow['in_image'] = in_image
            # If we already are showing the query dont show it here
            if sidebyside:
                _kwshow['draw_ell'] = annote_mode == 1
                _kwshow['draw_lines'] = annote_mode >= 1
                viz_matches.show_matches(ibs, qres, aid, **_kwshow)
            else:
                _kwshow['draw_ell'] = annote_mode >= 1
                if annote_mode == 2:
                    # TODO Find a better name
                    _kwshow['color'] = aid2_color[aid]
                    _kwshow['sel_fx2'] = qres.aid2_fm[aid][:, 1]
                viz_chip.show_chip(ibs, aid, annote=False, **_kwshow)
                viz_matches.annotate_matches(ibs, qres, aid, show_query=not show_query)

        printDBG('[show_qres()] Plotting Chips %s:' % vh.get_aidstrs(aid_list))
        if aid_list is None:
            return
        # Do lazy load before show
        vh.get_chips(ibs, aid_list, **kwargs)
        vh.get_kpts(ibs, aid_list, **kwargs)
        for ox, aid in enumerate(aid_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_aids == aid)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(aid, orank, pnum)

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
        _plot_matches_aids(gt_aids, nQuerySubplts, (nRows, nGTCols))
        _plot_matches_aids(top_aids, shift_topN, (nRows, nTopNCols))
        #figtitle += ' q%s name=%s' % (ibsfuncs.aidstr(qres.qaid), ibs.aid2_name(qres.qaid))
        figtitle += aug
        df2.set_figtitle(figtitle, incanvas=not vh.NO_LBL_OVERRIDE)

    # Result Interaction
    df2.adjust_subplots_safe()
    printDBG('[show_qres()] Finished')
    return fig
