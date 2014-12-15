from __future__ import absolute_import, division, print_function
from plottool import draw_func2 as df2
import utool as ut  # NOQA
import numpy as np
from ibeis import ibsfuncs
from . import viz_helpers as vh
from . import viz_chip
from . import viz_matches
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_qres]')


DEFAULT_NTOP = 3


@utool.indent_func
@profile
def show_qres_top(ibs, qres, **kwargs):
    """
    Wrapper around show_qres.
    """
    N = kwargs.get('N', DEFAULT_NTOP)
    top_aids = qres.get_top_aids(num=N)
    aidstr = ibsfuncs.aidstr(qres.qaid)
    figtitle = kwargs.get('figtitle', '')
    if len(figtitle) > 0:
        figtitle = ' ' + figtitle
    kwargs['figtitle'] = ('q%s -- TOP %r' % (aidstr, N)) + figtitle
    return show_qres(ibs, qres, top_aids=top_aids,
                     # dont use these. use annot mode instead
                     #draw_kpts=False,
                     #draw_ell=False,
                     #all_kpts=False,
                     **kwargs)


@utool.indent_func
@profile
def show_qres_analysis(ibs, qres, **kwargs):
    """
    Wrapper around show_qres.

    KWARGS:
        aid_list - show matches against aid_list (default top 3)
    """
    print('[show_qres] qres.show_analysis()')
    # Parse arguments
    N = kwargs.get('N', DEFAULT_NTOP)
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
        #ut.embed()
        # Get the missed groundtruth annotations
        # qres.daids comes from qreq_.get_external_daids()
        matchable_aids = qres.daids
        #matchable_aids = ibs.get_recognition_database_aids()
        #matchable_aids = list(qres.aid2_fm.keys())
        _gtaids = ibs.get_annot_groundtruth(qres.qaid, daid_list=matchable_aids)
        # No need to display highly ranked groundtruth. It will already show up
        _gtaids = np.setdiff1d(_gtaids, top_aids)
        # Sort missed grountruth by score
        _gtscores = qres.get_aid_scores(_gtaids)
        _gtaids = utool.sortedby(_gtaids, _gtscores, reverse=True)
        if len(_gtaids) > 3:
            # Hack to not show too many unmatched groundtruths
            #_isexmp = ibs.get_annot_exemplar_flags(_gtaids)
            _gtaids = _gtaids[0:3]
        showgt_aids = _gtaids

    return show_qres(ibs, qres, gt_aids=showgt_aids, top_aids=top_aids,
                     figtitle=figtitle, show_query=show_query, **kwargs)


@utool.indent_func
def show_qres(ibs, qres, **kwargs):
    """
    Display Query Result Logic

    Defaults to: query chip, groundtruth matches, and top matches
<<<<<<< HEAD
    python -c "import utool, ibeis; print(utool.auto_docstr('ibeis.viz.viz_qres', 'show_qres'))"

    Args:
        ibs (IBEISController):  ibeis controller object
        qres (QueryResult):  object of feature correspondences and scores
    
    Returns:
        ?: fig
    
    CommandLine:
        python -m ibeis.viz.viz_qres --test-show_qres
    
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_qres import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs.query_chips(ibs.get_valid_aids()[0:1])[0]
        >>> # execute function
        >>> fig = show_qres(ibs, qres, sidebyside=False, show_query=True, top_aids=3)
        >>> # verify results
        >>> fig.show()

=======
    qres.ishow calls down into this

    Kwargs:

        in_image (bool) show result  in image view if True else chip view

        annot_mode (int):
            if annot_mode == 0, then draw lines and ellipse
            elif annot_mode == 1, then dont draw lines or ellipse
            elif annot_mode == 2, then draw only lines
>>>>>>> 416025526ff287851988040477e64dd33b774a3c
    """
    annot_mode = kwargs.get('annot_mode', 1) % 3  # this is toggled
    figtitle    = kwargs.get('figtitle', '')
    make_figtitle = kwargs.get('make_figtitle', False)
    aug         = kwargs.get('aug', '')
    top_aids    = kwargs.get('top_aids', DEFAULT_NTOP)
    gt_aids     = kwargs.get('gt_aids',   [])
    all_kpts    = kwargs.get('all_kpts', False)
    show_query  = kwargs.get('show_query', False)
    in_image    = kwargs.get('in_image', False)
    sidebyside  = kwargs.get('sidebyside', True)
    name_scoring  = kwargs.get('name_scoring', False)

    fnum = df2.kwargs_fnum(kwargs)

    if make_figtitle is True:
        figtitle = qres.make_title(pack=True)

    fig = df2.figure(fnum=fnum, docla=True, doclf=True)

    if isinstance(top_aids, int):
        top_aids = qres.get_top_aids(num=top_aids, name_scoring=name_scoring, ibs=ibs)

    nTop   = len(top_aids)

    #max_nCols = 5
    max_nCols = 5
    if nTop in [6, 7]:
        max_nCols = 3
    if nTop in [8]:
        max_nCols = 4

    try:
        assert len(list(set(top_aids).intersection(set(gt_aids)))) == 0, 'gts should be missed.  not in top'
    except AssertionError as ex:
        utool.printex(ex, keys=['top_aids', 'gt_aids'])
        raise

    printDBG(qres.get_inspect_str())
    ranked_aids = qres.get_top_aids()
    #--------------------------------------------------
    # Get grid / cell information to build subplot grid
    #--------------------------------------------------
    # Show query or not
    nQuerySubplts = 1 if show_query else 0
    # The top row is given slots for ground truths and querys
    # all aids in gt_aids should not be in top aids
    nGtSubplts    = nQuerySubplts + (0 if gt_aids is None else len(gt_aids))
    # The bottom rows are for the top results
    nTopNSubplts  = nTop
    nTopNCols     = min(max_nCols, nTopNSubplts)
    nGTCols       = min(max_nCols, nGtSubplts)
    nGTCols       = max(nGTCols, nTopNCols)
    nTopNCols     = nGTCols
    # Get number of rows to show groundtruth
    nGtRows       = 0 if nGTCols   == 0 else int(np.ceil(nGtSubplts   / nGTCols))
    # Get number of rows to show results
    nTopNRows     = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells      = nGtRows * nGTCols
    # Total number of rows
    nRows         = nTopNRows + nGtRows

    DEBUG_SHOW_QRES = True

    if DEBUG_SHOW_QRES:
        allgt_aids = ibs.get_annot_groundtruth(qres.qaid)
        nSelGt = len(gt_aids)
        nAllGt = len(allgt_aids)
        print('[show_qres]========================')
        print('[show_qres]----------------')
        print('[show_qres] * annot_mode=%r' % (annot_mode,))
        print('[show_qres] #nTop=%r #missed_gts=%r/%r' % (nTop, nSelGt, nAllGt))
        print('[show_qres] * -----')
        print('[show_qres] * nRows=%r' % (nRows,))
        print('[show_qres] * nGtSubplts=%r' % (nGtSubplts,))
        print('[show_qres] * nTopNSubplts=%r' % (nTopNSubplts,))
        print('[show_qres] * nQuerySubplts=%r' % (nQuerySubplts,))
        print('[show_qres] * -----')
        print('[show_qres] * nGTCols=%r' % (nGTCols,))
        print('[show_qres] * -----')
        print('[show_qres] * fnum=%r' % (fnum,))
        print('[show_qres] * figtitle=%r' % (figtitle,))
        print('[show_qres] * max_nCols=%r' % (max_nCols,))
        print('[show_qres] * show_query=%r' % (show_query,))
        print('[show_qres] * kwargs=%s' % (utool.dict_str(kwargs),))

    # HACK:
    _color_list = df2.distinct_colors(nTop)
    aid2_color = {aid: _color_list[ox] for ox, aid in enumerate(top_aids)}

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        """ helper for show_qres """
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annot_mode)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['pnum'] = pnum
        _kwshow['aid2_color'] = aid2_color
        _kwshow['draw_ell'] = annot_mode >= 1
        viz_chip.show_chip(ibs, qres.qaid, annote=False, **_kwshow)

    def _plot_matches_aids(aid_list, plotx_shift, rowcols):
        """ helper for show_qres to draw many aids """
        def _show_matches_fn(aid, orank, pnum):
            """ Helper function for drawing matches to one aid """
            aug = 'rank=%r\n' % orank
            #printDBG('[show_qres()] plotting: %r'  % (pnum,))
            #draw_ell = annot_mode == 1
            #draw_lines = annot_mode >= 1
            _kwshow  = dict(draw_ell=annot_mode, draw_pts=False, draw_lines=annot_mode,
                            ell_alpha=.5, all_kpts=all_kpts)
            _kwshow.update(kwargs)
            _kwshow['fnum'] = fnum
            _kwshow['pnum'] = pnum
            _kwshow['title_aug'] = aug
            _kwshow['in_image'] = in_image
            # If we already are showing the query dont show it here
            if sidebyside:
                _kwshow['draw_ell'] = annot_mode == 1
                _kwshow['draw_lines'] = annot_mode >= 1
                viz_matches.show_matches(ibs, qres, aid, **_kwshow)
            else:
                _kwshow['draw_ell'] = annot_mode >= 1
                if annot_mode == 2:
                    # TODO Find a better name
                    _kwshow['color'] = aid2_color[aid]
                    _kwshow['sel_fx2'] = qres.aid2_fm[aid][:, 1]
                viz_chip.show_chip(ibs, aid, annote=False, **_kwshow)
                viz_matches.annotate_matches(ibs, qres, aid, show_query=not show_query)

        if DEBUG_SHOW_QRES:
            print('[show_qres()] Plotting Chips %s:' % vh.get_aidstrs(aid_list))
        if aid_list is None:
            return
        # Do lazy load before show
        vh.get_chips(ibs, aid_list, **kwargs)
        vh.get_kpts(ibs, aid_list, **kwargs)
        for ox, aid in enumerate(aid_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_aids == aid)[0]
            # This pair has no matches between them.
            if len(oranks) == 0:
                orank = -1
                _show_matches_fn(aid, orank, pnum)
                #if DEBUG_SHOW_QRES:
                #    print('skipping pnum=%r' % (pnum,))
                continue
            if DEBUG_SHOW_QRES:
                print('pnum=%r' % (pnum,))
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

if __name__ == '__main__':
    print ("Utool dir: %r" % (dir(utool),))
    utool.doctest_module()
