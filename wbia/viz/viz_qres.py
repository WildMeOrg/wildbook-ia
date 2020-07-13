# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import wbia.plottool as pt
import utool as ut
import numpy as np
from wbia.other import ibsfuncs
from wbia.viz import viz_helpers as vh
from wbia.viz import viz_chip
from wbia.viz import viz_matches  # NOQA

(print, rrr, profile) = ut.inject2(__name__)


DEFAULT_NTOP = 3


def show_qres_top(ibs, cm, qreq_=None, **kwargs):
    """
    Wrapper around show_qres.
    """
    N = kwargs.get('N', DEFAULT_NTOP)
    # name_scoring = kwargs.get('name_scoring', False)
    # if isinstance(cm, chip_match.ChipMatch):
    top_aids = cm.get_top_aids(N)
    # else:
    #    top_aids = cm.get_top_aids(num=N, ibs=ibs, name_scoring=name_scoring)
    aidstr = ibsfuncs.aidstr(cm.qaid)
    figtitle = kwargs.get('figtitle', '')
    if len(figtitle) > 0:
        figtitle = ' ' + figtitle
    kwargs['figtitle'] = ('q%s -- TOP %r' % (aidstr, N)) + figtitle
    return show_qres(
        ibs,
        cm,
        top_aids=top_aids,
        qreq_=qreq_,
        # dont use these. use annot mode instead
        # draw_kpts=False,
        # draw_ell=False,
        # all_kpts=False,
        **kwargs,
    )


def show_qres_analysis(ibs, cm, qreq_=None, **kwargs):
    """
    Wrapper around show_qres.

    KWARGS:
        aid_list - show matches against aid_list (default top 3)

    Args:
        ibs (IBEISController):  wbia controller object
        cm (ChipMatch):  object of feature correspondences and scores
        qreq_ (QueryRequest):  query request object with hyper-parameters(default = None)

    Kwargs:
        N, show_gt, show_query, aid_list, figtitle, viz_name_score, viz_name_score

    CommandLine:
        python -m wbia.viz.viz_qres --exec-show_qres_analysis --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.viz.viz_qres import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm(
        >>>     defaultdb='PZ_MTEST', default_qaids=[1],
        >>>     default_daids=[2, 3, 4, 5, 6, 7, 8, 9])
        >>> kwargs = dict(show_query=False, viz_name_score=True,
        >>>               show_timedelta=True, N=3, show_gf=True)
        >>> ibs = qreq_.ibs
        >>> show_qres_analysis(ibs, cm, qreq_, **kwargs)
        >>> ut.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.viz.viz_qres import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm(
        >>>     defaultdb='PZ_MTEST', default_qaids=[1],
        >>>     default_daids=[2])
        >>> kwargs = dict(show_query=False, viz_name_score=True,
        >>>               show_timedelta=True, N=3, show_gf=True)
        >>> ibs = qreq_.ibs
        >>> show_qres_analysis(ibs, cm, qreq_, **kwargs)
        >>> ut.show_if_requested()
    """
    if ut.NOT_QUIET:
        print('[show_qres] cm.show_analysis()')
    # Parse arguments
    N = kwargs.get('N', DEFAULT_NTOP)
    show_gt = kwargs.pop('show_gt', True)
    show_gf = kwargs.pop('show_gf', False)
    show_query = kwargs.pop('show_query', True)
    aid_list = kwargs.pop('aid_list', None)
    figtitle = kwargs.pop('figtitle', None)
    viz_name_score = kwargs.get('viz_name_score', True)
    failed_to_match = False

    if aid_list is None:
        # Compare to aid_list instead of using top ranks
        top_aids = cm.get_top_aids(N)
        if len(top_aids) == 0:
            failed_to_match = True
            print('WARNING! No matches found for this query')
        if figtitle is None:
            if len(top_aids) == 0:
                figtitle = 'WARNING: no matches found!' + ibsfuncs.aidstr(cm.qaid)
            else:
                topscore = cm.get_annot_scores(top_aids)[0]
                figtitle = 'q%s -- topscore=%r' % (ibsfuncs.aidstr(cm.qaid), topscore)
    else:
        print('[analysis] showing a given list of aids')
        top_aids = aid_list
        if figtitle is None:
            figtitle = 'comparing to ' + ibsfuncs.aidstr(top_aids) + figtitle

    # Get any groundtruth if you are showing it
    showgt_aids = []
    if show_gt:
        # Get the missed groundtruth annotations
        assert qreq_ is not None
        matchable_aids = qreq_.daids
        _gtaids = ibs.get_annot_groundtruth(cm.qaid, daid_list=matchable_aids)

        if viz_name_score:
            # Only look at the groundtruth if a name isnt in the top list
            _gtnids = ibs.get_annot_name_rowids(_gtaids)
            top_nids = ibs.get_annot_name_rowids(top_aids)
            _valids = ~np.in1d(_gtnids, top_nids)
            _gtaids = ut.compress(_gtaids, _valids)

        # No need to display highly ranked groundtruth. It will already show up
        _gtaids = np.setdiff1d(_gtaids, top_aids)
        # Sort missed grountruth by score
        _gtscores = cm.get_annot_scores(_gtaids)
        _gtaids = ut.sortedby(_gtaids, _gtscores, reverse=True)
        if viz_name_score:
            if len(_gtaids) > 1:
                _gtaids = _gtaids[0:1]
        else:
            if len(_gtaids) > 3:
                # Hack to not show too many unmatched groundtruths
                # _isexmp = ibs.get_annot_exemplar_flags(_gtaids)
                _gtaids = _gtaids[0:3]
        showgt_aids = _gtaids

    if show_gf:
        # Show only one top-scoring groundfalse example
        top_nids = ibs.get_annot_name_rowids(top_aids)
        is_groundfalse = top_nids != ibs.get_annot_name_rowids(cm.qaid)
        gf_idxs = np.nonzero(is_groundfalse)[0]
        if len(gf_idxs) > 0:
            best_gf_idx = gf_idxs[0]
            isvalid = ~is_groundfalse
            isvalid[best_gf_idx] = True
            # Filter so there is only one groundfalse
            top_aids = top_aids.compress(isvalid)
        # else:
        #     if len(top_aids) == 1:
        #         top_aids = top_aids.tolist()
        #         top_aids.append(None)

        # else:
        #     # seems like there were no results. Must be bad feature detections
        #     # maybe too much spatial verification
        #     top_aids = []

        # if len(showgt_aids) != 0:
        #     # Hack to just include gtaids in normal list
        #     top_aids = np.append(top_aids, showgt_aids)
        #     showgt_aids = []

    if viz_name_score:
        # Make sure that there is only one of each name in the list
        top_nids = ibs.get_annot_name_rowids(top_aids)
        top_aids = ut.compress(top_aids, ut.flag_unique_items(top_nids))

    return show_qres(
        ibs,
        cm,
        gt_aids=showgt_aids,
        top_aids=top_aids,
        figtitle=figtitle,
        show_query=show_query,
        qreq_=qreq_,
        failed_to_match=failed_to_match,
        **kwargs,
    )


def show_qres(ibs, cm, qreq_=None, **kwargs):
    r"""
    Display Query Result Logic
    Defaults to: query chip, groundtruth matches, and top matches

    Args:
        ibs (wbia.IBEISController): wbia controller object
        cm (wbia.ChipMatch): object of feature correspondences and scores
        qreq_ (wbia.QueryRequest):  query request object with hyper-parameters(default = None)

    Kwargs:
        annot_mode, figtitle, make_figtitle, aug, top_aids, all_kpts,
        show_query, in_image, sidebyside, name_scoring, max_nCols,
        failed_to_match, fnum
        in_image (bool) show result  in image view if True else chip view
        annot_mode (int):
            if annot_mode == 0, then draw lines and ellipse
            elif annot_mode == 1, then dont draw lines or ellipse
            elif annot_mode == 2, then draw only lines
            elif annot_mode == 3, draw heatmask only
        See: viz_matches.show_name_matches, viz_helpers.get_query_text

    Returns:
        mpl.Figure: fig

    CommandLine:
        ./main.py --query 1 -y --db PZ_MTEST --noshow-qres
        python -m wbia.viz.viz_qres show_qres --show
        python -m wbia.viz.viz_qres show_qres --show --top-aids=10 --db=PZ_MTEST \
                --sidebyside --annot_mode=0 --notitle --no-viz_name_score \
                --qaids=5 --max_nCols=2 --adjust=.01,.01,.01

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_qres import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> kwargs = dict(
        >>>     top_aids=ut.get_argval('--top-aids', type_=int, default=3),
        >>>     sidebyside=not ut.get_argflag('--no-sidebyside'),
        >>>     annot_mode=ut.get_argval('--annot_mode', type_=int, default=1),
        >>>     viz_name_score=not ut.get_argflag('--no-viz_name_score'),
        >>>     simplemode=ut.get_argflag('--simplemode'),
        >>>     max_nCols=ut.get_argval('--max_nCols', type_=int, default=None)
        >>> )
        >>> ibs = qreq_.ibs
        >>> fig = show_qres(ibs, cm, show_query=False, qreq_=qreq_, **kwargs)
        >>> ut.show_if_requested()
    """
    # ut.print_dict(kwargs)
    annot_mode = kwargs.get('annot_mode', 1) % 4  # this is toggled
    figtitle = kwargs.get('figtitle', '')
    aug = kwargs.get('aug', '')
    top_aids = kwargs.get('top_aids', DEFAULT_NTOP)
    gt_aids = kwargs.get('gt_aids', [])
    all_kpts = kwargs.get('all_kpts', False)
    show_query = kwargs.get('show_query', False)
    in_image = kwargs.get('in_image', False)
    sidebyside = kwargs.get('sidebyside', True)
    simplemode = kwargs.get('simplemode', False)
    colorbar_ = kwargs.get('colorbar_', False)
    # name_scoring   = kwargs.get('name_scoring', False)
    viz_name_score = kwargs.get('viz_name_score', qreq_ is not None)
    max_nCols = kwargs.get('max_nCols', None)
    failed_to_match = kwargs.get('failed_to_match', False)

    fnum = pt.ensure_fnum(kwargs.get('fnum', None))

    if ut.VERBOSE and ut.NOT_QUIET:
        print(
            'query_info = '
            + ut.repr2(
                ibs.get_annot_info(
                    cm.qaid,
                    default=True,
                    gname=False,
                    name=False,
                    notes=False,
                    exemplar=False,
                ),
                nl=4,
            )
        )
        print(
            'top_aids_info = '
            + ut.repr2(
                ibs.get_annot_info(
                    top_aids,
                    default=True,
                    gname=False,
                    name=False,
                    notes=False,
                    exemplar=False,
                    reference_aid=cm.qaid,
                ),
                nl=4,
            )
        )

    fig = pt.figure(fnum=fnum, docla=True, doclf=True)

    if isinstance(top_aids, int):
        top_aids = cm.get_top_aids(top_aids)

    if failed_to_match:
        # HACK to visually indicate failure to match in analysis
        show_query = True
        top_aids = [None] + top_aids

    nTop = len(top_aids)

    if max_nCols is None:
        max_nCols = 5
        if nTop in [6, 7]:
            max_nCols = 3
        if nTop in [8]:
            max_nCols = 4

    try:
        assert (
            len(list(set(top_aids).intersection(set(gt_aids)))) == 0
        ), 'gts should be missed.  not in top'
    except AssertionError as ex:
        ut.printex(ex, keys=['top_aids', 'gt_aids'])
        raise

    if ut.DEBUG2:
        print(cm.get_inspect_str())

    # --------------------------------------------------
    # Get grid / cell information to build subplot grid
    # --------------------------------------------------
    # Show query or not
    nQuerySubplts = 1 if show_query else 0
    # The top row is given slots for ground truths and querys
    # all aids in gt_aids should not be in top aids
    nGtSubplts = nQuerySubplts + (0 if gt_aids is None else len(gt_aids))
    # The bottom rows are for the top results
    nTopNSubplts = nTop
    nTopNCols = min(max_nCols, nTopNSubplts)
    nGTCols = min(max_nCols, nGtSubplts)
    nGTCols = max(nGTCols, nTopNCols)
    nTopNCols = nGTCols
    # Get number of rows to show groundtruth
    nGtRows = 0 if nGTCols == 0 else int(np.ceil(nGtSubplts / nGTCols))
    # Get number of rows to show results
    nTopNRows = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells = nGtRows * nGTCols
    # Total number of rows
    nRows = nTopNRows + nGtRows

    DEBUG_SHOW_QRES = 0

    if DEBUG_SHOW_QRES:
        allgt_aids = ibs.get_annot_groundtruth(cm.qaid)
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
        print('[show_qres] * kwargs=%s' % (ut.repr2(kwargs),))

    # HACK:
    _color_list = pt.distinct_colors(nTop)
    aid2_color = {aid: _color_list[ox] for ox, aid in enumerate(top_aids)}
    ranked_aids = cm.get_top_aids()

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        """ helper for show_qres """
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        # print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annot_mode)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['pnum'] = pnum
        _kwshow['aid2_color'] = aid2_color
        _kwshow['draw_ell'] = annot_mode >= 1
        viz_chip.show_chip(ibs, cm.qaid, annote=False, qreq_=qreq_, **_kwshow)

    def _plot_matches_aids(aid_list, plotx_shift, rowcols):
        """ helper for show_qres to draw many aids """
        _kwshow = dict(
            draw_ell=annot_mode,
            draw_pts=False,
            draw_lines=annot_mode,
            ell_alpha=0.5,
            all_kpts=all_kpts,
        )
        _kwshow.update(kwargs)
        _kwshow['fnum'] = fnum
        _kwshow['in_image'] = in_image
        _kwshow['colorbar_'] = colorbar_
        if sidebyside:
            # Draw each match side by side the query
            _kwshow['draw_ell'] = annot_mode in {1}
            _kwshow['draw_lines'] = annot_mode in {1, 2}
            _kwshow['heatmask'] = annot_mode in {3}
        else:
            # print('annot_mode = %r' % (annot_mode,))
            _kwshow['draw_ell'] = annot_mode == 1
            # _kwshow['draw_pts'] = annot_mode >= 1
            # _kwshow['draw_lines'] = False
            _kwshow['show_query'] = False

        def _show_matches_fn(aid, orank, pnum):
            """ Helper function for drawing matches to one aid """
            aug = 'rank=%r\n' % orank
            _kwshow['pnum'] = pnum
            _kwshow['title_aug'] = aug
            # draw_ell = annot_mode == 1
            # draw_lines = annot_mode >= 1
            # If we already are showing the query dont show it here
            if sidebyside:
                # Draw each match side by side the query
                if viz_name_score:
                    cm.show_single_namematch(qreq_, ibs.get_annot_nids(aid), **_kwshow)
                else:
                    if simplemode:
                        _kwshow['draw_border'] = False
                        _kwshow['draw_lbl'] = False
                        _kwshow['notitle'] = True
                        _kwshow['vert'] = False
                        _kwshow['modifysize'] = True
                    cm.show_single_annotmatch(qreq_, aid, **_kwshow)
                    # viz_matches.show_matches(ibs, cm, aid, qreq_=qreq_, **_kwshow)
            else:
                # Draw each match by themselves
                data_config2_ = None if qreq_ is None else qreq_.extern_data_config2
                # _kwshow['draw_border'] = kwargs.get('draw_border', True)
                # _kwshow['notitle'] = ut.get_argflag(('--no-title', '--notitle'))
                viz_chip.show_chip(
                    ibs,
                    aid,
                    annote=False,
                    notitle=True,
                    data_config2_=data_config2_,
                    **_kwshow,
                )

        if DEBUG_SHOW_QRES:
            print('[show_qres()] Plotting Chips %s:' % vh.get_aidstrs(aid_list))
        if aid_list is None:
            return
        # Do lazy load before show
        # data_config2_ = None if qreq_ is None else qreq_.extern_data_config2

        # tblhack = getattr(qreq_, 'tablename', None)
        # HACK FOR HUMPBACKS
        # (Also in viz_matches)
        # if tblhack == 'vsone' or (qreq_ is not None and not qreq_._isnewreq):
        #     # precompute
        #     pass
        #     #ibs.get_annot_chips(aid_list, config2_=data_config2_, ensure=True)
        #     #ibs.get_annot_kpts(aid_list, config2_=data_config2_, ensure=True)

        for ox, aid in enumerate(aid_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_aids == aid)[0]
            # This pair has no matches between them.
            if len(oranks) == 0:
                orank = -1
                if aid is None:
                    pt.imshow_null(
                        'Failed to find matches\nfor qaid=%r' % (cm.qaid),
                        fnum=fnum,
                        pnum=pnum,
                        fontsize=18,
                    )
                else:
                    _show_matches_fn(aid, orank, pnum)
                # if DEBUG_SHOW_QRES:
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
        pt.imshow_null('[viz_qres] No matches. nRows=0', fnum=fnum)
    else:
        fig = pt.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), docla=True, doclf=True)
        pt.plt.subplot(nRows, nGTCols, 1)
        # Plot Query
        if show_query:
            _show_query_fn(0, (nRows, nGTCols))
        # Plot Ground Truth (if given)
        _plot_matches_aids(gt_aids, nQuerySubplts, (nRows, nGTCols))
        # Plot Results
        _plot_matches_aids(top_aids, shift_topN, (nRows, nTopNCols))
        figtitle += aug
    if failed_to_match:
        figtitle += '\n No matches found'

    incanvas = kwargs.get('with_figtitle', not vh.NO_LBL_OVERRIDE)
    pt.set_figtitle(figtitle, incanvas=incanvas)

    # Result Interaction
    return fig


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_qres
        python -m wbia.viz.viz_qres --allexamples
        python -m wbia.viz.viz_qres --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
