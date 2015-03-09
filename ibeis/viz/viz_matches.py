from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
import plottool.plot_helpers as ph
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[viz_matches]', DEBUG=False)


@utool.indent_func
def show_matches2(ibs, aid1, aid2, fm=None, fs=None, fm_norm=None, sel_fm=[],
                  H1=None, H2=None, qreq_=None, **kwargs):
    """
    TODO: use this as the main function.
    Have the qres version be a wrapper
    Integrate ChipMatch2
    """
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.get('draw_fmatches', True)
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [aid1, aid2], qreq_=qreq_, **kwargs)
    if draw_fmatches:
        kpts1, kpts2 = vh.get_kpts( ibs, [aid1, aid2], qreq_=qreq_, **kwargs)
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
        ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm,
                                               fs=fs, fm_norm=fm_norm,
                                               H1=H1, H2=H2, lbl1=lbl1, lbl2=lbl2, **kwargs)
    except Exception as ex:
        utool.printex(ex, 'consider qr.remove_corrupted_queries',
                      '[viz_matches]')
        print('')
        raise
    (x1, y1, w1, h1) = xywh1
    (x2, y2, w2, h2) = xywh2
    # TODO: MOVE TO ANNOTATE MATCHES
    if len(sel_fm) > 0:
        # Draw any selected matches
        sm_kw = dict(rect=True, colors=df2.BLUE)
        df2.plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_matches2(ibs, aid1, aid2, fm, fs, xywh2=xywh2, xywh1=xywh1,
                      offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


def annotate_matches2(ibs, aid1, aid2, fm, fs,
                      offset1=(0, 0),
                      offset2=(0, 0),
                      xywh2=None,  # (0, 0, 0, 0),
                      xywh1=None,  # (0, 0, 0, 0),
                      qreq_=None,
                      **kwargs):
    """
    TODO: use this as the main function.
    Have the qres version be a wrapper
    """
    # TODO Use this function when you clean show_matches
    in_image    = kwargs.get('in_image', False)
    show_query  = kwargs.get('show_query', True)
    draw_border = kwargs.get('draw_border', True)
    draw_lbl    = kwargs.get('draw_lbl', True)

    printDBG('[viz] annotate_matches2()')
    truth = ibs.get_match_truth(aid1, aid2)
    truth_color = vh.get_truth_color(truth)
    # Build title

    #score         = kwargs.pop('score', None)
    #rawscore      = kwargs.pop('rawscore', None)
    #aid2_raw_rank = kwargs.pop('aid2_raw_rank', None)
    print(kwargs)
    title = vh.get_query_text(ibs, None, aid2, truth, qaid=aid1, **kwargs)
    # Build xlbl
    ax = df2.gca()
    ph.set_plotdat(ax, 'viztype', 'matches')
    ph.set_plotdat(ax, 'qaid', aid1)
    ph.set_plotdat(ax, 'aid1', aid1)
    ph.set_plotdat(ax, 'aid2', aid2)
    if draw_lbl:
        name1, name2 = ibs.get_annot_names([aid1, aid2])
        nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2], distinguish_unknowns=False)
        #lbl1 = repr(name1)  + ' : ' + 'q' + vh.get_aidstrs(aid1)
        #lbl2 = repr(name2)  + ' : ' +  vh.get_aidstrs(aid2)
        lbl1_list = []
        lbl2_list = []
        if kwargs.get('show_aid', True):
            lbl1_list.append('q' + vh.get_aidstrs(aid1))
            lbl2_list.append(vh.get_aidstrs(aid2))
        if kwargs.get('show_name', True):
            lbl1_list.append(repr(str(name1)))
            lbl2_list.append(repr(str(name2)))
        if kwargs.get('show_nid', True):
            lbl1_list.append(vh.get_nidstrs(nid1))
            lbl2_list.append(vh.get_nidstrs(nid2))
        lbl1 = ' : '.join(lbl1_list)
        lbl2 = ' : '.join(lbl2_list)
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

        if xywh2 is None:
            #xywh2 = (xy[0], xy[1], w, h)
            # weird when sidebyside is off y seems to be inverted
            xywh2 = (0,  0, w, h)

        if not show_query and xywh1 is None:
            kpts2, = ibs.get_annot_kpts((aid2,), qreq_=qreq_)
            #df2.draw_kpts2(kpts2.take(fm.T[1], axis=0))
            # Draw any selected matches
            #sm_kw = dict(rect=True, colors=df2.BLUE)
            df2.plot_fmatch(None, xywh2, None, kpts2, fm, fs=fs, **kwargs)
        if draw_border:
            df2.draw_border(ax, truth_color, 4, offset=offset2)
        if draw_lbl:
            # Custom user lbl for chips 1 and 2
            if show_query:
                (x1, y1, w1, h1) = xywh1
                df2.absolute_lbl(x1 + w1, y1, lbl1)
            (x2, y2, w2, h2) = xywh2
            df2.absolute_lbl(x2 + w2, y2, lbl2)
        # No matches draw a red box
    if fm is None or len(fm) == 0:
        if draw_border:
            df2.draw_boxedX(bbox2, theta=theta2)


# OLD QRES BASED FUNCS STILL IN USE


@utool.indent_func
def show_matches(ibs, qres, aid2, sel_fm=[], qreq_=None, **kwargs):
    """
    shows single annotated match result.

    Args:
        ibs (IBEISController):
        qres (QueryResult):  object of feature correspondences and scores
        aid2 (int): result annotation id
        sel_fm (list): selected features match indices

    Kwargs:
        vert (bool)

    Returns:
        tuple: (ax, xywh1, xywh2)

    CommandLine:
        python -m ibeis.viz.viz_matches --test-show_matches --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_matches import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
        >>> aid2 = 2
        >>> sel_fm = []
        >>> # execute function
        >>> (ax, xywh1, xywh2) = show_matches(ibs, qres, aid2, sel_fm)
        >>> # verify results
        >>> result = str((ax, xywh1, xywh2))
        >>> print(result)
        >>> #if not utool.get_argflag('--noshow'):
        >>> if utool.get_argflag('--show'):
        >>>    execstr = df2.present()
        >>>    exec(execstr)
    """
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.get('draw_fmatches', True)
    aid1 = qres.qaid
    fm = qres.aid2_fm.get(aid2, [])
    fs = qres.aid2_fs.get(aid2, [])
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = vh.get_chips(ibs, [aid1, aid2], qreq_=qreq_, **kwargs)
    if draw_fmatches:
        kpts1, kpts2 = vh.get_kpts( ibs, [aid1, aid2], qreq_=qreq_, **kwargs)
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
    # TODO: MOVE TO ANNOTATE MATCHES
    if len(sel_fm) > 0:
        # Draw any selected matches
        sm_kw = dict(rect=True, colors=df2.BLUE)
        df2.plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_matches(ibs, qres, aid2, xywh2=xywh2, xywh1=xywh1,
                     offset1=offset1, offset2=offset2, qreq_=qreq_, **kwargs)
    return ax, xywh1, xywh2


@utool.indent_func
def annotate_matches(ibs, qres, aid2,
                     offset1=(0, 0),
                     offset2=(0, 0),
                     xywh2=None,  # (0, 0, 0, 0),
                     xywh1=None,  # (0, 0, 0, 0),
                     qreq_=None,
                     **kwargs):
    """
    Helper function
    Draws annotation on top of a matching chip plot

    does not draw feature matches. that is done in plottool.draw_func2.show_chipmatch2
    this handles things like the labels and borders based on
    groundtruth score
    """
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
        nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2], distinguish_unknowns=False)
        #lbl1 = repr(name1)  + ' : ' + 'q' + vh.get_aidstrs(aid1)
        #lbl2 = repr(name2)  + ' : ' +  vh.get_aidstrs(aid2)
        lbl1_list = []
        lbl2_list = []
        if kwargs.get('show_aid', True):
            lbl1_list.append('q' + vh.get_aidstrs(aid1))
            lbl2_list.append(vh.get_aidstrs(aid2))
        if kwargs.get('show_name', True):
            lbl1_list.append(repr(name1))
            lbl2_list.append(repr(name2))
        if kwargs.get('show_nid', True):
            lbl1_list.append(vh.get_nidstrs(nid1))
            lbl2_list.append(vh.get_nidstrs(nid2))
        lbl1 = ' : '.join(lbl1_list)
        lbl2 = ' : '.join(lbl2_list)
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

        if xywh2 is None:
            #xywh2 = (xy[0], xy[1], w, h)
            # weird when sidebyside is off y seems to be inverted
            xywh2 = (0,  0, w, h)

        if not show_query and xywh1 is None:
            fm = qres.aid2_fm.get(aid2, [])
            fs = qres.aid2_fs.get(aid2, [])
            kpts2, = ibs.get_annot_kpts((aid2,), qreq_=qreq_)
            #df2.draw_kpts2(kpts2.take(fm.T[1], axis=0))
            # Draw any selected matches
            #sm_kw = dict(rect=True, colors=df2.BLUE)
            df2.plot_fmatch(None, xywh2, None, kpts2, fm, fs=fs, **kwargs)
        if draw_border:
            df2.draw_border(ax, truth_color, 4, offset=offset2)
        if draw_lbl:
            # Custom user lbl for chips 1 and 2
            if show_query:
                (x1, y1, w1, h1) = xywh1
                df2.absolute_lbl(x1 + w1, y1, lbl1)
            (x2, y2, w2, h2) = xywh2
            df2.absolute_lbl(x2 + w2, y2, lbl2)
        # No matches draw a red box
    if aid2 not in qres.aid2_fm or len(qres.aid2_fm[aid2]) == 0:
        if draw_border:
            df2.draw_boxedX(bbox2, theta=theta2)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_matches
        python -m ibeis.viz.viz_matches --allexamples
        python -m ibeis.viz.viz_matches --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
