from __future__ import absolute_import, division, print_function
import utool as ut
import plottool.draw_func2 as df2
import plottool.plot_helpers as ph
from ibeis.viz import viz_helpers as vh
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[viz_matches]', DEBUG=False)


def _get_annot_pair_info(ibs, aid1, aid2, qreq_, draw_fmatches):
    rchip1, kpts1 = get_query_annot_pair_info(ibs, aid1, qreq_, draw_fmatches)
    rchip2, kpts2 = ut.get_list_column(get_data_annot_pair_info(ibs, [aid2], qreq_, draw_fmatches), 0)
    return rchip1, rchip2, kpts1, kpts2


def get_query_annot_pair_info(ibs, qaid, qreq_, draw_fmatches):
    query_config2_ = None if qreq_ is None else qreq_.get_external_query_config2()
    rchip1 = vh.get_chips(ibs, [qaid], config2_=query_config2_)[0]
    if draw_fmatches:
        kpts1 = vh.get_kpts(ibs, [qaid], config2_=query_config2_)[0]
    else:
        kpts1 = None
    return rchip1, kpts1


def get_data_annot_pair_info(ibs, aid_list, qreq_, draw_fmatches):
    data_config2_ = None if qreq_ is None else qreq_.get_external_data_config2()
    rchip2_list = vh.get_chips(ibs, aid_list, config2_=data_config2_)
    if draw_fmatches:
        kpts2_list = vh.get_kpts(ibs, aid_list, config2_=data_config2_)
    else:
        kpts2_list = [None] * len(aid_list)
    return rchip2_list, kpts2_list


def show_name_matches(ibs, qaid, name_daid_list, name_fm_list, name_fs_list, name_H1_list, qreq_=None, **kwargs):
    """
    kwargs = {}
    draw_fmatches = True

    CommandLine:
        python -m ibeis.viz.viz_matches --test-show_name_matches --show --verobse

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots import chip_match
        >>> from ibeis.viz.viz_matches import *  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
        >>> func = chip_match.ChipMatch2.show_single_namematch
        >>> sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
        >>> setup = ut.regex_replace('viz_matches.show_name_matches', '#', sourcecode)
        >>> print(setup)
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1])
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> qaid = cm.qaid
        >>> dnid = ibs.get_annot_nids(cm.qaid)
        >>> nidx = cm.nid2_nidx[dnid]
        >>> groupxs = cm.name_groupxs[nidx]
        >>> name_daid_list = ut.list_take(cm.daid_list, groupxs)
        >>> name_fm_list   = ut.list_take(cm.fm_list, groupxs)
        >>> homog = False
        >>> name_H1_list   = None if not homog or cm.H_list is None else ut.list_take(cm.H_list, groupxs)
        >>> name_fsv_list  = None if cm.fsv_list is None else ut.list_take(cm.fsv_list, groupxs)
        >>> name_fs_list   = None if name_fsv_list is None else [fsv.prod(axis=1) for fsv in name_fsv_list]
        >>> kwargs = {}
        >>> show_name_matches(ibs, qaid, name_daid_list, name_fm_list, name_fs_list, name_H1_list, qreq_=qreq_, **kwargs)
        >>> ut.quit_if_noshow()
        >>> ut.show_if_requested()
    """
    draw_fmatches = kwargs.get('draw_fmatches', True)
    rchip1, kpts1 = get_query_annot_pair_info(ibs, qaid, qreq_, draw_fmatches)
    rchip2_list, kpts2_list = get_data_annot_pair_info(ibs, name_daid_list, qreq_, draw_fmatches)
    fm_list = name_fm_list
    fs_list = name_fs_list
    show_multichip_match(rchip1, rchip2_list, kpts1, kpts2_list, fm_list, fs_list)


def show_multichip_match(rchip1, rchip2_list, kpts1, kpts2_list, fm_list, fs_list, fnum=None, pnum=None):
    """ move to df2
    rchip = rchip1
    H = H1 = None
    target_wh = None

    """
    import vtool.image as gtool
    import plottool as pt
    import numpy as np
    def preprocess_chips(rchip, H, target_wh):
        rchip_ = rchip if H is None else gtool.warpHomog(rchip, H, target_wh)
        return rchip_

    if fnum is None:
        fnum = pt.next_fnum()

    target_wh1 = None
    H1 = None
    rchip1_ = preprocess_chips(rchip1, H1, target_wh1)
    wh1 = gtool.get_size(rchip1_)
    rchip2_list_ = [preprocess_chips(rchip2, None, wh1) for rchip2 in rchip2_list]
    wh2_list = [gtool.get_size(rchip2) for rchip2 in rchip2_list_]

    match_img, offset_list, sf_list = pt.stack_image_list_special(rchip1_, rchip2_list_)

    wh_list = np.array(ut.flatten([[wh1], wh2_list])) * sf_list

    offset1 = offset_list[0]
    wh1 = wh_list[0]
    sf1 = sf_list[0]

    fig, ax = pt.imshow(match_img, fnum=fnum, pnum=pnum)

    for offset2, wh2, sf2, kpts2, fm2, fs2 in zip(offset_list[1:], wh_list[1:], sf_list[1:], kpts2_list, fm_list, fs_list):
        xywh1 = (offset1[0], offset1[1], wh1[0], wh1[1])
        xywh2 = (offset2[0], offset2[1], wh2[0], wh2[1])
        if kpts1 is not None and kpts2 is not None:
            pt.plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm2, fs2, fm_norm=None,
                           H1=None, H2=None, scale_factor1=sf1,
                           scale_factor2=sf2, colorbar_=False)

    # Show the stacked chips
    #annotate_matches2(ibs, aid1, aid2, fm, fs, xywh2=xywh2, xywh1=xywh1,
    #                  offset1=offset1, offset2=offset2, **kwargs)


#@ut.indent_func
def show_matches2(ibs, aid1, aid2, fm=None, fs=None, fm_norm=None, sel_fm=[],
                  H1=None, H2=None, qreq_=None, **kwargs):
    """
    TODO: use this as the main function.
    Have the qres version be a wrapper
    Integrate ChipMatch2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> cm.show_single_annotmatch(qreq_, daid)
    """
    if qreq_ is None:
        print('[viz_matches] WARNING: qreq_ is None')
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.get('draw_fmatches', True)
    # Read query and result info (chips, names, ...)
    rchip1, rchip2, kpts1, kpts2 = _get_annot_pair_info(ibs, aid1, aid2, qreq_, draw_fmatches)

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
        ut.printex(ex, 'consider qr.remove_corrupted_queries',
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

    #printDBG('[viz] annotate_matches2()')
    truth = ibs.get_match_truth(aid1, aid2)
    truth_color = vh.get_truth_color(truth)
    # Build title

    #score         = kwargs.pop('score', None)
    #rawscore      = kwargs.pop('rawscore', None)
    #aid2_raw_rank = kwargs.pop('aid2_raw_rank', None)
    #print(kwargs)
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
            data_config2 = None if qreq_ is None else qreq_.get_external_data_config2()
            kpts2 = ibs.get_annot_kpts([aid2], config2_=data_config2)[0]
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


#@ut.indent_func
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
        >>> #if not ut.get_argflag('--noshow'):
        >>> if ut.get_argflag('--show'):
        >>>    execstr = df2.present()
        >>>    exec(execstr)
    """
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.get('draw_fmatches', True)
    aid1 = qres.qaid
    fm = qres.aid2_fm.get(aid2, [])
    fs = qres.aid2_fs.get(aid2, [])
    # Read query and result info (chips, names, ...)
    rchip1, rchip2, kpts1, kpts2 = _get_annot_pair_info(ibs, aid1, aid2, qreq_, draw_fmatches)

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
        ut.printex(ex, 'consider qr.remove_corrupted_queries',
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


#@ut.indent_func
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

    DEPRICATE

    does not draw feature matches. that is done in plottool.draw_func2.show_chipmatch2
    this handles things like the labels and borders based on
    groundtruth score
    """
    # TODO Use this function when you clean show_matches
    fm = qres.aid2_fm.get(aid2, [])
    fs = qres.aid2_fs.get(aid2, [])
    aid1 = qres.qaid
    #truth = ibs.get_match_truth(aid1, aid2)
    #title = vh.get_query_text(ibs, qres, aid2, truth, **kwargs)
    score = qres.get_aid_scores([aid2])[0]
    rawscore = qres.get_aid_scores([aid2], rawscore=True)[0]
    aid2_raw_rank = qres.get_aid_ranks([aid2])[0]

    return annotate_matches2(ibs, aid1, aid2, fm, fs, offset1, offset2, xywh2, xywh1, qreq_, score=score, rawscore=rawscore, aid2_raw_rank=aid2_raw_rank, **kwargs)

    #in_image    = kwargs.get('in_image', False)
    #show_query  = kwargs.get('show_query', True)
    #draw_border = kwargs.get('draw_border', True)
    #draw_lbl    = kwargs.get('draw_lbl', True)

    #printDBG('[viz] annotate_matches()')
    #truth_color = vh.get_truth_color(truth)
    ## Build title
    ## Build xlbl
    #ax = df2.gca()
    #ph.set_plotdat(ax, 'viztype', 'matches')
    #ph.set_plotdat(ax, 'qaid', aid1)
    #ph.set_plotdat(ax, 'aid1', aid1)
    #ph.set_plotdat(ax, 'aid2', aid2)
    #if draw_lbl:
    #    name1, name2 = ibs.get_annot_names([aid1, aid2])
    #    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2], distinguish_unknowns=False)
    #    #lbl1 = repr(name1)  + ' : ' + 'q' + vh.get_aidstrs(aid1)
    #    #lbl2 = repr(name2)  + ' : ' +  vh.get_aidstrs(aid2)
    #    lbl1_list = []
    #    lbl2_list = []
    #    if kwargs.get('show_aid', True):
    #        lbl1_list.append('q' + vh.get_aidstrs(aid1))
    #        lbl2_list.append(vh.get_aidstrs(aid2))
    #    if kwargs.get('show_name', True):
    #        lbl1_list.append(repr(name1))
    #        lbl2_list.append(repr(name2))
    #    if kwargs.get('show_nid', True):
    #        lbl1_list.append(vh.get_nidstrs(nid1))
    #        lbl2_list.append(vh.get_nidstrs(nid2))
    #    lbl1 = ' : '.join(lbl1_list)
    #    lbl2 = ' : '.join(lbl2_list)
    #else:
    #    lbl1, lbl2 = None, None
    #if vh.NO_LBL_OVERRIDE:
    #    title = ''
    #df2.set_title(title, ax)
    ## Plot annotations over images
    #if in_image:
    #    bbox1, bbox2 = vh.get_bboxes(ibs, [aid1, aid2], [offset1, offset2])
    #    theta1, theta2 = ibs.get_annot_thetas([aid1, aid2])
    #    # HACK!
    #    if show_query:
    #        df2.draw_bbox(bbox1, bbox_color=df2.ORANGE, lbl=lbl1, theta=theta1)
    #    bbox_color2 = truth_color if draw_border else df2.ORANGE
    #    df2.draw_bbox(bbox2, bbox_color=bbox_color2, lbl=lbl2, theta=theta2)
    #else:
    #    xy, w, h = df2.get_axis_xy_width_height(ax)
    #    bbox2 = (xy[0], xy[1], w, h)
    #    theta2 = 0

    #    if xywh2 is None:
    #        #xywh2 = (xy[0], xy[1], w, h)
    #        # weird when sidebyside is off y seems to be inverted
    #        xywh2 = (0,  0, w, h)

    #    if not show_query and xywh1 is None:
    #        data_config2 = None if qreq_ is None else qreq_.get_external_data_config2()
    #        kpts2 = ibs.get_annot_kpts([aid2], config2_=data_config2)[0]
    #        #df2.draw_kpts2(kpts2.take(fm.T[1], axis=0))
    #        # Draw any selected matches
    #        #sm_kw = dict(rect=True, colors=df2.BLUE)
    #        df2.plot_fmatch(None, xywh2, None, kpts2, fm, fs=fs, **kwargs)
    #    if draw_border:
    #        df2.draw_border(ax, truth_color, 4, offset=offset2)
    #    if draw_lbl:
    #        # Custom user lbl for chips 1 and 2
    #        if show_query:
    #            (x1, y1, w1, h1) = xywh1
    #            df2.absolute_lbl(x1 + w1, y1, lbl1)
    #        (x2, y2, w2, h2) = xywh2
    #        df2.absolute_lbl(x2 + w2, y2, lbl2)
    #    # No matches draw a red box
    #if fm is None or len(fm) == 0:
    #    if draw_border:
    #        df2.draw_boxedX(bbox2, theta=theta2)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_matches --test-show_matches --show

        python -m ibeis.viz.viz_matches
        python -m ibeis.viz.viz_matches --allexamples
        python -m ibeis.viz.viz_matches --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
