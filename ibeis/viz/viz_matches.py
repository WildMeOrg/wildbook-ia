from __future__ import absolute_import, division, print_function
import utool as ut
import plottool as pt
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


def show_name_matches(ibs, qaid, name_daid_list, name_fm_list, name_fs_list,
                      name_H1_list, name_featflag_list, qreq_=None, **kwargs):
    """
    kwargs = {}
    draw_fmatches = True

    Called from chip_match.py

    CommandLine:
        python -m ibeis.viz.viz_matches --test-show_name_matches --show --verbose

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots import chip_match
        >>> from ibeis.model.hots import name_scoring
        >>> from ibeis.viz.viz_matches import *  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
        >>> import numpy as np
        >>> func = chip_match.ChipMatch2.show_single_namematch
        >>> sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
        >>> setup = ut.regex_replace('viz_matches.show_name_matches', '#', sourcecode)
        >>> homog = False
        >>> print(ut.indent(setup, '>>> '))
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1])
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> dnid = ibs.get_annot_nids(cm.qaid)
        >>> # +--- COPIED SECTION
        >>> from ibeis.viz import viz_matches
        >>> qaid = cm.qaid
        >>> # <GET NAME GROUPXS>
        >>> nidx = cm.nid2_nidx[dnid]
        >>> groupxs = cm.name_groupxs[nidx]
        >>> daids = np.take(cm.daid_list, groupxs)
        >>> groupxs = groupxs.compress(daids != cm.qaid)
        >>> # </GET NAME GROUPXS>
        >>> # sort annots in this name by the chip score
        >>> group_sortx = cm.csum_score_list.take(groupxs).argsort()[::-1]
        >>> sorted_groupxs = groupxs.take(group_sortx)
        >>> # get the info for this name
        >>> name_fm_list  = ut.list_take(cm.fm_list, sorted_groupxs)
        >>> REMOVE_EMPTY_MATCHES = len(sorted_groupxs) > 3
        >>> if REMOVE_EMPTY_MATCHES:
        >>>     isvalid_list = [len(fm) > 0 for fm in name_fm_list]
        >>>     name_fm_list = ut.list_compress(name_fm_list, isvalid_list)
        >>>     sorted_groupxs = sorted_groupxs.compress(isvalid_list)
        >>> name_H1_list   = None if not homog or cm.H_list is None else ut.list_take(cm.H_list, sorted_groupxs)
        >>> name_fsv_list  = None if cm.fsv_list is None else ut.list_take(cm.fsv_list, sorted_groupxs)
        >>> name_fs_list   = None if name_fsv_list is None else [fsv.prod(axis=1) for fsv in name_fsv_list]
        >>> name_daid_list = ut.list_take(cm.daid_list, sorted_groupxs)
        >>> # find features marked as invalid by name scoring
        >>> featflag_list  = name_scoring.get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_=qreq_)
        >>> name_featflag_list = ut.list_take(featflag_list, sorted_groupxs)
        >>> # Get the scores for names and chips
        >>> name_score = cm.name_score_list[nidx]
        >>> name_rank = ut.listfind(cm.name_score_list.argsort()[::-1].tolist(), nidx)
        >>> name_annot_scores = cm.csum_score_list.take(sorted_groupxs)
        >>> # L___ COPIED SECTION
        >>> kwargs = {}
        >>> show_name_matches(ibs, qaid, name_daid_list, name_fm_list, name_fs_list, name_H1_list, name_featflag_list, qreq_=qreq_, **kwargs)
        >>> ut.quit_if_noshow()
        >>> ut.show_if_requested()
    """
    import numpy as np
    from ibeis import constants as const
    draw_fmatches = kwargs.get('draw_fmatches', True)
    rchip1, kpts1 = get_query_annot_pair_info(ibs, qaid, qreq_, draw_fmatches)
    rchip2_list, kpts2_list = get_data_annot_pair_info(ibs, name_daid_list, qreq_, draw_fmatches)
    fm_list = name_fm_list
    fs_list = name_fs_list
    featflag_list = name_featflag_list
    offset_list, sf_list, bbox_list = show_multichip_match(rchip1, rchip2_list, kpts1, kpts2_list, fm_list, fs_list, featflag_list, **kwargs)
    aid_list = [qaid] + name_daid_list
    annotate_matches3(ibs, aid_list, bbox_list, offset_list, qreq_=None, **kwargs)
    ax = pt.gca()
    title = vh.get_query_text(ibs, None, name_daid_list, False, qaid=qaid, **kwargs)

    pt.set_title(title, ax)
    name_equality = ibs.get_annot_nids(qaid) == np.array(ibs.get_annot_nids(name_daid_list))
    truth = 1 if np.all(name_equality) else (2 if np.any(name_equality) else 0)
    if any(ibs.is_aid_unknown(name_daid_list)) or ibs.is_aid_unknown(qaid):
        truth = const.TRUTH_UNKNOWN
    truth_color = vh.get_truth_color(truth)
    pt.draw_border(ax, color=truth_color, lw=4)

    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey([qaid] * len(name_daid_list), name_daid_list)
    annotmatch_rowid_list = ut.filter_Nones(annotmatch_rowid_list)
    # Case tags
    tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowid_list)
    tag_list = ut.unique_keep_order2(ut.flatten(tags_list))

    name_rank = kwargs.get('name_rank', None)
    if name_rank is None:
        xlabel = {1: 'Genuine', 0: 'Imposter', 2: 'Unknown'}[truth]
        #xlabel = {1: 'True', 0: 'False', 2: 'Unknown'}[truth]
    else:
        if name_rank == 0:
            xlabel = {1: 'True Positive', 0: 'False Positive', 2: 'Unknown'}[truth]
        else:
            xlabel = {1: 'False Negative', 0: 'True Negative', 2: 'Unknown'}[truth]
    #xlabel_list = []
    #if any(ibs.get_annotmatch_is_photobomb(annotmatch_rowid_list)):
    #    xlabel_list += [' Photobomb']
    #if any(ibs.get_annotmatch_is_scenerymatch(annotmatch_rowid_list)):
    #    xlabel_list += [' Scenery']
    #if any(ibs.get_annotmatch_is_nondistinct(annotmatch_rowid_list)):
    #    xlabel_list += [' Nondistinct']
    #if any(ibs.get_annotmatch_is_hard(annotmatch_rowid_list)):
    #    xlabel_list += [' Hard']
    if len(tag_list) > 0:
        xlabel += '\n' + ', '.join(tag_list)

    ax.set_xlabel(xlabel)


def show_multichip_match(rchip1, rchip2_list, kpts1, kpts2_list, fm_list, fs_list, featflag_list, fnum=None, pnum=None, **kwargs):
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

    num = 0 if len(rchip2_list) < 3 else 1
    vert = True if len(rchip2_list) > 1 else False

    match_img, offset_list, sf_list = pt.stack_image_list_special(rchip1_, rchip2_list_, num=num, vert=vert)

    wh_list = np.array(ut.flatten([[wh1], wh2_list])) * sf_list

    offset1 = offset_list[0]
    wh1 = wh_list[0]
    sf1 = sf_list[0]

    fig, ax = pt.imshow(match_img, fnum=fnum, pnum=pnum)

    if kwargs.get('show_matches', True):
        NONVOTE_MODE = kwargs.get('nonvote_mode', 'filter')
        ut.flatten(fs_list)
        #ut.embed()
        flat_fs, cumlen_list = ut.invertible_flatten2(fs_list)
        flat_colors = pt.scores_to_color(np.array(flat_fs), 'hot')
        colors_list = ut.unflatten2(flat_colors, cumlen_list)
        for _tup in zip(offset_list[1:], wh_list[1:], sf_list[1:], kpts2_list, fm_list, fs_list, featflag_list, colors_list):
            offset2, wh2, sf2, kpts2, fm2_, fs2_, featflags, colors = _tup
            xywh1 = (offset1[0], offset1[1], wh1[0], wh1[1])
            xywh2 = (offset2[0], offset2[1], wh2[0], wh2[1])
            #colors = pt.scores_to_color(fs2)
            if kpts1 is not None and kpts2 is not None:
                if NONVOTE_MODE == 'filter':
                    fm2 = fm2_.compress(featflags, axis=0)
                    fs2 = fs2_.compress(featflags, axis=0)
                elif NONVOTE_MODE == 'only':
                    fm2 = fm2_.compress(np.logical_not(featflags), axis=0)
                    fs2 = fs2_.compress(np.logical_not(featflags), axis=0)
                else:
                    # TODO: optional coloring of nonvotes instead
                    fm2 = fm2_
                    fs2 = fs2_
                pt.plot_fmatch(xywh1, xywh2, kpts1, kpts2, fm2, fs2, fm_norm=None,
                               H1=None, H2=None, scale_factor1=sf1,
                               scale_factor2=sf2, colorbar_=False, colors=colors,
                               **kwargs)
        pt.colorbar(flat_fs, flat_colors)
    bbox_list = [(x, y, w, h) for (x, y), (w, h) in zip(offset_list, wh_list)]
    return offset_list, sf_list, bbox_list

    # Show the stacked chips
    #annotate_matches2(ibs, aid1, aid2, fm, fs, xywh2=xywh2, xywh1=xywh1,
    #                  offset1=offset1, offset2=offset2, **kwargs)


def annotate_matches3(ibs, aid_list, bbox_list, offset_list, qreq_=None, **kwargs):
    """
    TODO: use this as the main function.
    Have the qres version be a wrapper
    """
    # TODO Use this function when you clean show_matches
    in_image    = kwargs.get('in_image', False)
    #show_query  = kwargs.get('show_query', True)
    #draw_border = kwargs.get('draw_border', True)
    draw_lbl    = kwargs.get('draw_lbl', True)
    # List of annotation scores for each annot in the name
    name_annot_scores = kwargs.get('name_annot_scores', None)

    #printDBG('[viz] annotate_matches2()')
    #truth = ibs.get_match_truth(aid1, aid2)
    #truth_color = vh.get_truth_color(truth)
    # Build title

    #score         = kwargs.pop('score', None)
    #rawscore      = kwargs.pop('rawscore', None)
    #aid2_raw_rank = kwargs.pop('aid2_raw_rank', None)
    #print(kwargs)
    #title = vh.get_query_text(ibs, None, aid2, truth, qaid=aid1, **kwargs)
    # Build xlbl
    ax = pt.gca()
    ph.set_plotdat(ax, 'viztype', 'multi_match')
    ph.set_plotdat(ax, 'qaid', aid_list[0])
    ph.set_plotdat(ax, 'num_matches', len(aid_list) - 1)
    ph.set_plotdat(ax, 'aid_list', aid_list[1:])
    for count, aid in enumerate(aid_list, start=1):
        ph.set_plotdat(ax, 'aid%d' % (count,), aid)

    if draw_lbl:
        # Build labels
        nid_list = ibs.get_annot_nids(aid_list, distinguish_unknowns=False)
        name_list = ibs.get_annot_names(aid_list)
        lbls_list = [[] for _ in range(len(aid_list))]
        if kwargs.get('show_aid', True):
            for count, (lbls, aid) in enumerate(zip(lbls_list, aid_list)):
                lbls.append(('q' if count == 0 else '') + vh.get_aidstrs(aid))
        if kwargs.get('show_name', False):
            for (lbls, name) in zip(lbls_list, name_list):
                lbls.append(repr(str(name)))
        if kwargs.get('show_nid', True):
            for count, (lbls, nid) in enumerate(zip(lbls_list, nid_list)):
                # only label the first two images with nids
                LABEL_ALL_NIDS = False
                if count <= 1 or LABEL_ALL_NIDS:
                    lbls.append(vh.get_nidstrs(nid))
        if kwargs.get('show_annot_score', True) and name_annot_scores is not None:
            for (lbls, score) in zip(lbls_list[1:], name_annot_scores):
                lbls.append(ut.num_fmt(score))
        lbl_list = [' : '.join(lbls) for lbls in lbls_list]
    else:
        lbl_list = [None] * len(aid_list)
    #pt.set_title(title, ax)
    # Plot annotations over images
    if in_image:
        in_image_bbox_list = vh.get_bboxes(ibs, aid_list, offset_list)
        in_image_theta_list = ibs.get_annot_thetas(aid_list)
        # HACK!
        #if show_query:
        #    pt.draw_bbox(bbox1, bbox_color=pt.ORANGE, lbl=lbl1, theta=theta1)
        bbox_color = pt.ORANGE
        #bbox_color2 = truth_color if draw_border else pt.ORANGE
        for bbox, theta, lbl in zip(in_image_bbox_list, in_image_theta_list, lbl_list):
            pt.draw_bbox(bbox, bbox_color=bbox_color, lbl=lbl, theta=theta)
            pass
    else:
        xy, w, h = pt.get_axis_xy_width_height(ax)
        #bbox2 = (xy[0], xy[1], w, h)
        #theta2 = 0

        #if xywh2 is None:
        #    #xywh2 = (xy[0], xy[1], w, h)
        #    # weird when sidebyside is off y seems to be inverted
        #    xywh2 = (0,  0, w, h)

        #if not show_query and xywh1 is None:
        #    data_config2 = None if qreq_ is None else qreq_.get_external_data_config2()
        #    kpts2 = ibs.get_annot_kpts([aid2], config2_=data_config2)[0]
        #    #pt.draw_kpts2(kpts2.take(fm.T[1], axis=0))
        #    # Draw any selected matches
        #    #sm_kw = dict(rect=True, colors=pt.BLUE)
        #    pt.plot_fmatch(None, xywh2, None, kpts2, fm, fs=fs, **kwargs)
        #if draw_border:
        #    pt.draw_border(ax, truth_color, 4, offset=offset2)
        if draw_lbl:
            # Custom user lbl for chips 1 and 2
            #if show_query:
            #    (x1, y1, w1, h1) = xywh1
            #    pt.absolute_lbl(x1 + w1, y1, lbl1)
            for bbox, lbl in zip(bbox_list, lbl_list):
                (x, y, w, h) = bbox
                pt.absolute_lbl(x + w, y, lbl)
        # No matches draw a red box
    #if fm is None or len(fm) == 0:
    #    if draw_border:
    #        pt.draw_boxedX(bbox2, theta=theta2)


#@ut.indent_func
def show_matches2(ibs, aid1, aid2, fm=None, fs=None, fm_norm=None, sel_fm=[],
                  H1=None, H2=None, qreq_=None, **kwargs):
    """
    TODO: use this as the main function.
    Have the qres version be a wrapper
    Integrate ChipMatch2

    Example:
        >>> # DISABLE_DOCTEST
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
        ax, xywh1, xywh2 = pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm,
                                               fs=fs, fm_norm=fm_norm,
                                               H1=H1, H2=H2, lbl1=lbl1,
                                              lbl2=lbl2, sel_fm=sel_fm,
                                              **kwargs)
    except Exception as ex:
        ut.printex(ex, 'consider qr.remove_corrupted_queries',
                      '[viz_matches]')
        print('')
        raise
    # Moved the code into show_chipmatch
    #if len(sel_fm) > 0:
    #    # Draw any selected matches
    #    sm_kw = dict(rect=True, colors=pt.BLUE)
    #    pt.plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    (x1, y1, w1, h1) = xywh1
    (x2, y2, w2, h2) = xywh2
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
    notitle     = kwargs.get('notitle', False)

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
    ax = pt.gca()
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
    if not notitle:
        pt.set_title(title, ax)
    # Plot annotations over images
    if in_image:
        bbox1, bbox2 = vh.get_bboxes(ibs, [aid1, aid2], [offset1, offset2])
        theta1, theta2 = ibs.get_annot_thetas([aid1, aid2])
        # HACK!
        if show_query:
            pt.draw_bbox(bbox1, bbox_color=pt.ORANGE, lbl=lbl1, theta=theta1)
        bbox_color2 = truth_color if draw_border else pt.ORANGE
        pt.draw_bbox(bbox2, bbox_color=bbox_color2, lbl=lbl2, theta=theta2)
    else:
        xy, w, h = pt.get_axis_xy_width_height(ax)
        bbox2 = (xy[0], xy[1], w, h)
        theta2 = 0

        if xywh2 is None:
            #xywh2 = (xy[0], xy[1], w, h)
            # weird when sidebyside is off y seems to be inverted
            xywh2 = (0,  0, w, h)

        if not show_query and xywh1 is None:
            data_config2 = None if qreq_ is None else qreq_.get_external_data_config2()
            kpts2 = ibs.get_annot_kpts([aid2], config2_=data_config2)[0]
            #pt.draw_kpts2(kpts2.take(fm.T[1], axis=0))
            # Draw any selected matches
            #sm_kw = dict(rect=True, colors=pt.BLUE)
            pt.plot_fmatch(None, xywh2, None, kpts2, fm, fs=fs, **kwargs)
        if draw_border:
            pt.draw_border(ax, truth_color, 4, offset=offset2)
        if draw_lbl:
            # Custom user lbl for chips 1 and 2
            if show_query:
                (x1, y1, w1, h1) = xywh1
                pt.absolute_lbl(x1 + w1, y1, lbl1)
            (x2, y2, w2, h2) = xywh2
            pt.absolute_lbl(x2 + w2, y2, lbl2)
        # No matches draw a red box
    if fm is None or len(fm) == 0:
        if draw_border:
            pt.draw_boxedX(bbox2, theta=theta2)


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
        >>>    execstr = pt.present()
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
        ax, xywh1, xywh2 = pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2,
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
        sm_kw = dict(rect=True, colors=pt.BLUE)
        pt.plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
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
    #ax = pt.gca()
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
    #pt.set_title(title, ax)
    ## Plot annotations over images
    #if in_image:
    #    bbox1, bbox2 = vh.get_bboxes(ibs, [aid1, aid2], [offset1, offset2])
    #    theta1, theta2 = ibs.get_annot_thetas([aid1, aid2])
    #    # HACK!
    #    if show_query:
    #        pt.draw_bbox(bbox1, bbox_color=pt.ORANGE, lbl=lbl1, theta=theta1)
    #    bbox_color2 = truth_color if draw_border else pt.ORANGE
    #    pt.draw_bbox(bbox2, bbox_color=bbox_color2, lbl=lbl2, theta=theta2)
    #else:
    #    xy, w, h = pt.get_axis_xy_width_height(ax)
    #    bbox2 = (xy[0], xy[1], w, h)
    #    theta2 = 0

    #    if xywh2 is None:
    #        #xywh2 = (xy[0], xy[1], w, h)
    #        # weird when sidebyside is off y seems to be inverted
    #        xywh2 = (0,  0, w, h)

    #    if not show_query and xywh1 is None:
    #        data_config2 = None if qreq_ is None else qreq_.get_external_data_config2()
    #        kpts2 = ibs.get_annot_kpts([aid2], config2_=data_config2)[0]
    #        #pt.draw_kpts2(kpts2.take(fm.T[1], axis=0))
    #        # Draw any selected matches
    #        #sm_kw = dict(rect=True, colors=pt.BLUE)
    #        pt.plot_fmatch(None, xywh2, None, kpts2, fm, fs=fs, **kwargs)
    #    if draw_border:
    #        pt.draw_border(ax, truth_color, 4, offset=offset2)
    #    if draw_lbl:
    #        # Custom user lbl for chips 1 and 2
    #        if show_query:
    #            (x1, y1, w1, h1) = xywh1
    #            pt.absolute_lbl(x1 + w1, y1, lbl1)
    #        (x2, y2, w2, h2) = xywh2
    #        pt.absolute_lbl(x2 + w2, y2, lbl2)
    #    # No matches draw a red box
    #if fm is None or len(fm) == 0:
    #    if draw_border:
    #        pt.draw_boxedX(bbox2, theta=theta2)


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
