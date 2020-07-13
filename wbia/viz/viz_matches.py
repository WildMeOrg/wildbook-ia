# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import wbia.plottool as pt
import wbia.plottool.plot_helpers as ph
from wbia.viz import viz_helpers as vh

(print, rrr, profile) = ut.inject2(__name__)


def _get_annot_pair_info(ibs, aid1, aid2, qreq_, draw_fmatches, **kwargs):
    kpts1 = kwargs.get('kpts1', None)
    kpts2 = kwargs.get('kpts2', None)
    as_fpath = kwargs.get('as_fpath', False)
    kpts2_list = None if kpts2 is None else [kpts2]
    rchip1, kpts1 = get_query_annot_pair_info(
        ibs, aid1, qreq_, draw_fmatches, kpts1=kpts1, as_fpath=as_fpath
    )
    annot2_data_list = get_data_annot_pair_info(
        ibs, [aid2], qreq_, draw_fmatches, as_fpath=as_fpath, kpts2_list=kpts2_list
    )
    rchip2, kpts2 = ut.get_list_column(annot2_data_list, 0)
    return rchip1, rchip2, kpts1, kpts2


def get_query_annot_pair_info(
    ibs, qaid, qreq_, draw_fmatches, kpts1=None, as_fpath=False
):
    # print('!!! qqreq_ = %r' % (qreq_,))
    query_config2_ = None if qreq_ is None else qreq_.extern_query_config2
    tblhack = getattr(qreq_, 'tablename', None)
    # print('!!! query_config2_ = %r' % (query_config2_,))
    if (
        not tblhack
        or tblhack.lower()
        in [
            'bc_dtw',
            'oc_wdtw',
            'curvrankdorsal',
            'curvrankfinfindrhybriddorsal',
            'curvrankfluke',
            'deepsense',
            'finfindr',
            'kaggle7',
            'kaggleseven',
        ]
    ) and getattr(qreq_, '_isnewreq', None):
        if (
            hasattr(qreq_, 'get_fmatch_overlayed_chip')
            and draw_fmatches
            and draw_fmatches != 'hackoff'
        ):
            rchip1 = qreq_.get_fmatch_overlayed_chip(qaid, config=query_config2_)
            draw_fmatches = False
        else:
            rchip1 = ibs.depc_annot.get_property(
                'chips', qaid, 'img', config=query_config2_
            )
            draw_fmatches = False
    else:
        rchip1 = vh.get_chips(ibs, [qaid], config2_=query_config2_, as_fpath=as_fpath)[0]
    if draw_fmatches:
        if kpts1 is None:
            kpts1 = vh.get_kpts(ibs, [qaid], config2_=query_config2_)[0]
    else:
        kpts1 = None
    return rchip1, kpts1


def get_data_annot_pair_info(
    ibs, aid_list, qreq_, draw_fmatches, scale_down=False, kpts2_list=None, as_fpath=False
):
    data_config2_ = None if qreq_ is None else qreq_.extern_data_config2
    # print('!!! data_config2_ = %r' % (data_config2_,))
    # print('!!! dqreq_ = %r' % (qreq_,))
    tblhack = getattr(qreq_, 'tablename', None)
    if (
        not tblhack
        or tblhack.lower()
        in [
            'bc_dtw',
            'oc_wdtw',
            'curvrankdorsal',
            'curvrankfinfindrhybriddorsal',
            'curvrankfluke',
            'deepsense',
            'finfindr',
            'kaggle7',
            'kaggleseven',
        ]
    ) and getattr(qreq_, '_isnewreq', None):
        if (
            hasattr(qreq_, 'get_fmatch_overlayed_chip')
            and draw_fmatches
            and draw_fmatches != 'hackoff'
        ):
            rchip2_list = qreq_.get_fmatch_overlayed_chip(aid_list, config=data_config2_)
            # rchip2_list = ibs.depc_annot.get_property('chips', aid_list, 'img', config=data_config2_)
            draw_fmatches = False
        else:
            rchip2_list = ibs.depc_annot.get_property(
                'chips', aid_list, 'img', config=data_config2_
            )
            draw_fmatches = False
        # vh.get_chips(ibs, aid_list, config2_=data_config2_)
    else:
        rchip2_list = vh.get_chips(
            ibs, aid_list, config2_=data_config2_, as_fpath=as_fpath
        )
    if draw_fmatches:
        if kpts2_list is None:
            kpts2_list = vh.get_kpts(ibs, aid_list, config2_=data_config2_)
    else:
        kpts2_list = [None] * len(aid_list)
    if scale_down:
        pass
    return rchip2_list, kpts2_list


# @ut.tracefunc_xml
def show_name_matches(
    ibs,
    qaid,
    name_daid_list,
    name_fm_list,
    name_fs_list,
    name_H1_list,
    name_featflag_list,
    qreq_=None,
    **kwargs
):
    """
    Called from chip_match.py

    Args:
        ibs (IBEISController):  wbia controller object
        qaid (int):  query annotation id
        name_daid_list (list):
        name_fm_list (list):
        name_fs_list (list):
        name_H1_list (list):
        name_featflag_list (list):
        qreq_ (QueryRequest):  query request object with hyper-parameters(default = None)

    Kwargs:
        draw_fmatches, name_rank, fnum, pnum, colorbar_, nonvote_mode,
        fastmode, show_matches, fs, fm_norm, lbl1, lbl2, rect, draw_border,
        cmap, H1, H2, scale_factor1, scale_factor2, draw_pts, draw_ell,
        draw_lines, show_nMatches, all_kpts, in_image, show_query, draw_lbl,
        name_annot_scores, score, rawscore, aid2_raw_rank, show_name,
        show_nid, show_aid, show_annot_score, show_truth, name_score,
        show_name_score, show_name_rank, show_timedelta

    CommandLine:
        python -m wbia.viz.viz_matches --exec-show_name_matches
        python -m wbia.viz.viz_matches --test-show_name_matches --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_matches import *  # NOQA
        >>> from wbia.algo.hots import chip_match
        >>> from wbia.algo.hots import name_scoring
        >>> import vtool as vt
        >>> from wbia.algo.hots import _pipeline_helpers as plh  # NOQA
        >>> import numpy as np
        >>> func = chip_match.ChipMatch.show_single_namematch
        >>> sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True,
        >>>                                     strip_docstr=True)
        >>> setup = ut.regex_replace('viz_matches.show_name_matches', '#', sourcecode)
        >>> homog = False
        >>> print(ut.indent(setup, '>>> '))
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1])
        >>> cm = cm_list[0]
        >>> cm.score_name_nsum(qreq_)
        >>> dnid = ibs.get_annot_nids(cm.qaid)
        >>> # +--- COPIED SECTION
        >>> locals_ = locals()
        >>> var_list = ut.exec_func_src(
        >>>     func, locals_=locals_,
        >>>     sentinal='name_annot_scores = cm.annot_score_list.take(sorted_groupxs')
        >>> exec(ut.execstr_dict(var_list))
        >>> # L___ COPIED SECTION
        >>> kwargs = {}
        >>> show_name_matches(ibs, qaid, name_daid_list, name_fm_list,
        >>>                   name_fs_list, name_h1_list, name_featflag_list,
        >>>                   qreq_=qreq_, **kwargs)
        >>> ut.quit_if_noshow()
        >>> ut.show_if_requested()
    """
    # print("SHOW NAME MATCHES")
    # print(ut.repr2(kwargs, nl=True))
    # from wbia import constants as const
    from wbia import tag_funcs

    draw_fmatches = kwargs.pop('draw_fmatches', True)
    rchip1, kpts1 = get_query_annot_pair_info(ibs, qaid, qreq_, draw_fmatches)
    rchip2_list, kpts2_list = get_data_annot_pair_info(
        ibs, name_daid_list, qreq_, draw_fmatches
    )

    heatmask = kwargs.pop('heatmask', False)
    if heatmask:
        from vtool.coverage_kpts import make_kpts_heatmask
        import numpy as np
        import vtool as vt

        wh1 = vt.get_size(rchip1)
        fx1 = np.unique(np.hstack([fm.T[0] for fm in name_fm_list]))
        heatmask1 = make_kpts_heatmask(kpts1[fx1], wh1)
        rchip1 = vt.overlay_alpha_images(heatmask1, rchip1)
        # Hack cast back to uint8
        rchip1 = (rchip1 * 255).astype(np.uint8)

        rchip2_list_ = rchip2_list
        rchip2_list = []

        for rchip2, kpts2, fm in zip(rchip2_list_, kpts2_list, name_fm_list):
            fx2 = fm.T[1]
            wh2 = vt.get_size(rchip2)
            heatmask2 = make_kpts_heatmask(kpts2[fx2], wh2)
            rchip2 = vt.overlay_alpha_images(heatmask2, rchip2)
            # Hack cast back to uint8
            rchip2 = (rchip2 * 255).astype(np.uint8)
            rchip2_list.append(rchip2)
    #
    fm_list = name_fm_list
    fs_list = name_fs_list
    featflag_list = name_featflag_list
    offset_list, sf_list, bbox_list = show_multichip_match(
        rchip1, rchip2_list, kpts1, kpts2_list, fm_list, fs_list, featflag_list, **kwargs
    )
    aid_list = [qaid] + name_daid_list
    annotate_matches3(
        ibs,
        aid_list,
        bbox_list,
        offset_list,
        name_fm_list,
        name_fs_list,
        qreq_=None,
        **kwargs
    )
    ax = pt.gca()
    title = vh.get_query_text(ibs, None, name_daid_list, False, qaid=qaid, **kwargs)

    pt.set_title(title, ax)

    # Case tags
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(
        [qaid] * len(name_daid_list), name_daid_list
    )
    annotmatch_rowid_list = ut.filter_Nones(annotmatch_rowid_list)
    tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowid_list)
    if not ut.get_argflag('--show'):  # False:
        tags_list = tag_funcs.consolodate_annotmatch_tags(tags_list)
    tag_list = ut.unique_ordered(ut.flatten(tags_list))

    name_rank = kwargs.get('name_rank', None)
    truth = get_multitruth(ibs, aid_list)

    xlabel = {1: 'Correct ID', 0: 'Incorrect ID', 2: 'Unknown ID'}[truth]

    if False:
        if name_rank is None:
            xlabel = {1: 'Genuine', 0: 'Imposter', 2: 'Unknown'}[truth]
            # xlabel = {1: 'True', 0: 'False', 2: 'Unknown'}[truth]
        else:
            if name_rank == 0:
                xlabel = {1: 'True Positive', 0: 'False Positive', 2: 'Unknown'}[truth]
            else:
                xlabel = {1: 'False Negative', 0: 'True Negative', 2: 'Unknown'}[truth]

    if len(tag_list) > 0:
        xlabel += '\n' + ', '.join(tag_list)

    noshow_truth = ut.get_argflag('--noshow_truth')
    if not noshow_truth:
        pt.set_xlabel(xlabel)
    return ax


def get_multitruth(ibs, aid_list):
    import numpy as np

    if ibs.is_aid_unknown(aid_list[0]):
        return 2
    name_equality = ibs.get_annot_nids(aid_list[0]) == np.array(
        ibs.get_annot_nids(aid_list[1:])
    )
    truth = 1 if np.all(name_equality) else (2 if np.any(name_equality) else 0)
    return truth


def annotate_matches3(
    ibs,
    aid_list,
    bbox_list,
    offset_list,
    name_fm_list,
    name_fs_list,
    qreq_=None,
    **kwargs
):
    """
    TODO: use this as the main function.
    """
    # TODO Use this function when you clean show_matches
    in_image = kwargs.get('in_image', False)
    # show_query  = kwargs.get('show_query', True)
    draw_border = kwargs.get('draw_border', True)
    draw_lbl = kwargs.get('draw_lbl', True)
    notitle = kwargs.get('notitle', False)
    # List of annotation scores for each annot in the name

    # printDBG('[viz] annotate_matches3()')
    # truth = ibs.get_match_truth(aid1, aid2)

    # name_equality = (
    #    np.array(ibs.get_annot_nids(aid_list[1:])) == ibs.get_annot_nids(aid_list[0])
    # ).tolist()
    # truth = 1 if all(name_equality) else (2 if any(name_equality) else 0)
    # truth_color = vh.get_truth_color(truth)
    # # Build title

    # score         = kwargs.pop('score', None)
    # rawscore      = kwargs.pop('rawscore', None)
    # aid2_raw_rank = kwargs.pop('aid2_raw_rank', None)
    # print(kwargs)
    # title = vh.get_query_text(ibs, None, aid2, truth, qaid=aid1, **kwargs)
    # Build xlbl
    ax = pt.gca()
    ph.set_plotdat(ax, 'viztype', 'multi_match')
    ph.set_plotdat(ax, 'qaid', aid_list[0])
    ph.set_plotdat(ax, 'num_matches', len(aid_list) - 1)
    ph.set_plotdat(ax, 'aid_list', aid_list[1:])
    for count, aid in enumerate(aid_list, start=1):
        ph.set_plotdat(ax, 'aid%d' % (count,), aid)

    # name_equality = (ibs.get_annot_nids(aid_list[0]) ==
    #                 np.array(ibs.get_annot_nids(aid_list[1:])))
    # truth = 1 if np.all(name_equality) else (2 if np.any(name_equality) else 0)
    truth = get_multitruth(ibs, aid_list)
    if any(ibs.is_aid_unknown(aid_list[1:])) or ibs.is_aid_unknown(aid_list[0]):
        truth = ibs.const.EVIDENCE_DECISION.UNKNOWN
    truth_color = vh.get_truth_color(truth)

    name_annot_scores = kwargs.get('name_annot_scores', None)
    if len(aid_list) == 2:
        # HACK; generalize to multple annots
        title = vh.get_query_text(
            ibs, None, aid_list[1], truth, qaid=aid_list[0], **kwargs
        )
        if not notitle:
            pt.set_title(title, ax)

    if draw_lbl:
        # Build labels
        nid_list = ibs.get_annot_nids(aid_list, distinguish_unknowns=False)
        name_list = ibs.get_annot_names(aid_list)
        lbls_list = [[] for _ in range(len(aid_list))]
        if kwargs.get('show_name', False):
            for count, (lbls, name) in enumerate(zip(lbls_list, name_list)):
                lbls.append(ut.repr2((name)))
        if kwargs.get('show_nid', True):
            for count, (lbls, nid) in enumerate(zip(lbls_list, nid_list)):
                # only label the first two images with nids
                LABEL_ALL_NIDS = False
                if count <= 1 or LABEL_ALL_NIDS:
                    # lbls.append(vh.get_nidstrs(nid))
                    lbls.append(('q' if count == 0 else '') + vh.get_nidstrs(nid))
        if kwargs.get('show_aid', True):
            for count, (lbls, aid) in enumerate(zip(lbls_list, aid_list)):
                lbls.append(('q' if count == 0 else '') + vh.get_aidstrs(aid))
        if kwargs.get('show_annot_score', True) and name_annot_scores is not None:
            max_digits = kwargs.get('score_precision', None)
            for (lbls, score) in zip(lbls_list[1:], name_annot_scores):
                lbls.append(ut.num_fmt(score, max_digits=max_digits))
        lbl_list = [' : '.join(lbls) for lbls in lbls_list]
    else:
        lbl_list = [None] * len(aid_list)
    # Plot annotations over images
    if in_image:
        in_image_bbox_list = vh.get_bboxes(ibs, aid_list, offset_list)
        in_image_theta_list = ibs.get_annot_thetas(aid_list)
        # HACK!
        # if show_query:
        #    pt.draw_bbox(bbox1, bbox_color=pt.ORANGE, lbl=lbl1, theta=theta1)
        bbox_color = pt.ORANGE
        bbox_color = truth_color if draw_border else pt.ORANGE
        for bbox, theta, lbl in zip(in_image_bbox_list, in_image_theta_list, lbl_list):
            pt.draw_bbox(bbox, bbox_color=bbox_color, lbl=lbl, theta=theta)
            pass
    else:
        xy, w, h = pt.get_axis_xy_width_height(ax)
        if draw_border:
            pt.draw_border(ax, color=truth_color, lw=4)
        if draw_lbl:
            # Custom user lbl for chips 1 and 2
            for bbox, lbl in zip(bbox_list, lbl_list):
                (x, y, w, h) = bbox
                pt.absolute_lbl(x + w, y, lbl)
    # No matches draw a red box
    if True:
        no_matches = name_fm_list is None or all(
            [True if fm is None else len(fm) == 0 for fm in name_fm_list]
        )
        if no_matches:
            xy, w, h = pt.get_axis_xy_width_height(ax)
            # axes_bbox = (xy[0], xy[1], w, h)
            if draw_border:
                pass
                # pt.draw_boxedX(axes_bbox, theta=0)


def annotate_matches2(
    ibs,
    aid1,
    aid2,
    fm,
    fs,
    offset1=(0, 0),
    offset2=(0, 0),
    xywh2=None,  # (0, 0, 0, 0),
    xywh1=None,  # (0, 0, 0, 0),
    qreq_=None,
    **kwargs
):
    """
    TODO: use this as the main function.
    """
    if True:
        aid_list = [aid1, aid2]
        bbox_list = [xywh1, xywh2]
        offset_list = [offset1, offset2]
        name_fm_list = [fm]
        name_fs_list = [fs]
        return annotate_matches3(
            ibs,
            aid_list,
            bbox_list,
            offset_list,
            name_fm_list,
            name_fs_list,
            qreq_=qreq_,
            **kwargs
        )
    else:
        # TODO: make sure all of this functionality is incorporated into annotate_matches3
        in_image = kwargs.get('in_image', False)
        show_query = kwargs.get('show_query', True)
        draw_border = kwargs.get('draw_border', True)
        draw_lbl = kwargs.get('draw_lbl', True)
        notitle = kwargs.get('notitle', False)

        truth = ibs.get_match_truth(aid1, aid2)
        truth_color = vh.get_truth_color(truth)
        # Build title
        title = vh.get_query_text(ibs, None, aid2, truth, qaid=aid1, **kwargs)
        # Build xlbl
        ax = pt.gca()
        ph.set_plotdat(ax, 'viztype', 'matches')
        ph.set_plotdat(ax, 'qaid', aid1)
        ph.set_plotdat(ax, 'aid1', aid1)
        ph.set_plotdat(ax, 'aid2', aid2)
        if draw_lbl:
            name1, name2 = ibs.get_annot_names([aid1, aid2])
            nid1, nid2 = ibs.get_annot_name_rowids(
                [aid1, aid2], distinguish_unknowns=False
            )
            # lbl1 = repr(name1)  + ' : ' + 'q' + vh.get_aidstrs(aid1)
            # lbl2 = repr(name2)  + ' : ' +  vh.get_aidstrs(aid2)
            lbl1_list = []
            lbl2_list = []
            if kwargs.get('show_aid', True):
                lbl1_list.append('q' + vh.get_aidstrs(aid1))
                lbl2_list.append(vh.get_aidstrs(aid2))
            if kwargs.get('show_name', True):
                lbl1_list.append(repr((name1)))
                lbl2_list.append(repr((name2)))
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
                # xywh2 = (xy[0], xy[1], w, h)
                # weird when sidebyside is off y seems to be inverted
                xywh2 = (0, 0, w, h)

            if not show_query and xywh1 is None:
                data_config2 = None if qreq_ is None else qreq_.extern_data_config2
                # FIXME, pass data in
                kpts2 = ibs.get_annot_kpts([aid2], config2_=data_config2)[0]
                # pt.draw_kpts2(kpts2.take(fm.T[1], axis=0))
                # Draw any selected matches
                # sm_kw = dict(rect=True, colors=pt.BLUE)
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
        if True:
            # No matches draw a red box
            if fm is None or len(fm) == 0:
                if draw_border:
                    pass
                    # pt.draw_boxedX(bbox2, theta=theta2)


# @ut.indent_func
# @ut.tracefunc_xml
def show_matches2(
    ibs,
    aid1,
    aid2,
    fm=None,
    fs=None,
    fm_norm=None,
    sel_fm=[],
    H1=None,
    H2=None,
    qreq_=None,
    **kwargs
):
    """
    TODO: DEPRICATE and use special case of show_name_matches
    Integrate ChipMatch

    Used in:
        Found 1 line(s) in '/home/joncrall/code/wbia_cnn/wbia_cnn/ingest_wbia.py':
        ingest_wbia.py : 827 |        >>>     wbia.viz.viz_matches.show_matches2(ibs, aid1, aid2, fm=None, kpts1=kpts1, kpts2=kpts2)
        ----------------------
        Found 4 line(s) in '/home/joncrall/code/wbia/wbia/viz/viz_matches.py':
        viz_matches.py : 423 |def show_matches2(ibs, aid1, aid2, fm=None, fs=None, fm_norm=None, sel_fm=[],
        viz_matches.py : 430 |        python -m wbia.viz.viz_matches --exec-show_matches2 --show
        viz_matches.py : 431 |        python -m wbia --tf ChipMatch.ishow_single_annotmatch show_matches2 --show
        viz_matches.py : 515 |    return show_matches2(ibs, aid1, aid2, fm, fs, qreq_=qreq_, **kwargs)
        ----------------------
        Found 1 line(s) in '/home/joncrall/code/wbia/wbia/viz/interact/interact_matches.py':
        interact_matches.py : 372 |            tup = viz.viz_matches.show_matches2(ibs, self.qaid, self.daid,
        ----------------------
        Found 2 line(s) in '/home/joncrall/code/wbia/wbia/algo/hots/chip_match.py':
        chip_match.py : 204 |        viz_matches.show_matches2(qreq_.ibs, cm.qaid, daid, qreq_=qreq_,
        chip_match.py : 219 |            wbia.viz.viz_matches.show_matches2
        ----------------------
        Found 1 line(s) in '/home/joncrall/code/wbia/wbia/algo/hots/scoring.py':
        scoring.py : 562 |        viz.viz_matches.show_matches2(qreq_.ibs, qaid, daid, fm, fs,

    CommandLine:
        python -m wbia.viz.viz_matches --exec-show_matches2 --show
        python -m wbia --tf ChipMatch.ishow_single_annotmatch show_matches2 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.chip_match import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm(defaultdb='PZ_MTEST', default_qaids=[18])
        >>> cm.score_name_nsum(qreq_)
        >>> daid = cm.get_top_aids()[0]
        >>> cm.show_single_annotmatch(qreq_, daid)
        >>> ut.show_if_requested()
    """
    if qreq_ is None:
        print('[viz_matches] WARNING: qreq_ is None')
    kwargs = kwargs.copy()
    in_image = kwargs.get('in_image', False)
    draw_fmatches = kwargs.pop('draw_fmatches', True)
    # Read query and result info (chips, names, ...)
    rchip1, rchip2, kpts1, kpts2 = _get_annot_pair_info(
        ibs, aid1, aid2, qreq_, draw_fmatches, **kwargs
    )
    ut.delete_keys(kwargs, ['kpts1', 'kpts2'])
    if fm is None:
        assert len(kpts1) == len(kpts2), 'keypoints should be in correspondence'
        import numpy as np

        fm = np.vstack((np.arange(len(kpts1)), np.arange(len(kpts1)))).T

    # Build annotation strings / colors
    lbl1 = 'q' + vh.get_aidstrs(aid1)
    lbl2 = vh.get_aidstrs(aid2)
    if in_image:  # HACK!
        lbl1 = None
        lbl2 = None
    # Draws the chips and keypoint matches
    try:
        ax, xywh1, xywh2 = pt.show_chipmatch2(
            rchip1,
            rchip2,
            kpts1,
            kpts2,
            fm,
            fs=fs,
            fm_norm=fm_norm,
            H1=H1,
            H2=H2,
            lbl1=lbl1,
            lbl2=lbl2,
            sel_fm=sel_fm,
            **kwargs
        )
    except Exception as ex:
        ut.printex(ex, 'consider qr.remove_corrupted_queries', '[viz_matches]')
        print('')
        raise
    # Moved the code into show_chipmatch
    # if len(sel_fm) > 0:
    #    # Draw any selected matches
    #    sm_kw = dict(rect=True, colors=pt.BLUE)
    #    pt.plot_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **sm_kw)
    (x1, y1, w1, h1) = xywh1
    (x2, y2, w2, h2) = xywh2
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_matches2(
        ibs,
        aid1,
        aid2,
        fm,
        fs,
        xywh2=xywh2,
        xywh1=xywh1,
        offset1=offset1,
        offset2=offset2,
        **kwargs
    )
    return ax, xywh1, xywh2


# OLD QRES BASED FUNCS STILL IN USE


# @ut.indent_func
def show_matches(ibs, cm, aid2, sel_fm=[], qreq_=None, **kwargs):
    """
    DEPRICATE

    shows single annotated match result.

    Args:
        ibs (IBEISController):
        cm (ChipMatch):  object of feature correspondences and scores
        aid2 (int): result annotation id
        sel_fm (list): selected features match indices

    Kwargs:
        vert (bool)

    Returns:
        tuple: (ax, xywh1, xywh2)

    """
    fm = cm.aid2_fm.get(aid2, [])
    fs = cm.aid2_fs.get(aid2, [])
    aid1 = cm.qaid
    return show_matches2(ibs, aid1, aid2, fm, fs, qreq_=qreq_, **kwargs)


@profile
def show_multichip_match(
    rchip1,
    rchip2_list,
    kpts1,
    kpts2_list,
    fm_list,
    fs_list,
    featflag_list,
    fnum=None,
    pnum=None,
    **kwargs
):
    """
    move to df2
    rchip = rchip1
    H = H1 = None
    target_wh = None

    """
    import vtool.image as gtool
    import wbia.plottool as pt
    import numpy as np
    import vtool as vt

    kwargs = kwargs.copy()

    colorbar_ = kwargs.pop('colorbar_', True)
    stack_larger = kwargs.pop('stack_larger', False)
    stack_side = kwargs.pop('stack_side', False)
    # mode for features disabled by name scoring
    NONVOTE_MODE = kwargs.get('nonvote_mode', 'filter')

    def preprocess_chips(rchip, H, target_wh):
        rchip_ = rchip if H is None else gtool.warpHomog(rchip, H, target_wh)
        return rchip_

    if fnum is None:
        fnum = pt.next_fnum()

    target_wh1 = None
    H1 = None
    rchip1_ = preprocess_chips(rchip1, H1, target_wh1)
    # Hack to visually identify the query
    rchip1_ = vt.draw_border(
        rchip1_,
        out=rchip1_,
        thickness=15,
        color=(pt.UNKNOWN_PURP[0:3] * 255).astype(np.uint8).tolist(),
    )
    wh1 = gtool.get_size(rchip1_)
    rchip2_list_ = [preprocess_chips(rchip2, None, wh1) for rchip2 in rchip2_list]
    wh2_list = [gtool.get_size(rchip2) for rchip2 in rchip2_list_]

    num = 0 if len(rchip2_list) < 3 else 1
    # vert = True if len(rchip2_list) > 1 else False
    vert = True if len(rchip2_list) > 1 else None
    # num = 0

    if False and kwargs.get('fastmode', False):
        # This doesn't actually help the speed very much
        stackkw = dict(
            # Hack draw results faster Q
            # initial_sf=.4,
            # initial_sf=.9,
            use_larger=stack_larger,
            # use_larger=True,
        )
    else:
        stackkw = dict()
    # use_larger = True
    # vert = kwargs.get('fastmode', False)

    if stack_side:
        # hack to stack all database images vertically
        num = 0

    # TODO: heatmask

    match_img, offset_list, sf_list = vt.stack_image_list_special(
        rchip1_, rchip2_list_, num=num, vert=vert, **stackkw
    )
    wh_list = np.array(ut.flatten([[wh1], wh2_list])) * sf_list

    offset1 = offset_list[0]
    wh1 = wh_list[0]
    sf1 = sf_list[0]

    fig, ax = pt.imshow(match_img, fnum=fnum, pnum=pnum)

    if kwargs.get('show_matches', True):
        ut.flatten(fs_list)
        # ut.embed()
        flat_fs, cumlen_list = ut.invertible_flatten2(fs_list)
        flat_colors = pt.scores_to_color(np.array(flat_fs), 'hot')
        colors_list = ut.unflatten2(flat_colors, cumlen_list)
        for _tup in zip(
            offset_list[1:],
            wh_list[1:],
            sf_list[1:],
            kpts2_list,
            fm_list,
            fs_list,
            featflag_list,
            colors_list,
        ):
            offset2, wh2, sf2, kpts2, fm2_, fs2_, featflags, colors = _tup
            xywh1 = (offset1[0], offset1[1], wh1[0], wh1[1])
            xywh2 = (offset2[0], offset2[1], wh2[0], wh2[1])
            # colors = pt.scores_to_color(fs2)
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
                pt.plot_fmatch(
                    xywh1,
                    xywh2,
                    kpts1,
                    kpts2,
                    fm2,
                    fs2,
                    fm_norm=None,
                    H1=None,
                    H2=None,
                    scale_factor1=sf1,
                    scale_factor2=sf2,
                    colorbar_=False,
                    colors=colors,
                    **kwargs
                )
        if colorbar_:
            pt.colorbar(flat_fs, flat_colors)
    bbox_list = [(x, y, w, h) for (x, y), (w, h) in zip(offset_list, wh_list)]
    return offset_list, sf_list, bbox_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_matches --test-show_matches --show

        python -m wbia.viz.viz_matches
        python -m wbia.viz.viz_matches --allexamples
        python -m wbia.viz.viz_matches --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
