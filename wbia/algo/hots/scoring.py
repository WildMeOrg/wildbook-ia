# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import vtool as vt
import utool as ut
from wbia.algo.hots import _pipeline_helpers as plh  # NOQA

print, rrr, profile = ut.inject2(__name__)


def score_chipmatch_list(qreq_, cm_list, score_method, progkw=None):
    """
    CommandLine:
        python -m wbia.algo.hots.scoring --test-score_chipmatch_list
        python -m wbia.algo.hots.scoring --test-score_chipmatch_list:1
        python -m wbia.algo.hots.scoring --test-score_chipmatch_list:0 --show

    Example0:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> # (IMPORTANT)
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver()
        >>> score_method = qreq_.qparams.prescore_method
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> cm = cm_list[0]
        >>> assert cm.score_list.argmax() == 0
        >>> ut.quit_if_noshow()
        >>> cm.show_single_annotmatch(qreq_)
        >>> ut.show_if_requested()

    Example1:
        >>> # SLOW_DOCTEST
        >>> # (IMPORTANT)
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
        >>> qaid = qreq_.qaids[0]
        >>> cm = cm_list[0]
        >>> score_method = qreq_.qparams.score_method
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> assert cm.score_list.argmax() == 0
        >>> ut.quit_if_noshow()
        >>> cm.show_single_annotmatch(qreq_)
        >>> ut.show_if_requested()
    """
    if progkw is None:
        progkw = dict(freq=1, time_thresh=30.0, adjust=True)
    lbl = 'scoring %s' % (score_method)
    # Choose the appropriate scoring mechanism
    print('[scoring] score %d chipmatches with %s' % (len(cm_list), score_method,))
    if score_method == 'sumamech':
        for cm in ut.ProgressIter(cm_list, lbl=lbl, **progkw):
            cm.score_name_sumamech(qreq_)
    if score_method == 'csum':
        for cm in ut.ProgressIter(cm_list, lbl=lbl, **progkw):
            cm.score_name_maxcsum(qreq_)
    elif score_method == 'nsum':
        for cm in ut.ProgressIter(cm_list, lbl=lbl, **progkw):
            cm.score_name_nsum(qreq_)
    else:
        raise NotImplementedError('[hs] unknown scoring method:' + score_method)


def get_name_shortlist_aids(
    daid_list,
    dnid_list,
    annot_score_list,
    name_score_list,
    nid2_nidx,
    nNameShortList,
    nAnnotPerName,
):
    r"""
    CommandLine:
        python -m wbia.algo.hots.scoring --test-get_name_shortlist_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> daid_list        = np.array([11, 12, 13, 14, 15, 16, 17])
        >>> dnid_list        = np.array([21, 21, 21, 22, 22, 23, 24])
        >>> annot_score_list = np.array([ 6,  2,  3,  5,  6,  3,  2])
        >>> name_score_list  = np.array([ 8,  9,  5,  4])
        >>> nid2_nidx        = {21:0, 22:1, 23:2, 24:3}
        >>> nNameShortList, nAnnotPerName = 3, 2
        >>> args = (daid_list, dnid_list, annot_score_list, name_score_list,
        ...         nid2_nidx, nNameShortList, nAnnotPerName)
        >>> top_daids = get_name_shortlist_aids(*args)
        >>> result = str(top_daids)
        >>> print(result)
        [15, 14, 11, 13, 16]
    """
    unique_nids, groupxs = vt.group_indices(np.array(dnid_list))
    grouped_annot_scores = vt.apply_grouping(annot_score_list, groupxs)
    grouped_daids = vt.apply_grouping(np.array(daid_list), groupxs)
    # Ensure name score list is aligned with the unique_nids
    aligned_name_score_list = name_score_list.take(ut.dict_take(nid2_nidx, unique_nids))
    # Sort each group by the name score
    group_sortx = aligned_name_score_list.argsort()[::-1]
    _top_daid_groups = ut.take(grouped_daids, group_sortx)
    _top_annot_score_groups = ut.take(grouped_annot_scores, group_sortx)
    top_daid_groups = ut.listclip(_top_daid_groups, nNameShortList)
    top_annot_score_groups = ut.listclip(_top_annot_score_groups, nNameShortList)
    # Sort within each group by the annotation score
    top_daid_sortx_groups = [
        annot_score_group.argsort()[::-1] for annot_score_group in top_annot_score_groups
    ]
    top_sorted_daid_groups = vt.ziptake(top_daid_groups, top_daid_sortx_groups)
    top_clipped_daids = [
        ut.listclip(sorted_daid_group, nAnnotPerName)
        for sorted_daid_group in top_sorted_daid_groups
    ]
    top_daids = ut.flatten(top_clipped_daids)
    return top_daids


def make_chipmatch_shortlists(
    qreq_, cm_list, nNameShortList, nAnnotPerName, score_method='nsum'
):
    """
    Makes shortlists for reranking

    CommandLine:
        python -m wbia.algo.hots.scoring --test-make_chipmatch_shortlists --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
        >>> score_method    = 'nsum'
        >>> nNameShortList  = 5
        >>> nAnnotPerName   = 6
        >>> # apply scores
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> cm_input = cm_list[0]
        >>> #assert cm_input.dnid_list.take(cm_input.argsort())[0] == cm_input.qnid
        >>> cm_shortlist = make_chipmatch_shortlists(qreq_, cm_list, nNameShortList, nAnnotPerName)
        >>> cm_input.print_rawinfostr()
        >>> cm = cm_shortlist[0]
        >>> cm.print_rawinfostr()
        >>> # should be sorted already from the shortlist take
        >>> top_nid_list = cm.dnid_list
        >>> top_aid_list = cm.daid_list
        >>> qnid = cm.qnid
        >>> print('top_aid_list = %r' % (top_aid_list,))
        >>> print('top_nid_list = %r' % (top_nid_list,))
        >>> print('qnid = %r' % (qnid,))
        >>> rankx = top_nid_list.tolist().index(qnid)
        >>> assert rankx == 0, 'qnid=%r should be first rank, not rankx=%r' % (qnid, rankx)
        >>> max_num_rerank = nNameShortList * nAnnotPerName
        >>> min_num_rerank = nNameShortList
        >>> ut.assert_inbounds(len(top_nid_list), min_num_rerank, max_num_rerank, 'incorrect number in shortlist', eq=True)
        >>> ut.quit_if_noshow()
        >>> cm.show_single_annotmatch(qreq_, daid=top_aid_list[0])
        >>> ut.show_if_requested()
    """
    print(
        '[scoring] Making shortlist nNameShortList=%r, nAnnotPerName=%r'
        % (nNameShortList, nAnnotPerName)
    )
    cm_shortlist = []
    for cm in cm_list:
        assert cm.score_list is not None, 'score list must be computed'
        assert cm.annot_score_list is not None, 'annot_score_list must be computed'
        # FIXME: this should just always be name
        if score_method == 'nsum':
            top_aids = cm.get_name_shortlist_aids(nNameShortList, nAnnotPerName)
        elif score_method == 'csum':
            top_aids = cm.get_annot_shortlist_aids(nNameShortList * nAnnotPerName)
        else:
            raise AssertionError(score_method)
        cm_subset = cm.shortlist_subset(top_aids)
        cm_shortlist.append(cm_subset)
    return cm_shortlist


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.scoring
        python -m wbia.algo.hots.scoring --allexamples
        python -m wbia.algo.hots.scoring --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
