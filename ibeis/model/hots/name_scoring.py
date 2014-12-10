from __future__ import absolute_import, division, print_function
import numpy as np
import vtool as vt
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[nscoring]', DEBUG=False)


def get_one_score_per_name(ibs, aid_list, score_list):
    # TODO : rectify with code in pipeline
    """
    Converts annotation scores to name scores

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *   # NOQA
        >>> import ibeis
        >>> from ibeis.dev import results_all
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> daid_list = ibs.get_valid_aids()
        >>> qaid_list = daid_list[0:1]
        >>> cfgdict = dict()
        >>> qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list, cfgdict)
        >>> qres = qaid2_qres[qaid_list[0]]
        >>> aid_list, score_list = qres.get_aids_and_scores()
        >>> nscoretup = get_one_score_per_name(ibs, aid_list, score_list)
        >>> (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscoretup
        >>> result = str(sorted_nids[0:3])
        >>> print(result)
        [2 6 5]
    """
    score_arr = np.array(score_list)
    aid_list  = np.array(aid_list)
    nid_list  = np.array(ibs.get_annot_name_rowids(aid_list))
    unique_nids, groupxs = vt.group_indicies(nid_list)
    grouped_scores = np.array(vt.apply_grouping(score_arr, groupxs))
    grouped_aids   = np.array(vt.apply_grouping(aid_list, groupxs))
    # Build representative score per group
    group_nscore = np.array([scores.max() for scores in grouped_scores])
    group_sortx = group_nscore.argsort()[::-1]
    # Top nids
    sorted_nids = unique_nids.take(group_sortx, axis=0)
    sorted_nscore = group_nscore.take(group_sortx, axis=0)
    # Initial sort of aids
    _sorted_aids   = grouped_aids.take(group_sortx, axis=0)
    _sorted_scores = grouped_scores.take(group_sortx, axis=0)
    # Secondary sort of aids
    sorted_sortx  = [scores.argsort()[::-1] for scores in _sorted_scores]
    sorted_scores = [scores.take(sortx) for scores, sortx in zip(_sorted_scores, sorted_sortx)]
    sorted_aids   = [aids.take(sortx) for aids, sortx in zip(_sorted_aids, sorted_sortx)]
    nscoretup     = (sorted_nids, sorted_nscore, sorted_aids, sorted_scores)
    return nscoretup


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.name_scoring
        python -m ibeis.model.hots.name_scoring --allexamples
        python -m ibeis.model.hots.name_scoring --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
