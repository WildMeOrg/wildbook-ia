# -*- coding: utf-8 -*-
"""
TODO: DEPRICATE WITH QRES

IBEIS AGNOSTIC DEFINITIONS ARE NOW IN VTOOL
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)


def get_nTruePositive(atrank, was_retrieved, gt_ranks):
    """ the number of documents we got right """
    TP = (np.logical_and(was_retrieved, gt_ranks <= atrank)).sum()
    return TP


def get_nFalseNegative(TP, atrank, nGroundTruth):
    """ the number of documents we should have retrieved but didn't """
    # FN = min((atrank + 1) - TP, nGroundTruth - TP)
    # nRetreived = (atrank + 1)
    FN = nGroundTruth - TP
    # min(atrank, nGroundTruth - TP)
    return FN


def get_nFalsePositive(TP, atrank):
    """ the number of documents we should not have retrieved """
    # FP = min((atrank + 1) - TP, nGroundTruth)
    nRetreived = atrank + 1
    FP = nRetreived - TP
    return FP


def get_precision(TP, FP):
    """ precision positive predictive value """
    precision = TP / (TP + FP)
    return precision


def get_recall(TP, FN):
    """ recall, true positive rate, sensitivity, hit rate """
    recall = TP / (TP + FN)
    return recall


def get_average_percision_(qres, ibs=None, gt_aids=None):
    """
    gets average percision using the PASCAL definition

    FIXME: Use only the groundtruth that could have been matched in the
    database. (shouldn't be an issue until we start using daid subsets)

    References:
        http://en.wikipedia.org/wiki/Information_retrieval

    """
    recall_range_, p_interp_curve = get_interpolated_precision_vs_recall_(
        qres, ibs=ibs, gt_aids=gt_aids
    )

    if recall_range_ is None:
        ave_p = np.nan
    else:
        ave_p = p_interp_curve.sum() / p_interp_curve.size

    return ave_p


def get_interpolated_precision_vs_recall_(qres, ibs=None, gt_aids=None):
    tup = get_precision_recall_curve_(qres, ibs=ibs, gt_aids=gt_aids)
    ofrank_curve, precision_curve, recall_curve = tup
    recall_range_, p_interp_curve = interpolate_precision_recall_(
        precision_curve, recall_curve
    )
    return recall_range_, p_interp_curve


def interpolate_precision_recall_(precision_curve, recall_curve, nSamples=11):
    if precision_curve is None:
        return None, None

    recall_range_ = np.linspace(0, 1, nSamples)

    def p_interp(r):
        precision_candidates = precision_curve[recall_curve >= r]
        if len(precision_candidates) == 0:
            return 0
        return precision_candidates.max()

    p_interp_curve = np.array([p_interp(r) for r in recall_range_])
    return recall_range_, p_interp_curve


def get_precision_recall_curve_(qres, ibs=None, gt_aids=None):
    """

    CommandLine:
        python -m wbia.algo.hots.precision_recall --test-get_precision_recall_curve_ --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.hots_query_result import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> qaids = ibs.get_valid_aids()[14:15]
        >>> daids = ibs.get_valid_aids()
        >>> qres = ibs.query_chips(qaids, daids)[0]
        >>> gt_aids = None
        >>> atrank  = 18
        >>> nSamples = 20
        >>> ofrank_curve, precision_curve, recall_curve = qres.get_precision_recall_curve(ibs=ibs, gt_aids=gt_aids)
        >>> recall_range_, p_interp_curve = interpolate_precision_recall_(precision_curve, recall_curve, nSamples=nSamples)
        >>> print((recall_range_, p_interp_curve))
        >>> ut.quit_if_noshow()
        >>> draw_precision_recall_curve_(recall_range_, p_interp_curve)
        >>> ut.show_if_requested()

    References:
        http://en.wikipedia.org/wiki/Precision_and_recall
    """
    gt_ranks = np.array(qres.get_gt_ranks(ibs=ibs, gt_aids=gt_aids, fillvalue=None))
    was_retrieved = np.array([rank is not None for rank in gt_ranks])
    nGroundTruth = len(gt_ranks)

    if nGroundTruth == 0:
        return None, None, None

    # From oxford:
    # Precision is defined as the ratio of retrieved positive images to the total number retrieved.
    # Recall is defined as the ratio of the number of retrieved positive images to the total
    # number of positive images in the corpus.

    # with ut.EmbedOnException():
    max_rank = gt_ranks.max()
    if max_rank is None:
        max_rank = 0
    ofrank_curve = np.arange(max_rank + 1)

    truepos_curve = np.array(
        [get_nTruePositive(ofrank, was_retrieved, gt_ranks) for ofrank in ofrank_curve]
    )

    falsepos_curve = np.array(
        [
            get_nFalsePositive(TP, atrank)
            for TP, atrank in zip(truepos_curve, ofrank_curve)
        ],
        dtype=np.float32,
    )

    falseneg_curve = np.array(
        [
            get_nFalseNegative(TP, atrank, nGroundTruth)
            for TP, atrank in zip(truepos_curve, ofrank_curve)
        ],
        dtype=np.float32,
    )

    precision_curve = get_precision(truepos_curve, falsepos_curve)
    recall_curve = get_recall(truepos_curve, falseneg_curve)

    # print(np.vstack([precision_curve, recall_curve]).T)
    return ofrank_curve, precision_curve, recall_curve


def show_precision_recall_curve_(qres, ibs=None, gt_aids=None, fnum=1):
    """
    CHANGE NAME TO REFERENCE QRES
    """
    recall_range_, p_interp_curve = get_interpolated_precision_vs_recall_(
        qres, ibs=ibs, gt_aids=gt_aids
    )
    title_pref = (qres.make_smaller_title() + '\n',)
    return draw_precision_recall_curve_(recall_range_, p_interp_curve, title_pref, fnum)


def draw_precision_recall_curve_(recall_range_, p_interp_curve, title_pref=None, fnum=1):
    import wbia.plottool as pt

    if recall_range_ is None:
        recall_range_ = np.array([])
        p_interp_curve = np.array([])
    fig = pt.figure(fnum=fnum, docla=True, doclf=True)  # NOQA

    if recall_range_ is None:
        ave_p = np.nan
    else:
        ave_p = p_interp_curve.sum() / p_interp_curve.size

    pt.plot2(
        recall_range_,
        p_interp_curve,
        marker='o--',
        x_label='recall',
        y_label='precision',
        unitbox=True,
        flipx=False,
        color='r',
        title='Interplated Precision Vs Recall\n' + 'avep = %r' % ave_p,
    )
    print('Interplated Precision')
    print(ut.repr2(list(zip(recall_range_, p_interp_curve))))
    # fig.show()


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.precision_recall
        python -m wbia.algo.hots.precision_recall --allexamples
        python -m wbia.algo.hots.precision_recall --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
