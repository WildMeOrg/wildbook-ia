# -*- coding: utf-8 -*-
"""
GOALS:
    1) vsmany
       * works resaonable for very few and very many
       * stars with small k and then k becomes a percent or log percent
       * distinctiveness from different location

    2) 1-vs-1
       * uses distinctiveness and foreground when available
       * start with ratio test and ransac

    3) First N decision are interactive until we learn a good threshold

    4) Always show numbers between 0 and 1 spatial verification is based on
    single best exemplar

       x - build encoder
       x - test encoder
       x - monotonicity (both nondecreasing and strictly increasing)
       x - cache encoder
       x - cache maitainance (deleters and listers)
       o - Incemental learning
       o - Spceies sensitivity

    * Add ability for user to relearn encoder from labeled database.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool
import numpy as np
import utool as ut
import vtool as vt
import six  # NOQA
print, rrr, profile = utool.inject2(__name__, '[scorenorm]', DEBUG=False)


def learn_annotscore_normalizer(qreq_, **learnkw):
    """
    Takes the result of queries and trains a score encoder

    Args:
        ibs (ibeis.IBEISController):
        cm_list (list):  object of feature correspondences and scores
        cfgstr (str):

    Returns:
        vtool.ScoreNormalizer: freshly trained score encoder

    Args:
        qreq_ (ibeis.QueryRequest):  query request object with hyper-parameters
        cfgstr (str):
        prefix (?):

    Returns:
        vtool.ScoreNormalizer: encoder

    CommandLine:
        python -m ibeis.algo.hots.scorenorm --exec-learn_annotscore_normalizer --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.scorenorm import *  # NOQA
        >>> import ibeis
        >>> qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST', a=['default'], p=['default'])
        >>> encoder = learn_annotscore_normalizer(qreq_, cfgstr, prefix)
        >>> result = ('encoder = %s' % (ut.repr2(encoder),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> encoder.visualize()
        >>> ut.show_if_requested()
    """
    ibs = qreq_.ibs
    cm_list = ibs.query_chips(qreq_=qreq_)
    datatup = get_annotscore_training_data(ibs, cm_list)
    (tp_support, tn_support, tp_support_labels, tn_support_labels) = datatup
    if len(tp_support) < 2 or len(tn_support) < 2:
        print('len(tp_support) = %r' % (len(tp_support),))
        print('len(tn_support) = %r' % (len(tn_support),))
        print('Warning: [scorenorm] not enough data')
        import warnings
        warnings.warn('Warning: [scorenorm] not enough data')
    # timestamp = ut.get_printable_timestamp()
    encoder = vt.ScoreNormalizer()
    vt.fit()
    return encoder


def get_annotscore_training_data(ibs, cm_list):
    """
    Returns "good" taining examples
    """
    good_tp_nscores = []
    good_tn_nscores = []
    good_tp_aidnid_pairs = []
    good_tn_aidnid_pairs = []

    trainable = [ibs.get_annot_has_groundtruth(cm.qaid, daid_list=cm.daid_list)
                 for cm in cm_list]
    cm_list_ = ut.compress(cm_list, trainable)

    for cm in cm_list_:
        qaid = cm.qaid
        qnid = ibs.get_annot_name_rowids(cm.qaid)

        nscoretup = cm.get_ranked_nids_and_aids()
        (sorted_nids, sorted_nscores, sorted_aids, sorted_scores) = nscoretup

        sorted_ndiff = -np.diff(sorted_nscores.tolist())
        sorted_nids = np.array(sorted_nids)
        is_positive  = sorted_nids == qnid
        is_negative = np.logical_and(~is_positive, sorted_nids > 0)
        # Only take data from results with positive and negative examples
        if not np.any(is_positive) or not np.any(is_negative):
            continue
        gt_rank = np.nonzero(is_positive)[0][0]
        gf_rank = np.nonzero(is_negative)[0][0]
        # Only take correct groundtruth scores
        if gt_rank == 0 and len(sorted_nscores) > gf_rank:
            if len(sorted_ndiff) > gf_rank:
                good_tp_nscores.append(sorted_nscores[gt_rank])
                good_tn_nscores.append(sorted_nscores[gf_rank])
                good_tp_aidnid_pairs.append((qaid, sorted_nids[gt_rank]))
                good_tn_aidnid_pairs.append((qaid, sorted_nids[gf_rank]))
    tp_support = np.array(good_tp_nscores)
    tn_support = np.array(good_tn_nscores)
    tp_support_labels = good_tp_aidnid_pairs
    tn_support_labels = good_tp_aidnid_pairs
    return (tp_support, tn_support, tp_support_labels, tn_support_labels)


def get_featscore_training_data(qreq_, cm_list, disttypes_=None,
                                namemode=True):
    from ibeis.algo.hots import chip_match
    fsv_col_lbls = None
    tp_fsvs_list = []
    tn_fsvs_list = []
    for cm in ut.ProgressIter(cm_list,
                              lbl='building featscore lists',
                              adjust=True, freq=1):
        try:
            if disttypes_ is None:
                # Use precomputed fsv distances
                fsv_col_lbls = cm.fsv_col_lbls
                tp_fsv, tn_fsv = chip_match.get_training_fsv(
                    cm, namemode=namemode)
            else:
                # Investigate independant computed dists
                fsv_col_lbls = disttypes_
                tp_fsv, tn_fsv = chip_match.get_training_desc_dist(
                    cm, qreq_, fsv_col_lbls, namemode=namemode)
            tp_fsvs_list.extend(tp_fsv)
            tn_fsvs_list.extend(tn_fsv)
        except chip_match.UnbalancedExampleException:
            continue
    fsv_tp = np.vstack(tp_fsvs_list)
    fsv_tn = np.vstack(tn_fsvs_list)
    return fsv_tp, fsv_tn, fsv_col_lbls


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.scorenorm
        python -m ibeis.algo.hots.scorenorm --allexamples
        python -m ibeis.algo.hots.scorenorm --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
