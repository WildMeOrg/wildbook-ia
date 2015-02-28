from __future__ import absolute_import, division, print_function
import numpy as np
import vtool as vt
import utool as ut
import six
from ibeis.model.hots import hstypes
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
from collections import namedtuple
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[nscoring]', DEBUG=False)

NameScoreTup = namedtuple('NameScoreTup', ('sorted_nids', 'sorted_nscore',
                                           'sorted_aids', 'sorted_scores'))


def dupvote_dense_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_):
    """
    dupvotes gives duplicate name votes a weight close to 0.

    Dense version of name weighting

    Each query feature is only allowed to vote for each name at most once.
    IE: a query feature can vote for multiple names, but it cannot vote
    for the same name twice.

    CommandLine:
        python dev.py --allgt -t best --db PZ_MTEST
        python dev.py --allgt -t nsum --db PZ_MTEST
        python dev.py --allgt -t dupvote --db PZ_MTEST

    CommandLine:
        # Compares with dupvote on and dupvote off
        ./dev.py -t custom:dupvote_weight=0.0 custom:dupvote_weight=1.0  --db GZ_ALL --show --va -w --qaid 1032

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #tup = nn_weights.testdata_nn_weights('testdb1', slice(0, 1), slice(0, 11))
        >>> dbname = 'testdb1'  # 'GZ_ALL'  # 'testdb1'
        >>> cfgdict = dict(K=10, Knorm=10, codename='nsum')
        >>> ibs, qreq_ = plh.get_pipeline_testdata(dbname=dbname, qaid_list=[2], daid_list=[1, 2, 3], cfgdict=cfgdict)
        >>> print(print(qreq_.get_infostr()))
        >>> pipeline_locals_ = plh.testrun_pipeline_upto(qreq_, 'weight_neighbors')
        >>> qaid2_nns, qaid2_nnvalid0 = ut.dict_take(pipeline_locals_, ['qaid2_nns', 'qaid2_nnvalid0'])
        >>> # Test Function Call
        >>> qaid2_dupvote_weight = dupvote_dense_match_weighter(qaid2_nns, qaid2_nnvalid0, qreq_)
        >>> # Check consistency
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> qfx2_dupvote_weight = qaid2_dupvote_weight[qaid]
        >>> flags = qfx2_dupvote_weight  > .5
        >>> qfx2_topnid = ibs.get_annot_name_rowids(qreq_.indexer.get_nn_aids(qaid2_nns[qaid][0]))
        >>> isunique_list = [ut.isunique(row[flag]) for row, flag in zip(qfx2_topnid, flags)]
        >>> assert all(isunique_list), 'dupvote should only allow one vote per name'

    """
    K = qreq_.qparams.K
    def find_dupvotes(nns, qfx2_invalid0):
        if len(qfx2_invalid0) == 0:
            # hack for empty query features (should never happen, but it
            # inevitably will)
            qfx2_dupvote_weight = np.empty((0, K), dtype=hstypes.FS_DTYPE)
        else:
            (qfx2_idx, qfx2_dist) = nns
            qfx2_topidx = qfx2_idx.T[0:K].T
            qfx2_topaid = qreq_.indexer.get_nn_aids(qfx2_topidx)
            qfx2_topnid = qreq_.ibs.get_annot_name_rowids(qfx2_topaid)
            qfx2_topnid[qfx2_invalid0] = 0
            # A duplicate vote is when any vote for a name after the first
            qfx2_isnondup = np.array([ut.flag_unique_items(topnids) for topnids in qfx2_topnid])
            # set invalids to be duplicates as well (for testing)
            qfx2_isnondup[qfx2_invalid0] = False
            # Database feature index to chip index
            qfx2_dupvote_weight = (qfx2_isnondup.astype(hstypes.FS_DTYPE) * (1 - 1E-7)) + 1E-7
        return qfx2_dupvote_weight

    # convert ouf of dict format
    qaid_list = list(six.iterkeys(qaid2_nns))
    nns_list        = ut.dict_take(qaid2_nns, qaid_list)
    nnvalid0_list   = ut.dict_take(qaid2_nnvalid0, qaid_list)
    nninvalid0_list = [np.bitwise_not(qfx2_valid0) for qfx2_valid0 in nnvalid0_list]
    dupvote_weight_list = [
        find_dupvotes(nns, qfx2_invalid0)
        for nns, qfx2_invalid0 in zip(nns_list, nninvalid0_list)
    ]
    # convert into dict format
    qaid2_dupvote_weight = dict(zip(qaid_list, dupvote_weight_list))
    return qaid2_dupvote_weight


def group_scores_by_name(ibs, aid_list, score_list):
    """
    Converts annotation scores to name scores.
    Over multiple annotations finds keypoints best match and uses that score.

    CommandLine:
        python -m ibeis.model.hots.name_scoring --test-group_scores_by_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *   # NOQA
        >>> import ibeis
        >>> from ibeis.dev import results_all
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> daid_list = ibs.get_valid_aids()
        >>> qaid_list = daid_list[0:1]
        >>> cfgdict = dict()
        >>> qaid2_qres, qreq_ = ibs._query_chips4(
        ...     qaid_list, daid_list, cfgdict=cfgdict, return_request=True,
        ...     use_cache=False, save_qcache=False)
        >>> qres = qaid2_qres[qaid_list[0]]
        >>> print(qres.get_inspect_str())
        >>> print(qres.get_inspect_str(ibs=ibs, name_scoring=True))
        >>> aid_list, score_list = qres.get_aids_and_scores()
        >>> nscoretup = group_scores_by_name(ibs, aid_list, score_list)
        >>> (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscoretup
        >>> ut.assert_eq(sorted_nids[0], 1)

    # TODO: this code needs a really good test case
    #>>> result = np.array_repr(sorted_nids[0:2])
    #>>> print(result)
    #array([1, 5])

    Ignore::
        # hack in dict of Nones prob for testing
        import six
        qres.aid2_prob = {aid:None for aid in six.iterkeys(qres.aid2_score)}

    array([ 1,  5, 26])
    [2 6 5]
    """
    assert len(score_list) == len(aid_list), 'scores and aids must be associated'
    score_arr = np.array(score_list)
    aid_list  = np.array(aid_list)
    nid_list  = np.array(ibs.get_annot_name_rowids(aid_list))
    # Group scores by name
    unique_nids, groupxs = vt.group_indices(nid_list)
    grouped_scores = np.array(vt.apply_grouping(score_arr, groupxs))
    grouped_aids   = np.array(vt.apply_grouping(aid_list, groupxs))
    # Build representative score per group
    # (find each keypoints best match per annotation within the name)
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
    nscoretup     = NameScoreTup(sorted_nids, sorted_nscore, sorted_aids, sorted_scores)
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
