# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range, map  # NOQA
import numpy as np
import vtool as vt
import utool as ut
import itertools
from wbia.algo.hots import hstypes
from wbia.algo.hots import _pipeline_helpers as plh  # NOQA
from collections import namedtuple

(print, rrr, profile) = ut.inject2(__name__, '[nscoring]')

NameScoreTup = namedtuple(
    'NameScoreTup', ('sorted_nids', 'sorted_nscore', 'sorted_aids', 'sorted_scores')
)


def testdata_chipmatch():
    from wbia.algo.hots import chip_match

    # only the first indicies will matter in these test
    # feature matches
    fm_list = [
        np.array([(0, 9), (1, 9), (2, 9), (3, 9)], dtype=np.int32),
        np.array([(0, 9), (1, 9), (2, 9), (3, 9)], dtype=np.int32),
        np.array([(0, 9), (1, 9), (2, 9), (3, 9)], dtype=np.int32),
        np.array([(4, 9), (5, 9), (6, 9), (3, 9)], dtype=np.int32),
        np.array([(0, 9), (1, 9), (2, 9), (3, 9), (4, 9)], dtype=np.int32),
    ]
    # score each feature match as 1
    fsv_list = [
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
    ]
    cm = chip_match.ChipMatch(
        qaid=1,
        daid_list=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        fm_list=fm_list,
        fsv_list=fsv_list,
        dnid_list=np.array([1, 1, 2, 2, 3], dtype=np.int32),
        fsv_col_lbls=['count'],
    )
    return cm


@profile
def compute_fmech_score(cm, qreq_=None, hack_single_ori=False):
    r"""
    nsum. This is the fmech scoring mechanism.


    Args:
        cm (wbia.ChipMatch):

    Returns:
        tuple: (unique_nids, nsum_score_list)

    CommandLine:
        python -m wbia.algo.hots.name_scoring --test-compute_fmech_score
        python -m wbia.algo.hots.name_scoring --test-compute_fmech_score:0
        python -m wbia.algo.hots.name_scoring --test-compute_fmech_score:2
        utprof.py -m wbia.algo.hots.name_scoring --test-compute_fmech_score:2
        utprof.py -m wbia.algo.hots.pipeline --test-request_wbia_query_L0:0 --db PZ_Master1 -a timectrl:qindex=0:256

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> cm = testdata_chipmatch()
        >>> nsum_score_list = compute_fmech_score(cm)
        >>> assert np.all(nsum_score_list == [ 4.,  7.,  5.])

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_)
        >>> cm._cast_scores()
        >>> #cm.qnid = 1   # Hack for testdb1 names
        >>> nsum_score_list = compute_fmech_score(cm, qreq_)
        >>> #assert np.all(nsum_nid_list == cm.unique_nids), 'nids out of alignment'
        >>> flags = (cm.unique_nids == cm.qnid)
        >>> max_true = nsum_score_list[flags].max()
        >>> max_false = nsum_score_list[~flags].max()
        >>> assert max_true > max_false, 'is this truely a hard case?'
        >>> assert max_true > 1.2, 'score=%r should be higher for aid=18' % (max_true,)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18], cfgdict=dict(query_rotation_heuristic=True))
        >>> cm = cm_list[0]
        >>> cm.score_name_nsum(qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)

    Example3:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('testdb1', qaid_list=[1], cfgdict=dict(query_rotation_heuristic=True))
        >>> cm = cm_list[0]
        >>> cm.score_name_nsum(qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)
    """
    # assert qreq_ is not None
    if hack_single_ori is None:
        try:
            hack_single_ori = qreq_ is not None and (
                qreq_.qparams.query_rotation_heuristic
                or qreq_.qparams.rotation_invariance
            )
        except AttributeError:
            hack_single_ori = True
    # The core for each feature match
    #
    # The query feature index for each feature match
    fm_list = cm.fm_list
    fs_list = cm.get_fsv_prod_list()
    fx1_list = [fm.T[0] for fm in fm_list]
    if hack_single_ori:
        # Group keypoints with the same xy-coordinate.
        # Combine these feature so each only recieves one vote
        kpts1 = qreq_.ibs.get_annot_kpts(cm.qaid, config2_=qreq_.extern_query_config2)
        xys1_ = vt.get_xys(kpts1).T
        fx1_to_comboid = vt.compute_unique_arr_dataids(xys1_)
        fcombo_ids = [fx1_to_comboid.take(fx1) for fx1 in fx1_list]
    else:
        # use the feature index itself as a combo id
        # so each feature only recieves one vote
        fcombo_ids = fx1_list

    if False:
        import ubelt as ub

        for ids in fcombo_ids:
            ub.find_duplicates(ids)

    # Group annotation matches by name
    # nsum_nid_list, name_groupxs = vt.group_indices(cm.dnid_list)
    # nsum_nid_list = cm.unique_nids
    name_groupxs = cm.name_groupxs

    nsum_score_list = []
    # For all indicies matched to a particular name
    for name_idxs in name_groupxs:
        # Get feat indicies and scores corresponding to the name's annots
        name_combo_ids = ut.take(fcombo_ids, name_idxs)
        name_fss = ut.take(fs_list, name_idxs)
        # Flatten over annots in the name
        fs = np.hstack(name_fss)
        if len(fs) == 0:
            nsum_score_list.append(0)
            continue
        combo_ids = np.hstack(name_combo_ids)
        # Features (with the same id) can't vote for this name twice
        group_idxs = vt.group_indices(combo_ids)[1]
        flagged_idxs = [idxs[fs.take(idxs).argmax()] for idxs in group_idxs]
        # Detail: sorting the idxs preseveres summation order
        # this fixes the numerical issue where nsum and csum were off
        flagged_idxs = np.sort(flagged_idxs)
        name_score = fs.take(flagged_idxs).sum()

        nsum_score_list.append(name_score)
    nsum_score_list = np.array(nsum_score_list)

    return nsum_score_list


@profile
def get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_=None):
    """
    DEPRICATE

    Computes flags to desribe which features can or can not vote

    CommandLine:
        python -m wbia.algo.hots.name_scoring --exec-get_chipmatch_namescore_nonvoting_feature_flags

    Example:
        >>> # ENABLE_DOCTEST
        >>> # FIXME: breaks when fg_on=True
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> from wbia.algo.hots import name_scoring
        >>> # Test to make sure name score and chips score are equal when per_name=1
        >>> qreq_, args = plh.testdata_pre('spatial_verification', defaultdb='PZ_MTEST', a=['default:dpername=1,qsize=1,dsize=10'], p=['default:K=1,fg_on=True'])
        >>> cm_list = args.cm_list_FILT
        >>> ibs = qreq_.ibs
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_)
        >>> featflat_list = get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_)
        >>> assert all(list(map(np.all, featflat_list))), 'all features should be able to vote in K=1, per_name=1 case'
    """
    try:
        hack_single_ori = qreq_ is not None and (
            qreq_.qparams.query_rotation_heuristic or qreq_.qparams.rotation_invariance
        )
    except AttributeError:
        hack_single_ori = True
        pass
    # The core for each feature match
    fs_list = cm.get_fsv_prod_list()
    # The query feature index for each feature match
    fm_list = cm.fm_list
    kpts1 = (
        None
        if not hack_single_ori
        else qreq_.ibs.get_annot_kpts(cm.qaid, config2_=qreq_.extern_query_config2)
    )
    dnid_list = cm.dnid_list
    name_groupxs = cm.name_groupxs
    featflag_list = get_namescore_nonvoting_feature_flags(
        fm_list, fs_list, dnid_list, name_groupxs, kpts1=kpts1
    )
    return featflag_list


@profile
def get_namescore_nonvoting_feature_flags(
    fm_list, fs_list, dnid_list, name_groupxs, kpts1=None
):
    r"""
    DEPRICATE

    fm_list = [fm[:min(len(fm), 10)] for fm in fm_list]
    fs_list = [fs[:min(len(fs), 10)] for fs in fs_list]
    """
    fx1_list = [fm.T[0] for fm in fm_list]
    # Group annotation matches by name
    name_grouped_fx1_list = vt.apply_grouping_(fx1_list, name_groupxs)
    name_grouped_fs_list = vt.apply_grouping_(fs_list, name_groupxs)
    # Stack up all matches to a particular name, keep track of original indicies via offets
    name_invertable_flat_fx1_list = list(
        map(ut.invertible_flatten2_numpy, name_grouped_fx1_list)
    )
    name_grouped_fx1_flat = ut.get_list_column(name_invertable_flat_fx1_list, 0)
    name_grouped_invertable_cumsum_list = ut.get_list_column(
        name_invertable_flat_fx1_list, 1
    )
    name_grouped_fs_flat = list(map(np.hstack, name_grouped_fs_list))
    if kpts1 is not None:
        xys1_ = vt.get_xys(kpts1).T
        kpts_xyid_list = vt.compute_unique_data_ids(xys1_)
        # Make nested group for every name by query feature index (accounting for duplicate orientation)
        name_grouped_comboid_flat = list(
            kpts_xyid_list.take(fx1) for fx1 in name_grouped_fx1_flat
        )
        xyid_groupxs_list = list(
            vt.group_indices(xyid_flat)[1] for xyid_flat in name_grouped_comboid_flat
        )
        name_group_fx1_groupxs_list = xyid_groupxs_list
    else:
        # Make nested group for every name by query feature index
        fx1_groupxs_list = [
            vt.group_indices(fx1_flat)[1] for fx1_flat in name_grouped_fx1_flat
        ]
        name_group_fx1_groupxs_list = fx1_groupxs_list
    name_grouped_fid_grouped_fs_list = [
        vt.apply_grouping(fs_flat, fid_groupxs)
        for fs_flat, fid_groupxs in zip(name_grouped_fs_flat, name_group_fx1_groupxs_list)
    ]

    # Flag which features are valid in this grouped space. Only one keypoint should be able to vote
    # for each group
    name_grouped_fid_grouped_isvalid_list = [
        np.array([fs_group.max() == fs_group for fs_group in fid_grouped_fs_list])
        for fid_grouped_fs_list in name_grouped_fid_grouped_fs_list
    ]

    # Go back to being grouped only in name space
    # dtype = np.bool
    name_grouped_isvalid_flat_list = [
        vt.invert_apply_grouping2(fid_grouped_isvalid_list, fid_groupxs, dtype=np.bool)
        for fid_grouped_isvalid_list, fid_groupxs in zip(
            name_grouped_fid_grouped_isvalid_list, name_group_fx1_groupxs_list
        )
    ]

    name_grouped_isvalid_unflat_list = [
        ut.unflatten2(isvalid_flat, invertable_cumsum_list)
        for isvalid_flat, invertable_cumsum_list in zip(
            name_grouped_isvalid_flat_list, name_grouped_invertable_cumsum_list
        )
    ]

    # Reports which features were valid in name scoring for every annotation
    featflag_list = vt.invert_apply_grouping(
        name_grouped_isvalid_unflat_list, name_groupxs
    )
    return featflag_list


@profile
def align_name_scores_with_annots(
    annot_score_list, annot_aid_list, daid2_idx, name_groupxs, name_score_list
):
    r"""
    takes name scores and gives them to the best annotation

    Returns:
        score_list: list of scores aligned with cm.daid_list and cm.dnid_list

    Args:
        annot_score_list (list): score associated with each annot
        name_groupxs (list): groups annot_score lists into groups compatible with name_score_list
        name_score_list (list): score assocated with name
        nid2_nidx (dict): mapping from nids to index in name score list

    CommandLine:
        python -m wbia.algo.hots.name_scoring --test-align_name_scores_with_annots
        python -m wbia.algo.hots.name_scoring --test-align_name_scores_with_annots --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.evaluate_csum_annot_score(qreq_)
        >>> cm.evaluate_nsum_name_score(qreq_)
        >>> # Annot aligned lists
        >>> annot_score_list = cm.algo_annot_scores['csum']
        >>> annot_aid_list   = cm.daid_list
        >>> daid2_idx        = cm.daid2_idx
        >>> # Name aligned lists
        >>> name_score_list  = cm.algo_name_scores['nsum']
        >>> name_groupxs     = cm.name_groupxs
        >>> # Execute Function
        >>> score_list = align_name_scores_with_annots(annot_score_list, annot_aid_list, daid2_idx, name_groupxs, name_score_list)
        >>> # Check that the correct name gets the highest score
        >>> target = name_score_list[cm.nid2_nidx[cm.qnid]]
        >>> test_index = np.where(score_list == target)[0][0]
        >>> cm.score_list = score_list
        >>> ut.assert_eq(ibs.get_annot_name_rowids(cm.daid_list[test_index]), cm.qnid)
        >>> assert ut.isunique(cm.dnid_list[score_list > 0]), 'bad name score'
        >>> top_idx = cm.algo_name_scores['nsum'].argmax()
        >>> assert cm.get_top_nids()[0] == cm.unique_nids[top_idx], 'bug in alignment'
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.name_scoring import *  # NOQA
        >>> annot_score_list = []
        >>> annot_aid_list   = []
        >>> daid2_idx        = {}
        >>> # Name aligned lists
        >>> name_score_list  = np.array([], dtype=np.float32)
        >>> name_groupxs     = []
        >>> # Execute Function
        >>> score_list = align_name_scores_with_annots(annot_score_list, annot_aid_list, daid2_idx, name_groupxs, name_score_list)
    """
    if len(name_groupxs) == 0:
        score_list = np.empty(0, dtype=name_score_list.dtype)
        return score_list
    else:
        # Group annot aligned indicies by nid
        annot_aid_list = np.array(annot_aid_list)
        # nid_list, groupxs  = vt.group_indices(annot_nid_list)
        grouped_scores = vt.apply_grouping(annot_score_list, name_groupxs)
        grouped_annot_aids = vt.apply_grouping(annot_aid_list, name_groupxs)
        flat_grouped_aids = np.hstack(grouped_annot_aids)
        # flat_groupxs  = np.hstack(name_groupxs)
        # if __debug__:
        #    sum_scores = np.array([scores.sum() for scores in grouped_scores])
        #    max_scores = np.array([scores.max() for scores in grouped_scores])
        #    assert np.all(name_score_list <= sum_scores)
        #    assert np.all(name_score_list > max_scores)
        # +------------
        # Find the position of the highest name_scoring annotation for each name
        # IN THE FLATTENED GROUPED ANNOT_AID_LIST (this was the bug)
        offset_list = np.array([annot_scores.argmax() for annot_scores in grouped_scores])
        # Find the starting position of eatch group use chain to start offsets with 0
        _padded_scores = itertools.chain([[]], grouped_scores[:-1])
        sizeoffset_list = np.array([len(annot_scores) for annot_scores in _padded_scores])
        baseindex_list = sizeoffset_list.cumsum()
        # Augment starting position with offset index
        annot_idx_list = np.add(baseindex_list, offset_list)
        # L______________
        best_aid_list = flat_grouped_aids[annot_idx_list]
        best_idx_list = ut.dict_take(daid2_idx, best_aid_list)
        # give the annotation domain a name score
        # score_list = np.zeros(len(annot_score_list), dtype=name_score_list.dtype)
        score_list = np.full(
            len(annot_score_list), fill_value=-np.inf, dtype=name_score_list.dtype
        )
        # score_list = np.full(len(annot_score_list), fill_value=np.nan, dtype=name_score_list.dtype)
        # score_list = np.nan(len(annot_score_list), dtype=name_score_list.dtype)
        # HACK: we need to set these to 'low' values and we also have to respect negatives
        # score_list[:] = -np.inf
        # make sure that the nid_list from group_indicies and the nids belonging to
        # name_score_list (cm.unique_nids) are in alignment
        # nidx_list = np.array(ut.dict_take(nid2_nidx, nid_list))

        # THIS ASSUMES name_score_list IS IN ALIGNMENT WITH BOTH cm.unique_nids and
        # nid_list (which should be == cm.unique_nids)
        score_list[best_idx_list] = name_score_list
        return score_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.algo.hots.name_scoring
        python -m wbia.algo.hots.name_scoring --allexamples
        python -m wbia.algo.hots.name_scoring --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
