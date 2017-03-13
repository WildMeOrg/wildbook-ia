# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range, map  # NOQA
import numpy as np
import vtool as vt
import utool as ut
import itertools
from ibeis.algo.hots import hstypes
from ibeis.algo.hots import _pipeline_helpers as plh  # NOQA
from collections import namedtuple
(print, rrr, profile) = ut.inject2(__name__, '[nscoring]')

NameScoreTup = namedtuple('NameScoreTup', ('sorted_nids', 'sorted_nscore',
                                           'sorted_aids', 'sorted_scores'))


def testdata_chipmatch():
    from ibeis.algo.hots import chip_match
    # only the first indicies will matter in these test
    # feature matches
    fm_list = [
        np.array([(0, 9), (1, 9), (2, 9), (3, 9)], dtype=np.int32),
        np.array([(0, 9), (1, 9), (2, 9), (3, 9)], dtype=np.int32),
        np.array([(0, 9), (1, 9), (2, 9), (3, 9)], dtype=np.int32),
        np.array([(4, 9), (5, 9), (6, 9), (3, 9)], dtype=np.int32),
        np.array([(0, 9), (1, 9), (2, 9), (3, 9), (4, 9)], dtype=np.int32)
    ]
    # score each feature match as 1
    fsv_list = [
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,)], dtype=hstypes.FS_DTYPE),
        np.array([(1,), (1,), (1,), (1,), (1, )], dtype=hstypes.FS_DTYPE),
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
def compute_nsum_score(cm, qreq_=None):
    r"""
    nsum

    Args:
        cm (ibeis.ChipMatch):

    Returns:
        tuple: (unique_nids, nsum_score_list)

    CommandLine:
        python -m ibeis.algo.hots.name_scoring --test-compute_nsum_score
        python -m ibeis.algo.hots.name_scoring --test-compute_nsum_score:0
        python -m ibeis.algo.hots.name_scoring --test-compute_nsum_score:2
        utprof.py -m ibeis.algo.hots.name_scoring --test-compute_nsum_score:2
        utprof.py -m ibeis.algo.hots.pipeline --test-request_ibeis_query_L0:0 --db PZ_Master1 -a timectrl:qindex=0:256

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> # build test data
        >>> cm = testdata_chipmatch()
        >>> # execute function
        >>> (unique_nids, nsum_score_list) = compute_nsum_score(cm)
        >>> result = ut.list_str((unique_nids, nsum_score_list), label_list=['unique_nids', 'nsum_score_list'], with_dtype=False)
        >>> print(result)
        unique_nids = np.array([1, 2, 3])
        nsum_score_list = np.array([ 4.,  7.,  5.])

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_)
        >>> cm._cast_scores()
        >>> #cm.qnid = 1   # Hack for testdb1 names
        >>> nsum_nid_list, nsum_score_list = compute_nsum_score(cm, qreq_)
        >>> assert np.all(nsum_nid_list == cm.unique_nids), 'nids out of alignment'
        >>> flags = (nsum_nid_list == cm.qnid)
        >>> max_true = nsum_score_list[flags].max()
        >>> max_false = nsum_score_list[~flags].max()
        >>> assert max_true > max_false, 'is this truely a hard case?'
        >>> assert max_true > 1.2, 'score=%r should be higher for aid=18' % (max_true,)
        >>> nsum_nid_list2, nsum_score_list2, _ = compute_nsum_score2(cm, qreq_)
        >>> assert np.allclose(nsum_score_list2, nsum_score_list), 'something is very wrong'
        >>> #assert np.all(nsum_score_list2 == nsum_score_list), 'could be a percision issue'

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18], cfgdict=dict(query_rotation_heuristic=True))
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)

    Example3:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('testdb1', qaid_list=[1], cfgdict=dict(query_rotation_heuristic=True))
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)

    Example4:
        >>> # ENABLE_DOCTEST
        >>> # FIXME: breaks when fg_on=True
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> from ibeis.algo.hots import name_scoring
        >>> from ibeis.algo.hots import scoring
        >>> import ibeis
        >>> # Test to make sure name score and chips score are equal when per_name=1
        >>> qreq_, args = plh.testdata_pre(
        >>>     'spatial_verification', defaultdb='PZ_MTEST',
        >>>     a=['default:dpername=1,qsize=1,dsize=10'],
        >>>     p=['default:K=1,fg_on=True,sqrd_dist_on=True'])
        >>> cm = args.cm_list_FILT[0]
        >>> ibs = qreq_.ibs
        >>> # Ensure there is only one aid per database name
        >>> assert isinstance(ibs, ibeis.IBEISController)
        >>> stats = ut.get_stats(ibs.get_num_annots_per_name(qreq_.daids)[0], use_nan=True)
        >>> print('per_name_stats = %s' % (ut.dict_str(stats, nl=False),))
        >>> assert stats['mean'] == 1 and stats['std'] == 0, 'this test requires one annot per name in the database'
        >>> cm.evaluate_dnids(qreq_)
        >>> cm.assert_self(qreq_)
        >>> cm._cast_scores()
        >>> # cm.fs_list = cm.fs_list.astype(np.float)
        >>> nsum_nid_list, nsum_score_list = name_scoring.compute_nsum_score(cm, qreq_)
        >>> nsum_nid_list2, nsum_score_list2, _ = name_scoring.compute_nsum_score2(cm, qreq_)
        >>> csum_score_list = scoring.compute_csum_score(cm)
        >>> vt.asserteq(nsum_score_list, csum_score_list)
        >>> vt.asserteq(nsum_score_list, csum_score_list, thresh=0, iswarning=True)
        >>> vt.asserteq(nsum_score_list2, csum_score_list, thresh=0, iswarning=True)
        >>> #assert np.allclose(nsum_score_list, csum_score_list), 'should be the same when K=1 and per_name=1'
        >>> #assert all(nsum_score_list  == csum_score_list), 'should be the same when K=1 and per_name=1'
        >>> #assert all(nsum_score_list2 == csum_score_list), 'should be the same when K=1 and per_name=1'
        >>> # Evaluate parts of the sourcecode


    Ignore:
        assert all(nsum_score_list3 == csum_score_list), 'should be the same when K=1 and per_name=1'
        fm_list = fm_list[0:1]
        fs_list = fs_list[0:1]
        featflag_list2 = featflag_list2[0:1]
        dnid_list = dnid_list[0:1]
        name_groupxs2 = name_groupxs2[0:1]
        nsum_nid_list2 = nsum_nid_list2[0:1]

    """
    #assert qreq_ is not None
    try:
        hack_single_ori =  qreq_ is not None and (qreq_.qparams.query_rotation_heuristic or qreq_.qparams.rotation_invariance)
    except AttributeError:
        hack_single_ori =  True
    # The core for each feature match
    #
    # The query feature index for each feature match
    fm_list = cm.fm_list
    fs_list = cm.get_fsv_prod_list()
    dnid_list = cm.dnid_list
    #--
    fx1_list = [fm.T[0] for fm in fm_list]
    """
    # Try a rebase?
    fx1_list = list(map(vt.compute_unique_data_ids_, fx1_list))
    """
    # Group annotation matches by name
    nsum_nid_list, name_groupxs = vt.group_indices(dnid_list)
    name_grouped_fx1_list = vt.apply_grouping_(fx1_list, name_groupxs)
    name_grouped_fs_list  = vt.apply_grouping_(fs_list,  name_groupxs)
    # Stack up all matches to a particular name
    name_grouped_fx1_flat = list(map(np.hstack, name_grouped_fx1_list))
    name_grouped_fs_flat  = list(map(np.hstack, name_grouped_fs_list))
    """
    assert np.all(name_grouped_fs_list[0][0] == fs_list[0])
    assert np.all(name_grouped_fs_flat[0] == fs_list[0])
    """
    if hack_single_ori:
        # keypoints with the same xy can only have one of them vote
        kpts1 = qreq_.ibs.get_annot_kpts(cm.qaid, config2_=qreq_.extern_query_config2)
        xys1_ = vt.get_xys(kpts1).T
        kpts_xyid_list = vt.compute_unique_arr_dataids(xys1_)
        # Make nested group for every name by query feature index (accounting for duplicate orientation)
        name_grouped_xyid_flat = [kpts_xyid_list.take(fx1) for fx1 in name_grouped_fx1_flat]
        feat_groupxs_list = [vt.group_indices(xyid_flat)[1] for xyid_flat in name_grouped_xyid_flat]
    else:
        # make unique indicies using feature indexes
        feat_groupxs_list = [vt.group_indices(fx1_flat)[1] for fx1_flat in name_grouped_fx1_flat]
    # Make nested group for every name by unique query feature index
    feat_grouped_fs_list = [[fs_flat.take(xs, axis=0) for xs in feat_groupxs]
                            for fs_flat, feat_groupxs in zip(name_grouped_fs_flat, feat_groupxs_list)]
    """
    np.array(feat_grouped_fs_list)[0].T[0] == fs_list
    """
    if False:
        valid_fs_list = [
            np.array([group.max() for group in grouped_fs])
            #np.array([group[group.argmax()] for group in grouped_fs])
            for grouped_fs in feat_grouped_fs_list
        ]
        nsum_score_list4 = np.array([valid_fs.sum() for valid_fs in valid_fs_list])  # NOQA
    # Prevent a feature from voting twice:
    # take only the max score that a query feature produced
    #name_grouped_valid_fs_list1 =[np.array([fs_group.max() for fs_group in feat_grouped_fs])
    #                            for feat_grouped_fs in feat_grouped_fs_list]
    nsum_score_list = np.array([np.sum([fs_group.max() for fs_group in feat_grouped_fs])
                                for feat_grouped_fs in feat_grouped_fs_list])
    return nsum_nid_list, nsum_score_list


@profile
def compute_nsum_score2(cm, qreq_=None):
    r"""
    Example3:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('testdb1', qaid_list=[1], cfgdict=dict(fg_on=False, query_rotation_heuristic=True))
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_)
        >>> nsum_nid_list1, nsum_score_list1, featflag_list1 = compute_nsum_score2(cm, qreq_)
        >>> nsum_nid_list2, nsum_score_list2 = compute_nsum_score(cm, qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)
    """
    featflag_list2 = get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_)
    fs_list = cm.get_fsv_prod_list()
    name_groupxs2 = cm.name_groupxs
    nsum_nid_list2 = cm.unique_nids
    #--
    valid_fs_list2 = vt.zipcompress(fs_list, featflag_list2)
    name_grouped_valid_fs_list2 = vt.apply_grouping_(valid_fs_list2,  name_groupxs2)
    nsum_score_list2 = np.array([sum(list(map(np.sum, valid_fs_group)))
                                 for valid_fs_group in name_grouped_valid_fs_list2])
    if False:
        nsum_score_list3 = np.array([  # NOQA
            np.sum([fs_group.sum() for fs_group in valid_fs_group])
            for valid_fs_group in name_grouped_valid_fs_list2])
    return nsum_nid_list2, nsum_score_list2, featflag_list2


@profile
def get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_=None):
    """
    Computes flags to desribe which features can or can not vote

    CommandLine:
        python -m ibeis.algo.hots.name_scoring --exec-get_chipmatch_namescore_nonvoting_feature_flags

    Example:
        >>> # ENABLE_DOCTEST
        >>> # FIXME: breaks when fg_on=True
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> from ibeis.algo.hots import name_scoring
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
        hack_single_ori =  qreq_ is not None and (qreq_.qparams.query_rotation_heuristic or qreq_.qparams.rotation_invariance)
    except AttributeError:
        hack_single_ori =  True
        pass
    # The core for each feature match
    fs_list = cm.get_fsv_prod_list()
    # The query feature index for each feature match
    fm_list = cm.fm_list
    kpts1 = None if not hack_single_ori else qreq_.ibs.get_annot_kpts(cm.qaid, config2_=qreq_.extern_query_config2)
    dnid_list = cm.dnid_list
    name_groupxs = cm.name_groupxs
    featflag_list = get_namescore_nonvoting_feature_flags(fm_list, fs_list, dnid_list, name_groupxs, kpts1=kpts1)
    return featflag_list


@profile
def get_namescore_nonvoting_feature_flags(fm_list, fs_list, dnid_list, name_groupxs, kpts1=None):
    r"""
    fm_list = [fm[:min(len(fm), 10)] for fm in fm_list]
    fs_list = [fs[:min(len(fs), 10)] for fs in fs_list]
    """
    fx1_list = [fm.T[0] for fm in fm_list]
    # Group annotation matches by name
    name_grouped_fx1_list = vt.apply_grouping_(fx1_list, name_groupxs)
    name_grouped_fs_list  = vt.apply_grouping_(fs_list,  name_groupxs)
    # Stack up all matches to a particular name, keep track of original indicies via offets
    name_invertable_flat_fx1_list = list(map(ut.invertible_flatten2_numpy, name_grouped_fx1_list))
    name_grouped_fx1_flat = ut.get_list_column(name_invertable_flat_fx1_list, 0)
    name_grouped_invertable_cumsum_list = ut.get_list_column(name_invertable_flat_fx1_list, 1)
    name_grouped_fs_flat = list(map(np.hstack, name_grouped_fs_list))
    if kpts1 is not None:
        xys1_ = vt.get_xys(kpts1).T
        kpts_xyid_list = vt.compute_unique_data_ids(xys1_)
        # Make nested group for every name by query feature index (accounting for duplicate orientation)
        name_grouped_xyid_flat = list(kpts_xyid_list.take(fx1) for fx1 in name_grouped_fx1_flat)
        xyid_groupxs_list = list(vt.group_indices(xyid_flat)[1] for xyid_flat in name_grouped_xyid_flat)
        name_group_fx1_groupxs_list = xyid_groupxs_list
    else:
        # Make nested group for every name by query feature index
        fx1_groupxs_list = [vt.group_indices(fx1_flat)[1] for fx1_flat in name_grouped_fx1_flat]
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
    #dtype = np.bool
    name_grouped_isvalid_flat_list = [
        vt.invert_apply_grouping2(fid_grouped_isvalid_list, fid_groupxs, dtype=np.bool)
        for fid_grouped_isvalid_list, fid_groupxs in zip(name_grouped_fid_grouped_isvalid_list, name_group_fx1_groupxs_list)
    ]

    name_grouped_isvalid_unflat_list = [
        ut.unflatten2(isvalid_flat, invertable_cumsum_list)
        for isvalid_flat, invertable_cumsum_list in zip(name_grouped_isvalid_flat_list, name_grouped_invertable_cumsum_list)
    ]

    # Reports which features were valid in name scoring for every annotation
    featflag_list = vt.invert_apply_grouping(name_grouped_isvalid_unflat_list, name_groupxs)
    return featflag_list


@profile
def align_name_scores_with_annots(annot_score_list, annot_aid_list, daid2_idx, name_groupxs, name_score_list):
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
        python -m ibeis.algo.hots.name_scoring --test-align_name_scores_with_annots
        python -m ibeis.algo.hots.name_scoring --test-align_name_scores_with_annots --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.evaluate_csum_score(qreq_)
        >>> cm.evaluate_nsum_score(qreq_)
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
        >>> assert cm.get_top_nids()[0] == cm.unique_nids[cm.nsum_score_list.argmax()], 'bug in alignment'
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_)
        >>> ut.show_if_requested()

    Example:
        >>> from ibeis.algo.hots.name_scoring import *  # NOQA
        >>> annot_score_list = []
        >>> annot_aid_list   = []
        >>> daid2_idx        = {}
        >>> # Name aligned lists
        >>> name_score_list  = np.array([], dtype=np.float32)
        >>> name_groupxs     = []
        >>> # Execute Function
        >>> score_list = align_name_scores_with_annots(annot_score_list, annot_aid_list, daid2_idx, name_groupxs, name_score_list)

    Ignore:
        dict(zip(cm.dnid_list, cm.score_list))
        dict(zip(cm.unique_nids, cm.nsum_score_list))
        np.all(nid_list == cm.unique_nids)
    """
    if len(name_groupxs) == 0:
        score_list = np.empty(0, dtype=name_score_list.dtype)
        return score_list
    else:
        # Group annot aligned indicies by nid
        annot_aid_list = np.array(annot_aid_list)
        #nid_list, groupxs  = vt.group_indices(annot_nid_list)
        grouped_scores     = vt.apply_grouping(annot_score_list, name_groupxs)
        grouped_annot_aids = vt.apply_grouping(annot_aid_list, name_groupxs)
        flat_grouped_aids  = np.hstack(grouped_annot_aids)
        #flat_groupxs  = np.hstack(name_groupxs)
        #if __debug__:
        #    sum_scores = np.array([scores.sum() for scores in grouped_scores])
        #    max_scores = np.array([scores.max() for scores in grouped_scores])
        #    assert np.all(name_score_list <= sum_scores)
        #    assert np.all(name_score_list > max_scores)
        # +------------
        # Find the position of the highest name_scoring annotation for each name
        # IN THE FLATTENED GROUPED ANNOT_AID_LIST (this was the bug)
        offset_list = np.array([annot_scores.argmax() for annot_scores in grouped_scores])
        # Find the starting position of eatch group use chain to start offsets with 0
        _padded_scores  = itertools.chain([[]], grouped_scores[:-1])
        sizeoffset_list = np.array([len(annot_scores) for annot_scores in _padded_scores])
        baseindex_list  = sizeoffset_list.cumsum()
        # Augment starting position with offset index
        annot_idx_list = np.add(baseindex_list, offset_list)
        # L______________
        best_aid_list = flat_grouped_aids[annot_idx_list]
        best_idx_list = ut.dict_take(daid2_idx, best_aid_list)
        # give the annotation domain a name score
        #score_list = np.zeros(len(annot_score_list), dtype=name_score_list.dtype)
        score_list = np.full(len(annot_score_list), fill_value=-np.inf, dtype=name_score_list.dtype)
        #score_list = np.full(len(annot_score_list), fill_value=np.nan, dtype=name_score_list.dtype)
        #score_list = np.nan(len(annot_score_list), dtype=name_score_list.dtype)
        # HACK: we need to set these to 'low' values and we also have to respect negatives
        #score_list[:] = -np.inf
        # make sure that the nid_list from group_indicies and the nids belonging to
        # name_score_list (cm.unique_nids) are in alignment
        #nidx_list = np.array(ut.dict_take(nid2_nidx, nid_list))

        # THIS ASSUMES name_score_list IS IN ALIGNMENT WITH BOTH cm.unique_nids and
        # nid_list (which should be == cm.unique_nids)
        score_list[best_idx_list] = name_score_list
        return score_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.name_scoring
        python -m ibeis.algo.hots.name_scoring --allexamples
        python -m ibeis.algo.hots.name_scoring --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
