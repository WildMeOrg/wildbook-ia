from __future__ import absolute_import, division, print_function
from six.moves import zip, range, map  # NOQA
import numpy as np
import itertools
import vtool as vt
import utool as ut
#import six
#from ibeis.model.hots import scoring
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
from collections import namedtuple
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[nscoring]', DEBUG=False)

NameScoreTup = namedtuple('NameScoreTup', ('sorted_nids', 'sorted_nscore',
                                           'sorted_aids', 'sorted_scores'))


def testdata_chipmatch():
    from ibeis.model.hots import chip_match
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
        np.array([(1,), (1,), (1,), (1,)], dtype=np.float32),
        np.array([(1,), (1,), (1,), (1,)], dtype=np.float32),
        np.array([(1,), (1,), (1,), (1,)], dtype=np.float32),
        np.array([(1,), (1,), (1,), (1,)], dtype=np.float32),
        np.array([(1,), (1,), (1,), (1,), (1, )], dtype=np.float32),
    ]
    cm = chip_match.ChipMatch2(
        qaid=1,
        daid_list=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        fm_list=fm_list,
        fsv_list=fsv_list,
        dnid_list=np.array([1, 1, 2, 2, 3], dtype=np.int32),
        fsv_col_lbls=['count'],
    )
    #print(cm.get_rawinfostr())
    #if False:
    #    # DEBUG
    #    cm.rrr()
    #    print(cm.get_rawinfostr())
    #    print(cm.get_cvs_str(ibs=qreq_.ibs, numtop=None))
    return cm


def compute_nsum_score(cm, qreq_=None):
    r"""
    nsum

    Args:
        cm (ChipMatch2):

    Returns:
        tuple: (unique_nids, nsum_score_list)

    CommandLine:
        python -m ibeis.model.hots.name_scoring --test-compute_nsum_score:0
        python -m ibeis.model.hots.name_scoring --test-compute_nsum_score:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> # build test data
        >>> cm = testdata_chipmatch()
        >>> # execute function
        >>> (unique_nids, nsum_score_list) = compute_nsum_score(cm)
        >>> result = ut.list_str((unique_nids, nsum_score_list), label_list=['unique_nids', 'nsum_score_list'])
        >>> print(result)
        unique_nids = np.array([1, 2, 3], dtype=np.int32)
        nsum_score_list = np.array([ 4.,  7.,  5.], dtype=np.float32)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_.ibs)
        >>> #cm.qnid = 1   # Hack for testdb1 names
        >>> nsum_nid_list, nsum_score_list = compute_nsum_score(cm)
        >>> assert np.all(nsum_nid_list == cm.unique_nids), 'nids out of alignment'
        >>> flags = (nsum_nid_list == cm.qnid)
        >>> max_true = nsum_score_list[flags].max()
        >>> max_false = nsum_score_list[~flags].max()
        >>> assert max_true > max_false, 'is this truely a hard case?'
        >>> assert max_true > 1.2, 'score=%r should be higher for aid=18' % (max_true,)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18], cfgdict=dict(augment_queryside_hack=True))
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> #cm.evaluate_dnids(qreq_.ibs)
        >>> #cm.qnid = 1   # Hack for testdb1 names
        >>> #nsum_nid_list, nsum_score_list = compute_nsum_score(cm, qreq_=qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)

    Example3:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('testdb1', qaid_list=[1], cfgdict=dict(augment_queryside_hack=True))
        >>> cm = cm_list[0]
        >>> cm.score_nsum(qreq_)
        >>> #cm.evaluate_dnids(qreq_.ibs)
        >>> #cm.qnid = 1   # Hack for testdb1 names
        >>> #nsum_nid_list, nsum_score_list = compute_nsum_score(cm, qreq_=qreq_)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)
    """
    HACK_SINGLE_ORI =  qreq_ is not None and (qreq_.qparams.augment_queryside_hack or qreq_.qparams.rotation_invariance)
    # The core for each feature match
    qaid = cm.qaid
    dnid_list = cm.dnid_list
    fs_list = cm.get_fsv_prod_list()
    fm_list = cm.fm_list
    #
    # The query feature index for each feature match
    fx1_list = [fm.T[0] for fm in fm_list]
    # Group annotation matches by name
    nsum_nid_list, name_groupxs = vt.group_indices(dnid_list)
    name_grouped_fx1_list = vt.apply_grouping_(fx1_list, name_groupxs)
    name_grouped_fs_list  = vt.apply_grouping_(fs_list,  name_groupxs)
    # Stack up all matches to a particular name
    name_grouped_fx1_flat = (map(np.hstack, name_grouped_fx1_list))
    name_grouped_fs_flat  = (map(np.hstack, name_grouped_fs_list))
    if HACK_SINGLE_ORI:
        # keypoints with the same xy can only have one of them vote
        kpts1 = qreq_.ibs.get_annot_kpts(qaid, config2_=qreq_.get_external_query_config2())
        xys1_ = vt.get_xys(kpts1).T
        kpts_xyid_list = vt.compute_unique_data_ids(xys1_)
        # Make nested group for every name by query feature index (accounting for duplicate orientation)
        name_grouped_xyid_flat = (kpts_xyid_list.take(fx1) for fx1 in name_grouped_fx1_flat)
        xyid_groupxs_list = (vt.group_indices(xyid_flat)[1] for xyid_flat in name_grouped_xyid_flat)
        feat_grouped_fs_list = list(
            vt.apply_grouping(fs_flat, xyid_groupxs)
            for fs_flat, xyid_groupxs in zip(name_grouped_fs_flat, xyid_groupxs_list)
        )
    else:
        # Make nested group for every name by query feature index
        fx1_groupxs_list = (vt.group_indices(fx1_flat)[1] for fx1_flat in name_grouped_fx1_flat)
        feat_grouped_fs_list = list(
            vt.apply_grouping(fs_flat, fx1_groupxs)
            for fs_flat, fx1_groupxs in zip(name_grouped_fs_flat, fx1_groupxs_list)
        )
    # Prevent a feature from voting twice:
    # take only the max score that a query feature produced
    best_fs_list = list(
        np.array([fs_group.max() for fs_group in feat_grouped_fs])
        for feat_grouped_fs in feat_grouped_fs_list
    )
    nsum_score_list = np.array([fs.sum() for fs in best_fs_list])
    # DO NOT Return sorted by the name score
    #name_score_sortx = nsum_score_list.argsort()[::-1]
    #nsum_score_list = nsum_score_list.take(name_score_sortx)
    #nsum_nid_list   = nsum_nid_list.take(name_score_sortx)
    return nsum_nid_list, nsum_score_list


def compute_nsum_score2(cm, qreq_=None):
    r"""
    Example3:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('testdb1', qaid_list=[1])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('testdb1', qaid_list=[1], cfgdict=dict(augment_queryside_hack=True))
        >>> cm = cm_list[0]
        >>> cm.evaluate_dnids(qreq_.ibs)
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_, ori=True)
    """
    featflag_list = get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_)
    fs_list = cm.get_fsv_prod_list()
    name_groupxs = cm.name_groupxs
    valid_fs_list = vt.zipcompress(fs_list, featflag_list)
    name_grouped_valid_fs_list = vt.apply_grouping_(valid_fs_list,  name_groupxs)
    nsum_nid_list = cm.dnid_list
    nsum_score_list = [sum(list(map(np.sum, valid_fs_group))) for valid_fs_group in name_grouped_valid_fs_list]
    return nsum_nid_list, nsum_score_list, featflag_list


def get_chipmatch_namescore_nonvoting_feature_flags(cm, qreq_=None):
    """

    CommandLine:
        python -m ibeis.model.hots.name_scoring --exec-get_chipmatch_namescore_nonvoting_feature_flags

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> from ibeis.model.hots import name_scoring
        >>> from ibeis.model.hots import scoring
        >>> # Test to make sure name score and chips score are equal when per_name=1
        >>> ibs, qreq_list, cms_list = plh.testdata_pre_sver2('PZ_MTEST', ['default:dpername=1,qsize=1,dsize=10'], ['default:K=1'])
        >>> qreq_, cm_list = qreq_list[0], cms_list[0]
        >>> cm = cm_list[0]
        >>> # Ensure there is only one aid per database name
        >>> stats = ibs.get_annot_per_name_stats(qreq_.get_external_daids())
        >>> print('per_name_stats = %s' % (ut.dict_str(stats, nl=False),))
        >>> assert stats['mean'] == 1 and stats['std'] == 0, 'this test requires one annot per name in the database'
        >>> cm.evaluate_dnids(qreq_.ibs)
        >>> cm.assert_self()
        >>> nsum_nid_list, nsum_score = name_scoring.compute_nsum_score(cm)
        >>> csum_score = scoring.compute_csum_score(cm)
        >>> assert all(nsum_score == csum_score), 'should be the same when K=1 and per_name=1'
        >>> # Evaluate parts of the sourcecode

        #>>> func1 = name_scoring.get_chipmatch_namescore_nonvoting_feature_flags
        #>>> func2 = name_scoring.get_namescore_nonvoting_feature_flags
        #>>> sourcecode1 = ut.get_func_sourcecode(func1, stripdef=True, stripret=True, strip_docstr=True, remove_linenums=[-1, -2])
        #>>> locals_ = locals()
        #>>> globals_ = globals()
        #>>> six.exec_(sourcecode1, globals_, locals_)
        #>>> sourcecode2 = ut.get_func_sourcecode(func2, stripdef=True, stripret=True, strip_docstr=True)
        #>>> six.exec_(sourcecode2, globals_, locals_)
        #assert locals_['kpts1'] is None
        #locals_['featflag_list']
    """
    HACK_SINGLE_ORI =  qreq_ is not None and (qreq_.qparams.augment_queryside_hack or qreq_.qparams.rotation_invariance)
    # The core for each feature match
    fs_list = cm.get_fsv_prod_list()
    # The query feature index for each feature match
    fm_list = cm.fm_list
    kpts1 = None if not HACK_SINGLE_ORI else qreq_.ibs.get_annot_kpts(cm.qaid, config2_=qreq_.get_external_query_config2())
    dnid_list = cm.dnid_list
    name_groupxs = cm.name_groupxs
    featflag_list = get_namescore_nonvoting_feature_flags(fm_list, fs_list, dnid_list, name_groupxs, kpts1=kpts1)
    return featflag_list


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
        python -m ibeis.model.hots.name_scoring --test-align_name_scores_with_annots
        python -m ibeis.model.hots.name_scoring --test-align_name_scores_with_annots --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> cm.evaluate_csum_score(qreq_)
        >>> cm.evaluate_nsum_score(qreq_)
        >>> # Annot aligned lists
        >>> annot_score_list = cm.csum_score_list
        >>> annot_aid_list   = cm.daid_list
        >>> daid2_idx        = cm.daid2_idx
        >>> # Name aligned lists
        >>> name_score_list  = cm.nsum_score_list
        >>> name_groupxs     = cm.name_groupxs
        >>> # Execute Function
        >>> score_list = align_name_scores_with_annots(annot_score_list, annot_aid_list, daid2_idx, name_groupxs, name_score_list)
        >>> target = name_score_list[cm.nid2_nidx[cm.qnid]]
        >>> # Check Results
        >>> test_index = np.where(score_list == target)[0][0]
        >>> cm.score_list = score_list
        >>> ut.assert_eq(ibs.get_annot_name_rowids(cm.daid_list[test_index]), cm.qnid)
        >>> assert ut.isunique(cm.dnid_list[score_list > 0]), 'bad name score'
        >>> assert cm.get_top_nids()[0] == cm.unique_nids[cm.nsum_score_list.argmax()], 'bug in alignment'
        >>> ut.quit_if_noshow()
        >>> cm.show_ranked_matches(qreq_)

    Example:
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
        score_list = np.zeros(len(annot_score_list), dtype=name_score_list.dtype)
        # make sure that the nid_list from group_indicies and the nids belonging to
        # name_score_list (cm.unique_nids) are in alignment
        #nidx_list = np.array(ut.dict_take(nid2_nidx, nid_list))

        # THIS ASSUMES name_score_list IS IN ALIGNMENT WITH BOTH cm.unique_nids and
        # nid_list (which should be == cm.unique_nids)
        score_list[best_idx_list] = name_score_list
        return score_list


#def get_best_annot_per_name_indices(cm):
#    grouped_scores = vt.apply_grouping(cm.annot_score_list, cm.name_groupxs)
#    # Find the position of the highest name_scoring annotation for each name
#    offset_list = np.array([annot_scores.argmax() for annot_scores in grouped_scores])
#    # Find the starting position of eatch group use chain to start offsets with 0
#    _padded_scores  = itertools.chain([[]], grouped_scores[:-1])
#    sizeoffset_list = np.array([len(annot_scores) for annot_scores in _padded_scores])
#    baseindex_list  = sizeoffset_list.cumsum()
#    # Augment starting position with offset index
#    annot_idx_list = np.add(baseindex_list, offset_list)


def group_scores_by_name(ibs, aid_list, score_list):
    r"""
    Converts annotation scores to name scores.
    Over multiple annotations finds keypoints best match and uses that score.

    CommandLine:
        python -m ibeis.model.hots.name_scoring --test-group_scores_by_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *   # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import results_all
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

    TODO:
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

        Timeit::
            import ibeis
            ibs = ibeis.opendb('PZ_MTEST')
            aid_list = ibs.get_valid_aids()
            aid_arr = np.array(aid_list)
            %timeit ibs.get_annot_name_rowids(aid_list)
            %timeit ibs.get_annot_name_rowids(aid_arr)


    """
    assert len(score_list) == len(aid_list), 'scores and aids must be associated'
    score_arr = np.array(score_list)
    nid_list  = np.array(ibs.get_annot_name_rowids(aid_list))
    aid_list  = np.array(aid_list)
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
