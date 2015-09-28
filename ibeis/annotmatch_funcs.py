# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import vtool as vt
from six.moves import zip, map
import numpy as np
from ibeis.control import controller_inject
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[annotmatch_funcs]')

# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)


def setup_pzmtest_subgraph(ibs):
    import ibeis
    ibs = ibeis.opendb(db='PZ_MTEST')
    #nids = ibs.get_valid_nids()
    #aids = ibs.get_name_aids(nids)

    rowids = ibs._get_all_annotmatch_rowids()
    aids1 = ibs.get_annotmatch_aid1(rowids)
    aids2 = ibs.get_annotmatch_aid2(rowids)

    for aid1, aid2 in zip(aids1, aids2):
        ibs.mark_annot_pair_as_positive_match(aid1, aid2)
        ibs.mark_annot_pair_as_positive_match(aid2, aid1)


def get_annotmatch_subgraph(ibs):
    r"""

    http://bokeh.pydata.org/en/latest/
    https://github.com/jsexauer/networkx_viewer

    TODO: Need a special visualization
        In the web I need:
            * graph of annotations matches.
            * can move them around.
            * edit lines between them.
            * http://stackoverflow.com/questions/15373530/web-graph-visualization-tool

            This should  share functionality with a name view.

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_subgraph

        # Networkx example
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:0 --show

    Ignore:

        from ibeis import viz

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> result = get_annotmatch_subgraph(ibs)
        >>> ut.show_if_requested()
    """
    #import ibeis
    #ibs = ibeis.opendb(db='PZ_MTEST')
    #rowids = ibs._get_all_annotmatch_rowids()
    #aids1 = ibs.get_annotmatch_aid1(rowids)
    #aids2 = ibs.get_annotmatch_aid2(rowids)

    #
    #
    nids = ibs.get_valid_nids()
    nids = nids[0:5]
    aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)

    # Enumerate annotmatch properties
    rng = np.random.RandomState(0)
    edge_props = {
        'weight': rng.rand(len(aids1)),
        'reviewer_confidence': rng.rand(len(aids1)),
        'algo_confidence': rng.rand(len(aids1)),
    }

    # Remove data that does not need to be visualized
    # (dont show all the aids if you dont have to)
    thresh = .5
    flags = edge_props['weight'] > thresh
    aids1 = ut.list_compress(aids1, flags)
    aids2 = ut.list_compress(aids2, flags)
    edge_props = {key: ut.list_compress(val, flags) for key, val in edge_props.items()}

    edge_keys = list(edge_props.keys())
    edge_vals = ut.dict_take(edge_props, edge_keys)

    unique_aids = list(set(aids1 + aids2))

    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aids1, aids2, *edge_vals))
    node_lbls = [('aid', 'int')]
    edge_lbls = [('weight', 'float')]
    from ibeis.viz import viz_graph
    netx_graph = viz_graph.make_netx_graph(nodes, edges, node_lbls, edge_lbls)
    fnum = None
    #zoom = kwargs.get('zoom', .4)
    zoom = .4
    viz_graph.viz_netx_chipgraph(ibs, netx_graph, fnum=fnum, with_images=True, zoom=zoom)


@register_ibs_method
def mark_annot_pair_as_reviewed(ibs, aid1, aid2):
    """ denote that this match was reviewed and keep whatever status it is given """
    isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
    if isunknown1 or isunknown2:
        truth = ibs.const.TRUTH_UNKNOWN
    else:
        nid1, nid2 = ibs.get_annot_name_rowids((aid1, aid2))
        truth = ibs.const.TRUTH_UNKNOWN if (nid1 == nid2) else ibs.const.TRUTH_NOT_MATCH
    ibs.add_or_update_annotmatch(aid1, aid2, truth, 1.0)


@register_ibs_method
def add_or_update_annotmatch(ibs, aid1, aid2, truth, confidence):
    annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])[0]
    # TODO: sql add or update?
    if annotmatch_rowid is not None:
        ibs.set_annotmatch_truth([annotmatch_rowid], [confidence])
        ibs.set_annotmatch_confidence([annotmatch_rowid], [confidence])
    else:
        ibs.add_annotmatch([aid1], [aid2], annotmatch_truth_list=[truth], annotmatch_confidence_list=[confidence])


@register_ibs_method
def mark_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun=False, on_nontrivial_merge=None):
    """
    Safe way to perform links. Errors on invalid operations.

    TODO: ELEVATE THIS FUNCTION
    Change into make_task_mark_annot_pair_as_positive_match and it returns what
    needs to be done.

    Need to test several cases:
        uknown, unknown
        knownA, knownA
        knownB, knownA
        unknown, knownA
        knownA, unknown

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  query annotation id
        aid2 (int):  matching annotation id

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-mark_annot_pair_as_positive_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> status = mark_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(status)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        if not ut.QUIET:
            print('... _set_annot_name_rowids(aids=%r, nids=%r)' % (aid_list, nid_list))
            print('... names = %r' % (ibs.get_name_texts(nid_list)))
        assert len(aid_list) == len(nid_list), 'list must correspond'
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            #ibs.add_or_update_annotmatch(aid1, aid2, const.TRUTH_MATCH, [1.0])
        # Return the new annots in this name
        _aids_list = ibs.get_name_aids(nid_list)
        _combo_aids_list = [_aids + [aid] for _aids, aid, in zip(_aids_list, aid_list)]
        status = _combo_aids_list
        return status
    print('[marking_match] aid1 = %r, aid2 = %r' % (aid1, aid2))

    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('...images already matched')
        status = None
        ibs.mark_annot_pair_as_reviewed(aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...match unknown1 to unknown2 into 1 new name')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids * 2)
        elif not isunknown1 and not isunknown2:
            print('...merge known1 into known2')
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
            trivial_merge = len(aid1_and_groundtruth) == 1 and len(aid2_and_groundtruth) == 1
            if not trivial_merge:
                if on_nontrivial_merge is None:
                    raise Exception('no function is set up to handle nontrivial merges!')
                else:
                    on_nontrivial_merge(ibs, aid1, aid2)
            status =  _set_annot_name_rowids(aid1_and_groundtruth, [nid2] * len(aid1_and_groundtruth))
        elif isunknown2 and not isunknown1:
            print('...match unknown2 into known1')
            status =  _set_annot_name_rowids([aid2], [nid1])
        elif isunknown1 and not isunknown2:
            print('...match unknown1 into known2')
            status =  _set_annot_name_rowids([aid1], [nid2])
        else:
            raise AssertionError('impossible state')
    return status


@register_ibs_method
def mark_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun=False, on_nontrivial_split=None):
    """
    TODO: ELEVATE THIS FUNCTION

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        dryrun (bool):

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-mark_annot_pair_as_negative_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> # execute function
        >>> result = mark_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(result)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        print('... _set_annot_name_rowids(%r, %r)' % (aid_list, nid_list))
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            #ibs.add_or_update_annotmatch(aid1, aid2, const.TRUTH_NOT_MATCH, [1.0])
    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('images are marked as having the same name... we must tread carefully')
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        if len(aid1_groundtruth) == 1 and aid1_groundtruth == [aid2]:
            # this is the only safe case for same name split
            # Change so the names are not the same
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            if on_nontrivial_split is None:
                raise Exception('no function is set up to handle nontrivial splits!')
            else:
                on_nontrivial_split(ibs, aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...nonmatch unknown1 and unknown2 into 2 new names')
            next_nids = ibs.make_next_nids(num=2)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids)
        elif not isunknown1 and not isunknown2:
            print('...nonmatch known1 and known2... nothing to do (yet)')
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            status = None
        elif isunknown2 and not isunknown1:
            print('...nonmatch unknown2 -> newname and known1')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid2], next_nids)
        elif isunknown1 and not isunknown2:
            print('...nonmatch unknown1 -> newname and known2')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            raise AssertionError('impossible state')
    return status


@register_ibs_method
def get_annot_pair_timdelta(ibs, aid_list1, aid_list2):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    Returns:
        list: timedelta_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-get_annot_pair_timdelta

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> from six.moves import filter
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids(hasgt=True)
        >>> unixtimes = ibs.get_annot_image_unixtimes(aid_list)
        >>> aid_list = ut.list_compress(aid_list, np.array(unixtimes) != -1)
        >>> gt_aids_list = ibs.get_annot_groundtruth(aid_list, daid_list=aid_list)
        >>> aid_list1 = [aid for aid, gt_aids in zip(aid_list, gt_aids_list) if len(gt_aids) > 0][0:5]
        >>> aid_list2 = [gt_aids[0] for gt_aids in gt_aids_list if len(gt_aids) > 0][0:5]
        >>> timedelta_list = ibs.get_annot_pair_timdelta(aid_list1, aid_list2)
        >>> result = ut.numpy_str(timedelta_list, precision=2)
        >>> print(result)
        np.array([  7.57e+07,   7.57e+07,   2.41e+06,   1.98e+08,   9.69e+07], dtype=np.float64)

    """
    #unixtime_list1 = np.array(ibs.get_annot_image_unixtimes(aid_list1), dtype=np.float)
    #unixtime_list2 = np.array(ibs.get_annot_image_unixtimes(aid_list2), dtype=np.float)
    unixtime_list1 = ibs.get_annot_image_unixtimes_asfloat(aid_list1)
    unixtime_list2 = ibs.get_annot_image_unixtimes_asfloat(aid_list2)
    #unixtime_list1[unixtime_list1 == -1] = np.nan
    #unixtime_list2[unixtime_list2 == -1] = np.nan
    timedelta_list = np.abs(unixtime_list1 - unixtime_list2)
    return timedelta_list


# AUTOGENED CONSTANTS:


@register_ibs_method
def get_annot_has_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    num_reviewed_list = ibs.get_annot_num_reviewed_matching_aids(aid_list)
    has_reviewed_list = [num_reviewed > 0 for num_reviewed in num_reviewed_list]
    return has_reviewed_list


@register_ibs_method
def get_annot_num_reviewed_matching_aids(ibs, aid1_list, eager=True, nInput=None):
    r"""
    Args:
        aid_list (int):  list of annotation ids
        eager (bool):
        nInput (None):

    Returns:
        list: num_annot_reviewed_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-get_annot_num_reviewed_matching_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid1_list = ibs.get_valid_aids()
        >>> eager = True
        >>> nInput = None
        >>> # execute function
        >>> num_annot_reviewed_list = get_annot_num_reviewed_matching_aids(ibs, aid_list, eager, nInput)
        >>> # verify results
        >>> result = str(num_annot_reviewed_list)
        >>> print(result)
    """
    aids_list = ibs.get_annot_reviewed_matching_aids(aid1_list, eager=eager, nInput=nInput)
    num_annot_reviewed_list = list(map(len, aids_list))
    return num_annot_reviewed_list


def get_annotmatch_rowids_from_aid1(ibs, aid1_list, eager=True, nInput=None):
    """
    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    """
    ANNOT_ROWID1 = 'annot_rowid1'
    params_iter = [(aid1,) for aid1 in aid1_list]
    colnames = ('annotmatch_rowid',)
    andwhere_colnames = (ANNOT_ROWID1,)
    annotmach_rowid_list = ibs.db.get_where2(
        ibs.const.ANNOTMATCH_TABLE, colnames, params_iter,
        andwhere_colnames=andwhere_colnames, eager=eager, unpack_scalars=False, nInput=nInput)
    return annotmach_rowid_list


@register_ibs_method
def get_annot_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    """
    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    """
    ANNOT_ROWID1 = 'annot_rowid1'
    ANNOT_ROWID2 = 'annot_rowid2'
    #params_iter = [(aid, aid) for aid in aid_list]
    #[(aid, aid) for aid in aid_list]
    #colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    #where_colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    params_iter = [(aid,) for aid in aid_list]
    colnames = (ANNOT_ROWID2,)
    andwhere_colnames = (ANNOT_ROWID1,)
    aids_list = ibs.db.get_where2(ibs.const.ANNOTMATCH_TABLE, colnames,
                                  params_iter,
                                  andwhere_colnames=andwhere_colnames,
                                  eager=eager, unpack_scalars=False,
                                  nInput=nInput)
    #logicop = 'OR'
    #aids_list = ibs.db.get_where3(
    #    const.ANNOTMATCH_TABLE, colnames, params_iter,
    #    where_colnames=where_colnames, logicop=logicop, eager=eager,
    #    unpack_scalars=False, nInput=nInput)
    return aids_list


@register_ibs_method
def get_annot_pair_truth(ibs, aid1_list, aid2_list):
    """
    CAREFUL: uses annot match table for truth, so only works if reviews have happend
    """
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    annotmatch_truth_list = ibs.get_annotmatch_truth(annotmatch_rowid_list)
    return annotmatch_truth_list


@register_ibs_method
def get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list):
    r"""
    Args:
        aid1_list (list):
        aid2_list (list):

    Returns:
        list: annotmatch_reviewed_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-get_annot_pair_is_reviewed

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> pairs = list(ut.product(aid_list, aid_list))
        >>> aid1_list = ut.get_list_column(pairs, 0)
        >>> aid2_list = ut.get_list_column(pairs, 1)
        >>> # execute function
        >>> annotmatch_reviewed_list = get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list)
        >>> # verify results
        >>> reviewed_pairs = ut.list_compress(pairs, annotmatch_reviewed_list)
        >>> result = len(reviewed_pairs)
        >>> print(result)
        104
    """
    flag_non_None_items = lambda list_: (item_ is not None for item_ in list_)
    annotmatch_truth_list1 = ibs.get_annot_pair_truth(aid1_list, aid2_list)
    annotmatch_truth_list2 = ibs.get_annot_pair_truth(aid2_list, aid1_list)
    annotmatch_truth_list = ut.or_lists(
        flag_non_None_items(annotmatch_truth_list1),
        flag_non_None_items(annotmatch_truth_list2))
    #annotmatch_reviewed_list = [truth is not None for truth in annotmatch_truth_list]
    return annotmatch_truth_list


ANNOTMATCH_PROPS_STANDARD = [
    'SceneryMatch',
    'Photobomb',
    'Hard',
    'NonDistinct',
]

ANNOTMATCH_PROPS_OTHER = [
    'Occlusion',
    'Viewpoint',
    'Pose',
    'Lighting',
    'Quality',  # quality causes failure

    'Orientation',  # orientation caused failure

    'EdgeMatch',  # descriptors on the edge of the naimal produce strong matches
    'Interesting',   # flag a case as interesting

    'JoinCase',  # case should actually be marked as correct
    'SplitCase',  # case should actually be marked as correct

    'random',  # gf case has random matches, the gt is to blame

    'BadShoulder',  # gf is a bad shoulder match
    'BadTail',  # gf is a bad tail match
]

OLD_ANNOTMATCH_PROPS = [
    'TooLargeMatches',  # really big nondistinct matches
    'TooSmallMatches',  # really big nondistinct matches
    'ScoringIssue',  # matches should be scored differently
    'BadCoverage',  # matches were not in good places (missing matches)
    'ViewpointOMG',  # gf is a bad tail match
    'ViewpointCanDo',  # gf is a bad tail match
    'shouldhavemore',
    'Shadowing',  # shadow causes failure
    'success',  # A good success case
    'GoodCoverage',  # matches were spread out correctly (scoring may be off though)
]

# Changes to prop names
PROP_MAPPING = {
    'ViewpointCanDo' : 'Correctable',
    'ViewpointOMG'   : 'Uncorrectable',

    'Shadowing'      : 'Lighting',

    'success': None,
    'GoodCoverage':  None,

    'Hard'           : 'NeedsWork',
    'shouldhavemore' : 'NeedsWork',
    'BadCoverage'    : 'NeedsWork',
    'ScoringIssue'   : 'NeedsWork',

    'TooSmallMatches' : 'FeatureScale',
    'TooLargeMatches' : 'FeatureScale',

    #'BadShoulder' : 'BadShoulder',
    #'GoodCoverage': None,
}

for key, val in PROP_MAPPING.items():
    if key in ANNOTMATCH_PROPS_OTHER:
        ANNOTMATCH_PROPS_OTHER.remove(key)
    if val is not None and val not in ANNOTMATCH_PROPS_OTHER:
        ANNOTMATCH_PROPS_OTHER.append(val)

ANNOTMATCH_PROPS_OTHER_SET = set([_.lower() for _ in ANNOTMATCH_PROPS_OTHER])
ANNOTMATCH_PROPS_OLD_SET = set([_.lower() for _ in OLD_ANNOTMATCH_PROPS])
ANNOTMATCH_PROPS_STANDARD_SET = set([_.lower() for _ in ANNOTMATCH_PROPS_STANDARD])


def consolodate_tags(case_tags):
    remove_tags = [
        'needswork',
        'correctable',
        'uncorrectable',
        'interesting',
        'splitcase',
        'joincase',
        'orientation',
        'random',
        #'badtail', 'badshoulder', 'splitcase', 'joincase', 'goodcoverage', 'interesting', 'hard'
    ]
    tags_dict = {
        #'quality': 'Quality',
        #'scoringissue': 'ScoringIssue',
        #'orientation': 'Orientation',
        #'pose': 'Pose',
        #'lighting': 'Lighting',
        #'occlusion': 'Occlusion',
        #'featurescale': 'FeatureScale',
        #'edgematch': 'EdgeMatches',
        'featurescale': 'Pose',
        'edgematch': 'Pose',
        'badtail': 'NonDistinct',
        'badshoulder': 'NonDistinct',
        #'toolargematches': 'CoarseFeatures',
        #'badcoverage': 'LowCoverage',
        #'shouldhavemore': 'LowCoverage',
        #'viewpoint': 'Viewpoint',
    }

    def filter_tags(tags):
        return [t for t in tags if t.lower() not in remove_tags]

    def map_tags(tags):
        return [tags_dict.get(t.lower(), t) for t in tags]

    def cap_tags(tags):
        return [t[0].upper() + t[1:] for t in tags]

    filtered_tags = list(map(filter_tags, case_tags))
    mapped_tags = list(map(map_tags, filtered_tags))
    unique_tags = list(map(ut.unique_keep_order2,  mapped_tags))
    new_tags = list(map(cap_tags, unique_tags))

    return new_tags


def rename_and_reduce_tags(ibs, annotmatch_rowids):
    """
    Script to update tags to newest values

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-rename_and_reduce_tags --db PZ_Master1

    Example:
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> #ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> annotmatch_rowids = filter_annotmatch_by_tags(ibs, min_num=1)
        >>> rename_and_reduce_tags(ibs, annotmatch_rowids)
    """

    tags_list_ = get_annotmatch_case_tags(ibs, annotmatch_rowids)
    def fix_tags(tags):
        return {ibs.const.__STR__(t.lower()) for t in tags}
    tags_list = list(map(fix_tags, tags_list_))

    prop_mapping = {
        ibs.const.__STR__(key.lower()): val
        for key, val in six.iteritems(PROP_MAPPING)
    }

    bad_tags = fix_tags(prop_mapping.keys())

    for rowid, tags in zip(annotmatch_rowids, tags_list):
        old_tags = tags.intersection(bad_tags)
        for tag in old_tags:
            ibs.set_annotmatch_prop(tag, [rowid], [False])
        new_tags = ut.dict_take(prop_mapping, old_tags)
        for tag in new_tags:
            if tag is not None:
                ibs.set_annotmatch_prop(tag, [rowid], [True])


def get_cate_categories():
    standard = ANNOTMATCH_PROPS_STANDARD
    other = ANNOTMATCH_PROPS_OTHER
    #case_list = standard + other
    return standard, other


def export_tagged_chips(ibs, aid_list):
    """
    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-export_tagged_chips  --tags Hard interesting --db PZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> any_tags = ut.get_argval('--tags', type_=list, default=None)
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> aid_pairs = filter_aidpairs_by_tags(ibs, any_tags=any_tags, min_num=1)
        >>> aid_list = np.unique(aid_pairs.flatten())
        >>> print(aid_list)
        >>> print('len(aid_list) = %r' % (len(aid_list),))
        >>> export_tagged_chips(ibs, aid_list)
    """
    visual_uuid_hashid = ibs.get_annot_hashid_visual_uuid(aid_list, _new=True)
    zip_fpath = 'exported_chips' +  visual_uuid_hashid + '.zip'
    chip_fpath = ibs.get_annot_chip_fpath(aid_list)
    ut.archive_files(zip_fpath, chip_fpath)


def get_aidpair_tags(ibs, aid1_list, aid2_list, directed=True):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        aid2_list (list):
        directed (bool): (default = True)

    Returns:
        list: tags_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_aidpair_tags --db PZ_Master1 --tags Hard interesting

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> any_tags = ut.get_argval('--tags', type_=list, default=None)
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> aid_pairs = filter_aidpairs_by_tags(ibs, any_tags=any_tags, min_num=1)
        >>> aid1_list = aid_pairs.T[0]
        >>> aid2_list = aid_pairs.T[1]
        >>> undirected_tags = get_aidpair_tags(ibs, aid1_list, aid2_list, directed=False)
        >>> tagged_pairs = list(zip(aid_pairs.tolist(), undirected_tags))
        >>> print(ut.list_str(tagged_pairs))
        >>> print(ut.dict_str(ut.groupby_tags(tagged_pairs, undirected_tags), nl=2))
    """
    aid_pairs = np.vstack([aid1_list, aid2_list]).T
    if directed:
        annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey(aid_pairs.T[0], aid_pairs.T[1])
        tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowid)
    else:
        expanded_aid_pairs = np.vstack([aid_pairs, aid_pairs[:, ::-1]])
        expanded_annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey(expanded_aid_pairs.T[0], expanded_aid_pairs.T[1])
        expanded_edgeids = vt.get_undirected_edge_ids(expanded_aid_pairs)
        unique_edgeids, groupxs = vt.group_indices(expanded_edgeids)
        expanded_tags_list = ibs.get_annotmatch_case_tags(expanded_annotmatch_rowid)
        grouped_tags = vt.apply_grouping(np.array(expanded_tags_list, dtype=object), groupxs)
        undirected_tags = [list(set(ut.flatten(tags))) for tags in grouped_tags]
        edgeid2_tags = dict(zip(unique_edgeids, undirected_tags))
        input_edgeids = expanded_edgeids[:len(aid_pairs)]
        tags_list = ut.dict_take(edgeid2_tags, input_edgeids)
    return tags_list


@register_ibs_method
def filter_aidpairs_by_tags(ibs, any_tags=None, all_tags=None, min_num=None, max_num=None):
    """
    list(zip(aid_pairs, undirected_tags))
    """
    filtered_annotmatch_rowids = filter_annotmatch_by_tags(
        ibs, None, any_tags=any_tags, all_tags=all_tags, min_num=min_num,
        max_num=max_num)
    aid1_list = np.array(ibs.get_annotmatch_aid1(filtered_annotmatch_rowids))
    aid2_list = np.array(ibs.get_annotmatch_aid2(filtered_annotmatch_rowids))
    aid_pairs = np.vstack([aid1_list, aid2_list]).T
    # Dont double count
    vt.get_undirected_edge_ids(aid_pairs)
    xs = vt.find_best_undirected_edge_indexes(aid_pairs)
    aid1_list = aid1_list.take(xs)
    aid2_list = aid2_list.take(xs)
    aid_pairs = np.vstack([aid1_list, aid2_list]).T
    return aid_pairs
    #directed_tags = get_aidpair_tags(ibs, aid_pairs.T[0], aid_pairs.T[1], directed=True)
    #valid_tags_list = ibs.get_annotmatch_case_tags(filtered_annotmatch_rowids)


def filter_annotmatch_by_tags(ibs, annotmatch_rowids=None, any_tags=None,
                              all_tags=None, min_num=None, max_num=None):
    r"""
    ignores case

    Args:
        ibs (IBEISController):  ibeis controller object
        flags (?):

    Returns:
        list

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-filter_annotmatch_by_tags --show
        python -m ibeis.annotmatch_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --min-num=1
        python -m ibeis.annotmatch_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags JoinCase
        python -m ibeis.annotmatch_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags SplitCase
        python -m ibeis.annotmatch_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags occlusion
        python -m ibeis.annotmatch_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags viewpoint

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> #ibs = ibeis.opendb(defaultdb='testdb1')
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> #tags = ['Photobomb', 'SceneryMatch']
        >>> any_tags = ut.get_argval('--tags', type_=list, default=['SceneryMatch', 'Photobomb'])
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> prop = any_tags[0]
        >>> filtered_annotmatch_rowids = filter_annotmatch_by_tags(ibs, None, any_tags=any_tags, min_num=min_num)
        >>> aid1_list = np.array(ibs.get_annotmatch_aid1(filtered_annotmatch_rowids))
        >>> aid2_list = np.array(ibs.get_annotmatch_aid2(filtered_annotmatch_rowids))
        >>> aid_pairs = np.vstack([aid1_list, aid2_list]).T
        >>> # Dont double count
        >>> xs = vt.find_best_undirected_edge_indexes(aid_pairs)
        >>> aid1_list = aid1_list.take(xs)
        >>> aid2_list = aid2_list.take(xs)
        >>> valid_tags_list = ibs.get_annotmatch_case_tags(filtered_annotmatch_rowids)
        >>> print('valid_tags_list = %s' % (ut.list_str(valid_tags_list, nl=1),))
        >>> #
        >>> print('Aid pairs with any_tags=%s' % (any_tags,))
        >>> print('Aid pairs with min_num=%s' % (min_num,))
        >>> print('aid_pairs = ' + ut.list_str(list(zip(aid1_list, aid2_list))))

        #>>> #timedelta_list = get_annot_pair_timdelta(ibs, aid1_list, aid2_list)
        #>>> #ut.quit_if_noshow()
        #>>> #import plottool as pt
        #>>> #pt.draw_timedelta_pie(timedelta_list, label='timestamp of tags=%r' % (tags,))
        #>>> #ut.show_if_requested()
    """

    def fix_tags(tags):
        return {ibs.const.__STR__(t.lower()) for t in tags}

    if annotmatch_rowids is None:
        annotmatch_rowids = ibs._get_all_annotmatch_rowids()

    tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowids)
    tags_list = [fix_tags(tags_) for tags_ in tags_list]

    annotmatch_rowids_ = annotmatch_rowids
    tags_list_ = tags_list

    if min_num is not None:
        flags = [len(tags_) >= min_num for tags_ in tags_list_]
        annotmatch_rowids_ = ut.list_compress(annotmatch_rowids_, flags)
        tags_list_ = ut.list_compress(tags_list_, flags)

    if max_num is not None:
        flags = [len(tags_) <= max_num for tags_ in tags_list_]
        annotmatch_rowids_ = ut.list_compress(annotmatch_rowids_, flags)
        tags_list_ = ut.list_compress(tags_list_, flags)

    if any_tags is not None:
        any_tags = fix_tags(set(ut.ensure_iterable(any_tags)))
        flags = [len(any_tags.intersection(tags_)) > 0 for tags_ in tags_list_]
        annotmatch_rowids_ = ut.list_compress(annotmatch_rowids_, flags)
        tags_list_ = ut.list_compress(tags_list_, flags)

    if all_tags is not None:
        all_tags = fix_tags(set(ut.ensure_iterable(all_tags)))
        flags = [len(all_tags.intersection(tags_)) == len(all_tags) for tags_ in tags_list_]
        annotmatch_rowids_ = ut.list_compress(annotmatch_rowids_, flags)
        tags_list_ = ut.list_compress(tags_list_, flags)

    filtered_annotmatch_rowids = annotmatch_rowids_
    return filtered_annotmatch_rowids

    #case_list = get_cate_categories()

    #for case in case_list:
    #    flag_list = ibs.get_annotmatch_prop(case, annotmatch_rowids)


@register_ibs_method
def get_annotmatch_case_tags(ibs, annotmatch_rowids):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        annotmatch_rowids (?):

    Returns:
        list: filtered_aid_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_case_tags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> annotmatch_rowids = ibs._get_all_annotmatch_rowids()
        >>> tags_list = get_annotmatch_case_tags(ibs, annotmatch_rowids)
        >>> result = ('tags_list = %s' % (str(tags_list),))
        >>> print(result)
        tags_list = [[u'occlusion', u'pose', 'Hard', 'NonDistinct'], [], ['Hard']]
    """
    standard, other = get_cate_categories()
    annotmatch_notes_list = ibs.get_annotmatch_note(annotmatch_rowids)
    tags_list = [[] if note is None else _parse_note(note) for note in annotmatch_notes_list]
    for case in standard:
        flag_list = ibs.get_annotmatch_prop(case, annotmatch_rowids)
        for tags in ut.list_compress(tags_list, flag_list):
            tags.append(case)
    return tags_list


@register_ibs_method
def get_annotmatch_prop(ibs, prop, annotmatch_rowids):
    r"""
    hacky getter for dynamic properties of annotmatches using notes table

    Args:
        prop (str):
        annotmatch_rowids (?):

    Returns:
        list: filtered_aid_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_prop

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Test setting and getting standard keys
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> prop = 'hard'
        >>> annotmatch_rowids = ibs._get_all_annotmatch_rowids()
        >>> flag_list = get_annotmatch_prop(ibs, prop, annotmatch_rowids)
        >>> flag_list = ('filtered_aid_list = %s' % (str(flag_list),))
        >>> subset_rowids = annotmatch_rowids[::2]
        >>> set_annotmatch_prop(ibs, prop, subset_rowids, [True] * len(subset_rowids))
        >>> flag_list2 = get_annotmatch_prop(ibs, prop, annotmatch_rowids)
        >>> print('flag_list2 = %r' % (flag_list2,))

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Test setting and getting non-standard keys
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> prop = 'occlusion'
        >>> annotmatch_rowids = ibs._get_all_annotmatch_rowids()
        >>> flag_list = get_annotmatch_prop(ibs, prop, annotmatch_rowids)
        >>> flag_list = ('filtered_aid_list = %s' % (str(flag_list),))
        >>> subset_rowids = annotmatch_rowids[1::2]
        >>> subset_rowids1 = annotmatch_rowids[::2]
        >>> set_annotmatch_prop(ibs, prop, subset_rowids1, [True] * len(subset_rowids))
        >>> set_annotmatch_prop(ibs, 'pose', subset_rowids1, [True] * len(subset_rowids))
        >>> flag_list2 = get_annotmatch_prop(ibs, prop, annotmatch_rowids)
        >>> print('flag_list2 = %r' % (flag_list2,))
    """
    if prop.lower() in ANNOTMATCH_PROPS_STANDARD_SET:
        getter = getattr(ibs, 'get_annotmatch_is_' + prop.lower())
        flag_list = getter(annotmatch_rowids)
        return flag_list
    elif prop.lower() in ANNOTMATCH_PROPS_OTHER_SET or prop.lower() in ANNOTMATCH_PROPS_OLD_SET:
        return get_annotmatch_other_prop(ibs, prop, annotmatch_rowids)
    else:
        raise NotImplementedError('Unknown prop=%r' % (prop,))


@register_ibs_method
def set_annotmatch_prop(ibs, prop, annotmatch_rowids, flags):
    """
    hacky setter for dynamic properties of annotmatches using notes table
    """
    if prop.lower() in ANNOTMATCH_PROPS_STANDARD_SET:
        setter = getattr(ibs, 'set_annotmatch_is_' + prop.lower())
        return setter(annotmatch_rowids, flags)
    elif prop.lower() in ANNOTMATCH_PROPS_OTHER_SET or prop.lower() in ANNOTMATCH_PROPS_OLD_SET:
        return set_annotmatch_other_prop(ibs, prop, annotmatch_rowids, flags)
    else:
        raise NotImplementedError('Unknown prop=%r' % (prop,))


def _parse_note(note):
    """ convert a note into tags """
    return [tag.strip() for tag in note.split(';') if len(tag) > 0]


def _remove_tag(tags, prop):
    """ convert a note into tags """
    try:
        tags.remove(prop)
    except ValueError:
        pass
    return tags


def get_annotmatch_other_prop(ibs, prop, annotmatch_rowids):
    prop = prop.lower()
    annotmatch_notes_list = ibs.get_annotmatch_note(annotmatch_rowids)
    tags_list = [None if note is None else _parse_note(note)
                 for note in annotmatch_notes_list]
    flag_list = [None if tags is None else int(prop in tags)
                 for tags in tags_list]
    return flag_list


def set_annotmatch_other_prop(ibs, prop, annotmatch_rowids, flags):
    """
    sets nonstandard properties using the notes column
    """
    prop = prop.lower()
    annotmatch_notes_list = ibs.get_annotmatch_note(annotmatch_rowids)
    ensured_notes = ['' if note is None else note for note in annotmatch_notes_list]
    tags_list = [_parse_note(note) for note in ensured_notes]
    # Remove from all
    new_tags_list = [_remove_tag(tags, prop) for tags in tags_list]
    # then add to specified ones
    for tags, flag in zip(new_tags_list, flags):
        if flag:
            tags.append(prop)
    new_notes_list = [
        ';'.join(tags) for tags in new_tags_list
    ]
    ibs.set_annotmatch_note(annotmatch_rowids, new_notes_list)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.annotmatch_funcs
        python -m ibeis.annotmatch_funcs --allexamples
        python -m ibeis.annotmatch_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
