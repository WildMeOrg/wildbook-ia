# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import zip, map
import numpy as np
import vtool as vt
import utool as ut
from wbia.control import controller_inject

print, rrr, profile = ut.inject2(__name__)


# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


# TODO : make a annot_tags file


ANNOTMATCH_PROPS_STANDARD = [
    # 'SceneryMatch',
    # 'Photobomb',
    # 'Hard',
    # 'NonDistinct',
]

ANNOTMATCH_PROPS_OTHER = [
    'SceneryMatch',
    'Photobomb',
    'Hard',
    'NonDistinct',
    'Occlusion',
    'Viewpoint',
    'MildViewpoint',
    'Pose',
    'Lighting',
    'Quality',  # quality causes failure
    'Orientation',  # orientation caused failure
    'EdgeMatch',  # descriptors on the edge of the naimal produce strong matches
    'Interesting',  # flag a case as interesting
    'JoinCase',  # case should actually be marked as correct
    'SplitCase',  # case should actually be marked as correct
    'random',  # gf case has random matches, the gt is to blame
    'BadShoulder',  # gf is a bad shoulder match
    'BadTail',  # gf is a bad tail match
    'TimeDeltaError',
    # These annots have almost the same information
    'NearDuplicate',
    'CorrectPhotobomb',  # FIXME: this is a terrible name
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
    'ViewpointCanDo': 'Correctable',
    'ViewpointOMG': 'Uncorrectable',
    'Shadowing': 'Lighting',
    'success': None,
    'GoodCoverage': None,
    # 'Hard'           : 'NeedsWork',
    'shouldhavemore': 'NeedsWork',
    'BadCoverage': 'NeedsWork',
    'ScoringIssue': 'NeedsWork',
    'TooSmallMatches': 'FeatureScale',
    'TooLargeMatches': 'FeatureScale',
    # 'BadShoulder' : 'BadShoulder',
    # 'GoodCoverage': None,
}

for key, val in PROP_MAPPING.items():
    if key in ANNOTMATCH_PROPS_OTHER:
        ANNOTMATCH_PROPS_OTHER.remove(key)
    if val is not None and val not in ANNOTMATCH_PROPS_OTHER:
        ANNOTMATCH_PROPS_OTHER.append(val)

ANNOTMATCH_PROPS_OTHER_SET = set([_.lower() for _ in ANNOTMATCH_PROPS_OTHER])
ANNOTMATCH_PROPS_OLD_SET = set([_.lower() for _ in OLD_ANNOTMATCH_PROPS])
# ANNOTMATCH_PROPS_STANDARD_SET = set([_.lower() for _ in ANNOTMATCH_PROPS_STANDARD])


def consolodate_annotmatch_tags(old_tags):
    # return case_tags
    remove_tags = [
        'hard',
        'needswork',
        'correctable',
        'uncorrectable',
        'interesting',
        'splitcase',
        'joincase',
        # 'orientation',
        'random',
        # 'badtail', 'badshoulder', 'splitcase', 'joincase', 'goodcoverage', 'interesting', 'hard'
    ]
    tags_dict = {
        # 'quality': 'Quality',
        # 'scoringissue': 'ScoringIssue',
        # 'orientation': 'Orientation',
        # 'orientation': 'MildViewpoint',
        'orientation': 'Viewpoint',
        # 'pose': 'SimilarPose',
        'pose': 'NonDistinct',
        # 'lighting': 'Lighting',
        # 'occlusion': 'Occlusion',
        # 'featurescale': 'FeatureScale',
        # 'edgematch': 'EdgeMatches',
        # 'featurescale': 'Pose',
        # 'featurescale': 'FeatureScale',
        'nondistinct': 'NonDistinct',
        'featurescale': 'NonDistinct',
        'edgematch': 'SimilarPose',
        'badtail': 'NonDistinct',
        'badshoulder': 'NonDistinct',
        # 'mildviewpoint': 'MildViewpoint',
        'mildviewpoint': 'Viewpoint',
        # 'toolargematches': 'CoarseFeatures',
        # 'badcoverage': 'LowCoverage',
        # 'shouldhavemore': 'LowCoverage',
        # 'viewpoint': 'Viewpoint',
    }

    def filter_tags(tags):
        return [t for t in tags if t.lower() not in remove_tags]

    def map_tags(tags):
        return [tags_dict.get(t.lower(), t) for t in tags]

    def cap_tags(tags):
        return [t[0].upper() + t[1:] for t in tags]

    filtered_tags = list(map(filter_tags, old_tags))
    mapped_tags = list(map(map_tags, filtered_tags))
    unique_tags = list(map(ut.unique_ordered, mapped_tags))
    new_tags = list(map(cap_tags, unique_tags))

    return new_tags


def rename_and_reduce_tags(ibs, annotmatch_rowids):
    """
    Script to update tags to newest values

    CommandLine:
        python -m wbia.tag_funcs --exec-rename_and_reduce_tags --db PZ_Master1

    Ignore:
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> #ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> annotmatch_rowids = filter_annotmatch_by_tags(ibs, min_num=1)
        >>> rename_and_reduce_tags(ibs, annotmatch_rowids)
    """
    tags_list_ = get_annotmatch_case_tags(ibs, annotmatch_rowids)

    def fix_tags(tags):
        return {six.text_type(t.lower()) for t in tags}

    tags_list = list(map(fix_tags, tags_list_))

    prop_mapping = {
        six.text_type(key.lower()): val for key, val in six.iteritems(PROP_MAPPING)
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
    # case_list = standard + other
    return standard, other


def export_tagged_chips(ibs, aid_list, dpath='.'):
    """
    DEPRICATE

    CommandLine:
        python -m wbia.tag_funcs --exec-export_tagged_chips --tags Hard interesting needswork --db PZ_Master1
        python -m wbia.tag_funcs --exec-export_tagged_chips --logic=or --any_startswith quality occlusion --has_any lighting needswork interesting hard --db GZ_Master1 --dpath=/media/raid
        python -m wbia.tag_funcs --exec-export_tagged_chips --db GZ_Master1 --min_num=1  --dpath /media/raid

    Example:
        >>> # SCRIPT
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> kwargs = ut.argparse_dict(ut.get_kwdefaults2(filterflags_general_tags), type_hint=ut.ddict(list, logic=str))
        >>> ut.print_dict(kwargs, 'filter args')
        >>> aid_list = ibs.filter_annots_by_tags(**kwargs)
        >>> print('len(aid_list) = %r' % (len(aid_list),))
        >>> dpath = ut.get_argval('--dpath', default='')
        >>> all_tags = ut.flatten(ibs.get_annot_all_tags(aid_list))
        >>> filtered_tag_hist = ut.dict_hist(all_tags)
        >>> ut.print_dict(filtered_tag_hist, key_order_metric='val')
        >>> export_tagged_chips(ibs, aid_list, dpath)
    """
    visual_uuid_hashid = ibs.get_annot_hashid_visual_uuid(aid_list)
    zip_fpath = ut.unixjoin(
        dpath, 'exported_chips2_' + ibs.get_dbname() + visual_uuid_hashid + '.zip'
    )
    chip_fpath = ibs.get_annot_chip_fpath(aid_list)
    ut.archive_files(zip_fpath, chip_fpath, common_prefix=True)


@register_ibs_method
def filter_annots_by_tags(ibs, aid_list=None, **kwargs):
    """
    Filter / Find / Search for annotations with particular tags

    CommandLine:
        python -m wbia.tag_funcs --exec-filter_annots_by_tags --helpx
        python -m wbia.tag_funcs --exec-filter_annots_by_tags --db GZ_Master1
        python -m wbia.tag_funcs --exec-filter_annots_by_tags --db GZ_Master1 --min_num=1
        python -m wbia.tag_funcs --exec-filter_annots_by_tags --db GZ_Master1 --has_any=lighting --has_all=lighting:underexposed --show

    SeeAlso:
        python -m wbia.init.filter_annots --exec-filter_annots_general

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> kwargs = ut.argparse_dict(ut.get_kwdefaults2(filterflags_general_tags), type_hint=ut.ddict(list, logic=str))
        >>> ut.print_dict(kwargs, 'filter args')
        >>> aid_list = ibs.filter_annots_by_tags(aid_list, **kwargs)
        >>> print('len(aid_list) = %r' % (len(aid_list),))
        >>> # print results
        >>> all_tags = ut.flatten(ibs.get_annot_all_tags(aid_list))
        >>> filtered_tag_hist = ut.dict_hist(all_tags)
        >>> ut.print_dict(filtered_tag_hist, key_order_metric='val')
        >>> print('len(aid_list) = %r' % (len(aid_list),))
        >>> print('sum(tags) = %r' % (sum(filtered_tag_hist.values()),))
        >>> ut.quit_if_noshow()
        >>> import wbia.viz.interact
        >>> wbia.viz.interact.interact_chip.interact_multichips(ibs, aid_list)
        >>> ut.show_if_requested()
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    tags_list = ibs.get_annot_all_tags(aid_list)
    flags = filterflags_general_tags(tags_list, **kwargs)
    aid_list = ut.compress(aid_list, flags)
    return aid_list


@register_ibs_method
def filterflags_annot_tags(ibs, aid_list, **kwargs):
    """
    Filter / Find / Search for annotations with particular tags
    """
    tags_list = ibs.get_annot_all_tags(aid_list)
    flags = filterflags_general_tags(tags_list, **kwargs)
    return flags


@register_ibs_method
def get_aidpair_tags(ibs, aid1_list, aid2_list, directed=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid1_list (list):
        aid2_list (list):
        directed (bool): (default = True)

    Returns:
        list: tags_list

    CommandLine:
        python -m wbia.tag_funcs --exec-get_aidpair_tags --db PZ_Master1 --tags Hard interesting

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> has_any = ut.get_argval('--tags', type_=list, default=None)
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> aid_pairs = filter_aidpairs_by_tags(ibs, has_any=has_any, min_num=1)
        >>> aid1_list = aid_pairs.T[0]
        >>> aid2_list = aid_pairs.T[1]
        >>> undirected_tags = get_aidpair_tags(ibs, aid1_list, aid2_list, directed=False)
        >>> tagged_pairs = list(zip(aid_pairs.tolist(), undirected_tags))
        >>> print(ut.repr2(tagged_pairs))
        >>> tag_dict = ut.groupby_tags(tagged_pairs, undirected_tags)
        >>> print(ut.repr2(tag_dict, nl=2))
        >>> print(ut.repr2(ut.map_dict_vals(len, tag_dict)))
    """
    aid_pairs = np.vstack([aid1_list, aid2_list]).T
    if directed:
        annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey(
            aid_pairs.T[0], aid_pairs.T[1]
        )
        tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowid)
    else:
        annotmatch_rowid = ibs.get_annotmatch_rowid_from_undirected_superkey(
            aid_pairs.T[0], aid_pairs.T[1]
        )
        tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowid)
        if False:
            expanded_aid_pairs = np.vstack([aid_pairs, aid_pairs[:, ::-1]])
            expanded_annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey(
                expanded_aid_pairs.T[0], expanded_aid_pairs.T[1]
            )
            expanded_edgeids = vt.get_undirected_edge_ids(expanded_aid_pairs)
            unique_edgeids, groupxs = vt.group_indices(expanded_edgeids)
            expanded_tags_list = ibs.get_annotmatch_case_tags(expanded_annotmatch_rowid)
            grouped_tags = vt.apply_grouping(
                np.array(expanded_tags_list, dtype=object), groupxs
            )
            undirected_tags = [list(set(ut.flatten(tags))) for tags in grouped_tags]
            edgeid2_tags = dict(zip(unique_edgeids, undirected_tags))
            input_edgeids = expanded_edgeids[: len(aid_pairs)]
            tags_list = ut.dict_take(edgeid2_tags, input_edgeids)
    return tags_list


@register_ibs_method
def filter_aidpairs_by_tags(
    ibs, has_any=None, has_all=None, min_num=None, max_num=None, am_rowids=None
):
    """
    list(zip(aid_pairs, undirected_tags))
    """
    # annotmatch_rowids = ibs.get_annotmatch_rowids_from_aid(aid_list)

    filtered_annotmatch_rowids = filter_annotmatch_by_tags(
        ibs,
        am_rowids,
        has_any=has_any,
        has_all=has_all,
        min_num=min_num,
        max_num=max_num,
    )
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
    # directed_tags = get_aidpair_tags(ibs, aid_pairs.T[0], aid_pairs.T[1], directed=True)
    # valid_tags_list = ibs.get_annotmatch_case_tags(filtered_annotmatch_rowids)


def filter_annotmatch_by_tags(ibs, annotmatch_rowids=None, **kwargs):
    r"""
    ignores case

    Args:
        ibs (IBEISController):  wbia controller object
        flags (?):

    Returns:
        list

    CommandLine:
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --min-num=1
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags JoinCase
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags SplitCase
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags occlusion
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags viewpoint
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags SceneryMatch
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags Photobomb

        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db GZ_Master1 --tags needswork

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> #ibs = wbia.opendb(defaultdb='testdb1')
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> #tags = ['Photobomb', 'SceneryMatch']
        >>> has_any = ut.get_argval('--tags', type_=list, default=['SceneryMatch', 'Photobomb'])
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> prop = has_any[0]
        >>> filtered_annotmatch_rowids = filter_annotmatch_by_tags(ibs, None, has_any=has_any, min_num=min_num)
        >>> aid1_list = np.array(ibs.get_annotmatch_aid1(filtered_annotmatch_rowids))
        >>> aid2_list = np.array(ibs.get_annotmatch_aid2(filtered_annotmatch_rowids))
        >>> aid_pairs = np.vstack([aid1_list, aid2_list]).T
        >>> # Dont double count
        >>> xs = vt.find_best_undirected_edge_indexes(aid_pairs)
        >>> aid1_list = aid1_list.take(xs)
        >>> aid2_list = aid2_list.take(xs)
        >>> valid_tags_list = ibs.get_annotmatch_case_tags(filtered_annotmatch_rowids)
        >>> print('valid_tags_list = %s' % (ut.repr2(valid_tags_list, nl=1),))
        >>> #
        >>> print('Aid pairs with has_any=%s' % (has_any,))
        >>> print('Aid pairs with min_num=%s' % (min_num,))
        >>> print('aid_pairs = ' + ut.repr2(list(zip(aid1_list, aid2_list))))
        >>> # Show timedelta info
        >>> ut.quit_if_noshow()
        >>> timedelta_list = ibs.get_annot_pair_timedelta(aid1_list, aid2_list)
        >>> import wbia.plottool as pt
        >>> pt.draw_timedelta_pie(timedelta_list, label='timestamp of tags=%r' % (has_any,))
        >>> ut.show_if_requested()
    """
    if annotmatch_rowids is None:
        annotmatch_rowids = ibs._get_all_annotmatch_rowids()

    tags_list = ibs.get_annotmatch_case_tags(annotmatch_rowids)

    flags = filterflags_general_tags(tags_list, **kwargs)
    filtered_annotmatch_rowids = ut.compress(annotmatch_rowids, flags)
    return filtered_annotmatch_rowids


# TODO: ut.filterflags_general_tags


def filterflags_general_tags(
    tags_list,
    has_any=None,
    has_all=None,
    has_none=None,
    min_num=None,
    max_num=None,
    any_startswith=None,
    any_endswith=None,
    any_match=None,
    none_match=None,
    logic='and',
):
    r"""
    maybe integrate into utool? Seems pretty general

    Args:
        tags_list (list):
        has_any (None): (default = None)
        has_all (None): (default = None)
        min_num (None): (default = None)
        max_num (None): (default = None)

    CommandLine:
        python -m wbia.tag_funcs --exec-filterflags_general_tags
        python -m wbia.tag_funcs --exec-filterflags_general_tags:0  --helpx
        python -m wbia.tag_funcs --exec-filterflags_general_tags:0
        python -m wbia.tag_funcs --exec-filterflags_general_tags:0  --none_match n
        python -m wbia.tag_funcs --exec-filterflags_general_tags:0  --has_none=n,o
        python -m wbia.tag_funcs --exec-filterflags_general_tags:1
        python -m wbia.tag_funcs --exec-filterflags_general_tags:2

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> tags_list = [['v'], [], ['P'], ['P', 'o'], ['n', 'o',], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['q', 'v'], ['n'], ['n'], ['N']]
        >>> kwargs = ut.argparse_dict(ut.get_kwdefaults2(filterflags_general_tags), type_hint=list)
        >>> print('kwargs = %r' % (kwargs,))
        >>> flags = filterflags_general_tags(tags_list, **kwargs)
        >>> print(flags)
        >>> result = ut.compress(tags_list, flags)
        >>> print('result = %r' % (result,))

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> tags_list = [['v'], [], ['P'], ['P'], ['n', 'o',], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n'], ['N']]
        >>> has_all = 'n'
        >>> min_num = 1
        >>> flags = filterflags_general_tags(tags_list, has_all=has_all, min_num=min_num)
        >>> result = ut.compress(tags_list, flags)
        >>> print('result = %r' % (result,))

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> tags_list = [['vn'], ['vn', 'no'], ['P'], ['P'], ['n', 'o',], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n', 'nP'], ['NP']]
        >>> kwargs = {
        >>>     'any_endswith': 'n',
        >>>     'any_match': None,
        >>>     'any_startswith': 'n',
        >>>     'has_all': None,
        >>>     'has_any': None,
        >>>     'has_none': None,
        >>>     'max_num': 3,
        >>>     'min_num': 1,
        >>>     'none_match': ['P'],
        >>> }
        >>> flags = filterflags_general_tags(tags_list, **kwargs)
        >>> filtered = ut.compress(tags_list, flags)
        >>> result = ('result = %s' % (ut.repr2(filtered),))
        result = [['vn', 'no'], ['n', 'o'], ['n', 'N'], ['n'], ['n', 'nP']]
    """
    import re
    import operator

    def fix_tags(tags):
        return {six.text_type(t.lower()) for t in tags}

    if logic is None:
        logic = 'and'

    logic_func = {'and': np.logical_and, 'or': np.logical_or}[logic]

    default_func = {'and': np.ones, 'or': np.zeros}[logic]

    tags_list_ = [fix_tags(tags_) for tags_ in tags_list]
    flags = default_func(len(tags_list_), dtype=np.bool)

    if min_num is not None:
        flags_ = [len(tags_) >= min_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if max_num is not None:
        flags_ = [len(tags_) <= max_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_any is not None:
        has_any = fix_tags(set(ut.ensure_iterable(has_any)))
        flags_ = [len(has_any.intersection(tags_)) > 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_none is not None:
        has_none = fix_tags(set(ut.ensure_iterable(has_none)))
        flags_ = [len(has_none.intersection(tags_)) == 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_all is not None:
        has_all = fix_tags(set(ut.ensure_iterable(has_all)))
        flags_ = [
            len(has_all.intersection(tags_)) == len(has_all) for tags_ in tags_list_
        ]
        logic_func(flags, flags_, out=flags)

    def check_item(tags_, fields, op, compare):
        t_flags = [any([compare(t, f) for f in fields]) for t in tags_]
        num_passed = sum(t_flags)
        flag = op(num_passed, 0)
        return flag

    def flag_tags(tags_list, fields, op, compare):
        flags = [check_item(tags_, fields, op, compare) for tags_ in tags_list_]
        return flags

    def execute_filter(flags, tags_list, fields, op, compare):
        if fields is not None:
            fields = ut.ensure_iterable(fields)
            flags_ = flag_tags(tags_list, fields, op, compare)
            logic_func(flags, flags_, out=flags)
        return flags

    flags = execute_filter(
        flags, tags_list, any_startswith, operator.gt, six.text_type.startswith
    )

    flags = execute_filter(
        flags, tags_list, any_endswith, operator.gt, six.text_type.endswith
    )

    flags = execute_filter(
        flags, tags_list, any_match, operator.gt, lambda t, f: re.match(f, t)
    )

    flags = execute_filter(
        flags, tags_list, none_match, operator.eq, lambda t, f: re.match(f, t)
    )
    return flags


@register_ibs_method
@profile
def get_annotmatch_case_tags(ibs, annotmatch_rowids):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        annotmatch_rowids (?):

    Returns:
        list: filtered_aid_list

    CommandLine:
        python -m wbia.tag_funcs --exec-get_annotmatch_case_tags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> annotmatch_rowids = ibs._get_all_annotmatch_rowids()
        >>> tags_list = get_annotmatch_case_tags(ibs, annotmatch_rowids)
        >>> result = ('tags_list = %s' % (str(tags_list),))
        >>> print(result)
        tags_list = [[u'occlusion', u'pose', 'Hard', 'NonDistinct'], [], ['Hard']]
    """
    standard, other = get_cate_categories()
    annotmatch_tag_texts_list = ibs.get_annotmatch_tag_text(annotmatch_rowids)
    tags_list = [
        [] if note is None else _parse_tags(note) for note in annotmatch_tag_texts_list
    ]
    # NEW = False
    # if NEW:
    #    # hack for faster tag parsing
    #    from wbia.control import _autogen_annotmatch_funcs as _aaf
    #    import itertools
    #    colnames = (_aaf.ANNOTMATCH_IS_HARD, _aaf.ANNOTMATCH_IS_SCENERYMATCH,
    #                _aaf.ANNOTMATCH_IS_PHOTOBOMB, _aaf.ANNOTMATCH_IS_NONDISTINCT)
    #    id_iter = annotmatch_rowids
    #    annotmatch_is_col = ibs.db.get(
    #        ibs.const.ANNOTMATCH_TABLE, colnames, id_iter, id_colname='rowid',
    #        eager=True, nInput=None, unpack_scalars=True)
    #    annotmatch_is_col = [col if col is not None else [None] * len(colnames)
    #                         for col in annotmatch_is_col]
    #    standardtags = [x[len('annotmatch_is_'):] for x in colnames]
    #    standard_tags_list = ut.list_zipcompress(itertools.repeat(standardtags), annotmatch_is_col)
    #    tags_list = [tags1 + tags2 for tags1, tags2 in zip(tags_list, standard_tags_list)]
    # else:
    #    for case in standard:
    #        flag_list = ibs.get_annotmatch_prop(case, annotmatch_rowids)
    #        for tags in ut.compress(tags_list, flag_list):
    #            tags.append(case)
    tags_list = [[six.text_type(t) for t in tags] for tags in tags_list]
    # if ut.get_argval('--consol') or True:
    #    tags_list = consolodate_annotmatch_tags(tags_list)
    return tags_list


@profile
def get_annotmatch_standard_prop(ibs, prop, annotmatch_rowids):
    getter = getattr(ibs, 'get_annotmatch_is_' + prop.lower())
    flag_list = getter(annotmatch_rowids)
    return flag_list


@register_ibs_method
@profile
def get_annotmatch_prop(ibs, prop, annotmatch_rowids):
    r"""
    hacky getter for dynamic properties of annotmatches using notes table

    Args:
        prop (str):
        annotmatch_rowids (?):

    Returns:
        list: filtered_aid_list

    CommandLine:
        python -m wbia.tag_funcs --exec-get_annotmatch_prop

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Test setting and getting standard keys
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
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
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
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
    # if prop.lower() in ANNOTMATCH_PROPS_STANDARD_SET:
    #    return ibs.get_annotmatch_standard_prop(prop, annotmatch_rowids)
    for prop_ in ut.ensure_iterable(prop):
        flag1 = prop_.lower() not in ANNOTMATCH_PROPS_OTHER_SET
        flag2 = prop_.lower() not in ANNOTMATCH_PROPS_OLD_SET
        if flag1 and flag2:
            raise NotImplementedError('Unknown prop_=%r' % (prop_,))
    return get_annotmatch_other_prop(ibs, prop, annotmatch_rowids)


@register_ibs_method
def set_annotmatch_prop(ibs, prop, annotmatch_rowids, flags):
    """
    hacky setter for dynamic properties of annotmatches using notes table
    """
    print(
        '[ibs] set_annotmatch_prop prop=%s for %d pairs' % (prop, len(annotmatch_rowids))
    )
    # if prop.lower() in ANNOTMATCH_PROPS_STANDARD_SET:
    #    setter = getattr(ibs, 'set_annotmatch_is_' + prop.lower())
    #    return setter(annotmatch_rowids, flags)
    if (
        prop.lower() in ANNOTMATCH_PROPS_OTHER_SET
        or prop.lower() in ANNOTMATCH_PROPS_OLD_SET
    ):
        return set_annotmatch_other_prop(ibs, prop, annotmatch_rowids, flags)
    else:
        raise NotImplementedError(
            'Unknown prop=%r not in %r' % (prop, ANNOTMATCH_PROPS_OTHER_SET)
        )


def _parse_tags(note):
    """ convert a note into tags """
    return [tag.strip() for tag in note.split(';') if len(tag) > 0]


def _remove_tag(tags, prop):
    """ convert a note into tags """
    try:
        tags.remove(prop)
    except ValueError:
        pass
    return tags


@profile
def get_annotmatch_other_prop(ibs, prop, annotmatch_rowids):
    annotmatch_tag_texts_list = ibs.get_annotmatch_tag_text(annotmatch_rowids)
    flag_list = get_textformat_tag_flags(prop, annotmatch_tag_texts_list)
    return flag_list


def set_annotmatch_other_prop(ibs, prop, annotmatch_rowids, flags):
    """
    sets nonstandard properties using the notes column
    """
    annotmatch_tag_texts_list = ibs.get_annotmatch_tag_text(annotmatch_rowids)
    new_notes_list = set_textformat_tag_flags(prop, annotmatch_tag_texts_list, flags)
    ibs.set_annotmatch_tag_text(annotmatch_rowids, new_notes_list)


@profile
def get_textformat_tag_flags(prop, text_list):
    """ general text tag getter hack """
    tags_list = [None if note is None else _parse_tags(note) for note in text_list]
    if ut.isiterable(prop):
        props_ = [p.lower() for p in prop]
        flags_list = [
            [None if tags is None else int(prop_ in tags) for tags in tags_list]
            for prop_ in props_
        ]
        return flags_list
    else:
        prop = prop.lower()
        flag_list = [None if tags is None else int(prop in tags) for tags in tags_list]
        return flag_list


def set_textformat_tag_flags(prop, text_list, flags):
    """ general text tag setter hack """
    prop = prop.lower()
    ensured_text = ['' if note is None else note for note in text_list]
    tags_list = [_parse_tags(note) for note in ensured_text]
    # Remove from all
    new_tags_list = [_remove_tag(tags, prop) for tags in tags_list]
    # then add to specified ones
    for tags, flag in zip(new_tags_list, flags):
        if flag:
            tags.append(prop)
    new_text_list = [';'.join(tags) for tags in new_tags_list]
    return new_text_list


ANNOT_TAGS = [
    'occlusion',
    'lighting',
    'quality',
    'pose',
    'error',
    'interesting',
    'error:viewpoint',
    'error:quality',
    'occlusion:large',
    'occlusion:medium',
    'occlusion:small',
    'lighting:shadowed',
    'lighting:overexposed',
    'lighting:underexposed',
    'quality:washedout',
    'quality:blury',
    'pose:novel',
    'pose:common',
    'error:bbox',
    'error:mask',
    'error:other',
]


def get_available_annot_tags():
    return ANNOT_TAGS


def get_annot_prop(ibs, prop, aid_list):
    """
    Annot tags
    """
    text_list = ibs.get_annot_tag_text(aid_list)
    flag_list = get_textformat_tag_flags(prop, text_list)
    return flag_list


@register_ibs_method
def set_annot_prop(ibs, prop, aid_list, flags):
    """
    sets nonstandard properties using the notes column
    """
    text_list = ibs.get_annot_tag_text(aid_list)
    new_text_list = set_textformat_tag_flags(prop, text_list, flags)
    ibs.set_annot_tag_text(aid_list, new_text_list)


@register_ibs_method
def append_annot_case_tags(ibs, aid_list, tag_list):
    """
    Generally appends tags to annotations. Careful not to introduce too many
    random tags. Maybe we should just let that happen and introduce tag-aliases

    Note: this is more of a set add rather than a list append

    TODO: remove
    """
    # Ensure each item is a list
    # tags_list = [tag if isinstance(tag, list) else [tag] for tag in tag_list]
    if isinstance(tag_list, six.string_types):
        # Apply single tag to everybody
        tag_list = [tag_list] * len(aid_list)
    tags_list = [ut.ensure_iterable(tag) for tag in tag_list]
    text_list = ibs.get_annot_tag_text(aid_list)
    orig_tags_list = [[] if note is None else _parse_tags(note) for note in text_list]
    new_tags_list = [ut.unique(t1 + t2) for t1, t2 in zip(tags_list, orig_tags_list)]
    ibs.set_annot_case_tags(aid_list, new_tags_list)


@register_ibs_method
def set_annot_case_tags(ibs, aid_list, new_tags_list):
    """
    Completely overwrite case tags
    """
    for tag in new_tags_list:
        assert isinstance(tag, list), 'each set of tags must be a list of strs'
    new_text_list = [';'.join(tags) for tags in new_tags_list]
    ibs.set_annot_tag_text(aid_list, new_text_list)


@register_ibs_method
def remove_annot_case_tags(ibs, aid_list, tag_list):
    if isinstance(tag_list, six.string_types):
        # Apply single tag to everybody
        tag_list = [tag_list] * len(aid_list)
    tags_list = [ut.ensure_iterable(tag) for tag in tag_list]
    text_list = ibs.get_annot_tag_text(aid_list)
    orig_tags_list = [[] if note is None else _parse_tags(note) for note in text_list]
    new_tags_list = [ut.setdiff(t2, t1) for t1, t2 in zip(tags_list, orig_tags_list)]
    new_text_list = [';'.join(tags) for tags in new_tags_list]
    ibs.set_annot_tag_text(aid_list, new_text_list)


@register_ibs_method
def overwrite_annot_case_tags(ibs, aid_list, tag_list):
    """
    Completely replaces annotation tags.
    BE VERY CAREFUL WITH THIS FUNCTION
    """
    assert all([ut.isiterable(tag) for tag in tag_list])
    # text_list = ibs.get_annot_tag_text(aid_list)
    # orig_tags_list = [[] if note is None else _parse_tags(note) for note in text_list]
    new_tags_list = tag_list
    new_text_list = [';'.join(tags) for tags in new_tags_list]
    ibs.set_annot_tag_text(aid_list, new_text_list)


@register_ibs_method
def remove_all_annot_case_tags(ibs, aid_list):
    ibs.set_annot_tag_text(aid_list, [''] * len(aid_list))


@register_ibs_method
def get_annot_case_tags(ibs, aid_list):
    r"""
    returns list of tags. Use instead of get_annot_tag_text
    TODO:
        rename to get_annot_unary_tags

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids

    Returns:
        list: tags_list

    CommandLine:
        python -m wbia.tag_funcs --exec-get_annot_case_tags

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> from wbia.tag_funcs import _parse_tags # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> tags_list = get_annot_case_tags(ibs, aid_list)
        >>> result = ('tags_list = %s' % (str(tags_list),))
        >>> print(result)

    Ignore:
        # FIXME incorrporate old tag notes
        aid_list = ibs.get_valid_aids()
        notes_list = ibs.get_annot_notes(aid_list)
        flags = [len(notes) > 0 for notes in notes_list]
        aid_list = ut.compress(aid_list, flags)
        notes_list = ut.compress(notes_list, flags)

        import re
        notes_list = [note.replace('rfdetect', '') for note in notes_list]
        notes_list = [note.replace('<COMMA>', ';') for note in notes_list]
        notes_list = [note.replace('jpg', '') for note in notes_list]
        notes_list = [note.replace('<HARDCASE>', '') for note in notes_list]
        notes_list = [note.strip() for note in notes_list]
        notes_list = [re.sub(';;*', ';', note) for note in notes_list]
        notes_list = [note.strip(';') for note in notes_list]
        notes_list = [note.strip(':') for note in notes_list]
        notes_list = [note.strip() for note in notes_list]

        flags = [len(notes) < 70 and len(notes) > 0 for notes in notes_list]
        aid_list = ut.compress(aid_list, flags)
        notes_list = ut.compress(notes_list, flags)

        flags = ['M;' not in notes and 'F;' not in notes and 'H1' not in notes for notes in notes_list]
        flags = [ 'M;' not in notes and 'F;' not in notes and 'H1' not in notes for notes in notes_list]
        aid_list = ut.compress(aid_list, flags)
        notes_list = ut.compress(notes_list, flags)

        flags = ['aliases' not in notes for notes in notes_list]
        aid_list = ut.compress(aid_list, flags)
        notes_list = ut.compress(notes_list, flags)

        #flags = [not re.match(';\d*;', note) for note  in notes_list]
        flags = [not re.match(r'\d\d*', note) for note  in notes_list]
        aid_list = ut.compress(aid_list, flags)
        notes_list = ut.compress(notes_list, flags)

        flags = [not notes.startswith('Foal;') for notes in notes_list]
        aid_list = ut.compress(aid_list, flags)
        notes_list = ut.compress(notes_list, flags)

        old_tags_list = [_parse_tags(note) for note in notes_list]

        old_tags = list(set(ut.flatten(old_tags_list)))
        old_tags = sorted([tag for tag in old_tags if not re.match(r'\d\d*', tag)])

        old_to_new = {
            'gash':  None,
            'pose':  'novelpose',
            'vocalizing': 'novelpose'
            'occlusion':  'occlusion',
        }

    Ignore:
        python -m wbia.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags viewpoint

    """
    text_list = ibs.get_annot_tag_text(aid_list)
    tags_list = [[] if note is None else _parse_tags(note) for note in text_list]
    return tags_list


@register_ibs_method
@profile
def get_annot_annotmatch_tags(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids

    Returns:
        list: annotmatch_tags_list

    CommandLine:
        python -m wbia.tag_funcs --exec-get_annot_annotmatch_tags --db GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> all_tags = ut.flatten(get_annot_annotmatch_tags(ibs, aid_list))
        >>> tag_hist = ut.dict_hist(all_tags)
        >>> ut.print_dict(tag_hist)
    """
    annotmatch_rowids = ibs.get_annotmatch_rowids_from_aid(aid_list)
    unflat_tags_list = ibs.unflat_map(ibs.get_annotmatch_case_tags, annotmatch_rowids)
    annotmatch_tags_list = [
        list(set(ut.flatten(_unflat_tags))) for _unflat_tags in unflat_tags_list
    ]
    return annotmatch_tags_list


@register_ibs_method
@profile
def get_annot_all_tags(ibs, aid_list=None):
    """
    CommandLine:
        python -m wbia.tag_funcs --exec-get_annot_all_tags --db GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tag_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> all_tags = ut.flatten(ibs.get_annot_all_tags(aid_list))
        >>> tag_hist = ut.dict_hist(all_tags)
        >>> ut.print_dict(tag_hist)
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()

    annotmatch_tags_list = ibs.get_annot_annotmatch_tags(aid_list)
    annot_tags_list = ibs.get_annot_case_tags(aid_list)
    both_tags_list = list(
        map(
            ut.unique_ordered,
            map(ut.flatten, zip(annot_tags_list, annotmatch_tags_list)),
        )
    )
    return both_tags_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.tag_funcs
        python -m wbia.tag_funcs --allexamples
        python -m wbia.tag_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
