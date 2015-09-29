# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import vtool as vt
from six.moves import zip, map
import numpy as np
from ibeis.control import controller_inject
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[tag_funcs]')


# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)


# TODO : make a annot_tags file


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


def consolodate_annotmatch_tags(case_tags):
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
        python -m ibeis.tag_funcs --exec-rename_and_reduce_tags --db PZ_Master1

    Example:
        >>> from ibeis.tag_funcs import *  # NOQA
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
        python -m ibeis.tag_funcs --exec-export_tagged_chips  --tags Hard interesting --db PZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.tag_funcs import *  # NOQA
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
        python -m ibeis.tag_funcs --exec-get_aidpair_tags --db PZ_Master1 --tags Hard interesting

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.tag_funcs import *  # NOQA
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
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --min-num=1
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags JoinCase
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags SplitCase
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags occlusion
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags viewpoint

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.tag_funcs import *  # NOQA
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
        python -m ibeis.tag_funcs --exec-get_annotmatch_case_tags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.tag_funcs import *  # NOQA
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
        python -m ibeis.tag_funcs --exec-get_annotmatch_prop

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Test setting and getting standard keys
        >>> from ibeis.tag_funcs import *  # NOQA
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
        >>> from ibeis.tag_funcs import *  # NOQA
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
    annotmatch_notes_list = ibs.get_annotmatch_note(annotmatch_rowids)
    flag_list = get_tags_in_textformat(prop, annotmatch_notes_list)
    return flag_list


def set_annotmatch_other_prop(ibs, prop, annotmatch_rowids, flags):
    """
    sets nonstandard properties using the notes column
    """
    annotmatch_notes_list = ibs.get_annotmatch_note(annotmatch_rowids)
    new_notes_list = set_tags_in_textformat(prop, annotmatch_notes_list, flags)
    ibs.set_annotmatch_note(annotmatch_rowids, new_notes_list)


def get_tags_in_textformat(prop, text_list):
    """ general text tag getter hack """
    prop = prop.lower()
    tags_list = [None if note is None else _parse_note(note)
                 for note in text_list]
    flag_list = [None if tags is None else int(prop in tags)
                 for tags in tags_list]
    return flag_list


def set_tags_in_textformat(prop, text_list, flags):
    """ general text tag setter hack """
    prop = prop.lower()
    ensured_text = ['' if note is None else note for note in text_list]
    tags_list = [_parse_note(note) for note in ensured_text]
    # Remove from all
    new_tags_list = [_remove_tag(tags, prop) for tags in tags_list]
    # then add to specified ones
    for tags, flag in zip(new_tags_list, flags):
        if flag:
            tags.append(prop)
    new_text_list = [
        ';'.join(tags) for tags in new_tags_list
    ]
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
    text_list = ibs.get_annot_tags(aid_list)
    flag_list = get_tags_in_textformat(prop, text_list)
    return flag_list


@register_ibs_method
def set_annot_prop(ibs, prop, aid_list, flags):
    """
    sets nonstandard properties using the notes column

    http://localhost:5000/group_review/?aid_list=1,2,3,4
    """
    text_list = ibs.get_annot_tags(aid_list)
    new_text_list = set_tags_in_textformat(prop, text_list, flags)
    ibs.set_annot_tags(aid_list, new_text_list)


@register_ibs_method
def get_annot_case_tags(ibs, aid_list):
    r"""
    returns list of tags. Use instead of get_annot_tags

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids

    Returns:
        list: tags_list

    CommandLine:
        python -m ibeis.tag_funcs --exec-get_annot_case_tags

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.tag_funcs import *  # NOQA
        >>> from ibeis.tag_funcs import _parse_note # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> tags_list = get_annot_case_tags(ibs, aid_list)
        >>> result = ('tags_list = %s' % (str(tags_list),))
        >>> print(result)

    Ignore:
        aid_list = ibs.get_valid_aids()
        notes_list = ibs.get_annot_notes(aid_list)
        flags = [len(notes) > 0 for notes in notes_list]
        aid_list = ut.list_compress(aid_list, flags)
        notes_list = ut.list_compress(notes_list, flags)

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
        aid_list = ut.list_compress(aid_list, flags)
        notes_list = ut.list_compress(notes_list, flags)

        flags = ['M;' not in notes and 'F;' not in notes and 'H1' not in notes for notes in notes_list]
        flags = [ 'M;' not in notes and 'F;' not in notes and 'H1' not in notes for notes in notes_list]
        aid_list = ut.list_compress(aid_list, flags)
        notes_list = ut.list_compress(notes_list, flags)

        flags = ['aliases' not in notes for notes in notes_list]
        aid_list = ut.list_compress(aid_list, flags)
        notes_list = ut.list_compress(notes_list, flags)

        #flags = [not re.match(';\d*;', note) for note  in notes_list]
        flags = [not re.match(r'\d\d*', note) for note  in notes_list]
        aid_list = ut.list_compress(aid_list, flags)
        notes_list = ut.list_compress(notes_list, flags)

        flags = [not notes.startswith('Foal;') for notes in notes_list]
        aid_list = ut.list_compress(aid_list, flags)
        notes_list = ut.list_compress(notes_list, flags)

        old_tags_list = [_parse_note(note) for note in notes_list]

        old_tags = list(set(ut.flatten(old_tags_list)))
        old_tags = sorted([tag for tag in old_tags if not re.match(r'\d\d*', tag)])

        old_to_new = {
            'gash':  None,
            'pose':  'novelpose',
            'vocalizing': 'novelpose'
            'occlusion':  'occlusion',
        }

    Ignore:
        python -m ibeis.tag_funcs --exec-filter_annotmatch_by_tags --show --db PZ_Master1 --tags viewpoint

    """
    text_list = ibs.get_annot_tags(aid_list)
    tags_list = [[] if note is None else _parse_note(note) for note in text_list]
    return tags_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.tag_funcs
        python -m ibeis.tag_funcs --allexamples
        python -m ibeis.tag_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
