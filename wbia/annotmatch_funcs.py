# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import ubelt as ub  # NOQA
import numpy as np
from six.moves import zip, map, filter, range  # NOQA
from functools import partial  # NOQA
from wbia.control import controller_inject

print, rrr, profile = ut.inject2(__name__)

# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


# def get_annotmatch_rowids_subset_from_aids(ibs, aids):
#     pass


@register_ibs_method
@profile
def get_annotmatch_rowids_from_aid1(ibs, aid1_list, eager=True, nInput=None):
    """
    TODO autogenerate

    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    Args:
        ibs (IBEISController):  wbia controller object
        aid1_list (list):
        eager (bool): (default = True)
        nInput (None): (default = None)

    Returns:
        list: annotmatch_rowid_list
    """
    from wbia.control import manual_annotmatch_funcs

    colnames = (manual_annotmatch_funcs.ANNOTMATCH_ROWID,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid1_list)
    if True:
        # HACK IN INDEX
        ibs.db.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS aid1_to_am ON {ANNOTMATCH_TABLE} ({annot_rowid1});
            """.format(
                ANNOTMATCH_TABLE=ibs.const.ANNOTMATCH_TABLE,
                annot_rowid1=manual_annotmatch_funcs.ANNOT_ROWID1,
            )
        ).fetchall()
    where_colnames = [manual_annotmatch_funcs.ANNOT_ROWID1]
    annotmatch_rowid_list = ibs.db.get_where_eq(
        ibs.const.ANNOTMATCH_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    annotmatch_rowid_list = list(map(sorted, annotmatch_rowid_list))
    return annotmatch_rowid_list


@register_ibs_method
@profile
def get_annotmatch_rowids_from_aid2(
    ibs, aid2_list, eager=True, nInput=None, force_method=None
):
    """
    # This one is slow because aid2 is the second part of the index
    Returns a list of the aids that were reviewed as candidate matches to the input aid
    """
    from wbia.control import manual_annotmatch_funcs

    if nInput is None:
        nInput = len(aid2_list)
    if True:
        # HACK IN INDEX
        ibs.db.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS aid2_to_am ON {ANNOTMATCH_TABLE} ({annot_rowid2});
            """.format(
                ANNOTMATCH_TABLE=ibs.const.ANNOTMATCH_TABLE,
                annot_rowid2=manual_annotmatch_funcs.ANNOT_ROWID2,
            )
        ).fetchall()
    colnames = (manual_annotmatch_funcs.ANNOTMATCH_ROWID,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid2_list)
    where_colnames = [manual_annotmatch_funcs.ANNOT_ROWID2]
    annotmatch_rowid_list = ibs.db.get_where_eq(
        ibs.const.ANNOTMATCH_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    annotmatch_rowid_list = list(map(sorted, annotmatch_rowid_list))
    return annotmatch_rowid_list


@register_ibs_method
@profile
def get_annotmatch_rowids_from_aid(
    ibs, aid_list, eager=True, nInput=None, force_method=None
):
    """
    Undirected version
    Returns a list of the aids that were reviewed as candidate matches to the input aid
    aid_list = ibs.get_valid_aids()

    CommandLine:
        python -m wbia.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid
        python -m wbia.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid:1 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> ut.exec_funckw(ibs.get_annotmatch_rowids_from_aid, globals())
        >>> aid_list = ibs.get_valid_aids()[0:4]
        >>> annotmatch_rowid_list = ibs.get_annotmatch_rowids_from_aid(aid_list,
        >>>                                                        eager, nInput)
        >>> result = ('annotmatch_rowid_list = %s' % (str(annotmatch_rowid_list),))
        >>> print(result)
    """
    # from wbia.control import manual_annotmatch_funcs
    if nInput is None:
        nInput = len(aid_list)
    if nInput == 0:
        return []
    rowids1 = ibs.get_annotmatch_rowids_from_aid1(aid_list)
    rowids2 = ibs.get_annotmatch_rowids_from_aid2(aid_list)
    annotmatch_rowid_list = [ut.unique(ut.flatten(p)) for p in zip(rowids1, rowids2)]
    # Ensure funciton output is consistent
    annotmatch_rowid_list = list(map(sorted, annotmatch_rowid_list))
    return annotmatch_rowid_list


@register_ibs_method
@profile
def get_annotmatch_rowid_from_undirected_superkey(ibs, aids1, aids2):
    # The directed nature of this makes a few things difficult and may cause
    # odd behavior
    am_rowids = ibs.get_annotmatch_rowid_from_superkey(aids1, aids2)
    idxs = ut.where([r is None for r in am_rowids])
    # Check which ones are None
    aids1_ = ut.take(aids1, idxs)
    aids2_ = ut.take(aids2, idxs)
    am_rowids_ = ibs.get_annotmatch_rowid_from_superkey(aids2_, aids1_)
    # Use the other rowid if found
    for idx, rowid in zip(idxs, am_rowids_):
        am_rowids[idx] = rowid
    return am_rowids


@register_ibs_method
def get_annotmatch_rowid_from_edges(ibs, aid_pairs):
    """
    Edegs are undirected
    """
    aid_pairs = np.array(aid_pairs)
    aids1 = aid_pairs.T[0]
    aids2 = aid_pairs.T[1]
    return ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)


@register_ibs_method
def get_annotmatch_rowids_in_cliques(ibs, aids_list):
    # Equivalent call:
    # ibs.get_annotmatch_rowids_between_groups(ibs, aids_list, aids_list)
    import itertools

    ams_list = [
        ibs.get_annotmatch_rowid_from_undirected_superkey(
            *zip(*itertools.combinations(aids, 2))
        )
        for aids in ut.ProgIter(aids_list, lbl='loading clique am rowids')
    ]
    ams_list = [[] if ams is None else ut.filter_Nones(ams) for ams in ams_list]
    return ams_list


@register_ibs_method
def get_annotmatch_rowids_between_groups(ibs, aids1_list, aids2_list):
    ams_list = []
    lbl = 'loading between group am rowids'
    for aids1, aids2 in ut.ProgIter(list(zip(aids1_list, aids2_list)), lbl=lbl):
        ams = get_annotmatch_rowids_between(ibs, aids1, aids2)
        ams_list.append(ams)
    return ams_list


@register_ibs_method
def get_annotmatch_rowids_between(ibs, aids1, aids2, method=None):
    """

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> aids1 = aids2 = [1, 2, 3, 4, 5, 6]
        >>> rowids_between = ibs.get_annotmatch_rowids_between
        >>> ams1 = sorted(rowids_between(aids1, aids2, method=1))
        >>> ams2 = sorted(rowids_between(aids1, aids2, method=2))
        >>> assert len(ub.find_duplicates(ams1)) == 0
        >>> assert len(ub.find_duplicates(ams2)) == 0
        >>> assert sorted(ams2) == sorted(ams1)
    """
    if method is None:
        if len(aids1) * len(aids2) > 5000:
            method = 1
        else:
            method = 2
    if method == 1:
        # Strategy 1: get all existing rows and see what intersects
        # This is better when the enumerated set of rows would be larger than
        # the database size
        unflat_rowids1L = ibs.get_annotmatch_rowids_from_aid1(aids1)
        unflat_rowids1R = ibs.get_annotmatch_rowids_from_aid2(aids1)
        unflat_rowids2L = ibs.get_annotmatch_rowids_from_aid1(aids2)
        unflat_rowids2R = ibs.get_annotmatch_rowids_from_aid2(aids2)

        am_rowids1L = {r for r in ut.iflatten(unflat_rowids1L) if r is not None}
        am_rowids1R = {r for r in ut.iflatten(unflat_rowids1R) if r is not None}
        am_rowids2L = {r for r in ut.iflatten(unflat_rowids2L) if r is not None}
        am_rowids2R = {r for r in ut.iflatten(unflat_rowids2R) if r is not None}

        ams12 = am_rowids1L.intersection(am_rowids2R)
        ams21 = am_rowids2L.intersection(am_rowids1R)
        ams = sorted(ams12.union(ams21))
        # ams = sorted(am_rowids1.intersection(am_rowids2))
        # rowids2 = ibs.get_annotmatch_rowids_from_aid2(aid_list)
        # unflat_rowids1 = ibs.get_annotmatch_rowids_from_aid(aids1)
        # unflat_rowids2 = ibs.get_annotmatch_rowids_from_aid(aids2)
        # am_rowids1 = {r for r in ut.iflatten(unflat_rowids1) if r is not None}
        # am_rowids2 = {r for r in ut.iflatten(unflat_rowids2) if r is not None}
        # ams = sorted(am_rowids1.intersection(am_rowids2))
        # ams = ut.isect(am_rowids1, am_rowids2)
    elif method == 2:
        # Strategy 2: enumerate what rows could exist and see what does exist
        # This is better when the enumerated set of rows would be smaller than
        # the database size
        edges = list(ut.product_nonsame(aids1, aids2))
        if len(edges) == 0:
            ams = []
        else:
            aids1_, aids2_ = ut.listT(edges)
            # ams = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1_, aids2_)
            ams = ibs.get_annotmatch_rowid_from_superkey(aids1_, aids2_)
            if ams is None:
                ams = []
            ams = ut.filter_Nones(ams)
    return ams


@register_ibs_method
def add_annotmatch_undirected(ibs, aids1, aids2, **kwargs):
    if len(aids1) == 0 and len(aids2) == 0:
        return []
    edges = list(zip(aids1, aids2))
    from wbia.algo.graph import nx_utils as nxu

    # Enforce new undirected constraint
    edges = ut.estarmap(nxu.e_, edges)
    aids1, aids2 = list(zip(*edges))

    am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
    idxs = ut.where([r is None for r in am_rowids])
    # Check which ones are None
    aids1_ = ut.take(aids1, idxs)
    aids2_ = ut.take(aids2, idxs)
    # Create anything that is None
    am_rowids_ = ibs.add_annotmatch(aids2_, aids1_)
    # Use the other rowid if found
    for idx, rowid in zip(idxs, am_rowids_):
        am_rowids[idx] = rowid
    return am_rowids


@register_ibs_method
def get_annot_pair_timedelta(ibs, aid_list1, aid_list2):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    Returns:
        list: timedelta_list

    CommandLine:
        python -m wbia.annotmatch_funcs --test-get_annot_pair_timedelta

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids(hasgt=True)
        >>> unixtimes = ibs.get_annot_image_unixtimes_asfloat(aid_list)
        >>> aid_list = ut.compress(aid_list, ~np.isnan(unixtimes))
        >>> gt_aids_list = ibs.get_annot_groundtruth(aid_list, daid_list=aid_list)
        >>> flags = np.array(list(map(len, gt_aids_list))) > 0
        >>> aid_list1 = ut.compress(aid_list, flags)[0:5]
        >>> aid_list2 = ut.take_column(gt_aids_list, 0)[0:5]
        >>> timedelta_list = ibs.get_annot_pair_timedelta(aid_list1, aid_list2)
        >>> result = ut.repr2(timedelta_list, precision=1)
        >>> print(result)
        np.array([7.6e+07, 7.6e+07, 2.4e+06, 2.0e+08, 9.7e+07])
    """
    unixtime_list1 = ibs.get_annot_image_unixtimes_asfloat(aid_list1)
    unixtime_list2 = ibs.get_annot_image_unixtimes_asfloat(aid_list2)
    timedelta_list = np.abs(unixtime_list1 - unixtime_list2)
    return timedelta_list


@register_ibs_method
def get_annotedge_timedelta(ibs, edges):
    return ibs.get_annot_pair_timedelta(*zip(*edges))


@register_ibs_method
def get_annotedge_viewdist(ibs, edges):
    edges = np.array(edges)
    unique_annots = ibs.annots(np.unique(edges)).view()
    annots1 = unique_annots.view(edges.T[0])
    annots2 = unique_annots.view(edges.T[1])
    view_ints1 = annots1.viewpoint_int
    view_ints2 = annots2.viewpoint_int

    DIST = ibs.const.VIEW.DIST
    view_dists = [
        DIST[tup] if tup in DIST else DIST[tup[::-1]]
        for tup in zip(view_ints1, view_ints2)
    ]
    view_dists = np.array(ut.replace_nones(view_dists, np.nan))
    return view_dists


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
        python -m wbia.annotmatch_funcs --test-get_annot_num_reviewed_matching_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb2')
        >>> aid1_list = ibs.get_valid_aids()
        >>> eager = True
        >>> nInput = None
        >>> num_annot_reviewed_list = get_annot_num_reviewed_matching_aids(ibs, aid_list, eager, nInput)
        >>> result = str(num_annot_reviewed_list)
        >>> print(result)
    """
    aids_list = ibs.get_annot_reviewed_matching_aids(
        aid1_list, eager=eager, nInput=nInput
    )
    num_annot_reviewed_list = list(map(len, aids_list))
    return num_annot_reviewed_list


@register_ibs_method
def get_annot_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    """
    Returns a list of the aids that were reviewed as candidate matches to the input aid
    """
    ANNOT_ROWID1 = 'annot_rowid1'
    ANNOT_ROWID2 = 'annot_rowid2'
    params_iter = [(aid,) for aid in aid_list]
    colnames = (ANNOT_ROWID2,)
    where_colnames = (ANNOT_ROWID1,)
    aids_list = ibs.db.get_where_eq(
        ibs.const.ANNOTMATCH_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        unpack_scalars=False,
        nInput=nInput,
    )
    return aids_list


@register_ibs_method
def get_annotmatch_aids(ibs, annotmatch_rowid_list):
    ANNOT_ROWID1 = 'annot_rowid1'
    ANNOT_ROWID2 = 'annot_rowid2'
    id_iter = annotmatch_rowid_list
    colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    aid_pairs = ibs.db.get(
        ibs.const.ANNOTMATCH_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return aid_pairs


@register_ibs_method
def get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list):
    r"""
    Args:
        aid1_list (list):
        aid2_list (list):

    Returns:
        list: annotmatch_reviewed_list

    CommandLine:
        python -m wbia.annotmatch_funcs --test-get_annot_pair_is_reviewed

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> pairs = list(ut.product(aid_list, aid_list))
        >>> aid1_list = ut.get_list_column(pairs, 0)
        >>> aid2_list = ut.get_list_column(pairs, 1)
        >>> annotmatch_reviewed_list = get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list)
        >>> reviewed_pairs = ut.compress(pairs, annotmatch_reviewed_list)
        >>> result = len(reviewed_pairs)
        >>> print(result)
        104
    """
    am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(aid1_list, aid2_list)
    return [
        None if user is None else user.startswith('user:')
        for user in ibs.get_annotmatch_reviewer(am_rowids)
    ]


@register_ibs_method
def set_annot_pair_as_reviewed(ibs, aid1, aid2):
    """ denote that this match was reviewed and keep whatever status it is given """
    isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
    if isunknown1 or isunknown2:
        truth = ibs.const.EVIDENCE_DECISION.UNKNOWN
    else:
        nid1, nid2 = ibs.get_annot_name_rowids((aid1, aid2))
        truth = (
            ibs.const.EVIDENCE_DECISION.POSITIVE
            if (nid1 == nid2)
            else ibs.const.EVIDENCE_DECISION.NEGATIVE
        )

    # Ensure a row exists for this pair
    annotmatch_rowids = ibs.add_annotmatch_undirected([aid1], [aid2])

    # Old functionality, remove. Reviewing should not set truth
    confidence = ibs.const.CONFIDENCE.CODE_TO_INT['guessing']
    ibs.set_annotmatch_evidence_decision(annotmatch_rowids, [truth])
    user_id = ut.get_user_name() + '@' + ut.get_computer_name()
    ibs.set_annotmatch_reviewer(annotmatch_rowids, ['user:' + user_id])
    ibs.set_annotmatch_confidence(annotmatch_rowids, [confidence])
    print('... set truth=%r' % (truth,))


@register_ibs_method
def set_annot_pair_as_positive_match(
    ibs, aid1, aid2, dryrun=False, on_nontrivial_merge=None, logger=None
):
    """
    Safe way to perform links. Errors on invalid operations.

    TODO: ELEVATE THIS FUNCTION
    Change into make_task_set_annot_pair_as_positive_match and it returns what
    needs to be done.

    Need to test several cases:
        uknown, unknown
        knownA, knownA
        knownB, knownA
        unknown, knownA
        knownA, unknown

    Args:
        ibs (IBEISController):  wbia controller object
        aid1 (int):  query annotation id
        aid2 (int):  matching annotation id

    CommandLine:
        python -m wbia.annotmatch_funcs --test-set_annot_pair_as_positive_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> status = set_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun)
        >>> print(status)
    """

    def _set_annot_name_rowids(aid_list, nid_list):
        if not ut.QUIET:
            print('... _set_annot_name_rowids(aids=%r, nids=%r)' % (aid_list, nid_list))
            print('... names = %r' % (ibs.get_name_texts(nid_list)))
        assert len(aid_list) == len(nid_list), 'list must correspond'
        if not dryrun:
            if logger is not None:
                log = logger.info
                previous_names = ibs.get_annot_names(aid_list)
                new_names = ibs.get_name_texts(nid_list)
                annot_uuids = ibs.get_annot_uuids(aid_list)
                annot_uuid_pair = ibs.get_annot_uuids((aid1, aid2))
                log(
                    (
                        'REVIEW_PAIR AS TRUE: (annot_uuid_pair=%r) '
                        'CHANGE NAME of %d (annot_uuids=%r) '
                        'WITH (previous_names=%r) to (new_names=%r)'
                    )
                    % (
                        annot_uuid_pair,
                        len(annot_uuids),
                        annot_uuids,
                        previous_names,
                        new_names,
                    )
                )

            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.set_annot_pair_as_reviewed(aid1, aid2)
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
        ibs.set_annot_pair_as_reviewed(aid1, aid2)
        if logger is not None:
            log = logger.info
            annot_uuid_pair = ibs.get_annot_uuids((aid1, aid2))
            log('REVIEW_PAIR AS TRUE: (annot_uuid_pair=%r) NO CHANGE' % annot_uuid_pair)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...match unknown1 to unknown2 into 1 new name')
            next_nids = ibs.make_next_nids(num=1)
            status = _set_annot_name_rowids([aid1, aid2], next_nids * 2)
        elif not isunknown1 and not isunknown2:
            print('...merge known1 into known2')
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
            trivial_merge = (
                len(aid1_and_groundtruth) == 1 and len(aid2_and_groundtruth) == 1
            )
            if not trivial_merge:
                if on_nontrivial_merge is None:
                    raise Exception('no function is set up to handle nontrivial merges!')
                else:
                    on_nontrivial_merge(ibs, aid1, aid2)
            status = _set_annot_name_rowids(
                aid1_and_groundtruth, [nid2] * len(aid1_and_groundtruth)
            )
        elif isunknown2 and not isunknown1:
            print('...match unknown2 into known1')
            status = _set_annot_name_rowids([aid2], [nid1])
        elif isunknown1 and not isunknown2:
            print('...match unknown1 into known2')
            status = _set_annot_name_rowids([aid1], [nid2])
        else:
            raise AssertionError('impossible state')
    return status


@register_ibs_method
def set_annot_pair_as_negative_match(
    ibs, aid1, aid2, dryrun=False, on_nontrivial_split=None, logger=None
):
    """
    TODO: ELEVATE THIS FUNCTION

    Args:
        ibs (IBEISController):  wbia controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        dryrun (bool):

    CommandLine:
        python -m wbia.annotmatch_funcs --test-set_annot_pair_as_negative_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> result = set_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun)
        >>> print(result)
    """

    def _set_annot_name_rowids(aid_list, nid_list):
        print('... _set_annot_name_rowids(%r, %r)' % (aid_list, nid_list))
        if not dryrun:
            if logger is not None:
                log = logger.info
                previous_names = ibs.get_annot_names(aid_list)
                new_names = ibs.get_name_texts(nid_list)
                annot_uuids = ibs.get_annot_uuids(aid_list)
                annot_uuid_pair = ibs.get_annot_uuids((aid1, aid2))
                log(
                    (
                        'REVIEW_PAIR AS FALSE: (annot_uuid_pair=%r) '
                        'CHANGE NAME of %d (annot_uuids=%r) '
                        'WITH (previous_names=%r) to (new_names=%r)'
                    )
                    % (
                        annot_uuid_pair,
                        len(annot_uuids),
                        annot_uuids,
                        previous_names,
                        new_names,
                    )
                )
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.set_annot_pair_as_reviewed(aid1, aid2)

    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('images are marked as having the same name... we must tread carefully')
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        if len(aid1_groundtruth) == 1 and aid1_groundtruth == [aid2]:
            # this is the only safe case for same name split
            # Change so the names are not the same
            next_nids = ibs.make_next_nids(num=1)
            status = _set_annot_name_rowids([aid1], next_nids)
        else:
            if on_nontrivial_split is None:
                raise Exception('no function is set up to handle nontrivial splits!')
            else:
                on_nontrivial_split(ibs, aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...nomatch unknown1 and unknown2 into 2 new names')
            next_nids = ibs.make_next_nids(num=2)
            status = _set_annot_name_rowids([aid1, aid2], next_nids)
        elif not isunknown1 and not isunknown2:
            print('...nomatch known1 and known2... nothing to do (yet)')
            ibs.set_annot_pair_as_reviewed(aid1, aid2)
            status = None
            if logger is not None:
                log = logger.info
                annot_uuid_pair = ibs.get_annot_uuids((aid1, aid2))
                log(
                    'REVIEW_PAIR AS FALSE: (annot_uuid_pair=%r) NO CHANGE'
                    % annot_uuid_pair
                )
        elif isunknown2 and not isunknown1:
            print('...nomatch unknown2 -> newname and known1')
            next_nids = ibs.make_next_nids(num=1)
            status = _set_annot_name_rowids([aid2], next_nids)
        elif isunknown1 and not isunknown2:
            print('...nomatch unknown1 -> newname and known2')
            next_nids = ibs.make_next_nids(num=1)
            status = _set_annot_name_rowids([aid1], next_nids)
        else:
            raise AssertionError('impossible state')
    return status


@register_ibs_method
def get_match_truth(ibs, aid1, aid2):
    return ibs.get_match_truths([aid1], [aid2])[0]


@register_ibs_method
def get_match_truths(ibs, aids1, aids2):
    r"""
    Uses NIDS to verify truth.
    TODO: rectify with annotmatch table

    Args:
        ibs (IBEISController):  wbia controller object
        aids1 (list):
        aids2 (list):

    Returns:
        list[int]: truth_codes - see
            wbia.constants.EVIDENCE_DECISION.INT_TO_CODE for code definitions

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_match_truths

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annotmatch_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aids1 = ibs.get_valid_aids()
        >>> aids2 = ut.list_roll(ibs.get_valid_aids(), -1)
        >>> truth_codes = get_match_truths(ibs, aids1, aids2)
        >>> print('truth_codes = %s' % ut.repr2(truth_codes))
        >>> target = np.array([3, 1, 3, 3, 1, 0, 0, 3, 3, 3, 3, 0, 3])
        >>> assert np.all(truth_codes == target)
    """
    nids1 = np.array(ibs.get_annot_name_rowids(aids1))
    nids2 = np.array(ibs.get_annot_name_rowids(aids2))
    isunknowns1 = np.array(ibs.is_nid_unknown(nids1))
    isunknowns2 = np.array(ibs.is_nid_unknown(nids2))
    any_unknown = np.logical_or(isunknowns1, isunknowns2)
    truth_codes = np.array((nids1 == nids2), dtype=np.int32)
    truth_codes[any_unknown] = ibs.const.EVIDENCE_DECISION.UNKNOWN
    return truth_codes


@register_ibs_method
def get_match_text(ibs, aid1, aid2):
    truth = ibs.get_match_truth(aid1, aid2)
    text = ibs.const.EVIDENCE_DECISION.INT_TO_NICE.get(truth, None)
    return text


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.annotmatch_funcs
        python -m wbia.annotmatch_funcs --allexamples
        python -m wbia.annotmatch_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
