# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'wbia.templates.template_generator')"
sh Tgen.sh --key review --invert --Tcfg with_getters=True with_setters=False --modfname manual_review_funcs

# TODO: Fix this name it is too special case
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
from six.moves import zip, map, reduce

# import numpy as np
# import vtool as vt
import numpy as np
import ubelt as ub  # NOQA
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject  # NOQA
import utool as ut
import uuid
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


VERBOSE_SQL = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


REVIEW_ROWID = 'review_rowid'
REVIEW_UUID = 'review_uuid'
REVIEW_AID1 = 'annot_1_rowid'
REVIEW_AID2 = 'annot_2_rowid'
REVIEW_COUNT = 'review_count'
REVIEW_EVIDENCE_DECISION = 'review_evidence_decision'
REVIEW_META_DECISION = 'review_meta_decision'
REVIEW_USER_IDENTITY = 'review_user_identity'
REVIEW_USER_CONFIDENCE = 'review_user_confidence'
REVIEW_TAGS = 'review_tags'
REVIEW_TIME_CLIENT_START = 'review_client_start_time_posix'
REVIEW_TIME_CLIENT_END = 'review_client_end_time_posix'
REVIEW_TIME_SERVER_START = 'review_server_start_time_posix'
REVIEW_TIME_SERVER_END = 'review_server_end_time_posix'


def e_(u, v):
    return (u, v) if u < v else (v, u)


def hack_create_aidpair_index(ibs):
    # HACK IN INDEX
    sqlfmt = ut.codeblock(
        """
        CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({index_cols});
        """
    )
    sqlcmd = sqlfmt.format(
        index_name='aidpair_to_rowid',
        table=ibs.const.REVIEW_TABLE,
        index_cols=','.join([REVIEW_AID1, REVIEW_AID2]),
    )
    ibs.staging.connection.execute(sqlcmd).fetchall()
    sqlcmd = sqlfmt.format(
        index_name='aid1_to_rowids',
        table=ibs.const.REVIEW_TABLE,
        index_cols=','.join([REVIEW_AID1]),
    )
    ibs.staging.connection.execute(sqlcmd).fetchall()
    sqlcmd = sqlfmt.format(
        index_name='aid2_to_rowids',
        table=ibs.const.REVIEW_TABLE,
        index_cols=','.join([REVIEW_AID2]),
    )
    ibs.staging.connection.execute(sqlcmd).fetchall()


@register_ibs_method
@accessor_decors.ider
@register_api('/api/review/', methods=['GET'])
def _get_all_review_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    all_known_review_rowids = ibs.staging.get_all_rowids(const.REVIEW_TABLE)
    return all_known_review_rowids


@register_ibs_method
def _set_review_uuids(ibs, review_rowid_list, review_uuid_list):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    id_iter = ((review_rowid,) for review_rowid in review_rowid_list)
    val_iter = ((review_uuid,) for review_uuid in review_uuid_list)
    ibs.staging.set(const.REVIEW_TABLE, (REVIEW_UUID,), val_iter, id_iter)


@register_ibs_method
def get_review_rowid_from_superkey(
    ibs, aid_1_list, aid_2_list, count_list, eager=False, nInput=None
):
    """ Returns review_rowid_list

    Args:
        superkey lists: review_rowid_list, aid_list

    Returns:
        review_rowid_list
    """
    colnames = (REVIEW_ROWID,)
    params_iter = zip(aid_1_list, aid_2_list, count_list)
    where_colnames = [REVIEW_AID1, REVIEW_AID2, REVIEW_COUNT]
    review_rowid_list = list(
        ibs.staging.get_where_eq(
            const.REVIEW_TABLE,
            colnames,
            params_iter,
            where_colnames,
            eager=eager,
            nInput=nInput,
        )
    )
    return review_rowid_list


@register_ibs_method
@accessor_decors.adder
@register_api('/api/review/', methods=['POST'])
def add_review(
    ibs,
    aid_1_list,
    aid_2_list,
    evidence_decision_list,
    meta_decision_list=None,
    review_uuid_list=None,
    identity_list=None,
    user_confidence_list=None,
    tags_list=None,
    review_client_start_time_posix=None,
    review_client_end_time_posix=None,
    review_server_start_time_posix=None,
    review_server_end_time_posix=None,
):
    r"""
    Adds a list of reviews.

    Returns:
        list: review_id_list - review rowids

    RESTful:
        Method: POST
        URL:    /api/review/

    CommandLine:
        python -m wbia.control.manual_review_funcs --test-add_review

    Doctest:
        >>> import wbia
        >>> from wbia.control.manual_review_funcs import *
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.staging.get_table_as_pandas('reviews')
        >>> # ensure it is empty
        >>> rowids = ibs.staging.get_all_rowids('reviews')
        >>> ibs.staging.delete_rowids('reviews', rowids)
        >>> ut.exec_funckw(ibs.add_review, globals())
        >>> # Add some dummy reviews
        >>> aid_1_list = [1, 2, 3, 2]
        >>> aid_2_list = [2, 3, 4, 3]
        >>> evidence_decision_list = [1, 0, 1, 2]
        >>> new_rowids = ibs.add_review(aid_1_list, aid_2_list,
        >>>                             evidence_decision_list)
        >>> assert new_rowids == [1, 2, 3, 4]
        >>> table = ibs.staging.get_table_as_pandas('reviews')
        >>> print(table)
        >>> # Then delete them
        >>> ibs.staging.delete_rowids('reviews', new_rowids)
    """
    assert len(aid_1_list) == len(aid_2_list)
    assert len(aid_1_list) == len(evidence_decision_list)
    diff_list = -np.array(aid_2_list)
    assert np.all(diff_list != 0), 'Cannot add a review state between an aid and itself'
    n_input = len(aid_1_list)

    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    aid_pair_list = [e_(u, v) for u, v in zip(aid_1_list, aid_2_list)]
    aid_1_list = [pair[0] for pair in aid_pair_list]
    aid_2_list = [pair[1] for pair in aid_pair_list]

    if True:
        # Get current review counts from database
        unique_pairs = list(set(aid_pair_list))
        count_base = [
            0 if counts is None or len(counts) == 0 else max(max(counts), len(counts))
            for counts in ibs.get_review_counts_from_pairs(unique_pairs)
        ]
        pair_to_count = {edge: count for edge, count in zip(unique_pairs, count_base)}
        count_list = []
        for edge in aid_pair_list:
            pair_to_count[edge] += 1
            count = pair_to_count[edge]
            count_list.append(count)
    # else:
    #     # Alternative implemenation
    #     unique_pairs, groupxs = ut.group_indices(aid_pair_list)
    #     unique_base = [
    #         0 if counts is None or len(counts) == 0 else
    #         max(max(counts), len(counts))
    #         for counts in ibs.get_review_counts_from_pairs(unique_pairs)
    #     ]
    #     grouped_base = [[b] * len(g) for b, g in zip(unique_base, groupxs)]
    #     grouped_offsets = [list(range(n)) for n in map(len, groupxs)]
    #     base = np.array(ut.ungroup(grouped_base, groupxs))
    #     offsets = np.array(ut.ungroup(grouped_offsets, groupxs))
    #     count_list = offsets + base + 1

    if review_uuid_list is None:
        review_uuid_list = [uuid.uuid4() for _ in range(n_input)]
    if meta_decision_list is None:
        meta_decision_list = [None for _ in range(n_input)]
    if identity_list is None:
        # identity_list = [ut.get_computer_name()] * len(aid_1_list)
        identity_list = [None] * n_input
    if tags_list is None:
        tag_str_list = [None] * n_input
    else:
        tag_str_list = [';'.join(map(str, tag_list)) for tag_list in tags_list]
    if user_confidence_list is None:
        user_confidence_list = [None] * n_input

    if review_client_start_time_posix is None:
        review_client_start_time_posix = [None] * n_input
    if review_client_end_time_posix is None:
        review_client_end_time_posix = [None] * n_input
    if review_server_start_time_posix is None:
        review_server_start_time_posix = [None] * n_input
    if review_server_end_time_posix is None:
        review_server_end_time_posix = [None] * n_input

    assert n_input == len(identity_list)
    assert n_input == len(tag_str_list)
    assert n_input == len(user_confidence_list)
    assert n_input == len(review_uuid_list)
    assert n_input == len(count_list)

    superkey_paramx = (
        0,
        1,
        2,
    )
    # TODO Allow for better ensure=False without using partial
    # Just autogenerate these functions
    colnames = [
        REVIEW_AID1,
        REVIEW_AID2,
        REVIEW_COUNT,
        REVIEW_UUID,
        REVIEW_EVIDENCE_DECISION,
        REVIEW_META_DECISION,
        REVIEW_USER_IDENTITY,
        REVIEW_USER_CONFIDENCE,
        REVIEW_TAGS,
        REVIEW_TIME_CLIENT_START,
        REVIEW_TIME_CLIENT_END,
        REVIEW_TIME_SERVER_START,
        REVIEW_TIME_SERVER_END,
    ]
    params_iter = list(
        zip(
            aid_1_list,
            aid_2_list,
            count_list,
            review_uuid_list,
            evidence_decision_list,
            meta_decision_list,
            identity_list,
            user_confidence_list,
            tag_str_list,
            review_client_start_time_posix,
            review_client_end_time_posix,
            review_server_start_time_posix,
            review_server_end_time_posix,
        )
    )
    review_rowid_list = ibs.staging.add_cleanly(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        ibs.get_review_rowid_from_superkey,
        superkey_paramx,
    )
    return review_rowid_list


@register_ibs_method
@accessor_decors.deleter
# @cache_invalidator(const.REVIEW_TABLE)
@register_api('/api/review/', methods=['DELETE'])
def delete_review(ibs, review_rowid_list):
    r"""
    deletes reviews from the database

    RESTful:
        Method: DELETE
        URL:    /api/review/
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d reviews' % len(review_rowid_list))
    ibs.staging.delete_rowids(const.REVIEW_TABLE, review_rowid_list)


@register_ibs_method
def get_review_rowids_from_edges(ibs, edges, eager=True, nInput=None, directed=False):
    colnames = (REVIEW_ROWID,)
    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    # params_iter = edges
    if directed:
        params_iter = edges
    else:
        params_iter = [e_(u, v) for u, v in edges]
    where_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_rowids_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return review_rowids_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_review_exists_from_edges(ibs, edges, eager=True, nInput=None):
    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    # params_iter = (e_(u, v) for u, v in edges)
    params_iter = edges
    where_colnames = [REVIEW_AID1, REVIEW_AID2]
    exists_list = ibs.staging.exists_where_eq(
        const.REVIEW_TABLE,
        params_iter,
        where_colnames,
        eager=False,
        nInput=nInput,
        unpack_scalars=True,
    )
    exists_list = map(bool, exists_list)
    if eager:
        exists_list = list(exists_list)
    return exists_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/review/rowids/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_rowids_from_aid_tuple(
    ibs, aid_1_list, aid_2_list, eager=True, nInput=None
):
    r"""
    Aid pairs are undirected

    Returns:
        list_ (list): review_rowid_list - review rowid list of lists

    RESTful:
        Method: GET
        URL:    /api/review/rowid/tuple/
    """
    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    edges = (e_(u, v) for u, v in zip(aid_1_list, aid_2_list))
    return get_review_rowids_from_edges(ibs, edges, eager=eager, nInput=nInput)
    # colnames = (REVIEW_ROWID,)
    # where_colnames = [REVIEW_AID1, REVIEW_AID2]
    # review_rowids_list = ibs.staging.get_where_eq(
    #     const.REVIEW_TABLE, colnames, params_iter, where_colnames,
    #     eager=eager, nInput=nInput, unpack_scalars=False)
    # return review_rowids_list


@register_ibs_method
def get_review_rowids_between(ibs, aids1, aids2=None, method=None):
    """
    Find staging rowids between sets of aids

    Doctest:
        >>> from wbia.control.manual_review_funcs import *
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> aids1 = aids2 = [1, 2, 3, 4, 5, 6]
        >>> rowids_between = ibs.get_review_rowids_between
        >>> ids1 = sorted(rowids_between(aids1, aids2, method=1))
        >>> ids2 = sorted(rowids_between(aids1, aids2, method=2))
        >>> assert len(ub.find_duplicates(ids1)) == 0
        >>> assert len(ub.find_duplicates(ids2)) == 0
        >>> assert ids1 == ids2
    """
    if aids2 is None:
        aids2 = aids1
    if method is None:
        if len(aids1) * len(aids2) > 5000:
            method = 1
        else:
            method = 2

    if method == 1:
        # Strategy 1: get all existing rows and see what intersects
        # This is better when the enumerated set of rows would be larger than
        # the database size
        rowids11 = set(ut.flatten(ibs.get_review_rowids_from_aid1(aids1)))
        rowids12 = set(ut.flatten(ibs.get_review_rowids_from_aid2(aids1)))
        if aids1 is aids2:
            rowids = list(reduce(set.intersection, [rowids11, rowids12]))
        else:
            rowids21 = set(ut.flatten(ibs.get_review_rowids_from_aid1(aids2)))
            rowids22 = set(ut.flatten(ibs.get_review_rowids_from_aid2(aids2)))
            rowids = list(
                reduce(set.intersection, [rowids11, rowids12, rowids21, rowids22])
            )
    elif method == 2:
        # Strategy 2: enumerate what rows could exist and see what does exist
        # This is better when the enumerated set of rows would be smaller than
        # the database size
        edges = list(ut.product_nonsame(aids1, aids2))
        if len(edges) == 0:
            rowids = []
        else:
            rowids = ibs.get_review_rowids_from_edges(edges, directed=True)
            if rowids is None:
                rowids = []
            rowids = ut.filter_Nones(rowids)
            rowids = ut.flatten(rowids)
    else:
        raise ValueError('no method=%r' % (method,))
    return rowids


# @register_ibs_method
# @accessor_decors.getter_1to1
# @register_api('/api/review/rowids/undirected/', methods=['GET'], __api_plural_check__=False)
# def get_review_rowids_from_undirected_tuple(ibs, aid_1_list, aid_2_list):
#     aids1, aids2 = aid_1_list, aid_2_list
#     review_rowids_dir1 = ibs.get_review_rowids_from_aid_tuple(aids1, aids2)
#     review_rowids_dir2 = ibs.get_review_rowids_from_aid_tuple(aids2, aids1)
#     def _join_rowids(rowids1, rowids2):
#         if rowids1 is None and rowids2 is None:
#             return None
#         else:
#             if rowids1 is None:
#                 rowids1 = []
#             elif rowids2 is None:
#                 rowids2 = []
#             return rowids1 + rowids2
#     review_rowids_list = [
#         _join_rowids(rowids1, rowids2)
#         for rowids1, rowids2 in zip(review_rowids_dir1, review_rowids_dir2)
#     ]
#     return review_rowids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/count/', methods=['GET'])
def get_review_count(ibs, review_rowid_list):
    review_count_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_COUNT,), review_rowid_list
    )
    return review_count_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/counts/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_counts_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_counts_list - review counts

    RESTful:
        Method: GET
        URL:    /api/review/counts/tuple/
    """
    aid_pairs = zip(aid_1_list, aid_2_list)
    review_counts_list = ibs.get_review_counts_from_pairs(aid_pairs)
    return review_counts_list


@register_ibs_method
def get_review_counts_from_pairs(ibs, aid_pairs, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_counts_list - review counts

    RESTful:
        Method: GET
        URL:    /api/review/counts/tuple/
    """
    colnames = (REVIEW_COUNT,)
    params_iter = aid_pairs
    where_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_counts_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return review_counts_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decision/', methods=['GET'])
def get_review_decision(ibs, review_rowid_list):
    review_decision_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_EVIDENCE_DECISION,), review_rowid_list
    )
    return review_decision_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/uuid/', methods=['GET'])
def get_review_uuid(ibs, review_rowid_list):
    review_uuid_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_UUID,), review_rowid_list
    )
    return review_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decision/str/', methods=['GET'])
def get_review_decision_str(ibs, review_rowid_list):
    review_decision_list = ibs.get_review_decision(review_rowid_list)
    review_decision_str_list = [
        const.EVIDENCE_DECISION.INT_TO_NICE.get(review_decision)
        for review_decision in review_decision_list
    ]
    return review_decision_str_list


@register_ibs_method
@register_api('/api/review/decisions/only/', methods=['GET'], __api_plural_check__=False)
def get_review_decisions_from_only(ibs, aid_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_tuple_decisions_list - review decisions

    RESTful:
        Method: GET
        URL:    /api/review/decisions/only/
    """
    colnames = (
        REVIEW_AID1,
        REVIEW_AID2,
        REVIEW_EVIDENCE_DECISION,
    )
    params_iter = [(aid,) for aid in aid_list]
    where_clause = '%s=?' % (REVIEW_AID1)
    review_tuple_decisions_list = ibs.staging.get_where(
        const.REVIEW_TABLE, colnames, params_iter, where_clause, unpack_scalars=False
    )
    return review_tuple_decisions_list


@register_ibs_method
@register_api('/api/review/rowids/only/', methods=['GET'], __api_plural_check__=False)
def get_review_rowids_from_only(ibs, aid_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_rowids

    RESTful:
        Method: GET
        URL:    /api/review/rowids/only/
    """
    colnames = (REVIEW_ROWID,)
    params_iter = [(aid,) for aid in aid_list]
    where_clause = '%s=?' % (REVIEW_AID1)
    review_rowids = ibs.staging.get_where(
        const.REVIEW_TABLE, colnames, params_iter, where_clause, unpack_scalars=False
    )
    return review_rowids


@register_ibs_method
def get_review_rowids_from_single(ibs, aid_list, eager=True, nInput=None):
    colnames = (REVIEW_ROWID,)
    params_iter = [(aid, aid) for aid in aid_list]
    where_clause = '%s=? OR %s=?' % (REVIEW_AID1, REVIEW_AID2)
    review_rowids = ibs.staging.get_where(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_clause=where_clause,
        unpack_scalars=False,
    )
    return review_rowids


@register_ibs_method
def get_review_rowids_from_aid1(ibs, aid_list, eager=True, nInput=None):
    colnames = (REVIEW_ROWID,)
    params_iter = [(aid,) for aid in aid_list]
    where_clause = '%s=?' % (REVIEW_AID1,)
    review_rowids = ibs.staging.get_where(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_clause=where_clause,
        unpack_scalars=False,
    )
    return review_rowids


@register_ibs_method
def get_review_rowids_from_aid2(ibs, aid_list, eager=True, nInput=None):
    colnames = (REVIEW_ROWID,)
    params_iter = [(aid,) for aid in aid_list]
    where_clause = '%s=?' % (REVIEW_AID2,)
    review_rowids = ibs.staging.get_where(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_clause=where_clause,
        unpack_scalars=False,
    )
    return review_rowids


# @register_ibs_method
# @accessor_decors.getter_1to1
# @register_api('/api/review/decisions/single/', methods=['GET'], __api_plural_check__=False)
# def get_review_decisions_from_single(ibs, aid_list, eager=True, nInput=None):
#     r"""
#     Returns:
#         list_ (list): review_tuple_decisions_list - review decisions

#     RESTful:
#         Method: GET
#         URL:    /api/review/identities/single/
#     """
#     colnames = (REVIEW_AID1, REVIEW_AID2, REVIEW_EVIDENCE_DECISION,)
#     params_iter = zip(aid_list, aid_list, )
#     where_colnames = [REVIEW_AID1, REVIEW_AID2]
#     review_tuple_decisions_list = ibs.staging.get_where_eq(
#         const.REVIEW_TABLE, colnames, params_iter, where_colnames,
#         eager=eager, nInput=nInput, op='OR', unpack_scalars=False)
#     return review_tuple_decisions_list


# @register_ibs_method
# @accessor_decors.getter_1to1
# @register_api('/api/review/decisions/tuple/', methods=['GET'], __api_plural_check__=False)
# def get_review_decisions_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
#     r"""
#     Returns:
#         list_ (list): review_decisions_list - review decisions

#     RESTful:
#         Method: GET
#         URL:    /api/review/identities/tuple/
#     """
#     colnames = (REVIEW_EVIDENCE_DECISION,)
#     params_iter = zip(aid_1_list, aid_2_list)
#     where_colnames = [REVIEW_AID1, REVIEW_AID2]
#     review_decisions_list = ibs.staging.get_where_eq(
#         const.REVIEW_TABLE, colnames, params_iter, where_colnames,
#         eager=eager, nInput=nInput, unpack_scalars=False)
#     return review_decisions_list


# @register_ibs_method
# @accessor_decors.getter_1to1
# @register_api('/api/review/decisions/str/tuple/', methods=['GET'], __api_plural_check__=False)
# def get_review_decisions_str_from_tuple(ibs, aid_1_list, aid_2_list, **kwargs):
#     r"""
#     Returns:
#         list_ (list): review_decisions_list - review decisions

#     RESTful:
#         Method: GET
#         URL:    /api/review/identities/str/tuple/
#     """
#     review_decisions_list = ibs.get_review_decisions_from_tuple(
#         aid_1_list, aid_2_list, **kwargs)
#     review_decision_str_list = [
#         [
#             const.EVIDENCE_DECISION.INT_TO_NICE.get(review_decision)
#             for review_decision in review_decision_list
#         ]
#         for review_decision_list in review_decisions_list
#     ]
#     return review_decision_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/identity/', methods=['GET'])
def get_review_identity(ibs, review_rowid_list):
    review_identity_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_USER_IDENTITY,), review_rowid_list
    )
    return review_identity_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/confidence/', methods=['GET'])
def get_review_user_confidence(ibs, review_rowid_list):
    user_confidence_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_USER_CONFIDENCE,), review_rowid_list
    )
    return user_confidence_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api(
    '/api/review/identities/tuple/', methods=['GET'], __api_plural_check__=False
)
def get_review_identities_from_tuple(
    ibs, aid_1_list, aid_2_list, eager=True, nInput=None
):
    r"""
    Returns:
        list_ (list): review_identities_list - review identities

    RESTful:
        Method: GET
        URL:    /api/review/identities/tuple/
    """
    colnames = (REVIEW_USER_IDENTITY,)
    params_iter = zip(aid_1_list, aid_2_list)
    where_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_identities_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return review_identities_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/posix/', methods=['GET'])
def get_review_posix_time(ibs, review_rowid_list):
    return ibs.get_review_posix_server_end_time(review_rowid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/server/start/posix/', methods=['GET'])
def get_review_posix_server_start_time(ibs, review_rowid_list):
    review_posix_time_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_TIME_SERVER_START,), review_rowid_list
    )
    return review_posix_time_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/client/start/posix/', methods=['GET'])
def get_review_posix_client_start_time(ibs, review_rowid_list):
    review_posix_time_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_TIME_CLIENT_START,), review_rowid_list
    )
    return review_posix_time_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/client/end/posix/', methods=['GET'])
def get_review_posix_client_end_time(ibs, review_rowid_list):
    review_posix_time_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_TIME_CLIENT_END,), review_rowid_list
    )
    return review_posix_time_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/server/end/posix/', methods=['GET'])
def get_review_posix_server_end_time(ibs, review_rowid_list):
    review_posix_time_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_TIME_SERVER_END,), review_rowid_list
    )
    return review_posix_time_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_review_aid_tuple(ibs, review_rowid_list, eager=True, nInput=None):
    colnames = (
        REVIEW_AID1,
        REVIEW_AID2,
    )
    params_iter = zip(review_rowid_list)
    where_colnames = [REVIEW_ROWID]
    aid_tuple_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
    )
    return aid_tuple_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/posix/tuple/', methods=['GET'])
def get_review_posix_times_from_tuple(
    ibs, aid_1_list, aid_2_list, eager=True, nInput=None
):
    r"""
    Returns:
        list_ (list): identity_list - review posix times

    RESTful:
        Method: GET
        URL:    /api/review/time/posix/tuple/
    """
    colnames = (REVIEW_TIME_SERVER_END,)
    params_iter = zip(aid_1_list, aid_2_list)
    where_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_posix_times_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return review_posix_times_list


# def _parse_tag_str(tag_str):
#     if tag_str is None or len(tag_str) == 0:
#         return None
#     else:
#         return tag_str.split(';')


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/tags/', methods=['GET'], __api_plural_check__=False)
def get_review_tags(ibs, review_rowid_list):
    review_tag_str_list = ibs.staging.get(
        const.REVIEW_TABLE, (REVIEW_TAGS,), review_rowid_list
    )
    review_tags_list = [
        None
        if review_tag_str is None or len(review_tag_str) == 0
        else review_tag_str.split(';')
        for review_tag_str in review_tag_str_list
    ]
    return review_tags_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/tags/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_tags_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_tags_list - review tags (list of strings)

    RESTful:
        Method: GET
        URL:    /api/review/tags/tuple/
    """
    colnames = (REVIEW_TAGS,)
    params_iter = zip(aid_1_list, aid_2_list)
    where_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_tag_strs_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    review_tags_list = [
        []
        if review_tag_str_list is None
        else [
            None
            if review_tag_str is None or len(review_tag_str) == 0
            else review_tag_str.split(';')
            for review_tag_str in review_tag_str_list
        ]
        for review_tag_str_list in review_tag_strs_list
    ]
    return review_tags_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/metadata/', methods=['GET'])
def get_review_metadata(ibs, review_rowid_list, return_raw=False):
    r"""
    Returns:
        list_ (list): review metadata dictionary

    RESTful:
        Method: GET
        URL:    /api/review/metadata/
    """
    metadata_str_list = ibs.staging.get(
        const.REVIEW_TABLE, ('review_metadata_json',), review_rowid_list
    )
    metadata_list = []
    for metadata_str in metadata_str_list:
        if metadata_str in [None, '']:
            metadata_dict = {}
        else:
            metadata_dict = metadata_str if return_raw else ut.from_json(metadata_str)
        metadata_list.append(metadata_dict)
    return metadata_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/review/metadata/', methods=['PUT'])
def set_review_metadata(ibs, review_rowid_list, metadata_dict_list):
    r"""
    Sets the review's metadata using a metadata dictionary

    RESTful:
        Method: PUT
        URL:    /api/review/metadata/

    CommandLine:
        python -m wbia.control.manual_review_funcs --test-set_review_metadata

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_review_funcs import *  # NOQA
        >>> import wbia
        >>> import random
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> review_rowid_list = ibs.add_review([1], [2], [0])
        >>> metadata_dict_list = [
        >>>     {'test': random.uniform(0.0, 1.0)},
        >>> ]
        >>> print(ut.repr2(metadata_dict_list))
        >>> ibs.set_review_metadata(review_rowid_list, metadata_dict_list)
        >>> # verify results
        >>> metadata_dict_list_ = ibs.get_review_metadata(review_rowid_list)
        >>> print(ut.repr2(metadata_dict_list_))
        >>> assert metadata_dict_list == metadata_dict_list_
        >>> metadata_str_list = [ut.to_json(metadata_dict) for metadata_dict in metadata_dict_list]
        >>> print(ut.repr2(metadata_str_list))
        >>> metadata_str_list_ = ibs.get_review_metadata(review_rowid_list, return_raw=True)
        >>> print(ut.repr2(metadata_str_list_))
        >>> assert metadata_str_list == metadata_str_list_
        >>> ibs.delete_review(review_rowid_list)
    """
    id_iter = ((review_rowid,) for review_rowid in review_rowid_list)
    metadata_str_list = []
    for metadata_dict in metadata_dict_list:
        metadata_str = ut.to_json(metadata_dict)
        metadata_str_list.append(metadata_str)
    val_list = ((metadata_str,) for metadata_str in metadata_str_list)
    ibs.staging.set(const.REVIEW_TABLE, ('review_metadata_json',), val_list, id_iter)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_review_funcs
        python -m wbia.control.manual_review_funcs --allexamples
        python -m wbia.control.manual_review_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
