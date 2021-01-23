# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'wbia.templates.template_generator')"
sh Tgen.sh --key edge weight --invert --Tcfg with_getters=True with_setters=False --modfname manual_edge_funcs

# TODO: Fix this name it is too special case
"""
import logging
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
logger = logging.getLogger('wbia')


VERBOSE_SQL = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


WEIGHT_ROWID = 'weight_rowid'
WEIGHT_UUID = 'weight_uuid'
WEIGHT_AID1 = 'annot_1_rowid'
WEIGHT_AID2 = 'annot_2_rowid'
WEIGHT_COUNT = 'weight_count'
WEIGHT_VALUE = 'weight_value'
WEIGHT_ALGO_IDENTITY = 'weight_algo_identity'
WEIGHT_ALGO_CONFIDENCE = 'weight_algo_confidence'
WEIGHT_TAGS = 'weight_tags'


def e_(u, v):
    return (u, v) if u < v else (v, u)


def hack_create_aidpair_index(ibs):
    # HACK IN INDEX
    sqlfmt = ut.codeblock(
        """
        CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({index_cols});
        """
    )
    with ibs.staging.connect() as conn:
        sqlcmd = sqlfmt.format(
            index_name='aidpair_to_rowid',
            table=ibs.const.WEIGHT_TABLE,
            index_cols=','.join([WEIGHT_AID1, WEIGHT_AID2]),
        )
        conn.execute(sqlcmd)
        sqlcmd = sqlfmt.format(
            index_name='aid1_to_rowids',
            table=ibs.const.WEIGHT_TABLE,
            index_cols=','.join([WEIGHT_AID1]),
        )
        conn.execute(sqlcmd)
        sqlcmd = sqlfmt.format(
            index_name='aid2_to_rowids',
            table=ibs.const.WEIGHT_TABLE,
            index_cols=','.join([WEIGHT_AID2]),
        )
        conn.execute(sqlcmd)


@register_ibs_method
@accessor_decors.ider
@register_api('/api/edge/weight/', methods=['GET'])
def _get_all_edge_weight_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    all_known_edge_weight_rowids = ibs.staging.get_all_rowids(const.WEIGHT_TABLE)
    return all_known_edge_weight_rowids


@register_ibs_method
def _set_edge_weight_uuids(ibs, weight_rowid_list, weight_uuid_list):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    id_iter = ((weight_rowid,) for weight_rowid in weight_rowid_list)
    val_iter = ((weight_uuid,) for weight_uuid in weight_uuid_list)
    ibs.staging.set(const.WEIGHT_TABLE, (WEIGHT_UUID,), val_iter, id_iter)


@register_ibs_method
def get_edge_weight_rowid_from_superkey(
    ibs, aid_1_list, aid_2_list, count_list, eager=False, nInput=None
):
    """Returns weight_rowid_list

    Args:
        superkey lists: weight_rowid_list, aid_list

    Returns:
        weight_rowid_list
    """
    colnames = (WEIGHT_ROWID,)
    params_iter = zip(aid_1_list, aid_2_list, count_list)
    where_colnames = [WEIGHT_AID1, WEIGHT_AID2, WEIGHT_COUNT]
    weight_rowid_list = list(
        ibs.staging.get_where_eq(
            const.WEIGHT_TABLE,
            colnames,
            params_iter,
            where_colnames,
            eager=eager,
            nInput=nInput,
        )
    )
    return weight_rowid_list


@register_ibs_method
@accessor_decors.adder
@register_api('/api/edge/weight/', methods=['POST'])
def add_edge_weight(
    ibs,
    aid_1_list,
    aid_2_list,
    value_list,
    identity_list,
    weight_uuid_list=None,
    algo_confidence_list=None,
    tags_list=None,
):
    r"""
    Adds a list of edge weights.

    Returns:
        list: weight_id_list - edge weight rowids

    RESTful:
        Method: POST
        URL:    /api/edge/weight/

    CommandLine:
        python -m wbia.control.manual_edge_funcs --test-add_edge_weight

    Doctest:
        >>> import wbia
        >>> from wbia.control.manual_edge_funcs import *
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.staging.get_table_as_pandas('weights')
        >>> # ensure it is empty
        >>> rowids = ibs.staging.get_all_rowids('weights')
        >>> ibs.staging.delete_rowids('weights', rowids)
        >>> ut.exec_funckw(ibs.add_edge_weight, globals())
        >>> # Add some dummy edge weights
        >>> aid_1_list = [1, 2, 3, 2]
        >>> aid_2_list = [2, 3, 4, 3]
        >>> value_list = [1, 0, 1, 2]
        >>> new_rowids = ibs.add_edge_weight(aid_1_list, aid_2_list,
        >>>                             value_list)
        >>> assert new_rowids == [1, 2, 3, 4]
        >>> table = ibs.staging.get_table_as_pandas('weights')
        >>> print(table)
        >>> # Then delete them
        >>> ibs.staging.delete_rowids('weights', new_rowids)
    """
    assert len(aid_1_list) == len(aid_2_list)
    assert len(aid_1_list) == len(value_list)
    diff_list = -np.array(aid_2_list)
    assert np.all(
        diff_list != 0
    ), 'Cannot add a edge weight state between an aid and itself'
    n_input = len(aid_1_list)

    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    aid_pair_list = [e_(u, v) for u, v in zip(aid_1_list, aid_2_list)]
    aid_1_list = [pair[0] for pair in aid_pair_list]
    aid_2_list = [pair[1] for pair in aid_pair_list]

    if True:
        # Get current edge weight counts from database
        unique_pairs = list(set(aid_pair_list))
        count_base = [
            0 if counts is None or len(counts) == 0 else max(max(counts), len(counts))
            for counts in ibs.get_edge_weight_counts_from_pairs(unique_pairs)
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
    #         for counts in ibs.get_edge_weight_counts_from_pairs(unique_pairs)
    #     ]
    #     grouped_base = [[b] * len(g) for b, g in zip(unique_base, groupxs)]
    #     grouped_offsets = [list(range(n)) for n in map(len, groupxs)]
    #     base = np.array(ut.ungroup(grouped_base, groupxs))
    #     offsets = np.array(ut.ungroup(grouped_offsets, groupxs))
    #     count_list = offsets + base + 1

    if weight_uuid_list is None:
        weight_uuid_list = [uuid.uuid4() for _ in range(n_input)]
    if tags_list is None:
        tag_str_list = [None] * n_input
    else:
        tag_str_list = [';'.join(map(str, tag_list)) for tag_list in tags_list]
    if algo_confidence_list is None:
        algo_confidence_list = [None] * n_input

    assert n_input == len(identity_list)
    assert n_input == len(tag_str_list)
    assert n_input == len(algo_confidence_list)
    assert n_input == len(weight_uuid_list)
    assert n_input == len(count_list)

    superkey_paramx = (
        0,
        1,
        2,
    )
    # TODO Allow for better ensure=False without using partial
    # Just autogenerate these functions
    colnames = [
        WEIGHT_AID1,
        WEIGHT_AID2,
        WEIGHT_COUNT,
        WEIGHT_UUID,
        WEIGHT_VALUE,
        WEIGHT_ALGO_IDENTITY,
        WEIGHT_ALGO_CONFIDENCE,
        WEIGHT_TAGS,
    ]
    params_iter = list(
        zip(
            aid_1_list,
            aid_2_list,
            count_list,
            weight_uuid_list,
            value_list,
            identity_list,
            algo_confidence_list,
            tag_str_list,
        )
    )
    weight_rowid_list = ibs.staging.add_cleanly(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        ibs.get_edge_weight_rowid_from_superkey,
        superkey_paramx,
    )
    return weight_rowid_list


@register_ibs_method
@accessor_decors.deleter
# @cache_invalidator(const.WEIGHT_TABLE)
@register_api('/api/edge/weight/', methods=['DELETE'])
def delete_edge_weight(ibs, weight_rowid_list):
    r"""
    deletes edge weights from the database

    RESTful:
        Method: DELETE
        URL:    /api/edge/weight/
    """
    if ut.VERBOSE:
        logger.info('[ibs] deleting %d edge weights' % len(weight_rowid_list))
    ibs.staging.delete_rowids(const.WEIGHT_TABLE, weight_rowid_list)


@register_ibs_method
def get_edge_weight_rowids_from_edges(
    ibs, edges, eager=True, nInput=None, directed=False
):
    colnames = (WEIGHT_ROWID,)
    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    # params_iter = edges
    if directed:
        params_iter = edges
    else:
        params_iter = [e_(u, v) for u, v in edges]
    where_colnames = [WEIGHT_AID1, WEIGHT_AID2]
    weight_rowids_list = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return weight_rowids_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_edge_weight_exists_from_edges(ibs, edges, eager=True, nInput=None):
    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    # params_iter = (e_(u, v) for u, v in edges)
    params_iter = edges
    where_colnames = [WEIGHT_AID1, WEIGHT_AID2]
    exists_list = ibs.staging.exists_where_eq(
        const.WEIGHT_TABLE,
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
@register_api(
    '/api/edge/weight/rowids/tuple/', methods=['GET'], __api_plural_check__=False
)
def get_edge_weight_rowids_from_aid_tuple(
    ibs, aid_1_list, aid_2_list, eager=True, nInput=None
):
    r"""
    Aid pairs are undirected

    Returns:
        list_ (list): weight_rowid_list - edge weight rowid list of lists

    RESTful:
        Method: GET
        URL:    /api/edge/weight/rowid/tuple/
    """
    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    edges = (e_(u, v) for u, v in zip(aid_1_list, aid_2_list))
    return get_edge_weight_rowids_from_edges(ibs, edges, eager=eager, nInput=nInput)
    # colnames = (WEIGHT_ROWID,)
    # where_colnames = [WEIGHT_AID1, WEIGHT_AID2]
    # weight_rowids_list = ibs.staging.get_where_eq(
    #     const.WEIGHT_TABLE, colnames, params_iter, where_colnames,
    #     eager=eager, nInput=nInput, unpack_scalars=False)
    # return weight_rowids_list


@register_ibs_method
def get_edge_weight_rowids_between(ibs, aids1, aids2=None, method=None):
    """
    Find staging rowids between sets of aids

    Doctest:
        >>> from wbia.control.manual_edge_funcs import *
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> aids1 = aids2 = [1, 2, 3, 4, 5, 6]
        >>> rowids_between = ibs.get_edge_weight_rowids_between
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
        rowids11 = set(ut.flatten(ibs.get_edge_weight_rowids_from_aid1(aids1)))
        rowids12 = set(ut.flatten(ibs.get_edge_weight_rowids_from_aid2(aids1)))
        if aids1 is aids2:
            rowids = list(reduce(set.intersection, [rowids11, rowids12]))
        else:
            rowids21 = set(ut.flatten(ibs.get_edge_weight_rowids_from_aid1(aids2)))
            rowids22 = set(ut.flatten(ibs.get_edge_weight_rowids_from_aid2(aids2)))
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
            rowids = ibs.get_edge_weight_rowids_from_edges(edges, directed=True)
            if rowids is None:
                rowids = []
            rowids = ut.filter_Nones(rowids)
            rowids = ut.flatten(rowids)
    else:
        raise ValueError('no method=%r' % (method,))
    return rowids


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/count/', methods=['GET'])
def get_edge_weight_count(ibs, weight_rowid_list):
    weight_count_list = ibs.staging.get(
        const.WEIGHT_TABLE, (WEIGHT_COUNT,), weight_rowid_list
    )
    return weight_count_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api(
    '/api/edge/weight/counts/tuple/', methods=['GET'], __api_plural_check__=False
)
def get_edge_weight_counts_from_tuple(
    ibs, aid_1_list, aid_2_list, eager=True, nInput=None
):
    r"""
    Returns:
        list_ (list): weight_counts_list - edge weight counts

    RESTful:
        Method: GET
        URL:    /api/edge/weight/counts/tuple/
    """
    aid_pairs = zip(aid_1_list, aid_2_list)
    weight_counts_list = ibs.get_edge_weight_counts_from_pairs(aid_pairs)
    return weight_counts_list


@register_ibs_method
def get_edge_weight_counts_from_pairs(ibs, aid_pairs, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): weight_counts_list - edge weight counts

    RESTful:
        Method: GET
        URL:    /api/edge/weight/counts/tuple/
    """
    colnames = (WEIGHT_COUNT,)
    params_iter = aid_pairs
    where_colnames = [WEIGHT_AID1, WEIGHT_AID2]
    weight_counts_list = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return weight_counts_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/value/', methods=['GET'])
def get_edge_weight_value(ibs, weight_rowid_list):
    weight_value_list = ibs.staging.get(
        const.WEIGHT_TABLE, (WEIGHT_VALUE,), weight_rowid_list
    )
    return weight_value_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/uuid/', methods=['GET'])
def get_edge_weight_uuid(ibs, weight_rowid_list):
    weight_uuid_list = ibs.staging.get(
        const.WEIGHT_TABLE, (WEIGHT_UUID,), weight_rowid_list
    )
    return weight_uuid_list


@register_ibs_method
@register_api(
    '/api/edge/weight/values/only/', methods=['GET'], __api_plural_check__=False
)
def get_edge_weight_values_from_only(ibs, aid_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): weight_tuple_values_list - edge weight values

    RESTful:
        Method: GET
        URL:    /api/edge/weight/values/only/
    """
    colnames = (
        WEIGHT_AID1,
        WEIGHT_AID2,
        WEIGHT_VALUE,
    )
    params_iter = [(aid,) for aid in aid_list]
    weight_tuple_values_list = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE, colnames, params_iter, (WEIGHT_AID1,), unpack_scalars=False
    )
    return weight_tuple_values_list


@register_ibs_method
@register_api(
    '/api/edge/weight/rowids/only/', methods=['GET'], __api_plural_check__=False
)
def get_edge_weight_rowids_from_only(ibs, aid_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): weight_rowids

    RESTful:
        Method: GET
        URL:    /api/edge/weight/rowids/only/
    """
    colnames = (WEIGHT_ROWID,)
    params_iter = [(aid,) for aid in aid_list]
    weight_rowids = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE, colnames, params_iter, (WEIGHT_AID1,), unpack_scalars=False
    )
    return weight_rowids


@register_ibs_method
def get_edge_weight_rowids_from_single(ibs, aid_list, eager=True, nInput=None):
    colnames = (WEIGHT_ROWID,)
    params_iter = [(aid, aid) for aid in aid_list]
    where_clause = '%s=? OR %s=?' % (WEIGHT_AID1, WEIGHT_AID2)
    weight_rowids = ibs.staging.get_where(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        where_clause=where_clause,
        unpack_scalars=False,
    )
    return weight_rowids


@register_ibs_method
def get_edge_weight_rowids_from_aid1(ibs, aid_list, eager=True, nInput=None):
    colnames = (WEIGHT_ROWID,)
    params_iter = [(aid,) for aid in aid_list]
    weight_rowids = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        (WEIGHT_AID1,),
        unpack_scalars=False,
    )
    return weight_rowids


@register_ibs_method
def get_edge_weight_rowids_from_aid2(ibs, aid_list, eager=True, nInput=None):
    colnames = (WEIGHT_ROWID,)
    params_iter = [(aid,) for aid in aid_list]
    weight_rowids = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        (WEIGHT_AID2,),
        unpack_scalars=False,
    )
    return weight_rowids


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/identity/', methods=['GET'])
def get_edge_weight_identity(ibs, weight_rowid_list):
    weight_identity_list = ibs.staging.get(
        const.WEIGHT_TABLE, (WEIGHT_ALGO_IDENTITY,), weight_rowid_list
    )
    return weight_identity_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/confidence/', methods=['GET'])
def get_edge_weight_confidence(ibs, weight_rowid_list):
    algo_confidence_list = ibs.staging.get(
        const.WEIGHT_TABLE, (WEIGHT_ALGO_CONFIDENCE,), weight_rowid_list
    )
    return algo_confidence_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api(
    '/api/edge/weight/identities/tuple/', methods=['GET'], __api_plural_check__=False
)
def get_edge_weight_identities_from_tuple(
    ibs, aid_1_list, aid_2_list, eager=True, nInput=None
):
    r"""
    Returns:
        list_ (list): weight_identities_list - edge weight identities

    RESTful:
        Method: GET
        URL:    /api/edge/weight/identities/tuple/
    """
    colnames = (WEIGHT_ALGO_IDENTITY,)
    params_iter = zip(aid_1_list, aid_2_list)
    where_colnames = [WEIGHT_AID1, WEIGHT_AID2]
    weight_identities_list = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    return weight_identities_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_edge_weight_aid_tuple(ibs, weight_rowid_list, eager=True, nInput=None):
    colnames = (
        WEIGHT_AID1,
        WEIGHT_AID2,
    )
    params_iter = zip(weight_rowid_list)
    where_colnames = [WEIGHT_ROWID]
    aid_tuple_list = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
    )
    return aid_tuple_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/tags/', methods=['GET'], __api_plural_check__=False)
def get_edge_weight_tags(ibs, weight_rowid_list):
    weight_tag_str_list = ibs.staging.get(
        const.WEIGHT_TABLE, (WEIGHT_TAGS,), weight_rowid_list
    )
    weight_tags_list = [
        None
        if weight_tag_str is None or len(weight_tag_str) == 0
        else weight_tag_str.split(';')
        for weight_tag_str in weight_tag_str_list
    ]
    return weight_tags_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/tags/tuple/', methods=['GET'], __api_plural_check__=False)
def get_edge_weight_tags_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): weight_tags_list - edge weight tags (list of strings)

    RESTful:
        Method: GET
        URL:    /api/edge/weight/tags/tuple/
    """
    colnames = (WEIGHT_TAGS,)
    params_iter = zip(aid_1_list, aid_2_list)
    where_colnames = [WEIGHT_AID1, WEIGHT_AID2]
    weight_tag_strs_list = ibs.staging.get_where_eq(
        const.WEIGHT_TABLE,
        colnames,
        params_iter,
        where_colnames,
        eager=eager,
        nInput=nInput,
        unpack_scalars=False,
    )
    weight_tags_list = [
        []
        if weight_tag_str_list is None
        else [
            None
            if weight_tag_str is None or len(weight_tag_str) == 0
            else weight_tag_str.split(';')
            for weight_tag_str in weight_tag_str_list
        ]
        for weight_tag_str_list in weight_tag_strs_list
    ]
    return weight_tags_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/edge/weight/metadata/', methods=['GET'])
def get_edge_weight_metadata(ibs, weight_rowid_list, return_raw=False):
    r"""
    Returns:
        list_ (list): edge weight metadata dictionary

    RESTful:
        Method: GET
        URL:    /api/edge/weight/metadata/
    """
    metadata_str_list = ibs.staging.get(
        const.WEIGHT_TABLE, ('weight_metadata_json',), weight_rowid_list
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
@register_api('/api/edge/weight/metadata/', methods=['PUT'])
def set_edge_weight_metadata(ibs, weight_rowid_list, metadata_dict_list):
    r"""
    Sets the edge weight's metadata using a metadata dictionary

    RESTful:
        Method: PUT
        URL:    /api/edge/weight/metadata/

    CommandLine:
        python -m wbia.control.manual_edge_funcs --test-set_edge_weight_metadata

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_edge_funcs import *  # NOQA
        >>> import wbia
        >>> import random
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> weight_rowid_list = ibs.add_edge_weight([1], [2], [0])
        >>> metadata_dict_list = [
        >>>     {'test': random.uniform(0.0, 1.0)},
        >>> ]
        >>> print(ut.repr2(metadata_dict_list))
        >>> ibs.set_edge_weight_metadata(weight_rowid_list, metadata_dict_list)
        >>> # verify results
        >>> metadata_dict_list_ = ibs.get_edge_weight_metadata(weight_rowid_list)
        >>> print(ut.repr2(metadata_dict_list_))
        >>> assert metadata_dict_list == metadata_dict_list_
        >>> metadata_str_list = [ut.to_json(metadata_dict) for metadata_dict in metadata_dict_list]
        >>> print(ut.repr2(metadata_str_list))
        >>> metadata_str_list_ = ibs.get_edge_weight_metadata(weight_rowid_list, return_raw=True)
        >>> print(ut.repr2(metadata_str_list_))
        >>> assert metadata_str_list == metadata_str_list_
        >>> ibs.delete_edge_weight(weight_rowid_list)
    """
    id_iter = ((weight_rowid,) for weight_rowid in weight_rowid_list)
    metadata_str_list = []
    for metadata_dict in metadata_dict_list:
        metadata_str = ut.to_json(metadata_dict)
        metadata_str_list.append(metadata_str)
    val_list = ((metadata_str,) for metadata_str in metadata_str_list)
    ibs.staging.set(const.WEIGHT_TABLE, ('weight_metadata_json',), val_list, id_iter)
