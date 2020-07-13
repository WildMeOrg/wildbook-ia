# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'wbia.templates.template_generator')"  # NOQA
sh Tgen.sh --key name --invert --Tcfg with_getters=True with_setters=False --modfname manual_name_funcs  # NOQA
sh Tgen.sh --key name --invert --Tcfg with_getters=True with_setters=True --modfname manual_name_funcs --funcname-filter=sex  # NOQA

"""
from __future__ import absolute_import, division, print_function

# TODO: Fix this name it is too special case
import uuid
import functools
import six  # NOQA

# from six.moves import range
from wbia import constants as const
from wbia.other import ibsfuncs
import numpy as np
import vtool as vt
from wbia.control import accessor_decors, controller_inject  # NOQA
import utool as ut
from wbia.control.controller_inject import make_ibs_register_decorator
import os

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


ANNOT_ROWID = 'annot_rowid'
ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
NAME_ROWID = 'name_rowid'

NAME_UUID = 'name_uuid'
NAME_TEXT = 'name_text'
NAME_ALIAS_TEXT = 'name_alias_text'
NAME_NOTE = 'name_note'
NAME_SEX = 'name_sex'
NAME_TEMP_FLAG = 'name_temp_flag'


def testdata_ibs(defaultdb='testdb1'):
    import wbia

    ibs = wbia.opendb(defaultdb=defaultdb)
    config2_ = None  # qreq_.qparams
    return ibs, config2_


@register_ibs_method
@accessor_decors.ider
def _get_all_known_name_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    # all_known_nids = ibs._get_all_known_lblannot_rowids(const.INDIVIDUAL_KEY)
    all_known_nids = ibs.db.get_all_rowids(const.NAME_TABLE)
    return all_known_nids


@register_ibs_method
@accessor_decors.ider
def _get_all_name_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    # all_known_nids = ibs._get_all_known_lblannot_rowids(const.INDIVIDUAL_KEY)
    all_known_nids = ibs.db.get_all_rowids(const.NAME_TABLE)
    return all_known_nids


@register_ibs_method
def _get_all_known_nids(ibs):
    r"""
    alias
    """
    return _get_all_known_name_rowids(ibs)


@register_ibs_method
@accessor_decors.adder
@register_api('/api/name/', methods=['POST'])
def add_names(ibs, name_text_list, name_uuid_list=None, name_note_list=None):
    r"""
    Adds a list of names.

    Returns:
        name_rowid_list (list): their nids

    RESTful:
        Method: POST
        URL:    /api/name/
    """
    if name_note_list is None:
        name_note_list = [''] * len(name_text_list)
    # Get random uuids
    if name_uuid_list is None:
        name_uuid_list = [uuid.uuid4() for _ in range(len(name_text_list))]
    get_rowid_from_superkey = functools.partial(
        ibs.get_name_rowids_from_text, ensure=False
    )
    superkey_paramx = (1,)
    colnames = [NAME_UUID, NAME_TEXT, NAME_NOTE]
    params_iter = list(zip(name_uuid_list, name_text_list, name_note_list))
    name_rowid_list = ibs.db.add_cleanly(
        const.NAME_TABLE, colnames, params_iter, get_rowid_from_superkey, superkey_paramx,
    )
    return name_rowid_list
    # OLD WAY
    # # nid_list_ = [namenid_dict[name] for name in name_list_]
    # name_text_list_ = ibs.sanitize_name_texts(name_text_list)
    # # All names are individuals and so may safely receive the INDIVIDUAL_KEY lblannot
    # lbltype_rowid = ibs.lbltype_ids[const.INDIVIDUAL_KEY]
    # lbltype_rowid_list = [lbltype_rowid] * len(name_text_list_)
    # nid_list = ibs.add_lblannots(lbltype_rowid_list, name_text_list_, note_list)
    # # nid_list = [const.UNKNOWN_NAME_ROWID if rowid is None else rowid for rowid in nid_list]
    # return nid_list


# def init_default_speciesvalue():
#    #const.KEY_DEFAULTS[const.SPECIES_KEY]
#    note_list = ['default value']
#    # Get random uuids
#    import uuid
#    lblannot_uuid_list = [uuid.UUID('00000000-0000-0000-0000-000000000001')]
#    value_list = [const.KEY_DEFAULTS[const.SPECIES_KEY]]
#    colnames = ['species_uuid', 'species_rowid', 'species_text', 'species_note']
#    params_iter = list(zip(lblannot_uuid_list, lbltype_rowid_list, value_list, note_list))
#    get_rowid_from_superkey = ibs.get_species_rowid_from_species_text
#    superkey_paramx = (1, 2)
#    species_rowid_list = ibs.db.add_cleanly(const.SPECIES_TABLE, colnames, params_iter,
#                                            get_rowid_from_superkey, superkey_paramx)


@register_ibs_method
@register_api('/api/name/sanitize/', methods=['PUT'])
def sanitize_name_texts(ibs, name_text_list):
    r"""

    RESTful:
        Method: PUT
        URL:    /api/name/sanitize
    """
    ibsfuncs.assert_valid_names(name_text_list)
    name_text_list_ = [
        None if name_text == const.UNKNOWN else name_text for name_text in name_text_list
    ]
    return name_text_list_


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/name/', methods=['DELETE'])
def delete_names(ibs, name_rowid_list, safe=True, strict=False, verbose=ut.VERBOSE):
    r"""
    deletes names from the database

    CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
    at least ensure that no annot is associated with any of these nids

    RESTful:
        Method: DELETE
        URL:    /api/name/

    # Ignore:
    #     >>> # UNPORTED_DOCTEST
    #     >>> gpath_list = grabdata.get_test_gpaths(ndata=None)[0:4]
    #     >>> gid_list = ibs.add_images(gpath_list)
    #     >>> bbox_list = [(0, 0, 100, 100)]*len(gid_list)
    #     >>> name_list = ['a', 'b', 'a', 'd']
    #     >>> aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list, name_list=name_list)
    #     >>> assert len(aid_list) != 0, "No annotations added"
    #     >>> nid_list = ibs.get_valid_nids()
    #     >>> assert len(nid_list) != 0, "No names added"
    #     >>> nid = nid_list[0]
    #     >>> assert nid is not None, "nid is None"
    #     >>> ibs.delete_names(nid)
    #     >>> all_nids = ibs.get_valid_nids()
    #     >>> assert nid not in all_nids, "NID not deleted"

    """
    if verbose:
        print('[ibs] deleting %d names' % len(name_rowid_list))
    if safe:
        aids_list = ibs.get_name_aids(name_rowid_list)
        aid_list = ut.flatten(aids_list)
        if strict:
            assert (
                len(aid_list) == 0
            ), 'should not be any annots belonging to a deleted name'
        else:
            if verbose:
                print(
                    '[ibs] deleting %d annots that belonged to those names'
                    % len(aid_list)
                )
            if len(aid_list) > 0:
                ibs.delete_annot_nids(aid_list)
    ibs.db.delete_rowids(const.NAME_TABLE, name_rowid_list)
    # return len(name_rowid_list)
    # ibs.delete_lblannots(nid_list)


@register_ibs_method
@accessor_decors.ider
# @register_api('/api/name/nids/empty/', methods=['GET'])
def get_empty_nids(ibs, _nid_list=None):
    r"""
    get name rowids that do not have any annotations (not including UNKONWN)

    Returns:
        list: nid_list - all names without any animals (does not include unknown names)
        an nid is not invalid if it has a valid alias

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_empty_nids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> new_nid_list = ibs.make_next_nids(num=2)
        >>> empty_nids = ibs.get_empty_nids()
        >>> assert len(empty_nids) == 2, 'get_empty_nids fails1'
        >>> assert new_nid_list == empty_nids, 'get_empty_nids fails2'
        >>> ibs.delete_empty_nids()
        >>> empty_nids2 = ibs.get_empty_nids()
        >>> assert len(empty_nids2) == 0, 'get_empty_nids fails3'
        >>> result = str(empty_nids2)
        >>> print(result)
        []
    """
    recursive = _nid_list is not None
    recstr = '\t\t' if recursive else '\t'
    if _nid_list is None:
        _nid_list = ibs._get_all_known_name_rowids()
    if len(_nid_list) == 0:
        return []
    args = (len(_nid_list),)
    if recursive:
        print('\tCHECKING %d NIDS FOR EMPTY (RECURSIVE)' % args)
    else:
        print('CHECKING %d NIDS FOR EMPTY' % args)
    nRois_list = ibs.get_name_num_annotations(_nid_list)
    # Filter names with rois
    isempty_list = (nRois <= 0 for nRois in nRois_list)
    empty_nid_list = list(ut.iter_compress(_nid_list, isempty_list))
    # Filter names with aliases (TODO: use transitivity to determine validity)
    alias_text_list = ibs.get_name_alias_texts(empty_nid_list)
    hasalias_list = [alias_text is not None for alias_text in alias_text_list]
    # Find nids with aliases and without alias
    alias_nid_list = list(ut.ifilter_items(empty_nid_list, hasalias_list))
    no_alias_nid_list = list(ut.ifilterfalse_items(empty_nid_list, hasalias_list))
    # Find name texts and then nids of the original nids that have valid aliases
    alias_text_list = ibs.get_name_alias_texts(alias_nid_list)
    alias_nid_list = ibs.get_name_rowids_from_text(alias_text_list)
    # Find the empty aliases, recursively
    print('%sFound %d empty NIDs' % (recstr, len(empty_nid_list),))
    print('%sFound %d empty NIDs without an alias' % (recstr, len(no_alias_nid_list),))
    message = ' checking these recursively' if len(alias_nid_list) > 0 else ''
    print(
        '%sFound %d empty NIDs with an alias...%s'
        % (recstr, len(alias_nid_list), message,)
    )
    empty_alias_nid_list = ibs.get_empty_nids(_nid_list=alias_nid_list)
    # Compile the full list of nids without any associated annotations
    empty_nid_list = empty_nid_list + no_alias_nid_list + empty_alias_nid_list
    if not recursive:
        print(
            '\tFound %d empty NIDs with an alias that is recursively empty'
            % (len(empty_alias_nid_list),)
        )
    empty_nid_list = list(set(empty_nid_list))
    # Sanity check
    nRois_list = ibs.get_name_num_annotations(empty_nid_list)
    isempty_list = [nRois <= 0 for nRois in nRois_list]
    assert isempty_list.count(True) == len(empty_nid_list)
    return empty_nid_list


@register_ibs_method
# @register_api('/api/name/nids/empty/', methods=['DELETE'])
def delete_empty_nids(ibs):
    r"""
    Removes names that have no Rois from the database
    """
    print('[ibs] deleting empty nids')
    invalid_nids = ibs.get_empty_nids()
    print('[ibs] ... %d empty nids' % (len(invalid_nids),))
    ibs.delete_names(invalid_nids)


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/name/annot/rowid/', methods=['GET'])
@profile
def get_name_aids(ibs, nid_list, enable_unknown_fix=True, is_staged=False):
    r"""
    # TODO: Rename to get_anot_rowids_from_name_rowid

    Returns:
         list: aids_list a list of list of aids in each name

    RESTful:
        Method: GET
        URL:    /api/name/annot/rowid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> # Map annotations to name ids
        >>> aid_list = ibs.get_valid_aids()
        >>> nid_list = ibs.get_annot_name_rowids(aid_list)
        >>> # Get annotation ids for each name
        >>> aids_list = ibs.get_name_aids(nid_list)
        >>> # Run Assertion Test
        >>> groupid2_items = ut.group_items(aids_list, nid_list)
        >>> grouped_items = list(six.itervalues(groupid2_items))
        >>> passed_iter = map(ut.allsame, grouped_items)
        >>> passed_list = list(passed_iter)
        >>> assert all(passed_list), 'problem in get_name_aids'
        >>> # Print gropued items
        >>> print(ut.repr2(groupid2_items, newlines=False))

    Ignore;
        from wbia.control.manual_name_funcs import *  # NOQA
        import wbia
        #ibs = wbia.opendb('testdb1')
        #ibs = wbia.opendb('PZ_MTEST')
        ibs = wbia.opendb('PZ_Master0')
        #ibs = wbia.opendb('GZ_ALL')

        nid_list = ibs.get_valid_nids()
        nid_list_ = [const.UNKNOWN_NAME_ROWID if nid <= 0 else nid for nid in nid_list]

        with ut.Timer('sql'):
            #aids_list1 = ibs.get_name_aids(nid_list, enable_unknown_fix=False)
            aids_list1 = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,), nid_list_, id_colname=NAME_ROWID, unpack_scalars=False)

        with ut.Timer('hackquery + group'):
            opstr = '''
            SELECT annot_rowid, name_rowid
            FROM annotations
            WHERE name_rowid IN
                (%s)
                ORDER BY name_rowid ASC, annot_rowid ASC
            ''' % (', '.join(map(str, nid_list)))
            pair_list = ibs.db.connection.execute(opstr).fetchall()
            aids = np.array(ut.get_list_column(pair_list, 0))
            nids = np.array(ut.get_list_column(pair_list, 1))
            unique_nids, groupx = vt.group_indices(nids)
            grouped_aids_ = vt.apply_grouping(aids, groupx)
            aids_list5 = [sorted(arr.tolist()) for arr in grouped_aids_]

        for aids1, aids5 in zip(aids_list1, aids_list5):
            if (aids1) != (aids5):
                print(aids1)
                print(aids5)
                print('-----')

        ut.assert_lists_eq(list(map(tuple, aids_list5)), list(map(tuple, aids_list1)))

        with ut.Timer('numpy'):
            # alt method
            valid_aids = np.array(ibs.get_valid_aids())
            valid_nids = np.array(ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False))
            aids_list2 = [valid_aids.take(np.flatnonzero(valid_nids == nid)).tolist() for nid in nid_list_]

        with ut.Timer('numpy2'):
            # alt method
            valid_aids = np.array(ibs.get_valid_aids())
            valid_nids = np.array(ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False))
            aids_list3 = [valid_aids.take(np.flatnonzero(np.equal(valid_nids, nid))).tolist() for nid in nid_list_]

        with ut.Timer('numpy3'):
            # alt method
            valid_aids = np.array(ibs.get_valid_aids())
            valid_nids = np.array(ibs.db.get_all_col_rows(const.ANNOTATION_TABLE, NAME_ROWID))
            aids_list4 = [valid_aids.take(np.flatnonzero(np.equal(valid_nids, nid))).tolist() for nid in nid_list_]
        assert aids_list2 == aids_list3
        assert aids_list3 == aids_list4
        assert aids_list1 == aids_list2

        valid_aids = ibs.get_valid_aids()
        %timeit ibs.db.get_all_col_rows('annotations', 'rowid')
        %timeit ibs.db.get_all_col_rows('annotations', 'name_rowid')
        %timeit ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False)
        %timeit ibs.get_valid_aids()
        %timeit ibs.get_annot_name_rowids(ibs.get_valid_aids(), distinguish_unknowns=False)
        valid_nids1 = ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False)
        valid_nids2 = ibs.db.get_all_col_rows('annotations', 'name_rowid')
        assert valid_nids1 == valid_nids2

    ibs.db.fname
    ibs.db.fpath

    import sqlite3

    con = sqlite3.connect(ibs.db.fpath)

    opstr = '''
    SELECT annot_rowid, name_rowid
    FROM annotations
    WHERE name_rowid IN
        (SELECT name_rowid FROM name)
        ORDER BY name_rowid ASC, annot_rowid ASC
    '''

    annot_rowid_list = con.execute(opstr).fetchall()
    aid_list = ut.get_list_column(annot_rowid_list, 0)
    nid_list = ut.get_list_column(annot_rowid_list, 1)


    # HACKY HACKY HACK

    with ut.Timer('hackquery + group'):
        #nid_list = ibs.get_valid_nids()[10:15]
        nid_list = ibs.get_valid_nids()
        opstr = '''
        SELECT annot_rowid, name_rowid
        FROM annotations
        WHERE name_rowid IN
            (%s)
            ORDER BY name_rowid ASC, annot_rowid ASC
        ''' % (', '.join(map(str, nid_list)))
        pair_list = ibs.db.connection.execute(opstr).fetchall()
        aids = np.array(ut.get_list_column(pair_list, 0))
        nids = np.array(ut.get_list_column(pair_list, 1))
        unique_nids, groupx = vt.group_indices(nids)
        grouped_aids_ = vt.apply_grouping(aids, groupx)
        grouped_aids = [arr.tolist() for arr in grouped_aids_]

    SELECT
       name_rowid, COUNT(annot_rowid) AS number, GROUP_CONCAT(annot_rowid) AS aid_list
    FROM annotations
    WHERE name_rowid in (SELECT name_rowid FROM name)
     GROUP BY name_rowid
    ORDER BY name_rowid ASC


    import vtool as vt
    vt
    vt.aid_list[0]


    annot_rowid_list = con.execute(opstr).fetchall()
    opstr = '''
        SELECT annot_rowid
        FROM annotations
        WHERE name_rowid=?
        '''

    cur = ibs.db.connection.cursor()

    cur = con.execute('BEGIN IMMEDIATE TRANSACTION')
    cur = ibs.db.connection
    res = [cur.execute(opstr, (nid,)).fetchall() for nid in nid_list_]
    cur.execute('COMMIT TRANSACTION')

    res = [ibs.db.cur.execute(opstr, (nid,)).fetchall() for nid in nid_list_]

    """
    # FIXME: THIS FUNCTION IS VERY SLOW
    # ADD A LOCAL CACHE TO FIX THIS SPEED
    # ALSO FIX GET_IMAGE_AIDS
    # really a getter for the annotation table not the name table
    # return [[] for nid in nid_list]
    # TODO: should a query of the UNKNOWN_NAME_ROWID return anything?
    # TODO: don't even run negative aids as queries
    nid_list_ = [const.UNKNOWN_NAME_ROWID if nid <= 0 else nid for nid in nid_list]

    NEW_INDEX_HACK = True
    USE_GROUPING_HACK = False
    if NEW_INDEX_HACK:
        # FIXME: This index should when the database is defined.
        # Ensure that an index exists on the image column of the annotation table
        # print(len(nid_list_))
        ibs.db.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS nid_to_aids ON annotations (name_rowid);
            """
        ).fetchall()
        aids_list = ibs.db.get(
            const.ANNOTATION_TABLE,
            (ANNOT_ROWID,),
            nid_list_,
            id_colname=NAME_ROWID,
            unpack_scalars=False,
        )
        # %timeit ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,), nid_list_, id_colname=NAME_ROWID, unpack_scalars=False)
        # The index maxes the following query very efficient
    elif USE_GROUPING_HACK:
        # This code doesn't work because it doesn't respect empty names
        input_list, inverse_unique = np.unique(nid_list_, return_inverse=True)
        input_str = ', '.join(list(map(str, input_list)))
        opstr = """
        SELECT annot_rowid, name_rowid
        FROM {ANNOTATION_TABLE}
        WHERE name_rowid IN
            ({input_str})
            ORDER BY name_rowid ASC, annot_rowid ASC
        """.format(
            input_str=input_str, ANNOTATION_TABLE=const.ANNOTATION_TABLE
        )
        pair_list = ibs.db.connection.execute(opstr).fetchall()
        aidscol = np.array(ut.get_list_column(pair_list, 0))
        nidscol = np.array(ut.get_list_column(pair_list, 1))
        unique_nids, groupx = vt.group_indices(nidscol)
        grouped_aids_ = vt.apply_grouping(aidscol, groupx)
        # aids_list = [sorted(arr.tolist()) for arr in grouped_aids_]
        structured_aids_list = [arr.tolist() for arr in grouped_aids_]
        aids_list = np.array(structured_aids_list)[inverse_unique].tolist()
    else:
        USE_NUMPY_IMPL = True
        # USE_NUMPY_IMPL = False
        # Use qt if getting one at a time otherwise perform bulk operation
        USE_NUMPY_IMPL = len(nid_list_) > 1
        # USE_NUMPY_IMPL = len(nid_list_) > 10
        if USE_NUMPY_IMPL:
            # This seems to be 30x faster for bigger inputs
            valid_aids = np.array(ibs._get_all_aids())
            valid_nids = np.array(
                ibs.db.get_all_col_rows(const.ANNOTATION_TABLE, NAME_ROWID)
            )
            # np.array(ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False))

            # MEMORY HOG LIKE A SON OF A BITCH
            # aids_list = [
            #     valid_aids.take(np.flatnonzero(
            #         np.equal(valid_nids, nid))).tolist()
            #     for nid in nid_list_
            # ]

            temp = np.zeros((len(valid_nids),), dtype=np.bool)
            aids_dict = {}
            nid_list_unique = np.unique(nid_list_)
            for nid in nid_list_unique:
                bool_list = np.equal(valid_nids, nid, out=temp)
                flattened = np.flatnonzero(bool_list)
                aid_list = [] if nid < 0 else valid_aids.take(flattened)
                aid_list = aid_list.tolist()
                aids_dict[nid] = aid_list

            aids_list = ut.dict_take(aids_dict, nid_list_)
        else:
            # SQL IMPL
            aids_list = ibs.db.get(
                const.ANNOTATION_TABLE,
                (ANNOT_ROWID,),
                nid_list_,
                id_colname=NAME_ROWID,
                unpack_scalars=False,
            )
    if enable_unknown_fix:
        # enable_unknown_fix == distinguish_unknowns
        # negative name rowids correspond to unknown annoations wherex annot_rowid = -name_rowid
        # aids_list = [None if nid is None else ([-nid] if nid < 0 else aids)
        #             for nid, aids in zip(nid_list, aids_list)]
        # Not sure if this should fail or return empty list on None nid
        aids_list = [
            [] if nid is None else ([-nid] if nid < 0 else aids)
            for nid, aids in zip(nid_list, aids_list)
        ]
        # aids_list = [[-nid] if nid < 0 else aids
        #             for nid, aids in zip(nid_list, aids_list)]
    aids_list = [
        ibs.filter_annotation_set(aid_list_, is_staged=is_staged)
        for aid_list_ in aids_list
    ]
    return aids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/name/annot/uuid/', methods=['GET'])
def get_name_annot_uuids(ibs, nid_list, **kwargs):
    aids_list = ibs.get_name_aids(nid_list, **kwargs)
    annot_uuids_list = [ibs.get_annot_uuids(aid_list) for aid_list in aids_list]
    return annot_uuids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/name/annot/rowid/exemplar/', methods=['GET'])
def get_name_exemplar_aids(ibs, nid_list):
    r"""
    Returns:
        list_ (list):  a list of list of cids in each name


    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_exemplar_aids

    RESTful:
        Method: GET
        URL:    /api/name/annot/rowid/examplar/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> nid_list = ibs.get_annot_name_rowids(aid_list)
        >>> exemplar_aids_list = ibs.get_name_exemplar_aids(nid_list)
        >>> result = exemplar_aids_list
        >>> print(result)
        [[], [2, 3], [2, 3], [], [5, 6], [5, 6], [7], [8], [], [10], [], [12], [13]]
    """
    # Get all annot ids for each name
    aids_list = ibs.get_name_aids(nid_list, enable_unknown_fix=True)
    # Flag any annots that are not exemplar and remove them
    flags_list = ibsfuncs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
    exemplar_aids_list = [
        ut.compress(aids, flags) for aids, flags in zip(aids_list, flags_list)
    ]
    return exemplar_aids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/name/annot/uuid/exemplar/', methods=['GET'])
def get_name_exemplar_name_uuids(ibs, nid_list, **kwargs):
    aids_list = ibs.get_name_exemplar_aids(nid_list, **kwargs)
    annot_uuids_list = [ibs.get_annot_uuids(aid_list) for aid_list in aids_list]
    return annot_uuids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/name/image/rowid/', methods=['GET'])
def get_name_gids(ibs, nid_list):
    r"""
    Returns:
        list_ (list): the image ids associated with name ids

    RESTful:
        Method: GET
        URL:    /api/name/image/rowid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_name_rowids()
        >>> gids_list = ibs.get_name_gids(nid_list)
        >>> result = gids_list
        >>> print(result)
        [[2, 3], [5, 6], [7], [8], [10], [12], [13]]
    """
    # TODO: Optimize
    aids_list = ibs.get_name_aids(nid_list, enable_unknown_fix=True)
    gids_list = ibsfuncs.unflat_map(ibs.get_annot_gids, aids_list)
    return gids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/name/image/uuid/', methods=['GET'])
def get_name_image_uuids(ibs, nid_list):
    r"""
    DEPRICATE

    Returns:
        list_ (list): the image ids associated with name ids

    RESTful:
        Method: GET
        URL:    /api/name/image/uuid/
    """
    # TODO: Optimize
    gids_list = ibs.get_name_gids(nid_list)
    image_uuids_list = [ibs.get_image_uuids(gid_list) for gid_list in gids_list]
    return image_uuids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/uuid/', methods=['GET'])
def get_name_uuids(ibs, nid_list):
    r"""
    Returns:
        list_ (list): uuids_list - name uuids

    RESTful:
        Method: GET
        URL:    /api/name/uuid/
    """
    uuids_list = ibs.db.get(const.NAME_TABLE, (NAME_UUID,), nid_list)
    # notes_list = ibs.get_lblannot_notes(nid_list)
    return uuids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/note/', methods=['GET'])
def get_name_notes(ibs, name_rowid_list):
    r"""
    Returns:
        list_ (list): notes_list - name notes

    RESTful:
        Method: GET
        URL:    /api/name/note/
    """
    notes_list = ibs.db.get(const.NAME_TABLE, (NAME_NOTE,), name_rowid_list)
    # notes_list = ibs.get_lblannot_notes(nid_list)
    return notes_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/metadata/', methods=['GET'])
def get_name_metadata(ibs, name_rowid_list, return_raw=False):
    r"""
    Returns:
        list_ (list): name metadata dictionary

    RESTful:
        Method: GET
        URL:    /api/name/metadata/
    """
    metadata_str_list = ibs.db.get(
        const.NAME_TABLE, ('name_metadata_json',), name_rowid_list
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
@accessor_decors.getter_1to1
@register_api('/api/name/num/annot/', methods=['GET'])
def get_name_num_annotations(ibs, nid_list):
    r"""
    Returns:
        list_ (list):  the number of annotations for each name

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_num_annotations

    RESTful:
        Method: GET
        URL:    /api/name/num/annot/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_name_rowids()
        >>> result = get_name_num_annotations(ibs, nid_list)
        >>> print(result)
        [2, 2, 1, 1, 1, 1, 1]
    """
    # TODO: Optimize
    return list(map(len, ibs.get_name_aids(nid_list, enable_unknown_fix=True)))


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/num/annot/exemplar/', methods=['GET'])
def get_name_num_exemplar_annotations(ibs, nid_list):
    r"""
    Returns:
        list_ (list):  the number of annotations, which are exemplars for each name

    RESTful:
        Method: GET
        URL:    /api/name/num/annot/exemplar/
    """
    return list(map(len, ibs.get_name_exemplar_aids(nid_list)))


@register_ibs_method
@register_api('/api/name/temp/', methods=['GET'])
def get_name_temp_flag(ibs, name_rowid_list, eager=True, nInput=None):
    r"""
    name_temp_flag_list <- name.name_temp_flag[name_rowid_list]

    gets data from the "native" column "name_temp_flag" in the "name" table

    Args:
        name_rowid_list (list):

    Returns:
        list: name_temp_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = name_temp_flag
        tbl = name

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_temp_flag

    RESTful:
        Method: GET
        URL:    /api/name/temp/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> name_rowid_list = ibs._get_all_name_rowids()
        >>> eager = True
        >>> name_temp_flag_list = ibs.get_name_temp_flag(name_rowid_list, eager=eager)
        >>> assert len(name_rowid_list) == len(name_temp_flag_list)
    """
    id_iter = name_rowid_list
    colnames = (NAME_TEMP_FLAG,)
    name_temp_flag_list = ibs.db.get(
        const.NAME_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return name_temp_flag_list


@register_ibs_method
@register_api('/api/name/temp/', methods=['PUT'])
def set_name_temp_flag(
    ibs, name_rowid_list, name_temp_flag_list, duplicate_behavior='error'
):
    r"""
    name_temp_flag_list -> name.name_temp_flag[name_rowid_list]

    Args:
        name_rowid_list
        name_temp_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = name
        col = name_temp_flag

    RESTful:
        Method: PUT
        URL:    /api/name/temp/
    """
    id_iter = name_rowid_list
    colnames = (NAME_TEMP_FLAG,)
    ibs.db.set(
        const.NAME_TABLE,
        colnames,
        name_temp_flag_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/alias/text/', methods=['GET'], __api_plural_check__=False)
def get_name_alias_texts(ibs, name_rowid_list):
    r"""
    Returns:
        list_ (list): name_alias_text_list

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_texts

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_alias_texts

    RESTful:
        Method: GET
        URL:    /api/name/alias/text/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> name_rowid_list = ibs.get_valid_nids()
        >>> # execute function
        >>> name_alias_text_list = get_name_alias_texts(ibs, name_rowid_list)
        >>> # verify results
        >>> result = str(name_alias_text_list)
        >>> print(result)
        [None, None, None, None, None, None, None]
    """
    name_alias_text_list = ibs.db.get(
        const.NAME_TABLE, (NAME_ALIAS_TEXT,), name_rowid_list
    )
    return name_alias_text_list


@register_ibs_method
@accessor_decors.cache_invalidator(
    const.ANNOTATION_TABLE, [ANNOT_SEMANTIC_UUID], rowidx=None
)
@accessor_decors.setter
@register_api('/api/name/alias/text/', methods=['PUT'], __api_plural_check__=False)
def set_name_alias_texts(ibs, name_rowid_list, name_alias_text_list):
    r"""
    Returns:
        list_ (list): name_alias_text_list

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_texts

    RESTful:
        Method: PUT
        URL:    /api/name/alias/text/
    """
    # ibsfuncs.assert_valid_names(name_alias_text_list)
    val_list = ((value,) for value in name_alias_text_list)
    ibs.db.set(const.NAME_TABLE, (NAME_ALIAS_TEXT,), val_list, name_rowid_list)
    # TODO: ibs.update_annot_semantic_uuids(aid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/text/', methods=['GET'])
def get_name_texts(ibs, name_rowid_list, apply_fix=True):
    r"""
    Returns:
        list_ (list): text names

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_texts

    RESTful:
        Method: GET
        URL:    /api/name/text/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> name_rowid_list = ibs._get_all_known_name_rowids()
        >>> name_text_list = get_name_texts(ibs, name_rowid_list)
        >>> result = ut.repr2(name_text_list)
        >>> print(result)
        ['easy', 'hard', 'jeff', 'lena', 'occl', 'polar', 'zebra']
    """
    # FIXME: Use standalone name table
    # TODO:
    # Change the temporary negative indexes back to the unknown NID for the
    # SQL query. Then augment the lblannot list to distinguish unknown lblannots
    # name_text_list = ibs.get_lblannot_values(nid_list, const.INDIVIDUAL_KEY)
    # name_text_list = ibs.get_lblannot_values(nid_list, const.INDIVIDUAL_KEY)
    name_text_list = ibs.db.get(const.NAME_TABLE, (NAME_TEXT,), name_rowid_list)
    if apply_fix:
        name_text_list = [
            const.UNKNOWN
            if rowid == const.UNKNOWN_NAME_ROWID or name_text is None
            else name_text
            for name_text, rowid in zip(name_text_list, name_rowid_list)
        ]
    return name_text_list


@register_ibs_method
def get_num_names(ibs, **kwargs):
    r"""
    Number of valid names

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_num_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> result = get_num_names(ibs)
        >>> print(result)
        7
    """
    nid_list = ibs.get_valid_nids(**kwargs)
    return len(nid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/rowid/text/', methods=['GET'])
def get_name_rowids_from_text(ibs, name_text_list, ensure=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        name_text_list (list):
        ensure (bool): adds as new names if non-existant (default = True)

    Returns:
        name_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_rowids_from_text:0
        python -m wbia.control.manual_name_funcs --test-get_name_rowids_from_text:1

    TODO:
        should ensure be defaulted to False?

    RESTful:
        Method: GET
        URL:    /api/name/rowid/text/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> import utool as ut
        >>> ibs = wbia.opendb('testdb1')
        >>> name_text_list = [u'Fred', u'Sue', '____', u'zebra_grevys', 'TYPO', '____']
        >>> ensure = False
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> print(ut.repr2(list(zip(name_text_list, name_rowid_list))))
        >>> ensure = True
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> print(ut.repr2(list(zip(name_text_list, name_rowid_list))))
        >>> ibs.print_name_table()
        >>> result = str(name_rowid_list) + '\n'
        >>> typo_rowids = ibs.get_name_rowids_from_text(['TYPO', 'Fred', 'Sue', 'zebra_grevys'])
        >>> ibs.delete_names(typo_rowids)
        >>> result += str(ibs._get_all_known_name_rowids())
        >>> print('----')
        >>> ibs.print_name_table()
        >>> print(result)
        [8, 9, 0, 10, 11, 0]
        [1, 2, 3, 4, 5, 6, 7]
    """
    if ensure:
        name_rowid_list = ibs.add_names(name_text_list)
    else:
        name_rowid_list = ibs.get_name_rowids_from_text_(name_text_list)
    return name_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_name_rowids_from_text_(ibs, name_text_list, ensure=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        name_text_list (list):

    Returns:
        name_rowid_list (list):

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-get_name_rowids_from_text_

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> import utool as ut  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> name_text_list = [u'Fred', 'easy', u'Sue', '____', u'zebra_grevys', 'TYPO', 'jeff']
        >>> name_rowid_list = ibs.get_name_rowids_from_text_(name_text_list)
        >>> ibs.print_name_table()
        >>> result = str(name_rowid_list)
        >>> print(result)
        [None, 1, None, 0, None, None, 3]
    """
    name_text_list_ = ibs.sanitize_name_texts(name_text_list)
    name_rowid_list = ibs.db.get(
        const.NAME_TABLE, (NAME_ROWID,), name_text_list_, id_colname=NAME_TEXT
    )
    name_rowid_list = [
        const.UNKNOWN_NAME_ROWID if text is None or text == const.UNKNOWN else rowid
        for rowid, text in zip(name_rowid_list, name_text_list_)
    ]
    return name_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/rowid/uuid/', methods=['GET'])
def get_name_rowids_from_uuid(ibs, uuid_list, nid_hack=False, ensure=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        name_text_list (list):

    Returns:
        name_rowid_list (list):
    """
    name_rowid_list = ibs.db.get(
        const.NAME_TABLE, (NAME_ROWID,), uuid_list, id_colname=NAME_UUID
    )
    if nid_hack:
        name_rowid_list = [
            name_uuid if name_rowid is None else name_rowid
            for name_uuid, name_rowid in zip(uuid_list, name_rowid_list)
        ]
    return name_rowid_list


@register_ibs_method
@register_api('/api/name/dict/', methods=['GET'])
def get_name_nids_with_gids(ibs, nid_list=None):
    if nid_list is None:
        nid_list = sorted(ibs.get_valid_nids())
    name_list = ibs.get_name_texts(nid_list)
    gids_list = ibs.get_name_gids(nid_list)

    zipped = zip(nid_list, name_list, gids_list)
    combined_dict = {name: (nid, gid_list) for nid, name, gid_list in zipped}
    return combined_dict


@register_ibs_method
@accessor_decors.ider
@register_api('/api/name/', methods=['GET'])
def get_valid_nids(ibs, imgsetid=None, filter_empty=False, min_pername=None):
    r"""
    Returns:
        list_ (list): all valid names with at least one animal
        (does not include unknown names)

    RESTful:
        Method: GET
        URL:    /api/name/
    """
    if imgsetid is None:
        _nid_list = ibs._get_all_known_name_rowids()
    else:
        _nid_list = ibs.get_imageset_nids(imgsetid)
    # HACK FOR UNKNOWN. Makes things crash
    # _nid_list += [0]
    nid_list = _nid_list

    if filter_empty:
        min_pername = 1 if min_pername is None else max(min_pername, 1)

    if min_pername is not None:
        nAnnot_list = ibs.get_name_num_annotations(nid_list)
        flag_list = np.array(nAnnot_list) >= min_pername
        nid_list = ut.compress(nid_list, flag_list)
    return nid_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/name/note/', methods=['PUT'])
def set_name_notes(ibs, name_rowid_list, notes_list):
    r"""
    Sets a note for each name (multiple annotations)

    RESTful:
        Method: PUT
        URL:    /api/name/note/
    """
    # ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    # ibs.set_lblannot_notes(nid_list, notes_list)
    val_list = ((value,) for value in notes_list)
    ibs.db.set(const.NAME_TABLE, (NAME_NOTE,), val_list, name_rowid_list)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/name/metadata/', methods=['PUT'])
def set_name_metadata(ibs, name_rowid_list, metadata_dict_list):
    r"""
    Sets the name's metadata using a metadata dictionary

    RESTful:
        Method: PUT
        URL:    /api/name/metadata/

    """
    id_iter = ((gid,) for gid in name_rowid_list)
    metadata_str_list = []
    for metadata_dict in metadata_dict_list:
        metadata_str = ut.to_json(metadata_dict)
        metadata_str_list.append(metadata_str)
    val_list = ((metadata_str,) for metadata_str in metadata_str_list)
    ibs.db.set(const.NAME_TABLE, ('name_metadata_json',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/name/text/', methods=['PUT'])
def set_name_texts(
    ibs,
    name_rowid_list,
    name_text_list,
    verbose=False,
    notify_wildbook=False,
    assert_wildbook=False,
    update_json_log=True,
):
    r"""
    Changes the name text. Does not affect the animals of this name.
    Effectively just changes the TEXT UUID

    CommandLine:
        python -m wbia.control.manual_name_funcs --test-set_name_texts

    RESTful:
        Method: PUT
        URL:    /api/name/text/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs.get_valid_nids()[0:2]
        >>> name_list = ibs.get_name_texts(nid_list)
        >>> result = set_name_texts(ibs, nid_list, name_list)
        >>> print(result)
    """
    import wbia

    if verbose:
        print('[ibs] setting %d name texts' % (len(name_rowid_list),))
    if notify_wildbook and wbia.ENABLE_WILDBOOK_SIGNAL:
        print('[ibs] notifying WildBook of name text changes')
        status_list = ibs.wildbook_signal_name_changes(name_rowid_list, name_text_list)

        wb_signaled = status_list is not None
        if assert_wildbook and wb_signaled:
            assert status_list, 'The request to WB failed'
            failed_nid_list = list(ut.ifilterfalse_items(name_rowid_list, status_list))
            args = (
                len(failed_nid_list),
                failed_nid_list,
            )
            msg = 'Failed to update %d WB names, nid_list = %r' % args
            assert len(failed_nid_list) == 0, msg
    ibsfuncs.assert_valid_names(name_text_list)
    old_name_text_list = ibs.get_name_texts(name_rowid_list)
    # sanitize_name_texts(ibs, name_text_list):
    # ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    # ibs.set_lblannot_values(nid_list, name_list)
    val_list = ((value,) for value in name_text_list)
    ibs.db.set(const.NAME_TABLE, (NAME_TEXT,), val_list, name_rowid_list)
    # Database updated, log name changes
    if update_json_log:
        import time

        json_log_path = ibs.get_logdir_local()
        json_log_filename = 'names.updates.json'
        json_log_filepath = os.path.join(json_log_path, json_log_filename)
        print('Logging name changes to: %r' % (json_log_filepath,))
        # Log has never been made, create one
        if not os.path.exists(json_log_filepath):
            json_dict = {
                'updates': [],
            }
            json_str = ut.to_json(json_dict, pretty=True)
            with open(json_log_filepath, 'w') as json_log_file:
                json_log_file.write(json_str)
        # Get current log state
        with open(json_log_filepath, 'r') as json_log_file:
            json_str = json_log_file.read()
        json_dict = ut.from_json(json_str)
        db_name = ibs.get_db_name()
        db_init_uuid = ibs.get_db_init_uuid()
        # Zip all the updates together and write to updates list in dictionary
        zipped = zip(name_rowid_list, old_name_text_list, name_text_list)
        for name_rowid, old_name_text, new_name_text in zipped:
            json_dict['updates'].append(
                {
                    'time_unixtime': time.time(),
                    'db_name': db_name,
                    'db_init_uuid': db_init_uuid,
                    'name_rowid': name_rowid,
                    'name_old_text': old_name_text,
                    'name_new_text': new_name_text,
                }
            )
        # Write new log state
        json_str = ut.to_json(json_dict, pretty=True)
        with open(json_log_filepath, 'w') as json_log_file:
            json_log_file.write(json_str)


@register_ibs_method
@register_api('/api/name/sex/', methods=['GET'])
def get_name_sex(ibs, name_rowid_list, eager=True, nInput=None):
    r"""
    name_sex_list <- name.name_sex[name_rowid_list]

    gets data from the "native" column "name_sex" in the "name" table

    Args:
        name_rowid_list (list):

    Returns:
        list: name_sex_list

    TemplateInfo:
        Tgetter_table_column
        col = name_sex
        tbl = name

    RESTful:
        Method: GET
        URL:    /api/name/sex/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> name_rowid_list = ibs._get_all_name_rowids()
        >>> eager = True
        >>> name_sex_list = ibs.get_name_sex(name_rowid_list, eager=eager)
        >>> assert len(name_rowid_list) == len(name_sex_list)
    """
    id_iter = name_rowid_list
    colnames = (NAME_SEX,)
    name_sex_list = ibs.db.get(
        const.NAME_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return name_sex_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/name/sex/', methods=['PUT'])
def set_name_sex(ibs, name_rowid_list, name_sex_list, duplicate_behavior='error'):
    r"""
    name_sex_list -> name.name_sex[name_rowid_list]

    Args:
        name_rowid_list
        name_sex_list

    TemplateInfo:
        Tsetter_native_column
        tbl = name
        col = name_sex

    RESTful:
        Method: PUT
        URL:    /api/name/sex/
    """
    id_iter = name_rowid_list
    colnames = (NAME_SEX,)
    ibs.db.set(
        const.NAME_TABLE,
        colnames,
        name_sex_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/name/sex/text/', methods=['GET'])
def get_name_sex_text(ibs, name_rowid_list, eager=True, nInput=None):
    r"""

    RESTful:
        Method: GET
        URL:    /api/name/sex/text/
    """
    name_sex_list = ibs.get_name_sex(name_rowid_list, eager=eager, nInput=nInput)
    name_sex_text_list = ut.dict_take(const.SEX_INT_TO_TEXT, name_sex_list)
    return name_sex_text_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/name/sex/text/', methods=['PUT'])
def set_name_sex_text(ibs, name_rowid_list, name_sex_text_list):
    r"""

    RESTful:
        Method: PUT
        URL:    /api/name/sex/text/
    """
    name_sex_list = ut.dict_take(const.SEX_TEXT_TO_INT, name_sex_text_list)
    return ibs.set_name_sex(name_rowid_list, name_sex_list)


@register_ibs_method
@register_api('/api/name/age/months/min/', methods=['GET'], __api_plural_check__=False)
def get_name_age_months_est_min(ibs, name_rowid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/name/age/months/min/
    """
    aids_list = ibs.get_name_aids(name_rowid_list)
    age_list = [ibs.get_annot_age_months_est_min(aid_list) for aid_list in aids_list]
    return age_list


@register_ibs_method
@register_api('/api/name/age/months/max/', methods=['GET'], __api_plural_check__=False)
def get_name_age_months_est_max(ibs, name_rowid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/name/age/months/max/
    """
    aids_list = ibs.get_name_aids(name_rowid_list)
    age_list = [ibs.get_annot_age_months_est_max(aid_list) for aid_list in aids_list]
    return age_list


@register_ibs_method
@register_api('/api/name/imageset/rowid/', methods=['GET'])
def get_name_imgsetids(ibs, nid_list):
    r"""

    RESTful:
        Method: GET
        URL:    /api/name/imageset/rowid/
    """
    import utool as ut

    name_aids_list = ibs.get_name_aids(nid_list)
    name_aid_list = ut.flatten(name_aids_list)
    name_gid_list = ibs.get_annot_gids(name_aid_list)
    name_imgsetids_list = ibs.get_image_imgsetids(name_gid_list)
    name_imgsetid_list = ut.flatten(name_imgsetids_list)
    name_imgsetids = list(set(name_imgsetid_list))
    return name_imgsetids


@register_ibs_method
@register_api('/api/name/imageset/uuid/', methods=['GET'])
def get_name_imgset_uuids(ibs, nid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/name/imageset/uuid/
    """
    name_imgsetids = ibs.get_name_imgsetids(nid_list)
    name_uuids_list = [
        ibs.get_imageset_uuids(name_imgsetid) for name_imgsetid in name_imgsetids
    ]
    return name_uuids_list


# def get_imageset_nids(ibs,


@register_ibs_method
@accessor_decors.getter
def get_name_has_split(ibs, nid_list):
    r"""
    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_name_speeds

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()
        >>> splits_list = ibs.get_name_has_split(nid_list)
        >>> result = str(splits_list)
        >>> print(result)
    """
    aids_list_ = ibs.get_name_aids(nid_list)
    # ibs.check_name_mapping_consistency(aids_list_)

    def get_valid_aids_clique_annotmatch_rowids(aids):
        import itertools

        aid_pairs = list(itertools.combinations(aids, 2))
        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)
        am_ids = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        am_ids = ut.filter_Nones(am_ids)
        return am_ids

    amids_list = [get_valid_aids_clique_annotmatch_rowids(aids) for aids in aids_list_]
    flags_list = ibs.unflat_map(
        ut.partial(ibs.get_annotmatch_prop, 'SplitCase'), amids_list
    )
    has_splits = list(map(any, flags_list))
    return has_splits


@register_ibs_method
def get_name_speeds(ibs, nid_list):
    r"""
    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_name_speeds

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()
        >>> speeds_list = get_name_speeds(ibs, nid_list)
        >>> result = str(speeds_list)
        >>> print(result)
    """
    aids_list = ibs.get_name_aids(nid_list)
    # ibs.check_name_mapping_consistency(aids_list_)
    # aids_list = [(aids) for aids in aids_list_]
    # speeds_list = ibs.get_unflat_annots_speeds_list(aids_list)
    speeds_list = ibs.get_unflat_annots_speeds_list2(aids_list)
    return speeds_list


@register_ibs_method
@accessor_decors.getter
def get_name_hourdiffs(ibs, nid_list):
    """
    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_name_hourdiffs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()
        >>> hourdiffs_list = ibs.get_name_hourdiffs(nid_list)
        >>> result = hourdiffs_list
        >>> print(hourdiffs_list)
    """
    aids_list_ = ibs.get_name_aids(nid_list)
    # ibs.check_name_mapping_consistency(aids_list_)
    # HACK FILTERING SHOULD NOT OCCUR HERE
    aids_list = [(aids) for aids in aids_list_]
    hourdiffs_list = ibs.get_unflat_annots_hourdists_list(aids_list)
    return hourdiffs_list


@register_ibs_method
@accessor_decors.getter
def get_name_max_hourdiff(ibs, nid_list):
    hourdiffs_list = ibs.get_name_hourdiffs(nid_list)
    maxhourdiff_list_ = np.array(
        [vt.safe_max(hourdiff, nans=False) for hourdiff in hourdiffs_list]
    )
    maxhourdiff_list = np.array(maxhourdiff_list_)
    return maxhourdiff_list


@register_ibs_method
@accessor_decors.getter
def get_name_max_speed(ibs, nid_list):
    """
    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_name_max_speed

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()
        >>> maxspeed_list = ibs.get_name_max_speed(nid_list)
        >>> result = maxspeed_list
        >>> print(maxspeed_list)
    """
    speeds_list = ibs.get_name_speeds(nid_list)
    maxspeed_list = np.array([vt.safe_max(speeds, nans=False) for speeds in speeds_list])
    return maxspeed_list


@register_ibs_method
def get_name_gps_tracks(ibs, nid_list=None, aid_list=None):
    """
    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_name_gps_tracks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_name_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> #ibs = wbia.opendb('PZ_Master0')
        >>> ibs = wbia.opendb('testdb1')
        >>> #nid_list = ibs.get_valid_nids()
        >>> aid_list = ibs.get_valid_aids()
        >>> nid_list, gps_track_list, aid_track_list = ibs.get_name_gps_tracks(aid_list=aid_list)
        >>> nonempty_list = list(map(lambda x: len(x) > 0, gps_track_list))
        >>> ut.compress(nid_list, nonempty_list)
        >>> ut.compress(gps_track_list, nonempty_list)
        >>> ut.compress(aid_track_list, nonempty_list)
        >>> aid_track_list = list(map(sorted, aid_track_list))
        >>> result = str(aid_track_list)
        >>> print(result)
        [[11], [], [4], [1], [2, 3], [5, 6], [7], [8], [10], [12], [13]]
    """
    assert aid_list is None or nid_list is None, 'only specify one please'
    if aid_list is None:
        aids_list_ = ibs.get_name_aids(nid_list)
    else:
        aids_list_, nid_list = ibs.group_annots_by_name(aid_list)
    aids_list = [
        ut.sortedby(aids, ibs.get_annot_image_unixtimes(aids)) for aids in aids_list_
    ]
    gids_list = ibs.unflat_map(ibs.get_annot_gids, aids_list)
    gpss_list = ibs.unflat_map(ibs.get_image_gps, gids_list)

    isvalids_list = [
        [gps[0] != -1.0 or gps[1] != -1.0 for gps in gpss] for gpss in gpss_list
    ]
    gps_track_list = [
        ut.compress(gpss, isvalids) for gpss, isvalids in zip(gpss_list, isvalids_list)
    ]
    aid_track_list = [
        ut.compress(aids, isvalids) for aids, isvalids in zip(aids_list, isvalids_list)
    ]
    return nid_list, gps_track_list, aid_track_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_name_funcs
        python -m wbia.control.manual_name_funcs --allexamples

        python -m wbia.control.manual_name_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
