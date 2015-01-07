from __future__ import absolute_import, division, print_function
# TODO: Fix this name it is too special case
import uuid
import functools
import six  # NOQA
#from six.moves import range
from ibeis import constants as const
from ibeis import ibsfuncs
import numpy as np
import vtool as vt
from ibeis.control.accessor_decors import (adder, deleter, setter, getter_1to1,
                                           getter_1toM, ider)
from ibeis.control import accessor_decors  # NOQA
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[autogen_name_species]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


ANNOT_ROWID         = 'annot_rowid'
ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
NAME_ROWID          = 'name_rowid'
SPECIES_ROWID       = 'species_rowid'

NAME_UUID = 'name_uuid'
NAME_TEXT = 'name_text'
NAME_ALIAS_TEXT = 'name_alias_text'
NAME_NOTE = 'name_note'
SPECIES_UUID = 'species_uuid'
SPECIES_TEXT = 'species_text'
SPECIES_NOTE = 'species_note'


@register_ibs_method
@ider
def _get_all_known_name_rowids(ibs):
    """
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    #all_known_nids = ibs._get_all_known_lblannot_rowids(const.INDIVIDUAL_KEY)
    all_known_nids = ibs.db.get_all_rowids(const.NAME_TABLE)
    return all_known_nids


@register_ibs_method
def _get_all_known_nids(ibs):
    """ alias """
    return _get_all_known_name_rowids(ibs)


@register_ibs_method
@ider
def _get_all_species_rowids(ibs):
    """
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    #all_known_species_rowids = ibs._get_all_known_lblannot_rowids(const.SPECIES_KEY)
    all_known_species_rowids = ibs.db.get_all_rowids(const.SPECIES_TABLE)
    return all_known_species_rowids


@register_ibs_method
@adder
def add_names(ibs, name_text_list, name_uuid_list=None, name_note_list=None):
    """
    Adds a list of names.

    Returns:
        name_rowid_list (list): their nids
    """
    if name_note_list is None:
        name_note_list = [''] * len(name_text_list)
    # Get random uuids
    if name_uuid_list is None:
        name_uuid_list = [uuid.uuid4() for _ in range(len(name_text_list))]
    get_rowid_from_superkey = functools.partial(ibs.get_name_rowids_from_text, ensure=False)
    superkey_paramx = (1,)
    colnames = [NAME_UUID, NAME_TEXT, NAME_NOTE]
    params_iter = list(zip(name_uuid_list, name_text_list, name_note_list))
    name_rowid_list = ibs.db.add_cleanly(const.NAME_TABLE, colnames, params_iter,
                                             get_rowid_from_superkey, superkey_paramx)
    return name_rowid_list
    # OLD WAY
    ## nid_list_ = [namenid_dict[name] for name in name_list_]
    #name_text_list_ = ibs.sanatize_name_texts(name_text_list)
    ## All names are individuals and so may safely receive the INDIVIDUAL_KEY lblannot
    #lbltype_rowid = ibs.lbltype_ids[const.INDIVIDUAL_KEY]
    #lbltype_rowid_list = [lbltype_rowid] * len(name_text_list_)
    #nid_list = ibs.add_lblannots(lbltype_rowid_list, name_text_list_, note_list)
    ##nid_list = [ibs.UNKNOWN_NAME_ROWID if rowid is None else rowid for rowid in nid_list]
    #return nid_list


#def init_default_speciesvalue():
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
def sanatize_species_texts(ibs, species_text_list):
    """ changes unknown species to the unknown value """
    ibsfuncs.assert_valid_species(ibs, species_text_list, iswarning=True)
    def _sanatize_species_text(species_text):
        if species_text is None:
            return None
        elif species_text in const.VALID_SPECIES:
            return species_text
        else:
            return const.UNKNOWN
    species_text_list_ = [_sanatize_species_text(species_text)
                          for species_text in species_text_list]
    # old but same logic
    #species_text_list_ = [None if species_text is None else
    #                      species_text if species_text in const.VALID_SPECIES else
    #                      const.UNKNOWN
    #                      for species_text in species_text_list]
    # oldest different logic
    #species_text_list_ = [None
    #                      if species_text is None or species_text == const.UNKNOWN
    #                      else species_text.lower()
    #                      for species_text in species_text_list]
    #species_text_list_ = [species_text if species_text in const.VALID_SPECIES else None
    #                      for species_text in species_text_list_]
    return species_text_list_


@register_ibs_method
def sanatize_name_texts(ibs, name_text_list):
    ibsfuncs.assert_valid_names(name_text_list)
    name_text_list_ = [None
                       if name_text == const.UNKNOWN
                       else name_text
                       for name_text in name_text_list]
    return name_text_list_


@register_ibs_method
@adder
def add_species(ibs, species_text_list, species_uuid_list=None, note_list=None):
    """
    Adds a list of species.

    Returns:
        list: speciesid_list - species rowids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> ibs = ibeis.opendb('testdb1')
        >>> species_text_list = [
        ...     u'jaguar', u'zebra_plains', u'zebra_plains', '____', 'TYPO',
        ...     '____', u'zebra_grevys', u'bear_polar']
        >>> species_rowid_list = ibs.add_species(species_text_list)
        >>> print(ut.list_str(list(zip(species_text_list, species_rowid_list))))
        >>> ibs.print_species_table()
        >>> species_text = ibs.get_species_texts(species_rowid_list)
        >>> # Ensure we leave testdb1 in a clean state
        >>> ibs.delete_species(ibs.get_species_rowids_from_text(['jaguar', 'TYPO']))
        >>> all_species_rowids = ibs._get_all_species_rowids()
        >>> result = str(species_text) + '\n'
        >>> result += str(all_species_rowids) + '\n'
        >>> result += str(ibs.get_species_texts(all_species_rowids))
        >>> print(result)
        ['____', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']
        [1, 2, 3]
        [u'zebra_plains', u'zebra_grevys', u'bear_polar']

    [u'jaguar', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']
    [8, 9, 10]
    [u'zebra_plains', u'zebra_grevys', u'bear_polar']

    """
    if note_list is None:
        note_list = [''] * len(species_text_list)
    # Sanatize to remove ____
    species_text_list_ = ibs.sanatize_species_texts(species_text_list)
    # Get random uuids
    if species_uuid_list is None:
        species_uuid_list = [uuid.uuid4() for _ in range(len(species_text_list))]
    superkey_paramx = (1,)
    # TODO Allow for better ensure=False without using partial
    # Just autogenerate these functions
    get_rowid_from_superkey = functools.partial(ibs.get_species_rowids_from_text, ensure=False)
    colnames = [SPECIES_UUID, SPECIES_TEXT, SPECIES_NOTE]
    params_iter = list(zip(species_uuid_list, species_text_list_, note_list))
    species_rowid_list = ibs.db.add_cleanly(const.SPECIES_TABLE, colnames, params_iter,
                                             get_rowid_from_superkey, superkey_paramx)
    return species_rowid_list
    #value_list = ibs.sanatize_species_texts(species_text_list)
    #lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
    #lbltype_rowid_list = [lbltype_rowid] * len(species_text_list)
    #species_rowid_list = ibs.add_lblannots(lbltype_rowid_list, value_list, note_list)
    ##species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID if rowid is None else
    ##                      rowid for rowid in species_rowid_list]
    #return species_rowid_list


@register_ibs_method
@deleter
#@cache_invalidator(const.NAME_TABLE)
def delete_names(ibs, name_rowid_list):
    """
    deletes names from the database

    CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
    at least ensure that no annot is associated with any of these nids
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d names' % len(name_rowid_list))
    ibs.db.delete_rowids(const.NAME_TABLE, name_rowid_list)
    #ibs.delete_lblannots(nid_list)


@register_ibs_method
@deleter
#@cache_invalidator(const.SPECIES_TABLE)
def delete_species(ibs, species_rowid_list):
    """
    deletes species from the database

    CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
    at least ensure that no annot is associated with any of these species rowids
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d speciess' % len(species_rowid_list))
    ibs.db.delete_rowids(const.SPECIES_TABLE, species_rowid_list)
    #ibs.delete_lblannots(species_rowid_list)


@register_ibs_method
@ider
def get_invalid_nids(ibs):
    """
    Returns:
        list: nid_list - all names without any animals (does not include unknown names)
        an nid is not invalid if it has a valid alias

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_invalid_nids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> nids_list = get_invalid_nids(ibs)
        >>> result = str(nids_list)
        >>> print(result)
        []
    """
    _nid_list = ibs._get_all_known_name_rowids()
    nRois_list = ibs.get_name_num_annotations(_nid_list)
    # Filter names with rois
    isempty_list = (nRois <= 0 for nRois in nRois_list)
    nid_list = list(ut.ifilter_items(_nid_list, isempty_list))
    # Filter names with aliases (TODO: use transitivity to determine validity)
    hasalias_list = [alias_text is not None for alias_text in ibs.get_name_alias_texts(nid_list)]
    nid_list = list(ut.ifilterfalse_items(nid_list, hasalias_list))
    return nid_list


@register_ibs_method
@getter_1toM
#@cache_getter(const.NAME_TABLE, ANNOT_ROWID, native_rowids=False)
def get_name_aids(ibs, nid_list, enable_unknown_fix=True):
    """
    # TODO: Rename to get_anot_rowids_from_name_rowid

    Returns:
         list: aids_list a list of list of aids in each name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> # Map annotations to name ids
        >>> aid_list = ibs.get_valid_aids()
        >>> nid_list = ibs.get_annot_name_rowids(aid_list)
        >>> # Get annotation ids for each name
        >>> aids_list = ibs.get_name_aids(nid_list)
        >>> # Run Assertion Test
        >>> groupid2_items = ut.group_items(aids_list, nid_list)
        >>> grouped_items = list(six.itervalues(groupid2_items))
        >>> passed_iter = map(ut.list_allsame, grouped_items)
        >>> passed_list = list(passed_iter)
        >>> assert all(passed_list), 'problem in get_name_aids'
        >>> # Print gropued items
        >>> print(ut.dict_str(groupid2_items, newlines=False))

    Ignore;
        from ibeis.control.manual_name_species_funcs import *  # NOQA
        import ibeis
        #ibs = ibeis.opendb('testdb1')
        #ibs = ibeis.opendb('PZ_MTEST')
        ibs = ibeis.opendb('PZ_Master0')
        #ibs = ibeis.opendb('GZ_ALL')

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
            unique_nids, groupx = vt.group_indicies(nids)
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
        unique_nids, groupx = vt.group_indicies(nids)
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
    #return [[] for nid in nid_list]
    # TODO: should a query of the UNKNOWN_NAME_ROWID return anything?
    # TODO: don't even run negative aids as queries
    nid_list_ = [const.UNKNOWN_NAME_ROWID if nid <= 0 else nid for nid in nid_list]
    USE_GROUPING_HACK = False
    if USE_GROUPING_HACK:
        # This code doesn't work because it doesn't respect empty names
        input_list, inverse_unique = np.unique(nid_list_, return_inverse=True)
        input_str = ', '.join(list(map(str, input_list)))
        opstr = '''
        SELECT annot_rowid, name_rowid
        FROM {ANNOTATION_TABLE}
        WHERE name_rowid IN
            ({input_str})
            ORDER BY name_rowid ASC, annot_rowid ASC
        '''.format(input_str=input_str, ANNOTATION_TABLE=const.ANNOTATION_TABLE)
        pair_list = ibs.db.connection.execute(opstr).fetchall()
        aidscol = np.array(ut.get_list_column(pair_list, 0))
        nidscol = np.array(ut.get_list_column(pair_list, 1))
        unique_nids, groupx = vt.group_indicies(nidscol)
        grouped_aids_ = vt.apply_grouping(aidscol, groupx)
        #aids_list = [sorted(arr.tolist()) for arr in grouped_aids_]
        structured_aids_list = [arr.tolist() for arr in grouped_aids_]
        aids_list = np.array(structured_aids_list)[inverse_unique].tolist()
    else:
        USE_NUMPY_IMPL = True
        #USE_NUMPY_IMPL = False
        # Use qt if getting one at a time otherwise perform bulk operation
        USE_NUMPY_IMPL = len(nid_list_) > 1
        #USE_NUMPY_IMPL = len(nid_list_) > 10
        if USE_NUMPY_IMPL:
            # This seems to be 30x faster for bigger inputs
            valid_aids = np.array(ibs._get_all_aids())
            valid_nids = np.array(ibs.db.get_all_col_rows(const.ANNOTATION_TABLE, NAME_ROWID))
            #np.array(ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False))
            aids_list = [valid_aids.take(np.flatnonzero(np.equal(valid_nids, nid))).tolist() for nid in nid_list_]
        else:
            # SQL IMPL
            aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,), nid_list_, id_colname=NAME_ROWID, unpack_scalars=False)
    if enable_unknown_fix:
        # negative name rowids correspond to unknown annoations wherex annot_rowid = -name_rowid
        aids_list = [[-nid] if nid < 0 else aids for nid, aids in zip(nid_list, aids_list)]
    return aids_list


@register_ibs_method
@getter_1toM
def get_name_exemplar_aids(ibs, nid_list):
    """
    Returns:
        list_ (list):  a list of list of cids in each name


    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_exemplar_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
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
    exemplar_aids_list = [ut.filter_items(aids, flags) for aids, flags in
                          zip(aids_list, flags_list)]
    return exemplar_aids_list


@register_ibs_method
@getter_1toM
def get_name_gids(ibs, nid_list):
    """
    Returns:
        list_ (list): the image ids associated with name ids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
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
@getter_1to1
def get_name_uuids(ibs, name_rowid_list):
    """
    Returns:
        list_ (list): uuids_list - name uuids
    """
    uuids_list = ibs.db.get(const.NAME_TABLE, (NAME_UUID,), name_rowid_list)
    #notes_list = ibs.get_lblannot_notes(nid_list)
    return uuids_list


@register_ibs_method
@getter_1to1
def get_name_notes(ibs, name_rowid_list):
    """
    Returns:
        list_ (list): notes_list - name notes
    """
    notes_list = ibs.db.get(const.NAME_TABLE, (NAME_NOTE,), name_rowid_list)
    #notes_list = ibs.get_lblannot_notes(nid_list)
    return notes_list


@register_ibs_method
@getter_1to1
def get_name_num_annotations(ibs, nid_list):
    """
    Returns:
        list_ (list):  the number of annotations for each name

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_num_annotations

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_name_rowids()
        >>> result = get_name_num_annotations(ibs, nid_list)
        >>> print(result)
        [2, 2, 1, 1, 1, 1, 1]
    """
    # TODO: Optimize
    return list(map(len, ibs.get_name_aids(nid_list, enable_unknown_fix=True)))


@register_ibs_method
@getter_1to1
def get_name_num_exemplar_annotations(ibs, nid_list):
    """
    Returns:
        list_ (list):  the number of annotations, which are exemplars for each name
    """
    return list(map(len, ibs.get_name_exemplar_aids(nid_list)))


@register_ibs_method
@getter_1to1
def get_name_alias_texts(ibs, name_rowid_list):
    """
    Returns:
        list_ (list): name_alias_text_list

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_texts

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_alias_texts

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> name_rowid_list = ibs.get_valid_nids()
        >>> # execute function
        >>> name_alias_text_list = get_name_alias_texts(ibs, name_rowid_list)
        >>> # verify results
        >>> result = str(name_alias_text_list)
        >>> print(result)
        [None, None, None, None, None, None, None]
    """
    name_alias_text_list = ibs.db.get(const.NAME_TABLE, (NAME_ALIAS_TEXT,), name_rowid_list)
    return name_alias_text_list


@register_ibs_method
@accessor_decors.dev_cache_invalidator(const.ANNOTATION_TABLE, ANNOT_SEMANTIC_UUID, native_rowids=False)
@setter
def set_name_alias_texts(ibs, name_rowid_list, name_alias_text_list):
    """
    Returns:
        list_ (list): name_alias_text_list

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_texts
    """
    #ibsfuncs.assert_valid_names(name_alias_text_list)
    val_list = ((value,) for value in name_alias_text_list)
    ibs.db.set(const.NAME_TABLE, (NAME_ALIAS_TEXT,), val_list, name_rowid_list)
    # TODO: ibs.update_annot_semantic_uuids(aid_list)


@register_ibs_method
@getter_1to1
def get_name_texts(ibs, name_rowid_list, apply_fix=True):
    """
    Returns:
        list_ (list): text names

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_texts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> name_rowid_list = ibs._get_all_known_name_rowids()
        >>> name_text_list = get_name_texts(ibs, name_rowid_list)
        >>> result = str(name_text_list)
        >>> print(result)
        [u'easy', u'hard', u'jeff', u'lena', u'occl', u'polar', u'zebra']
    """
    # FIXME: Use standalone name table
    # TODO:
    # Change the temporary negative indexes back to the unknown NID for the
    # SQL query. Then augment the lblannot list to distinguish unknown lblannots
    #name_text_list = ibs.get_lblannot_values(nid_list, const.INDIVIDUAL_KEY)
    #name_text_list = ibs.get_lblannot_values(nid_list, const.INDIVIDUAL_KEY)
    name_text_list = ibs.db.get(const.NAME_TABLE, (NAME_TEXT,), name_rowid_list)
    if apply_fix:
        name_text_list = [const.UNKNOWN
                          if rowid == ibs.UNKNOWN_NAME_ROWID or name_text is None
                          else name_text
                          for name_text, rowid in zip(name_text_list, name_rowid_list)]
    return name_text_list


@register_ibs_method
def get_num_names(ibs, **kwargs):
    r"""
    Number of valid names

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_num_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = get_num_names(ibs)
        >>> print(result)
        7
    """
    nid_list = ibs.get_valid_nids(**kwargs)
    return len(nid_list)


@register_ibs_method
@getter_1to1
def get_species_rowids_from_text(ibs, species_text_list, ensure=True):
    r"""
    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_species_rowids_from_text:0
        python -m ibeis.control.manual_name_species_funcs --test-get_species_rowids_from_text:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> ibs = ibeis.opendb('testdb1')
        >>> species_text_list = [
        ...     u'jaguar', u'zebra_plains', u'zebra_plains', '____', 'TYPO',
        ...     '____', u'zebra_grevys', u'bear_polar']
        >>> ensure = False
        >>> species_rowid_list = ibs.get_species_rowids_from_text(species_text_list, ensure)
        >>> print(ut.list_str(list(zip(species_text_list, species_rowid_list))))
        >>> ensure = True
        >>> species_rowid_list = ibs.get_species_rowids_from_text(species_text_list, ensure)
        >>> print(ut.list_str(list(zip(species_text_list, species_rowid_list))))
        >>> ibs.print_species_table()
        >>> species_text = ibs.get_species_texts(species_rowid_list)
        >>> # Ensure we leave testdb1 in a clean state
        >>> ibs.delete_species(ibs.get_species_rowids_from_text(['jaguar', 'TYPO']))
        >>> all_species_rowids = ibs._get_all_species_rowids()
        >>> result = str(species_text) + '\n'
        >>> result += str(all_species_rowids) + '\n'
        >>> result += str(ibs.get_species_texts(all_species_rowids))
        >>> print(result)
        [u'jaguar', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']
        [1, 2, 3]
        [u'zebra_plains', u'zebra_grevys', u'bear_polar']

    [u'jaguar', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']
    [8, 9, 10]
    [u'zebra_plains', u'zebra_grevys', u'bear_polar']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> import utool as ut  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> species_text_list = [
        ...     u'jaguar', u'zebra_plains', u'zebra_plains', '____', 'TYPO',
        ...     '____', u'zebra_grevys', u'bear_polar']
        >>> ensure = False
        >>> species_rowid_list = ibs.get_species_rowids_from_text(species_text_list, ensure)

    """
    if ensure:
        species_rowid_list = ibs.add_species(species_text_list)
    else:
        species_text_list_ = ibs.sanatize_species_texts(species_text_list)
        #lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
        #lbltype_rowid_list = [lbltype_rowid] * len(species_text_list_)
        #species_rowid_list = ibs.get_lblannot_rowid_from_superkey(lbltype_rowid_list, species_text_list_)
        ## Ugg species and names need their own table
        #species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID if rowid is None else
        #                      rowid for rowid in species_rowid_list]
        species_rowid_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_ROWID,), species_text_list_, id_colname=SPECIES_TEXT)
        # BIG HACK FOR ENFORCING UNKNOWN SPECIESS HAVE ROWID 0
        species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID if text is None or text == const.UNKNOWN else rowid
                               for rowid, text in zip(species_rowid_list, species_text_list_)]
    return species_rowid_list


@register_ibs_method
@getter_1to1
def get_name_rowids_from_text(ibs, name_text_list, ensure=True):
    r"""

    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_name_rowids_from_text:0
        python -m ibeis.control.manual_name_species_funcs --test-get_name_rowids_from_text:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> ibs = ibeis.opendb('testdb1')
        >>> name_text_list = [u'Fred', u'Sue', '____', u'zebra_grevys', 'TYPO', '____']
        >>> ensure = False
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> print(ut.list_str(list(zip(name_text_list, name_rowid_list))))
        >>> ensure = True
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> print(ut.list_str(list(zip(name_text_list, name_rowid_list))))
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

    [0, 0, 0, 0, 0, 0]
    [1, 2, 3, 4, 5, 6, 7]
    [11, 12, 0, 13, 14, 0]
    [1, 2, 3, 4, 5, 6, 7]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> import utool as ut  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> name_text_list = [u'Fred', 'easy', u'Sue', '____', u'zebra_grevys', 'TYPO', 'jeff']
        >>> ensure = False
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> ibs.print_name_table()
        >>> result = str(name_rowid_list)
        >>> print(result)
        [None, 1, None, 0, None, None, 3]

    """
    if ensure:
        name_rowid_list = ibs.add_names(name_text_list)
    else:
        name_text_list_ = ibs.sanatize_name_texts(name_text_list)
        #lbltype_rowid = ibs.lbltype_ids[const.INDIVIDUAL_KEY]
        #lbltype_rowid_list = [lbltype_rowid] * len(name_text_list_)
        #name_rowid_list = ibs.get_lblannot_rowid_from_superkey(lbltype_rowid_list, name_text_list_)
        ## Ugg species and names need their own table
        #name_rowid_list = [ibs.UNKNOWN_NAME_ROWID if rowid is None else
        #                      rowid for rowid in name_rowid_list]
        #val_iter = [(text,) for text in name_text_list_]
        name_rowid_list = ibs.db.get(const.NAME_TABLE, (NAME_ROWID,), name_text_list_, id_colname=NAME_TEXT)
        # BIG HACK FOR ENFORCING UNKNOWN NAMES HAVE ROWID 0
        #name_rowid_list = [ibs.UNKNOWN_NAME_ROWID if rowid is None else
        #                      rowid for rowid in name_rowid_list]
        #name_rowid_list = [ibs.UNKNOWN_NAME_ROWID if val is None or val == const.UNKNOWN else rowid
        #                       for rowid, val in zip(name_rowid_list, name_text_list)]
        name_rowid_list = [ibs.UNKNOWN_NAME_ROWID if text is None or text == const.UNKNOWN else rowid
                               for rowid, text in zip(name_rowid_list, name_text_list_)]

    return name_rowid_list


@register_ibs_method
@getter_1to1
def get_species_texts(ibs, species_rowid_list):
    """
    Returns:
        list: species_text_list text names

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_species_texts --enableall

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> species_rowid_list = ibs._get_all_species_rowids()
        >>> result = get_species_texts(ibs, species_rowid_list)
        >>> print(result)
        [u'zebra_plains', u'zebra_grevys', u'bear_polar']
    """
    # FIXME: use standalone species table
    #species_text_list = ibs.get_lblannot_values(species_rowid_list, const.SPECIES_KEY)
    species_text_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_TEXT,), species_rowid_list)
    species_text_list = [const.UNKNOWN
                         if rowid == ibs.UNKNOWN_SPECIES_ROWID else species_text
                         for species_text, rowid in zip(species_text_list, species_rowid_list)]
    return species_text_list


@register_ibs_method
@getter_1to1
def get_species_uuids(ibs, species_rowid_list):
    """
    Returns:
        list_ (list): uuids_list - species uuids
    """
    uuids_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_UUID,), species_rowid_list)
    #notes_list = ibs.get_lblannot_notes(nid_list)
    return uuids_list


@register_ibs_method
@getter_1to1
def get_species_notes(ibs, species_rowid_list):
    """
    Returns:
        list_ (list): notes_list - species notes
    """
    notes_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_NOTE,), species_rowid_list)
    #notes_list = ibs.get_lblannot_notes(nid_list)
    return notes_list


@register_ibs_method
@ider
def get_valid_nids(ibs, eid=None, filter_empty=False):
    """
    Returns:
        list_ (list): all valid names with at least one animal
        (does not include unknown names)
    """
    if eid is None:
        _nid_list = ibs._get_all_known_name_rowids()
    else:
        _nid_list = ibs.get_encounter_nids(eid)
    if filter_empty:
        nRois_list = ibs.get_name_num_annotations(_nid_list)
        nonempty_list = (nRois > 0 for nRois in nRois_list)
        nid_list = list(ut.ifilter_items(_nid_list, nonempty_list))
    else:
        nid_list = _nid_list
    return nid_list


@register_ibs_method
@setter
def set_name_notes(ibs, name_rowid_list, notes_list):
    """ Sets a note for each name (multiple annotations) """
    #ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    #ibs.set_lblannot_notes(nid_list, notes_list)
    val_list = ((value,) for value in notes_list)
    ibs.db.set(const.NAME_TABLE, (NAME_NOTE,), val_list, name_rowid_list)


@register_ibs_method
@setter
def set_name_texts(ibs, name_rowid_list, name_text_list):
    """
    Changes the name text. Does not affect the animals of this name.
    Effectively just changes the TEXT UUID

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-set_name_texts

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> nid_list = ibs.get_valid_nids()[0:2]
        >>> name_list = ibs.get_name_texts(nid_list)
        >>> # result = set_name_texts(ibs, nid_list, name_list)
        >>> print(result)
    """
    ibsfuncs.assert_valid_names(name_text_list)
    #sanatize_name_texts(ibs, name_text_list):
    #ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    #ibs.set_lblannot_values(nid_list, name_list)
    val_list = ((value,) for value in name_text_list)
    ibs.db.set(const.NAME_TABLE, (NAME_TEXT,), val_list, name_rowid_list)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_name_species_funcs
        python -m ibeis.control.manual_name_species_funcs --allexamples
        python -m ibeis.control.manual_name_species_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
