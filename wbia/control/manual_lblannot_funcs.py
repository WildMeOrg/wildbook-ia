# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import uuid
from six.moves import range
from wbia import constants as const
from wbia.other import ibsfuncs
from wbia.control.accessor_decors import (
    adder,
    deleter,
    setter,
    getter_1to1,
    getter_1toM,
    ider,
)
import utool as ut
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)
LBLANNOT_VALUE = 'lblannot_value'


# @ider
# def _get_all_alr_rowids(ibs):
#    all_alr_rowids = ibs.db.get_all_rowids(const.AL_RELATION_TABLE)
#    return all_alr_rowids


@register_ibs_method
@ider
def _get_all_lblannot_rowids(ibs):
    all_lblannot_rowids = ibs.db.get_all_rowids(const.LBLANNOT_TABLE)
    return all_lblannot_rowids


@register_ibs_method
@adder
def add_annot_relationship(ibs, aid_list, lblannot_rowid_list, alr_confidence_list=None):
    """
    Adds a relationship between annots and lblannots
        (annotations and labels of annotations)
    """
    if alr_confidence_list is None:
        alr_confidence_list = [0.0] * len(aid_list)
    colnames = (
        'annot_rowid',
        'lblannot_rowid',
        'alr_confidence',
    )
    params_iter = list(zip(aid_list, lblannot_rowid_list, alr_confidence_list))
    get_rowid_from_superkey = ibs.get_alrid_from_superkey
    superkey_paramx = (0, 1)  # TODO HAVE SQL GIVE YOU THESE NUMBERS
    alrid_list = ibs.db.add_cleanly(
        const.AL_RELATION_TABLE,
        colnames,
        params_iter,
        get_rowid_from_superkey,
        superkey_paramx,
    )
    return alrid_list


@register_ibs_method
@adder
def add_lblannots(
    ibs, lbltype_rowid_list, value_list, note_list=None, lblannot_uuid_list=None
):
    """ Adds new lblannots (labels of annotations)
    creates a new uuid for any new pair(type, value)
    #TODO: reverse order of rowid_list value_list in input
    """
    if note_list is None:
        note_list = [''] * len(value_list)
    # Get random uuids
    if lblannot_uuid_list is None:
        lblannot_uuid_list = [uuid.uuid4() for _ in range(len(value_list))]
    colnames = ['lblannot_uuid', 'lbltype_rowid', LBLANNOT_VALUE, 'lblannot_note']
    params_iter = list(zip(lblannot_uuid_list, lbltype_rowid_list, value_list, note_list))
    get_rowid_from_superkey = ibs.get_lblannot_rowid_from_superkey
    superkey_paramx = (1, 2)
    lblannot_rowid_list = ibs.db.add_cleanly(
        const.LBLANNOT_TABLE,
        colnames,
        params_iter,
        get_rowid_from_superkey,
        superkey_paramx,
    )
    return lblannot_rowid_list


@register_ibs_method
@getter_1to1
def get_lblannot_rowid_from_superkey(ibs, lbltype_rowid_list, value_list):
    """
    Returns:
        list_ (list):  lblannot_rowid_list from the superkey (lbltype, value)
    """
    colnames = ('lblannot_rowid',)
    params_iter = zip(lbltype_rowid_list, value_list)
    where_clause = 'lbltype_rowid=? AND lblannot_value=?'
    lblannot_rowid_list = ibs.db.get_where(
        const.LBLANNOT_TABLE, colnames, params_iter, where_clause
    )
    # BIG HACK FOR ENFORCING UNKNOWN LBLANNOTS HAVE ROWID 0
    lblannot_rowid_list = [
        const.UNKNOWN_LBLANNOT_ROWID if val is None or val == const.UNKNOWN else rowid
        for rowid, val in zip(lblannot_rowid_list, value_list)
    ]
    return lblannot_rowid_list


@register_ibs_method
@deleter
def delete_annot_relations(ibs, aid_list):
    """ Deletes the relationship between an annotation and a label """
    alrids_list = ibs.get_annot_alrids(aid_list)
    alrid_list = ut.flatten(alrids_list)
    ibs.db.delete_rowids(const.AL_RELATION_TABLE, alrid_list)


@register_ibs_method
@deleter
def delete_annot_relations_oftype(ibs, aid_list, _lbltype):
    """ Deletes the relationship between an annotation and a label """
    alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
    alrid_list = ut.flatten(alrids_list)
    ibs.db.delete_rowids(const.AL_RELATION_TABLE, alrid_list)


@register_ibs_method
@deleter
def delete_lblannots(ibs, lblannot_rowid_list):
    """ deletes lblannots from the database """
    if ut.VERBOSE:
        print('[ibs] deleting %d lblannots' % len(lblannot_rowid_list))
    ibs.db.delete_rowids(const.LBLANNOT_TABLE, lblannot_rowid_list)


@register_ibs_method
@getter_1to1
def get_alr_annot_rowids(ibs, alrid_list):
    """
    Args:
        alrid_list (list of rowids): annot + label relationship rows
    get the annot_rowid belonging to each relationship """
    annot_rowids_list = ibs.db.get(const.AL_RELATION_TABLE, ('annot_rowid',), alrid_list)
    return annot_rowids_list


@register_ibs_method
def get_alr_annot_rowids_from_lblannot_rowid(ibs, lblannot_rowid_list):
    """
    This is a 1toM getter

    Get annotation rowids of labels. There may be more than one annotation
    per label.

    Args:
        lblannot_rowid_list (list): of lblannot (labels of annotations) rowids

    Returns:
        aids_list (list): of lists annotation rowids
    """
    # verbose = len(lblannot_rowid_list) > 20
    # TODO: Optimize IF POSSIBLE
    # FIXME: SLOW
    # if verbose:
    #    print(ut.get_caller_name(N=list(range(0, 20))))
    where_clause = 'lblannot_rowid=?'
    params_iter = [(lblannot_rowid,) for lblannot_rowid in lblannot_rowid_list]
    aids_list = ibs.db.get_where(
        const.AL_RELATION_TABLE,
        ('annot_rowid',),
        params_iter,
        where_clause,
        unpack_scalars=False,
    )
    return aids_list


@register_ibs_method
@getter_1to1
def get_alr_confidence(ibs, alrid_list):
    """
    Args:
        alrid_list (list of rowids): annot + label relationship rows
    Returns:
        alr_confidence_list (list of rowids): confidence in an annotation relationship
    """
    alr_confidence_list = ibs.db.get(
        const.AL_RELATION_TABLE, ('alr_confidence',), alrid_list
    )
    return alr_confidence_list


@register_ibs_method
@getter_1to1
def get_alr_lblannot_rowids(ibs, alrid_list):
    """
    Args:
        alrid_list (list of rowids): annot + label relationship rows
    Returns:
        lblannot_rowids_list (list of rowids): label rowids (of annotations)
    """
    lblannot_rowids_list = ibs.db.get(
        const.AL_RELATION_TABLE, ('lblannot_rowid',), alrid_list
    )
    return lblannot_rowids_list


@register_ibs_method
@getter_1to1
def get_alrid_from_superkey(ibs, aid_list, lblannot_rowid_list):
    """
    Args:
        aid_list (list): list of annotation row-ids
        lblannot_rowid_list (list): list of lblannot row-ids
    Returns:
        alrid_list (list): annot-label relationship id list
    """
    colnames = ('annot_rowid',)
    params_iter = zip(aid_list, lblannot_rowid_list)
    where_clause = 'annot_rowid=? AND lblannot_rowid=?'
    alrid_list = ibs.db.get_where(
        const.AL_RELATION_TABLE, colnames, params_iter, where_clause
    )
    return alrid_list


@register_ibs_method
@getter_1to1
def get_annot_lblannot_value_of_lbltype(ibs, aid_list, _lbltype, lblannot_value_getter):
    """
    Returns:
        lblannot_value_list (list): a list of strings ['fred', 'sue', ...] for each chip
        identifying the animal
    """
    lbltype_dict_list = ibs.get_annot_lblannot_rowids_oftype(aid_list, _lbltype)
    DEFAULT_VALUE = const.KEY_DEFAULTS[_lbltype]
    # FIXME: Use filters and unflat maps
    lblannot_value_list = [
        lblannot_value_getter(lblannot_rowids)[0]
        if len(lblannot_rowids) > 0
        else DEFAULT_VALUE
        for lblannot_rowids in lbltype_dict_list
    ]
    return lblannot_value_list


@register_ibs_method
@getter_1to1
def get_lblannot_lbltypes_rowids(ibs, lblannot_rowid_list):
    lbltype_rowid_list = ibs.db.get(
        const.LBLANNOT_TABLE, ('lbltype_rowid',), lblannot_rowid_list
    )
    return lbltype_rowid_list


@register_ibs_method
@getter_1to1
def get_lblannot_notes(ibs, lblannot_rowid_list):
    lblannotnotes_list = ibs.db.get(
        const.LBLANNOT_TABLE, ('lblannot_note',), lblannot_rowid_list
    )
    return lblannotnotes_list


@register_ibs_method
@getter_1to1
def get_lblannot_rowid_from_uuid(ibs, lblannot_uuid_list):
    """
    UNSAFE

    Returns:
        lblannot_rowid_list from the superkey (lbltype, value)
    """
    raise AssertionError(
        'CALL TO get_lblannot_rowid_from_uuid IS UNSAFE.  USE get_lblannot_rowid_from_superkey'
    )
    colnames = ('lblannot_rowid',)
    params_iter = lblannot_uuid_list
    id_colname = 'lblannot_uuid'
    lblannot_rowid_list = ibs.db.get(
        const.LBLANNOT_TABLE, colnames, params_iter, id_colname=id_colname
    )
    return lblannot_rowid_list


@register_ibs_method
@getter_1to1
def get_lblannot_uuids(ibs, lblannot_rowid_list):
    lblannotuuid_list = ibs.db.get(
        const.LBLANNOT_TABLE, ('lblannot_uuid',), lblannot_rowid_list
    )
    return lblannotuuid_list


@register_ibs_method
@getter_1to1
def get_lblannot_values(ibs, lblannot_rowid_list, _lbltype=None):
    """
    Returns:
        text lblannots
    """
    # TODO: Remove keyword argument
    # ibsfuncs.assert_lblannot_rowids_are_type(ibs, lblannot_rowid_list,  ibs.lbltype_ids[_lbltype])
    lblannot_value_list = ibs.db.get(
        const.LBLANNOT_TABLE, (LBLANNOT_VALUE,), lblannot_rowid_list
    )
    return lblannot_value_list


@register_ibs_method
@setter
def set_alr_confidence(ibs, alrid_list, confidence_list):
    """ sets annotation-lblannot-relationship confidence """
    id_iter = ((alrid,) for alrid in alrid_list)
    val_iter = ((confidence,) for confidence in confidence_list)
    colnames = ('alr_confidence',)
    ibs.db.set(const.AL_RELATION_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@setter
def set_alr_lblannot_rowids(ibs, alrid_list, lblannot_rowid_list):
    """
    Associates whatever annotation is at row(alrid) with a new
    lblannot_rowid. (effectively changes the label value of the rowid)
    """
    id_iter = ((alrid,) for alrid in alrid_list)
    val_iter = ((lblannot_rowid,) for lblannot_rowid in lblannot_rowid_list)
    colnames = ('lblannot_rowid',)
    ibs.db.set(const.AL_RELATION_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@setter
def set_annot_lblannot_from_rowid(ibs, aid_list, lblannot_rowid_list, _lbltype):
    """ Sets items/lblannot_rowids of a list of annotations."""
    # Get the alrids_list for the aids, using the lbltype as a filter
    alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
    # Find the aids which already have relationships (of _lbltype)
    setflag_list = [len(alrids) > 0 for alrids in alrids_list]
    # Add the relationship if it doesn't exist
    aid_list_to_add = ut.get_dirty_items(aid_list, setflag_list)
    lblannot_rowid_list_to_add = ut.get_dirty_items(lblannot_rowid_list, setflag_list)
    # set the existing relationship if one already exists
    alrids_list_to_set = ut.compress(alrids_list, setflag_list)
    lblannot_rowid_list_to_set = ut.compress(lblannot_rowid_list, setflag_list)
    # Assert each annot has only one relationship of this type
    ibsfuncs.assert_singleton_relationship(ibs, alrids_list_to_set)
    alrid_list_to_set = ut.flatten(alrids_list_to_set)
    # Add the new relationships
    ibs.add_annot_relationship(aid_list_to_add, lblannot_rowid_list_to_add)
    # Set the old relationships
    ibs.set_alr_lblannot_rowids(alrid_list_to_set, lblannot_rowid_list_to_set)


@register_ibs_method
@setter
def set_annot_lblannot_from_value(ibs, aid_list, value_list, _lbltype, ensure=True):
    """
    Associates the annot and lblannot of a specific type and value
    Adds the lblannot if it doesnt exist.
    Wrapper around convenience function for set_annot_from_lblannot_rowid
    """
    assert value_list is not None
    assert _lbltype is not None
    if ensure:
        pass
    # a value consisting of an empty string or all spaces is set to the default
    DEFAULT_VALUE = const.KEY_DEFAULTS[_lbltype]
    EMPTY_KEY = const.EMPTY_KEY
    # setting a name to DEFAULT_VALUE or EMPTY is equivalent to unnaming it
    value_list_ = [
        DEFAULT_VALUE if value.strip() == EMPTY_KEY else value for value in value_list
    ]
    notdefault_list = [value != DEFAULT_VALUE for value in value_list_]
    aid_list_to_delete = ut.get_dirty_items(aid_list, notdefault_list)
    # Set all the valid valids
    aids_to_set = ut.compress(aid_list, notdefault_list)
    values_to_set = ut.compress(value_list_, notdefault_list)
    ibs.delete_annot_relations_oftype(aid_list_to_delete, _lbltype)
    # remove the relationships that have now been unnamed
    # Convert names into lblannot_rowid
    # FIXME: This function should not be able to set label realationships
    # to labels that have not been added!!
    # This is an inefficient way of getting lblannot_rowids!
    lbltype_rowid_list = [ibs.lbltype_ids[_lbltype]] * len(values_to_set)
    # auto ensure
    lblannot_rowid_list = ibs.add_lblannots(lbltype_rowid_list, values_to_set)
    # Call set_annot_from_lblannot_rowid to finish the conditional adding
    ibs.set_annot_lblannot_from_rowid(aids_to_set, lblannot_rowid_list, _lbltype)


@register_ibs_method
def set_lblannot_notes(ibs, lblannot_rowid_list, value_list):
    """
    Updates the value for lblannots. Note this change applies to
    all annotations related to this lblannot_rowid
    """
    id_iter = ((rowid,) for rowid in lblannot_rowid_list)
    val_list = ((value,) for value in value_list)
    ibs.db.set(const.LBLANNOT_TABLE, ('lblannot_note',), val_list, id_iter)


@register_ibs_method
def set_lblannot_values(ibs, lblannot_rowid_list, value_list):
    """
    Updates the value for lblannots. Note this change applies to
    all annotations related to this lblannot_rowid
    """
    id_iter = ((rowid,) for rowid in lblannot_rowid_list)
    val_list = ((value,) for value in value_list)
    ibs.db.set(const.LBLANNOT_TABLE, (LBLANNOT_VALUE,), val_list, id_iter)


@register_ibs_method
@getter_1toM
def get_annot_alrids(ibs, aid_list):
    """ FIXME: __name__
    Get all the relationship ids belonging to the input annotations
    if lblannot lbltype is specified the relationship ids are filtered to
    be only of a specific lbltype/category/type
    """
    alrids_list = ibs.db.get(
        const.AL_RELATION_TABLE,
        ('alr_rowid',),
        aid_list,
        id_colname='annot_rowid',
        unpack_scalars=False,
    )
    return alrids_list


@register_ibs_method
@getter_1toM
def get_annot_alrids_oftype(ibs, aid_list, lbltype_rowid):
    """
    Get all the relationship ids belonging to the input annotations where the
    relationship ids are filtered to be only of a specific lbltype/category/type
    """
    alrids_list = ibs.get_annot_alrids(aid_list)
    # Get lblannot_rowid of each relationship
    lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    # Get the type of each lblannot
    lbltype_rowids_list = ibsfuncs.unflat_map(
        ibs.get_lblannot_lbltypes_rowids, lblannot_rowids_list
    )
    # only want the nids of individuals, not species, for example
    valids_list = [
        [typeid == lbltype_rowid for typeid in rowids] for rowids in lbltype_rowids_list
    ]
    alrids_list = [
        ut.compress(alrids, valids) for alrids, valids in zip(alrids_list, valids_list)
    ]

    def resolution_func_first(alrid_list):
        return [alrid_list[0]]

    alrids_list = [
        resolution_func_first(alrid_list) if len(alrid_list) > 1 else alrid_list
        for alrid_list in alrids_list
    ]
    assert all([len(alrid_list) < 2 for alrid_list in alrids_list]), (
        'More than one type per lbltype.  ALRIDS: '
        + str(alrids_list)
        + ', ROW: '
        + str(lbltype_rowid)
        + ', KEYS:'
        + str(ibs.lbltype_ids)
    )
    return alrids_list


@register_ibs_method
@getter_1toM
def get_annot_lblannot_rowids(ibs, aid_list):
    """
    Returns:
        list_ (list): the name id of each annotation. """
    # Get all the annotation lblannot relationships
    # filter out only the ones which specify names
    alrids_list = ibs.get_annot_alrids(aid_list)
    lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    return lblannot_rowids_list


@register_ibs_method
@getter_1toM
def get_annot_lblannot_rowids_oftype(ibs, aid_list, _lbltype=None):
    """
    Returns:
        list_ (list): the name id of each annotation. """
    # Get all the annotation lblannot relationships
    # filter out only the ones which specify names
    assert _lbltype is not None, 'should be using lbltype_rowids anyway'
    alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
    lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    return lblannot_rowids_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_lblannot_funcs
        python -m wbia.control.manual_lblannot_funcs --allexamples
        python -m wbia.control.manual_lblannot_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
