from __future__ import absolute_import, division, print_function
# TODO: Fix this name it is too special case
#import uuid
import six  # NOQA
#from six.moves import range
from ibeis import constants as const
from ibeis import ibsfuncs
from ibeis.control.accessor_decors import (adder, deleter, setter, getter_1to1,
                                           getter_1toM, ider)
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[autogen_name_species]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


ANNOT_ROWID         = 'annot_rowid'
NAME_ROWID          = 'name_rowid'
SPECIES_ROWID       = 'species_rowid'
LBLANNOT_VALUE = 'lblannot_value'


@register_ibs_method
@ider
def _get_all_known_name_rowids(ibs):
    """
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    all_known_nids = ibs._get_all_known_lblannot_rowids(const.INDIVIDUAL_KEY)
    return all_known_nids


@register_ibs_method
@ider
def _get_all_species_rowids(ibs):
    """
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    all_known_species_rowids = ibs._get_all_known_lblannot_rowids(const.SPECIES_KEY)
    return all_known_species_rowids


@register_ibs_method
@adder
def add_names(ibs, name_text_list, note_list=None):
    """
    Adds a list of names.

    Returns:
        nid_list (list): their nids
    """
    # nid_list_ = [namenid_dict[name] for name in name_list_]
    name_text_list_ = ibs.sanatize_name_texts(name_text_list)
    # All names are individuals and so may safely receive the INDIVIDUAL_KEY lblannot
    lbltype_rowid = ibs.lbltype_ids[const.INDIVIDUAL_KEY]
    lbltype_rowid_list = [lbltype_rowid] * len(name_text_list_)
    nid_list = ibs.add_lblannots(lbltype_rowid_list, name_text_list_, note_list)
    #nid_list = [ibs.UNKNOWN_NAME_ROWID if rowid is None else rowid for rowid in nid_list]
    return nid_list


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
    ibsfuncs.assert_valid_species(ibs, species_text_list, iswarning=True)
    species_text_list_ = [None
                          if species_text is None or species_text == const.UNKNOWN
                          else species_text.lower()
                          for species_text in species_text_list]
    species_text_list_ = [species_text if species_text in const.VALID_SPECIES else None
                          for species_text in species_text_list_]
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
def add_species(ibs, species_text_list, note_list=None):
    """
    Adds a list of species.

    Returns:
        list: speciesid_list - species rowids

    """
    value_list = ibs.sanatize_species_texts(species_text_list)
    lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
    lbltype_rowid_list = [lbltype_rowid] * len(species_text_list)
    species_rowid_list = ibs.add_lblannots(lbltype_rowid_list, value_list, note_list)
    #species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID if rowid is None else
    #                      rowid for rowid in species_rowid_list]
    return species_rowid_list


@register_ibs_method
@deleter
#@cache_invalidator(const.LBLANNOT_TABLE)
def delete_names(ibs, nid_list):
    """
    deletes names from the database

    CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
    at least ensure that no annot is associated with any of these nids
    """
    ibs.delete_lblannots(nid_list)


@register_ibs_method
@ider
def get_invalid_nids(ibs):
    """
    Returns:
        list: nid_list - all names without any animals (does not include unknown names)
    """
    _nid_list = ibs._get_all_known_name_rowids()
    nRois_list = ibs.get_name_num_annotations(_nid_list)
    nid_list = [nid for nid, nRois in zip(_nid_list, nRois_list)
                if nRois <= 0]
    return nid_list


@register_ibs_method
@getter_1toM
def get_name_aids(ibs, nid_list, enable_unknown_fix=False):
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
    """
    nid_list_ = [const.UNKNOWN_LBLANNOT_ROWID if nid <= 0 else nid for nid in nid_list]
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,),
                           nid_list_, id_colname=NAME_ROWID,
                           unpack_scalars=False)
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
    nid_list_ = [const.UNKNOWN_LBLANNOT_ROWID if nid <= 0 else nid for nid in nid_list]
    # Get all annot ids for each name
    aids_list = ibs.get_name_aids(nid_list_)
    # Flag any annots that are not exemplar and remove them
    flags_list = ibsfuncs.unflat_map(ibs.get_annot_exemplar_flag, aids_list)
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
    aids_list = ibs.get_name_aids(nid_list)
    gids_list = ibsfuncs.unflat_map(ibs.get_annot_gids, aids_list)
    return gids_list


@register_ibs_method
@getter_1to1
def get_name_notes(ibs, nid_list):
    """
    Returns:
        list_ (list): name notes
    """
    notes_list = ibs.get_lblannot_notes(nid_list)
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
    return list(map(len, ibs.get_name_aids(nid_list)))


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
def get_name_texts(ibs, nid_list):
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
        >>> nid_list = ibs._get_all_known_nids()
        >>> name_text_list = get_name_texts(ibs, nid_list)
        >>> result = str(name_text_list)
        >>> print(result)
        [u'easy', u'hard', u'jeff', u'lena', u'occl', u'polar', u'zebra']
    """
    # FIXME: Use standalone name table
    # TODO:
    # Change the temporary negative indexes back to the unknown NID for the
    # SQL query. Then augment the lblannot list to distinguish unknown lblannots
    name_text_list = ibs.get_lblannot_values(nid_list, const.INDIVIDUAL_KEY)
    name_text_list = [const.UNKNOWN
                      if rowid == ibs.UNKNOWN_NAME_ROWID or name_text is None
                      else name_text
                      for name_text, rowid in zip(name_text_list, nid_list)]
    return name_text_list


@register_ibs_method
def get_num_names(ibs, **kwargs):
    """
    Number of valid name (subset of lblannot)

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
    """
    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m ibeis.control.manual_name_species_funcs --test-get_species_rowids_from_text

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
        >>> ibs.print_lblannot_table()
        >>> species_text = ibs.get_species_texts(species_rowid_list)
        >>> result = species_text
        >>> print(result)
        [u'jaguar', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']

    """
    if ensure:
        species_rowid_list = ibs.add_species(species_text_list)
    else:
        species_text_list_ = ibs.sanatize_species_texts(species_text_list)
        lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
        lbltype_rowid_list = [lbltype_rowid] * len(species_text_list_)
        species_rowid_list = ibs.get_lblannot_rowid_from_superkey(lbltype_rowid_list, species_text_list_)
        # Ugg species and names need their own table
        species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID if rowid is None else
                              rowid for rowid in species_rowid_list]
    return species_rowid_list


@register_ibs_method
@getter_1to1
def get_name_rowids_from_text(ibs, name_text_list, ensure=True):
    """
    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_name_species_funcs import *  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> ibs = ibeis.opendb('testdb1')
        >>> name_text_list = [
        ...     u'Fred', u'Sue', '____',
        ...     u'zebra_grevys', 'TYPO', '____']
        >>> ensure = False
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> print(ut.list_str(list(zip(name_text_list, name_rowid_list))))
        >>> ensure = True
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_text_list, ensure)
        >>> print(ut.list_str(list(zip(name_text_list, name_rowid_list))))
        >>> ibs.print_lblannot_table()
        >>> result = name_rowid_list
        >>> print(result)
        [11, 12, 0, 13, 14, 0]
    """
    if ensure:
        name_rowid_list = ibs.add_names(name_text_list)
    else:
        name_text_list_ = ibs.sanatize_name_texts(name_text_list)
        lbltype_rowid = ibs.lbltype_ids[const.INDIVIDUAL_KEY]
        lbltype_rowid_list = [lbltype_rowid] * len(name_text_list_)
        name_rowid_list = ibs.get_lblannot_rowid_from_superkey(lbltype_rowid_list, name_text_list_)
        # Ugg species and names need their own table
        name_rowid_list = [ibs.UNKNOWN_NAME_ROWID if rowid is None else
                              rowid for rowid in name_rowid_list]
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
    species_text_list = ibs.get_lblannot_values(species_rowid_list, const.SPECIES_KEY)
    species_text_list = [const.UNKNOWN
                         if rowid == ibs.UNKNOWN_SPECIES_ROWID else species_text
                         for species_text, rowid in zip(species_text_list, species_rowid_list)]
    return species_text_list


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
    nRois_list = ibs.get_name_num_annotations(_nid_list)
    if filter_empty:
        nid_list = [nid for nid, nRois in zip(_nid_list, nRois_list)
                    if nRois > 0]
    else:
        nid_list = _nid_list
    return nid_list


@register_ibs_method
@setter
def set_name_notes(ibs, nid_list, notes_list):
    """ Sets a note for each name (multiple annotations) """
    ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    ibs.set_lblannot_notes(nid_list, notes_list)


@register_ibs_method
@setter
def set_name_texts(ibs, nid_list, name_list):
    """
    Changes the name text. Does not affect the animals of this name.
    Effectively an alias.

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
    ibsfuncs.assert_valid_names(name_list)
    #sanatize_name_texts(ibs, name_text_list):
    ibsfuncs.assert_lblannot_rowids_are_type(ibs, nid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    ibs.set_lblannot_values(nid_list, name_list)


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
