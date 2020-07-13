# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'wbia.templates.template_generator')"
sh Tgen.sh --key species --invert --Tcfg with_getters=True with_setters=False --modfname manual_species_funcs

# TODO: Fix this name it is too special case
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import uuid
import functools
import six  # NOQA
from six.moves import range, zip, map  # NOQA

# import numpy as np
# import vtool as vt
import numpy as np
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject  # NOQA
import utool as ut
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


SPECIES_ROWID = 'species_rowid'
SPECIES_UUID = 'species_uuid'
SPECIES_TEXT = 'species_text'
SPECIES_NICE = 'species_nice'
SPECIES_CODE = 'species_code'
SPECIES_NOTE = 'species_note'
SPECIES_ENABLED = 'species_toggle_enabled'


@register_ibs_method
@accessor_decors.ider
@register_api('/api/species/', methods=['GET'], __api_plural_check__=False)
def _get_all_species_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    # all_known_species_rowids = ibs._get_all_known_lblannot_rowids(const.SPECIES_KEY)
    all_known_species_rowids = ibs.db.get_all_rowids(const.SPECIES_TABLE)
    return all_known_species_rowids


@register_ibs_method
@accessor_decors.ider
def get_all_species_texts(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    species_rowid_list = ibs._get_all_species_rowids()
    species_text_list = ibs.get_species_texts(species_rowid_list)
    return species_text_list


@register_ibs_method
@accessor_decors.ider
def get_all_species_nice(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    species_rowid_list = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowid_list)
    return species_nice_list


@register_ibs_method
# @register_api('/api/species/sanitize/', methods=['PUT'])
def sanitize_species_texts(ibs, species_text_list):
    r"""
    changes unknown species to the unknown value

    Args:
        ibs (IBEISController):  wbia controller object
        species_text_list (list):

    Returns:
        list: species_text_list_

    CommandLine:
        python -m wbia.control.manual_species_funcs --test-sanitize_species_texts

    RESTful:
        Method: POST
        URL:    /api/species/sanitize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_species_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> species_text_list = ['foo', 'bar', 'zebra_plains']
        >>> # execute function
        >>> species_text_list_ = sanitize_species_texts(ibs, species_text_list)
        >>> # verify results
        >>> result = ut.repr2(species_text_list_, nl=False)
        >>> print(result)
        ['foo', 'bar', 'zebra_plains']
    """
    # valid_species = ibs.get_all_species_texts()
    # ibsfuncs.assert_valid_species_texts(ibs, species_text_list, iswarning=True)
    # def _sanitize_species_text(species_text):
    #     if species_text is None:
    #         return None
    #     elif species_text in valid_species:
    #         return species_text
    #     else:
    #         return const.UNKNOWN
    # species_text_list_ = [_sanitize_species_text(species_text)
    #                       for species_text in species_text_list]
    # # old but same logic
    # #species_text_list_ = [None if species_text is None else
    # #                      species_text if species_text in valid_species else
    # #                      const.UNKNOWN
    # #                      for species_text in species_text_list]
    # # oldest different logic
    # #species_text_list_ = [None
    # #                      if species_text is None or species_text == const.UNKNOWN
    # #                      else species_text.lower()
    # #                      for species_text in species_text_list]
    # #species_text_list_ = [species_text if species_text in valid_species else None
    # #                      for species_text in species_text_list_]
    # return species_text_list_
    return species_text_list


def _convert_species_nice_to_text(species_nice_list):
    import re

    def _convert(nice):
        nice = re.sub(r'[ ]+', '_', nice)
        nice = re.sub(r'[^a-zA-Z0-9_\+]+', '', nice)
        nice = re.sub(r'[_]+', '_', nice)
        nice = nice.lower()
        return nice

    return [_convert(species_nice) for species_nice in species_nice_list]


def _convert_species_nice_to_code(species_nice_list):
    import re

    def _convert(text):
        text = re.sub(r'[_]+', ' ', text)
        text = text.title()
        text = re.sub(r'[^A-Z0-9\+]+', '', text)
        return text

    species_text_list = _convert_species_nice_to_text(species_nice_list)
    return [_convert(species_text) for species_text in species_text_list]


@register_ibs_method
@accessor_decors.adder
@register_api('/api/species/', methods=['POST'], __api_plural_check__=False)
def add_species(
    ibs,
    species_nice_list,
    species_text_list=None,
    species_code_list=None,
    species_uuid_list=None,
    species_note_list=None,
    skip_cleaning=False,
):
    r"""
    Adds a list of species.

    Returns:
        list: speciesid_list - species rowids

    RESTful:
        Method: POST
        URL:    /api/species/

    CommandLine:
        python -m wbia.control.manual_species_funcs --test-add_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_species_funcs import *  # NOQA
        >>> import wbia
        >>> import utool as ut
        >>> ibs = wbia.opendb('testdb1')
        >>> species_text_list = [
        ...     'jaguar', 'zebra_plains', 'zebra_plains', '____', 'TYPO',
        ...     '____', 'zebra_grevys', 'bear_polar+head']
        >>> species_rowid_list = ibs.add_species(species_text_list)
        >>> print(ut.repr2(list(zip(species_text_list, species_rowid_list))))
        >>> ibs.print_species_table()
        >>> species_text = ibs.get_species_texts(species_rowid_list)
        >>> # Ensure we leave testdb1 in a clean state
        >>> ibs.delete_species(ibs.get_species_rowids_from_text(['jaguar', 'TYPO']))
        >>> all_species_rowids = ibs._get_all_species_rowids()
        >>> result =  ut.repr2(species_text, nl=False) + '\n'
        >>> result += ut.repr2(all_species_rowids, nl=False) + '\n'
        >>> result += ut.repr2(ibs.get_species_texts(all_species_rowids), nl=False) + '\n'
        >>> result += ut.repr2(ibs.get_species_codes(all_species_rowids), nl=False)
        >>> print(result)
        ['jaguar', 'zebra_plains', 'zebra_plains', '____', 'typo', '____', 'zebra_grevys', 'bear_polar+head']
        [1, 2, 3, 6]
        ['zebra_plains', 'zebra_grevys', 'bear_polar', 'bear_polar+head']
        ['PZ', 'GZ', 'PB', 'BP+H']
    """
    # Strip all spaces
    species_nice_list = [
        const.UNKNOWN if _ is None else _.strip() for _ in species_nice_list
    ]

    if species_text_list is None:
        species_text_list = _convert_species_nice_to_text(species_nice_list)
    if species_code_list is None:
        species_code_list = _convert_species_nice_to_code(species_nice_list)
    if species_note_list is None:
        species_note_list = [''] * len(species_text_list)
    if species_uuid_list is None:
        species_uuid_list = [uuid.uuid4() for _ in range(len(species_text_list))]

    # Sanatize to remove invalid names
    flag_list = np.array(
        [
            species_nice is None
            or species_nice.strip() in ['_', const.UNKNOWN, 'none', 'None', '']
            for species_nice in species_nice_list
        ]
    )
    species_uuid_list = ut.filterfalse_items(species_uuid_list, flag_list)
    species_nice_list = ut.filterfalse_items(species_nice_list, flag_list)
    species_text_list = ut.filterfalse_items(species_text_list, flag_list)
    species_code_list = ut.filterfalse_items(species_code_list, flag_list)
    species_note_list = ut.filterfalse_items(species_note_list, flag_list)

    superkey_paramx = (1,)
    # TODO Allow for better ensure=False without using partial
    # Just autogenerate these functions
    get_rowid_from_superkey = functools.partial(
        ibs.get_species_rowids_from_text, ensure=False
    )
    colnames = [SPECIES_UUID, SPECIES_TEXT, SPECIES_NICE, SPECIES_CODE, SPECIES_NOTE]
    params_iter = list(
        zip(
            species_uuid_list,
            species_text_list,
            species_nice_list,
            species_code_list,
            species_note_list,
        )
    )
    species_rowid_list = ibs.db.add_cleanly(
        const.SPECIES_TABLE,
        colnames,
        params_iter,
        get_rowid_from_superkey,
        superkey_paramx,
    )
    temp_list = np.array([-1] * len(flag_list))
    temp_list[flag_list == False] = np.array(species_rowid_list)  # NOQA
    temp_list[flag_list == True] = const.UNKNOWN_SPECIES_ROWID  # NOQA
    species_rowid_list = list(temp_list)
    assert -1 not in species_rowid_list

    # Clean species
    if not skip_cleaning:
        species_mapping_dict = ibs._clean_species()
        if species_mapping_dict is not None:
            species_rowid_list = [
                species_mapping_dict.get(species_rowid, species_rowid)
                for species_rowid in species_rowid_list
            ]

    return species_rowid_list
    # value_list = ibs.sanitize_species_texts(species_text_list)
    # lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
    # lbltype_rowid_list = [lbltype_rowid] * len(species_text_list)
    # species_rowid_list = ibs.add_lblannots(lbltype_rowid_list, value_list, species_note_list)
    # # species_rowid_list = [const.UNKNOWN_SPECIES_ROWID if rowid is None else
    #                      rowid for rowid in species_rowid_list]
    # return species_rowid_list


@register_ibs_method
@accessor_decors.deleter
# @cache_invalidator(const.SPECIES_TABLE)
@register_api('/api/species/', methods=['DELETE'], __api_plural_check__=False)
def delete_species(ibs, species_rowid_list):
    r"""
    deletes species from the database

    CAREFUL. YOU PROBABLY DO NOT WANT TO USE THIS
    at least ensure that no annot is associated with any of these species rowids

    RESTful:
        Method: DELETE
        URL:    /api/species/
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d speciess' % len(species_rowid_list))
    ibs.db.delete_rowids(const.SPECIES_TABLE, species_rowid_list)
    # ibs.delete_lblannots(species_rowid_list)


@register_ibs_method
# @accessor_decors.deleter
def delete_empty_species(ibs):
    r"""
    deletes empty species from the database
    """
    species_text_set = set(ibs.get_all_species_texts())
    aid_list = ibs.get_valid_aids()
    used_species_text_set = set(ibs.get_annot_species_texts(aid_list))
    unused_species_text_set = species_text_set - used_species_text_set
    unused_species_text_list = list(unused_species_text_set)
    unused_species_rowid_list = ibs.get_species_rowids_from_text(unused_species_text_list)
    print('Deleting unused species: %r' % (unused_species_text_list,))
    ibs.delete_species(unused_species_rowid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/rowid/text/', methods=['GET'], __api_plural_check__=False)
def get_species_rowids_from_text(ibs, species_text_list, ensure=True, **kwargs):
    r"""
    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m wbia.control.manual_species_funcs --test-get_species_rowids_from_text:0
        python -m wbia.control.manual_species_funcs --test-get_species_rowids_from_text:1

    RESTful:
        Method: GET
        URL:    /api/species/rowid/text/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_species_funcs import *  # NOQA
        >>> import wbia
        >>> import utool as ut
        >>> ibs = wbia.opendb('testdb1')
        >>> species_text_list = [
        ...     u'jaguar', u'zebra_plains', u'zebra_plains', '____', 'TYPO',
        ...     '____', u'zebra_grevys', u'bear_polar']
        >>> ensure = False
        >>> species_rowid_list = ibs.get_species_rowids_from_text(species_text_list, ensure)
        >>> print(ut.repr2(list(zip(species_text_list, species_rowid_list))))
        >>> ensure = True
        >>> species_rowid_list = ibs.get_species_rowids_from_text(species_text_list, ensure)
        >>> print(ut.repr2(list(zip(species_text_list, species_rowid_list))))
        >>> ibs.print_species_table()
        >>> species_text = ibs.get_species_texts(species_rowid_list)
        >>> # Ensure we leave testdb1 in a clean state
        >>> ibs.delete_species(ibs.get_species_rowids_from_text(['jaguar', 'TYPO']))
        >>> all_species_rowids = ibs._get_all_species_rowids()
        >>> result = ut.repr2(species_text, nl=False) + '\n'
        >>> result += ut.repr2(all_species_rowids, nl=False) + '\n'
        >>> result += ut.repr2(ibs.get_species_texts(all_species_rowids), nl=False)
        >>> print(result)
        ['jaguar', 'zebra_plains', 'zebra_plains', '____', 'typo', '____', 'zebra_grevys', 'bear_polar']
        [1, 2, 3, 6]
        ['zebra_plains', 'zebra_grevys', 'bear_polar', 'bear_polar+head']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_species_funcs import *  # NOQA
        >>> import wbia
        >>> import utool as ut  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> species_text_list = [
        ...     u'jaguar', u'zebra_plains', u'zebra_plains', '____', 'TYPO',
        ...     '____', u'zebra_grevys', u'bear_polar']
        >>> ensure = False
        >>> species_rowid_list = ibs.get_species_rowids_from_text(species_text_list, ensure)

    """
    if ensure:
        species_rowid_list = ibs.add_species(species_text_list, **kwargs)
    else:
        species_text_list_ = ibs.sanitize_species_texts(species_text_list)
        # lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
        # lbltype_rowid_list = [lbltype_rowid] * len(species_text_list_)
        # species_rowid_list = ibs.get_lblannot_rowid_from_superkey(lbltype_rowid_list, species_text_list_)
        # Ugg species and names need their own table
        # species_rowid_list = [const.UNKNOWN_SPECIES_ROWID if rowid is None else
        #                      rowid for rowid in species_rowid_list]
        species_rowid_list = ibs.db.get(
            const.SPECIES_TABLE,
            (SPECIES_ROWID,),
            species_text_list_,
            id_colname=SPECIES_TEXT,
        )
        # BIG HACK FOR ENFORCING UNKNOWN SPECIESS HAVE ROWID 0
        species_rowid_list = [
            const.UNKNOWN_SPECIES_ROWID
            if text is None or text == const.UNKNOWN
            else rowid
            for rowid, text in zip(species_rowid_list, species_text_list_)
        ]
    return species_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/rowid/uuid/', methods=['GET'], __api_plural_check__=False)
def get_species_rowids_from_uuids(ibs, species_uuid_list):
    r"""
    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m wbia.control.manual_species_funcs --test-get_species_rowids_from_text:0
        python -m wbia.control.manual_species_funcs --test-get_species_rowids_from_text:1

    RESTful:
        Method: GET
        URL:    /api/species/rowid/uuid/
    """
    species_rowid_list = ibs.db.get(
        const.SPECIES_TABLE, (SPECIES_ROWID,), species_uuid_list, id_colname=SPECIES_UUID
    )
    species_rowid_list = [
        const.UNKNOWN_SPECIES_ROWID if text is None or text == const.UNKNOWN else rowid
        for rowid, text in zip(species_rowid_list, species_uuid_list)
    ]
    return species_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/uuid/', methods=['GET'], __api_plural_check__=False)
def get_species_uuids(ibs, species_rowid_list):
    r"""
    Returns:
        list_ (list): uuids_list - species uuids

    RESTful:
        Method: GET
        URL:    /api/species/uuid/
    """
    uuids_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_UUID,), species_rowid_list)
    # notes_list = ibs.get_lblannot_notes(nid_list)
    return uuids_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.SPECIES_TABLE, SPECIES_TEXT)
@register_api('/api/species/text/', methods=['GET'], __api_plural_check__=False)
def get_species_texts(ibs, species_rowid_list):
    r"""
    Returns:
        list: species_text_list text names

    CommandLine:
        python -m wbia.control.manual_species_funcs --test-get_species_texts --enableall

    RESTful:
        Method: GET
        URL:    /api/species/text/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_species_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> species_rowid_list = ibs._get_all_species_rowids()
        >>> result = get_species_texts(ibs, species_rowid_list)
        >>> result = ut.repr2(result)
        >>> print(result)
        ['zebra_plains', 'zebra_grevys', 'bear_polar', 'bear_polar+head']
    """
    # FIXME: use standalone species table
    # species_text_list = ibs.get_lblannot_values(species_rowid_list, const.SPECIES_KEY)
    species_text_list = ibs.db.get(
        const.SPECIES_TABLE, (SPECIES_TEXT,), species_rowid_list
    )
    species_text_list = [
        const.UNKNOWN if rowid == const.UNKNOWN_SPECIES_ROWID else species_text
        for species_text, rowid in zip(species_text_list, species_rowid_list)
    ]
    species_text_list = [
        const.UNKNOWN if code is None else code for code in species_text_list
    ]
    return species_text_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.SPECIES_TABLE, SPECIES_NICE)
@register_api('/api/species/nice/', methods=['GET'], __api_plural_check__=False)
def get_species_nice(ibs, species_rowid_list):
    r"""
    Returns:
        list: species_text_list nice names

    CommandLine:
        python -m wbia.control.manual_species_funcs --test-get_species_nice --enableall

    RESTful:
        Method: GET
        URL:    /api/species/nice/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_species_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs._clean_species()
        >>> species_rowid_list = ibs._get_all_species_rowids()
        >>> result = get_species_nice(ibs, species_rowid_list)
        >>> result = ut.repr2(result)
        >>> print(result)
        ['Zebra (Plains)', "Zebra (Grevy's)", 'Polar Bear', 'bear_polar+head']
    """
    # FIXME: use standalone species table
    # species_nice_list = ibs.get_lblannot_values(species_rowid_list, const.SPECIES_KEY)
    species_nice_list = ibs.db.get(
        const.SPECIES_TABLE, (SPECIES_NICE,), species_rowid_list
    )
    species_nice_list = [
        const.UNKNOWN if rowid == const.UNKNOWN_SPECIES_ROWID else species_nice
        for species_nice, rowid in zip(species_nice_list, species_rowid_list)
    ]
    species_nice_list = [
        'Unknown' if code is None else code for code in species_nice_list
    ]
    return species_nice_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/code/', methods=['GET'], __api_plural_check__=False)
def get_species_codes(ibs, species_rowid_list):
    r"""
    Returns:
        list_ (list): code_list - species codes

    RESTful:
        Method: GET
        URL:    /api/species/code/
    """
    species_code_list = ibs.db.get(
        const.SPECIES_TABLE, (SPECIES_CODE,), species_rowid_list
    )
    species_code_list = [
        'UNKNOWN' if code is None else code for code in species_code_list
    ]
    return species_code_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/note/', methods=['GET'], __api_plural_check__=False)
def get_species_notes(ibs, species_rowid_list):
    r"""
    Returns:
        list_ (list): notes_list - species notes

    RESTful:
        Method: GET
        URL:    /api/species/note/
    """
    notes_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_NOTE,), species_rowid_list)
    # notes_list = ibs.get_lblannot_notes(nid_list)
    return notes_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/species/enabled/', methods=['GET'])
def get_species_enabled(ibs, species_rowid_list):
    r"""
    Returns:
        list_ (list): "Species Enabled" flag, true if the species is enabled
    """
    enabled_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_ENABLED,), species_rowid_list)
    return enabled_list


@register_ibs_method
@accessor_decors.setter
def _set_species_texts(ibs, species_rowid_list, species_text_list):
    r"""
    Sets the species nice names
    """
    id_iter = ((species_rowid,) for species_rowid in species_rowid_list)
    val_list = ((enabled,) for enabled in species_text_list)
    ibs.db.set(const.SPECIES_TABLE, (SPECIES_TEXT,), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
def _set_species_nice(ibs, species_rowid_list, species_nice_list):
    r"""
    Sets the species nice names
    """
    id_iter = ((species_rowid,) for species_rowid in species_rowid_list)
    val_list = ((enabled,) for enabled in species_nice_list)
    ibs.db.set(const.SPECIES_TABLE, (SPECIES_NICE,), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
def _set_species_code(ibs, species_rowid_list, species_code_list):
    r"""
    Sets the species nice names
    """
    id_iter = ((species_rowid,) for species_rowid in species_rowid_list)
    val_list = ((enabled,) for enabled in species_code_list)
    ibs.db.set(const.SPECIES_TABLE, (SPECIES_CODE,), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/species/enabled/', methods=['PUT'])
def set_species_enabled(ibs, species_rowid_list, enabled_list):
    r"""
    Sets the species all instances enabled bit
    """
    id_iter = ((species_rowid,) for species_rowid in species_rowid_list)
    val_list = ((enabled,) for enabled in enabled_list)
    ibs.db.set(const.SPECIES_TABLE, (SPECIES_ENABLED,), val_list, id_iter)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_species_funcs
        python -m wbia.control.manual_species_funcs --allexamples
        python -m wbia.control.manual_species_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
