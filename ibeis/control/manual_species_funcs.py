"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"
sh Tgen.sh --key species --invert --Tcfg with_getters=True with_setters=False --modfname manual_species_funcs

"""
from __future__ import absolute_import, division, print_function
# TODO: Fix this name it is too special case
import uuid
import functools
import six  # NOQA
#from six.moves import range
from ibeis import constants as const
from ibeis import ibsfuncs
#import numpy as np
#import vtool as vt
from ibeis.control import accessor_decors, controller_inject  # NOQA
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_species]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)

SPECIES_ROWID   = 'species_rowid'
SPECIES_UUID    = 'species_uuid'
SPECIES_TEXT    = 'species_text'
SPECIES_NOTE    = 'species_note'


@register_ibs_method
@accessor_decors.ider
def _get_all_species_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    #all_known_species_rowids = ibs._get_all_known_lblannot_rowids(const.SPECIES_KEY)
    all_known_species_rowids = ibs.db.get_all_rowids(const.SPECIES_TABLE)
    return all_known_species_rowids


@register_ibs_method
#@register_api('/api/species/sanitize', methods=['PUT'])
def sanitize_species_texts(ibs, species_text_list):
    r"""
    changes unknown species to the unknown value

    Args:
        ibs (IBEISController):  ibeis controller object
        species_text_list (list):

    Returns:
        list: species_text_list_

    CommandLine:
        python -m ibeis.control.manual_species_funcs --test-sanitize_species_texts

    RESTful:
        Method: POST
        URL:    /api/species/sanitize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_species_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> species_text_list = ['foo', 'bar', 'zebra_plains']
        >>> # execute function
        >>> species_text_list_ = sanitize_species_texts(ibs, species_text_list)
        >>> # verify results
        >>> result = str(species_text_list_)
        >>> print(result)
        ['____', '____', 'zebra_plains']
    """
    ibsfuncs.assert_valid_species_texts(ibs, species_text_list, iswarning=True)
    def _sanitize_species_text(species_text):
        if species_text is None:
            return None
        elif species_text in const.VALID_SPECIES:
            return species_text
        else:
            return const.UNKNOWN
    species_text_list_ = [_sanitize_species_text(species_text)
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
@accessor_decors.adder
@register_api('/api/species/', methods=['POST'])
def add_species(ibs, species_text_list, species_uuid_list=None, note_list=None):
    r"""
    Adds a list of species.

    Returns:
        list: speciesid_list - species rowids

    RESTful:
        Method: POST
        URL:    /api/species/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_species_funcs import *  # NOQA
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
        [u'jaguar', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']
        [1, 2, 3]
        [u'zebra_plains', u'zebra_grevys', u'bear_polar']

    [u'jaguar', u'zebra_plains', u'zebra_plains', '____', '____', '____', u'zebra_grevys', u'bear_polar']
    [8, 9, 10]
    [u'zebra_plains', u'zebra_grevys', u'bear_polar']

    """
    if note_list is None:
        note_list = [''] * len(species_text_list)
    # Sanatize to remove ____
    species_text_list_ = ibs.sanitize_species_texts(species_text_list)
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
    #value_list = ibs.sanitize_species_texts(species_text_list)
    #lbltype_rowid = ibs.lbltype_ids[const.SPECIES_KEY]
    #lbltype_rowid_list = [lbltype_rowid] * len(species_text_list)
    #species_rowid_list = ibs.add_lblannots(lbltype_rowid_list, value_list, note_list)
    ##species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID if rowid is None else
    ##                      rowid for rowid in species_rowid_list]
    #return species_rowid_list


@register_ibs_method
@accessor_decors.deleter
#@cache_invalidator(const.SPECIES_TABLE)
@register_api('/api/species/', methods=['DELETE'])
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
    #ibs.delete_lblannots(species_rowid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/rowids_from_text/', methods=['GET'])
def get_species_rowids_from_text(ibs, species_text_list, ensure=True):
    r"""
    Returns:
        species_rowid_list (list): Creates one if it doesnt exist

    CommandLine:
        python -m ibeis.control.manual_species_funcs --test-get_species_rowids_from_text:0
        python -m ibeis.control.manual_species_funcs --test-get_species_rowids_from_text:1

    RESTful:
        Method: GET
        URL:    /api/species/rowids_from_text/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_species_funcs import *  # NOQA
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
        >>> from ibeis.control.manual_species_funcs import *  # NOQA
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
        species_text_list_ = ibs.sanitize_species_texts(species_text_list)
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
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.SPECIES_TABLE, SPECIES_TEXT)
@register_api('/api/species/texts/', methods=['GET'])
def get_species_texts(ibs, species_rowid_list):
    r"""
    Returns:
        list: species_text_list text names

    CommandLine:
        python -m ibeis.control.manual_species_funcs --test-get_species_texts --enableall

    RESTful:
        Method: GET
        URL:    /api/species/texts/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_species_funcs import *  # NOQA
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
@accessor_decors.getter_1to1
@register_api('/api/species/uuids/', methods=['GET'])
def get_species_uuids(ibs, species_rowid_list):
    r"""
    Returns:
        list_ (list): uuids_list - species uuids

    RESTful:
        Method: GET
        URL:    /api/species/uuids/
    """
    uuids_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_UUID,), species_rowid_list)
    #notes_list = ibs.get_lblannot_notes(nid_list)
    return uuids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/species/notes/', methods=['GET'])
def get_species_notes(ibs, species_rowid_list):
    r"""
    Returns:
        list_ (list): notes_list - species notes

    RESTful:
        Method: GET
        URL:    /api/species/notes/
    """
    notes_list = ibs.db.get(const.SPECIES_TABLE, (SPECIES_NOTE,), species_rowid_list)
    #notes_list = ibs.get_lblannot_notes(nid_list)
    return notes_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.control.manual_species_funcs
        python -m ibeis.control.manual_species_funcs --allexamples
        python -m ibeis.control.manual_species_funcs --allexamples --noface --nosrc

    RESTful:
        Method: GET
        URL:    /api/species/notes/
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
