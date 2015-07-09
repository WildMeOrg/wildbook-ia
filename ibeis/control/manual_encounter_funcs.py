from __future__ import absolute_import, division, print_function
import six
from ibeis import constants as const
from ibeis.control import accessor_decors, controller_inject
from ibeis.control.controller_inject import make_ibs_register_decorator
import functools
import utool as ut
import uuid
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_encounter]')


ENCOUNTER_END_TIME_POSIX   = 'encounter_end_time_posix'
ENCOUNTER_GPS_LAT          = 'encounter_gps_lat'
ENCOUNTER_GPS_LON          = 'encounter_gps_lon'
ENCOUNTER_NOTE             = 'encounter_note'
ENCOUNTER_PROCESSED_FLAG   = 'encounter_processed_flag'
ENCOUNTER_ROWID            = 'encounter_rowid'
ENCOUNTER_SHIPPED_FLAG     = 'encounter_shipped_flag'
ENCOUNTER_START_TIME_POSIX = 'encounter_start_time_posix'
ENCOUNTER_SMART_WAYPOINT_ID = 'encounter_smart_waypoint_id'
ENCOUNTER_SMART_XML_FNAME   = 'encounter_smart_xml_fname'


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_ibs_method
@accessor_decors.ider
def _get_all_encounter_rowids(ibs):
    r"""
    Returns:
        list_ (list):  all unfiltered eids (encounter rowids)
    """
    all_eids = ibs.db.get_all_rowids(const.ENCOUNTER_TABLE)
    return all_eids


@register_ibs_method
@accessor_decors.ider
def _get_all_eids(ibs):
    r"""
    alias
    """
    return _get_all_encounter_rowids(ibs)


@register_ibs_method
@accessor_decors.ider
@register_api('/api/encounter/', methods=['GET'])
def get_valid_eids(ibs, min_num_gids=0, processed=None, shipped=None):
    r"""
    Returns:
        list_ (list):  list of all encounter ids

    RESTful:
        Method: GET
        URL:    /api/encounter/
    """
    eid_list = ibs._get_all_eids()
    if min_num_gids > 0:
        num_gids_list = ibs.get_encounter_num_gids(eid_list)
        flag_list = [num_gids >= min_num_gids for num_gids in num_gids_list]
        eid_list  = ut.filter_items(eid_list, flag_list)
    if processed is not None:
        flag_list = ibs.get_encounter_processed_flags(eid_list)
        isvalid_list = [ flag == 1 if processed else flag == 0 for flag in flag_list]
        eid_list  = ut.filter_items(eid_list, isvalid_list)
    if shipped is not None:
        flag_list = ibs.get_encounter_shipped_flags(eid_list)
        isvalid_list = [ flag == 1 if shipped else flag == 0 for flag in flag_list]
        eid_list  = ut.filter_items(eid_list, isvalid_list)

    return eid_list


@register_ibs_method
@accessor_decors.adder
@register_api('/api/encounter/', methods=['POST'])
def add_encounters(ibs, enctext_list, encounter_uuid_list=None, config_rowid_list=None,
                   notes_list=None):
    r"""
    Adds a list of encounters.

    Args:
        enctext_list (list):
        encounter_uuid_list (list):
        config_rowid_list (list):
        notes_list (list):

    Returns:
        eid_list (list): added encounter rowids

    RESTful:
        Method: POST
        URL:    /api/encounter/
    """
    if ut.VERBOSE:
        print('[ibs] adding %d encounters' % len(enctext_list))
    # Add encounter text names to database
    if notes_list is None:
        notes_list = [''] * len(enctext_list)
    if encounter_uuid_list is None:
        encounter_uuid_list = [uuid.uuid4() for _ in range(len(enctext_list))]
    if config_rowid_list is None:
        config_rowid_list = [ibs.MANUAL_CONFIGID] * len(enctext_list)
    colnames = ['encounter_text', 'encounter_uuid', 'config_rowid', 'encounter_note']
    params_iter = zip(enctext_list, encounter_uuid_list, config_rowid_list, notes_list)
    get_rowid_from_superkey = functools.partial(ibs.get_encounter_eids_from_text, ensure=False)
    eid_list = ibs.db.add_cleanly(const.ENCOUNTER_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return eid_list


# SETTERS::ENCOUNTER


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/text/', methods=['PUT'])
def set_encounter_text(ibs, eid_list, encounter_text_list):
    r"""
    Sets names of encounters (groups of animals)

    RESTful:
        Method: PUT
        URL:    /api/encounter/text/
    """
    # Special set checks
    if any(ibs.is_special_encounter(eid_list)):
        raise ValueError('cannot rename special encounters')
    id_iter = ((eid,) for eid in eid_list)
    val_list = ((encounter_text,) for encounter_text in encounter_text_list)
    ibs.db.set(const.ENCOUNTER_TABLE, ('encounter_text',), val_list, id_iter)

#
# GETTERS::ENCOUNTER


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/num_imgs_reviewed/', methods=['GET'])
def get_encounter_num_imgs_reviewed(ibs, eid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/encounter/num_imgs_reviewed/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> # Reset and compute encounters
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> num_reviwed_list = ibs.get_encounter_num_imgs_reviewed(eid_list)
        >>> result = num_reviwed_list
        >>> print(result)
        [0, 0]
    """
    gids_list = ibs.get_encounter_gids(eid_list)
    flags_list = ibs.unflat_map(ibs.get_image_reviewed, gids_list)
    num_reviwed_list = [sum(flags) for flags in flags_list]
    return num_reviwed_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/num_annotmatch_reviewed/', methods=['GET'])
def get_encounter_num_annotmatch_reviewed(ibs, eid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/encounter/num_annotmatch_reviewed/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> eid_list = ibs._get_all_encounter_rowids()
        >>> num_annots_reviewed_list = ibs.get_encounter_num_annotmatch_reviewed(eid_list)
    """
    aids_list = ibs.get_encounter_custom_filtered_aids(eid_list)
    has_revieweds_list = ibs.unflat_map(ibs.get_annot_has_reviewed_matching_aids, aids_list)
    num_annots_reviewed_list = list(map(sum, has_revieweds_list))
    return num_annots_reviewed_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/num_names_with_exemplar/', methods=['GET'])
def get_encounter_num_names_with_exemplar(ibs, eid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/encounter/num_names_with_exemplar/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> eid_list = ibs._get_all_encounter_rowids()
        >>> num_annots_reviewed_list = ibs.get_encounter_num_annotmatch_reviewed(eid_list)
    """
    aids_list = ibs.get_encounter_custom_filtered_aids(eid_list)
    exflags_list = ibs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
    nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    groups_list = [ut.group_items(exflags, nids)
                   for exflags, nids in zip(exflags_list, nids_list)]
    #num_names_list = [len(groups) for groups in groups_list]
    num_exemplared_names_list = [
        sum([any(exflags) for exflags in six.itervalues(groups)])
        for groups in groups_list
    ]
    return num_exemplared_names_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/fraction_names_with_exemplar/', methods=['GET'])
def get_encounter_fraction_names_with_exemplar(ibs, eid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/encounter/fraction_names_with_exemplar/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb2')
        >>> eid_list = ibs._get_all_encounter_rowids()
        >>> fraction_exemplared_names_list = ibs.get_encounter_fraction_names_with_exemplar(eid_list)
    """
    aids_list = ibs.get_encounter_custom_filtered_aids(eid_list)
    #exflags_list = ibs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
    nids_list = list(map(list, map(set, ibs.unflat_map(ibs.get_annot_name_rowids, aids_list))))
    aids_list_list = ibs.unflat_map(ibs.get_name_aids, nids_list)
    flags_list_list = list(map(lambda x: ibs.unflat_map(ibs.get_annot_exemplar_flags, x), aids_list_list))
    #groups_list = [ut.group_items(exflags, nids)
    #               for exflags, nids in zip(exflags_list, nids_list)]
    num_names_list = list(map(len, nids_list))
    num_exemplared_names_list = [
        sum([any(exflags) for exflags in flags_list])
        for flags_list in flags_list_list
    ]
    fraction_exemplared_names_list = [
        None if num_names == 0 else num_exemplared_names / num_names
        for num_exemplared_names,  num_names in zip(num_exemplared_names_list,  num_names_list)
    ]
    return fraction_exemplared_names_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/fraction_annotmatch_reviewed/', methods=['GET'])
def get_encounter_fraction_annotmatch_reviewed(ibs, eid_list):
    r"""
    Auto-docstr for 'get_encounter_fraction_annotmatch_reviewed'

    RESTful:
        Method: GET
        URL:    /api/encounter/fraction_annotmatch_reviewed/
    """
    aids_list = ibs.get_encounter_custom_filtered_aids(eid_list)
    flags_list = ibs.unflat_map(ibs.get_annot_has_reviewed_matching_aids, aids_list)
    fraction_annotmatch_reviewed_list = [None if len(flags) == 0 else sum(flags) / len(flags)
                                         for flags in flags_list]
    return fraction_annotmatch_reviewed_list


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/encounter/custom_filtered_aids/', methods=['GET'])
def get_encounter_custom_filtered_aids(ibs, eid_list):
    r"""
    hacks to filter aids to only certain views and qualities

    RESTful:
        Method: GET
        URL:    /api/encounter/custom_filtered_aids/
    """
    aids_list_ = ibs.get_encounter_aids(eid_list)
    # HACK: Get percentage for the annots we currently care about
    aids_list = [ibs.filter_aids_custom(aids) for aids in aids_list_]
    return aids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/fraction_imgs_reviewed/', methods=['GET'])
def get_encounter_fraction_imgs_reviewed(ibs, eid_list):
    r"""
    Auto-docstr for 'get_encounter_fraction_imgs_reviewed'

    RESTful:
        Method: GET
        URL:    /api/encounter/fraction_imgs_reviewed/
    """
    gids_list = ibs.get_encounter_gids(eid_list)
    flags_list = ibs.unflat_map(ibs.get_image_reviewed, gids_list)
    fraction_imgs_reviewed_list = [None if len(flags) == 0 else sum(flags) / len(flags)
                                   for flags in flags_list]
    return fraction_imgs_reviewed_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, 'percent_names_with_exemplar_str', debug=False)  # HACK
@register_api('/api/encounter/percent_names_with_exemplar_str/', methods=['GET'])
def get_encounter_percent_names_with_exemplar_str(ibs, eid_list):
    r"""
    Auto-docstr for 'get_encounter_percent_names_with_exemplar_str'

    RESTful:
        Method: GET
        URL:    /api/encounter/percent_names_with_exemplar_str/
    """
    fraction_exemplared_names_list = ibs.get_encounter_fraction_names_with_exemplar(eid_list)
    percent_exemplared_names_list_str = list(map(ut.percent_str, fraction_exemplared_names_list))
    return percent_exemplared_names_list_str


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, 'percent_imgs_reviewed_str', debug=False)  # HACK
@register_api('/api/encounter/percent_imgs_reviewed_str/', methods=['GET'])
def get_encounter_percent_imgs_reviewed_str(ibs, eid_list):
    r"""
    Auto-docstr for 'get_encounter_percent_imgs_reviewed_str'

    RESTful:
        Method: GET
        URL:    /api/encounter/percent_imgs_reviewed_str/
    """
    fraction_imgs_reviewed_list = ibs.get_encounter_fraction_imgs_reviewed(eid_list)
    percent_imgs_reviewed_str_list = list(map(ut.percent_str, fraction_imgs_reviewed_list))
    return percent_imgs_reviewed_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, 'percent_annotmatch_reviewed_str', debug=False)  # HACK
@register_api('/api/encounter/percent_annotmatch_reviewed_str/', methods=['GET'])
def get_encounter_percent_annotmatch_reviewed_str(ibs, eid_list):
    r"""
    Auto-docstr for 'get_encounter_percent_annotmatch_reviewed_str'

    RESTful:
        Method: GET
        URL:    /api/encounter/percent_annotmatch_reviewed_str/
    """
    fraction_annotmatch_reviewed_list = ibs.get_encounter_fraction_annotmatch_reviewed(eid_list)
    percent_annotmach_reviewed_str_list = list(map(ut.percent_str, fraction_annotmatch_reviewed_list))
    return percent_annotmach_reviewed_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/num_gids/', methods=['GET'])
def get_encounter_num_gids(ibs, eid_list):
    r"""
    Returns:
        nGids_list (list): number of images in each encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/num_gids/
    """
    nGids_list = list(map(len, ibs.get_encounter_gids(eid_list)))
    return nGids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/num_aids/', methods=['GET'])
def get_encounter_num_aids(ibs, eid_list):
    r"""
    Returns:
        nGids_list (list): number of images in each encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/num_aids/
    """
    nAids_list = list(map(len, ibs.get_encounter_aids(eid_list)))
    return nAids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/encounter/aids/', methods=['GET'])
def get_encounter_aids(ibs, eid_list):
    r"""
    Returns:
        aids_list (list):  a list of list of aids in each encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/aids/

    Args:
        ibs (IBEISController):  ibeis controller object
        eid_list (list):

    Returns:
        list: aids_list

    CommandLine:
        python -m ibeis.control.manual_encounter_funcs --test-get_encounter_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> ibs.delete_encounters(ibs.get_valid_eids())
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> aids_list = get_encounter_aids(ibs, eid_list)
        >>> result = ('aids_list = %s' % (str(aids_list),))
        >>> print(result)
    """
    gids_list = ibs.get_encounter_gids(eid_list)
    aids_list_ = ibs.unflat_map(ibs.get_image_aids, gids_list)
    aids_list = list(map(ut.flatten, aids_list_))
    #print('get_encounter_aids')
    #print('eid_list = %r' % (eid_list,))
    #print('gids_list = %r' % (gids_list,))
    #print('aids_list_ = %r' % (aids_list_,))
    #print('aids_list = %r' % (aids_list,))
    return aids_list


@register_ibs_method
@accessor_decors.getter_1toM
@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, 'image_rowids')
@register_api('/api/encounter/gids/', methods=['GET'])
def get_encounter_gids(ibs, eid_list):
    r"""
    Returns:
        gids_list (list):  a list of list of gids in each encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/gids/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    gids_list = ibs.db.get(const.EG_RELATION_TABLE, ('image_rowid',), eid_list, id_colname='encounter_rowid', unpack_scalars=False)
    #print('get_encounter_gids')
    #print('eid_list = %r' % (eid_list,))
    #print('gids_list = %r' % (gids_list,))
    return gids_list


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/encounter/egrids/', methods=['GET'])
def get_encounter_egrids(ibs, eid_list=None, gid_list=None):
    r"""
    Returns:
        list_ (list):  a list of encounter-image-relationship rowids for each encouterid

    RESTful:
        Method: GET
        URL:    /api/encounter/egrids/
    """
    # WEIRD FUNCTION FIXME
    assert eid_list is not None or gid_list is not None, "Either eid_list or gid_list must be None"
    if eid_list is not None and gid_list is None:
        # TODO: Group type
        params_iter = ((eid,) for eid in eid_list)
        where_clause = 'encounter_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
    elif gid_list is not None and eid_list is None:
        # TODO: Group type
        params_iter = ((gid,) for gid in gid_list)
        where_clause = 'image_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
    else:
        # TODO: Group type
        params_iter = ((eid, gid,) for eid, gid in zip(eid_list, gid_list))
        where_clause = 'encounter_rowid=? AND image_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
    return egrids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/encounter/nids/', methods=['GET'])
def get_encounter_nids(ibs, eid_list):
    r"""
    Returns:
        list_ (list):  a list of list of known nids in each encounter

    CommandLine:
        python -m ibeis.control.manual_encounter_funcs --test-get_encounter_nids

    RESTful:
        Method: GET
        URL:    /api/encounter/nids/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs.delete_encounters(ibs.get_valid_eids())
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> nids_list = ibs.get_encounter_nids(eid_list)
        >>> result = nids_list
        >>> print(result)
        [[1, 2, 3], [4, 5, 6, 7]]
    """
    aids_list = ibs.get_encounter_aids(eid_list)
    nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    #nids_list_ = [[nid[0] for nid in nids if len(nid) > 0] for nids in nids_list]
    # Remove any unknown anmes
    nids_list = [[nid for nid in nids if nid > 0] for nids in nids_list]

    nids_list = list(map(ut.unique_ordered, nids_list))
    #print('get_encounter_nids')
    #print('eid_list = %r' % (eid_list,))
    #print('aids_list = %r' % (aids_list,))
    #print('nids_list_ = %r' % (nids_list_,))
    #print('nids_list = %r' % (nids_list,))
    return nids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/uuid/', methods=['GET'])
def get_encounter_uuid(ibs, eid_list):
    r"""
    Returns:
        list_ (list): encounter_uuid of each eid in eid_list

    RESTful:
        Method: GET
        URL:    /api/encounter/uuid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    encuuid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_uuid',), eid_list, id_colname='encounter_rowid')
    return encuuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/configid/', methods=['GET'])
def get_encounter_configid(ibs, eid_list):
    r"""
    Returns:
        list_ (list): config_rowid of each eid in eid_list

    RESTful:
        Method: GET
        URL:    /api/encounter/configid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    config_rowid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('config_rowid',), eid_list, id_colname='encounter_rowid')
    return config_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/text/', methods=['GET'])
def get_encounter_text(ibs, eid_list):
    r"""
    Returns:
        list_ (list): encounter_text of each eid in eid_list

    RESTful:
        Method: GET
        URL:    /api/encounter/text/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    enctext_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_text',), eid_list, id_colname='encounter_rowid')
    return enctext_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/eids_from_text/', methods=['GET'])
def get_encounter_eids_from_text(ibs, enctext_list, ensure=True):
    r"""
    Returns:
        list_ (list): a list of eids corresponding to each encounter enctext
    #FIXME: make new naming scheme for non-primary-key-getters
    get_encounter_eids_from_text_from_text

    RESTful:
        Method: GET
        URL:    /api/encounter/eids_from_text/
    """
    if ensure:
        eid_list = ibs.add_encounters(enctext_list)
    else:
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        eid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_rowid',), enctext_list, id_colname='encounter_text')
    return eid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/note/', methods=['GET'])
def get_encounter_note(ibs, eid_list):
    r"""
    Returns:
        list_ (list): encounter_note of each eid in eid_list

    RESTful:
        Method: GET
        URL:    /api/encounter/note/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    encnote_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_note',), eid_list, id_colname='encounter_rowid')
    return encnote_list


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/encounter/', methods=['DELETE'])
def delete_encounters(ibs, eid_list):
    r"""
    Removes encounters and thier relationships (images are not effected)

    RESTful:
        Method: DELETE
        URL:    /api/encounter/
    """
    # Optimization hack, less SQL calls
    #egrid_list = ut.flatten(ibs.get_encounter_egrids(eid_list=eid_list))
    #ibs.db.delete_rowids(const.EG_RELATION_TABLE, egrid_list)
    #ibs.db.delete(const.EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')
    if ut.VERBOSE:
        print('[ibs] deleting %d encounters' % len(eid_list))
    ibs.delete_egr_encounter_relations(eid_list)
    ibs.db.delete_rowids(const.ENCOUNTER_TABLE, eid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/end_time_posix/', methods=['GET'])
def get_encounter_end_time_posix(ibs, encounter_rowid_list):
    r"""
    encounter_end_time_posix_list <- encounter.encounter_end_time_posix[encounter_rowid_list]

    gets data from the "native" column "encounter_end_time_posix" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_end_time_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_end_time_posix
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/end_time_posix/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_end_time_posix_list = ibs.get_encounter_end_time_posix(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_end_time_posix_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_END_TIME_POSIX,)
    encounter_end_time_posix_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_end_time_posix_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/gps_lats/', methods=['GET'])
def get_encounter_gps_lats(ibs, encounter_rowid_list):
    r"""
    encounter_gps_lat_list <- encounter.encounter_gps_lat[encounter_rowid_list]

    gets data from the "native" column "encounter_gps_lat" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_gps_lat_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_gps_lat
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/gps_lats/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_gps_lat_list = ibs.get_encounter_gps_lats(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_gps_lat_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LAT,)
    encounter_gps_lat_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_gps_lat_list


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/encounter/info/', methods=['PUT'])
def update_encounter_info(ibs, encounter_rowid_list):
    r"""
    sets start and end time for encounters

    FIXME: should not need to bulk update, should be handled as it goes

    RESTful:
        Method: PUT
        URL:    /api/encounter/info/

    Example:
        >>> # DOCTEST_DISABLE
        >>> encounter_rowid_list = ibs.get_valid_eids()
    """
    gids_list_ = ibs.get_encounter_gids(encounter_rowid_list)
    hasgids_list = [len(gids) > 0 for gids in gids_list_]
    gids_list = ut.filter_items(gids_list_, hasgids_list)
    eid_list = ut.filter_items(encounter_rowid_list, hasgids_list)
    unixtimes_list = ibs.unflat_map(ibs.get_image_unixtime, gids_list)
    encounter_end_time_posix_list = [max(unixtimes) for unixtimes in unixtimes_list]
    encounter_start_time_posix_list = [min(unixtimes) for unixtimes in unixtimes_list]
    ibs.set_encounter_end_time_posix(eid_list, encounter_end_time_posix_list)
    ibs.set_encounter_start_time_posix(eid_list, encounter_start_time_posix_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/gps_lons/', methods=['GET'])
def get_encounter_gps_lons(ibs, encounter_rowid_list):
    r"""
    encounter_gps_lon_list <- encounter.encounter_gps_lon[encounter_rowid_list]

    gets data from the "native" column "encounter_gps_lon" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_gps_lon_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_gps_lon
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/gps_lons/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_gps_lon_list = ibs.get_encounter_gps_lons(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_gps_lon_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LON,)
    encounter_gps_lon_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_gps_lon_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/notes/', methods=['GET'])
def get_encounter_notes(ibs, encounter_rowid_list):
    r"""
    encounter_note_list <- encounter.encounter_note[encounter_rowid_list]

    gets data from the "native" column "encounter_note" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_note_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_note
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/notes/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_note_list = ibs.get_encounter_notes(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_note_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_NOTE,)
    encounter_note_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_note_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/processed_flags/', methods=['GET'])
def get_encounter_processed_flags(ibs, encounter_rowid_list):
    r"""
    encounter_processed_flag_list <- encounter.encounter_processed_flag[encounter_rowid_list]

    gets data from the "native" column "encounter_processed_flag" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_processed_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_processed_flag
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/processed_flags/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_processed_flag_list = ibs.get_encounter_processed_flags(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_processed_flag_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_PROCESSED_FLAG,)
    encounter_processed_flag_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_processed_flag_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/shipped_flags/', methods=['GET'])
def get_encounter_shipped_flags(ibs, encounter_rowid_list):
    r"""
    encounter_shipped_flag_list <- encounter.encounter_shipped_flag[encounter_rowid_list]

    gets data from the "native" column "encounter_shipped_flag" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_shipped_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_shipped_flag
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/shipped_flags/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_shipped_flag_list = ibs.get_encounter_shipped_flags(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_shipped_flag_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SHIPPED_FLAG,)
    encounter_shipped_flag_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_shipped_flag_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/encounter/start_time_posix/', methods=['GET'])
def get_encounter_start_time_posix(ibs, encounter_rowid_list):
    r"""
    encounter_start_time_posix_list <- encounter.encounter_start_time_posix[encounter_rowid_list]

    gets data from the "native" column "encounter_start_time_posix" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_start_time_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_start_time_posix
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/start_time_posix/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_start_time_posix_list = ibs.get_encounter_start_time_posix(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_start_time_posix_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_START_TIME_POSIX,)
    encounter_start_time_posix_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_start_time_posix_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/end_time_posix/', methods=['PUT'])
def set_encounter_end_time_posix(ibs, encounter_rowid_list, encounter_end_time_posix_list):
    r"""
    encounter_end_time_posix_list -> encounter.encounter_end_time_posix[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_end_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_end_time_posix

    RESTful:
        Method: PUT
        URL:    /api/encounter/end_time_posix/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_END_TIME_POSIX,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_end_time_posix_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/gps_lats/', methods=['PUT'])
def set_encounter_gps_lats(ibs, encounter_rowid_list, encounter_gps_lat_list):
    r"""
    encounter_gps_lat_list -> encounter.encounter_gps_lat[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_gps_lat_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_gps_lat

    RESTful:
        Method: PUT
        URL:    /api/encounter/gps_lats/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LAT,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, encounter_gps_lat_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/gps_lons/', methods=['PUT'])
def set_encounter_gps_lons(ibs, encounter_rowid_list, encounter_gps_lon_list):
    r"""
    encounter_gps_lon_list -> encounter.encounter_gps_lon[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_gps_lon_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_gps_lon

    RESTful:
        Method: PUT
        URL:    /api/encounter/gps_lons/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LON,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, encounter_gps_lon_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/notes/', methods=['PUT'])
def set_encounter_notes(ibs, encounter_rowid_list, encounter_note_list):
    r"""
    encounter_note_list -> encounter.encounter_note[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_note_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_note

    RESTful:
        Method: PUT
        URL:    /api/encounter/notes/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_NOTE,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, encounter_note_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/processed_flags/', methods=['PUT'])
def set_encounter_processed_flags(ibs, encounter_rowid_list, encounter_processed_flag_list):
    r"""
    encounter_processed_flag_list -> encounter.encounter_processed_flag[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_processed_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_processed_flag

    RESTful:
        Method: PUT
        URL:    /api/encounter/processed_flags/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_PROCESSED_FLAG,)
    val_iter = ((processed_flag,) for processed_flag in encounter_processed_flag_list)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/shipped_flags/', methods=['PUT'])
def set_encounter_shipped_flags(ibs, encounter_rowid_list, encounter_shipped_flag_list):
    r"""
    encounter_shipped_flag_list -> encounter.encounter_shipped_flag[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_shipped_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_shipped_flag

    RESTful:
        Method: PUT
        URL:    /api/encounter/shipped_flags/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SHIPPED_FLAG,)
    val_iter = ((shipped_flag,) for shipped_flag in encounter_shipped_flag_list)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/encounter/start_time_posix/', methods=['PUT'])
def set_encounter_start_time_posix(ibs, encounter_rowid_list, encounter_start_time_posix_list):
    r"""
    encounter_start_time_posix_list -> encounter.encounter_start_time_posix[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_start_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_start_time_posix

    RESTful:
        Method: PUT
        URL:    /api/encounter/start_time_posix/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_START_TIME_POSIX,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_start_time_posix_list, id_iter)


@register_ibs_method
#@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, ENCOUNTER_SMART_WAYPOINT_ID)
@register_api('/api/encounter/smart_waypoint_ids/', methods=['GET'])
def get_encounter_smart_waypoint_ids(ibs, encounter_rowid_list):
    r"""
    encounter_smart_waypoint_id_list <- encounter.encounter_smart_waypoint_id[encounter_rowid_list]

    gets data from the "native" column "encounter_smart_waypoint_id" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_smart_waypoint_id_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_smart_waypoint_id
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/smart_waypoint_ids/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_smart_waypoint_id_list = ibs.get_encounter_smart_waypoint_ids(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_smart_waypoint_id_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_WAYPOINT_ID,)
    encounter_smart_waypoint_id_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_smart_waypoint_id_list


@register_ibs_method
#@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, ENCOUNTER_SMART_XML_FNAME)
@register_api('/api/encounter/smart_xml_fnames/', methods=['GET'])
def get_encounter_smart_xml_fnames(ibs, encounter_rowid_list):
    r"""
    encounter_smart_xml_fname_list <- encounter.encounter_smart_xml_fname[encounter_rowid_list]

    gets data from the "native" column "encounter_smart_xml_fname" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_smart_xml_fname_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_smart_xml_fname
        tbl = encounter

    RESTful:
        Method: GET
        URL:    /api/encounter/smart_xml_fnames/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_encounter_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_smart_xml_fname_list = ibs.get_encounter_smart_xml_fnames(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_smart_xml_fname_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_XML_FNAME,)
    encounter_smart_xml_fname_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_smart_xml_fname_list


@register_ibs_method
@register_api('/api/encounter/smart_waypoint_ids/', methods=['PUT'])
def set_encounter_smart_waypoint_ids(ibs, encounter_rowid_list, encounter_smart_waypoint_id_list):
    r"""
    encounter_smart_waypoint_id_list -> encounter.encounter_smart_waypoint_id[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_smart_waypoint_id_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_smart_waypoint_id

    RESTful:
        Method: PUT
        URL:    /api/encounter/smart_waypoint_ids/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_WAYPOINT_ID,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_smart_waypoint_id_list, id_iter)


@register_ibs_method
@register_api('/api/encounter/smart_xml_fnames/', methods=['PUT'])
def set_encounter_smart_xml_fnames(ibs, encounter_rowid_list, encounter_smart_xml_fname_list):
    r"""
    encounter_smart_xml_fname_list -> encounter.encounter_smart_xml_fname[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_smart_xml_fname_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_smart_xml_fname

    RESTful:
        Method: PUT
        URL:    /api/encounter/smart_xml_fnames/
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_XML_FNAME,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_smart_xml_fname_list, id_iter)


def testdata_ibs():
    r"""
    Auto-docstr for 'testdata_ibs'
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.control.manual_encounter_funcs
        python -m ibeis.control.manual_encounter_funcs --allexamples
        python -m ibeis.control.manual_encounter_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
