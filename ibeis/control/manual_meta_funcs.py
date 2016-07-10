# -*- coding: utf-8 -*-
"""
controller functions for contributors, versions, configs, and other metadata
"""
from __future__ import absolute_import, division, print_function
#import uuid
import six  # NOQA
#from os.path import join
import functools
from six.moves import range, input, zip, map  # NOQA
from ibeis import constants as const
from ibeis.control import accessor_decors, controller_inject
import utool as ut
from ibeis.algo import Config
#from ibeis.other import ibsfuncs
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, profile = ut.inject2(__name__, '[manual_meta]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_ibs_method
@accessor_decors.adder
# @register_api('/api/contributor/', methods=['POST'])
def add_contributors(ibs, tag_list, uuid_list=None, name_first_list=None, name_last_list=None,
                     loc_city_list=None, loc_state_list=None,
                     loc_country_list=None, loc_zip_list=None,
                     notes_list=None):
    r"""
    Adds a list of contributors.

    Returns:
        contrib_id_list (list): contributor rowids

    RESTful:
        Method: POST
        URL:    /api/contributor/
    """
    import datetime
    def _valid_zip(_zip, default='00000'):
        _zip = str(_zip)
        if len(_zip) == 5 and _zip.isdigit():
            return _zip
        return default

    if ut.VERBOSE:
        print('[ibs] adding %d imagesets' % len(tag_list))
    # Add contributors to database
    if name_first_list is None:
        name_first_list = [''] * len(tag_list)
    if name_last_list is None:
        name_last_list = [''] * len(tag_list)
    if loc_city_list is None:
        loc_city_list = [''] * len(tag_list)
    if loc_state_list is None:
        loc_state_list = [''] * len(tag_list)
    if loc_country_list is None:
        loc_country_list = [''] * len(tag_list)
    if loc_zip_list is None:
        loc_zip_list = [''] * len(tag_list)
    if notes_list is None:
        notes_list = [ "Created %s" % (datetime.datetime.now(),) for _ in range(len(tag_list))]

    loc_zip_list = [ _valid_zip(_zip) for _zip in loc_zip_list]

    if uuid_list is None:
        #contrib_rowid_list = ibs.get_contributor_rowid_from_tag(tag_list)
        #uuid_list = ibs.get_contributor_uuid(contrib_rowid_list)
        #uuid_list = ibs.get_contributor_uuid(contrib_rowid_list)
        #uuid_list = [ uuid.uuid4() if uuid_ is None else uuid_ for uuid_ in uuid_list ]
        # DETERMENISTIC UUIDS
        zero_uuid = ut.get_zero_uuid()
        uuid_list = [ut.augment_uuid(zero_uuid, tag) for tag in tag_list]

    colnames = ['contributor_uuid', 'contributor_tag', 'contributor_name_first',
                'contributor_name_last', 'contributor_location_city',
                'contributor_location_state', 'contributor_location_country',
                'contributor_location_zip', 'contributor_note']
    params_iter = zip(uuid_list, tag_list, name_first_list,
                      name_last_list, loc_city_list, loc_state_list,
                      loc_country_list, loc_zip_list, notes_list)

    get_rowid_from_superkey = ibs.get_contributor_rowid_from_uuid
    #get_rowid_from_superkey = ibs.get_contributor_rowid_from_tag  # ?? is tag a superkey?
    contrib_id_list = ibs.db.add_cleanly(const.CONTRIBUTOR_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return contrib_id_list


@register_ibs_method
@accessor_decors.adder
def add_version(ibs, versiontext_list):
    r"""
    Adds an algorithm / actor configuration as a string
    """
    # FIXME: Configs are still handled poorly
    params_iter = ((versiontext,) for versiontext in versiontext_list)
    get_rowid_from_superkey = ibs.get_version_rowid_from_superkey
    versionid_list = ibs.db.add_cleanly(const.VERSIONS_TABLE, ('version_text',),
                                        params_iter, get_rowid_from_superkey)
    return versionid_list


@register_ibs_method
@accessor_decors.adder
# @register_api('/api/config/', methods=['POST'])
def add_config(ibs, cfgsuffix_list, contrib_rowid_list=None):
    r"""
    Adds an algorithm / actor configuration as a string

    RESTful:
        Method: POST
        URL:    /api/config/

    Args:
        ibs (IBEISController):  ibeis controller object
        cfgsuffix_list (list):
        contrib_rowid_list (list): (default = None)

    Returns:
        list: config_rowid_list

    CommandLine:
        python -m ibeis.control.manual_meta_funcs --exec-add_config

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_meta_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> cfgsuffix_list = ['_CHIP(sz450)']
        >>> contrib_rowid_list = None
        >>> config_rowid_list = add_config(ibs, cfgsuffix_list, contrib_rowid_list)
        >>> result = ('config_rowid_list = %s' % (str(config_rowid_list),))
        >>> print(result)
    """
    # FIXME: Configs are still handled poorly. This function is an ensure
    params_iter = ((suffix,) for suffix in cfgsuffix_list)
    get_rowid_from_superkey = ibs.get_config_rowid_from_suffix
    config_rowid_list = ibs.db.add_cleanly(const.CONFIG_TABLE, ('config_suffix',),
                                           params_iter, get_rowid_from_superkey)
    if contrib_rowid_list is not None:
        ibs.set_config_contributor_rowid(config_rowid_list, contrib_rowid_list)
    return config_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
def ensure_config_rowid_from_suffix(ibs, cfgsuffix_list):
    config_rowid_list = ibs.get_config_rowid_from_suffix(cfgsuffix_list)
    is_dirty_list = ut.flag_None_items(config_rowid_list)
    if any(is_dirty_list):
        # Only call adder if needed, adders cause debug output to be large
        return ibs.add_config(cfgsuffix_list)
    else:
        return config_rowid_list


#@register_ibs_method
#@accessor_decors.default_decorator
#@register_api('/api/query/config/rowid/', methods=['GET'])
#def get_query_config_rowid(ibs):
#    r"""
#    # FIXME: Configs are still handled poorly

#    RESTful:
#        Method: GET
#        URL:    /api/query/config/rowid/
#    """
#    query_cfg_suffix = ibs.cfg.query_cfg.get_cfgstr()
#    query_cfg_rowid = ibs.add_config(query_cfg_suffix)
#    return query_cfg_rowid

# SETTERS::METADATA


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/metadata/value/', methods=['PUT'])
def set_metadata_value(ibs, metadata_key_list, metadata_value_list, db):
    r"""
    Sets metadata key, value pairs

    RESTful:
        Method: PUT
        URL:    /api/metadata/value/
    """
    db = db[0]  # Unwrap tuple, required by @accessor_decors.setter decorator
    metadata_rowid_list = ibs.get_metadata_rowid_from_metadata_key(metadata_key_list, db)
    id_iter = ((metadata_rowid,) for metadata_rowid in metadata_rowid_list)
    val_list = ((metadata_value,) for metadata_value in metadata_value_list)
    db.set(const.METADATA_TABLE, ('metadata_value',), val_list, id_iter)


@register_ibs_method
def set_database_version(ibs, db, version):
    r"""
    Sets the specified database's version from the controller
    """
    db.set_db_version(version)

# SETTERS::CONTRIBUTORS


@register_ibs_method
# @register_api('/api/config/contributor/rowid/', methods=['PUT'])
def set_config_contributor_rowid(ibs, config_rowid_list, contrib_rowid_list):
    r"""
    Sets the config's contributor rowid

    RESTful:
        Method: PUT
        URL:    /api/config/contributor/rowid/
    """
    id_iter = ((config_rowid,) for config_rowid in config_rowid_list)
    val_list = ((contrib_rowid,) for contrib_rowid in contrib_rowid_list)
    ibs.db.set(const.CONFIG_TABLE, ('contributor_rowid',), val_list, id_iter)


@register_ibs_method
# @register_api('/api/contributor/new/temp/', methods=['POST'])
def add_new_temp_contributor(ibs, user_prompt=False, offset=None, autolocate=False):
    r"""

    RESTful:
        Method: POST
        URL:    /api/contributor/new/temp/
    """
    name_first = ibs.get_dbname()
    name_last = ut.get_computer_name() + ':' + ut.get_user_name() + ':' + ibs.get_dbdir()
    print('[collect_transfer_data] Contributor default first name: %s' % (name_first, ))
    print('[collect_transfer_data] Contributor default last name:  %s' % (name_last, ))
    if user_prompt:
        name_first = input('\n[collect_transfer_data] Change first name (Enter to use default): ')
        name_last  = input('\n[collect_transfer_data] Change last name (Enter to use default): ')

    if autolocate:
        success, location_city, location_state, location_country, location_zip = ut.geo_locate()
    else:
        success = False

    if success:
        print('\n[collect_transfer_data] Your location was be determined automatically.')
        print('[collect_transfer_data] Contributor default city: %s'    % (location_city, ))
        print('[collect_transfer_data] Contributor default state: %s'   % (location_state, ))
        print('[collect_transfer_data] Contributor default zip: %s'     % (location_country, ))
        print('[collect_transfer_data] Contributor default country: %s' % (location_zip, ))
        if user_prompt:
            location_city    = input('\n[collect_transfer_data] Change default location city (Enter to use default): ')
            location_state   = input('\n[collect_transfer_data] Change default location state (Enter to use default): ')
            location_zip     = input('\n[collect_transfer_data] Change default location zip (Enter to use default): ')
            location_country = input('\n[collect_transfer_data] Change default location country (Enter to use default): ')
    else:
        if user_prompt:
            print('\n')
        print('[collect_transfer_data] Your location could not be determined automatically.')
        if user_prompt:
            location_city    = input('[collect_transfer_data] Enter your location city (Enter to skip): ')
            location_state   = input('[collect_transfer_data] Enter your location state (Enter to skip): ')
            location_zip     = input('[collect_transfer_data] Enter your location zip (Enter to skip): ')
            location_country = input('[collect_transfer_data] Enter your location country (Enter to skip): ')
        else:
            location_city    = ''
            location_state   = ''
            location_zip     = ''
            location_country = ''

    #tag = '::'.join([name_first, name_last, location_city, location_state, location_zip, location_country])
    tag_components = [name_first, name_last, location_city, location_state, location_zip, location_country]
    if offset is not None:
        tag_components += [str(offset)]
    tag_components_clean = [comp.replace(';', '<semi>') for comp in tag_components]
    tag = ','.join(tag_components_clean)
    contrib_rowid = ibs.add_contributors(
        [tag], name_first_list=[name_first],
        name_last_list=[name_last], loc_city_list=[location_city],
        loc_state_list=[location_state], loc_country_list=[location_country],
        loc_zip_list=[location_zip])[0]
    return contrib_rowid


@register_ibs_method
def ensure_contributor_rowids(ibs, user_prompt=False, autolocate=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        user_prompt (bool):

    Returns:
        list:

    CommandLine:
        python -m ibeis.control.manual_meta_funcs --test-ensure_contributor_rowids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_meta_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(db='testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> ibs.delete_contributors(ibs.get_valid_contrib_rowids())
        >>> contrib_rowid_list1 = ibs.get_image_contributor_rowid(gid_list)
        >>> assert ut.allsame(contrib_rowid_list1)
        >>> ut.assert_eq(contrib_rowid_list1[0], None)
        >>> user_prompt = ut.get_argflag('--user-prompt')
        >>> autolocate = ut.get_argflag('--user-prompt')
        >>> # execute function
        >>> result = ensure_contributor_rowids(ibs, user_prompt, autolocate)
        >>> # verify results
        >>> ibs.print_contributor_table()
        >>> print(result)
        >>> contrib_rowid_list2 = ibs.get_image_contributor_rowid(gid_list)
        >>> assert ut.allsame(contrib_rowid_list2)
        >>> ut.assert_eq(contrib_rowid_list2[0], 1)
    """
    # TODO: Alter this check to support merging databases with more than one contributor, but none assigned to the manual config
    if not ut.QUIET:
        print('[ensure_contributor_rowids] Ensuring all images have contributors for dbname=%r' % (ibs.get_dbname()))
    contrib_rowid_list = ibs.get_valid_contrib_rowids()
    unassigned_gid_list = ibs.get_all_uncontributed_images()
    if not ut.QUIET:
        print('[ensure_contributor_rowids] %d Contributors exist. %d images are unassigned' %
              (len(contrib_rowid_list), len(unassigned_gid_list)))
    if len(unassigned_gid_list) > 0:
        new_contrib_rowid = ibs.add_new_temp_contributor(offset=len(contrib_rowid_list), user_prompt=user_prompt, autolocate=autolocate)
        # SET UNASSIGNED IMAGE CONTRIBUTORS
        ibs.set_image_contributor_rowid(unassigned_gid_list, [new_contrib_rowid] * len(unassigned_gid_list))
        ibs.ensure_imageset_configs_populated()
    # make sure that all images have assigned contributors
    # Get new non-conflicting contributor for unassigned images
    #contrib_rowid_list = list([new_contrib_rowid]) * len(unassigned_gid_list)
    #ibs.set_config_contributor_rowid(unassigned_gid_list, contrib_rowid_list)
    return ibs.get_valid_contrib_rowids()


@register_ibs_method
# @register_api('/api/contributor/gids/uncontributed/', methods=['GET'])
def get_all_uncontributed_images(ibs):
    r"""

    RESTful:
        Method: GET
        URL:    /api/contributor/gids/uncontributed/
    """
    gid_list = ibs.get_valid_gids()
    contrib_rowid_list = ibs.get_image_contributor_rowid(gid_list)
    is_unassigned = [contrib_rowid is None for contrib_rowid in contrib_rowid_list]
    unassigned_gid_list = ut.compress(gid_list, is_unassigned)
    #sum(is_unassigned)
    #len(is_unassignd)
    #unassigned_gid_list = [
    #    gid
    #    for gid, _contrib_rowid in zip(gid_list, contrib_rowid_list)
    #    if _contrib_rowid is None
    #]
    return unassigned_gid_list


@register_ibs_method
# @register_api('/api/contributor/configs/uncontributed/', methods=['GET'])
def get_all_uncontributed_configs(ibs):
    r"""

    RESTful:
        Method: GET
        URL:    /api/contributor/configs/uncontributed/
    """
    config_rowid_list = ibs.get_valid_configids()
    contrib_rowid_list = ibs.get_config_contributor_rowid(config_rowid_list)
    isunassigned_list = [_contrib_rowid is None for _contrib_rowid in contrib_rowid_list]
    unassigned_config_rowid_list = ut.compress(contrib_rowid_list, isunassigned_list)
    #unassigned_config_rowid_list = [
    #    config_rowid
    #    for config_rowid, _contrib_rowid in zip(config_rowid_list, contrib_rowid_list)
    #    if _contrib_rowid is None
    #]
    return unassigned_config_rowid_list


@register_ibs_method
# @register_api('/api/config/contributor/unassigned/', methods=['PUT'])
def set_config_contributor_unassigned(ibs, contrib_rowid):
    r"""

    RESTful:
        Method: PUT
        URL:    /api/config/contributor/unassigned/
    """
    # IS THIS NECESSARY?
    unassigned_config_rowid_list = ibs.get_all_uncontributed_configs()
    contrib_rowid_list = [contrib_rowid] * len(unassigned_config_rowid_list)
    ibs.set_config_contributor_rowid(unassigned_config_rowid_list, contrib_rowid_list)


@register_ibs_method
def ensure_imageset_configs_populated(ibs):
    r"""
    """
    imgsetid_list = ibs.get_valid_imgsetids()
    config_rowid_list = ibs.get_imageset_configid(imgsetid_list)
    isunassigned_list = [config_rowid is None for config_rowid in config_rowid_list]
    unassigned_imgsetid_list = ut.compress(imgsetid_list, isunassigned_list)
    #unassigned_imgsetid_list = [
    #    imgsetid
    #    for imgsetid, config_rowid in zip(imgsetid_list, config_rowid_list)
    #    if config_rowid is None
    #]
    id_iter = ((imgsetid,) for imgsetid in unassigned_imgsetid_list)
    config_rowid_list = list([ibs.MANUAL_CONFIGID]) * len(unassigned_imgsetid_list)
    val_list = ((config_rowid,) for config_rowid in config_rowid_list)
    ibs.db.set(const.IMAGESET_TABLE, ('config_rowid',), val_list, id_iter)


#
# GETTERS::.CONTRIBUTOR_TABLE


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/rowid/uuid/', methods=['GET'])
def get_contributor_rowid_from_uuid(ibs, uuid_list):
    r"""
    Returns:
        contrib_rowid_list (list):  a contributor

    RESTful:
        Method: GET
        URL:    /api/contributor/rowid/uuid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    contrib_rowid_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_rowid',), uuid_list, id_colname='contributor_uuid')
    return contrib_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/rowid/tag/', methods=['GET'])
def get_contributor_rowid_from_tag(ibs, tag_list):
    r"""
    Returns:
        contrib_rowid_list (list):  a contributor

    RESTful:
        Method: GET
        URL:    /api/contributor/rowid/tag/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    contrib_rowid_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_rowid',), tag_list, id_colname='contributor_tag')
    return contrib_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/uuid/', methods=['GET'])
def get_contributor_uuid(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_uuid_list (list):  a contributor's uuid

    RESTful:
        Method: GET
        URL:    /api/contributor/uuid/
    """
    contrib_uuid_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_uuid',), contrib_rowid_list)
    return contrib_uuid_list


#@register_ibs_method
#def get_contributor_tag(ibs, contrib_rowid_list):
#    Returns:
#        contrib_tag_list (list):  a contributor's tag
#    contrib_tag_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_tag',), contrib_rowid_list)
#    return contrib_tag_list

CONFIG_ROWID                 = 'config_rowid'
CONTRIBUTOR_LOCATION_CITY    = 'contributor_location_city'
CONTRIBUTOR_LOCATION_COUNTRY = 'contributor_location_country'
CONTRIBUTOR_LOCATION_STATE   = 'contributor_location_state'
CONTRIBUTOR_LOCATION_ZIP     = 'contributor_location_zip'
CONTRIBUTOR_NAME_FIRST       = 'contributor_name_first'
CONTRIBUTOR_NAME_LAST        = 'contributor_name_last'
CONTRIBUTOR_NOTE             = 'contributor_note'
CONTRIBUTOR_ROWID            = 'contributor_rowid'
CONTRIBUTOR_TAG              = 'contributor_tag'
CONTRIBUTOR_UUID             = 'contributor_uuid'
FEATWEIGHT_ROWID             = 'featweight_rowid'


def testdata_ibs():
    r"""
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    qreq_ = None
    return ibs, qreq_


@register_ibs_method
def _get_all_contributor_rowids(ibs):
    r"""
    all_contributor_rowids <- contributor.get_all_rowids()

    Returns:
        list_ (list): unfiltered contributor_rowids

    TemplateInfo:
        Tider_all_rowids
        tbl = contributor

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_meta_funcs import *  # NOQA
        >>> ibs, qreq_ = testdata_ibs()
        >>> ibs._get_all_contributor_rowids()
    """
    all_contributor_rowids = ibs.db.get_all_rowids(const.CONTRIBUTOR_TABLE)
    return all_contributor_rowids


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/tag/', methods=['GET'])
def get_contributor_tag(ibs, contributor_rowid_list, eager=True, nInput=None):
    r"""
    contributor_tag_list <- contributor.contributor_tag[contributor_rowid_list]

    gets data from the "native" column "contributor_tag" in the "contributor" table

    Args:
        contributor_rowid_list (list):

    Returns:
        list: contributor_tag_list -  a contributor's tag

    TemplateInfo:
        Tgetter_table_column
        col = contributor_tag
        tbl = contributor

    CommandLine:
        python -m ibeis.templates.template_generator --key contributor  --Tcfg with_api_cache=False with_deleters=False

    RESTful:
        Method: GET
        URL:    /api/contributor/tag/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_meta_funcs import *  # NOQA
        >>> ibs, qreq_ = testdata_ibs()
        >>> contributor_rowid_list = ibs._get_all_contributor_rowids()
        >>> eager = True
        >>> contributor_tag_list = ibs.get_contributor_tag(contributor_rowid_list, eager=eager)
        >>> assert len(contributor_rowid_list) == len(contributor_tag_list)
    """
    id_iter = contributor_rowid_list
    colnames = (CONTRIBUTOR_TAG,)
    contributor_tag_list = ibs.db.get(
        const.CONTRIBUTOR_TABLE, colnames, id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return contributor_tag_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/name/first/', methods=['GET'])
def get_contributor_first_name(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_name_first_list (list):  a contributor's first name

    RESTful:
        Method: GET
        URL:    /api/contributor/name/first/
    """
    contrib_name_first_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_name_first',), contrib_rowid_list)
    return contrib_name_first_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/name/last/', methods=['GET'])
def get_contributor_last_name(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_name_last_list (list):  a contributor's last name

    RESTful:
        Method: GET
        URL:    /api/contributor/name/last/
    """
    contrib_name_last_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_name_last',), contrib_rowid_list)
    return contrib_name_last_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/name/string/', methods=['GET'])
def get_contributor_name_string(ibs, contrib_rowid_list, include_tag=False):
    r"""
    Returns:
        contrib_name_list (list):  a contributor's full name

    RESTful:
        Method: GET
        URL:    /api/contributor/name/string/
    """
    first_list = ibs.get_contributor_first_name(contrib_rowid_list)
    last_list = ibs.get_contributor_last_name(contrib_rowid_list)
    if include_tag:
        tag_list = ibs.get_contributor_tag(contrib_rowid_list)
        name_list = zip(first_list, last_list, tag_list)
        contrib_name_list = [
            "%s %s (%s)" % (first, last, tag)
            for first, last, tag in name_list
        ]
    else:
        name_list = zip(first_list, last_list)
        contrib_name_list = [
            "%s %s" % (first, last)
            for first, last in name_list
        ]

    return contrib_name_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/city/', methods=['GET'])
def get_contributor_city(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_city_list (list):  a contributor's location - city

    RESTful:
        Method: GET
        URL:    /api/contributor/city/
    """
    contrib_city_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_city',), contrib_rowid_list)
    return contrib_city_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/state/', methods=['GET'])
def get_contributor_state(ibs, contrib_rowid_list):
    r"""
    Returns:
        list_ (list):  a contributor's location - state

    RESTful:
        Method: GET
        URL:    /api/contributor/state/
    """
    contrib_state_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_state',), contrib_rowid_list)
    return contrib_state_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/country/', methods=['GET'])
def get_contributor_country(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_country_list (list):  a contributor's location - country

    RESTful:
        Method: GET
        URL:    /api/contributor/country/
    """
    contrib_country_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_country',), contrib_rowid_list)
    return contrib_country_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/zip/', methods=['GET'])
def get_contributor_zip(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_zip_list (list):  a contributor's location - zip

    RESTful:
        Method: GET
        URL:    /api/contributor/zip/
    """
    contrib_zip_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_zip',), contrib_rowid_list)
    return contrib_zip_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/location/string/', methods=['GET'])
def get_contributor_location_string(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_list (list):  a contributor's location

    RESTful:
        Method: GET
        URL:    /api/contributor/location/string/
    """
    city_list = ibs.get_contributor_city(contrib_rowid_list)
    state_list = ibs.get_contributor_state(contrib_rowid_list)
    zip_list = ibs.get_contributor_zip(contrib_rowid_list)
    country_list = ibs.get_contributor_country(contrib_rowid_list)
    location_list = zip(city_list, state_list, zip_list, country_list)
    contrib_list = [
        "%s, %s\n%s %s" % (city, state, _zip, country)
        for city, state, _zip, country in location_list
    ]
    return contrib_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/note/', methods=['GET'])
def get_contributor_note(ibs, contrib_rowid_list):
    r"""
    Returns:
        contrib_note_list (list):  a contributor's note

    RESTful:
        Method: GET
        URL:    /api/contributor/note/
    """
    contrib_note_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_note',), contrib_rowid_list)
    return contrib_note_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/config/rowids/', methods=['GET'])
def get_contributor_config_rowids(ibs, contrib_rowid_list):
    r"""
    Returns:
        config_rowid_list (list):  config rowids for a contributor

    RESTful:
        Method: GET
        URL:    /api/contributor/config/rowids/
    """
    config_rowid_list = ibs.db.get(const.CONFIG_TABLE, ('config_rowid',), contrib_rowid_list, id_colname='contributor_rowid', unpack_scalars=False)
    return config_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/imageset/rowids/', methods=['GET'])
def get_contributor_imgsetids(ibs, config_rowid_list):
    r"""
    Returns:
        imgsetid_list (list):  imgsetids for a contributor

    RESTful:
        Method: GET
        URL:    /api/contributor/imageset/rowids/
    """
    imgsetid_list = ibs.db.get(const.IMAGESET_TABLE, ('imageset_rowid',), config_rowid_list, id_colname='config_rowid', unpack_scalars=False)
    return imgsetid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/contributor/image/rowid/', methods=['GET'])
def get_contributor_gids(ibs, contrib_rowid_list):
    r"""
    TODO: Template 1_M reverse getter

    Returns:
        gid_list (list):  gids for a contributor

    RESTful:
        Method: GET
        URL:    /api/contributor/gids/
    """
    gid_list = ibs.db.get(const.IMAGE_TABLE, ('image_rowid',), contrib_rowid_list, id_colname='contributor_rowid', unpack_scalars=False)
    return gid_list


#
# GETTERS::CONFIG_TABLE


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/config/rowid/suffix/', methods=['GET'])
def get_config_rowid_from_suffix(ibs, cfgsuffix_list):
    r"""
    Gets an algorithm configuration as a string

    DEPRICATE

    RESTful:
        Method: GET
        URL:    /api/contributor/config/rowid/suffix/

    Args:
        ibs (IBEISController):  ibeis controller object
        cfgsuffix_list (list):

    Returns:
        list: gid_list

    CommandLine:
        python -m ibeis.control.manual_meta_funcs --exec-get_config_rowid_from_suffix
    """
    # FIXME: This is causing a crash when converting old hotspotter databses.
    # probably because the superkey changed
    # SEE DBSchema.
    # TODO: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    config_rowid_list = ibs.db.get(const.CONFIG_TABLE, ('config_rowid',), cfgsuffix_list, id_colname='config_suffix')

    # executeone always returns a list
    #if config_rowid_list is not None and len(config_rowid_list) == 1:
    #    config_rowid_list = config_rowid_list[0]
    return config_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/config/rowid/', methods=['GET'])
def get_config_contributor_rowid(ibs, config_rowid_list):
    r"""
    Returns:
        cfgsuffix_list (list):  contributor's rowid for algorithm configs

    RESTful:
        Method: GET
        URL:    /api/contributor/config/rowid/
    """
    cfgsuffix_list = ibs.db.get(const.CONFIG_TABLE, ('contributor_rowid',), config_rowid_list)
    return cfgsuffix_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/contributor/config/suffixes/', methods=['GET'])
def get_config_suffixes(ibs, config_rowid_list):
    r"""
    Returns:
        cfgsuffix_list (list):  suffixes for algorithm configs

    RESTful:
        Method: GET
        URL:    /api/contributor/config/suffixes/
    """
    cfgsuffix_list = ibs.db.get(const.CONFIG_TABLE, ('config_suffix',), config_rowid_list)
    return cfgsuffix_list


@register_ibs_method
@accessor_decors.deleter
# @register_api('/api/contributor/', methods=['DELETE'])
def delete_contributors(ibs, contrib_rowid_list):
    r"""
    deletes contributors from the database and all information associated

    RESTful:
        Method: DELETE
        URL:    /api/contributor/
    """
    # TODO: FIXME TESTME
    if not ut.QUIET:
        print('[ibs] deleting %d contributors' % len(contrib_rowid_list))

    config_rowid_list = ut.flatten(ibs.get_contributor_config_rowids(contrib_rowid_list))
    # Delete configs (UNSURE IF THIS IS CORRECT)
    ibs.delete_configs(config_rowid_list)
    # CONTRIBUTORS SHOULD NOT DELETE IMAGES
    # Delete imagesets
    #imgsetid_list = ibs.get_valid_imgsetids()
    #imgsetid_config_list = ibs.get_imageset_configid(imgsetid_list)
    #valid_list = [config in config_rowid_list for config in imgsetid_config_list ]
    #imgsetid_list = ut.compress(imgsetid_list, valid_list)
    #ibs.delete_imagesets(imgsetid_list)
    # Remote image contributors ~~~Delete images~~~~
    gid_list = ut.flatten(ibs.get_contributor_gids(contrib_rowid_list))
    ibs.set_image_contributor_rowid(gid_list, [None] * len(gid_list))
    #ibs.delete_images(gid_list)
    # Delete contributors
    ibs.db.delete_rowids(const.CONTRIBUTOR_TABLE, contrib_rowid_list)


@register_ibs_method
@accessor_decors.ider
def _get_all_contrib_rowids(ibs):
    r"""
    Returns:
        list_ (list):  all unfiltered contrib_rowid (contributor rowid)
    """
    all_contrib_rowids = ibs.db.get_all_rowids(const.CONTRIBUTOR_TABLE)
    return all_contrib_rowids


@register_ibs_method
@accessor_decors.ider
# @register_api('/api/contributor/', methods=['GET'])
def get_valid_contrib_rowids(ibs):
    r"""
    Returns:
        list_ (list):  list of all contributor ids

    Returns:
        list: contrib_rowids_list

    CommandLine:
        python -m ibeis.control.manual_meta_funcs --test-get_valid_contrib_rowids

    RESTful:
        Method: GET
        URL:    /api/contributor/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_meta_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> contrib_rowids_list = get_valid_contrib_rowids(ibs)
        >>> result = str(contrib_rowids_list)
        >>> print(result)
    """
    contrib_rowids_list = ibs._get_all_contrib_rowids()
    return contrib_rowids_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/metadata/value/', methods=['GET'])
def get_metadata_value(ibs, metadata_key_list, db):
    r"""

    RESTful:
        Method: GET
        URL:    /api/metadata/value/
    """
    params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
    where_clause = 'metadata_key=?'
    # list of relationships for each image
    metadata_value_list = db.get_where(const.METADATA_TABLE, ('metadata_value',), params_iter, where_clause, unpack_scalars=True)
    return metadata_value_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/metadata/rowid/key/', methods=['GET'])
def get_metadata_rowid_from_metadata_key(ibs, metadata_key_list, db):
    r"""

    RESTful:
        Method: GET
        URL:    /api/metadata/rowid/key/
    """
    db = db[0]  # Unwrap tuple, required by @accessor_decors.getter_1to1 decorator
    params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
    where_clause = 'metadata_key=?'
    # list of relationships for each image
    metadata_rowid_list = db.get_where(const.METADATA_TABLE, ('metadata_rowid',), params_iter, where_clause, unpack_scalars=True)
    return metadata_rowid_list


@register_api('/api/core/version/', methods=['GET'])
def get_database_version_alias(ibs, db):
    r"""
    Alias: `func:get_database_version`

    RESTful:
        Method: GET
        URL:    /api/core/version/
    """
    return ibs.get_database_version(db)


@register_ibs_method
@accessor_decors.ider
@register_api('/api/core/db/version/', methods=['GET'])
def get_database_version(ibs, db):
    r"""
    Gets the specified database version from the controller

    RESTful:
        Method: GET
        URL:    /api/core/dbversion/
    """
    return db.get_db_version()


@register_ibs_method
@accessor_decors.adder
# @register_api('/api/metadata/', methods=['POST'])
def add_metadata(ibs, metadata_key_list, metadata_value_list, db):
    r"""
    Adds metadata

    Returns:
        metadata_rowid_list (list): metadata rowids

    RESTful:
        Method: POST
        URL:    /api/metadata/
    """
    if ut.VERBOSE:
        print('[ibs] adding %d metadata' % len(metadata_key_list))
    # Add imageset text names to database
    colnames = ['metadata_key', 'metadata_value']
    params_iter = zip(metadata_key_list, metadata_value_list)
    get_rowid_from_superkey = functools.partial(ibs.get_metadata_rowid_from_metadata_key, db=(db,))
    metadata_rowid_list = db.add_cleanly(const.METADATA_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return metadata_rowid_list


@register_ibs_method
@accessor_decors.deleter
# @register_api('/api/config/', methods=['DELETE'])
def delete_configs(ibs, config_rowid_list):
    r"""
    deletes images from the database that belong to fids

    RESTful:
        Method: DELETE
        URL:    /api/config/
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d configs' % len(config_rowid_list))
    ibs.db.delete_rowids(const.CONFIG_TABLE, config_rowid_list)


@register_ibs_method
def _init_config(ibs):
    r"""
    Loads the database's algorithm configuration

    TODO: per-species config
    """
    #####
    # <GENERAL CONFIG>
    config_fpath = ut.unixjoin(ibs.get_dbdir(), 'general_config.cPkl')
    try:
        general_config = ut.load_cPkl(config_fpath, verbose=ut.VERBOSE)
    except IOError:
        general_config = {}
    current_species = general_config.get('current_species', None)
    if ut.VERBOSE and ut.NOT_QUIET:
        print('[_init_config] general_config.current_species = %r' % (current_species,))
    # </GENERAL CONFIG>
    #####
    species_list = ibs.get_database_species()
    if current_species is None:
        species_list = ibs.get_database_species()
        current_species = species_list[0] if len(species_list) == 1 else None
    cfgname = 'cfg' if current_species is None else current_species
    if ut.VERBOSE and ut.NOT_QUIET:
        print('[_init_config] Loading database with species_list = %r ' % (species_list,))
        print('[_init_config] Using cfgname=%r' % (cfgname,))
    # try to be intelligent about the default speceis
    ibs._load_named_config(cfgname)


@register_ibs_method
def _init_burned_in_species(ibs):
    # Add missing "required" species
    species_nice_list = [
        'Giraffe (Masai)',
        'Giraffe (Reticulated)',
        'Other',
        'Zebra (Grevy\'s)',
        'Zebra (Hybrid)',
        'Zebra (Plains)',
    ]
    species_text_list = [
        'giraffe_masai',
        'giraffe_reticulated',
        'other',
        'zebra_grevys',
        'zebra_hybrid',
        'zebra_plains',
    ]
    species_code_list = [
        'GIRM',
        'GIR',
        'OTHER',
        'GZ',
        'HZ',
        'PZ',
    ]
    ibs.add_species(species_nice_list, species_text_list, species_code_list)
    print('[_init_burned_in_species] Burned in mising species...')


@register_ibs_method
def _load_named_config(ibs, cfgname=None):
    r"""
    """
    # TODO: update cfgs between versions
    # Try to load previous config otherwise default
    #use_config_cache = not (ut.is_developer() and not ut.get_argflag(('--nocache-pref',)))
    use_config_cache = not ut.get_argflag(('--nocache-pref',))
    ibs.cfg = Config.load_named_config(cfgname, ibs.get_dbdir(), use_config_cache)
    ibs.reset_table_cache()


@register_ibs_method
def _default_config(ibs, cfgname=None, new=True):
    r"""
    Resets the databases's algorithm configuration

    Args:
        ibs (IBEISController):  ibeis controller object
        cfgname (None):

    CommandLine:
        python -m ibeis.control.manual_meta_funcs --test-_default_config

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.Config import *  # NOQA
        >>> from ibeis.control.manual_meta_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> assert not hasattr(ibs.cfg.query_cfg.flann_cfg, 'badparam')
        >>> ibs.cfg.query_cfg.flann_cfg.badparam = True
        >>> assert ibs.cfg.query_cfg.flann_cfg.cb_index == .4
        >>> ibs.cfg.query_cfg.flann_cfg.cb_index = .5
        >>> assert ibs.cfg.query_cfg.flann_cfg.cb_index == .5
        >>> assert hasattr(ibs.cfg.query_cfg.flann_cfg, 'badparam')
        >>> # bad params is not initially removed but when you specify new it is
        >>> ibs._default_config(new=False)
        >>> assert ibs.cfg.query_cfg.flann_cfg.cb_index == .4
        >>> assert hasattr(ibs.cfg.query_cfg.flann_cfg, 'badparam')
        >>> # bad param should be removed when config is defaulted
        >>> ibs._default_config(new=True)
        >>> assert not hasattr(ibs.cfg.query_cfg.flann_cfg, 'badparam')

    cfg = ibs.cfg
    """
    #species_list = ibs.get_database_species()
    #if len(species_list) == 1:
    #    # try to be intelligent about the default speceis
    #    cfgname = species_list[0]
    ibs.cfg = Config._default_config(ibs.cfg, cfgname, new=new)
    ibs.reset_table_cache()


#@register_ibs_method
#@accessor_decors.default_decorator
#@register_api('/api/query/cfg/', methods=['PUT'])
#def set_query_cfg(ibs, query_cfg):
#    r"""
#    DEPRICATE

#    RESTful:
#        Method: PUT
#        URL:    /api/query/cfg/
#    """
#    Config.set_query_cfg(ibs.cfg, query_cfg)
#    ibs.reset_table_cache()
#    #if ibs.qreq is not None:
#    #    ibs.qreq.set_cfg(query_cfg)


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/query/cfg/', methods=['PUT'])
def update_query_cfg(ibs, **kwargs):
    r"""
    Updates query config only. Configs needs a restructure very badly
    DEPRICATE

    RESTful:
        Method: PUT
        URL:    /api/query/cfg/
    """
    Config.update_query_config(ibs.cfg, **kwargs)
    ibs.reset_table_cache()


@register_ibs_method
@accessor_decors.ider
# @register_api('/api/config/', methods=['GET'])
def get_valid_configids(ibs):
    r"""

    RESTful:
        Method: GET
        URL:    /api/config/
    """
    config_rowid_list = ibs.db.get_all_rowids(const.CONFIG_TABLE)
    return config_rowid_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.control.manual_meta_funcs
        python -m ibeis.control.manual_meta_funcs --allexamples
        python -m ibeis.control.manual_meta_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
