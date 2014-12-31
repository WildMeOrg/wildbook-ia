"""
controller functions for contributors, versions, configs, and other metadata
"""
from __future__ import absolute_import, division, print_function
import uuid
import six  # NOQA
#from os.path import join
import functools
from six.moves import range
from ibeis import constants as const
from ibeis.control.accessor_decors import (
    adder, deleter, setter, getter_1to1, default_decorator, ider)
import utool as ut
from ibeis.model import Config
#from ibeis import ibsfuncs
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_meta]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
@adder
def add_contributors(ibs, tag_list, uuid_list=None, name_first_list=None, name_last_list=None,
                     loc_city_list=None, loc_state_list=None,
                     loc_country_list=None, loc_zip_list=None,
                     notes_list=None):
    """
    Adds a list of contributors.

    Returns:
        contrib_id_list (list): contributor rowids
    """
    import datetime
    def _valid_zip(_zip, default='00000'):
        _zip = str(_zip)
        if len(_zip) == 5 and _zip.isdigit():
            return _zip
        return default

    if ut.VERBOSE:
        print('[ibs] adding %d encounters' % len(tag_list))
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
        uuid_list = [uuid.uuid4() for _ in range(len(tag_list))]

    colnames = ['contributor_uuid', 'contributor_tag', 'contributor_name_first',
                'contributor_name_last', 'contributor_location_city',
                'contributor_location_state', 'contributor_location_country',
                'contributor_location_zip', 'contributor_note']
    params_iter = zip(uuid_list, tag_list, name_first_list,
                      name_last_list, loc_city_list, loc_state_list,
                      loc_country_list, loc_zip_list, notes_list)

    get_rowid_from_superkey = ibs.get_contributor_rowid_from_uuid
    contrib_id_list = ibs.db.add_cleanly(const.CONTRIBUTOR_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return contrib_id_list


@register_ibs_method
@adder
def add_version(ibs, versiontext_list):
    """ Adds an algorithm / actor configuration as a string """
    # FIXME: Configs are still handled poorly
    params_iter = ((versiontext,) for versiontext in versiontext_list)
    get_rowid_from_superkey = ibs.get_version_rowid_from_superkey
    versionid_list = ibs.db.add_cleanly(const.VERSIONS_TABLE, ('version_text',),
                                        params_iter, get_rowid_from_superkey)
    return versionid_list


@register_ibs_method
@adder
def add_config(ibs, cfgsuffix_list, contrib_rowid_list=None):
    """ Adds an algorithm / actor configuration as a string """
    # FIXME: Configs are still handled poorly. This function is an ensure
    params_iter = ((suffix,) for suffix in cfgsuffix_list)
    get_rowid_from_superkey = ibs.get_config_rowid_from_suffix
    config_rowid_list = ibs.db.add_cleanly(const.CONFIG_TABLE, ('config_suffix',),
                                           params_iter, get_rowid_from_superkey)
    if contrib_rowid_list is not None:
        ibs.set_config_contributor_rowid(config_rowid_list, contrib_rowid_list)
    return config_rowid_list

# SETTERS::METADATA


@register_ibs_method
@setter
def set_metadata_value(ibs, metadata_key_list, metadata_value_list, db):
    """ Sets metadata key, value pairs
    """
    db = db[0]  # Unwrap tuple, required by @setter decorator
    metadata_rowid_list = ibs.get_metadata_rowid_from_metadata_key(metadata_key_list, db)
    id_iter = ((metadata_rowid,) for metadata_rowid in metadata_rowid_list)
    val_list = ((metadata_value,) for metadata_value in metadata_value_list)
    db.set(const.METADATA_TABLE, ('metadata_value',), val_list, id_iter)


@register_ibs_method
def set_database_version(ibs, db, version):
    """ Sets the specified database's version from the controller
    """
    db.set_db_version(version)

# SETTERS::CONTRIBUTORS


@register_ibs_method
def set_config_contributor_rowid(ibs, config_rowid_list, contrib_rowid_list):
    """ Sets the config's contributor rowid """
    id_iter = ((config_rowid,) for config_rowid in config_rowid_list)
    val_list = ((contrib_rowid,) for contrib_rowid in contrib_rowid_list)
    ibs.db.set(const.CONFIG_TABLE, ('contributor_rowid',), val_list, id_iter)


@register_ibs_method
def set_config_contributor_unassigned(ibs, contrib_rowid):
    config_rowid_list = ibs.get_valid_configids()
    contrib_rowid_list = ibs.get_config_contributor_rowid(config_rowid_list)
    unassigned_config_rowid_list = [
        config_rowid
        for config_rowid, _contrib_rowid in zip(config_rowid_list, contrib_rowid_list)
        if _contrib_rowid is None
    ]
    contrib_rowid_list = list([contrib_rowid]) * len(unassigned_config_rowid_list)
    ibs.set_config_contributor_rowid(config_rowid_list, contrib_rowid_list)


@register_ibs_method
def set_image_contributor_unassigned(ibs, contrib_rowid):
    gid_list = ibs.get_valid_gids()
    contrib_rowid_list = ibs.get_image_contributor_rowid(gid_list)
    unassigned_gid_list = [
        gid
        for gid, _contrib_rowid in zip(gid_list, contrib_rowid_list)
        if _contrib_rowid is None
    ]
    contrib_rowid_list = list([contrib_rowid]) * len(unassigned_gid_list)
    ibs.set_image_contributor_rowid(unassigned_gid_list, contrib_rowid_list)


@register_ibs_method
def ensure_encounter_configs_populated(ibs):
    eid_list = ibs.get_valid_eids()
    config_rowid_list = ibs.get_encounter_configid(eid_list)
    unassigned_eid_list = [
        eid
        for eid, config_rowid in zip(eid_list, config_rowid_list)
        if config_rowid is None
    ]
    id_iter = ((eid,) for eid in unassigned_eid_list)
    config_rowid_list = list([ibs.MANUAL_CONFIGID]) * len(unassigned_eid_list)
    val_list = ((config_rowid,) for config_rowid in config_rowid_list)
    ibs.db.set(const.ENCOUNTER_TABLE, ('config_rowid',), val_list, id_iter)


#
# GETTERS::.CONTRIBUTOR_TABLE


@register_ibs_method
@getter_1to1
def get_contributor_rowid_from_uuid(ibs, tag_list):
    """
    Returns:
        contrib_rowid_list (list):  a contributor """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    contrib_rowid_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_rowid',), tag_list, id_colname='contributor_uuid')
    return contrib_rowid_list


@register_ibs_method
@getter_1to1
def get_contributor_uuid(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_uuid_list (list):  a contributor's uuid """
    contrib_uuid_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_uuid',), contrib_rowid_list)
    return contrib_uuid_list


@register_ibs_method
@getter_1to1
def get_contributor_tag(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_tag_list (list):  a contributor's tag """
    contrib_tag_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_tag',), contrib_rowid_list)
    return contrib_tag_list


@register_ibs_method
@getter_1to1
def get_contributor_first_name(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_name_first_list (list):  a contributor's first name """
    contrib_name_first_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_name_first',), contrib_rowid_list)
    return contrib_name_first_list


@register_ibs_method
@getter_1to1
def get_contributor_last_name(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_name_last_list (list):  a contributor's last name """
    contrib_name_last_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_name_last',), contrib_rowid_list)
    return contrib_name_last_list


@register_ibs_method
@getter_1to1
def get_contributor_name_string(ibs, contrib_rowid_list, include_tag=False):
    """
    Returns:
        contrib_name_list (list):  a contributor's full name """
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
@getter_1to1
def get_contributor_city(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_city_list (list):  a contributor's location - city """
    contrib_city_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_city',), contrib_rowid_list)
    return contrib_city_list


@register_ibs_method
@getter_1to1
def get_contributor_state(ibs, contrib_rowid_list):
    """
    Returns:
        list_ (list):  a contributor's location - state """
    contrib_state_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_state',), contrib_rowid_list)
    return contrib_state_list


@register_ibs_method
@getter_1to1
def get_contributor_country(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_country_list (list):  a contributor's location - country """
    contrib_country_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_country',), contrib_rowid_list)
    return contrib_country_list


@register_ibs_method
@getter_1to1
def get_contributor_zip(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_zip_list (list):  a contributor's location - zip """
    contrib_zip_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_location_zip',), contrib_rowid_list)
    return contrib_zip_list


@register_ibs_method
@getter_1to1
def get_contributor_location_string(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_list (list):  a contributor's location """
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
@getter_1to1
def get_contributor_note(ibs, contrib_rowid_list):
    """
    Returns:
        contrib_note_list (list):  a contributor's note """
    contrib_note_list = ibs.db.get(const.CONTRIBUTOR_TABLE, ('contributor_note',), contrib_rowid_list)
    return contrib_note_list


@register_ibs_method
@getter_1to1
def get_contributor_config_rowids(ibs, contrib_rowid_list):
    """
    Returns:
        config_rowid_list (list):  config rowids for a contributor """
    config_rowid_list = ibs.db.get(const.CONFIG_TABLE, ('config_rowid',), contrib_rowid_list, id_colname='contributor_rowid', unpack_scalars=False)
    return config_rowid_list


@register_ibs_method
@getter_1to1
def get_contributor_eids(ibs, config_rowid_list):
    """
    Returns:
        eid_list (list):  eids for a contributor """
    eid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_rowid',), config_rowid_list, id_colname='config_rowid', unpack_scalars=False)
    return eid_list


@register_ibs_method
@getter_1to1
def get_contributor_gids(ibs, contrib_rowid_list):
    """
    Returns:
        gid_list (list):  eids for a contributor """
    gid_list = ibs.db.get(const.IMAGE_TABLE, ('image_rowid',), contrib_rowid_list, id_colname='contributor_rowid', unpack_scalars=False)
    return gid_list


#
# GETTERS::CONFIG_TABLE


@register_ibs_method
@getter_1to1
def get_config_rowid_from_suffix(ibs, cfgsuffix_list):
    """
    Gets an algorithm configuration as a string
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
def ensure_config_rowid_from_suffix(ibs, cfgsuffix_list):
    """
    Alias for adder
    """
    # FIXME: cfgsuffix should be renamed cfgstr? cfgtext?
    return ibs.add_config(cfgsuffix_list)


@register_ibs_method
@getter_1to1
def get_config_contributor_rowid(ibs, config_rowid_list):
    """
    Returns:
        cfgsuffix_list (list):  contributor's rowid for algorithm configs """
    cfgsuffix_list = ibs.db.get(const.CONFIG_TABLE, ('contributor_rowid',), config_rowid_list)
    return cfgsuffix_list


@register_ibs_method
@getter_1to1
def get_config_suffixes(ibs, config_rowid_list):
    """
    Returns:
        cfgsuffix_list (list):  suffixes for algorithm configs """
    cfgsuffix_list = ibs.db.get(const.CONFIG_TABLE, ('config_suffix',), config_rowid_list)
    return cfgsuffix_list


@register_ibs_method
@deleter
def delete_contributors(ibs, contrib_rowid_list):
    """ deletes contributors from the database and all information associated"""
    if ut.VERBOSE:
        print('[ibs] deleting %d contributors' % len(contrib_rowid_list))
    config_rowid_list = ut.flatten(ibs.get_contributor_config_rowids(contrib_rowid_list))
    # Delete configs
    ibs.delete_configs(config_rowid_list)
    # Delete encounters
    eid_list = ibs.get_valid_eids()
    eid_config_list = ibs.get_encounter_configid(eid_list)
    valid_list = [ config in config_rowid_list for config in eid_config_list ]
    eid_list = ut.filter_items(eid_list, valid_list)
    ibs.delete_encounters(eid_list)
    # Delete images
    gid_list = ut.flatten(ibs.get_contributor_gids(contrib_rowid_list))
    ibs.delete_images(gid_list)
    # Delete contributors
    ibs.db.delete_rowids(const.CONTRIBUTOR_TABLE, contrib_rowid_list)


@register_ibs_method
@ider
def _get_all_contrib_rowids(ibs):
    """
    Returns:
        list_ (list):  all unfiltered contrib_rowid (contributor rowid) """
    all_contrib_rowids = ibs.db.get_all_rowids(const.CONTRIBUTOR_TABLE)
    return all_contrib_rowids


@register_ibs_method
@ider
def get_valid_contrib_rowids(ibs):
    """
    Returns:
        list_ (list):  list of all contributor ids """
    contrib_rowids_list = ibs._get_all_contrib_rowids()
    return contrib_rowids_list


@register_ibs_method
@getter_1to1
def get_metadata_value(ibs, metadata_key_list, db):
    params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
    where_clause = 'metadata_key=?'
    # list of relationships for each image
    metadata_value_list = db.get_where(const.METADATA_TABLE, ('metadata_value',), params_iter, where_clause, unpack_scalars=True)
    return metadata_value_list


@register_ibs_method
@getter_1to1
def get_metadata_rowid_from_metadata_key(ibs, metadata_key_list, db):
    db = db[0]  # Unwrap tuple, required by @getter_1to1 decorator
    params_iter = ((metadata_key,) for metadata_key in metadata_key_list)
    where_clause = 'metadata_key=?'
    # list of relationships for each image
    metadata_rowid_list = db.get_where(const.METADATA_TABLE, ('metadata_rowid',), params_iter, where_clause, unpack_scalars=True)
    return metadata_rowid_list


@register_ibs_method
@ider
def get_database_version(ibs, db):
    ''' Gets the specified database version from the controller '''
    return db.get_db_version()


@register_ibs_method
@adder
def add_metadata(ibs, metadata_key_list, metadata_value_list, db):
    """
    Adds metadata

    Returns:
        metadata_rowid_list (list): metadata rowids
    """
    if ut.VERBOSE:
        print('[ibs] adding %d metadata' % len(metadata_key_list))
    # Add encounter text names to database
    colnames = ['metadata_key', 'metadata_value']
    params_iter = zip(metadata_key_list, metadata_value_list)
    get_rowid_from_superkey = functools.partial(ibs.get_metadata_rowid_from_metadata_key, db=(db,))
    metadata_rowid_list = db.add_cleanly(const.METADATA_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return metadata_rowid_list


@register_ibs_method
@deleter
def delete_configs(ibs, config_rowid_list):
    """ deletes images from the database that belong to fids"""
    if ut.VERBOSE:
        print('[ibs] deleting %d configs' % len(config_rowid_list))
    ibs.db.delete_rowids(const.CONFIG_TABLE, config_rowid_list)


@register_ibs_method
def _init_config(ibs):
    """
    Loads the database's algorithm configuration

    TODO: per-species config
    """
    species_list = ibs.get_database_species()
    # try to be intelligent about the default speceis
    cfgname = species_list[0] if len(species_list) == 1 else 'cfg'
    ibs._load_named_config(cfgname)


@register_ibs_method
def _load_named_config(ibs, cfgname=None):
    # TODO: update cfgs between versions
    # Try to load previous config otherwise default
    use_config_cache = not (ut.is_developer() or ut.get_argflag(('--nocache-pref',)))
    ibs.cfg = Config.load_named_config(cfgname, ibs.get_dbdir(), use_config_cache)
    ibs.reset_table_cache()


@register_ibs_method
def _default_config(ibs, cfgname=None, new=True):
    """
    Resets the databases's algorithm configuration

    Args:
        ibs (IBEISController):  ibeis controller object
        cfgname (None):

    CommandLine:
        python -m ibeis.control.manual_meta_funcs --test-_default_config

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
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


@register_ibs_method
@default_decorator
def set_query_cfg(ibs, query_cfg):
    Config.set_query_cfg(ibs.cfg, query_cfg)
    ibs.reset_table_cache()
    #if ibs.qreq is not None:
    #    ibs.qreq.set_cfg(query_cfg)


@register_ibs_method
@default_decorator
def update_query_cfg(ibs, **kwargs):
    """ Updates query config only. Configs needs a restructure very badly """
    Config.update_query_config(ibs.cfg, **kwargs)
    ibs.reset_table_cache()


@register_ibs_method
@ider
def get_valid_configids(ibs):
    config_rowid_list = ibs.db.get_all_rowids(const.CONFIG_TABLE)
    return config_rowid_list


@register_ibs_method
@default_decorator
def get_query_config_rowid(ibs):
    """ # FIXME: Configs are still handled poorly """
    query_cfg_suffix = ibs.cfg.query_cfg.get_cfgstr()
    query_cfg_rowid = ibs.add_config(query_cfg_suffix)
    return query_cfg_rowid

#@default_decorator
#def get_qreq_rowid(ibs):
#    """ # FIXME: Configs are still handled poorly """
#    assert ibs.qres is not None
#    qreq_rowid = ibs.qreq.get_cfgstr()
#    return qreq_rowid


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_meta_funcs
        python -m ibeis.control.manual_meta_funcs --allexamples
        python -m ibeis.control.manual_meta_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
