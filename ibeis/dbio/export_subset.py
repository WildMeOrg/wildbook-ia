#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database

TODO:
    NEED TO UPDATE FILE TO USE THE NEW NAME AND SPECIES TABLE NOW THAT THEY
    ARE NO LONGER LBLANNOTS
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from collections import namedtuple
import utool
import utool as ut
import datetime
# import ibeis
import inspect
from ibeis import ibsfuncs  # NOQA
from ibeis import constants as const
# from ibeis.constants import (AL_RELATION_TABLE, ANNOTATION_TABLE, CONFIG_TABLE,
#                              CONTRIBUTOR_TABLE, EG_RELATION_TABLE, ENCOUNTER_TABLE,
#                              GL_RELATION_TABLE, IMAGE_TABLE, LBLANNOT_TABLE,
#                              LBLIMAGE_TABLE, __STR__)
# from vtool import geometry

# Transfer data structures could become classes.
TransferData = namedtuple(
    'TransferData', (
        'transfer_database_name',
        'transfer_database_source',
        'transfer_export_time',
        'transfer_export_location_city',
        'transfer_export_location_state',
        'transfer_export_location_zip',
        'transfer_export_location_country',
        'contributor_td_list',
        'name_td',
        'species_td',
    ))

CONTRIBUTOR_TransferData = namedtuple(
    'CONTRIBUTOR_TransferData', (
        'contributor_uuid',
        'contributor_tag',
        'contributor_name_first',
        'contributor_name_last',
        'contributor_location_city',
        'contributor_location_state',
        'contributor_location_country',
        'contributor_location_zip',
        'contributor_note',
        'config_td',
        'encounter_td',
        'image_td',
    ))

NAME_TransferData = namedtuple(
    'NAME_TransferData', (
        'name_uuid_list',
        'name_text_list',
        'name_note_list',
    ))

SPECIES_TransferData = namedtuple(
    'SPECIES_TransferData', (
        'species_uuid_list',
        'species_text_list',
        'species_note_list',
    ))

CONFIG_TransferData = namedtuple(
    'CONFIG_TransferData', (
        'config_suffixes_list',
    ))

ENCOUNTER_TransferData = namedtuple(
    'ENCOUNTER_TransferData', (
        'config_INDEX_list',
        'encounter_uuid_list',
        'encounter_text_list',
        'encoutner_note_list',
    ))

IMAGE_TransferData = namedtuple(
    'IMAGE_TransferData', (
        'encounter_INDEXs_list',
        'image_path_list',
        'image_uuid_list',
        'image_ext_list',
        'image_original_name_list',
        'image_width_list',
        'image_height_list',
        'image_time_posix_list',
        'image_gps_lat_list',
        'image_gps_lon_list',
        'image_toggle_enabled_list',
        'image_toggle_reviewed_list',
        'image_note_list',
        'lblimage_td_list',
        'annotation_td_list',
    ))

ANNOTATION_TransferData = namedtuple(
    'ANNOTATION_TransferData', (
        'annot_parent_INDEX_list',
        'annot_uuid_list',
        'annot_theta_list',
        'annot_verts_list',
        'annot_viewpoint_list',
        'annot_detection_confidence_list',
        'annot_exemplar_flag_list',
        'annot_visual_uuid_list',
        'annot_semantic_uuid_list',
        'annot_note_list',
        'annot_name_INDEX_list',
        'annot_species_INDEX_list',
        'lblannot_td_list',
    ))

LBLIMAGE_TransferData = namedtuple(
    'LBLIMAGE_TransferData', (
        'config_INDEX_list',
        'glr_confidence_list',
        'lblimage_uuid_list',
        'lbltype_text_list',
        'lblimage_value_list',
        'lblimage_note_list',
    ))

LBLANNOT_TransferData = namedtuple(
    'LBLANNOT_TransferData', (
        'config_INDEX_list',
        'alr_confidence_list',
        'lblannot_uuid_list',
        'lbltype_text_list',
        'lblannot_value_list',
        'lblannot_note_list',
    ))


#############################
#############################
#############################


def _index(value, list_, warning=True):
    if value in list_:
        return list_.index(value)
    else:
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        if warning:
            print("[export_subset] WARNING: value (%r) not in list: %r (%r)" % (value, list_, calframe[1][3]))
        return None


def geo_locate(default="Unknown", timeout=1):
    try:
        import urllib2
        import json
        req = urllib2.Request('http://freegeoip.net/json/', headers={ 'User-Agent': 'Mozilla/5.0' })
        f = urllib2.urlopen(req, timeout=timeout)
        json_string = f.read()
        f.close()
        location = json.loads(json_string)
        location_city    = location['city']
        location_state   = location['region_name']
        location_country = location['country_name']
        location_zip     = location['zipcode']
        success = True
    except:
        success = False
        location_city    = default
        location_state   = default
        location_zip     = default
        location_country = default
    return success, location_city, location_state, location_country, location_zip


def ensure_contributor_rowids(ibs, user_prompt=False):
    # TODO: Alter this check to support merging databases with more than one contributor, but none assigned to the manual config
    contrib_rowid_list = ibs.get_valid_contrib_rowids()
    if len(contrib_rowid_list) == 0:
        print("[collect_transfer_data] There is no contributor for this database.  Add a contributor to continue merge.")
        name_first = ibs.get_dbname()
        name_last = utool.get_computer_name() + ":" + utool.get_user_name() + ":" + ibs.get_dbdir()
        print("[collect_transfer_data] Contributor default first name: %s" % (name_first, ))
        print("[collect_transfer_data] Contributor default last name:  %s" % (name_last, ))
        if user_prompt:
            name_first = raw_input("\n[collect_transfer_data] Change first name (Enter to use default): ")
            name_last  = raw_input("\n[collect_transfer_data] Change last name (Enter to use default): ")

        success, location_city, location_state, location_country, location_zip = geo_locate()

        if success:
            print("\n[collect_transfer_data] Your location was be determined automatically.")
            print("[collect_transfer_data] Contributor default city: %s"    % (location_city, ))
            print("[collect_transfer_data] Contributor default state: %s"   % (location_state, ))
            print("[collect_transfer_data] Contributor default zip: %s"     % (location_country, ))
            print("[collect_transfer_data] Contributor default country: %s" % (location_zip, ))
            if user_prompt:
                location_city    = raw_input("\n[collect_transfer_data] Change default location city (Enter to use default): ")
                location_state   = raw_input("\n[collect_transfer_data] Change default location state (Enter to use default): ")
                location_zip     = raw_input("\n[collect_transfer_data] Change default location zip (Enter to use default): ")
                location_country = raw_input("\n[collect_transfer_data] Change default location country (Enter to use default): ")
        else:
            if user_prompt:
                print("\n")
            print("[collect_transfer_data] Your location could not be determined automatically.")
            if user_prompt:
                location_city    = raw_input("[collect_transfer_data] Enter your location city (Enter to skip): ")
                location_state   = raw_input("[collect_transfer_data] Enter your location state (Enter to skip): ")
                location_zip     = raw_input("[collect_transfer_data] Enter your location zip (Enter to skip): ")
                location_country = raw_input("[collect_transfer_data] Enter your location country (Enter to skip): ")
            else:
                location_city    = ''
                location_state   = ''
                location_zip     = ''
                location_country = ''

        tag = "::".join([name_first, name_last, location_city, location_state, location_zip, location_country])
        contrib_rowid = ibs.add_contributors(
            [tag], name_first_list=[name_first],
            name_last_list=[name_last], loc_city_list=[location_city],
            loc_state_list=[location_state], loc_country_list=[location_country],
            loc_zip_list=[location_zip])
        contrib_rowid = contrib_rowid[0]
        print('[collect_transfer_data] New contributor\'s contrib_rowid: %s' % (contrib_rowid,))
        ibs.set_config_contributor_unassigned(contrib_rowid)
        ibs.set_image_contributor_unassigned(contrib_rowid)
        ibs.ensure_encounter_configs_populated()
    return ibs.get_valid_contrib_rowids()


#############################
#############################
#############################


def export_transfer_data(ibs_src, gid_list=None, user_prompt=False):
    """
    Packs all the data you are going to transfer from ibs_src
    info the transfer_data named tuple.

    Example:
        >>> # SLOW_DOCTEST
        >>> import ibeis
        >>> from ibeis.dbio import export_subset    # NOQA
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> ibs_src = ibeis.opendb(dbdir='/raid/work2/Turk/PZ_Master')
        >>> bulk_conflict_resolution = 'ignore'
        >>> gid_list = ibs.get_valid_gids()[::10]
        >>> user_prompt = False
    """
    if gid_list is None:
        nid_list = ibs_src._get_all_known_name_rowids()
        species_rowid_list = ibs_src._get_all_species_rowids()
    else:
        nid_list = list(set(ut.flatten(ibs_src.get_image_nids(gid_list))))
        species_rowid_list = ibs_src._get_all_species_rowids()
        #species_rowid_list = list(set(ut.flatten(ibs_src.species_rowid_list(gid_list))))
    # Create Name TransferData
    name_td = export_name_transfer_data(ibs_src, nid_list)  # NOQA
    # Create Species TranferData
    species_td = export_species_transfer_data(ibs_src, species_rowid_list)   # NOQA
    # Create Contributor TranferData
    contrib_rowid_list = ensure_contributor_rowids(ibs_src, user_prompt=user_prompt)
    if gid_list is not None:
        contrib_rowid_list = list(set(ibs_src.get_image_contributor_rowid(gid_list)))
    assert len(contrib_rowid_list) > 0, "There must be at least one contributor to merge"
    contributor_td_list = [
        export_contributor_transfer_data(ibs_src, contrib_rowid, nid_list,
                                         species_rowid_list, valid_gid_list=gid_list)
        for contrib_rowid in contrib_rowid_list
    ]
    # Geolocate and create database's TransferData object
    success, location_city, location_state, location_country, location_zip = geo_locate()
    td = TransferData(
        ibs_src.dbname,
        utool.get_computer_name() + ":" + utool.get_user_name() + ":" + ibs_src.workdir,
        "%s" % (datetime.datetime.now()),
        location_city,
        location_state,
        location_zip,
        location_country,
        contributor_td_list,
        name_td,
        species_td
    )
    return td


def export_contributor_transfer_data(ibs_src, contributor_rowid, nid_list,
                                     species_rowid_list, valid_gid_list=None):
    """

    CommandLine:
        python -m ibeis.dbio.export_subset --test-export_contributor_transfer_data

    Example:
        >>> # SLOW_DOCTEST
        >>> import ibeis
        >>> from ibeis.dbio import export_subset    # NOQA
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> ibs_src = ibeis.opendb(dbdir='/raid/work2/Turk/PZ_Master')
        >>> bulk_conflict_resolution = 'ignore'
        >>> gid_list = ibs_src.get_valid_gids()[::10]
        >>> user_prompt = False
        >>> nid_list = list(set(ut.flatten(ibs_src.get_image_nids(gid_list))))
        >>> species_rowid_list = ibs_src._get_all_species_rowids()
        >>> contrib_rowid_list = list(set(ibs_src.get_image_contributor_rowid(gid_list)))
        >>> valid_gid_list = gid_list
        >>> contributor_rowid = contrib_rowid_list[0]
        >>> contrib_td = export_contributor_transfer_data(ibs_src, contributor_rowid, nid_list, species_rowid_list, valid_gid_list=valid_gid_list)

    Dev::
        ibs = ibs_src
        configid_list = ibs.get_valid_configids()
        config_suffix_list = ibs.get_config_suffixes(configid_list)
        print(ut.list_str(list(zip(configid_list, config_suffix_list))))

        eid_list = ibs.get_valid_eids()
        enc_config_rowid_list = ibs.get_encounter_configid(eid_list)
        enc_suffix_list = ibs.get_config_suffixes(config_rowid_list)
        print(ut.list_str(list(zip(enc_config_rowid_list, enc_suffix_list))))

    """
    # Get configs
    #config_rowid_list = ibs_src.get_contributor_config_rowids(contributor_rowid)
    # Hack around config-less encounters
    config_rowid_list = ibs_src.get_valid_configids()
    config_td = export_config_transfer_data(ibs_src, config_rowid_list)
    # Get encounters
    #eid_list = ibs_src.get_valid_eids()
    eid_list = utool.flatten( ibs_src.get_contributor_eids(config_rowid_list) )
    encounter_td = export_encounter_transfer_data(ibs_src, eid_list, config_rowid_list)
    # Get images
    gid_list = ibs_src.get_contributor_gids(contributor_rowid)
    if valid_gid_list is not None:
        isvalid_list = [ gid in valid_gid_list for gid in gid_list ]
        gid_list = ut.filter_items(gid_list, isvalid_list)
    image_td = export_image_transfer_data(ibs_src, gid_list, config_rowid_list, eid_list,
                                          nid_list, species_rowid_list)
    # Create Contributor TransferData
    contributor_td = CONTRIBUTOR_TransferData(
        ibs_src.get_contributor_uuid(contributor_rowid),
        ibs_src.get_contributor_tag(contributor_rowid),
        ibs_src.get_contributor_first_name(contributor_rowid),
        ibs_src.get_contributor_last_name(contributor_rowid),
        ibs_src.get_contributor_city(contributor_rowid),
        ibs_src.get_contributor_state(contributor_rowid),
        ibs_src.get_contributor_country(contributor_rowid),
        ibs_src.get_contributor_zip(contributor_rowid),
        ibs_src.get_contributor_note(contributor_rowid),
        config_td,
        encounter_td,
        image_td
    )
    return contributor_td


def export_name_transfer_data(ibs_src, nid_list):
    # Create Name TransferData
    name_td = NAME_TransferData(
        ibs_src.get_name_uuids(nid_list),
        ibs_src.get_name_texts(nid_list),
        ibs_src.get_name_notes(nid_list)
    )
    return name_td


def export_species_transfer_data(ibs_src, species_rowid_list):
    # Create Species TransferData
    species_td = SPECIES_TransferData(
        ibs_src.get_species_uuids(species_rowid_list),
        ibs_src.get_species_texts(species_rowid_list),
        ibs_src.get_species_notes(species_rowid_list)
    )
    return species_td


def export_config_transfer_data(ibs_src, config_rowid_list):
    if config_rowid_list is None or len(config_rowid_list) == 0:
        return None
    # Create Config TransferData
    config_td = CONFIG_TransferData(
        ibs_src.get_config_suffixes(config_rowid_list)
    )
    return config_td


def export_encounter_transfer_data(ibs_src, eid_list, config_rowid_list):
    if eid_list is None or len(eid_list) == 0:
        return None
    # Get encounter data
    config_INDEX_list = [ _index(ibs_src.get_encounter_configid(eid), config_rowid_list) for eid in eid_list ]
    # Create Encounter TransferData
    encounter_td = ENCOUNTER_TransferData(
        config_INDEX_list,
        ibs_src.get_encounter_uuid(eid_list),
        ibs_src.get_encounter_enctext(eid_list),
        ibs_src.get_encounter_note(eid_list)
    )
    return encounter_td


def export_image_transfer_data(ibs_src, gid_list, config_rowid_list, eid_list, nid_list,
                               species_rowid_list):
    """
    builds transfer data for seleted image ids in ibs_src.
    NOTE: gid_list, config_rowid_list and eid_list do not correspond
    """
    if gid_list is None or len(gid_list) == 0:
        return None
    # Get image data
    image_size_list = ibs_src.get_image_sizes(gid_list)
    image_gps_list = ibs_src.get_image_gps(gid_list)
    # Get encounter INDEXs
    eids_list = ibs_src.get_image_eids(gid_list)
    encounter_INDEXs_list = [
        [_index(eid, eid_list) for eid in eid_list_]
        for eid_list_ in eids_list
    ]
    # Get image-label relationships
    glrids_list = ibs_src.get_image_glrids(gid_list)
    lblimage_td_list = [
        export_lblimage_transfer_data(ibs_src, glrid_list, config_rowid_list)
        for glrid_list in glrids_list
    ]
    # Get annotations
    aids_list = ibs_src.get_image_aids(gid_list)
    annot_td_list = [
        export_annot_transfer_data(ibs_src, aid_list, config_rowid_list, nid_list,
                                   species_rowid_list)
        for aid_list in aids_list
    ]
    # Create Image TransferData
    image_td = IMAGE_TransferData(
        encounter_INDEXs_list,
        ibs_src.get_image_paths(gid_list),
        ibs_src.get_image_uuids(gid_list),
        ibs_src.get_image_exts(gid_list),
        ibs_src.get_image_gnames(gid_list),
        [size[0] for size in image_size_list],
        [size[1] for size in image_size_list],
        ibs_src.get_image_unixtime(gid_list),
        [gps[0] for gps in image_gps_list],
        [gps[1] for gps in image_gps_list],
        ibs_src.get_image_enabled(gid_list),
        ibs_src.get_image_reviewed(gid_list),
        ibs_src.get_image_notes(gid_list),
        lblimage_td_list,
        annot_td_list
    )
    return image_td


def export_annot_transfer_data(ibs_src, aid_list, config_rowid_list, species_rowid_list):
    if aid_list is None or len(aid_list) == 0:
        return None
    # Get annotation parents
    annot_parent_rowid_list = ibs_src.get_annot_parent_aid(aid_list)
    # We can make this assumption because parts are not shared across an image.
    annot_parent_INDEX_list = [
        None if annot_parent_rowid is None else _index(annot_parent_rowid, aid_list)
        for annot_parent_rowid in annot_parent_rowid_list
    ]
    # Get annotation-label relationships
    alrids_list = ibs_src.get_annot_alrids(aid_list)
    lblannot_td_list = [
        export_lblannot_transfer_data(ibs_src, alrid_list, config_rowid_list)
        for alrid_list in alrids_list
    ]
    # Get names and species of annotations
    annot_name_rowid_list = ibs_src.get_annot_name_rowids(aid_list, distinguish_unknowns=False)
    annot_name_INDEX_list = [ _index(nid, nid_list) if nid != const.UNKNOWN_NAME_ROWID else None for nid in annot_name_rowid_list ]  # NOQA
    annot_species_rowid_list = ibs_src.get_annot_species_rowids(aid_list)
    annot_species_INDEX_list = [  # NOQA
        _index(species_rowid, species_rowid_list)
        for species_rowid in annot_species_rowid_list
    ]
    # Create Annotation TransferData
    annot_td = ANNOTATION_TransferData(
        annot_parent_INDEX_list,
        ibs_src.get_annot_uuids(aid_list),
        ibs_src.get_annot_thetas(aid_list),
        ibs_src.get_annot_verts(aid_list),
        ibs_src.get_annot_viewpoints(aid_list),
        ibs_src.get_annot_detect_confidence(aid_list),
        ibs_src.get_annot_exemplar_flags(aid_list),
        ibs_src.get_annot_visual_uuids(aid_list),
        ibs_src.get_annot_semantic_uuids(aid_list),
        ibs_src.get_annot_notes(aid_list),
        annot_name_INDEX_list,
        annot_species_INDEX_list,
        lblannot_td_list
    )
    return annot_td


def export_lblimage_transfer_data(ibs_src, glrid_list, config_rowid_list):
    if glrid_list is None or len(glrid_list) == 0:
        return None
    # Get lblimage config
    config_INDEX_list = [
        _index(ibs_src.get_glr_config_rowid(glrid), config_rowid_list)
        for glrid in glrid_list
    ]
    lblimage_rowid_list = ibs_src.get_glr_lblimage_rowids(glrid_list)
    lbltypes_rowid_list = ibs_src.get_lblimage_lbltypes_rowids(lblimage_rowid_list)
    # Create Lblimage TransferData
    lblimage_td = LBLIMAGE_TransferData(
        config_INDEX_list,
        ibs_src.get_glr_confidence(glrid_list),
        ibs_src.get_lblimage_uuids(lblimage_rowid_list),
        ibs_src.get_lbltype_text(lbltypes_rowid_list),
        ibs_src.get_lblimage_values(lblimage_rowid_list),
        ibs_src.get_lblimage_notes(lblimage_rowid_list)
    )
    return lblimage_td


def export_lblannot_transfer_data(ibs_src, alrid_list, config_rowid_list):
    if alrid_list is None or len(alrid_list) == 0:
        return None
    # Get lblannot config
    config_INDEX_list = [
        _index(ibs_src.get_alr_config_rowid(alrid), config_rowid_list)
        for alrid in alrid_list
    ]
    lblannot_rowid_list = ibs_src.get_alr_lblannot_rowids(alrid_list)
    lbltypes_rowid_list = ibs_src.get_lblannot_lbltypes_rowids(lblannot_rowid_list)
    # Create Lblannot TransferData
    lblannot_td = LBLANNOT_TransferData(
        config_INDEX_list,
        ibs_src.get_alr_confidence(alrid_list),
        ibs_src.get_lblannot_uuids(lblannot_rowid_list),
        ibs_src.get_lbltype_text(lbltypes_rowid_list),
        ibs_src.get_lblannot_values(lblannot_rowid_list),
        ibs_src.get_lblannot_notes(lblannot_rowid_list)
    )
    return lblannot_td


#############################
#############################
#############################


def import_transfer_data(ibs_dst, td, bulk_conflict_resolution='merge'):
    """
    Imports transfer data from any ibeis database and moves it into ibs_dst
    """
    nid_list = import_name_transfer_data(ibs_dst, td.name_td)
    species_rowid_list = import_species_transfer_data(ibs_dst, td.species_td)
    # Import the contributors
    added = []
    rejected = []
    for contributor_td in td.contributor_td_list:
        contrib_uuid = contributor_td.contributor_uuid
        success = import_contributor_transfer_data(
            ibs_dst,
            contributor_td,
            nid_list,
            species_rowid_list,
            bulk_conflict_resolution=bulk_conflict_resolution
        )
        if success:
            added.append(contrib_uuid)
        else:
            rejected.append(contrib_uuid)
    print("[import_transfer_data] ----------------------")
    print("[import_transfer_data] Database %r imported" % (td.transfer_database_name,))
    print("[import_transfer_data]   Contributors Accepted: %i" % (len(added),))
    print("[import_transfer_data]   Contributors Rejected: %i" % (len(rejected),))


def import_contributor_transfer_data(ibs_dst, contributor_td, nid_list, species_rowid_list,
                                     bulk_conflict_resolution='merge'):
    print("[import_transfer_data] Import Contributor: %r" % (contributor_td.contributor_uuid,))
    # Find conflicts
    contributor_rowid = ibs_dst.get_contributor_rowid_from_uuid([contributor_td.contributor_uuid])[0]
    if contributor_rowid is not None:
        # Resolve conflict
        if bulk_conflict_resolution == 'replace':
            print("[import_transfer_data]     Conflict Resolution - Replacing contributor: %r" % (contributor_td.contributor_uuid, ))
            # Delete current contributor
            ibs_dst.delete_contributors([contributor_rowid])
        elif bulk_conflict_resolution == 'ignore':
            print("[import_transfer_data]     Conflict Resolution - Ignoring contributor: %r" % (contributor_td.contributor_uuid, ))
            return True
        else:
            print("[import_transfer_data]     Conflict Resolution - Merging contributor: %r" % (contributor_td.contributor_uuid, ))
            # TODO: do a more sophisticated contributor merge
            return False

    contributor_rowid = ibs_dst.add_contributors(
        [contributor_td.contributor_tag],
        uuid_list=[contributor_td.contributor_uuid],
        name_first_list=[contributor_td.contributor_name_first],
        name_last_list=[contributor_td.contributor_name_last],
        loc_city_list=[contributor_td.contributor_location_city],
        loc_state_list=[contributor_td.contributor_location_state],
        loc_country_list=[contributor_td.contributor_location_country],
        loc_zip_list=[contributor_td.contributor_location_zip],
        notes_list=[contributor_td.contributor_note]
    )[0]
    # Import configs
    if contributor_td.config_td is not None:
        if utool.VERBOSE:
            print("[import_transfer_data]   Importing configs: %r" % (contributor_td.config_td.config_suffixes_list,))
        else:
            print("[import_transfer_data]   Importing configs")
        config_rowid_list = import_config_transfer_data(
            ibs_dst,
            contributor_td.config_td,
            contributor_rowid,
            bulk_conflict_resolution=bulk_conflict_resolution
        )
        print("[import_transfer_data]   ...imported %i configs" % (len(config_rowid_list),))
    else:
        config_rowid_list = []
        print("[import_transfer_data]   NO CONFIGS TO IMPORT (WARNING)")
    # Import encounters
    if contributor_td.encounter_td is not None:
        if utool.VERBOSE:
            print("[import_transfer_data]   Importing encounters: %r" % (contributor_td.encounter_td.encounter_uuid_list,))
        else:
            print("[import_transfer_data]   Importing encounters:")
            eid_list = import_encounter_transfer_data(
                ibs_dst,
                contributor_td.encounter_td,
                config_rowid_list,
                bulk_conflict_resolution=bulk_conflict_resolution
            )
        print("[import_transfer_data]   ...imported %i encounters" % (len(eid_list),))
    else:
        eid_list = []
        print("[import_transfer_data]   NO ENCOUNTERS TO IMPORT")
    # Import images
    if contributor_td.image_td is not None:
        if utool.VERBOSE:
            print("[import_transfer_data]   Importing images: %r" % (contributor_td.image_td.image_uuid_list,))
        else:
            print("[import_transfer_data]   Importing images:")
        gid_list = import_image_transfer_data(
            ibs_dst,
            contributor_td.image_td,
            contributor_rowid,
            eid_list,
            nid_list,
            species_rowid_list,
            config_rowid_list,
            bulk_conflict_resolution=bulk_conflict_resolution
        )
        print("[import_transfer_data]   ...imported %i images" % (len(gid_list),))
    else:
        print("[import_transfer_data]   NO IMAGES TO IMPORT")
    # Finished importing contributor
    print("[import_transfer_data] ...imported contributor: %s" % (contributor_rowid,))
    return True


def import_name_transfer_data(ibs_dst, name_td):
    # Import Name TransferData
    name_rowid_list = ibs_dst.add_names(
        name_td.name_text_list,
        name_td.name_uuid_list,
        name_td.name_note_list
    )
    return name_rowid_list


def import_species_transfer_data(ibs_dst, species_td):
    # Import Species TransferData
    species_rowid_list = ibs_dst.add_species(
        species_td.species_text_list,
        species_td.species_uuid_list,
        species_td.species_note_list
    )
    return species_rowid_list


def import_config_transfer_data(ibs_dst, config_td, contributor_rowid, bulk_conflict_resolution='merge'):
    # Find conflicts
    # Map input (because transfer objects are read-only)
    config_suffixes_list = config_td.config_suffixes_list
    # Find conflicts
    known_config_rowid_list = ibs_dst.get_config_rowid_from_suffix(config_suffixes_list)
    valid_list = [ known_config_rowid is None for known_config_rowid in known_config_rowid_list ]
    if not all(valid_list):
        # Resolve conflicts
        invalid_config_rowid_list = utool.filterfalse_items(known_config_rowid_list, valid_list)
        #invalid_indices = utool.filterfalse_items(range(len(known_config_rowid_list)), valid_list) # TODO
        if bulk_conflict_resolution == 'replace':
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Replacing configs: %r" % (invalid_config_rowid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Replacing %i configs..." % (len(invalid_config_rowid_list), ))
            # Delete invalid configs
            ibs_dst.delete_configs(invalid_config_rowid_list)
        elif bulk_conflict_resolution == 'ignore':
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Ignoring configs: %r" % (invalid_config_rowid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Ignoring %i configs..." % (len(invalid_config_rowid_list), ))
            config_suffixes_list = utool.filter_items(config_td.config_suffixes_list, valid_list)
        else:
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Merging configs: %r" % (invalid_config_rowid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Merging %i configs..." % (len(invalid_config_rowid_list), ))
            # TODO: do a more sophisticated config merge
    # Add configs
    config_rowid_list = ibs_dst.add_config(
        config_suffixes_list,
        contrib_rowid_list=[contributor_rowid] * len(config_suffixes_list)
    )
    return config_rowid_list


def import_encounter_transfer_data(ibs_dst, encounter_td, config_rowid_list, bulk_conflict_resolution='merge'):
    # Map input (because transfer objects are read-only)
    config_INDEX_list   = encounter_td.config_INDEX_list
    encounter_uuid_list = encounter_td.encounter_uuid_list
    encounter_text_list = encounter_td.encounter_text_list
    encoutner_note_list = encounter_td.encoutner_note_list
    # Find conflicts
    known_eid_list = ibs_dst.get_encounter_eids_from_text(encounter_text_list)
    valid_list = [ known_eid is None for known_eid in known_eid_list ]
    if not all(valid_list):
        # Resolve conflicts
        invalid_eid_list = utool.filterfalse_items(known_eid_list, valid_list)
        #invalid_indices = utool.filterfalse_items(range(len(known_eid_list)), valid_list)  # TODO
        if bulk_conflict_resolution == 'replace':
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Replacing encounters: %r" % (invalid_eid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Replacing %i encounters..." % (len(invalid_eid_list), ))
            # Delete invalid gids
            ibs_dst.delete_encounters(invalid_eid_list)
        elif bulk_conflict_resolution == 'ignore':
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Ignoring encounters: %r" % (invalid_eid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Ignoring %i encounters..." % (len(invalid_eid_list), ))
            config_INDEX_list   = utool.filter_items(config_INDEX_list,   valid_list)
            encounter_uuid_list = utool.filter_items(encounter_uuid_list, valid_list)
            encounter_text_list = utool.filter_items(encounter_text_list, valid_list)
            encoutner_note_list = utool.filter_items(encoutner_note_list, valid_list)
        else:
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Merging encounters: %r" % (invalid_eid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Merging %i encounters..." % (len(invalid_eid_list), ))
            # TODO: do a more sophisticated encounter merge
    # Add encounters
    config_rowid_list_ = [ config_rowid_list[i] for i in config_INDEX_list ]
    eid_list = ibs_dst.add_encounters(
        encounter_text_list,
        encounter_uuid_list=encounter_uuid_list,
        config_rowid_list=config_rowid_list_,
        notes_list=encoutner_note_list,
    )
    return eid_list


def import_image_transfer_data(ibs_dst, image_td, contributor_rowid,
                               eid_list, nid_list, species_rowid_list,
                               config_rowid_list, bulk_conflict_resolution='merge'):
    # Map input (because transfer objects are read-only)
    encounter_INDEXs_list      = image_td.encounter_INDEXs_list
    image_path_list            = image_td.image_path_list
    image_uuid_list            = image_td.image_uuid_list
    image_ext_list             = image_td.image_ext_list
    image_original_name_list   = image_td.image_original_name_list
    image_width_list           = image_td.image_width_list
    image_height_list          = image_td.image_height_list
    image_time_posix_list      = image_td.image_time_posix_list
    image_gps_lat_list         = image_td.image_gps_lat_list
    image_gps_lon_list         = image_td.image_gps_lon_list
    image_toggle_enabled_list  = image_td.image_toggle_enabled_list
    image_toggle_reviewed_list = image_td.image_toggle_reviewed_list
    image_note_list            = image_td.image_note_list
    lblimage_td_list           = image_td.lblimage_td_list
    annotation_td_list         = image_td.annotation_td_list
    # Find conflicts
    known_gid_list = ibs_dst.get_image_gids_from_uuid(image_uuid_list)
    valid_list = [ known_gid is None for known_gid in known_gid_list ]
    if not all(valid_list):
        # Resolve conflicts
        invalid_gid_list = utool.filterfalse_items(known_gid_list, valid_list)
        #invalid_indices = utool.filterfalse_items(range(len(known_gid_list)), valid_list)  # TODO
        if bulk_conflict_resolution == 'replace':
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Replacing images: %r" % (invalid_gid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Replacing %i images..." % (len(invalid_gid_list), ))
            # Delete invalid gids
            ibs_dst.delete_images(invalid_gid_list)
        elif bulk_conflict_resolution == 'ignore':
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Ignoring images: %r" % (invalid_gid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Ignoring %i images..." % (len(invalid_gid_list), ))
            encounter_INDEXs_list      = utool.filter_items(encounter_INDEXs_list,      valid_list)
            image_path_list            = utool.filter_items(image_path_list,            valid_list)
            image_uuid_list            = utool.filter_items(image_uuid_list,            valid_list)
            image_ext_list             = utool.filter_items(image_ext_list,             valid_list)
            image_original_name_list   = utool.filter_items(image_original_name_list,   valid_list)
            image_width_list           = utool.filter_items(image_width_list,           valid_list)
            image_height_list          = utool.filter_items(image_height_list,          valid_list)
            image_time_posix_list      = utool.filter_items(image_time_posix_list,      valid_list)
            image_gps_lat_list         = utool.filter_items(image_gps_lat_list,         valid_list)
            image_gps_lon_list         = utool.filter_items(image_gps_lon_list,         valid_list)
            image_toggle_enabled_list  = utool.filter_items(image_toggle_enabled_list,  valid_list)
            image_toggle_reviewed_list = utool.filter_items(image_toggle_reviewed_list, valid_list)
            image_note_list            = utool.filter_items(image_note_list,            valid_list)
            lblimage_td_list           = utool.filter_items(lblimage_td_list,           valid_list)
            annotation_td_list         = utool.filter_items(annotation_td_list,         valid_list)
        else:
            if utool.VERBOSE:
                print("[import_transfer_data]     Conflict Resolution - Merging images: %r" % (invalid_gid_list, ))
            else:
                print("[import_transfer_data]     Conflict Resolution - Merging %i images..." % (len(invalid_gid_list), ))
            print("IMAGE MERGING HAS NOT BEEN IMPLEMENTED, USE IGNORE OR REPLACE RESOLUTIONS FOR NOW")
            raise

    # Sanity Check
    assert len(image_uuid_list) == len(set(image_uuid_list)), "Not unique images"
    # Add images
    params_list = zip(image_uuid_list, image_path_list,
                      image_original_name_list, image_ext_list, image_width_list,
                      image_height_list, image_time_posix_list, image_gps_lat_list,
                      image_gps_lon_list, image_note_list)
    gid_list = ibs_dst.add_images(
        image_path_list,
        params_list=params_list
    )
    # Add new contributor and set image reviewed and enabled bits
    print("[import_transfer_data]     Associating images with contributors and setting reviewed and enabled bits...")
    contrib_rowid_list = [contributor_rowid] * len(gid_list)
    ibs_dst.set_image_contributor_rowid(gid_list, contrib_rowid_list)
    ibs_dst.set_image_reviewed(gid_list, image_toggle_reviewed_list)
    ibs_dst.set_image_enabled(gid_list, image_toggle_enabled_list)
    # Add images to appropriate encounters
    print("[import_transfer_data]     Associating images with new encounters...")
    for gid, encounter_INDEXs in zip(gid_list, encounter_INDEXs_list):
        for encounter_INDEX in encounter_INDEXs:
            if 0 <= encounter_INDEX and encounter_INDEX < len(eid_list):
                ibs_dst.set_image_eids([gid], [eid_list[encounter_INDEX]])
    # Add lblimages
    print("[import_transfer_data]     Importing lblimages...")
    glrid_total = 0
    for gid, lblimage_td in zip(gid_list, lblimage_td_list):
        if lblimage_td is not None:
            glrid_list = import_lblimage_transfer_data(
                ibs_dst,
                lblimage_td,
                gid,
                config_rowid_list
            )
            glrid_total += len(glrid_list)
    print("[import_transfer_data]     ...imported %i lblimages" % (glrid_total,))
    # Add annotations
    print("[import_transfer_data]     Importing annotations...")
    aid_total = 0
    for gid, annotation_td in zip(gid_list, annotation_td_list):
        if annotation_td is not None:
            if utool.VERBOSE:
                print("[import_transfer_data]     Importing annotations for image %r: %r " % (gid, annotation_td.annot_uuid_list,))
            aid_list = import_annot_transfer_data(
                ibs_dst,
                annotation_td,
                gid,
                nid_list,
                species_rowid_list,
                config_rowid_list
            )
            aid_total += len(aid_list)
            if utool.VERBOSE:
                print("[import_transfer_data]     ...imported %i annotations" % (len(aid_list),))
        elif utool.VERBOSE:
            print("[import_transfer_data]     NO ANNOTATIONS TO IMPORT FOR IMAGE %r" % (gid))
    print("[import_transfer_data]     ...imported %i annotations" % (aid_total,))
    return gid_list


def import_annot_transfer_data(ibs_dst, annot_td, parent_gid, config_rowid_list,
                               nid_list, species_rowid_list):
    parent_gid_list = [parent_gid] * len(annot_td.annot_uuid_list)
    name_rowid_list = [
        const.UNKNOWN_NAME_ROWID
        if annot_name_INDEX is None else
        nid_list[annot_name_INDEX]
        for annot_name_INDEX in annot_td.annot_name_INDEX_list
    ]
    species_rowid_list = [
        const.UNKNOWN_SPECIES_ROWID
        if annot_species_INDEX is None else
        species_rowid_list[annot_species_INDEX]
        for annot_species_INDEX in annot_td.annot_species_INDEX_list
    ]
    aid_list = ibs_dst.add_annots(
        parent_gid_list,
        theta_list=annot_td.annot_theta_list,
        detect_confidence_list=annot_td.annot_detection_confidence_list,
        notes_list=annot_td.annot_note_list,
        vert_list=annot_td.annot_verts_list,
        annot_uuid_list=annot_td.annot_uuid_list,
        viewpoint_list=annot_td.annot_viewpoint_list,
        annot_visual_uuid_list=annot_td.annot_visual_uuid_list,
        annot_semantic_uuid_list=annot_td.annot_semantic_uuid_list,
        nid_list=name_rowid_list,
        species_rowid_list=species_rowid_list,
        quiet_delete_thumbs=True  # Turns off thumbnail deletion print statements
    )
    if utool.VERBOSE:
        print("[import_transfer_data]       Setting the annotation's parent and exemplar bits...")
    # Adding parent rowids that come from the aid_list (only can come from this list because
    # aid parent rowids cannot span across images)
    for aid, annot_parent_INDEX in zip(aid_list, annot_td.annot_parent_INDEX_list):
        if annot_parent_INDEX is not None and 0 <= annot_parent_INDEX and annot_parent_INDEX < len(aid_list):
            parent_aid = aid_list[annot_parent_INDEX]
            ibs_dst.set_annot_parent_rowid([aid], [parent_aid])
    ibs_dst.set_annot_exemplar_flags(aid_list, annot_td.annot_exemplar_flag_list)
    # Add lblannots
    if utool.VERBOSE:
        print("[import_transfer_data]       Importing lblannots...")
    alrid_total = 0
    for aid, lblannot_td in zip(aid_list, annot_td.lblannot_td_list):
        if lblannot_td is not None:
            alrid_list = import_lblannot_transfer_data(
                ibs_dst,
                lblannot_td,
                aid,
                config_rowid_list
            )
            alrid_total += len(alrid_list)
    if utool.VERBOSE:
        print("[import_transfer_data]       ...imported %i lblimages" % (alrid_total,))
    return aid_list


def import_lblimage_transfer_data(ibs_dst, lblimage_td, gid, config_rowid_list):
    # Add lblimages
    lblimage_rowid_list = ibs_dst.add_lblimages(
        ibs_dst.get_lbltype_rowid_from_text(lblimage_td.lbltype_text_list),
        lblimage_td.lblimage_value_list,
        note_list=lblimage_td.lblimage_note_list,
        lblimage_uuid_list=lblimage_td.lblimage_uuid_list
    )
    # Add image-label relationships
    valid_list = [
        0 <= config_INDEX and config_INDEX < len(config_rowid_list)
        for config_INDEX in lblimage_td.config_INDEX_list
    ]
    lblimage_rowid_list = utool.filter_items(lblimage_rowid_list, valid_list)
    gid_list = [gid] * len(lblimage_rowid_list)
    config_rowid_list = [
        config_rowid_list[config_INDEX]
        for config_INDEX, valid in zip(lblimage_td.config_INDEX_list, valid_list)
        if valid
    ]
    glr_confidence_list = utool.filter_items(lblimage_td.glr_confidence_list, valid_list)
    glrid_list = ibs_dst.add_image_relationship(
        gid_list,
        lblimage_rowid_list,
        config_rowid_list=config_rowid_list,
        glr_confidence_list=glr_confidence_list,
    )
    return glrid_list


def import_lblannot_transfer_data(ibs_dst, lblannot_td, aid, config_rowid_list):
    # Add lblannots
    lblannot_rowid_list = ibs_dst.add_lblannots(
        ibs_dst.get_lbltype_rowid_from_text(lblannot_td.lbltype_text_list),
        lblannot_td.lblannot_value_list,
        note_list=lblannot_td.lblannot_note_list,
        lblannot_uuid_list=lblannot_td.lblannot_uuid_list
    )
    # Add annot-label relationships
    valid_list = [
        0 <= config_INDEX and config_INDEX < len(config_rowid_list)
        for config_INDEX in lblannot_td.config_INDEX_list
    ]
    lblannot_rowid_list = utool.filter_items(lblannot_rowid_list, valid_list)
    aid_list = [aid] * len(lblannot_rowid_list)
    config_rowid_list = [
        config_rowid_list[config_INDEX]
        for config_INDEX, valid in zip(lblannot_td.config_INDEX_list, valid_list)
        if valid
    ]
    alr_confidence_list = utool.filter_items(lblannot_td.alr_confidence_list, valid_list)
    alrid_list = ibs_dst.add_annot_relationship(
        aid_list,
        lblannot_rowid_list,
        config_rowid_list=config_rowid_list,
        alr_confidence_list=alr_confidence_list,
    )
    return alrid_list


#############################
#############################
#############################


def merge_databases(ibs_src, ibs_dst, gid_list=None, back=None, user_prompt=False, bulk_conflict_resolution='ignore'):
    """
    Conflict resolutions are only between contributors, configs, encounters and images.
    Annotations, lblannots, lblimages, their respective relationships, and image-encounter
    relationships all inherit the resolution from their associated image.

    Args:
        ibs_src (IBEISController): source controller

        ibs_dst (IBEISController): destination controller

        back (GUIBackend): optional gui to update

        user_prompt (bool): prompt user for information

        bulk_conflict_resolution (str): valid conflict_resolutions are::
            +---
            * 'replace' - delete original in ibs_dst and import new value
            * 'ignore' - ignore imported value and keep original in ibs_dst
            * 'merge' - (default) keep both values in both databases
            +---
            WARNING - this may cause near-duplicate information due to duplicate
            detections, duplicate manual annotations from different
            contributors.  Images and lblimages will not be duplicated.
            WARNING - this will only keep the meta data for an image, annotation
            from the source database (image_height, annotation_verts, etc.)
            WARNING - this may cause an exception to be raised
            +---

    Example:
        >>> # SLOW_DOCTEST
        >>> import ibeis
        >>> from ibeis.dbio import export_subset
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> ibs_src = ibeis.opendb(dbdir='testdb1')
        >>> bulk_conflict_resolution = 'ignore'
        >>> #gid_list = None
        >>> back = None
        >>> user_prompt = False
        >>> #ibs_src2 = ibeis.opendb(dbdir='PZ_MTEST')
        >>> print(ibs_src.get_infostr())
        >>> #print(ibs_src2.get_infostr())
        >>> ibs_dst = ibeis.opendb(dbdir='testdb_dst', allow_newdir=True, delete_ibsdir=True)
        >>> assert ibs_dst.get_num_names() == 0
        >>> assert ibs_dst.get_num_images() == 0
        >>> assert ibs_dst.get_num_annotations() == 0
        >>> #ibs_dst = ibs
        >>> export_subset.merge_databases(ibs_src, ibs_dst,
        ...                               bulk_conflict_resolution=bulk_conflict_resolution)
        >>> #export_subset.merge_databases(ibs_src2, ibs_dst, bulk_conflict_resolution='ignore')
        >>> print(ibs_dst.get_infostr())
    """
    # Export source database
    td = export_transfer_data(ibs_src, gid_list=gid_list, user_prompt=user_prompt)
    if utool.VERBOSE:
        print("\n\n[merge_databases] -------------------\n\n")
        print("[merge_databases] %s" % (td,))
        print("\n\n[merge_databases] -------------------\n\n")
    # Check that destination database has a valid contributor
    contrib_rowid_list = ensure_contributor_rowids(ibs_dst, user_prompt=user_prompt)  # NOQA
    # Import tansfer data object into the destination database
    # TODO: Implement db backup
    import_transfer_data(ibs_dst, td, bulk_conflict_resolution=bulk_conflict_resolution)
    # TODO: Implement db restore or db deletion
    ibs_dst.notify_observers()


# RELEVANT LEGACY CODE FOR IMAGE MERGING
# def check_conflicts(ibs_src, ibs_dst, transfer_data):
#     """
#     Check to make sure the destination database does not have any conflicts
#     with the incoming transfer.

#     Currently only checks that images do not have conflicting annotations.

#     Does not check label consistency.
#     """

#     # TODO: Check label consistency: ie check that labels with the
#     # same (type, value) should also have the same UUID
#     img_td      = transfer_data.img_td
#     #annot_td    = transfer_data.annot_td
#     #lblannot_td = transfer_data.lblannot_td
#     #alr_td      = transfer_data.alr_td

#     image_uuid_list1 = img_td.img_uuid_list
#     sameimg_gid_list2_ = ibs_dst.get_image_gids_from_uuid(image_uuid_list1)
#     issameimg = [gid is not None for gid in sameimg_gid_list2_]
#     # Check if databases contain the same images
#     if any(issameimg):
#         sameimg_gid_list2 = utool.filter_items(sameimg_gid_list2_, issameimg)
#         sameimg_image_uuids = utool.filter_items(image_uuid_list1, issameimg)
#         print('! %d/%d images are duplicates' % (len(sameimg_gid_list2), len(image_uuid_list1)))
#         # Check if sameimg images in dst has any annotations.
#         sameimg_aids_list2 = ibs_dst.get_image_aids(sameimg_gid_list2)
#         hasannots = [len(aids) > 0 for aids in sameimg_aids_list2]
#         if any(hasannots):
#             # TODO: Merge based on some merge stratagy parameter (like annotation timestamp)
#             sameimg_gid_list1 = ibs_src.get_image_gids_from_uuid(sameimg_image_uuids)
#             hasannot_gid_list2 = utool.filter_items(sameimg_gid_list2, hasannots)
#             hasannot_gid_list1 = utool.filter_items(sameimg_gid_list1, hasannots)
#             print('  !! %d/%d of those have annotations' % (len(hasannot_gid_list2), len(sameimg_gid_list2)))
#             # They had better be the same annotations!
#             assert_images_have_same_annnots(ibs_src, ibs_dst, hasannot_gid_list1, hasannot_gid_list2)
#             print('  ...phew, all of the annotations were the same.')
#         #raise AssertionError('dst dataset contains some of this data')


# def assert_images_have_same_annnots(ibs_src, ibs_dst, hasannot_gid_list1, hasannot_gid_list2):
#     """ Given a list of gids from each ibs, this function asserts that every
#         annontation in gid1 is the same as every annontation in gid2
#     """
#     from ibeis.ibsfuncs import unflat_map
#     hasannot_aids_list1 = ibs_src.get_image_aids(hasannot_gid_list1)
#     hasannot_aids_list2 = ibs_dst.get_image_aids(hasannot_gid_list2)
#     hasannot_auuids_list1 = unflat_map(ibs_src.get_annot_uuids, hasannot_aids_list1)
#     hasannot_auuids_list2 = unflat_map(ibs_dst.get_annot_uuids, hasannot_aids_list2)
#     hasannot_verts_list1 = unflat_map(ibs_src.get_annot_verts, hasannot_aids_list1)
#     hasannot_verts_list2 = unflat_map(ibs_dst.get_annot_verts, hasannot_aids_list2)
#     assert_same_annot_uuids(hasannot_auuids_list1, hasannot_auuids_list2)
#     assert_same_annot_verts(hasannot_verts_list1, hasannot_verts_list2)  # hack, check verts as well


# def assert_same_annot_uuids(hasannot_auuids_list1, hasannot_auuids_list2):
#     uuids_pair_iter = zip(hasannot_auuids_list1, hasannot_auuids_list2)
#     msg = ('The {count}-th image has inconsistent annotation:. '
#            'auuids1={auuids1} auuids2={auuids2}')
#     for count, (auuids1, auuids2) in enumerate(uuids_pair_iter):
#         assert auuids1 == auuids2, msg.format(
#             count=count, auuids1=auuids1, auuids2=auuids2,)


# def assert_same_annot_verts(hasannot_verts_list1, hasannot_verts_list2):
#     verts_pair_iter = zip(hasannot_verts_list1, hasannot_verts_list2)
#     msg = ('The {count}-th image has inconsistent annotation:. '
#            'averts1={averts1} averts2={averts2}')
#     for count, (averts1, averts2) in enumerate(verts_pair_iter):
#         assert averts1 == averts2, msg.format(
#             count=count, averts1=averts1, averts2=averts2,)


if __name__ == '__main__':
    """
    python -m ibeis.dbio.export_subset
    python -m ibeis.dbio.export_subset --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
