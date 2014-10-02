#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.io.export_subset))"
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from collections import namedtuple
import utool
import datetime
import ibeis
from ibeis import ibsfuncs # NOQA
from ibeis.constants import (AL_RELATION_TABLE, ANNOTATION_TABLE, CONFIG_TABLE, 
                             CONTRIBUTOR_TABLE, EG_RELATION_TABLE, ENCOUNTER_TABLE, 
                             GL_RELATION_TABLE, IMAGE_TABLE, LBLANNOT_TABLE, 
                             LBLIMAGE_TABLE, __STR__)
from vtool import geometry

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
        'annot_xtl_list',
        'annot_ytl_list',
        'annot_width_list',
        'annot_height_list',
        'annot_theta_list',
        'annot_num_verts_list',
        'annot_verts_list',
        'annot_viewpoint_list',
        'annot_detection_confidence_list',
        'annot_exemplar_flag_list',
        'annot_note_list',
        'lblannot_td_list',
    ))

LBLIMAGE_TransferData = namedtuple(
    'LBLIMAGE_TransferData', (
        'config_INDEX_list',
        'glr_confidence_list',
        'lblimage_uuid_list',
        'lbltype_rowid_list',
        'lblimage_value_list',
        'lblimage_note_list',
    ))

LBLANNOT_TransferData = namedtuple(
    'LBLANNOT_TransferData', (
        'config_INDEX_list',
        'alr_confidence_list',
        'lblannot_uuid_list',
        'lbltype_rowid_list',
        'lblannot_value_list',
        'lblannot_note_list',
    ))


#############################
#############################
#############################


def _index(value, list_, warning=False):
    if value in list_:
        return list_.index(value)
    else:
        print("[export_subset] WARNING: value (%r) not in list: %r" %(value, list_))
        return None


def geo_locate(default="Unknown"):
    try:
        import urllib2
        import json
        f = urllib2.urlopen('http://freegeoip.net/json/')
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


#############################
#############################
#############################


def export_contributor_transfer_data(ibs_src, contributor_rowid):
    # Get configs
    config_rowid_list = ibs_src.get_contributor_config_rowids(contributor_rowid)
    config_td = export_config_transfer_data(ibs_src, config_rowid_list) 
    # Get encounters
    eid_list = utool.flatten( ibs_src.get_contributor_eids(config_rowid_list) )
    encounter_td = export_encounter_transfer_data(ibs_src, eid_list, config_rowid_list)
    # Get images
    gid_list = ibs_src.get_contributor_gids(contributor_rowid)
    image_td = export_image_transfer_data(ibs_src, gid_list, config_rowid_list, eid_list)

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


def export_config_transfer_data(ibs_src, config_rowid_list):
    if config_rowid_list is None or len(config_rowid_list) == 0:
        return None
    # Get config data
    config_td = CONFIG_TransferData(
        ibs_src.get_config_suffixes(config_rowid_list)
    )
    return config_td


def export_encounter_transfer_data(ibs_src, eid_list, config_rowid_list):
    if eid_list is None or len(eid_list) == 0:
        return None
    # Get encounter data
    config_INDEX_list = [ _index(ibs_src.get_encounter_config(eid), config_rowid_list) for eid in eid_list ]
    encounter_td = ENCOUNTER_TransferData(
        config_INDEX_list,
        ibs_src.get_encounter_uuid(eid_list),
        ibs_src.get_encounter_enctext(eid_list),
        ibs_src.get_encounter_note(eid_list)
    )
    return encounter_td


def export_image_transfer_data(ibs_src, gid_list, config_rowid_list, eid_list):
    if gid_list is None or len(gid_list) == 0:
        return None
    # Get image data
    image_size_list = ibs_src.get_image_sizes(gid_list)
    image_gps_list = ibs_src.get_image_gps(gid_list)
    # Get encounter INDEXs
    eids_list = ibs_src.get_image_eids(gid_list)
    encounter_INDEXs_list = [ 
        [_index(eid, eid_list) for eid in eid_list]
        for eid_list in eids_list
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
        export_annot_transfer_data(ibs_src, aid_list, config_rowid_list)
        for aid_list in aids_list
    ]

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
        [gps[0] for gps in image_gps_list],
        ibs_src.get_image_enabled(gid_list),
        ibs_src.get_image_reviewed(gid_list),
        ibs_src.get_image_notes(gid_list),
        lblimage_td_list,
        annot_td_list
    )
    return image_td


def export_annot_transfer_data(ibs_src, aid_list, config_rowid_list):
    if aid_list is None or len(aid_list) == 0:
        return None
    # Get annotation parents
    annot_parent_rowid_list = ibs_src.get_annot_parent_aid(aid_list)
    # We can make this assumption because parts are not shared across an image.
    annot_parent_INDEX_list = [
        None if annot_parent_rowid is None else _index(annot_parent_rowid, aid_list)
        for annot_parent_rowid in annot_parent_rowid_list
    ]
    annot_bbox_list = ibs_src.get_annot_bboxes(aid_list)
    
    alrids_list = ibs_src.get_annot_alrids(aid_list)
    lblannot_td_list = [
        export_lblannot_transfer_data(ibs_src, alrid_list, config_rowid_list)
        for alrid_list in alrids_list
    ]

    annot_td = ANNOTATION_TransferData(
        annot_parent_INDEX_list,
        ibs_src.get_annot_uuids(aid_list),
        [bbox[0] for bbox in annot_bbox_list],
        [bbox[1] for bbox in annot_bbox_list],
        [bbox[2] for bbox in annot_bbox_list],
        [bbox[3] for bbox in annot_bbox_list],
        ibs_src.get_annot_thetas(aid_list),
        ibs_src.get_annot_num_verts(aid_list),
        ibs_src.get_annot_verts(aid_list),
        ibs_src.get_annot_viewpoints(aid_list),
        ibs_src.get_annot_detect_confidence(aid_list),
        ibs_src.get_annot_exemplar_flag(aid_list),
        ibs_src.get_annot_notes(aid_list),
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

    lblimage_td = LBLIMAGE_TransferData(
        config_INDEX_list, 
        ibs_src.get_glr_confidence(glrid_list),
        ibs_src.get_lblimage_uuids(lblimage_rowid_list),
        ibs_src.get_lblimage_lbltypes_rowids(lblimage_rowid_list),
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

    lblannot_td = LBLANNOT_TransferData(
        config_INDEX_list, 
        ibs_src.get_alr_confidence(alrid_list),
        ibs_src.get_lblannot_uuids(lblannot_rowid_list),
        ibs_src.get_lblannot_lbltypes_rowids(lblannot_rowid_list),
        ibs_src.get_lblannot_values(lblannot_rowid_list),
        ibs_src.get_lblannot_notes(lblannot_rowid_list)
    )
    return lblannot_td


def export_transfer_data(ibs_src, user_prompt=False):
    """
    Packs all the data you are going to transfer from ibs_src
    info the transfer_data named tuple.
    """
    contrib_rowid_list = ibs_src.get_valid_contrib_rowids()
    if len(contrib_rowid_list) == 0:
        print("[collect_transfer_data] There is no contributor for this database.  Add a contributor to continue merge.")
        name_first = ibs_src.dbname
        name_last = utool.get_computer_name() + ":" + utool.get_user_name() + ":" + ibs_src.workdir
        print("Contributor default first name: %s" %(name_first, ))
        print("Contributor default last name:  %s" %(name_last, ))
        if user_prompt:
            name_first = raw_input("\nChange first name (Enter to use default): ")
            name_last  = raw_input("\nChange last name (Enter to use default): ")

        success, location_city, location_state, location_country, location_zip = geo_locate()

        if success:
            print("\nYour location was be determined automatically.")
            print("Contributor default city: %s"    %(location_city, ))
            print("Contributor default state: %s"   %(location_state, ))
            print("Contributor default zip: %s"     %(location_country, ))
            print("Contributor default country: %s" %(location_zip, ))
            if user_prompt:
                location_city    = raw_input("\nChange default location city (Enter to use default): ")
                location_state   = raw_input("\nChange default location state (Enter to use default): ")
                location_zip     = raw_input("\nChange default location zip (Enter to use default): ")
                location_country = raw_input("\nChange default location country (Enter to use default): ")
        else:
            print("Your location could not be determined automatically.")
            if user_prompt:
                location_city    = raw_input("Enter your location city (Enter to skip): ")
                location_state   = raw_input("Enter your location state (Enter to skip): ")
                location_zip     = raw_input("Enter your location zip (Enter to skip): ")
                location_country = raw_input("Enter your location country (Enter to skip): ")
            else:
                location_city    = ''
                location_state   = ''
                location_zip     = ''
                location_country = ''

        tag = "::".join([name_first, name_last, location_city, location_state, location_zip, location_country])
        contrib_rowid = ibs_src.add_contributors([tag], name_first_list=[name_first], 
                                name_last_list=[name_last], loc_city_list=[location_city], 
                                loc_state_list=[location_state], loc_country_list=[location_country], 
                                loc_zip_list=[location_zip])
        contrib_rowid = contrib_rowid[0]
        print('New contributor\'s contrib_rowid: %s' %(contrib_rowid,))
        ibs_src.set_config_contributor_unassigned(contrib_rowid)
        ibs_src.set_image_contributor_unassigned(contrib_rowid)
        ibs_src.set_encounter_config_unassigned()

    contrib_rowid_list = ibs_src.get_valid_contrib_rowids()
    assert len(contrib_rowid_list) > 0, "There must be at least one contributor to merge"
    contributor_td_list = [ 
        export_contributor_transfer_data(ibs_src, contrib_rowid)
        for contrib_rowid in contrib_rowid_list
    ]

    success, location_city, location_state, location_country, location_zip = geo_locate()
    td = TransferData(
        ibs_src.dbname,
        utool.get_computer_name() + ":" + utool.get_user_name() + ":" + ibs_src.workdir,
        "%s" %(datetime.datetime.now()),
        location_city,    
        location_state,   
        location_zip,     
        location_country, 
        contributor_td_list
    )
    return td


#############################
#############################
#############################


def import_lblannot_transfer_data(ibs_dst, lblannot_td, aid, config_rowid_list):
    # Add lblannots
    lblannot_rowid_list = ibs_dst.add_lblannots(
        lblannot_td.lbltype_rowid_list, 
        lblannot_td.lblannot_value_list, 
        note_list=lblannot_td.lblannot_note_list, 
        lblannot_uuid_list=lblannot_td.lblannot_uuid_list
    )
    # Add annot-label relationships
    aid_list = [aid] * len(lblannot_rowid_list)
    config_rowid_list = [ config_rowid_list[config_INDEX] for config_INDEX in lblannot_td.config_INDEX_list ]
    alrid_list = ibs_dst.add_annot_relationship(
        aid_list,
        lblannot_rowid_list,
        config_rowid_list=config_rowid_list,
        alr_confidence_list=lblannot_td.alr_confidence_list,
    )
    return alrid_list


def import_lblimage_transfer_data(ibs_dst, lblimage_td, gid, config_rowid_list):
    # Add lblimages
    lblimage_rowid_list = ibs_dst.add_lblimages(
        lblimage_td.lbltype_rowid_list, 
        lblimage_td.lblimage_value_list, 
        note_list=lblimage_td.lblimage_note_list, 
        lblimage_uuid_list=lblimage_td.lblimage_uuid_list
    )
    # Add image-label relationships
    gid_list = [gid] * len(lblimage_rowid_list)
    config_rowid_list = [ config_rowid_list[config_INDEX] for config_INDEX in lblimage_td.config_INDEX_list ]
    glrid_list = ibs_dst.add_image_relationship(
        gid_list,
        lblimage_rowid_list,
        config_rowid_list=config_rowid_list,
        glr_confidence_list=lblimage_td.glr_confidence_list,
    )
    return glrid_list


def import_annot_transfer_data(ibs_src, annot_td):
    pass


def import_image_transfer_data(ibs_dst, image_td, contributor_rowid, eid_list, config_rowid_list):
    # Add images
    params_list = zip(image_td.image_uuid_list, image_td.image_path_list,
        image_td.image_original_name_list, image_td.image_ext_list, image_td.image_width_list,
        image_td.image_height_list, image_td.image_time_posix_list, image_td.image_gps_lat_list,
        image_td.image_gps_lon_list, image_td.image_note_list
    )
    gid_list = ibs_dst.add_images(
        image_td.image_path_list,
        params_list=params_list
    )
    # Add new contributor and set image reviewed and enabled bits
    contrib_rowid_list = [contributor_rowid] * len(gid_list)
    ibs_dst.set_image_contributor_rowid(gid_list, contrib_rowid_list)
    ibs_dst.set_image_reviewed(gid_list, image_td.image_toggle_reviewed_list)
    ibs_dst.set_image_enabled(gid_list, image_td.image_toggle_enabled_list)
    # Add images to appropriate encounters
    for gid, encounter_INDEXs in zip(gid_list, encounter_INDEXs_list):
        for encounter_INDEX in encounter_INDEXs:
            ibs.set_image_eids([gid], [eid_list[encounter_INDEX]])
    # Add lblimages
    for gid, lblimage_td in (gid_list, lblimage_td_list):
        if lblimage_td is not None:
            print("\n\t\tImporting Lblimages: %r" %(contributor_td.image_td.image_uuid_list,))
            glrid_list = import_lblimage_transfer_data(ibs_src, 
                lblimage_td, 
                gid, 
                config_rowid_list
            )
            print("\t\t...imported %i lblimages" %(len(glrid_list),))
        else:
            print("\n\t\tNO LBLIMAGES TO IMPORT FOR IMAGE %r" %(gid))
    # Add annotations 'annotation_td_list',
    for gid, annotation_td in (gid_list, lannotation_td_list):
        if annotation_td is not None:
            print("\n\t\tImporting Annotations: %r" %(contributor_td.image_td.image_uuid_list,))
            aid_list = import_annot_transfer_data(ibs_dst, 
                annotation_td,
                gid,
                config_rowid_list
            )
            print("\t\t...imported %i annotations" %(len(aid_list),))
        else:
            print("\n\t\tNO ANNOTATIONS TO IMPORT FOR IMAGE %r" %(gid))
    return gid_list


def import_encounter_transfer_data(ibs_dst, encounter_td, config_rowid_list):
    config_rowid_list_ = [ config_rowid_list[i] for i in encounter_td.config_INDEX_list ]
    eid_list = ibs_dst.add_encounters(
        encounter_td.encounter_text_list,
        encounter_uuid_list=encounter_td.encounter_uuid_list,
        config_rowid_list=config_rowid_list_,
        notes_list=encounter_td.encoutner_note_list,
    )
    return eid_list


def import_config_transfer_data(ibs_dst, config_td, contributor_rowid):
    config_rowid_list = ibs_dst.add_config(
        config_td.config_suffixes_list,
        contrib_rowid_list=[contributor_rowid] * len(config_td.config_suffixes_list)
    )
    return config_rowid_list


def import_contributor_transfer_data(ibs_dst, contributor_td):
    print("Import Contributor: %r" %(contributor_td.contributor_uuid,))
    
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
    )
    contributor_rowid = contributor_rowid[0]
    # Import configs
    if contributor_td.config_td is not None:
        print("\n\tImporting Configs: %r" %(contributor_td.config_td.config_suffixes_list,))
        config_rowid_list = import_config_transfer_data(ibs_dst, 
            contributor_td.config_td,
            contributor_rowid
        )
        print("\t...imported %i configs" %(len(config_rowid_list),))
    else:
        print("\n\tNO CONFIGS TO IMPORT (WARNING)")        
    # Import encounters
    if contributor_td.encounter_td is not None:
        print("\n\tImporting Encounters: %r" %(contributor_td.encounter_td.encounter_uuid_list,))
        eid_list = import_encounter_transfer_data(ibs_dst, 
            contributor_td.encounter_td,
            config_rowid_list
        )
        print("\t...imported %i encounters" %(len(eid_list),))
    else:
        print("\n\tNO ENCOUNTERS TO IMPORT")     
    # Import images
    if contributor_td.image_td is not None:
        print("\n\tImporting Images: %r" %(contributor_td.image_td.image_uuid_list,))
        gid_list = import_image_transfer_data(ibs_dst, 
            contributor_td.image_td,
            contributor_rowid,
            eid_list,
            config_rowid_list
        )
        print("\t...imported %i images" %(len(gid_list),))
    else:
        print("\n\tNO IMAGES TO IMPORT")    
    # Finished importing contributor
    print("\n...imported contributor\n")
    return True


def import_transfer_data(ibs_dst, td):
    """
    Imports transfer data from any ibeis database and moves it into ibs_dst
    """
    added = []
    rejected = []
    for contributor_td in td.contributor_td_list:
        contrib_uuid = contributor_td.contributor_uuid
        success = import_contributor_transfer_data(ibs_dst, contributor_td)
        if success:
            added.append(contrib_uuid)
        else:
            rejected.append(contrib_uuid)
    print("\n----------------------")
    print("Database %r imported\n\tContributors Accepted: %i\n\tContributors Rejected: %i" 
        %(td.transfer_database_name, len(added), len(rejected))
    )


#############################
#############################
#############################


def merge_databases(ibs_src, ibs_dst):
    """
    ibs_src = ibeis.opendb('testdb1')
    merge_databases(ibs, ibs_src)
    """
    td = export_transfer_data(ibs_src)
    print("\n\n-------------------\n\n")
    print(td)
    print("\n\n-------------------\n\n")
    import_transfer_data(ibs_dst, td)


def test_merge_with_mothers(ibs_dst):
    """
    merge the secified database with PZ_Mothers
    """
    ibs_src = ibeis.opendb('PZ_Mothers')
    merge_databases(ibs_src, ibs_dst)




