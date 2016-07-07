#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Exports subset of an IBEIS database to a new IBEIS database

"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
from collections import namedtuple
import utool as ut
import datetime
# import ibeis
import inspect
from ibeis.other import ibsfuncs  # NOQA
from ibeis import constants as const
# from ibeis.constants import (AL_RELATION_TABLE, ANNOTATION_TABLE, CONFIG_TABLE,
#                              CONTRIBUTOR_TABLE, GSG_RELATION_TABLE, IMAGESET_TABLE,
#                              GL_RELATION_TABLE, IMAGE_TABLE, LBLANNOT_TABLE,
#                              LBLIMAGE_TABLE, __STR__)
# from vtool import geometry

# Transfer data structures could become classes.
# TODO: Remove the nesting of transfer datas
# it should be a flat list
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
        'imageset_td',
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
        'species_nice_list',
        'species_text_list',
        'species_code_list',
        'species_note_list',
    ))

CONFIG_TransferData = namedtuple(
    'CONFIG_TransferData', (
        'config_suffixes_list',
    ))

IMAGESET_TransferData = namedtuple(
    'IMAGESET_TransferData', (
        'config_INDEX_list',
        'imageset_uuid_list',
        'imageset_text_list',
        'encoutner_note_list',
    ))

IMAGE_TransferData = namedtuple(
    'IMAGE_TransferData', (
        'imageset_INDEXs_list',
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
        'annot_yaw_list',
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


def tryindex(value, list_, warning=True):
    if value in list_:
        return list_.index(value)
    else:
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        if warning:
            print('[export_subset] WARNING: value (%r) not in list: %r (%r)' %
                  (value, list_, calframe[1][3]))
        return None


#############################
#############################
#############################


def export_transfer_data(ibs_src, gid_list=None):
    """
    STEP 1)

    Packs all the data you are going to transfer from ibs_src
    info the transfer_data named tuple.

    CommandLine:
        python -m ibeis.dbio.export_subset --test-export_transfer_data

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> import ibeis
        >>> from ibeis.dbio import export_subset    # NOQA
        >>> #ibs_src = ibeis.opendb(dbdir='/raid/work2/Turk/PZ_Master')
        >>> ibs_src = ibeis.opendb(db='testdb1')
        >>> bulk_conflict_resolution = 'ignore'
        >>> num = 5
        >>> ibs_src.ensure_contributor_rowids(user_prompt=False)
        >>> gid_list = ibs_src.get_valid_gids()[0:num]
        >>> td = export_transfer_data(ibs_src, gid_list=gid_list)
        >>> assert len(td.contributor_td_list) == 1, 'more than 1 contrib'
        >>> contributor_td = td.contributor_td_list[0]
        >>> image_td = contributor_td.image_td
        >>> assert len(image_td.annotation_td_list) == num
        >>> annotation_td = image_td.annotation_td_list[num - 1]
        >>> annot_td_dict = annotation_td._asdict()
        >>> # remove non-determenistic uuid
        >>> del annot_td_dict['annot_uuid_list']
        >>> result = ut.dict_str(annot_td_dict)
        >>> print(result)
        {
            'annot_parent_INDEX_list': [None],
            'annot_theta_list': [0.0],
            'annot_verts_list': [((0, 0), (1072, 0), (1072, 804), (0, 804))],
            'annot_yaw_list': [3.141592653589793],
            'annot_detection_confidence_list': [0.0],
            'annot_exemplar_flag_list': [1],
            'annot_visual_uuid_list': [UUID('5a1a53ba-fd44-b113-7f8c-fcf248d7047f')],
            'annot_semantic_uuid_list': [UUID('a0f8ba5a-82b4-578f-3995-ee92c0ebfe66')],
            'annot_note_list': [u''],
            'annot_name_INDEX_list': [1],
            'annot_species_INDEX_list': [0],
            'lblannot_td_list': [None],
        }

    print(ut.truncate_str(ibs.db.get_table_csv(const.IMAGE_TABLE, exclude_columns=['image_uuid', 'image_uri']), 10000))

    """
    if not ut.QUIET:
        print('Exporting transfer from ibs_src.dbname = %r' %
              (ibs_src.get_dbname()))
    if gid_list is None:
        gid_list = ibs_src.get_valid_gids()
    if not ut.QUIET:
        print('... with %d images' % (len(gid_list)))
    nid_list = list(set(ut.flatten(ibs_src.get_image_nids(gid_list))))
    species_rowid_list = ibs_src._get_all_species_rowids()
    # Create Name TransferData
    name_td = export_name_transfer_data(ibs_src, nid_list)  # NOQA
    # Create Species TranferData
    species_td = export_species_transfer_data(ibs_src, species_rowid_list)   # NOQA
    assert len(ibs_src.get_all_uncontributed_images()
               ) == 0, 'images are still uncontributed'
    # with ut.EmbedOnException():
    if gid_list is not None:
        contrib_rowid_list = list(
            set(ibs_src.get_image_contributor_rowid(gid_list)))
    assert len(
        contrib_rowid_list) > 0, 'There must be at least one contributor to merge'
    contributor_td_list = [
        export_contributor_transfer_data(ibs_src, contrib_rowid, nid_list,
                                         species_rowid_list, valid_gid_list=gid_list)
        for contrib_rowid in contrib_rowid_list
    ]
    # Geolocate and create database's TransferData object
    success, location_city, location_state, location_country, location_zip = ut.geo_locate()
    transfer_database_source = (
        ut.get_computer_name() + ':' + ut.get_user_name() + ':' + ibs_src.workdir)
    transfer_export_time = '%s' % (datetime.datetime.now())

    td = TransferData(
        ibs_src.dbname,
        transfer_database_source,
        transfer_export_time,
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

        imgsetid_list = ibs.get_valid_imgsetids()
        imageset_config_rowid_list = ibs.get_imageset_configid(imgsetid_list)
        imageset_suffix_list = ibs.get_config_suffixes(config_rowid_list)
        print(ut.list_str(list(zip(imageset_config_rowid_list, imageset_suffix_list))))

    """
    # Get configs
    #config_rowid_list = ibs_src.get_contributor_config_rowids(contributor_rowid)
    # Hack around config-less imagesets
    config_rowid_list = ibs_src.get_valid_configids()
    config_td = export_config_transfer_data(ibs_src, config_rowid_list)
    # Get imagesets
    #imgsetid_list = ibs_src.get_valid_imgsetids()
    imgsetid_list = ut.flatten(ibs_src.get_contributor_imgsetids(config_rowid_list))
    imageset_td = export_imageset_transfer_data(
        ibs_src, imgsetid_list, config_rowid_list)
    # Get images
    gid_list = ibs_src.get_contributor_gids(contributor_rowid)
    if valid_gid_list is not None:
        isvalid_list = [gid in valid_gid_list for gid in gid_list]
        gid_list = ut.compress(gid_list, isvalid_list)
    image_td = export_image_transfer_data(ibs_src, gid_list, config_rowid_list, imgsetid_list,
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
        imageset_td,
        image_td
    )
    return contributor_td


def export_name_transfer_data(ibs_src, nid_list):
    # TODO: autogenerate getter dictionaries
    # TODO: incorporate autogenerated getter dictionaries
    # name_getters = {
    #    'name_uuid'        : ibs_src.get_name_uuids,
    #    'name_text'        : ibs_src.get_name_texts,
    #    'name_notes'       : ibs_src.get_name_notes,
    #    'name_temp_flag'   : ibs_src.get_name_temp_flag,
    #    'name_alias_texts' : ibs_src.get_name_alias_texts,
    #}
    # Create Name TransferData
    name_td = NAME_TransferData(
        ibs_src.get_name_uuids(nid_list),
        ibs_src.get_name_texts(nid_list),
        ibs_src.get_name_notes(nid_list)
    )
    return name_td


def export_species_transfer_data(ibs_src, species_rowid_list):
    """
    ibs_src.db.print_schema()
    print(ibs.db.get_table_csv_header(ibeis.const.SPECIES_TABLE))
    """
    # Create Species TransferData
    species_td = SPECIES_TransferData(
        ibs_src.get_species_uuids(species_rowid_list),
        ibs_src.get_species_nice(species_rowid_list),
        ibs_src.get_species_texts(species_rowid_list),
        ibs_src.get_species_codes(species_rowid_list),
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


def export_imageset_transfer_data(ibs_src, imgsetid_list, config_rowid_list):
    if imgsetid_list is None or len(imgsetid_list) == 0:
        return None
    # Get imageset data
    config_INDEX_list = [
        tryindex(ibs_src.get_imageset_configid(imgsetid), config_rowid_list)
        for imgsetid in imgsetid_list]
    # Create ImageSet TransferData
    imageset_td = IMAGESET_TransferData(
        config_INDEX_list,
        ibs_src.get_imageset_uuid(imgsetid_list),
        ibs_src.get_imageset_text(imgsetid_list),
        ibs_src.get_imageset_note(imgsetid_list)
    )
    return imageset_td


def export_image_transfer_data(ibs_src, gid_list, config_rowid_list, imgsetid_list, nid_list,
                               species_rowid_list):
    """
    builds transfer data for seleted image ids in ibs_src.
    NOTE: gid_list, config_rowid_list and imgsetid_list do not correspond
    """
    if gid_list is None or len(gid_list) == 0:
        return None
    # Get image data
    #image_size_list = ibs_src.get_image_sizes(gid_list)
    #image_gps_list = ibs_src.get_image_gps(gid_list)
    # Get imageset INDEXs
    imgsetids_list = ibs_src.get_image_imgsetids(gid_list)
    imageset_INDEXs_list = [
        [tryindex(imgsetid, imgsetid_list) for imgsetid in imgsetid_list_]
        for imgsetid_list_ in imgsetids_list
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
        imageset_INDEXs_list,
        ibs_src.get_image_paths(gid_list),
        ibs_src.get_image_uuids(gid_list),
        ibs_src.get_image_exts(gid_list),
        ibs_src.get_image_gnames(gid_list),
        ibs_src.get_image_widths(gid_list),
        ibs_src.get_image_heights(gid_list),
        #[size[0] for size in image_size_list],
        #[size[1] for size in image_size_list],
        ibs_src.get_image_unixtime(gid_list),
        ibs_src.get_image_lat(gid_list),
        ibs_src.get_image_lon(gid_list),
        #[gps[0] for gps in image_gps_list],
        #[gps[1] for gps in image_gps_list],
        ibs_src.get_image_enabled(gid_list),
        ibs_src.get_image_reviewed(gid_list),
        ibs_src.get_image_notes(gid_list),
        lblimage_td_list,
        annot_td_list
    )
    return image_td


def export_annot_transfer_data(ibs_src, aid_list, config_rowid_list, nid_list, species_rowid_list):
    if aid_list is None or len(aid_list) == 0:
        return None
    # Get annotation parents
    annot_parent_rowid_list = ibs_src.get_annot_parent_aid(aid_list)
    # We can make this assumption because parts are not shared across an image.
    annot_parent_INDEX_list = [
        None if annot_parent_rowid is None else tryindex(
            annot_parent_rowid, aid_list)
        for annot_parent_rowid in annot_parent_rowid_list
    ]
    # Get annotation-label relationships
    alrids_list = ibs_src.get_annot_alrids(aid_list)
    lblannot_td_list = [
        export_lblannot_transfer_data(ibs_src, alrid_list, config_rowid_list)
        for alrid_list in alrids_list
    ]
    # Get names and species of annotations
    annot_name_rowid_list = ibs_src.get_annot_name_rowids(
        aid_list, distinguish_unknowns=False)
    annot_name_INDEX_list = [tryindex(nid, nid_list) if nid != const.UNKNOWN_NAME_ROWID else None for nid in annot_name_rowid_list]  # NOQA
    annot_species_rowid_list = ibs_src.get_annot_species_rowids(aid_list)
    annot_species_INDEX_list = [  # NOQA
        tryindex(species_rowid, species_rowid_list)
        for species_rowid in annot_species_rowid_list
    ]
    # Create Annotation TransferData
    annot_td = ANNOTATION_TransferData(
        annot_parent_INDEX_list,
        ibs_src.get_annot_uuids(aid_list),
        ibs_src.get_annot_thetas(aid_list),
        ibs_src.get_annot_verts(aid_list),
        ibs_src.get_annot_yaws(aid_list),
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
        tryindex(ibs_src.get_glr_config_rowid(glrid), config_rowid_list)
        for glrid in glrid_list
    ]
    lblimage_rowid_list = ibs_src.get_glr_lblimage_rowids(glrid_list)
    lbltypes_rowid_list = ibs_src.get_lblimage_lbltypes_rowids(
        lblimage_rowid_list)
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
        tryindex(ibs_src.get_alr_config_rowid(alrid), config_rowid_list)
        for alrid in alrid_list
    ]
    lblannot_rowid_list = ibs_src.get_alr_lblannot_rowids(alrid_list)
    lbltypes_rowid_list = ibs_src.get_lblannot_lbltypes_rowids(
        lblannot_rowid_list)
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
    print('[import_transfer_data] ----------------------')
    print('[import_transfer_data] Database %r imported' %
          (td.transfer_database_name,))
    print('[import_transfer_data]   Contributors Accepted: %i' % (len(added),))
    print('[import_transfer_data]   Contributors Rejected: %i' %
          (len(rejected),))


def import_contributor_transfer_data(ibs_dst, contributor_td, nid_list, species_rowid_list,
                                     bulk_conflict_resolution='merge'):
    print('[import_transfer_data] Import Contributor: %r' %
          (contributor_td.contributor_uuid,))
    # Find conflicts
    contributor_rowid = ibs_dst.get_contributor_rowid_from_uuid(
        [contributor_td.contributor_uuid])[0]
    if contributor_rowid is not None:
        # Resolve conflict
        if bulk_conflict_resolution == 'replace':
            print('[import_transfer_data]     Conflict Resolution - Replacing contributor: %r' %
                  (contributor_td.contributor_uuid, ))
            # Delete current contributor
            ibs_dst.delete_contributors([contributor_rowid])
        elif bulk_conflict_resolution == 'ignore':
            print('[import_transfer_data]     Conflict Resolution - Ignoring contributor: %r' %
                  (contributor_td.contributor_uuid, ))
            return True
        else:
            print('[import_transfer_data]     Conflict Resolution - Merging contributor: %r' %
                  (contributor_td.contributor_uuid, ))
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
        if ut.VERBOSE:
            print('[import_transfer_data]   Importing configs: %r' %
                  (contributor_td.config_td.config_suffixes_list,))
        else:
            print('[import_transfer_data]   Importing configs')
        config_rowid_list = import_config_transfer_data(
            ibs_dst,
            contributor_td.config_td,
            contributor_rowid,
            bulk_conflict_resolution=bulk_conflict_resolution
        )
        print('[import_transfer_data]   ...imported %i configs' %
              (len(config_rowid_list),))
    else:
        config_rowid_list = []
        print('[import_transfer_data]   NO CONFIGS TO IMPORT (WARNING)')
    # Import imagesets
    if contributor_td.imageset_td is not None:
        if ut.VERBOSE:
            print('[import_transfer_data]   Importing imagesets: %r' %
                  (contributor_td.imageset_td.imageset_uuid_list,))
        else:
            print('[import_transfer_data]   Importing imagesets:')
            imgsetid_list = import_imageset_transfer_data(
                ibs_dst,
                contributor_td.imageset_td,
                config_rowid_list,
                bulk_conflict_resolution=bulk_conflict_resolution
            )
        print('[import_transfer_data]   ...imported %i imagesets' %
              (len(imgsetid_list),))
    else:
        imgsetid_list = []
        print('[import_transfer_data]   NO IMAGESETS TO IMPORT')
    # Import images
    if contributor_td.image_td is not None:
        if ut.VERBOSE:
            print('[import_transfer_data]   Importing images: %r' %
                  (contributor_td.image_td.image_uuid_list,))
        else:
            print('[import_transfer_data]   Importing images:')
        gid_list = import_image_transfer_data(
            ibs_dst,
            contributor_td.image_td,
            contributor_rowid,
            imgsetid_list,
            nid_list,
            species_rowid_list,
            config_rowid_list,
            bulk_conflict_resolution=bulk_conflict_resolution
        )
        print('[import_transfer_data]   ...imported %i images' %
              (len(gid_list),))
    else:
        print('[import_transfer_data]   NO IMAGES TO IMPORT')
    # Finished importing contributor
    print('[import_transfer_data] ...imported contributor: %s' %
          (contributor_rowid,))
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
        species_td.species_nice_list,
        species_td.species_text_list,
        species_td.species_code_list,
        species_td.species_uuid_list,
        species_td.species_note_list
    )
    return species_rowid_list


def import_config_transfer_data(ibs_dst, config_td, contributor_rowid,
                                bulk_conflict_resolution='merge'):
    # Find conflicts
    # Map input (because transfer objects are read-only)
    config_suffixes_list = config_td.config_suffixes_list
    # Find conflicts
    known_config_rowid_list = ibs_dst.get_config_rowid_from_suffix(
        config_suffixes_list)
    valid_list = [
        known_config_rowid is None for known_config_rowid in known_config_rowid_list]
    if not all(valid_list):
        # Resolve conflicts
        invalid_config_rowid_list = ut.filterfalse_items(
            known_config_rowid_list, valid_list)
        # invalid_indices =
        # ut.filterfalse_items(range(len(known_config_rowid_list)),
        # valid_list) # TODO
        if bulk_conflict_resolution == 'replace':
            if ut.VERBOSE:
                print('[import_transfer_data]     Conflict Resolution - Replacing configs: %r' %
                      (invalid_config_rowid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Replacing %i configs...' %
                      (len(invalid_config_rowid_list), ))
            # Delete invalid configs
            ibs_dst.delete_configs(invalid_config_rowid_list)
        elif bulk_conflict_resolution == 'ignore':
            if ut.VERBOSE:
                print('[import_transfer_data]     Conflict Resolution - Ignoring configs: %r' %
                      (invalid_config_rowid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Ignoring %i configs...' %
                      (len(invalid_config_rowid_list), ))
            config_suffixes_list = ut.filter_items(
                config_td.config_suffixes_list, valid_list)
        else:
            if ut.VERBOSE:
                print('[import_transfer_data]     Conflict Resolution - Merging configs: %r' %
                      (invalid_config_rowid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Merging %i configs...' %
                      (len(invalid_config_rowid_list), ))
            # TODO: do a more sophisticated config merge
    # Add configs
    config_rowid_list = ibs_dst.add_config(
        config_suffixes_list,
        contrib_rowid_list=[contributor_rowid] * len(config_suffixes_list)
    )
    return config_rowid_list


def import_imageset_transfer_data(ibs_dst, imageset_td, config_rowid_list,
                                   bulk_conflict_resolution='merge'):
    # Map input (because transfer objects are read-only)
    config_INDEX_list = imageset_td.config_INDEX_list
    imageset_uuid_list = imageset_td.imageset_uuid_list
    imageset_text_list = imageset_td.imageset_text_list
    encoutner_note_list = imageset_td.encoutner_note_list
    # Find conflicts
    known_imgsetid_list = ibs_dst.get_imageset_imgsetids_from_text(imageset_text_list)
    valid_list = [known_imgsetid is None for known_imgsetid in known_imgsetid_list]
    if not all(valid_list):
        # Resolve conflicts
        invalid_imgsetid_list = ut.filterfalse_items(known_imgsetid_list, valid_list)
        # invalid_indices = ut.filterfalse_items(range(len(known_imgsetid_list)),
        # valid_list)  # TODO
        if bulk_conflict_resolution == 'replace':
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     Conflict Resolution - Replacing imagesets: %r' %
                    (invalid_imgsetid_list, ))
            else:
                print('[import_transfer_data]   Conflict Resolution - Replacing %i imagesets...' %
                      (len(invalid_imgsetid_list), ))
            # Delete invalid gids
            ibs_dst.delete_imagesets(invalid_imgsetid_list)
        elif bulk_conflict_resolution == 'ignore':
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     Conflict Resolution - Ignoring imagesets: %r' %
                    (invalid_imgsetid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Ignoring %i imagesets...' %
                      (len(invalid_imgsetid_list), ))
            config_INDEX_list = ut.filter_items(
                config_INDEX_list,   valid_list)
            imageset_uuid_list = ut.filter_items(
                imageset_uuid_list, valid_list)
            imageset_text_list = ut.filter_items(
                imageset_text_list, valid_list)
            encoutner_note_list = ut.filter_items(
                encoutner_note_list, valid_list)
        else:
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     Conflict Resolution - Merging imagesets: %r' %
                    (invalid_imgsetid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Merging %i imagesets...' %
                      (len(invalid_imgsetid_list), ))
            # TODO: do a more sophisticated imageset merge
    # Add imagesets
    config_rowid_list_ = [config_rowid_list[i] for i in config_INDEX_list]
    imgsetid_list = ibs_dst.add_imagesets(
        imageset_text_list,
        imageset_uuid_list=imageset_uuid_list,
        config_rowid_list=config_rowid_list_,
        notes_list=encoutner_note_list,
    )
    return imgsetid_list


def import_image_transfer_data(ibs_dst, image_td, contributor_rowid,
                               imgsetid_list, nid_list, species_rowid_list,
                               config_rowid_list, bulk_conflict_resolution='merge'):
    # Map input (because transfer objects are read-only)
    imageset_INDEXs_list = image_td.imageset_INDEXs_list
    image_path_list = image_td.image_path_list
    image_uuid_list = image_td.image_uuid_list
    image_ext_list = image_td.image_ext_list
    image_original_name_list = image_td.image_original_name_list
    image_width_list = image_td.image_width_list
    image_height_list = image_td.image_height_list
    image_time_posix_list = image_td.image_time_posix_list
    image_gps_lat_list = image_td.image_gps_lat_list
    image_gps_lon_list = image_td.image_gps_lon_list
    image_toggle_enabled_list = image_td.image_toggle_enabled_list
    image_toggle_reviewed_list = image_td.image_toggle_reviewed_list
    image_note_list = image_td.image_note_list
    lblimage_td_list = image_td.lblimage_td_list
    annotation_td_list = image_td.annotation_td_list
    # Find conflicts
    known_gid_list = ibs_dst.get_image_gids_from_uuid(image_uuid_list)
    valid_list = [known_gid is None for known_gid in known_gid_list]
    if not all(valid_list):
        # Resolve conflicts
        invalid_gid_list = ut.filterfalse_items(known_gid_list, valid_list)
        # invalid_indices = ut.filterfalse_items(range(len(known_gid_list)),
        # valid_list)  # TODO
        if bulk_conflict_resolution == 'replace':
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     Conflict Resolution - Replacing images: %r' %
                    (invalid_gid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Replacing %i images...' %
                      (len(invalid_gid_list), ))
            # Delete invalid gids
            ibs_dst.delete_images(invalid_gid_list)
        elif bulk_conflict_resolution == 'ignore':
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     Conflict Resolution - Ignoring images: %r' %
                    (invalid_gid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Ignoring %i images...' %
                      (len(invalid_gid_list), ))
            imageset_INDEXs_list = ut.filter_items(
                imageset_INDEXs_list,      valid_list)
            image_path_list = ut.filter_items(
                image_path_list,            valid_list)
            image_uuid_list = ut.filter_items(
                image_uuid_list,            valid_list)
            image_ext_list = ut.filter_items(
                image_ext_list,             valid_list)
            image_original_name_list = ut.filter_items(
                image_original_name_list,   valid_list)
            image_width_list = ut.filter_items(
                image_width_list,           valid_list)
            image_height_list = ut.filter_items(
                image_height_list,          valid_list)
            image_time_posix_list = ut.filter_items(
                image_time_posix_list,      valid_list)
            image_gps_lat_list = ut.filter_items(
                image_gps_lat_list,         valid_list)
            image_gps_lon_list = ut.filter_items(
                image_gps_lon_list,         valid_list)
            image_toggle_enabled_list = ut.filter_items(
                image_toggle_enabled_list,  valid_list)
            image_toggle_reviewed_list = ut.filter_items(
                image_toggle_reviewed_list, valid_list)
            image_note_list = ut.filter_items(
                image_note_list,            valid_list)
            lblimage_td_list = ut.filter_items(
                lblimage_td_list,           valid_list)
            annotation_td_list = ut.filter_items(
                annotation_td_list,         valid_list)
        else:
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     Conflict Resolution - Merging images: %r' %
                    (invalid_gid_list, ))
            else:
                print('[import_transfer_data]     Conflict Resolution - Merging %i images...' %
                      (len(invalid_gid_list), ))
            print(
                'IMAGE MERGING HAS NOT BEEN IMPLEMENTED, USE IGNORE OR REPLACE RESOLUTIONS FOR NOW')
            raise

    # Sanity Check
    assert len(image_uuid_list) == len(
        set(image_uuid_list)), 'Not unique images'
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
    print(
        '[import_transfer_data]     '
        'Associating images with contributors and setting reviewed and enabled bits...')
    contrib_rowid_list = [contributor_rowid] * len(gid_list)
    ibs_dst.set_image_contributor_rowid(gid_list, contrib_rowid_list)
    ibs_dst.set_image_reviewed(gid_list, image_toggle_reviewed_list)
    ibs_dst.set_image_enabled(gid_list, image_toggle_enabled_list)
    # Add images to appropriate imagesets
    print(
        '[import_transfer_data]     Associating images with new imagesets...')
    for gid, imageset_INDEXs in zip(gid_list, imageset_INDEXs_list):
        for imageset_INDEX in imageset_INDEXs:
            if 0 <= imageset_INDEX and imageset_INDEX < len(imgsetid_list):
                ibs_dst.set_image_imgsetids([gid], [imgsetid_list[imageset_INDEX]])
    # Add lblimages
    print('[import_transfer_data]     Importing lblimages...')
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
    print('[import_transfer_data]     ...imported %i lblimages' %
          (glrid_total,))
    # Add annotations
    print('[import_transfer_data]     Importing annotations...')
    aid_total = 0
    for gid, annotation_td in zip(gid_list, annotation_td_list):
        if annotation_td is not None:
            if ut.VERBOSE:
                print('[import_transfer_data]     Importing annotations for image %r: %r ' % (
                    gid, annotation_td.annot_uuid_list,))
            aid_list = import_annot_transfer_data(
                ibs_dst,
                annotation_td,
                gid,
                nid_list,
                species_rowid_list,
                config_rowid_list
            )
            aid_total += len(aid_list)
            if ut.VERBOSE:
                print(
                    '[import_transfer_data]     ...imported %i annotations' % (len(aid_list),))
        elif ut.VERBOSE:
            print(
                '[import_transfer_data]     NO ANNOTATIONS TO IMPORT FOR IMAGE %r' % (gid))
    print('[import_transfer_data]     ...imported %i annotations' %
          (aid_total,))
    return gid_list


def import_annot_transfer_data(ibs_dst, annot_td, parent_gid, nid_list,
                               species_rowid_list, config_rowid_list):
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
        yaw_list=annot_td.annot_yaw_list,
        annot_visual_uuid_list=annot_td.annot_visual_uuid_list,
        annot_semantic_uuid_list=annot_td.annot_semantic_uuid_list,
        nid_list=name_rowid_list,
        species_rowid_list=species_rowid_list,
        # Turns off thumbnail deletion print statements
        quiet_delete_thumbs=True
    )
    if ut.VERBOSE:
        print(
            '[import_transfer_data]       Setting the annotation\'s parent and exemplar bits...')
    # Adding parent rowids that come from the aid_list (only can come from this list because
    # aid parent rowids cannot span across images)
    for aid, annot_parent_INDEX in zip(aid_list, annot_td.annot_parent_INDEX_list):
        if (annot_parent_INDEX is not None and
           0 <= annot_parent_INDEX and
           annot_parent_INDEX < len(aid_list)):
            parent_aid = aid_list[annot_parent_INDEX]
            ibs_dst.set_annot_parent_rowid([aid], [parent_aid])
    ibs_dst.set_annot_exemplar_flags(
        aid_list, annot_td.annot_exemplar_flag_list)
    # Add lblannots
    if ut.VERBOSE:
        print('[import_transfer_data]       Importing lblannots...')
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
    if ut.VERBOSE:
        print('[import_transfer_data]       ...imported %i lblimages' %
              (alrid_total,))
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
    lblimage_rowid_list = ut.filter_items(lblimage_rowid_list, valid_list)
    gid_list = [gid] * len(lblimage_rowid_list)
    config_rowid_list = [
        config_rowid_list[config_INDEX]
        for config_INDEX, valid in zip(lblimage_td.config_INDEX_list, valid_list)
        if valid
    ]
    glr_confidence_list = ut.filter_items(
        lblimage_td.glr_confidence_list, valid_list)
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
    lblannot_rowid_list = ut.filter_items(lblannot_rowid_list, valid_list)
    aid_list = [aid] * len(lblannot_rowid_list)
    config_rowid_list = [
        config_rowid_list[config_INDEX]
        for config_INDEX, valid in zip(lblannot_td.config_INDEX_list, valid_list)
        if valid
    ]
    alr_confidence_list = ut.filter_items(
        lblannot_td.alr_confidence_list, valid_list)
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


def merge_databases(ibs_src, ibs_dst, gid_list=None, back=None,
                    user_prompt=False, bulk_conflict_resolution='ignore'):
    """
    STEP 0) MAIN DRIVER FUNCTION

    Conflict resolutions are only between contributors, configs, imagesets and images.
    Annotations, lblannots, lblimages, their respective relationships, and image-imageset
    relationships all inherit the resolution from their associated image.

    Args:
        ibs_src (IBEISController): source controller

        ibs_dst (IBEISController): destination controller (can be an empty database)

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

    CommandLine:
        python -m ibeis.dbio.export_subset --test-merge_databases
        python -m ibeis.dbio.export_subset --test-merge_databases:1

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> import ibeis
        >>> from ibeis.dbio import export_subset
        >>> ibs_src = ibeis.opendb(db='testdb1')
        >>> bulk_conflict_resolution = 'ignore'
        >>> #gid_list = None
        >>> back = None
        >>> user_prompt = False
        >>> #ibs_src2 = ibeis.opendb(dbdir='PZ_MTEST')
        >>> print(ibs_src.get_infostr())
        >>> #print(ibs_src2.get_infostr())
        >>> # OPEN A CLEAN DATABASE
        >>> ibs_dst = ibeis.opendb(dbdir='testdb_dst', allow_newdir=True, delete_ibsdir=True)
        >>> assert ibs_dst.get_num_names() == 0, 'dst database is not empty'
        >>> assert ibs_dst.get_num_images() == 0, 'dst database is not empty'
        >>> assert ibs_dst.get_num_annotations() == 0, 'dst database is not empty'
        >>> #ibs_dst = ibs
        >>> gid_list = ibs_src.get_valid_gids()
        >>> print('Execute test func')
        >>> export_subset.merge_databases(ibs_src, ibs_dst, gid_list,
        ...                               bulk_conflict_resolution=bulk_conflict_resolution)
        >>> #export_subset.merge_databases(ibs_src2, ibs_dst, bulk_conflict_resolution='ignore')
        >>> result = ibs_dst.get_infostr()
        >>> print('Result:')
        >>> print(result)
        dbname = 'testdb_dst'
        num_images = 13
        num_annotations = 13
        num_names = 7

    Example2:
        >>> # SLOW_DOCTEST
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> import ibeis
        >>> from ibeis.dbio import export_subset
        >>> ibs_src = ibeis.opendb(db='PZ_MTEST')
        >>> bulk_conflict_resolution = 'ignore'
        >>> #gid_list = None
        >>> back = None
        >>> user_prompt = False
        >>> #ibs_src2 = ibeis.opendb(dbdir='PZ_MTEST')
        >>> print(ibs_src.get_infostr())
        >>> #print(ibs_src2.get_infostr())
        >>> # OPEN A CLEAN DATABASE
        >>> ibs_dst = ibeis.opendb(dbdir='testdb_dst', allow_newdir=False, delete_ibsdir=False)
        >>> assert ibs_dst.get_num_names() > 0, 'dst database is empty'
        >>> assert ibs_dst.get_num_images() > 0, 'dst database is empty'
        >>> assert ibs_dst.get_num_annotations() > 0, 'dst database is empty'
        >>> #ibs_dst = ibs
        >>> gid_list = ibs_src.get_valid_gids()
        >>> print('Execute test func')
        >>> export_subset.merge_databases(ibs_src, ibs_dst, gid_list,
        ...                               bulk_conflict_resolution=bulk_conflict_resolution)
        >>> #export_subset.merge_databases(ibs_src2, ibs_dst, bulk_conflict_resolution='ignore')
        >>> result = ibs_dst.get_infostr()
        >>> print('Result:')
        >>> print(result)
        dbname = 'testdb_dst'
        num_images = 128
        num_annotations = 129
        num_names = 48


    """
    # First ensure that the source and destionation have contributors
    def DEBUG_UNCONTRIBUTED_IMAGES(ibs):
        unassigned_gid_list = ibs.get_all_uncontributed_images()
        print('There are %d/%d uncontributed images in %r' %
              (len(unassigned_gid_list), len(ibs.get_valid_gids()), ibs.get_dbname()))
        if len(unassigned_gid_list) > 0:
            dpath = ibs.get_dbdir()
            fname = '_debug_uncontributed_gids_' + ut.get_timestamp() + '.txt'
            fpath = ut.unixjoin(dpath, fname)
            ut.write_to(
                fpath, 'unassigned_gid_list = ' + repr(unassigned_gid_list))
    DEBUG_UNCONTRIBUTED_IMAGES(ibs_src)
    DEBUG_UNCONTRIBUTED_IMAGES(ibs_dst)
    # Check that destination database has a valid contributor
    ibs_src.ensure_contributor_rowids(user_prompt=user_prompt)
    ibs_dst.ensure_contributor_rowids(user_prompt=user_prompt)
    # Export source database
    td = export_transfer_data(ibs_src, gid_list=gid_list)
    if ut.VERBOSE:
        print('\n\n[merge_databases] -------------------\n\n')
        print('[merge_databases] %s' % (td,))
        print('\n\n[merge_databases] -------------------\n\n')
    # Import tansfer data object into the destination database
    # TODO: Implement db backup
    import_transfer_data(
        ibs_dst, td, bulk_conflict_resolution=bulk_conflict_resolution)
    # TODO: Implement db restore or db deletion
    ibs_dst.notify_observers()


def check_merge(ibs_src, ibs_dst):
    aid_list1 = ibs_src.get_valid_aids()
    gid_list1 = ibs_src.get_annot_gids(aid_list1)
    gname_list1 = ibs_src.get_image_uris(gid_list1)
    image_uuid_list1 = ibs_src.get_image_uuids(gid_list1)
    gid_list2 = ibs_dst.get_image_gids_from_uuid(image_uuid_list1)
    gname_list2 = ibs_dst.get_image_uris(gid_list2)
    # Asserts
    ut.assert_all_not_None(gid_list1, 'gid_list1')
    ut.assert_all_not_None(gid_list2, 'gid_list2')
    ut.assert_lists_eq(gname_list1, gname_list2, 'faild gname')
    # Image UUIDS should be consistent between databases
    image_uuid_list2 = ibs_dst.get_image_uuids(gid_list2)
    ut.assert_lists_eq(image_uuid_list1, image_uuid_list2, 'failed uuid')

    aids_list1 = ibs_src.get_image_aids(gid_list1)
    aids_list2 = ibs_dst.get_image_aids(gid_list2)

    avuuids_list1 = ibs_src.unflat_map(
        ibs_src.get_annot_visual_uuids, aids_list1)
    avuuids_list2 = ibs_dst.unflat_map(
        ibs_dst.get_annot_visual_uuids, aids_list2)

    issubset_list = [set(avuuids1).issubset(set(avuuids2))
                     for avuuids1, avuuids2 in zip(avuuids_list1, avuuids_list2)]
    assert all(issubset_list), 'ibs_src must be a subset of ibs_dst: issubset_list=%r' % (
        issubset_list,)
    #aids_depth1 = ut.depth_profile(aids_list1)
    #aids_depth2 = ut.depth_profile(aids_list2)
    # depth might not be true if ibs_dst is not empty
    #ut.assert_lists_eq(aids_depth1, aids_depth2, 'failed depth')
    print('Merge seems ok...')


def merge_databases2(ibs_src, ibs_dst, rowid_subsets=None):
    """
    New way of merging using the non-hacky sql table merge.
    However, its only workings due to major hacks.

    FIXME: annotmatch table

    CommandLine:
        python -m ibeis merge_databases2

        python -m ibeis.dbio.export_subset --test-merge_databases2:0
        python -m ibeis.dbio.export_subset --test-merge_databases2:0 --db1 PZ_Master0 --db2 PZ_Master1
        python -m ibeis.dbio.export_subset --test-merge_databases2:0 --db1 NNP_Master3 --db2 PZ_Master1

        python -m ibeis.dbio.export_subset --test-merge_databases2:0 --db1 GZ_ALL --db2 GZ_Master1
        python -m ibeis.dbio.export_subset --test-merge_databases2:0 --db1 lewa_grevys --db2 GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> import ibeis
        >>> db1 = ut.get_argval('--db1', str, default=None)
        >>> db2 = ut.get_argval('--db2', str, default=None)
        >>> dbdir1 = ut.get_argval('--dbdir1', str, default=None)
        >>> dbdir2 = ut.get_argval('--dbdir2', str, default=None)
        >>> delete_ibsdir = False
        >>> # Check for test mode instead of script mode
        >>> if db1 is None and db2 is None and dbdir1 is None and dbdir2 is None:
        ...     db1 = 'testdb1'
        ...     dbdir2 = 'testdb_dst'
        ...     delete_ibsdir = True
        >>> # Open the source and destination database
        >>> assert db1 is not None or dbdir1 is not None
        >>> assert db2 is not None or dbdir2 is not None
        >>> ibs_src = ibeis.opendb(db=db1, dbdir=dbdir1)
        >>> ibs_dst = ibeis.opendb(db=db2, dbdir=dbdir2, allow_newdir=True, delete_ibsdir=delete_ibsdir)
        >>> merge_databases2(ibs_src, ibs_dst)
        >>> check_merge(ibs_src, ibs_dst)
        >>> ibs_dst.print_dbinfo()
    """
    # TODO: ensure images are localized
    # otherwise this wont work
    print('BEGIN MERGE OF %r into %r' %
          (ibs_src.get_dbname(), ibs_dst.get_dbname()))
    # ibs_src.run_integrity_checks()
    # ibs_dst.run_integrity_checks()
    ibs_dst.update_annot_visual_uuids(ibs_dst.get_valid_aids())
    ibs_src.update_annot_visual_uuids(ibs_src.get_valid_aids())
    ibs_src.ensure_contributor_rowids()
    ibs_dst.ensure_contributor_rowids()
    ibs_src.fix_invalid_annotmatches()
    ibs_dst.fix_invalid_annotmatches()
    # Hack move of the external data
    if rowid_subsets is not None and const.IMAGE_TABLE in rowid_subsets:
        gid_list = rowid_subsets[const.IMAGE_TABLE]
    else:
        gid_list = ibs_src.get_valid_gids()
    imgpath_list = ibs_src.get_image_paths(gid_list)
    dst_imgdir = ibs_dst.get_imgdir()
    ut.copy_files_to(imgpath_list, dst_imgdir, overwrite=False, verbose=True)
    ignore_tables = ['lblannot', 'lblimage', 'image_lblimage_relationship',
                     'annotation_lblannot_relationship', 'keys']
    # TODO: Fix database merge to allow merging tables with more than one superkey
    # and no primary superkey, which requires an extension of the depcache
    error_tables = [
        'imageset_image_relationship',
        'annotgroup_annotation_relationship',
        'annotmatch',
    ]
    ignore_tables += error_tables
    ibs_dst.db.merge_databases_new(
        ibs_src.db, ignore_tables=ignore_tables, rowid_subsets=rowid_subsets)
    print('FINISHED MERGE %r into %r' %
          (ibs_src.get_dbname(), ibs_dst.get_dbname()))


def make_new_dbpath(ibs, id_label, id_list):
    """
    Creates a new database path unique to the exported subset of ids.
    """
    import ibeis
    tag_hash = ut.hashstr_arr(id_list, hashlen=8, alphabet=ut.ALPHABET_27)
    base_fmtstr = ibs.get_dbname() + '_' + id_label + 's=' + \
        tag_hash.replace('(', '_').replace(')', '_') + '_%d'
    dpath = ibeis.get_workdir()
    new_dbpath = ut.get_nonconflicting_path_old(base_fmtstr, dpath)
    return new_dbpath


def export_names(ibs, nid_list, new_dbpath=None):
    r"""
    exports a subset of names and other required info

    Args:
        ibs (IBEISController):  ibeis controller object
        nid_list (list):

    CommandLine:
        python -m ibeis.dbio.export_subset --test-export_names

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> ibs.delete_empty_nids()
        >>> nid_list = ibs._get_all_known_nids()[0:2]
        >>> # execute function
        >>> result = export_names(ibs, nid_list)
        >>> # verify results
        >>> print(result)
    """
    print('Exporting name nid_list=%r' % (nid_list,))
    if new_dbpath is None:
        new_dbpath = make_new_dbpath(ibs, 'nid', nid_list)

    aid_list = ut.flatten(ibs.get_name_aids(nid_list))
    gid_list = ut.unique_unordered(ibs.get_annot_gids(aid_list))

    return export_data(ibs, gid_list, aid_list, nid_list, new_dbpath=new_dbpath)


def export_images_temp(ibs):
    gid_list = ibs.get_valid_gids()
    reviewed_list = ibs.get_image_reviewed(gid_list)
    gid_list_ = [gid for gid, reviewed in zip(
        gid_list, reviewed_list) if reviewed == 1]
    export_images(ibs, gid_list_, '/Datasets/PZ_Master1-Sub3')


def find_gid_list(ibs, min_count=500, ensure_annots=False):
    import random
    gid_list = ibs.get_valid_gids()
    reviewed_list = ibs.get_image_reviewed(gid_list)

    if ensure_annots:
        aids_list = ibs.get_image_aids(gid_list)
        reviewed_list = [
            0 if len(aids) == 0 else reviewed
            for aids, reviewed in zip(aids_list, reviewed_list)
        ]

    # Filter by reviewed
    gid_list = [
        gid
        for gid, reviewed in zip(gid_list, reviewed_list)
        if reviewed == 1
    ]

    if len(gid_list) < min_count:
        return None

    while len(gid_list) > min_count:
        index = random.randint(0, len(gid_list) - 1)
        del gid_list[index]

    return gid_list


def __export_reviewed_subset(ibs, min_count=500, ensure_annots=False):
    from os.path import join
    gid_list = find_gid_list(
        ibs, min_count=min_count, ensure_annots=ensure_annots)
    if gid_list is None:
        return None
    new_dbpath = '/' + join('Datasets', 'BACKGROUND', ibs.dbname)
    print('Exporting to %r with %r images' % (new_dbpath, len(gid_list), ))
    return export_images(ibs, gid_list, new_dbpath=new_dbpath)


def export_images(ibs, gid_list, new_dbpath=None):
    """
    exports a subset of images and other required info

    TODO:
        PZ_Master1 needs to backproject information back on to NNP_Master3 and PZ_Master0

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):  list of annotation rowids
        new_dbpath (None): (default = None)

    Returns:
        str: new_dbpath
    """
    print('Exporting image gid_list=%r' % (gid_list,))
    if new_dbpath is None:
        new_dbpath = make_new_dbpath(ibs, 'gid', gid_list)

    aid_list = ut.unique_unordered(ut.flatten(ibs.get_image_aids(gid_list)))
    nid_list = ut.unique_unordered(ibs.get_annot_nids(aid_list))

    return export_data(ibs, gid_list, aid_list, nid_list, new_dbpath=new_dbpath)


def export_annots(ibs, aid_list, new_dbpath=None):
    """
    exports a subset of annotations and other required info

    TODO:
        PZ_Master1 needs to backproject information back on to NNP_Master3 and PZ_Master0

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids
        new_dbpath (None): (default = None)

    Returns:
        str: new_dbpath

    CommandLine:
        python -m ibeis.dbio.export_subset --exec-export_annots
        python -m ibeis.expt.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd
        python -m ibeis.expt.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd

        python -m ibeis.dbio.export_subset --exec-export_annots --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd --new_dbpath=PZ_ViewPoints

        python -m ibeis.expt.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a default:aids=all,is_known=True,view_pername=#primary>0&#primary1>0,per_name=4,size=200
        python -m ibeis.expt.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a default:aids=all,is_known=True,view_pername='#primary>0&#primary1>0',per_name=4,size=200 --acfginfo

    Example:
        >>> # SCRIPT
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> import ibeis
        >>> from ibeis.expt import experiment_helpers
        >>> ibs = ibeis.opendb(defaultdb='NNP_Master3')
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=[''])
        >>> acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, acfg_name_list)
        >>> aid_list = expanded_aids_list[0][0]
        >>> ibs.print_annot_stats(aid_list, yawtext_isect=True, per_image=True)
        >>> # Expand to get all annots in each chosen image
        >>> gid_list = ut.unique_ordered(ibs.get_annot_gids(aid_list))
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> ibs.print_annot_stats(aid_list, yawtext_isect=True, per_image=True)
        >>> new_dbpath = ut.get_argval('--new-dbpath', default='PZ_ViewPoints')
        >>> new_dbpath = export_annots(ibs, aid_list, new_dbpath)
        >>> result = ('new_dbpath = %s' % (str(new_dbpath),))
        >>> print(result)
    """
    print('Exporting annotations aid_list=%r' % (aid_list,))
    if new_dbpath is None:
        new_dbpath = make_new_dbpath(ibs, 'aid', aid_list)

    gid_list = ut.unique_unordered(ibs.get_annot_gids(aid_list))
    nid_list = ut.unique_unordered(ibs.get_annot_nids(aid_list))

    return export_data(ibs, gid_list, aid_list, nid_list, new_dbpath=new_dbpath)


def export_data(ibs, gid_list, aid_list, nid_list, new_dbpath=None):
    """
    exports a subset of data and other required info

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):  list of image rowids
        aid_list (list):  list of annotation rowids
        nid_list (list):  list of name rowids
        imgsetid_list (list):  list of imageset rowids
        gsgrid_list (list):  list of imageset-image pairs rowids
        new_dbpath (None): (default = None)

    Returns:
        str: new_dbpath
    """
    import ibeis

    imgsetid_list = ut.unique_unordered(ut.flatten(ibs.get_image_imgsetids(gid_list)))
    gsgrid_list = ut.unique_unordered(
        ut.flatten(ibs.get_image_gsgrids(gid_list)))

    annotmatch_rowid_list = ibs._get_all_annotmatch_rowids()
    flags1_list = [
        aid in set(aid_list) for aid in ibs.get_annotmatch_aid1(annotmatch_rowid_list)]
    flags2_list = [
        aid in set(aid_list) for aid in ibs.get_annotmatch_aid2(annotmatch_rowid_list)]
    flag_list = ut.and_lists(flags1_list, flags2_list)
    annotmatch_rowid_list = ut.compress(annotmatch_rowid_list, flag_list)
    #annotmatch_rowid_list = ibs.get_valid_aids(ibs.get_valid_aids())

    rowid_subsets = {
        const.ANNOTATION_TABLE: aid_list,
        const.NAME_TABLE: nid_list,
        const.IMAGE_TABLE: gid_list,
        const.ANNOTMATCH_TABLE: annotmatch_rowid_list,
        const.GSG_RELATION_TABLE: gsgrid_list,
        const.IMAGESET_TABLE: imgsetid_list,
    }
    ibs_dst = ibeis.opendb(dbdir=new_dbpath, allow_newdir=True)
    # Main merge driver
    merge_databases2(ibs, ibs_dst, rowid_subsets=rowid_subsets)
    print('Exported to %r' % (new_dbpath,))
    return new_dbpath


# RELEVANT LEGACY CODE FOR IMAGE MERGING
# def check_conflicts(ibs_src, ibs_dst, transfer_data):
#     """
#     Check to make sure the destination database does not have any conflicts
#     with the incoming transfer.

#     Currently only checks that images do not have conflicting annotations.

#     Does not check label consistency.
#     """

# TODO: Check label consistency: ie check that labels with the
# same (type, value) should also have the same UUID
#     img_td      = transfer_data.img_td
# annot_td    = transfer_data.annot_td
# lblannot_td = transfer_data.lblannot_td
# alr_td      = transfer_data.alr_td

#     image_uuid_list1 = img_td.img_uuid_list
#     sameimg_gid_list2_ = ibs_dst.get_image_gids_from_uuid(image_uuid_list1)
#     issameimg = [gid is not None for gid in sameimg_gid_list2_]
# Check if databases contain the same images
#     if any(issameimg):
#         sameimg_gid_list2 = ut.filter_items(sameimg_gid_list2_, issameimg)
#         sameimg_image_uuids = ut.filter_items(image_uuid_list1, issameimg)
#         print('! %d/%d images are duplicates' % (len(sameimg_gid_list2), len(image_uuid_list1)))
# Check if sameimg images in dst has any annotations.
#         sameimg_aids_list2 = ibs_dst.get_image_aids(sameimg_gid_list2)
#         hasannots = [len(aids) > 0 for aids in sameimg_aids_list2]
#         if any(hasannots):
# TODO: Merge based on some merge stratagy parameter (like annotation timestamp)
#             sameimg_gid_list1 = ibs_src.get_image_gids_from_uuid(sameimg_image_uuids)
#             hasannot_gid_list2 = ut.filter_items(sameimg_gid_list2, hasannots)
#             hasannot_gid_list1 = ut.filter_items(sameimg_gid_list1, hasannots)
#             print('  !! %d/%d of those have annotations' %
#             (len(hasannot_gid_list2), len(sameimg_gid_list2)))
# They had better be the same annotations!
#             assert_images_have_same_annnots(ibs_src, ibs_dst,
#             hasannot_gid_list1, hasannot_gid_list2)
#             print('  ...phew, all of the annotations were the same.')
# raise AssertionError('dst dataset contains some of this data')


# def assert_images_have_same_annnots(ibs_src, ibs_dst, hasannot_gid_list1, hasannot_gid_list2):
#     """ Given a list of gids from each ibs, this function asserts that every
#         annontation in gid1 is the same as every annontation in gid2
#     """
#     from ibeis.other.ibsfuncs import unflat_map
#     hasannot_aids_list1 = ibs_src.get_image_aids(hasannot_gid_list1)
#     hasannot_aids_list2 = ibs_dst.get_image_aids(hasannot_gid_list2)
#     hasannot_auuids_list1 = unflat_map(ibs_src.get_annot_uuids, hasannot_aids_list1)
#     hasannot_auuids_list2 = unflat_map(ibs_dst.get_annot_uuids, hasannot_aids_list2)
#     hasannot_verts_list1 = unflat_map(ibs_src.get_annot_verts, hasannot_aids_list1)
#     hasannot_verts_list2 = unflat_map(ibs_dst.get_annot_verts, hasannot_aids_list2)
#     assert_same_annot_uuids(hasannot_auuids_list1, hasannot_auuids_list2)
# assert_same_annot_verts(hasannot_verts_list1, hasannot_verts_list2)  #
# hack, check verts as well


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


def test_merge():
    r"""
    CommandLine:
        python -m ibeis.dbio.export_subset --test-test_merge

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.dbio.export_subset import *  # NOQA
        >>> result = test_merge()
        >>> print(result)
    """
    from ibeis.dbio import export_subset
    import ibeis
    ibs1 = ibeis.opendb('testdb2')
    ibs1.fix_invalid_annotmatches()
    ibs_dst = ibeis.opendb(
        db='testdb_dst2', allow_newdir=True, delete_ibsdir=True)
    export_subset.merge_databases2(ibs1, ibs_dst)
    #ibs_src = ibs1
    check_merge(ibs1, ibs_dst)

    ibs2 = ibeis.opendb('testdb1')
    ibs1.print_dbinfo()
    ibs2.print_dbinfo()
    ibs_dst.print_dbinfo()

    ibs_dst.print_dbinfo()

    export_subset.merge_databases2(ibs2, ibs_dst)
    #ibs_src = ibs2
    check_merge(ibs2, ibs_dst)

    ibs3 = ibeis.opendb('PZ_MTEST')
    export_subset.merge_databases2(ibs3, ibs_dst)
    #ibs_src = ibs2
    check_merge(ibs3, ibs_dst)

    ibs_dst.print_dbinfo()
    return ibs_dst

    #ibs_src.print_annotation_table(exclude_columns=['annot_verts',
    #'annot_semantic_uuid', 'annot_note', 'annot_parent_rowid',
    #'annot_exemplar_flag,'])
    # ibs_dst.print_annotation_table()


def check_database_overlap(ibs1, ibs2):
    """
    CommandLine:
        python -m ibeis.other.dbinfo --test-get_dbinfo:1 --db PZ_MTEST
        dev.py -t listdbs
        python -m ibeis.dbio.export_subset --exec-check_database_overlap
        --db PZ_MTEST --db2 PZ_MOTHERS

    CommandLine:
        python -m ibeis.dbio.export_subset --exec-check_database_overlap

        python -m ibeis.dbio.export_subset --exec-check_database_overlap --db1=PZ_MTEST --db2=PZ_Master0  # NOQA
        python -m ibeis.dbio.export_subset --exec-check_database_overlap --db1=NNP_Master3 --db2=PZ_Master0  # NOQA

        python -m ibeis.dbio.export_subset --exec-check_database_overlap --db1=GZ_Master0 --db2=GZ_ALL
        python -m ibeis.dbio.export_subset --exec-check_database_overlap --db1=GZ_ALL --db2=lewa_grevys

        python -m ibeis.dbio.export_subset --exec-check_database_overlap --db1=PZ_FlankHack --db2=PZ_Master1


    Example:
        >>> # SCRIPT
        >>> import ibeis
        >>> #ibs1 = ibeis.opendb(db='PZ_Master0')
        >>> #ibs2 = ibeis.opendb(dbdir='/raid/work2/Turk/PZ_Master')
        >>> db1 = ut.get_argval('--db1', str, default='PZ_MTEST')
        >>> db2 = ut.get_argval('--db2', str, default='testdb1')
        >>> dbdir1 = ut.get_argval('--dbdir1', str, default=None)
        >>> dbdir2 = ut.get_argval('--dbdir2', str, default=None)
        >>> ibs1 = ibeis.opendb(db=db1, dbdir=dbdir1)
        >>> ibs2 = ibeis.opendb(db=db2, dbdir=dbdir2)
        >>> check_database_overlap(ibs1, ibs2)
    """
    import numpy as np
    import vtool as vt

    def print_intersection(uuids1, uuids2, lbl=''):
        uuids1_ = set(uuids1)
        uuids2_ = set(uuids2)
        uuids_isect = uuids1_.intersection(uuids2_)
        print('Checking {lbl} intersection'.format(lbl=lbl))
        fmtkw1 = dict(
            lbl=lbl, num=len(uuids1_), percent=100 * len(uuids_isect) / len(uuids1_))
        fmtkw2 = dict(
            lbl=lbl, num=len(uuids2_), percent=100 * len(uuids_isect) / len(uuids2_))
        print(
            '  * Num {lbl} 1: {num}, Percentage {percent:.2f}%'.format(**fmtkw1))
        print(
            '  * Num {lbl} 2: {num}, Percentage {percent:.2f}%'.format(**fmtkw2))
        print(
            '  * Num {lbl} isect: {num}'.format(lbl=lbl, num=len(uuids_isect)))
        x_list1 = ut.find_list_indexes(uuids1, uuids_isect)
        x_list2 = ut.find_list_indexes(uuids2, uuids_isect)
        return x_list1, x_list2

    gids1 = ibs1.get_valid_gids()
    gids2 = ibs2.get_valid_gids()
    image_uuids1 = ibs1.get_image_uuids(gids1)
    image_uuids2 = ibs2.get_image_uuids(gids2)
    gx_list1, gx_list2 = print_intersection(
        image_uuids1, image_uuids2, 'images')
    gids_isect1 = ut.take(gids1, gx_list1)
    gids_isect2 = ut.take(gids2, gx_list2)
    SHOW_ISECT_GIDS = False
    if SHOW_ISECT_GIDS:
        if len(gx_list1) > 0:
            print('gids_isect1 = %r' % (gids_isect1,))
            print('gids_isect2 = %r' % (gids_isect2,))
            if False:
                # Debug code
                import ibeis.viz
                import plottool as pt
                gid_pairs = list(zip(gids_isect1, gids_isect2))
                pairs_iter = ut.ichunks(gid_pairs, chunksize=8)
                for fnum, pairs in enumerate(pairs_iter, start=1):
                    pnum_ = pt.make_pnum_nextgen(nRows=len(pairs), nCols=2)
                    for gid1, gid2 in pairs:
                        ibeis.viz.show_image(
                            ibs1, gid1, pnum=pnum_(), fnum=fnum)
                        ibeis.viz.show_image(
                            ibs2, gid2, pnum=pnum_(), fnum=fnum)

    aids1 = ibs1.get_valid_aids()
    aids2 = ibs2.get_valid_aids()
    if False:
        ibs1.update_annot_visual_uuids(aids1)
        ibs2.update_annot_visual_uuids(aids2)
        ibs1.update_annot_semantic_uuids(aids1)
        ibs2.update_annot_semantic_uuids(aids2)

    # Check to see which intersecting images have different annotations
    image_aids_isect1 = ibs1.get_image_aids(gids_isect1)
    image_aids_isect2 = ibs2.get_image_aids(gids_isect2)
    image_avuuids_isect1 = np.array(
        ibs1.unflat_map(ibs1.get_annot_visual_uuids, image_aids_isect1))
    image_avuuids_isect2 = np.array(
        ibs2.unflat_map(ibs2.get_annot_visual_uuids, image_aids_isect2))
    changed_image_xs = np.nonzero(
        image_avuuids_isect1 != image_avuuids_isect2)[0]
    if len(changed_image_xs) > 0:
        print('There are %d images with changes in annotation visual information' % (
            len(changed_image_xs),))
        changed_gids1 = ut.take(gids_isect1, changed_image_xs)
        changed_gids2 = ut.take(gids_isect2, changed_image_xs)

        SHOW_CHANGED_GIDS = False
        if SHOW_CHANGED_GIDS:
            print('gids_isect1 = %r' % (changed_gids2,))
            print('gids_isect2 = %r' % (changed_gids1,))
            if False:
                # Debug code
                import ibeis.viz
                import plottool as pt
                gid_pairs = list(zip(changed_gids1, changed_gids2))
                pairs_iter = ut.ichunks(gid_pairs, chunksize=8)
                for fnum, pairs in enumerate(pairs_iter, start=1):
                    pnum_ = pt.make_pnum_nextgen(nRows=len(pairs), nCols=2)
                    for gid1, gid2 in pairs:
                        ibeis.viz.show_image(
                            ibs1, gid1, pnum=pnum_(), fnum=fnum)
                        ibeis.viz.show_image(
                            ibs2, gid2, pnum=pnum_(), fnum=fnum)

    # Check for overlapping annotations (visual info only) in general
    annot_vuuids1 = ibs1.get_annot_visual_uuids(aids1)
    annot_vuuids2 = ibs2.get_annot_visual_uuids(aids2)
    avx_list1, avx_list2 = print_intersection(
        annot_vuuids1, annot_vuuids2, 'vuuids')

    # Check for overlapping annotations (visual + semantic info) in general
    annot_suuids1 = ibs1.get_annot_semantic_uuids(aids1)
    annot_suuids2 = ibs2.get_annot_semantic_uuids(aids2)
    asx_list1, asx_list2 = print_intersection(
        annot_suuids1, annot_suuids2, 'suuids')

    # Check which images with the same visual uuids have different semantic
    # uuids
    changed_ax_list1 = ut.setdiff_ordered(avx_list1, asx_list1)
    changed_ax_list2 = ut.setdiff_ordered(avx_list2, asx_list2)
    assert len(changed_ax_list1) == len(changed_ax_list2)
    assert ut.take(annot_vuuids1, changed_ax_list1) == ut.take(
        annot_vuuids2, changed_ax_list2)

    changed_aids1 = np.array(ut.take(aids1, changed_ax_list1))
    changed_aids2 = np.array(ut.take(aids2, changed_ax_list2))

    changed_sinfo1 = ibs1.get_annot_semantic_uuid_info(changed_aids1)
    changed_sinfo2 = ibs2.get_annot_semantic_uuid_info(changed_aids2)
    sinfo1_arr = np.array(changed_sinfo1)
    sinfo2_arr = np.array(changed_sinfo2)
    is_semantic_diff = sinfo2_arr != sinfo1_arr
    # Inspect semantic differences
    if np.any(is_semantic_diff):
        colxs, rowxs = np.nonzero(is_semantic_diff)
        colx2_rowids = ut.group_items(rowxs, colxs)
        prop2_rowids = ut.map_dict_keys(
            changed_sinfo1._fields.__getitem__, colx2_rowids)
        print('changed_value_counts = ' +
              ut.dict_str(ut.map_dict_vals(len, prop2_rowids)))
        ut.embed()
        yawx = changed_sinfo1._fields.index('yaw')

        # Show change in viewpoints
        if len(colx2_rowids[yawx]) > 0:
            vp_category_diff = ibsfuncs.viewpoint_diff(
                sinfo1_arr[yawx], sinfo2_arr[yawx]).astype(np.float)
            # Look for category changes
            #any_diff = np.floor(vp_category_diff) > 0
            #_xs    = np.nonzero(any_diff)[0]
            #_aids1 = changed_aids1.take(_xs)
            #_aids2 = changed_aids2.take(_xs)
            # Look for significant changes
            is_significant_diff = np.floor(vp_category_diff) > 1
            significant_xs = np.nonzero(is_significant_diff)[0]
            significant_aids1 = changed_aids1.take(significant_xs)
            significant_aids2 = changed_aids2.take(significant_xs)
            print('There are %d significant viewpoint changes' %
                  (len(significant_aids2),))
            #vt.ori_distance(sinfo1_arr[yawx], sinfo2_arr[yawx])
            #zip(ibs1.get_annot_yaw_texts(significant_aids1),
            #ibs2.get_annot_yaw_texts(significant_aids2))
            # print('yawdiff = %r' % )
            # if False:
            # Hack: Apply fixes
            # good_yaws = ibs2.get_annot_yaws(significant_aids2)
            # ibs1.set_annot_yaws(significant_aids1, good_yaws)
            #    pass
            if False:
                # Debug code
                import ibeis.viz
                import plottool as pt
                #aid_pairs = list(zip(_aids1, _aids2))
                aid_pairs = list(zip(significant_aids1, significant_aids2))
                pairs_iter = ut.ichunks(aid_pairs, chunksize=8)
                for fnum, pairs in enumerate(pairs_iter, start=1):
                    pnum_ = pt.make_pnum_nextgen(nRows=len(pairs), nCols=2)
                    for aid1, aid2 in pairs:
                        ibeis.viz.show_chip(
                            ibs1, aid1, pnum=pnum_(), fnum=fnum, show_yawtext=True, nokpts=True)
                        ibeis.viz.show_chip(
                            ibs2, aid2, pnum=pnum_(), fnum=fnum, show_yawtext=True, nokpts=True)

    #
    nAnnots_per_image1 = np.array(ibs1.get_image_num_annotations(gids1))
    nAnnots_per_image2 = np.array(ibs2.get_image_num_annotations(gids2))
    #
    images_without_annots1 = sum(nAnnots_per_image1 == 0)
    images_without_annots2 = sum(nAnnots_per_image2 == 0)
    print('images_without_annots1 = %r' % (images_without_annots1,))
    print('images_without_annots2 = %r' % (images_without_annots2,))

    nAnnots_per_image1

    class AlignedIndex(object):

        def __init__(self):
            self.iddict_ = {}

        def make_aligned_arrays(self, id_lists, data_lists):
            idx_lists = [vt.compute_unique_data_ids_(
                id_list, iddict_=self.iddict_) for id_list in id_lists]
            aligned_data = []
            for idx_list, data_array in zip(idx_lists, data_lists):
                array = np.full(len(self.iddict_), None)
                array[idx_list] = data_array
                aligned_data.append(array)
            return aligned_data

    # Try to figure out the conflicts
    # TODO: finishme

    self = AlignedIndex()
    id_lists = (image_uuids1, image_uuids2)
    data_lists = (nAnnots_per_image1, nAnnots_per_image2)
    aligned_data = self.make_aligned_arrays(id_lists, data_lists)
    nAnnots1_aligned, nAnnots2_aligned = aligned_data

    nAnnots_difference = nAnnots1_aligned - nAnnots2_aligned
    nAnnots_difference = np.nan_to_num(nAnnots_difference)
    print('images_with_different_num_annnots = %r' %
          (len(np.nonzero(nAnnots_difference)[0]),))


"""
def MERGE_NNP_MASTER_SCRIPT():
    print(ut.truncate_str(ibs_dst.db.get_table_csv(ibeis.const.ANNOTATION_TABLE,
        exclude_columns=['annot_verts', 'annot_semantic_uuid', 'annot_note', 'annot_parent_rowid']), 10000))
    print(ut.truncate_str(ibs_src1.db.get_table_csv(ibeis.const.ANNOTATION_TABLE,
        exclude_columns=['annot_verts', 'annot_semantic_uuid', 'annot_note', 'annot_parent_rowid']), 10000))
    print(ut.truncate_str(ibs_src1.db.get_table_csv(ibeis.const.ANNOTATION_TABLE), 10000))

    from ibeis.dbio.export_subset import *  # NOQA
    import ibeis
    # Step 1
    ibs_src1 = ibeis.opendb('GZC')
    ibs_dst = ibeis.opendb('NNP_Master3', allow_newdir=True)
    merge_databases2(ibs_src1, ibs_dst)

    ibs_src2 = ibeis.opendb('NNP_initial')
    merge_databases2(ibs_src2, ibs_dst)

    ## Step 2
    #ibs_src = ibeis.opendb('GZC')

    ## Check
    ibs1 = ibeis.opendb('NNP_initial')
    ibs2 = ibeis.opendb('GZC')
    ibs3 = ibs_dst

    #print(ibs1.get_image_time_statstr())
    #print(ibs2.get_image_time_statstr())
    #print(ibs3.get_image_time_statstr())
"""


if __name__ == '__main__':
    """
    python -m ibeis.dbio.export_subset
    python -m ibeis.dbio.export_subset --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
