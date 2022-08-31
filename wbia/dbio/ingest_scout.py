# -*- coding: utf-8 -*-
"""Converts a Scout-style raw data to WBIA database."""
import json
import logging
import random
from os.path import basename, exists, join, split, splitext

import utool as ut

import wbia
from wbia.detecttools.directory import Directory

logger = logging.getLogger('wbia')


SPECIES_MAPPING = {
    'baby elephant': 'elephant_savanna_baby',
    'carcasses': 'carcass_generic',
    'cat': 'cat_domestic',
    'cow': 'cow_domestic',
    'elephant': 'elephant_savanna',
    'elephant carcass': 'elephant_savanna_carcass',
    'giraffe': 'giraffe_generic',
    'horse': 'horse_domestic',
    'human': 'person',
    'building': 'building',
    'hut': 'building_hut',
    'manmade': 'building_other',
    'other animal': 'other',
    'vehicle': 'car',
    'zebra': 'zebra_generic',
}


def _convert_scout_to_wbia(
    scout_path,
    dbdir=None,
    purge=False,
    purge_existing_annotations=True,
    dry_run=False,
    ignore_directory_list=[],
    auto_localize=False,
    ensure_image=True,
    recursive=False,
    layer=1,
    scout_tag=None,
    **kwargs
):
    def _walk(direct_list):
        direct_list_ = []
        for direct_ in direct_list:
            logger.info('Processing {}'.format(direct_))
            found = False
            for file_ in direct_.file_list:
                base, ext = splitext(file_)
                ext = ext.lower().strip('.')
                if ext in ['json']:
                    found = True
                    break
            if found:
                direct_list_.append(direct_)
            direct_list_ += _walk(direct_.directory_list)

        return direct_list_

    if purge:
        ut.delete(dbdir)

    ibs = wbia.opendb(dbdir=dbdir)

    direct = Directory(scout_path, recursive=True)
    directory_list = direct.directory_list
    if recursive:
        directory_list = _walk(directory_list)

    global_species_set = set()
    global_dict = {}
    key_list = ['annotations', 'config', 'ignoreFilenameList', 'metadata']
    for directory in directory_list:
        logger.info(directory)

        if directory.base() in ignore_directory_list:
            logger.info('\tSkipping Directory')
            continue

        image_list = []
        json_list = []
        other_list = []
        for file in directory.file_list:
            base, ext = splitext(file)
            ext = ext.lower().strip('.')
            if ext in ['json']:
                json_list.append(file)
            elif ext in ['jpg', 'jpeg', 'png', 'tiff']:
                image_list.append(file)
            elif ext in ['ignore']:
                logger.info('Ignoring {!r}'.format(file))
                pass
            else:
                other_list.append(file)

        assert len(directory.directory_list) == 0
        assert len(json_list) == 1
        assert len(other_list) == 0

        json_filepath = json_list[0]
        with open(json_filepath, 'r') as json_file:
            json_dict = json.load(json_file)

        for key in key_list:
            if key not in json_dict:
                logger.info('\tMissing JSON Key: {!r}'.format(key))

        extra_key_set = set(json_dict.keys()) - set(key_list)
        if len(extra_key_set) > 0:
            logger.info('\tExtra JSON keys: {!r}'.format(extra_key_set))

        assert 'annotations' in json_dict

        filename_list = []
        for image_filepath in image_list:
            filename_list.append(basename(image_filepath))

        ignore_list = json_dict.get('ignoreFilenameList', [])

        filename_set = set(filename_list)

        unknown_ignored_list = set(ignore_list) - filename_set
        if len(unknown_ignored_list) > 0:
            logger.info('\tMissing Ignored Files: {!r}'.format(len(unknown_ignored_list)))

        filename_set = filename_set - set(ignore_list)
        annotation_dict = json_dict['annotations']

        filepath_list = []
        missing_list = []
        for filename in sorted(list(annotation_dict.keys())):
            filepath = join(directory.absolute_directory_path, filename)
            if exists(filepath):
                filepath_list.append(filepath)
            else:
                missing_list.append(filepath)

            filename_set = filename_set - {filename}

        if len(missing_list) > 0:
            logger.info('\tMissing %d Files' % (len(missing_list),))

        if len(ignore_list) > 0:
            logger.info('\tIgnoring %d Files' % (len(ignore_list),))

        unregistered_list = list(filename_set)
        if len(unregistered_list) > 0:
            logger.info('\tUnregistered %d Files' % (len(unregistered_list),))

        metadata_dict = {}
        for key in key_list:
            metadata_dict[key] = json_dict.get(key, {})

        for unregistered in unregistered_list:
            assert unregistered not in annotation_dict
            annotation_dict[unregistered] = []

        for filename in unregistered_list:
            filepath = join(directory.absolute_directory_path, filename)
            filepath_list.append(filepath)

        if layer == 1:
            imageset_text = directory.base()
        elif layer == 2:
            imageset_text = directory.absolute_directory_path
            # imageset_text, imageset_text_1 = split(imageset_text)
            # imageset_text, imageset_text_2 = split(imageset_text)
            # imageset_text = '%s %s' % (scout_tag, imageset_text_2, )
            imageset_text = imageset_text.split('/')
            imageset_text = imageset_text[8]

        # assert imageset_text not in global_dict
        if imageset_text in global_dict:
            assert layer in [2]
            for key in metadata_dict:
                value = metadata_dict[key]
                if isinstance(value, list):
                    global_dict[imageset_text]['metadata_dict'][key] += value
                elif isinstance(value, dict):
                    for key_ in value:
                        value_ = metadata_dict[key][key_]
                        global_dict[imageset_text]['metadata_dict'][key][key_] = value_
                else:
                    raise ValueError
        else:
            assert layer in [1, 2]
            global_dict[imageset_text] = {
                'metadata_dict': metadata_dict,
                'image_dict': {},
            }

        skipped = 0
        for filepath in filepath_list:
            assert exists(filepath)
            filename = basename(filepath)
            assert filename in annotation_dict

            if filename in ignore_list:
                continue

            annotation_list = annotation_dict[filename]

            bbox_list_ = []
            species_list_ = []
            for annotation in annotation_list:
                bbox = annotation.get('rectangle', None)
                species = annotation.get('type', None)

                if None in [bbox, species]:
                    skipped += 1
                    continue

                bbox = list(map(int, bbox.strip().split(',')))
                x0, y0, x1, y1 = bbox
                w = x1 - x0
                h = y1 - y0

                if w <= 0 or h <= 0:
                    skipped += 1
                    continue

                bbox = (x0, y0, w, h)
                species = SPECIES_MAPPING.get(species, species)
                global_species_set.add(species)

                bbox_list_.append(bbox)
                species_list_.append(species)

            assert filepath not in global_dict[imageset_text]['image_dict']
            global_dict[imageset_text]['image_dict'][filepath] = {
                'bbox_list': bbox_list_,
                'species_list': species_list_,
            }

        args = (skipped,)
        logger.info('\tSkipped %d annotations' % args)

    args = (
        len(global_species_set),
        global_species_set,
    )
    logger.info('Found %d species: %r' % args)

    imageset_text_list = sorted(list(global_dict.keys()))
    metadata_list = []
    for imageset_text in imageset_text_list:
        metadata = global_dict[imageset_text]['metadata_dict']
        metadata_list.append(metadata)

    if not dry_run:
        imageset_rowid_list = ibs.add_imagesets(imageset_text_list)
        ibs.set_imageset_metadata(imageset_rowid_list, metadata_list)

        all_gid_list = []
        global_gid_list = []
        global_bbox_list = []
        global_species_list = []

        for imageset_text, imageset_rowid in zip(imageset_text_list, imageset_rowid_list):
            filepath_list = sorted(list(global_dict[imageset_text]['image_dict'].keys()))
            gid_list = ibs.add_images(
                filepath_list,
                auto_localize=auto_localize,
                ensure_loadable=ensure_image,
                ensure_exif=ensure_image,
            )
            all_gid_list += gid_list

            ibs.set_image_imgsetids(gid_list, [imageset_rowid] * len(gid_list))

            for filepath, gid in zip(filepath_list, gid_list):
                local_bbox_list = global_dict[imageset_text]['image_dict'][filepath][
                    'bbox_list'
                ]
                local_species_list = global_dict[imageset_text]['image_dict'][filepath][
                    'species_list'
                ]
                assert len(local_bbox_list) == len(local_species_list)
                local_gid_list = [gid] * len(local_bbox_list)

                global_gid_list += local_gid_list
                global_bbox_list += local_bbox_list
                global_species_list += local_species_list

        if purge_existing_annotations:
            all_gid_list = list(set(all_gid_list))
            all_aids_list = ibs.get_image_aids(all_gid_list)
            all_aid_list = ut.flatten(all_aids_list)
            all_aid_list = list(set(all_aid_list))
            ibs.delete_annots(all_aid_list)

        assert len(global_gid_list) == len(global_bbox_list)
        assert len(global_gid_list) == len(global_species_list)
        ibs.add_annots(
            global_gid_list, bbox_list=global_bbox_list, species_list=global_species_list
        )

    return all_gid_list


def _process_scout_sequence_metadata(
    scout_path, scout_tag, ibs, gid_list, version=1, **kwargs
):
    import xlrd

    direct = Directory(scout_path, recursive=True)

    duplicate_filename_list = []
    duplicate_metadata_list = []

    metadata_dict = {}
    for directory in direct.directory_list:
        if directory.base() not in ['Metadata']:
            continue
        logger.info(directory)

        xlsx_list = []
        other_list = []
        for file in directory.file_list:
            base, ext = splitext(file)
            ext = ext.lower().strip('.')
            if ext in ['xlsx']:
                xlsx_list.append(file)
            else:
                other_list.append(file)

        assert len(xlsx_list) > 0
        assert len(other_list) == 0

        for xlsx in xlsx_list:
            workbook = xlrd.open_workbook(xlsx)

            for sheet in workbook.sheets():
                header_list = None
                if sheet.nrows > 0:
                    header_list = sheet.row_values(0)
                    first_header = header_list[0].lower()
                    has_photocode = first_header in ['photocode']
                    if has_photocode:
                        header_list = [header.strip() for header in header_list]
                        header_str = ''.join(header_list)
                        row_list = []
                        if len(header_str) > 0:
                            for row_index in range(1, sheet.nrows):
                                row = sheet.row_values(row_index)
                                row_str = ''.join([str(_).strip() for _ in row])
                                if len(row_str) > 0:
                                    row_list.append(row)
                        logger.info(sheet, sheet.nrows)
                        logger.info(header_list)
                        logger.info(len(row_list))
                    else:
                        header_list = None

                if header_list is not None:
                    for row in row_list:
                        metadata = dict(zip(header_list, row))
                        for key in metadata:
                            value = str(metadata[key])
                            if len(value) == 0:
                                metadata[key] = None
                        photocode = metadata.get('Photocode', None)
                        assert photocode is not None
                        photocode = photocode.strip()
                        if photocode not in metadata_dict:
                            metadata_dict[photocode] = metadata
                        else:
                            existing_metadata = metadata_dict[photocode]
                            duplicate_filename_list.append(photocode)
                            duplicate_metadata_list.append((metadata, existing_metadata))

    filepath_original_list = ibs.get_image_uris_original(gid_list)
    if version == 1:
        filename_list = [
            splitext(split(filepath_original)[1])[0]
            for filepath_original in filepath_original_list
        ]
    else:
        missing = 0
        filename_list = []
        seen_set = set()
        for filepath_original in filepath_original_list:
            filepath_original = filepath_original.replace(scout_path, '')
            filepath_original = filepath_original.replace('Photos/', '')
            start = filepath_original.split('/')[0]
            start = start.split('-')
            assert len(start) == 4
            _, day, month, year = start
            assert month == 'March'

            if 'ATS' in filepath_original:
                loc = 'ATS'
            elif 'AKP' in filepath_original:
                loc = 'AKP'
            else:
                raise ValueError

            if 'left' in filepath_original.lower():
                side = '1'
            elif (
                'right' in filepath_original.lower() or '_r_' in filepath_original.lower()
            ):
                side = '2'
            else:
                side = '2'
                continue

            filepath_original = filepath_original.split('/')[-2:]
            folder, tag = filepath_original
            folder = folder[:3]
            tag = tag.replace('.JPG', '').replace('DSC_', '')
            if len(tag) != 4:
                # tag_ = tag[-4:]
                # folder = 100
                # filename = '2017-03-%s-%s-%s-%s-%s' % (day, loc, side, folder, tag_, )
                assert tag.count('-') == 6 and ('ATS' in tag or 'AKP' in tag)
                filename = tag
            else:
                assert len(tag) == 4
                filename = '2017-03-{}-{}-{}-{}-{}'.format(
                    day,
                    loc,
                    side,
                    folder,
                    tag,
                )

            if filename not in metadata_dict:
                missing += 1
            # assert filename in metadata_dict
            if filename in seen_set:
                filename = filename + '-alt'
            seen_set.add(filename)
            filename_list.append(filename)

    assert len(filename_list) == len(set(filename_list))
    gid_dict = dict(zip(filename_list, gid_list))

    num_duplcates = len(duplicate_filename_list)
    num_missing_metadata = 0
    num_missing_images = 0

    gid_list_ = []
    metadata_list_ = []
    key_list = list(set(gid_dict.keys()) | set(metadata_dict.keys()))
    for key in key_list:
        gid = gid_dict.get(key, None)
        metadata = metadata_dict.get(key, None)
        if gid is None:
            num_missing_images += 1
        if metadata is None:
            num_missing_metadata += 1
        if None not in [gid, metadata]:
            gid_list_.append(gid)
            metadata_list_.append(metadata)
    assert len(gid_list_) == len(metadata_list_)
    logger.info(num_duplcates, num_missing_metadata, num_missing_images)

    # Set image metadata
    ibs.set_image_metadata(gid_list_, metadata_list_)

    # Add imagesets
    skipped = 0
    skipped_list = []
    metadata_dict_ = {}
    for gid, metadata in zip(gid_list_, metadata_list_):
        for key in ['Transect', 'Transect (Amal23)']:
            transect = metadata.get(key, None)
            if transect is not None:
                break

        for key in [
            'Camleft/right',
            'Camleft/right',
            'Camera left/ right',
            'Camera Side',
            'Camera',
        ]:
            photoside = metadata.get(key, None)
            if photoside is not None:
                break

        for key in ['Photo  No.']:
            photonum = metadata.get(key, None)
            if photonum is not None:
                break

        if None not in [transect, photonum, photoside]:
            transect = int(transect)
            photonum = int(photonum)
            photoside = photoside.strip().capitalize()
            assert photoside in ['Left', 'Right']
            if transect not in metadata_dict_:
                metadata_dict_[transect] = {}
            if photoside not in metadata_dict_[transect]:
                metadata_dict_[transect][photoside] = []
            metadata_dict_[transect][photoside].append((photonum, gid))
        else:
            skipped += 1
            skipped_list.append((transect, photoside, photonum, metadata.keys()))

    imageset_image_gid_list = []
    imageset_image_text_list = []
    imageset_imageset_text_list = []
    imageset_imageset_metadata_list = []
    for transect in sorted(metadata_dict_.keys()):
        for photoside in sorted(metadata_dict_[transect].keys()):
            imageset_text = '{} Transect {} ({})'.format(scout_tag, transect, photoside)
            values_list = metadata_dict_[transect][photoside]
            values_list = sorted(values_list)
            temp_list = ut.take_column(values_list, 1)
            imageset_image_gid_list += temp_list
            imageset_image_text_list += [imageset_text] * len(temp_list)
            key_list_ = ['index', 'gid']
            metadata_ = {
                'sequence': [
                    dict(zip(key_list_, value_list)) for value_list in values_list
                ]
            }
            imageset_imageset_text_list.append(imageset_text)
            imageset_imageset_metadata_list.append(metadata_)

    ibs.set_image_imagesettext(imageset_image_gid_list, imageset_image_text_list)
    imageset_imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(
        imageset_imageset_text_list
    )
    ibs.set_imageset_metadata(
        imageset_imageset_rowid_list, imageset_imageset_metadata_list
    )


def convert_scout2018_to_wbia(scout_path, dbdir=None, **kwargs):
    r"""Convert the raw 2018 Scout Phase 1 data to an wbia database.

    Args
        scout_path (str): Directory to folder *containing* raw Scout Phase 1 2018 data
        dbdir (str): Output directory

    CommandLine:
        python -m wbia convert_scout_to_wbia

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_scout import *  # NOQA
        >>> default_scout_path = join('/', 'data', 'raw', 'processed', 'scout_Elephants_S3', 'WildMe')
        >>> default_dbdir = join('/', 'data', 'wbia', 'ELPH_scout')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> scout_path = ut.get_argval('--scout', type_=str, default=default_scout_path)
        >>> result = convert_scout2018_to_wbia(scout_path, dbdir=dbdir, purge=False, dry_run=False)
        >>> logger.info(result)
    """
    ignore_directory_list = ['OlPejeta_2016']
    ibs, gid_list = _convert_scout_to_wbia(
        scout_path, dbdir, ignore_directory_list=ignore_directory_list, **kwargs
    )
    return ibs, gid_list


def convert_scout2019_to_wbia(scout_path, dbdir=None, **kwargs):
    r"""Convert the raw 2019 Scout Phase 1 data to an wbia database.

    Args
        scout_path (str): Directory to folder *containing* raw Scout Phase 1 2019 data
        dbdir (str): Output directory

    CommandLine:
        python -m wbia convert_scout_to_wbia

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_scout import *  # NOQA
        >>> default_scout_path = join('/', 'data', 'raw', 'processed', 'scout_Elephants_S3', 'WildMe', 'OlPejeta_2016')
        >>> default_dbdir = join('/', 'data', 'wbia', 'ELPH_scout')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> scout_path = ut.get_argval('--scout', type_=str, default=default_scout_path)
        >>> result = convert_scout2019_to_wbia(scout_path, dbdir=dbdir, purge=False, dry_run=False)
        >>> logger.info(result)
    """
    ignore_directory_list = ['Metadata']
    ibs, gid_list = _convert_scout_to_wbia(
        scout_path, dbdir, ignore_directory_list=ignore_directory_list, **kwargs
    )
    _process_scout_sequence_metadata(scout_path, 'OlPejeta-2016', ibs, gid_list, **kwargs)
    return ibs, gid_list


def convert_scout2019_sequences_to_wbia(dbdir=None, **kwargs):
    r"""Convert the raw 2019 Scout Phase 1 data to an wbia database.

    Args
        scout_path (str): Directory to folder *containing* raw Scout Phase 1 2019 data
        dbdir (str): Output directory

    CommandLine:
        python -m wbia convert_scout_to_wbia

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_scout import *  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'ELPH_scout')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> result = convert_scout2019_sequences_to_wbia(dbdir=dbdir, purge=False, dry_run=False)
        >>> logger.info(result)
    """
    scout_path_list = [
        (
            '/data/raw/processed/scout_Elephants_2019_Sequence/QENP_201809_29-30/Photos/',
            False,
            'QENP-2018',
            '/data/raw/processed/scout_Elephants_2019_Sequence/QENP_201809_29-30/',
        ),
        (
            '/data/raw/processed/scout_Elephants_2019_Sequence/Tsavo_201703_12/Photos/',
            True,
            'Tsavo-2017',
            '/data/raw/processed/scout_Elephants_2019_Sequence/Tsavo_201703_12/',
        ),
    ]
    for (
        scout_image_path,
        scout_recusrive,
        scout_tag,
        scout_metadata_path,
    ) in scout_path_list:
        assert exists(scout_image_path)
        assert exists(scout_metadata_path)
        layer = 2 if scout_recusrive else 1
        version = 2 if scout_recusrive else 1
        ibs, gid_list = _convert_scout_to_wbia(
            scout_image_path,
            dbdir,
            dry_run=True,
            recursive=scout_recusrive,
            layer=layer,
            scout_tag=scout_tag,
            **kwargs
        )
        _process_scout_sequence_metadata(
            scout_metadata_path, scout_tag, ibs, gid_list, version=version, **kwargs
        )
    return ibs, gid_list


def walk_scout_s3_directory(direct_list, validation_match_str='VALIDATION'):
    direct_list_ = []
    for direct_ in direct_list:
        found = []
        for file_ in direct_.file_list:
            path, filename = split(file_)
            base, ext = splitext(filename)
            ext = ext.lower().strip('.')
            if ext in ['json']:
                if validation_match_str not in base:
                    found.append(file_)
        if len(found) > 0:
            direct_list_.append((direct_, found))
        direct_list_ += walk_scout_s3_directory(direct_.directory_list)
    return direct_list_


def convert_scout_s3_to_wbia(dbdir, auto_localize=False, ensure_image=True):
    r"""
    Training:
        D MWS/WildMe/elephant/
        D MWS/WildMe/RR18_BIG_2015_09_23_R_AM/
        D MWS/WildMe/TA24_TPM_L_2016-10-30-A/
        D MWS/WildMe/TA24_TPM_R_2016-10-30-A/
        D MWS/WildMe/2012-08-16_AM_L_Azohi/
        D MWS/WildMe/2012-08-15_AM_R_Marealle/
        D MWS/WildMe/2012-08-14_PM_R_Chediel/

        D MWS/WildMe/OlPejeta_2016/20161106_Nikon_Left/
        D MWS/WildMe/OlPejeta_2016/20161106_Nikon_Right/
        D MWS/WildMe/OlPejeta_2016/20161108_Nikon_Left/
        D MWS/WildMe/OlPejeta_2016/20161108_Nikon_Right/

        D MWS/QENP_201809_29-30/Photos/
        D MWS/Tsavo_201703_12-24/Photos/
        D MWS/Katavi/Photos/

        ---

        -D MWS/QENP_201809_29-30/Photos/QENP\ 29\ Sep\ 2018\,\ Nikon\ Left/
        -D MWS/QENP_201809_29-30/Photos/QENP\ 29\ Sep\ 2018\,\ Nikon\ Right/
        -D MWS/QENP_201809_29-30/Photos/QENP\ 30\ Sep\ 2018\,\ Nikon\ Left/
        -D MWS/QENP_201809_29-30/Photos/QENP\ 30\ Sep\ 2018\,\ Nikon\ Right/
        -D MWS/Tsavo_201703_12-24/Photos/

        ---

        F MWS/Katavi/Photos/A/sde-A_20180923A/154D5600/sde-a_L_20180923101642.jpg
        F MWS/Katavi/Photos/A/sde-A_20180923A/154D5600/sde-a_L_20180923101644.jpg
        F MWS/Katavi/Photos/A/sde-A_20180923A/154D5600/sde-a_L_20180923101646.jpg
        F MWS/Katavi/Photos/A/sde-A_20180923A/154D5600/sde-a_L_20180923101648.jpg
        F MWS/Katavi/Photos/A/sde-A_20180925A/159D5600/sde-a_L_20180925090108.jpg
        F MWS/Katavi/Photos/A/sde-A_20180925A/159D5600/sde-a_L_20180925090110.jpg
        F MWS/Katavi/Photos/A/sde-A_20180925A/162D5600/sde-a_L_20180925103426.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/sde-a_L_20180928063108.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/164D5600/sde-a_L_20180928065722.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/171D5600/sde-a_L_20180928103238.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/171D5600/sde-a_L_20180928103240.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/171D5600/sde-a_L_20180928103242.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/171D5600/sde-a_L_20180928105040.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/171D5600/sde-a_L_20180928105424.jpg
        F MWS/Katavi/Photos/A/sde-A_20180928A/171D5600/sde-a_L_20180928105426.jpg
        F MWS/Katavi/Photos/A/sde-A_20180929A/171D5600/sde-a_L_20180929070410.jpg
        F MWS/Katavi/Photos/A/sde-A_20180929A/171D5600/sde-a_L_20180929070412.jpg
        F MWS/Katavi/Photos/A/sde-A_20180929A/178D5600/sde-a_L_20180929100236.jpg
        F MWS/Katavi/Photos/A/sde-A_20180929A/178D5600/sde-a_L_20180929100238.jpg
        F MWS/Katavi/Photos/A/sde-A_20180929A/178D5600/sde-a_L_20180929100240.jpg
        F MWS/Katavi/Photos/A/sde-A_20180929A/178D5600/sde-a_L_20180929100242.jpg
        F MWS/Katavi/Photos/A/sde-A_20180930A/179D5600/sde-a_L_20180930070536.jpg
        F MWS/Katavi/Photos/A/sde-A_20180930A/179D5600/sde-a_L_20180930070538.jpg
        F MWS/Katavi/Photos/A/sde-A_20180930A/179D5600/sde-a_L_20180930070544.jpg
        F MWS/Katavi/Photos/A/sde-A_20180930A/179D5600/sde-a_L_20180930070546.jpg
        F MWS/Katavi/Photos/A/sde-A_20181008A/222D5600/sde-a_L_20181008065514.jpg
        F MWS/Katavi/Photos/A/sde-A_20181008A/224D5600/sde-a_L_20181008075354.jpg
        F MWS/Katavi/Photos/A/sde-A_20181008A/229D5600/sde-a_L_20181008103704.jpg
        F MWS/Katavi/Photos/C/sde-C_20180927A/122D5600/sde-c_L_20180927082022.jpg
        F MWS/Katavi/Photos/C/sde-C_20180927A/122D5600/sde-c_L_20180927082024.jpg
        F MWS/Katavi/Photos/C/sde-C_20180927A/122D5600/sde-c_L_20180927082624.jpg
        F MWS/Katavi/Photos/C/sde-C_20180927A/122D5600/sde-c_L_20180927082626.jpg
        F MWS/Katavi/Photos/D/sde-D_20181003A/111D5600/sde-d_R_20181003082548.jpg
        F MWS/Katavi/Photos/D/sde-D_20181003A/111D5600/sde-d_R_20181003082550.jpg
        F MWS/Katavi/Photos/D/sde-D_20181003A/111D5600/sde-d_R_20181003082552.jpg
        F MWS/Katavi/Photos/D/sde-D_20181003A/111D5600/sde-d_R_20181003082554.jpg


    Testing:
        D MWS/OlPejeta_201903/Photos/
        D MWS/Tarangire_201903/Photos/

    Duplicated:
        MWS/OlPejeta_201611_06-08
        MWS/Katavi

    Ignored:
        MWS/WildMe/2012-08-13_rightbitok
        MWS/WildMe/2014-08-19_PM_Manyara_Ranch
        MWS/WildMe/2014-08-20_AM
        MWS/WildMe/2014-08-20_AM_R
        MWS/WildMe/2014-08-20_PM_Tarangire
        MWS/WildMe/2014-08-20_PM_Tarangire_R
        MWS/WildMe/2014-08-21_AM_Tarangire
        MWS/WildMe/2014-08-21_AM_Tarangire_R
        MWS/WildMe/2014-08-21_PM_Tarangire
        MWS/WildMe/2014-08-21_PM_Tarangire_R
        MWS/WildMe/2014-08-22_AM_Tarangire
        MWS/WildMe/2014-08-22_AM_Tarangire_R
        MWS/WildMe/2014-08-22_PM_Tarangire
        MWS/WildMe/2014-08-22_PM_Tarangire_R
        MWS/WildMe/20160915_Botswana_0
        MWS/WildMe/20161010_Botswana_1
        MWS/WildMe/20161013_Botswana_1
        MWS/WildMe/20161013_Botswana_2
        MWS/WildMe/5H-CFA
        MWS/WildMe/5H-SGR
        MWS/WildMe/5H-TWR
        MWS/WildMe/5H-ZGF
        MWS/WildMe/A032C025_1402281O
        MWS/WildMe/A032C026_140228ZV
        MWS/WildMe/all-kenya
        MWS/WildMe/Clip1HTK121
        MWS/WildMe/Clip1HTK151_1
        MWS/WildMe/Clip1HTK154_1
        MWS/WildMe/elephantSet1
        MWS/WildMe/elephantSet2
        MWS/WildMe/elephantSet3
        MWS/WildMe/TA23_MPK_L_2014-08-24-PM-Tarangire
        MWS/WildMe/TA23_MPK_R_2014-08-24-AM-Tarangire
        MWS/WildMe/TA23_MPK_R_2014-08-25-PM-Tarangire
        MWS/WildMe/TA23_MPZ_L_2014-08-20-AM-training
        MWS/WildMe/TA23_MPZ_L_2014-08-21-AM-Kibaoni
        MWS/WildMe/TA23_MPZ_L_2014-08-21-PM-Lolkisale
        MWS/WildMe/TA23_MPZ_L_2014-08-26-AM-Tarangire
        MWS/WildMe/TA23_MPZ_R_2014-08-20-AM-training
        MWS/WildMe/TA23_MPZ_R_2014-08-21-AM-Kibaoni
        MWS/WildMe/TA23_MPZ_R_2014-08-21-PM-Lolkisale
        MWS/WildMe/TA23_MPZ_training_pics
    """
    scout_prefix = '/data/raw/processed/'
    scout_path_list = [
        'MWS/WildMe/elephant/',
        'MWS/WildMe/RR18_BIG_2015_09_23_R_AM/',
        'MWS/WildMe/TA24_TPM_L_2016-10-30-A/',
        'MWS/WildMe/TA24_TPM_R_2016-10-30-A/',
        'MWS/WildMe/2012-08-16_AM_L_Azohi/',
        'MWS/WildMe/2012-08-15_AM_R_Marealle/',
        'MWS/WildMe/2012-08-14_PM_R_Chediel/',
        'MWS/WildMe/OlPejeta_2016/20161106_Nikon_Left/',
        'MWS/WildMe/OlPejeta_2016/20161106_Nikon_Right/',
        'MWS/WildMe/OlPejeta_2016/20161108_Nikon_Left/',
        'MWS/WildMe/OlPejeta_2016/20161108_Nikon_Right/',
        'MWS/QENP_201809_29-30/Photos/',
        'MWS/Tsavo_201703_12-24/Photos/',
        'MWS/Katavi/Photos/',
        # 'OlPejeta_201903/Photos/',
        # 'Tarangire_201903/Photos/',
    ]

    desired_species_set = {
        'elephant_savanna',
        'elephant_savanna_baby',
        'elephant_savanna_carcass',
    }

    ibs = wbia.opendb(dbdir=dbdir)

    # Locate all folders with JSON files (excluding VALIDATION files)
    direct_list = []
    for scout_path in scout_path_list:
        scout_path_absolute = join(scout_prefix, scout_path)
        direct = Directory(scout_path_absolute, recursive=True)
        direct_list += walk_scout_s3_directory([direct] + direct.directory_list)

    # Filter out multiple JSON files for a given folder
    def _filter(json_filepath_):
        if 'annotations_scout_april2019.json' in json_filepath_:
            return True
        if 'annotations_20161106_Nikon_Left_Corrected.json' in json_filepath_:
            return True
        if 'annotations_20161106_Nikon_Right_Corrected.json' in json_filepath_:
            return True
        return False

    json_filepath_list = []
    for direct, json_filepath_list_ in direct_list:
        assert 1 <= len(json_filepath_list_) and len(json_filepath_list_) <= 2

        if len(json_filepath_list_) == 2:
            json_filepath_list_filtered = [
                json_filepath
                for json_filepath in json_filepath_list_
                if _filter(json_filepath)
            ]
            assert len(json_filepath_list_filtered) == 1
            json_filepath_list_ = json_filepath_list_filtered

        assert len(json_filepath_list_) == 1
        json_filepath = json_filepath_list_[0]
        assert 'VALIDATION' not in json_filepath
        json_filepath_list.append(json_filepath)

    # De-duplicate list of JSON files
    json_filepath_list = sorted(set(json_filepath_list))

    # Load all JSON files and ensure their structure
    global_file_ext_set = set()
    global_species_set = set()
    global_image_set = set()
    global_pos_image_filtered_set = set()
    global_neg_image_filtered_set = set()
    global_neg_image_skipped_set = set()

    processed_dict = {}

    raw_file_ext_set = {'nef', 'NEF'}
    ignored_file_ext_set = {'db', 'dbf', 'prj', 'json', 'txt', 'shx', 'ignore', 'shp'}

    key_list = ['annotations', 'config', 'ignoreFilenameList', 'metadata']
    for json_filepath in json_filepath_list:
        error_message_list = []

        json_path, json_filename = split(json_filepath)

        try:
            with open(json_filepath, 'r') as json_file:
                json_dict = json.load(json_file)
        except json.JSONDecodeError:
            continue

        for key in key_list:
            if key not in json_dict:
                error_message_list.append('\tMissing JSON Key: {!r}'.format(key))

        extra_key_set = set(json_dict.keys()) - set(key_list)
        if len(extra_key_set) > 0:
            error_message_list.append('\tExtra JSON keys: {!r}'.format(extra_key_set))

        annotation_dict = json_dict.get('annotations', None)
        ignore_set = set(json_dict.get('ignoreFilenameList', []))

        assert annotation_dict is not None

        direct = Directory(json_path, recursive=False)
        filepath_list = direct.file_list

        filename_list = [basename(image_filepath) for image_filepath in filepath_list]
        filename_set = set(filename_list)

        local_file_ext_set = set()
        for filename in filename_set:
            base, ext = splitext(filename)
            ext = ext.strip('.')
            local_file_ext_set.add(ext)

        if len(ignore_set) > 0:
            # error_message_list.append('\tIgnoring %d Files' % (len(ignore_set), ))
            pass

        unknown_ignored_list = ignore_set - filename_set
        if len(unknown_ignored_list) > 0:
            error_message_list.append(
                '\tMissing %d Ignored Files' % (len(unknown_ignored_list),)
            )

        filename_set = filename_set - ignore_set

        pos_image_set = set()
        neg_image_set = set()
        missing_set = set()
        for filename in sorted(list(annotation_dict.keys())):

            if filename in ignore_set:
                continue

            annotation_list = annotation_dict[filename]

            bbox_list = []
            species_list = []
            for annotation in annotation_list:
                bbox = annotation.get('rectangle', None)
                species = annotation.get('type', None)

                if None in [bbox, species]:
                    continue

                bbox = list(map(int, bbox.strip().split(',')))
                x0, y0, x1, y1 = bbox
                w = x1 - x0
                h = y1 - y0

                if w <= 0 or h <= 0:
                    continue

                bbox = (x0, y0, w, h)
                species = SPECIES_MAPPING.get(species, species)
                global_species_set.add(species)

                if species in desired_species_set:
                    bbox_list.append(bbox)
                    species_list.append(species)

            assert len(bbox_list) == len(species_list)

            filepath = join(direct.absolute_directory_path, filename)

            assert filepath not in processed_dict
            processed_dict[filepath] = (bbox_list, species_list, json_filepath)

            if exists(filepath):
                if len(bbox_list) > 0:
                    pos_image_set.add(filepath)
                else:
                    neg_image_set.add(filepath)
            else:
                missing_set.add(filepath)

            base, ext = splitext(filename)
            ext = ext.strip('.')
            local_file_ext_set.add(ext)

        image_set = pos_image_set | neg_image_set
        global_image_set = global_image_set | image_set

        used_file_ext_set = local_file_ext_set - ignored_file_ext_set
        if len(raw_file_ext_set & used_file_ext_set) > 0:
            error_message_list.append(
                '\tUsing Extensions: {!r}'.format(used_file_ext_set)
            )
        global_file_ext_set = global_file_ext_set | local_file_ext_set

        if len(missing_set) > 0:
            error_message_list.append(
                '\tMissing %d Wanted Files (missing image file)' % (len(missing_set),)
            )

        unregistered_list = list(filename_set - image_set)
        if len(unregistered_list) > 0:
            # error_message_list.append('\tUnregistered %d Files (missing annotations)' % (len(unregistered_list), ))
            pass

        if len(error_message_list) > 0:
            logger.info('Processing: {!r}'.format(json_filepath))
            logger.info('\n'.join(error_message_list))

        num_positive = len(pos_image_set)
        num_negative = len(neg_image_set)
        num_desired = min(num_negative, num_positive)
        neg_image_list = list(neg_image_set)
        random.shuffle(neg_image_list)
        neg_image_filtered_list = neg_image_list[:num_desired]
        neg_image_filtered_set = set(neg_image_filtered_list)

        local_neg_image_skipped_set = neg_image_set - neg_image_filtered_set

        global_pos_image_filtered_set = global_pos_image_filtered_set | pos_image_set
        global_neg_image_filtered_set = (
            global_neg_image_filtered_set | neg_image_filtered_set
        )
        global_neg_image_skipped_set = (
            global_neg_image_skipped_set | local_neg_image_skipped_set
        )

    num_positive = len(global_pos_image_filtered_set)
    num_negative = len(global_neg_image_skipped_set)
    num_desired = min(num_negative, num_positive)
    global_neg_image_skipped_list = list(global_neg_image_skipped_set)
    random.shuffle(global_neg_image_skipped_list)
    global_neg_image_skipped_filtered_list = global_neg_image_skipped_list[:num_desired]
    global_neg_image_skipped_filtered_set = set(global_neg_image_skipped_filtered_list)
    assert len(global_neg_image_skipped_filtered_set & global_neg_image_filtered_set) == 0

    global_neg_image_filtered_total_set = (
        global_neg_image_filtered_set | global_neg_image_skipped_filtered_set
    )
    global_image_filtered_set = (
        global_pos_image_filtered_set | global_neg_image_filtered_total_set
    )

    logger.info('Encountered {!r} extensions'.format(global_file_ext_set))
    args = (
        len(global_species_set),
        global_species_set,
    )
    logger.info('Encountered %d species: %r' % args)
    logger.info('Found %d images' % (len(global_image_set),))
    logger.info('Kept %d images' % (len(global_image_filtered_set),))
    logger.info('\t%d positive' % (len(global_pos_image_filtered_set),))
    logger.info('\t%d negative (within folders)' % (len(global_neg_image_filtered_set),))
    logger.info('\t%d negative (global)' % (len(global_neg_image_skipped_filtered_set),))

    for filepath in global_pos_image_filtered_set:
        assert filepath in processed_dict
        bbox_list, species_list, json_filepath = processed_dict[filepath]
        assert len(bbox_list) == len(species_list)
        assert len(bbox_list) > 0
        assert len(species_list) > 0
        combined_species = set(species_list) | desired_species_set
        assert combined_species == desired_species_set
        assert len(combined_species & desired_species_set) > 0

    for filepath in global_neg_image_filtered_total_set:
        assert filepath in processed_dict
        bbox_list, species_list, json_filepath = processed_dict[filepath]
        assert len(bbox_list) == len(species_list)
        assert len(bbox_list) == 0
        assert len(species_list) == 0

    global_image_filtered_list = sorted(global_pos_image_filtered_set)
    gid_list = ibs.add_images(
        global_image_filtered_list,
        auto_localize=auto_localize,
        ensure_loadable=ensure_image,
        ensure_exif=ensure_image,
    )
    assert None not in gid_list

    imageset_text_list = []
    global_gid_list = []
    global_bbox_list = []
    global_species_list = []
    for gid, filepath in zip(gid_list, global_image_filtered_list):
        bbox_list, species_list, json_filepath = processed_dict[filepath]

        assert json_filepath.startswith(scout_prefix)
        json_filepath = json_filepath.replace(scout_prefix, '')
        imageset_text = None
        for scout_path in scout_path_list:
            if json_filepath.startswith(scout_path):
                imageset_text = scout_path
                break
        assert imageset_text is not None
        imageset_text = imageset_text.replace('/Photos/', '')
        imageset_text = imageset_text.strip('/')
        imageset_text_list.append(imageset_text)

        assert len(bbox_list) == len(species_list)
        global_gid_list += [gid] * len(bbox_list)
        global_bbox_list += bbox_list
        global_species_list += species_list

    ibs.set_image_imagesettext(gid_list, imageset_text_list)

    assert len(global_gid_list) == len(global_bbox_list)
    assert len(global_gid_list) == len(global_species_list)
    aid_list = ibs.add_annots(
        global_gid_list, bbox_list=global_bbox_list, species_list=global_species_list
    )

    return gid_list, aid_list
