# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Converts a Vulcan-style raw data to IBEIS database."""
from __future__ import absolute_import, division, print_function
from detecttools.directory import Directory
from os.path import join, splitext, exists, basename, split
import utool as ut
import ibeis
import json


SPECIES_MAPPING = {
    'baby elephant'    : 'elephant_savanna_baby',
    'carcasses'        : 'carcass_generic',
    'cat'              : 'cat_domestic',
    'cow'              : 'cow_domestic',
    'elephant'         : 'elephant_savanna',
    'elephant carcass' : 'elephant_savanna_carcass',
    'giraffe'          : 'giraffe_generic',
    'horse'            : 'horse_domestic',
    'human'            : 'person',
    'building'         : 'building',
    'hut'              : 'building_hut',
    'manmade'          : 'building_other',
    'other animal'     : 'other',
    'vehicle'          : 'car',
    'zebra'            : 'zebra_generic',
}


def _convert_vulcan_to_ibeis(vulcan_path, dbdir=None, purge=False,
                             purge_existing_annotations=True, dry_run=False,
                             ignore_directory_list=[], auto_localize=True,
                             ensure_image=True, **kwargs):
    if purge:
        ut.delete(dbdir)

    ibs = ibeis.opendb(dbdir=dbdir)

    direct = Directory(vulcan_path, recursive=True)

    global_species_set = set([])
    global_dict = {}
    key_list = ['annotations', 'config', 'ignoreFilenameList', 'metadata']
    for directory in direct.directory_list:
        print(directory)

        if directory.base() in ignore_directory_list:
            print('\tSkipping Directory')
            continue

        assert len(directory.directory_list) == 0

        image_list = []
        json_list = []
        other_list = []
        for file in directory.files():
            base, ext = splitext(file)
            ext = ext.lower().strip('.')
            if ext in ['json']:
                json_list.append(file)
            elif ext in ['jpg', 'jpeg', 'png', 'tiff']:
                image_list.append(file)
            elif ext in ['ignore']:
                print('Ignoring %r' % (file, ))
                pass
            else:
                other_list.append(file)

        assert len(json_list) == 1
        assert len(other_list) == 0

        json_filepath = json_list[0]
        with open(json_filepath, 'r') as json_file:
            json_dict = json.load(json_file)

        for key in key_list:
            if key not in json_dict:
                print('\tMissing JSON Key: %r' % (key, ))

        extra_key_set = set(json_dict.keys()) - set(key_list)
        if len(extra_key_set) > 0:
            print('\tExtra JSON keys: %r' % (extra_key_set, ))

        assert 'annotations' in json_dict

        filename_list = []
        for image_filepath in image_list:
            filename_list.append(basename(image_filepath))

        ignore_list = json_dict.get('ignoreFilenameList', [])

        filename_set = set(filename_list)

        unknown_ignored_list = set(ignore_list) - filename_set
        if len(unknown_ignored_list) > 0:
            print('\tMissing Ignored Files: %r' % (len(unknown_ignored_list), ))

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

            filename_set = filename_set - set([filename])

        if len(missing_list) > 0:
            print('\tMissing %d Files' % (len(missing_list), ))

        if len(ignore_list) > 0:
                print('\tIgnoring %d Files' % (len(ignore_list), ))

        unregistered_list = list(filename_set)
        if len(unregistered_list) > 0:
                print('\tUnregistered %d Files' % (len(unregistered_list), ))

        metadata_dict = {}
        for key in key_list:
            metadata_dict[key] = json_dict.get(key, {})

        for unregistered in unregistered_list:
            assert unregistered not in annotation_dict
            annotation_dict[unregistered] = []

        for filename in unregistered_list:
            filepath = join(directory.absolute_directory_path, filename)
            filepath_list.append(filepath)

        imageset_text = directory.base()
        assert imageset_text not in global_dict
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

        args = (skipped, )
        print('\tSkipped %d annotations' % args)

    args = (len(global_species_set), global_species_set, )
    print('Found %d species: %r' % args)

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
            gid_list = ibs.add_images(filepath_list, auto_localize=auto_localize,
                                      ensure_loadable=ensure_image,
                                      ensure_exif=ensure_image)
            all_gid_list += gid_list

            ibs.set_image_imgsetids(gid_list, [imageset_rowid] * len(gid_list))

            for filepath, gid in zip(filepath_list, gid_list):
                local_bbox_list = global_dict[imageset_text]['image_dict'][filepath]['bbox_list']
                local_species_list = global_dict[imageset_text]['image_dict'][filepath]['species_list']
                assert len(local_bbox_list) == len(local_species_list)
                local_gid_list = [gid] * len(local_bbox_list)

                global_gid_list += local_gid_list
                global_bbox_list += local_bbox_list
                global_species_list += local_species_list

        if purge_existing_annotations:
            all_gid_list  = list(set(all_gid_list))
            all_aids_list = ibs.get_image_aids(all_gid_list)
            all_aid_list  = ut.flatten(all_aids_list)
            all_aid_list  = list(set(all_aid_list))
            ibs.delete_annots(all_aid_list)

        assert len(global_gid_list) == len(global_bbox_list)
        assert len(global_gid_list) == len(global_species_list)
        ibs.add_annots(global_gid_list, bbox_list=global_bbox_list, species_list=global_species_list)

    return all_gid_list


def _process_vulcan_sequence_metadata(vulcan_path, ibs, gid_list, **kwargs):
    import xlrd

    direct = Directory(vulcan_path, recursive=True)

    duplicate_filename_list = []
    duplicate_metadata_list = []

    metadata_dict = {}
    for directory in direct.directory_list:
        if directory.base() not in ['Metadata']:
            continue
        print(directory)

        xlsx_list = []
        other_list = []
        for file in directory.files():
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
                if sheet.nrows > 0:
                    header_list = sheet.row_values(0)
                    header_list = [header.strip() for header in header_list]
                    header_str = ''.join(header_list)
                    row_list = []
                    if len(header_str) > 0:
                        for row_index in range(1, sheet.nrows):
                            row = sheet.row_values(row_index)
                            row_str = ''.join([str(_).strip() for _ in row])
                            if len(row_str) > 0:
                                row_list.append(row)
                    else:
                        header_list = None
                    print(sheet, sheet.nrows)
                    print(header_list)
                    print(len(row_list))

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
    filename_list = [
        splitext(split(filepath_original)[1])[0]
        for filepath_original in filepath_original_list
    ]
    assert len(filename_list) == len(set(filename_list))
    gid_dict = dict(zip(filename_list, gid_list))

    num_duplcates = len(duplicate_filename_list)
    num_missing_metadata = 0
    num_missing_images = 0

    gid_list = []
    metadata_list = []
    key_list = list(set(gid_dict.keys()) | set(metadata_dict.keys()))
    for key in key_list:
        gid = gid_dict.get(key, None)
        metadata = metadata_dict.get(key, None)
        if gid is None:
            num_missing_images += 1
        if metadata is None:
            num_missing_metadata += 1
        if None not in [gid, metadata]:
            gid_list.append(gid)
            metadata_list.append(metadata)
    assert len(gid_list) == len(metadata_list)

    ibs.set_image_metadata(gid_list, metadata_list)
    print(num_duplcates, num_missing_metadata, num_missing_images)


def convert_vulcan2018_to_ibeis(vulcan_path, dbdir=None, **kwargs):
    r"""Convert the raw 2018 Vulcan data to an ibeis database.

    Args
        vulcan_path (str): Directory to folder *containing* raw Vulcan 2018 data
        dbdir (str): Output directory

    CommandLine:
        python -m ibeis convert_vulcan_to_ibeis

    Example:
        >>> # SCRIPT
        >>> from ibeis.dbio.ingest_vulcan import *  # NOQA
        >>> default_vulcan_path = join('/', 'data', 'raw', 'processed', 'Vulcan_Elephants_2018')
        >>> default_dbdir = join('/', 'data', 'ibeis', 'ELPH_Vulcan')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> vulcan_path = ut.get_argval('--vulcan', type_=str, default=default_vulcan_path)
        >>> result = convert_vulcan2018_to_ibeis(vulcan_path, dbdir=dbdir, purge=False, dry_run=False)
        >>> print(result)
    """
    ibs, gid_list = _convert_vulcan_to_ibeis(vulcan_path, dbdir, **kwargs)
    return ibs, gid_list


def convert_vulcan2019_to_ibeis(vulcan_path, dbdir=None, **kwargs):
    r"""Convert the raw 2019 Vulcan data to an ibeis database.

    Args
        vulcan_path (str): Directory to folder *containing* raw Vulcan 2019 data
        dbdir (str): Output directory

    CommandLine:
        python -m ibeis convert_vulcan_to_ibeis

    Example:
        >>> # SCRIPT
        >>> from ibeis.dbio.ingest_vulcan import *  # NOQA
        >>> default_vulcan_path = join('/', 'data', 'raw', 'processed', 'Vulcan_Elephants_2019')
        >>> default_dbdir = join('/', 'data', 'ibeis', 'ELPH_Vulcan')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> vulcan_path = ut.get_argval('--vulcan', type_=str, default=default_vulcan_path)
        >>> result = convert_vulcan2019_to_ibeis(vulcan_path, dbdir=dbdir, purge=False, dry_run=False)
        >>> print(result)
    """
    ignore_directory_list = ['Metadata']
    ibs, gid_list = _convert_vulcan_to_ibeis(vulcan_path, dbdir,
                                             ignore_directory_list=ignore_directory_list,
                                             **kwargs)
    _process_vulcan_sequence_metadata(vulcan_path, ibs, gid_list, **kwargs)
    return ibs, gid_list


def convert_vulcan2019_sequences_to_ibeis(dbdir=None, **kwargs):
    r"""Convert the raw 2019 Vulcan data to an ibeis database.

    Args
        vulcan_path (str): Directory to folder *containing* raw Vulcan 2019 data
        dbdir (str): Output directory

    CommandLine:
        python -m ibeis convert_vulcan_to_ibeis

    Example:
        >>> # SCRIPT
        >>> from ibeis.dbio.ingest_vulcan import *  # NOQA
        >>> default_dbdir = join('/', 'data', 'ibeis', 'ELPH_Vulcan')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> result = convert_vulcan2019_sequences_to_ibeis(dbdir=dbdir, purge=False, dry_run=False)
        >>> print(result)
    """
    vulcan_path_list = [
        '/data/raw/processed/Vulcan_Elephants_2019_Sequence/QENP_201809_29-30/Photos/',
    ]
    for vulcan_path in vulcan_path_list:
        assert exists(vulcan_path)
        ibs, gid_list = _convert_vulcan_to_ibeis(vulcan_path, dbdir,
                                                 dry_run=True, auto_localize=False,
                                                 **kwargs)
    return ibs, gid_list
