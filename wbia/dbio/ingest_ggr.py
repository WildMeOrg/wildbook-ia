# -*- coding: utf-8 -*-
#!/usr/bin/env python  # NOQA
"""Converts a GGR-style raw data to IBEIS database."""
from __future__ import absolute_import, division, print_function
from wbia.detecttools.directory import Directory
from os.path import join, exists
import utool as ut
import wbia

(print, rrr, profile) = ut.inject2(__name__)


def _fix_ggr2018_directory_structure(ggr_path):

    # Manual fixes for bad directories

    src_uri = join(ggr_path, 'Clarine\\ Plane\\ Kurungu/')
    dst_uri = join(ggr_path, '231/')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '231B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(
        ggr_path,
        'Alex\\ Peltier\\ -\\ Plane\\ -\\ Ngurnit/giraffe\\ grevy\\ count\\ feb\\ 18/',
    )
    dst_uri = join(ggr_path, '232/')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '232B/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)

    src_uri = join(
        ggr_path, 'Mint\\ Media\\ Footage', 'Mpala\\ day\\ 1\\ spark', 'PANORAMA/'
    )
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, 'Mint\\ Media\\ Footage', 'Mpala\\ day\\ 1/')
    dst_uri = join(ggr_path, '233/')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '233B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, 'Mint\\ Media\\ Footage', 'Mpala\\ day\\ 1\\ spark/')
    dst_uri = join(ggr_path, '233', '233B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, 'Mint\\ Media\\ Footage', 'Mpala\\ day2\\ /')
    dst_uri = join(ggr_path, '233', '233B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, 'Mint\\ Media\\ Footage', 'Mpala\\ day\\ 2\\ spark/')
    dst_uri = join(ggr_path, '233', '233B/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '103\\ \\(1\\)/')
    dst_uri = join(ggr_path, '103/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '103\\ \\(ccef473b\\)/')
    dst_uri = join(ggr_path, '103/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '108\\ \\(1\\)/')
    dst_uri = join(ggr_path, '108/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '226A\\ \\(Shaba\\ Funan\\ Camp\\)/')
    dst_uri = join(ggr_path, '226/')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '226A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '121/*.*')
    dst_uri = join(ggr_path, '121', '121A/')
    ut.rsync(src_uri, dst_uri)
    for src_filepath in ut.glob(src_uri.replace('\\', '')):
        ut.delete(src_filepath)

    src_uri = join(ggr_path, '54', '54A\\(16\\)/')
    dst_uri = join(ggr_path, '54', '54A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '54', '54B\\(16\\)/')
    dst_uri = join(ggr_path, '54', '54B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '87', '87/')
    dst_uri = join(ggr_path, '87', '87A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '223', 'A/')
    dst_uri = join(ggr_path, '223', '223A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '223', 'B/')
    dst_uri = join(ggr_path, '223', '223B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '14', '15A/')
    dst_uri = join(ggr_path, '14', '14A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '73/')
    dst_uri = join(ggr_path, '85/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '117', '115A/')
    dst_uri = join(ggr_path, '117', '117A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '200\\ A/')
    dst_uri = join(ggr_path, '200', '200A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '200\\ B/')
    dst_uri = join(ggr_path, '200', '200B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '200\\ F/')
    dst_uri = join(ggr_path, '200', '200F/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '200A/')
    dst_uri = join(ggr_path, '201/')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '201A/')
    ut.rsync(src_uri, dst_uri)
    # ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '201\\ E/')
    dst_uri = join(ggr_path, '201', '201E/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '201\\ F/')
    dst_uri = join(ggr_path, '201', '201F/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '200A/')
    dst_uri = join(ggr_path, '202/')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '202A/')
    ut.rsync(src_uri, dst_uri)
    # ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '202\\ B/')
    dst_uri = join(ggr_path, '202', '202B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '200', '202\\ F/')
    dst_uri = join(ggr_path, '202', '202F/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '230', '230A', 'El\\ Karama/*.*')
    dst_uri = join(ggr_path, '230', '230A/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-1])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '136', '136B', '136B\\ Grevys\\ Rally/*.*')
    dst_uri = join(ggr_path, '136', '136B/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-1])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '160', '160E', '104DUSIT')
    if exists(src_uri):
        direct = Directory(src_uri, recursive=False)
        filename_list = direct.files()
        for filename in sorted(filename_list):
            dst_uri = filename.replace('104DUSIT/', '').replace('.JPG', '_.JPG')
            assert not exists(dst_uri)
            ut.rsync(filename, dst_uri)
        ut.delete(src_uri)

    src_uri = join(ggr_path, '222', '222B', '102DUSIT')
    if exists(src_uri):
        direct = Directory(src_uri, recursive=False)
        filename_list = direct.files()
        for filename in sorted(filename_list):
            dst_uri = filename.replace('102DUSIT/', '').replace('.JPG', '_.JPG')
            assert not exists(dst_uri)
            ut.rsync(filename, dst_uri)
        ut.delete(src_uri)

    # Errors found by QR codes

    # No conflicts
    src_uri = join(ggr_path, '5', '5A/')
    dst_uri = join(ggr_path, '5', '5B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '14', '14A/')
    dst_uri = join(ggr_path, '14', '14B/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '118', '118A/')
    dst_uri = join(ggr_path, '192')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '192A/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '119', '119A/')
    dst_uri = join(ggr_path, '189')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '189A/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '120', '120A/')
    dst_uri = join(ggr_path, '190')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '190A/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '138', '138C/')
    dst_uri = join(ggr_path, '169', '169C/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri)

    # Conflicts - Move first

    src_uri = join(ggr_path, '115', '115A/')
    dst_uri = join(ggr_path, '191')
    ut.ensuredir(dst_uri)
    dst_uri = join(dst_uri, '191A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    src_uri = join(ggr_path, '148', '148A/')
    dst_uri = join(ggr_path, '149', '149A-temp/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    # Conflicts - Move second

    src_uri = join(ggr_path, '117', '117A/')
    dst_uri = join(ggr_path, '115', '115A/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)

    src_uri = join(ggr_path, '149', '149A/')
    dst_uri = join(ggr_path, '148', '148A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    # Conflicts - Move third

    src_uri = join(ggr_path, '149', '149A-temp/')
    dst_uri = join(ggr_path, '149', '149A/')
    ut.rsync(src_uri, dst_uri)
    ut.delete(src_uri.replace('\\', ''))

    # Conflicts - Merge third

    src_uri = join(ggr_path, '57', '57A/')
    dst_uri = join(ggr_path, '25', '25A/')
    ut.rsync(src_uri, dst_uri)
    src_uri = src_uri.replace('\\', '')
    src_uri = '/'.join(src_uri.split('/')[:-2])
    ut.delete(src_uri)


def convert_ggr2018_to_wbia(
    ggr_path, dbdir=None, purge=True, dry_run=False, apply_updates=True, **kwargs
):
    r"""Convert the raw GGR2 (2018) data to an wbia database.

    Args
        ggr_path (str): Directory to folder *containing* raw GGR 2018 data
        dbdir (str): Output directory

    CommandLine:
        python -m wbia convert_ggr2018_to_wbia

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_ggr import *  # NOQA
        >>> default_ggr_path = join('/', 'data', 'wbia', 'GGR2', 'GGR2018data')
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ggr_path = ut.get_argval('--ggr', type_=str, default=default_ggr_path)
        >>> result = convert_ggr2018_to_wbia(ggr_path, dbdir=dbdir, purge=False, dry_run=True, apply_updates=False)
        >>> print(result)
    """
    ALLOWED_NUMBERS = list(range(1, 250))
    ALLOWED_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']

    ################################################################################

    if apply_updates:
        _fix_ggr2018_directory_structure(ggr_path)

    ################################################################################

    blacklist_filepath_set = set(
        [
            join(ggr_path, 'Cameras info.numbers'),
            join(ggr_path, 'Cameras info.xlsx'),
            join(ggr_path, 'GGR_photos_MRC_29.1.18.ods'),
            join(ggr_path, 'Cameras info-2.numbers'),
        ]
    )

    # Check root files
    direct = Directory(ggr_path)
    for filepath in direct.files(recursive=False):
        try:
            assert filepath in blacklist_filepath_set
            ut.delete(filepath)
        except AssertionError:
            print('Unresolved root file found in %r' % (filepath,))
            continue

    ################################################################################

    if purge:
        ut.delete(dbdir)
    ibs = wbia.opendb(dbdir=dbdir)

    ################################################################################

    # Check folder structure
    assert exists(ggr_path)
    direct = Directory(ggr_path, recursive=0)
    direct1_list = direct.directories()
    direct1_list.sort(key=lambda x: int(x.base()), reverse=False)
    for direct1 in direct1_list:
        if not dry_run:
            print('Processing directory: %r' % (direct1,))
        base1 = direct1.base()

        try:
            int(base1)
        except ValueError:
            print('Error found in %r' % (direct1,))
            continue

        try:
            assert len(direct1.files(recursive=False)) == 0
        except AssertionError:
            print('Files found in %r' % (direct1,))
            continue

        seen_letter_list = []
        direct1_ = Directory(direct1.absolute_directory_path, recursive=0)
        direct2_list = direct1_.directories()
        direct2_list.sort(key=lambda x: x.base(), reverse=False)
        for direct2 in direct2_list:
            base2 = direct2.base()

            try:
                assert base2.startswith(base1)
            except AssertionError:
                print('Folder name heredity conflict %r with %r' % (direct2, direct1,))
                continue

            try:
                assert len(base2) >= 2
                assert ' ' not in base2
                number = base2[:-1]
                letter = base2[-1]
                number = int(number)
                letter = letter.upper()
                assert number in ALLOWED_NUMBERS
                assert letter in ALLOWED_LETTERS
                seen_letter_list.append(letter)
            except ValueError:
                print('Error found in %r' % (direct2,))
                continue
            except AssertionError:
                print('Folder name format error found in %r' % (direct2,))
                continue

            direct2_ = Directory(
                direct2.absolute_directory_path, recursive=True, images=True
            )
            try:
                assert len(direct2_.directories()) == 0
            except AssertionError:
                print('Folders exist in file only level %r' % (direct2,))
                continue

            filepath_list = sorted(direct2_.files())

            if not dry_run:
                try:
                    gid_list = ibs.add_images(filepath_list)
                    gid_list = ut.filter_Nones(gid_list)
                    gid_list = sorted(list(set(gid_list)))

                    imageset_text = 'GGR2,%d,%s' % (number, letter,)
                    note_list = [
                        '%s,%05d' % (imageset_text, index + 1)
                        for index, gid in enumerate(gid_list)
                    ]
                    ibs.set_image_notes(gid_list, note_list)
                    ibs.set_image_imagesettext(gid_list, [imageset_text] * len(gid_list))
                except Exception as ex:  # NOQA
                    ut.embed()

        seen_letter_set = set(seen_letter_list)
        try:
            assert len(seen_letter_set) == len(seen_letter_list)
        except AssertionError:
            print(
                'Duplicate letters in %r with letters %r' % (direct1, seen_letter_list,)
            )
            continue

        try:
            assert 'A' in seen_letter_set
        except AssertionError:
            print('WARNING: A camera not found in %r' % (direct1,))
            continue

    return ibs
