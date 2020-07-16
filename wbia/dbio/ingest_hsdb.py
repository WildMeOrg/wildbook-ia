# -*- coding: utf-8 -*-
#!/usr/bin/env python  # NOQA
"""
Converts a hotspostter database to IBEIS
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join, exists
from wbia import constants as const
from wbia.init import sysres
from six.moves import zip, map
import utool as ut
import re
import csv

print, rrr, profile = ut.inject2(__name__)


SUCCESS_FLAG_FNAME = '_hsdb_to_ibeis_convert_success'


def is_hsdb(dbdir):
    return is_hsdbv4(dbdir) or is_hsdbv3(dbdir)


def is_hsdbv4(dbdir):
    has4 = (
        exists(join(dbdir, '_hsdb'))
        and exists(join(dbdir, '_hsdb', 'name_table.csv'))
        and exists(join(dbdir, '_hsdb', 'image_table.csv'))
        and exists(join(dbdir, '_hsdb', 'chip_table.csv'))
    )
    return has4


def is_hsdbv3(dbdir):
    has3 = (
        exists(join(dbdir, '.hs_internals'))
        and exists(join(dbdir, '.hs_internals', 'name_table.csv'))
        and exists(join(dbdir, '.hs_internals', 'image_table.csv'))
        and exists(join(dbdir, '.hs_internals', 'chip_table.csv'))
    )
    return has3


def get_hsinternal(hsdb_dir):
    internal_dir = join(hsdb_dir, '_hsdb')
    if not is_hsdbv4(hsdb_dir):
        internal_dir = join(hsdb_dir, '.hs_internals')
    return internal_dir


def is_hsinternal(dbdir):
    return exists(join(dbdir, '.hs_internals'))


def is_succesful_convert(dbdir):
    """ the sucess flag is only written if the _ibsdb was properly generated """
    return exists(join(dbdir, const.PATH_NAMES._ibsdb, SUCCESS_FLAG_FNAME))


def get_unconverted_hsdbs(workdir=None):
    r"""
    Args:
        workdir (None): (default = None)

    CommandLine:
        python -m wbia.dbio.ingest_hsdb --test-get_unconverted_hsdbs

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_hsdb import *  # NOQA
        >>> workdir = None
        >>> result = get_unconverted_hsdbs(workdir)
        >>> print(result)
    """
    import os
    import numpy as np

    if workdir is None:
        workdir = sysres.get_workdir()
    dbname_list = os.listdir(workdir)
    dbpath_list = np.array([join(workdir, name) for name in dbname_list])
    needs_convert = list(map(check_unconverted_hsdb, dbpath_list))
    needs_convert_hsdbs = ut.compress(dbpath_list, needs_convert)
    return needs_convert_hsdbs


def ingest_unconverted_hsdbs_in_workdir():
    workdir = sysres.get_workdir()
    needs_convert_hsdbs = get_unconverted_hsdbs(workdir)
    for hsdb in needs_convert_hsdbs:
        try:
            convert_hsdb_to_wbia(hsdb)
        except Exception as ex:
            ut.printex(ex)
            raise


def check_unconverted_hsdb(dbdir):
    """
    Returns if a directory is an unconverted hotspotter database
    """
    return is_hsdb(dbdir) and not is_succesful_convert(dbdir)


def testdata_ensure_unconverted_hsdb():
    r"""
    Makes an unconverted test datapath

    CommandLine:
        python -m wbia.dbio.ingest_hsdb --test-testdata_ensure_unconverted_hsdb

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_hsdb import *  # NOQA
        >>> result = testdata_ensure_unconverted_hsdb()
        >>> print(result)
    """
    import utool as ut

    assert ut.is_developer(), 'dev function only'
    # Make an unconverted test database
    ut.ensurepath('/raid/tests/tmp')
    ut.delete('/raid/tests/tmp/Frogs')
    ut.copy('/raid/tests/Frogs', '/raid/tests/tmp/Frogs')
    hsdb_dir = '/raid/tests/tmp/Frogs'
    return hsdb_dir


def convert_hsdb_to_wbia(hsdir, dbdir=None, **kwargs):
    r"""
    Args
        hsdir (str): Directory to folder *containing* _hsdb
        dbdir (str): Output directory (defaults to same as  hsdb)

    CommandLine:
        python -m wbia convert_hsdb_to_wbia --dbdir ~/work/Frogs
        python -m wbia convert_hsdb_to_wbia --hsdir "/raid/raw/RotanTurtles/Roatan HotSpotter Nov_21_2016"

    Ignore:
        from wbia.dbio.ingest_hsdb import *  # NOQA
        hsdir = "/raid/raw/RotanTurtles/Roatan HotSpotter Nov_21_2016"
        dbdir = "~/work/RotanTurtles"

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_hsdb import *  # NOQA
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=None)
        >>> hsdir = ut.get_argval('--hsdir', type_=str, default=dbdir)
        >>> result = convert_hsdb_to_wbia(hsdir)
        >>> print(result)
    """
    from wbia.control import IBEISControl
    import utool as ut

    if dbdir is None:
        dbdir = hsdir
    print('[ingest] Ingesting hsdb: %r -> %r' % (hsdir, dbdir))

    assert is_hsdb(
        hsdir
    ), 'not a hotspotter database. cannot even force convert: hsdir=%r' % (hsdir,)
    assert not is_succesful_convert(dbdir), 'hsdir=%r is already converted' % (hsdir,)
    # print('FORCE DELETE: %r' % (hsdir,))
    # ibsfuncs.delete_wbia_database(hsdir)
    imgdir = join(hsdir, 'images')

    internal_dir = get_hsinternal(hsdir)
    nametbl_fpath = join(internal_dir, 'name_table.csv')
    imgtbl_fpath = join(internal_dir, 'image_table.csv')
    chiptbl_fpath = join(internal_dir, 'chip_table.csv')

    # READ NAME TABLE
    name_text_list = ['____']
    name_hs_nid_list = [0]
    with open(nametbl_fpath, 'r') as nametbl_file:
        name_reader = csv.reader(nametbl_file)
        for ix, row in enumerate(name_reader):
            # if ix >= 3:
            if len(row) == 0 or row[0].strip().startswith('#'):
                continue
            else:
                hs_nid = int(row[0])
                name = row[1].strip()
                name_text_list.append(name)
                name_hs_nid_list.append(hs_nid)

    # READ IMAGE TABLE
    iamge_hs_gid_list = []
    image_gname_list = []
    image_reviewed_list = []
    with open(imgtbl_fpath, 'r') as imgtb_file:
        image_reader = csv.reader(imgtb_file)
        for ix, row in enumerate(image_reader):
            if len(row) == 0 or row[0].strip().startswith('#'):
                continue
            else:
                hs_gid = int(row[0])
                gname_ = row[1].strip()
                # aif in hotspotter is equivilant to reviewed in IBEIS
                reviewed = bool(row[2])
                iamge_hs_gid_list.append(hs_gid)
                image_gname_list.append(gname_)
                image_reviewed_list.append(reviewed)

    image_gpath_list = [join(imgdir, gname) for gname in image_gname_list]

    ut.debug_duplicate_items(image_gpath_list)
    # print(image_gpath_list)
    image_exist_flags = list(map(exists, image_gpath_list))
    missing_images = []
    for image_gpath, flag in zip(image_gpath_list, image_exist_flags):
        if not flag:
            missing_images.append(image_gpath)
            print('Image does not exist: %s' % image_gpath)

    if not all(image_exist_flags):
        print(
            'Only %d / %d image exist' % (sum(image_exist_flags), len(image_exist_flags))
        )

    SEARCH_FOR_IMAGES = False
    if SEARCH_FOR_IMAGES:
        # Hack to try and find the missing images
        from os.path import basename

        subfiles = ut.glob(hsdir, '*', recursive=True, fullpath=True, with_files=True)
        basename_to_existing = ut.group_items(subfiles, ut.lmap(basename, subfiles))

        can_copy_list = []
        for gpath in missing_images:
            gname = basename(gpath)
            if gname not in basename_to_existing:
                print('gname = %r' % (gname,))
                pass
            else:
                existing = basename_to_existing[gname]
                can_choose = True
                if len(existing) > 1:
                    if not ut.allsame(ut.lmap(ut.get_file_uuid, existing)):
                        can_choose = False
                if can_choose:
                    found = existing[0]
                    can_copy_list.append((found, gpath))
                else:
                    print(existing)

        src, dst = ut.listT(can_copy_list)
        ut.copy_list(src, dst)

    # READ CHIP TABLE
    chip_bbox_list = []
    chip_theta_list = []
    chip_hs_nid_list = []
    chip_hs_gid_list = []
    chip_note_list = []
    with open(chiptbl_fpath, 'r') as chiptbl_file:
        chip_reader = csv.reader(chiptbl_file)
        for ix, row in enumerate(chip_reader):
            if len(row) == 0 or row[0].strip().startswith('#'):
                continue
            else:
                hs_gid = int(row[1])
                hs_nid = int(row[2])
                bbox_text = row[3]
                theta = float(row[4])
                notes = '<COMMA>'.join([item.strip() for item in row[5:]])

                bbox_text = bbox_text.replace('[', '').replace(']', '').strip()
                bbox_text = re.sub('  *', ' ', bbox_text)
                bbox_strlist = bbox_text.split(' ')
                bbox = tuple(map(int, bbox_strlist))
                # bbox = [int(item) for item in bbox_strlist]
                chip_hs_nid_list.append(hs_nid)
                chip_hs_gid_list.append(hs_gid)
                chip_bbox_list.append(bbox)
                chip_theta_list.append(theta)
                chip_note_list.append(notes)

    names = ut.ColumnLists({'hs_nid': name_hs_nid_list, 'text': name_text_list})

    images = ut.ColumnLists(
        {
            'hs_gid': iamge_hs_gid_list,
            'gpath': image_gpath_list,
            'reviewed': image_reviewed_list,
            'exists': image_exist_flags,
        }
    )

    chips = ut.ColumnLists(
        {
            'hs_gid': chip_hs_gid_list,
            'hs_nid': chip_hs_nid_list,
            'bbox': chip_bbox_list,
            'theta': chip_theta_list,
            'note': chip_note_list,
        }
    )

    IGNORE_MISSING_IMAGES = True
    if IGNORE_MISSING_IMAGES:
        # Ignore missing information
        print('pre')
        print('chips = %r' % (chips,))
        print('images = %r' % (images,))
        print('names = %r' % (names,))
        missing_gxs = ut.where(ut.not_list(images['exists']))
        missing_gids = ut.take(images['hs_gid'], missing_gxs)
        gid_to_cxs = ut.dzip(*chips.group_indicies('hs_gid'))
        missing_cxs = ut.flatten(ut.take(gid_to_cxs, missing_gids))
        # Remove missing images and dependant chips
        images = images.remove(missing_gxs)
        chips = chips.remove(missing_cxs)
        valid_nids = set(chips['hs_nid'] + [0])
        isvalid = [nid in valid_nids for nid in names['hs_nid']]
        names = names.compress(isvalid)
        print('post')
        print('chips = %r' % (chips,))
        print('images = %r' % (images,))
        print('names = %r' % (names,))

    assert all(images['exists']), 'some images dont exist'

    # if gid is None:
    #     print('Not adding the ix=%r-th Chip. Its image is corrupted image.' % (ix,))
    #     # continue
    # # Build mappings to new indexes
    # names_nid_to_nid  = {names_nid: nid for (names_nid, nid) in zip(hs_nid_list, nid_list)}
    # names_nid_to_nid[1] = names_nid_to_nid[0]  # hsdb unknknown is 0 or 1
    # images_gid_to_gid = {images_gid: gid for (images_gid, gid) in zip(hs_gid_list, gid_list)}

    ibs = IBEISControl.request_IBEISController(dbdir=dbdir, check_hsdb=False, **kwargs)
    assert len(ibs.get_valid_gids()) == 0, 'target database is not empty'

    # Add names, images, and annotations
    names['ibs_nid'] = ibs.add_names(names['text'])
    images['ibs_gid'] = ibs.add_images(images['gpath'])  # any failed gids will be None

    if True:
        # Remove corrupted images
        print('pre')
        print('chips = %r' % (chips,))
        print('images = %r' % (images,))
        print('names = %r' % (names,))
        missing_gxs = ut.where(ut.flag_None_items(images['ibs_gid']))
        missing_gids = ut.take(images['hs_gid'], missing_gxs)
        gid_to_cxs = ut.dzip(*chips.group_indicies('hs_gid'))
        missing_cxs = ut.flatten(ut.take(gid_to_cxs, missing_gids))
        # Remove missing images and dependant chips
        chips = chips.remove(missing_cxs)
        images = images.remove(missing_gxs)
        print('post')
        print('chips = %r' % (chips,))
        print('images = %r' % (images,))
        print('names = %r' % (names,))

    # Index chips using new ibs rowids
    ibs_gid_lookup = ut.dzip(images['hs_gid'], images['ibs_gid'])
    ibs_nid_lookup = ut.dzip(names['hs_nid'], names['ibs_nid'])
    try:
        chips['ibs_gid'] = ut.take(ibs_gid_lookup, chips['hs_gid'])
    except KeyError:
        chips['ibs_gid'] = [ibs_gid_lookup.get(index, None) for index in chips['hs_gid']]
    try:
        chips['ibs_nid'] = ut.take(ibs_nid_lookup, chips['hs_nid'])
    except KeyError:
        chips['ibs_nid'] = [ibs_nid_lookup.get(index, None) for index in chips['hs_nid']]

    ibs.add_annots(
        chips['ibs_gid'],
        bbox_list=chips['bbox'],
        theta_list=chips['theta'],
        nid_list=chips['ibs_nid'],
        notes_list=chips['note'],
    )

    # aid_list = ibs.get_valid_aids()
    # flag_list = [True] * len(aid_list)
    # ibs.set_annot_exemplar_flags(aid_list, flag_list)
    # assert(all(ibs.get_annot_exemplar_flags(aid_list))), 'exemplars not set correctly'

    # Write file flagging successful conversion
    with open(join(ibs.get_ibsdir(), SUCCESS_FLAG_FNAME), 'w') as file_:
        file_.write('Successfully converted hsdir=%r' % (hsdir,))
    print('finished ingest')
    return ibs


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.dbio.ingest_hsdb
        python -m wbia.dbio.ingest_hsdb --allexamples
        python -m wbia.dbio.ingest_hsdb --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
