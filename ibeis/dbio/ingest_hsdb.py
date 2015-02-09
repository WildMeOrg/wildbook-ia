#!/usr/bin/env python2.7
"""
Converts a hotspostter database to IBEIS
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join, exists
#import ibeis
from ibeis import constants
from ibeis import ibsfuncs
from ibeis.dev import sysres
from six.moves import zip, map
import utool
import re
import csv
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[ingest_hsbd]')


SUCCESS_FLAG_FNAME = '_hsdb_to_ibeis_convert_success'

FORCE_DELETE = utool.get_argflag('--force-delete')


def is_succesful_convert(dbdir):
    return exists(join(dbdir, constants.PATH_NAMES._ibsdb, SUCCESS_FLAG_FNAME))


def get_unconverted_hsdbs(workdir=None):
    import os
    import numpy as np
    import vtool as vt
    if workdir is None:
        workdir = sysres.get_workdir()
    dbname_list = os.listdir(workdir)
    dbpath_list = np.array([join(workdir, name) for name in dbname_list])
    is_hsdb_list    = np.array(list(map(sysres.is_hsdb, dbpath_list)))
    is_ibs_cvt_list = np.array(list(map(is_succesful_convert, dbpath_list)))
    if FORCE_DELETE:
        needs_convert = is_hsdb_list
    else:
        needs_convert =  vt.and_lists(is_hsdb_list, True - is_ibs_cvt_list)
    needs_convert_hsdbs  = dbpath_list[needs_convert].tolist()
    return needs_convert_hsdbs


def ingest_unconverted_hsdbs_in_workdir():
    workdir = sysres.get_workdir()
    needs_convert_hsdbs = get_unconverted_hsdbs(workdir)
    for hsdb in needs_convert_hsdbs:
        try:
            convert_hsdb_to_ibeis(hsdb, force_delete=FORCE_DELETE)
        except Exception as ex:
            utool.printex(ex)
            raise


@utool.indent_func
def convert_hsdb_to_ibeis(hsdb_dir, force_delete=False):
    from ibeis.control import IBEISControl
    import utool as ut
    assert(sysres.is_hsdb(hsdb_dir)), 'not a hotspotter database. cannot even force convert: hsdb_dir=%r' % (hsdb_dir,)
    if force_delete:
        print('FORCE DELETE: %r' % (hsdb_dir,))
        ibsfuncs.delete_ibeis_database(hsdb_dir)
    print('[ingest] Ingesting hsdb: %r' % hsdb_dir)
    imgdir = join(hsdb_dir, 'images')

    ibs = IBEISControl.IBEISController(dbdir=hsdb_dir)

    # READ NAME TABLE
    names_name_list = ['____']
    name_nid_list   = [0]

    internal_dir = sysres.get_hsinternal(hsdb_dir)

    with open(join(internal_dir, 'name_table.csv'), 'rb') as nametbl_file:
        name_reader = csv.reader(nametbl_file)
        for ix, row in enumerate(name_reader):
            #if ix >= 3:
            if len(row) == 0 or row[0].strip().startswith('#'):
                continue
            else:
                nid = int(row[0])
                name = row[1].strip()
                names_name_list.append(name)
                name_nid_list.append(nid)

    # ADD NAMES TABLE
    nid_list = ibs.add_names(names_name_list)
    #print(names_name_list)
    #print(nid_list)

    assert len(nid_list) == len(names_name_list), 'bad name adder'

    image_gid_list   = []
    image_gname_list = []
    image_reviewed_list   = []
    with open(join(internal_dir, 'image_table.csv'), 'rb') as imgtb_file:
        image_reader = csv.reader(imgtb_file)
        for ix, row in enumerate(image_reader):
            #if ix >= 3:
            if len(row) == 0 or row[0].strip().startswith('#'):
                continue
            else:
                gid = int(row[0])
                gname_ = row[1].strip()
                reviewed = bool(row[2])  # aif in hotspotter is equivilant to reviewed in IBEIS
                image_gid_list.append(gid)
                image_gname_list.append(gname_)
                image_reviewed_list.append(reviewed)

    image_gpath_list = [join(imgdir, gname) for gname in image_gname_list]

    ut.debug_duplicate_items(image_gpath_list)
    #print(image_gpath_list)
    flags = list(map(exists, image_gpath_list))
    for image_gpath, flag in zip(image_gpath_list, flags):
        if not flag:
            print('Image does not exist: %s' % image_gpath)

    assert all(flags), 'some images dont exist'

    # Add Images Table
    gid_list = ibs.add_images(image_gpath_list)  # any failed gids will be None
    assert len(gid_list) == len(image_gpath_list), 'bad image adder'
    # Build mappings to new indexes
    names_nid_to_nid  = {names_nid: nid for (names_nid, nid) in zip(name_nid_list, nid_list)}
    names_nid_to_nid[1] = names_nid_to_nid[0]  # hsdb unknknown is 0 or 1
    images_gid_to_gid = {images_gid: gid for (images_gid, gid) in zip(image_gid_list, gid_list)}

    #get annotations from chip_table
    chip_bbox_list   = []
    chip_theta_list  = []
    chip_nid_list    = []
    chip_gid_list    = []
    chip_note_list   = []
    with open(join(internal_dir, 'chip_table.csv'), 'rb') as chiptbl_file:
        chip_reader = csv.reader(chiptbl_file)
        for ix, row in enumerate(chip_reader):
            if len(row) == 0 or row[0].strip().startswith('#'):
                continue
            else:
                images_gid = int(row[1])
                names_nid = int(row[2])
                bbox_text = row[3]
                theta = float(row[4])
                notes = '<COMMA>'.join([item.strip() for item in row[5:]])
                try:
                    nid = names_nid_to_nid[names_nid]
                except KeyError:
                    print(names_nid_to_nid)
                    print('Error no names_nid: %r' % names_nid)
                    raise
                gid = images_gid_to_gid[images_gid]
                bbox_text = bbox_text.replace('[', '').replace(']', '').strip()
                bbox_text = re.sub('  *', ' ', bbox_text)
                bbox_strlist = bbox_text.split(' ')
                bbox = tuple(map(int, bbox_strlist))
                if gid is None:
                    print('Not adding the ix=%r-th Chip. Its image is corrupted image.' % (ix,))
                    continue
                #bbox = [int(item) for item in bbox_strlist]
                chip_nid_list.append(nid)
                chip_gid_list.append(gid)
                chip_bbox_list.append(bbox)
                chip_theta_list.append(theta)
                chip_note_list.append(notes)

    # Add Chips Table
    ibs.add_annots(chip_gid_list, chip_bbox_list, chip_theta_list, nid_list=chip_nid_list, notes_list=chip_note_list)

    # Set all injested RIDS as exemplars
    aid_list = ibs.get_valid_aids()
    flag_list = [True] * len(aid_list)
    ibs.set_annot_exemplar_flags(aid_list, flag_list)
    assert(all(ibs.get_annot_exemplar_flags(aid_list))), 'exemplars not set correctly'

    # Write file flagging successful conversion
    with open(join(ibs.get_ibsdir(), SUCCESS_FLAG_FNAME), 'w') as file_:
        file_.write('Successfully converted hsdb_dir=%r' % (hsdb_dir,))
    print('finished ingest')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    db = utool.get_argval('--db', type_=str, default=None)
    dbdir = sysres.db_to_dbdir(db, allow_newdir=False, use_sync=False)
    convert_hsdb_to_ibeis(dbdir)
