#!/usr/bin/env python2.7
"""
Converts an IBEIS database to a hotspotter db
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join, relpath
#import ibeis
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[export_hsdb]')


def get_hsdb_image_gpaths(ibs, gid_list):
    imgdir = join(ibs.get_dbdir(), 'images')
    gpath_list_ = ibs.get_image_paths(gid_list)
    gpath_list  = [relpath(gpath, imgdir) for gpath in gpath_list_]
    return gpath_list


def export_ibeis_to_hotspotter(ibs):
    print('export to hsdb')
    ibs.inject_func(get_hsdb_image_gpaths)
    hsdb_dir = ibs.get_dbdir()
    internal_hsdb = join(hsdb_dir, '_hsdb')
    utool.ensuredir(internal_hsdb)

    # Build Image Table
    gid_list        = ibs.get_valid_gids()
    gpath_list      = ibs.get_hsdb_image_gpaths(gid_list)
    aif_list        = ibs.get_image_aifs(gid_list)
    image_table_csv = utool.make_csv_table(
        [gid_list, gpath_list, aif_list],
        ['gid', 'gname', 'aif'],
        '# image table')

    # Build Name Table
    nid_list       =  ibs.get_valid_nids()
    name_list      = ibs.get_names(nid_list)
    name_table_csv = utool.make_csv_table(
        [nid_list, name_list],
        ['nid', 'name'],
        '# name table')

    # Build Chip Table
    aid_list        = ibs.get_valid_aids()
    annotiongid_list     = ibs.get_annotion_gids(aid_list)
    annotionnid_list     = ibs.get_annotion_nids(aid_list)
    bbox_list       = map(list, ibs.get_annotion_bboxes(aid_list))
    theta_list      = ibs.get_annotion_thetas(aid_list)
    notes_list      = ibs.get_annotion_notes(aid_list)

    chip_column_list = [aid_list, annotiongid_list, annotionnid_list, bbox_list, theta_list, notes_list]
    chip_column_lbls = ['cid', 'gid', 'nid', '[tlx tly w h]', 'theta', 'notes']
    chip_column_types = [int, int, int, list, float, str]
    chip_table_csv = utool.make_csv_table(
        chip_column_list,
        chip_column_lbls,
        '# chip table', chip_column_types)

    if utool.VERBOSE:
        if len(aid_list) < 87:
            print(chip_table_csv)
        if len(nid_list) < 87:
            print(name_table_csv)
        if len(gid_list) < 87:
            print(image_table_csv)

    # Write Tables
    with open(join(internal_hsdb, 'HSDB_image_table.csv'), 'wb') as imgtbl_file:
        imgtbl_file.write(image_table_csv)
    with open(join(internal_hsdb, 'HSDB_name_table.csv'), 'wb') as nametbl_file:
        nametbl_file.write(name_table_csv)
    with open(join(internal_hsdb, 'HSDB_chip_table.csv'), 'wb') as chiptbl_file:
        chiptbl_file.write(chip_table_csv)


def dump_tables(ibs):
    """ Dumps hotspotter like tables to disk """
    ibsdir = ibs.get_ibsdir()
    gtbl_name = join(ibsdir, 'IBEIS_DUMP_images_table.csv')
    ntbl_name = join(ibsdir, 'IBEIS_DUMP_names_table.csv')
    rtbl_name = join(ibsdir, 'IBEIS_DUMP_annotations_table.csv')
    with open(gtbl_name, 'w') as file_:
        gtbl_str = ibs.db.get_table_csv('images', exclude_columns=[])
        file_.write(gtbl_str)
    with open(ntbl_name, 'w') as file_:
        ntbl_str = ibs.db.get_table_csv('names',  exclude_columns=[])
        file_.write(ntbl_str)
    with open(rtbl_name, 'w') as file_:
        rtbl_str = ibs.db.get_table_csv('annotations',   exclude_columns=[])
        file_.write(rtbl_str)


def get_flat_table(ibs):
    """ Dumps hotspotter flat tables """
    aid_list = ibs.get_valid_aids()
    column_tups = [
        (int,   'aids',   aid_list,),
        (str,   'names',  ibs.get_annotion_names(aid_list),),
        (list,  'bbox',   map(list, ibs.get_annotion_bboxes(aid_list),)),
        (float, 'theta',  ibs.get_annotion_thetas(aid_list),),
        (str,   'gpaths', ibs.get_annotion_gpaths(aid_list),),
        (str,   'notes',  ibs.get_annotion_notes(aid_list),),
        (str,   'uuids',  ibs.get_annotion_uuids(aid_list),),
    ]
    column_type   = [tup[0] for tup in column_tups]
    column_labels = [tup[1] for tup in column_tups]
    column_list   = [tup[2] for tup in column_tups]
    header = '\n'.join([
        '# Roi Flat Table',
        '# aid   - internal annotion index (not gaurenteed unique)',
        '# name  - animal identity',
        '# bbox  - bounding box [tlx tly w h] in image',
        '# theta - bounding box orientation',
        '# gpath - image filepath',
        '# notes - user defined notes',
        '# uuids - unique universal ids (gaurenteed unique)',
    ])
    flat_table_str = utool.make_csv_table(column_list, column_labels, header,
                                          column_type)
    return flat_table_str


def dump_flat_table(ibs):
    flat_table_fpath = join(ibs.dbdir, 'IBEIS_DUMP_flat_table.csv')
    flat_table_str = ibs.get_flat_table()
    print('[ibs] dumping flat table to: %r' % flat_table_fpath)
    with open(flat_table_fpath, 'w') as file_:
        file_.write(flat_table_str)


SUCCESS_FLAG_FNAME = '_hsdb_to_ibeis_convert_success'
