#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converts an IBEIS database to a hotspotter db
"""
from __future__ import absolute_import, division, print_function
from six.moves import map
from os.path import join, relpath
import utool as ut

print, rrr, profile = ut.inject2(__name__)


def get_hsdb_image_gpaths(ibs, gid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    Returns:
        list: gpath_list

    CommandLine:
        python -m wbia.dbio.export_hsdb --test-get_hsdb_image_gpaths

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dbio.export_hsdb import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> # execute function
        >>> gpath_list = get_hsdb_image_gpaths(ibs, gid_list)
        >>> # verify results
        >>> result = ut.repr2(gpath_list, nl=1)
        >>> print(result)
        [
            '../_ibsdb/images/66ec193a-1619-b3b6-216d-1784b4833b61.jpg',
            '../_ibsdb/images/d8903434-942f-e0f5-d6c2-0dcbe3137bf7.jpg',
        ]
    """
    imgdir = join(ibs.get_dbdir(), 'images')
    gpath_list_ = ibs.get_image_paths(gid_list)
    gpath_list = [ut.ensure_unixslash(relpath(gpath, imgdir)) for gpath in gpath_list_]
    return gpath_list


def get_hots_table_strings(ibs):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.dbio.export_hsdb --test-get_hots_table_strings

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dbio.export_hsdb import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.delete_empty_nids()
        >>> # execute function
        >>> csvtup = get_hots_table_strings(ibs)
        >>> # hack so hashtag is at the end of each line
        >>> result = '\n'.join(csvtup).replace('\n', '#\n') + '#'
        >>> # verify results
        >>> print(result)
        # image table#
        # num_rows=13#
        #   gid,                                                      gname,  aif#
              1,  ../_ibsdb/images/66ec193a-1619-b3b6-216d-1784b4833b61.jpg,    0#
              2,  ../_ibsdb/images/d8903434-942f-e0f5-d6c2-0dcbe3137bf7.jpg,    0#
              3,  ../_ibsdb/images/b73b72f4-4acb-c445-e72c-05ce02719d3d.jpg,    0#
              4,  ../_ibsdb/images/0cd05978-3d83-b2ee-2ac9-798dd571c3b3.jpg,    0#
              5,  ../_ibsdb/images/0a9bc03d-a75e-8d14-0153-e2949502aba7.jpg,    0#
              6,  ../_ibsdb/images/2deeff06-5546-c752-15dc-2bd0fdb1198a.jpg,    0#
              7,  ../_ibsdb/images/a9b70278-a936-c1dd-8a3b-bc1e9a998bf0.png,    0#
              8,  ../_ibsdb/images/42fdad98-369a-2cbc-67b1-983d6d6a3a60.jpg,    0#
              9,  ../_ibsdb/images/c459d381-fd74-1d99-6215-e42e3f432ea9.jpg,    0#
             10,  ../_ibsdb/images/33fd9813-3a2b-774b-3fcc-4360d1ae151b.jpg,    0#
             11,  ../_ibsdb/images/97e8ea74-873f-2092-b372-f928a7be30fa.jpg,    0#
             12,  ../_ibsdb/images/588bc218-83a5-d400-21aa-d499832632b0.jpg,    0#
             13,  ../_ibsdb/images/163a890c-36f2-981e-3529-c552b6d668a3.jpg,    0#
        # name table#
        # num_rows=7#
        #   nid,   name#
              1,   easy#
              2,   hard#
              3,   jeff#
              4,   lena#
              5,   occl#
              6,  polar#
              7,  zebra#
        # chip table#
        # num_rows=13#
        #   cid,  gid,  nid,      [tlx tly w h],  theta,                                        notes#
              1,    1,   -1,  [0  0  1047  715],   0.00,              aid 1 and 2 are correct matches#
              2,    2,    1,  [0  0  1035  576],   0.00,                                             #
              3,    3,    1,  [0  0  1072  804],   0.00,                                             #
              4,    4,   -4,  [0  0  1072  804],   0.00,                                             #
              5,    5,    2,  [0  0  1072  804],   0.00,                                             #
              6,    6,    2,   [0  0  450  301],   0.00,                                             #
              7,    7,    3,   [0  0  400  400],   0.00,  very simple image to debug feature detector#
              8,    8,    4,   [0  0  220  220],   0.00,                          standard test image#
              9,    9,   -9,   [0  0  450  284],   0.00,              this is actually a plains zebra#
             10,   10,    5,   [0  0  450  341],   0.00,              this is actually a plains zebra#
             11,   11,  -11,   [0  0  741  734],   0.00,                                             #
             12,   12,    6,   [0  0  673  634],   0.00,                                             #
             13,   13,    7,  [0  0  1114  545],   0.00,                                             #
    """
    print('export to hsdb')
    # ibs.inject_func(get_hsdb_image_gpaths)

    # Build Image Table
    gid_list = ibs.get_valid_gids()
    gpath_list = get_hsdb_image_gpaths(ibs, gid_list)
    reviewed_list = ibs.get_image_reviewed(gid_list)
    # aif in hotspotter is equivilant to reviewed in IBEIS
    image_table_csv = ut.make_csv_table(
        [gid_list, gpath_list, reviewed_list], ['gid', 'gname', 'aif'], '# image table'
    )

    # Build Name Table
    nid_list = ibs.get_valid_nids()
    name_list = ibs.get_name_texts(nid_list)
    name_table_csv = ut.make_csv_table(
        [nid_list, name_list], ['nid', 'name'], '# name table'
    )

    # Build Chip Table
    aid_list = ibs.get_valid_aids()
    annotationgid_list = ibs.get_annot_gids(aid_list)
    annotationnid_list = ibs.get_annot_name_rowids(aid_list)
    bbox_list = list(map(list, ibs.get_annot_bboxes(aid_list)))
    theta_list = ibs.get_annot_thetas(aid_list)
    notes_list = ibs.get_annot_notes(aid_list)

    chip_column_list = [
        aid_list,
        annotationgid_list,
        annotationnid_list,
        bbox_list,
        theta_list,
        notes_list,
    ]
    chip_column_lbls = ['cid', 'gid', 'nid', '[tlx tly w h]', 'theta', 'notes']
    chip_column_types = [int, int, int, list, float, str]
    chip_table_csv = ut.make_csv_table(
        chip_column_list, chip_column_lbls, '# chip table', chip_column_types
    )

    if ut.VERBOSE:
        if len(aid_list) < 87:
            print(chip_table_csv)
        if len(nid_list) < 87:
            print(name_table_csv)
        if len(gid_list) < 87:
            print(image_table_csv)
    return image_table_csv, name_table_csv, chip_table_csv


def export_wbia_to_hotspotter(ibs):
    # Dumps the files
    hsdb_dir = ibs.get_dbdir()
    internal_hsdb = join(hsdb_dir, '_hsdb')
    ut.ensuredir(internal_hsdb)
    image_table_csv, name_table_csv, chip_table_csv = get_hots_table_strings(ibs)
    # Write Tables
    with open(join(internal_hsdb, 'HSDB_image_table.csv'), 'wb') as imgtbl_file:
        imgtbl_file.write(image_table_csv)
    with open(join(internal_hsdb, 'HSDB_name_table.csv'), 'wb') as nametbl_file:
        nametbl_file.write(name_table_csv)
    with open(join(internal_hsdb, 'HSDB_chip_table.csv'), 'wb') as chiptbl_file:
        chiptbl_file.write(chip_table_csv)


def dump_hots_tables(ibs):
    """ Dumps hotspotter like tables to disk """
    ibsdir = ibs.get_ibsdir()
    gtbl_name = join(ibsdir, 'IBEIS_DUMP_images_table.csv')
    ntbl_name = join(ibsdir, 'IBEIS_DUMP_names_table.csv')
    rtbl_name = join(ibsdir, 'IBEIS_DUMP_annotations_table.csv')
    with open(gtbl_name, 'w') as file_:
        gtbl_str = ibs.db.get_table_csv('images', exclude_columns=[])
        file_.write(gtbl_str)
    with open(ntbl_name, 'w') as file_:
        ntbl_str = ibs.db.get_table_csv('names', exclude_columns=[])
        file_.write(ntbl_str)
    with open(rtbl_name, 'w') as file_:
        rtbl_str = ibs.db.get_table_csv('annotations', exclude_columns=[])
        file_.write(rtbl_str)


def get_hots_flat_table(ibs):
    """ Dumps hotspotter flat tables

    Args:
        ibs (IBEISController):  wbia controller object

    Returns:
        str: flat_table_str

    CommandLine:
        python -m wbia.dbio.export_hsdb --exec-get_hots_flat_table

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dbio.export_hsdb import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> flat_table_str = get_hots_flat_table(ibs)
        >>> result = ('flat_table_str = %s' % (str(flat_table_str),))
        >>> print(result)
    """
    aid_list = ibs.get_valid_aids()
    column_tups = [
        (int, 'aids', aid_list,),
        (str, 'names', ibs.get_annot_names(aid_list),),
        (list, 'bbox', list(map(list, ibs.get_annot_bboxes(aid_list),))),
        (float, 'theta', ibs.get_annot_thetas(aid_list),),
        (str, 'gpaths', ibs.get_annot_image_paths(aid_list),),
        (str, 'notes', ibs.get_annot_notes(aid_list),),
        (str, 'uuids', ibs.get_annot_uuids(aid_list),),
    ]
    column_type = [tup[0] for tup in column_tups]
    column_lbls = [tup[1] for tup in column_tups]
    column_list = [tup[2] for tup in column_tups]
    header = '\n'.join(
        [
            '# Roi Flat Table',
            '# aid   - internal annotation index (not gaurenteed unique)',
            '# name  - animal identity',
            '# bbox  - bounding box [tlx tly w h] in image',
            '# theta - bounding box orientation',
            '# gpath - image filepath',
            '# notes - user defined notes',
            '# uuids - unique universal ids (gaurenteed unique)',
        ]
    )
    flat_table_str = ut.make_csv_table(column_list, column_lbls, header, column_type)
    return flat_table_str


def dump_hots_flat_table(ibs):
    flat_table_fpath = join(ibs.dbdir, 'IBEIS_DUMP_flat_table.csv')
    flat_table_str = ibs.get_flat_table()
    print('[ibs] dumping flat table to: %r' % flat_table_fpath)
    with open(flat_table_fpath, 'w') as file_:
        file_.write(flat_table_str)


SUCCESS_FLAG_FNAME = '_hsdb_to_ibeis_convert_success'


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.dbio.export_hsdb
        python -m wbia.dbio.export_hsdb --allexamples
        python -m wbia.dbio.export_hsdb --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
