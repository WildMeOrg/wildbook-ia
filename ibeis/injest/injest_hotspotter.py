#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join, exists
import ibeis
from ibeis.dev import params
from itertools import izip
import utool
import re
import csv
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[injest_hotspotter]')


def injest_hsdb(hsdb_dir):
    print('Injesting: %r' % hsdb_dir)
    imgdir = join(hsdb_dir, 'images')

    main_locals = ibeis.main(dbdir=hsdb_dir, allow_newdir=True, gui=True)
    ibs = main_locals['ibs']  # IBEIS Control

    # READ NAME TABLE
    #names = open(join(workdir,'wildebeest/_hsdb/name_table.csv'),'rb')
    names_name_list = ['____']
    name_nid_list   = [0]
    with open(join(hsdb_dir, '_hsdb', 'name_table.csv'), 'rb') as nametbl_file:
        name_reader = csv.reader(nametbl_file)
        for i, row in enumerate(name_reader):
            if i >= 3:
                nid = int(row[0])
                name = row[1].strip()
                names_name_list.append(name)
                name_nid_list.append(nid)

    image_gid_list   = []
    image_gname_list = []
    image_aif_list   = []
    with open(join(hsdb_dir, '_hsdb/image_table.csv'), 'rb') as imgtb_file:
        image_reader = csv.reader(imgtb_file)
        for i, row in enumerate(image_reader):
            if i >= 3:
                gid = int(row[0])
                gname = row[1].strip()
                aif = bool(row[2])
                image_gid_list.append(gid)
                image_gname_list.append(gname)
                image_aif_list.append(aif)

    image_gpath_list = [join(imgdir, gname) for gname in image_gname_list]
    assert all(map(exists, image_gpath_list)), 'some images dont exist'

    # Add Images and Names Table
    gid_list = ibs.add_images(image_gpath_list)
    nid_list = ibs.add_names(names_name_list)
    # Build mappings to new indexes
    names_nid_to_nid  = {names_nid: nid for (names_nid, nid) in izip(name_nid_list, nid_list)}
    images_gid_to_gid = {images_gid: gid for (images_gid, gid) in izip(image_gid_list, gid_list)}

    #get rois from chip_table
    chip_bbox_list   = []
    chip_theta_list  = []
    chip_nid_list    = []
    chip_gid_list    = []
    chip_note_list   = []
    with open(join(hsdb_dir, '_hsdb/chip_table.csv'), 'rb') as chiptbl:
        chip_reader = csv.reader(chiptbl)
        for i, row in enumerate(chip_reader):
            if i >= 3:
                images_gid = int(row[1])
                names_nid = int(row[2])
                bbox_text = row[3]
                theta = float(row[4])
                notes = '<COMMA>'.join([item.strip() for item in row[5:]])
                nid = names_nid_to_nid[names_nid]
                gid = images_gid_to_gid[images_gid]
                bbox_text = bbox_text.replace('[', '').replace(']', '').strip()
                bbox_text = re.sub('  *', ' ', bbox_text)
                bbox_strlist = bbox_text.split(' ')
                bbox = map(int, bbox_strlist)
                #bbox = [int(item) for item in bbox_strlist]
                chip_nid_list.append(nid)
                chip_gid_list.append(gid)
                chip_bbox_list.append(bbox)
                chip_theta_list.append(theta)
                chip_note_list.append(notes)

    # Add Chips Table
    ibs.add_rois(chip_gid_list, chip_bbox_list, chip_theta_list, nid_list=chip_nid_list, notes_list=chip_note_list)


if __name__ == '__main__':
    import sys
    try:
        dbdir = sys.argv[1]
    except Exception:
        # TEST
        workdir = params.get_workdir()
        dbdir = join(workdir, 'PZ_MOTHERS')
    injest_hsdb(dbdir)
