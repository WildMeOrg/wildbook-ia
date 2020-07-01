#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import range, input
from os.path import join, expanduser, exists, basename  # NOQA
from wbia.control import IBEISControl
from wbia.other import ibsfuncs
from wbia.detecttools.directory import Directory
import utool

(print, rrr, profile) = utool.inject2(__name__)


if __name__ == '__main__':
    # FIX THIS TO POINT TO THE CORRECT DIRECTORY
    # prefix = expanduser(join('~', 'Desktop'))
    prefix = '/Volumes/EXTERNAL/BACKUPS/Dan_2014-03-26_Ol_Pejeta__100GB/Ol_pejeta_zebra_stuff__2GB/'

    print(
        """
          =====================
          PROCESSING ACTIVITIES
          =====================
          """
    )
    activities = {}
    columns = [3, 9, 10, 11, 12, 13, 14, 15]
    csv_fpath = join(prefix, 'OPC Zebra database all [exported]', 'csv')
    activity_csv_fpath = join(csv_fpath, 'Group-Habitat-Activity table.csv')
    exportedmdb_fpath = join(csv_fpath, 'Individual sightings.csv')

    utool.checkpath(activity_csv_fpath, verbose=True)
    utool.checkpath(exportedmdb_fpath, verbose=True)

    with open(join(activity_csv_fpath), 'r') as file_:
        lines = file_.read()
        for line in lines.splitlines()[1:]:
            line = [item.strip() for item in line.strip().split(',')]
            _id = line[2]
            if _id not in activities:
                activities[_id] = [line[col] for col in columns]

    originals = join(prefix, 'Ol_pejeta_zebra_photos2__1GB')
    images = Directory(originals)
    image_set = set(images.files())
    print(images)
    exts = []
    for image in images.files():
        exts.append(image.split('.')[-1])
    exts = list(set(exts))
    print('EXTENSIONS: %r ' % (exts,))

    print(
        """
          =====================
          PROCESSING IMAGESETS
          =====================
          """
    )
    used = []
    # e ncounters = open(join(prefix, 'e ncounters.csv'),'w')
    # animals = open(join(prefix, 'animals.csv'),'w')
    linenum = 0
    processed = []
    with open(join(exportedmdb_fpath), 'r') as file_:
        lines = file_.read()
        for line in lines.splitlines()[1:]:
            linenum += 1
            line = [item.strip() for item in line.strip().split(',')]
            if len(line) == 1:
                print('WARNING: INVALID DATA ON LINE', linenum, '[FIX TO CONTINUE]')
                input()
                continue
            filename = line[2].strip('"\'')
            sighting = line[1]
            files = [join(originals, filename + '.' + ext) in image_set for ext in exts]

            if sighting in activities and True in files:
                for i in range(len(files)):
                    if files[i]:
                        filename += '.' + exts[i]
                        break

                line = [join(originals, filename)] + activities[sighting]
                if filename not in used:
                    processed.append(line)
                    # animals.write(','.join(line) + '\n')
                    used.append(filename)
                # e ncounters.write(','.join(line) + '\n')

    print('USED:', float(len(used)) / len(images.files()))
    # print('processed: %s' % processed)

    print(
        """
          =====================
          FINISHED PROCESS
          =====================
          """
    )

    def _sloppy_data(string):
        string = string.replace('0212', '2012')
        string = string.replace('1212', '2012')
        string = string.replace('"', '')
        return string

    dbdir = join(prefix, 'converted')
    ibsfuncs.delete_wbia_database(dbdir)
    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    image_gpath_list = [item[0] for item in processed]
    notes_list = [','.join([basename(item[0])] + item[2:5]) for item in processed]
    times_list = [
        utool.exiftime_to_unixtime(_sloppy_data(item[1]), timestamp_format=2)
        for item in processed
    ]
    assert all(map(exists, image_gpath_list)), 'some images dont exist'

    gid_list = ibs.add_images(image_gpath_list)
    ibs.localize_images()
    ibs.set_image_notes(gid_list, notes_list)
    ibs.set_image_unixtime(gid_list, times_list)
    bbox_list = [(0, 0, w, h) for (w, h) in ibs.get_image_sizes(gid_list)]
    aid_list = ibs.add_annots(gid_list, bbox_list)
    name_list = [basename(image_path).split('.')[0] for image_path in image_gpath_list]
    ibs.set_annot_names(aid_list, name_list)
    ibs.db.commit()
