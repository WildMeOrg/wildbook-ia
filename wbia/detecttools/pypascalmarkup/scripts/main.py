# -*- coding: utf-8 -*-
import os
import re
import csv
import pypascalmarkup
import numpy as np
import ExifTags
from PIL import Image
from collections import defaultdict


# maps img_id -> img_name, to look up images names from chip_table's ImgID
def load_data(image_table_filename, chip_table_filename):
    image_ids = defaultdict(list)
    with open(image_table_filename, 'rb') as image_table_file:
        image_reader = csv.reader(image_table_file, delimiter=',', quotechar=',')
        for row in image_reader:
            # ignore the header and comments
            if row[0].startswith('#'):
                continue
            else:
                img_id = row[0].strip()
                img_name = row[1].strip()
                image_ids[img_id] = img_name

    # initialise all dict elements to an empty list
    images = defaultdict(list)
    with open(chip_table_filename, 'rb') as chip_table_file:
        chip_reader = csv.reader(chip_table_file, delimiter=',', quotechar=',')
        for row in chip_reader:
            # ignore the header and comments
            if row[0].startswith('#'):
                continue
            else:
                img_id = row[1].strip()
                # remove the square brackets at the front and back of ROI
                roi = list(map(int, re.sub('[^0-9]', ' ', row[3]).split()))
                img_name = image_ids[img_id]
                # convert from HotSpotter bounding box to PASCAL-VOC: xmax, xmin, ymax, ymin
                tlx, tly, w, h = roi
                images[img_name].append(np.array([tlx + w, tlx, tly + h, tly]))

    return images


def get_all_files(dir, ext='.jpg'):
    return [
        f
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and f.lower().endswith(ext)
    ]


if __name__ == '__main__':
    classes = [
        'elephant',
        'elephant',
        'giraffe',
        'rhino',
        'wilddog',
        'zebra_grevys',
        'zebra_plains',
    ]
    directories = [
        'Elephant_1',
        'Elephant_2',
        'Giraffe',
        'Rhino',
        'Wild_Dog',
        'Zebra_Grevys',
        'Zebra_Plain',
    ]
    # classname = 'zebra_grevys'
    for classname, directory in zip(classes, directories):
        images = load_data(directory + '/image_table.csv', directory + '/chip_table.csv')
        info = {
            'database_name': 'The IBEIS Database',
            'source': 'Mpala, Ol Pejeta, Kenya',
        }
        img_dir = directory + '/images'
        output_dir = '../data'
        out_fmt = '2014_%06d'

        processed, copied, c = 0, 0, 0
        # find the file in the output dir with the largest filename, start the naming from there
        while os.path.isfile(
            os.path.join(output_dir, 'Annotations', (out_fmt % c) + '.xml')
        ):
            c += 1  # this should be the number for the newest image file

        for name in images:
            src = os.path.join(img_dir, name)

            processed += 1
            if os.path.isfile(src):
                img = Image.open(src)
                # shutil.copyfile(src, dst_img)
                # print w, h, r, (int(np.round(w / r)), int(np.round(h / r)))
                try:
                    for orientation in list(ExifTags.TAGS.keys()):
                        if ExifTags.TAGS[orientation] == 'Orientation':
                            break
                    exif = dict(list(img._getexif().items()))

                    if exif[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img = img.rotate(90, expand=True)
                except Exception as e:
                    print(repr(e))
                    print('No EXIF data found for %s' % src)

                w, h = img.size
                if w < 900 and h < 900:
                    print('%s skipped because of size' % src)
                    continue
                if max(w, h) / float(min(w, h)) > 2:
                    print('%s skipped because of ratio' % src)
                    continue
                r = max(w, h) / 900.0

                new_img_name = (out_fmt % c) + '.jpg'
                dst_img = os.path.join(output_dir, 'JPEGImages', new_img_name)
                dst_ann = os.path.join(output_dir, 'Annotations', (out_fmt % c) + '.xml')
                c += 1
                copied += 1

                img = img.resize(
                    (int(np.round(w / r)), int(np.round(h / r))), Image.ANTIALIAS
                )
                img.save(dst_img)

                annotation = pypascalmarkup.PascalVOC_Markup_Annotation(
                    dst_img, 'IBEIS', new_img_name, **info
                )
                for roi in images[name]:
                    annotation.add_object(
                        classname, tuple(np.asarray(np.round(roi / r), np.int))
                    )
                with open(dst_ann, 'w') as xml_out:
                    xml_out.write(annotation.xml())

                print(
                    '%d of %d, copied %s to %s' % (processed, len(images), src, dst_img)
                )
                print('created corresponding annotation file at %s' % dst_ann)
            else:
                print('Could not find file %s, ignoring.' % src)

        print('***********************')
        print('copied %d of %d files' % (copied, processed))
        print('***********************')
