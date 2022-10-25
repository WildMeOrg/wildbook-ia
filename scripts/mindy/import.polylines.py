# -*- coding: utf-8 -*-
"""
Render Mindy annotation data.

pip install xmltodict
"""
from os.path import exists, join

import numpy as np
import utool as ut
import xmltodict
from PIL import ExifTags, Image

import wbia

ibs = wbia.opendb(dbdir='/data/db')


with open('annotations.xml', 'r') as xml_file:
    data = xmltodict.parse(xml_file.read())

annotations = data.get('annotations', [])
version = annotations.get('version')
meta = annotations.get('meta')
images = annotations.get('image')

print('Using annotation tool version {}'.format(version))
print('Exported on {}'.format(meta.get('dumped')))
print('Task:')
print(ut.repr3(meta.get('task')))

add_paths = []
add_orients = []
add_points = []

for image in images:
    filename = image.get('@name')
    polyline = image.get('polyline')
    points_str = polyline.get('@points')
    points_ = [point.strip().split(',') for point in points_str.strip().split(';')]
    points = [tuple(map(int, map(np.around, map(float, point)))) for point in points_]

    filepath = join('images', filename)

    assert exists(filepath)

    img = Image.open(filepath)
    exif_raw = img._getexif()
    if exif_raw is None:
        exif_raw = {}
    globals().update(locals())
    exif = {ExifTags.TAGS[k]: v for k, v in exif_raw.items() if k in ExifTags.TAGS}
    orient = exif.get('Orientation', None)
    img.close()

    add_paths.append(filepath)
    add_orients.append(orient)
    add_points.append(points)


add_gids = ibs.add_images(add_paths)
ibs.set_image_orientation(add_gids, add_orients)

add_aids = ibs.use_images_as_annotations(add_gids)
bbox_list = ibs.get_annot_bboxes(add_aids)
add_pids = ibs.add_parts(add_aids, bbox_list=bbox_list)

assert len(add_points) == len(add_pids)

contour_dict_list = []
for add_pid, add_point in zip(add_pids, add_points):
    xtl, ytl, w, h = ibs.get_part_boxes(add_pid)

    contour = {
        'contour': {
            'segment': [],
        },
    }

    for x, y in add_point:
        contour['contour']['segment'].append(
            {
                'x': x / w,
                'y': y / h,
                'r': w * 0.01,
            }
        )

    contour_dict_list.append(contour)

ibs.set_part_contour(add_pids, contour_dict_list)
ibs.precompute_web_viewpoint_thumbnails()
