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
add_bboxes = []
add_thetas = []
add_label = []
add_species = []

for image in images:
    filename = image.get('@name')
    bbox = image.get('box')
    label = bbox.get('@label')
    xtl = int(np.around(float(bbox.get('@xtl'))))
    ytl = int(np.around(float(bbox.get('@ytl'))))
    xbr = int(np.around(float(bbox.get('@xbr'))))
    ybr = int(np.around(float(bbox.get('@ybr'))))
    rotation = float(bbox.get('@rotation', 0.0))

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

    bbox = (xtl, ytl, xbr - xtl, ybr - ytl)
    theta = (rotation / 360.0) * (2.0 * np.pi)

    add_paths.append(filepath)
    add_orients.append(orient)
    add_bboxes.append(bbox)
    add_thetas.append(theta)
    add_label.append(label)


add_gids = ibs.add_images(add_paths)
ibs.set_image_orientation(add_gids, add_orients)

assert len(add_bboxes) == len(add_gids)
assert len(add_thetas) == len(add_gids)
assert len(add_species) == len(add_gids)
assert len(add_label) == len(add_gids)

add_aids = ibs.add_annots(
    add_gids,
    bbox_list=add_bboxes,
    theta_list=add_thetas,
)
# ibs.set_annot_species(add_aids, add_label)
# ibs.set_annot_yaw_texts(add_aids, add_label)
ibs.precompute_web_viewpoint_thumbnails()
