# -*- coding: utf-8 -*-
"""
Render Mindy annotation data.

pip install xmltodict
"""
from os.path import join

import cv2
import numpy as np
import utool as ut
import vtool as vt
import xmltodict
from PIL import ExifTags, Image

with open(join('example', 'boxes', 'annotations.xml'), 'r') as xml_file:
    data = xmltodict.parse(xml_file.read())

annotations = data.get('annotations', [])
version = annotations.get('version')
meta = annotations.get('meta')
image = annotations.get('image')

print('Using annotation tool version {}'.format(version))
print('Exported on {}'.format(meta.get('dumped')))
print('Task:')
print(ut.repr3(meta.get('task')))

filename = image.get('@name')
bbox = image.get('box')
label = bbox.get('@label')
xtl = int(np.around(float(bbox.get('@xtl'))))
ytl = int(np.around(float(bbox.get('@ytl'))))
xbr = int(np.around(float(bbox.get('@xbr'))))
ybr = int(np.around(float(bbox.get('@ybr'))))
rotation = float(bbox.get('@rotation'))

filepath = join('example', 'boxes', 'images', filename)

img = Image.open(filepath)
globals().update(locals())
exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
orient = exif.get('Orientation', None)
img.close()

img = cv2.imread(filepath)

if orient in [0, 1, None]:
    pass
elif orient in [3]:
    img = cv2.rotate(img, cv2.ROTATE_180)
elif orient in [6]:
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
elif orient in [8]:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
else:
    raise ValueError('unsupported exif')

bbox = (xtl, ytl, xbr - xtl, ybr - ytl)
theta = (rotation / 360.0) * (2.0 * np.pi)
R = vt.rotation_around_bbox_mat3x3(theta, bbox)

# Get verticies of the annotation polygon
verts = vt.verts_from_bbox(bbox, close=True)
# Rotate and transform vertices
xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
rotated_verts = np.round(trans_pts).astype(np.int).T.tolist()

prev_rotated_vert = None
first = True
for rotated_vert in rotated_verts:
    if prev_rotated_vert is not None:
        color = (0, 255, 0) if first else (255, 0, 0)
        img = cv2.line(img, prev_rotated_vert, rotated_vert, color, 3)
        first = False
    prev_rotated_vert = rotated_vert

if orient in [0, 1, None]:
    pass
elif orient in [3]:
    img = cv2.rotate(img, cv2.ROTATE_180)
elif orient in [6]:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
elif orient in [8]:
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
else:
    raise ValueError('unsupported exif')

cv2.imwrite(join('example', 'boxes', 'output.png'), img)
