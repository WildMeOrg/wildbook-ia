# -*- coding: utf-8 -*-
"""
Render Mindy annotation data.

pip install xmltodict
"""
import random
from os.path import join

import cv2
import numpy as np
import utool as ut
import xmltodict

with open(join('example', 'polylines', 'annotations.xml'), 'r') as xml_file:
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
polyline = image.get('polyline')
points_str = polyline.get('@points')
points_ = [point.strip().split(',') for point in points_str.strip().split(';')]
points = [tuple(map(int, map(np.around, map(float, point)))) for point in points_]

filepath = join('example', 'polylines', 'images', filename)

img = cv2.imread(filepath)

prev_point = None
for point in points:
    if prev_point is not None:
        img = cv2.line(img, prev_point, point, (255, 0, 0), 1)
    prev_point = point

chips = {}
idxs = list(range(len(points)))
while len(chips) < 6:
    index = random.choice(idxs)
    valid = True
    if index in chips:
        valid = False
    for val in chips.keys():
        if abs(val - index) < 10:
            valid = False
    if not valid:
        continue
    point = points[index]
    cx, cy = point
    xtl, ytl, xbr, ybr = cx - 40, cy - 40, cx + 40, cy + 40
    chip_ = img[ytl:ybr, xtl:xbr]
    chip = cv2.resize(chip_, (200, 200))
    if chip.shape == (200, 200, 3):
        chips[index] = chip

for index, point in enumerate(points):
    if index == 0:
        img = cv2.circle(img, point, 9, (255, 255, 255), -1)
        img = cv2.circle(img, point, 7, (0, 255, 0), -1)
    elif index == len(points) - 1:
        img = cv2.circle(img, point, 9, (255, 255, 255), -1)
        img = cv2.circle(img, point, 7, (0, 0, 255), -1)
    else:
        color = (0, 0, 255) if index in chips else (0, 255, 0)
        img = cv2.circle(img, point, 3, color)

strip = np.hstack(ut.take(chips, sorted(list(chips.keys()), reverse=True)))
canvas = np.vstack((img, strip))

cv2.imwrite(join('example', 'polylines', 'output.png'), canvas)
