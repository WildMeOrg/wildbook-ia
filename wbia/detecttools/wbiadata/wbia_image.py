#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import cv2
import os
import math
import xml.etree.ElementTree as xml

from . import common as com
from .wbia_object import IBEIS_Object


class IBEIS_Image(object):  # NOQA
    def __init__(ibsi, filename_xml, absolute_dataset_path, **kwargs):
        with open(filename_xml, 'r') as _xml:
            _xml = xml.XML(_xml.read().replace('\n', ''))

            ibsi.folder = com.get(_xml, 'folder')
            ibsi.absolute_dataset_path = absolute_dataset_path
            ibsi.filename = com.get(_xml, 'filename')

            source = com.get(_xml, 'source', text=False)
            ibsi.source_database = com.get(source, 'database')
            ibsi.source_annotation = com.get(source, 'annotation')
            ibsi.source_image = com.get(source, 'image')

            size = com.get(_xml, 'size', text=False)
            ibsi.width = int(com.get(size, 'width'))
            ibsi.height = int(com.get(size, 'height'))
            try:
                ibsi.depth = int(com.get(size, 'depth'))
            except TypeError:
                ibsi.depth = 3

            ibsi.segmented = com.get(size, 'segmented') == '1'

            ibsi.objects = []
            ibsi.objects_patches = []
            ibsi.objects_invalid = []
            for obj in com.get(_xml, 'object', text=False, singularize=False):
                temp = IBEIS_Object(obj, ibsi.width, ibsi.height)
                if (
                    temp.width > kwargs['object_min_width']
                    and temp.height > kwargs['object_min_height']
                ):
                    ibsi.objects.append(temp)
                else:
                    ibsi.objects_invalid.append(temp)

            flag = True
            for cat in ibsi.categories():
                if cat in kwargs['mine_exclude_categories']:
                    flag = False

            if kwargs['mine_negatives'] and flag:
                negatives = 0
                for i in range(kwargs['mine_max_attempts']):
                    if negatives >= kwargs['mine_max_keep']:
                        break

                    width = com.randInt(
                        kwargs['mine_width_min'],
                        min(ibsi.width - 1, kwargs['mine_width_max']),
                    )
                    height = com.randInt(
                        kwargs['mine_height_min'],
                        min(ibsi.height - 1, kwargs['mine_height_max']),
                    )
                    x = com.randInt(0, ibsi.width - width - 1)
                    y = com.randInt(0, ibsi.height - height - 1)

                    obj = {
                        'xmax': x + width,
                        'xmin': x,
                        'ymax': y + height,
                        'ymin': y,
                    }

                    overlap_names = ibsi._overlaps(
                        ibsi.objects, obj, kwargs['mine_overlap_margin']
                    )
                    if len(overlap_names) > 0:
                        continue

                    ibsi.objects.append(
                        IBEIS_Object(obj, ibsi.width, ibsi.height, name='MINED')
                    )
                    negatives += 1

            if kwargs['mine_patches']:
                patch_width = kwargs['mine_patch_width']
                patch_height = kwargs['mine_patch_height']
                x_length = float(ibsi.width - patch_width - 1)
                y_length = float(ibsi.height - patch_height - 1)
                x_bins = int(x_length / kwargs['mine_patch_stride_suggested'])
                y_bins = int(y_length / kwargs['mine_patch_stride_suggested'])

                x_bins = max(1, x_bins)
                y_bins = max(1, y_bins)
                patch_stride_x = x_length / x_bins
                patch_stride_y = y_length / y_bins
                # ibsi.show()
                for x in range(x_bins + 1):
                    for y in range(y_bins + 1):
                        x_min = int(x * patch_stride_x)
                        y_min = int(y * patch_stride_y)
                        x_max = x_min + patch_width
                        y_max = y_min + patch_height
                        assert (
                            0 <= x_min
                            and x_max < ibsi.width
                            and 0 <= y_min
                            and y_max < ibsi.height
                        )
                        # Add patch
                        obj = {
                            'xmax': x_max,
                            'xmin': x_min,
                            'ymax': y_max,
                            'ymin': y_min,
                        }
                        overlap_names = ibsi._overlaps(
                            ibsi.objects, obj, kwargs['mine_patch_overlap_margin']
                        )
                        if len(overlap_names) > 0:
                            for overlap_name in overlap_names:
                                name = '%s' % overlap_name.upper()
                                ibsi.objects_patches.append(
                                    IBEIS_Object(obj, ibsi.width, ibsi.height, name=name)
                                )
                        else:
                            ibsi.objects_patches.append(
                                IBEIS_Object(
                                    obj, ibsi.width, ibsi.height, name='NEGATIVE'
                                )
                            )

    def __str__(ibsi):
        return '<IBEIS Image Object | %s | %d objects>' % (
            ibsi.filename,
            len(ibsi.objects),
        )

    def __repr__(ibsi):
        return '<IBEIS Image Object | %s>' % (ibsi.filename)

    def __len__(ibsi):
        return len(ibsi.objects)

    def __lt__(ibsi1, ibsi2):
        if ibsi1.filename < ibsi2.filename:
            return -1
        if ibsi1.filename < ibsi2.filename:
            return 1
        return 0

    def _distance(pt1, pt2):
        (x1, y1) = pt1
        (x2, y2) = pt2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _overlaps(
        ibsi, objects, obj, margin=0.50, bins=['left', 'front', 'right', 'back']
    ):
        names = []
        for _obj in objects:
            x_overlap = max(0, min(obj['xmax'], _obj.xmax) - max(obj['xmin'], _obj.xmin))
            y_overlap = max(0, min(obj['ymax'], _obj.ymax) - max(obj['ymin'], _obj.ymin))
            area_overlap = float(x_overlap * y_overlap)
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax'] - obj['ymin']
            area_total = min(width * height, _obj.area)
            score = area_overlap / area_total
            # print(score)
            if score >= margin:
                names.append(_obj.name + ':' + _obj.pose_str)
        return list(set(names))

    def image_path(ibsi):
        return os.path.join(ibsi.absolute_dataset_path, 'JPEGImages', ibsi.filename)

    def categories(ibsi, unique=True, sorted_=True, patches=False):
        temp = [_object.name for _object in ibsi.objects]
        if patches:
            temp += [_object.name for _object in ibsi.objects_patches]
        if unique:
            temp = list(set(temp))
        if sorted_:
            temp = sorted(temp)
        return temp

    def bounding_boxes(ibsi, **kwargs):
        return [_object.bounding_box(**kwargs) for _object in ibsi.objects]

    def _accuracy_match(ibsi, prediction, object_list):

        # For this non-supressed prediction, compute and assign to the closest bndbox
        centerx, centery, minx, miny, maxx, maxy, confidence, supressed = prediction

        index_best = None
        score_best = -1.0
        for index, _object in enumerate(object_list):
            width = maxx - minx
            height = maxy - miny

            x_overlap = max(0, min(maxx, _object.xmax) - max(minx, _object.xmin))
            y_overlap = max(0, min(maxy, _object.ymax) - max(miny, _object.ymin))
            area_overlap = float(x_overlap * y_overlap)
            area_total = (width * height) + _object.area
            score = area_overlap / (area_total - area_overlap)

            if score >= score_best:
                # Wooo! Found a (probably) better candidate, but...
                if score == score_best:
                    # Well, this is awkward?
                    assert index_best is not None  # Just to be sure
                    _object_best = object_list[index_best]

                    a = ibsi._distance(
                        (centerx, centery), (_object_best.xcenter, _object_best.ycenter)
                    )
                    b = ibsi._distance(
                        (centerx, centery), (_object.xcenter, _object.ycenter)
                    )
                    if a < b:
                        # Not a better candidate based on distance
                        continue
                    elif a == b:
                        # First come, first serve
                        continue
                # Save new best
                score_best = score
                index_best = index

        return index_best, score_best

    def accuracy(ibsi, prediction_list, category, alpha=0.5):
        # PASCAL ACCURACY MEASUREMENT
        object_list = []
        for _object in ibsi.objects + ibsi.objects_invalid:
            if _object.name == category:
                object_list.append(_object)

        # Trivial case
        if len(object_list) == 0 and len(prediction_list) == 0:
            return 1.0, 0.0, 0.0, 0.0

        true_positive = 0
        false_positive = 0

        counters = [0] * len(object_list)
        for prediction in prediction_list:
            centerx, centery, minx, miny, maxx, maxy, confidence, supressed = prediction
            if supressed == 0.0:
                index_best, score_best = ibsi._accuracy_match(prediction, object_list)
                if score_best >= alpha:
                    counters[index_best] += 1
                    true_positive += 1
                else:
                    false_positive += 1

        false_negative = counters.count(0)
        precision = float(true_positive)
        recall = true_positive + false_positive + false_negative
        assert recall != 0
        return precision / recall, true_positive, false_positive, false_negative

    def show(
        ibsi,
        objects=True,
        parts=True,
        display=True,
        prediction_list=None,
        category=None,
        alpha=0.5,
        label=True,
    ):
        def _draw_box(
            img, annotation, xmin, ymin, xmax, ymax, color, stroke=2, position='top'
        ):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            width, height = cv2.getTextSize(annotation, font, scale, -1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, stroke)
            if label:
                if position in ['top']:
                    cv2.rectangle(
                        img, (xmin, ymin), (xmin + width, ymin + height), color, -1
                    )
                    cv2.putText(
                        img,
                        annotation,
                        (xmin + 5, ymin + height),
                        font,
                        0.4,
                        (255, 255, 255),
                    )
                elif position in ['bottom']:
                    cv2.rectangle(
                        img, (xmin, ymax - height), (xmin + width, ymax), color, -1
                    )
                    cv2.putText(
                        img, annotation, (xmin + 5, ymax), font, 0.4, (255, 255, 255)
                    )

        original = com.openImage(ibsi.image_path(), color=True)
        color_dict = {}
        for _object in ibsi.objects:
            color = com.randColor()
            color_dict[_object] = color
            _draw_box(
                original,
                _object.name.upper(),
                _object.xmin,
                _object.ymin,
                _object.xmax,
                _object.ymax,
                color,
            )

            if parts:
                for part in _object.parts:
                    _draw_box(
                        original,
                        part.name.upper(),
                        part.xmin,
                        part.ymin,
                        part.xmax,
                        part.ymax,
                        color,
                    )

        for _object in ibsi.objects_invalid:
            color = [0, 0, 0]
            color_dict[_object] = color
            _draw_box(
                original,
                _object.name.upper(),
                _object.xmin,
                _object.ymin,
                _object.xmax,
                _object.ymax,
                color,
            )

            if parts:
                for part in _object.parts:
                    _draw_box(
                        original,
                        part.name.upper(),
                        part.xmin,
                        part.ymin,
                        part.xmax,
                        part.ymax,
                        color,
                    )

        for _object in ibsi.objects_patches:
            if _object.name.upper() == 'NEGATIVE':
                continue
                color = [255, 0, 0]
            else:
                color = [0, 0, 255]
            color_dict[_object] = color
            _draw_box(
                original,
                _object.name.upper(),
                _object.xmin,
                _object.ymin,
                _object.xmax,
                _object.ymax,
                color,
            )

            if parts:
                for part in _object.parts:
                    _draw_box(
                        original,
                        part.name.upper(),
                        part.xmin,
                        part.ymin,
                        part.xmax,
                        part.ymax,
                        color,
                    )

        if prediction_list is not None:
            assert category is not None
            object_list = []
            for _object in ibsi.objects + ibsi.objects_invalid:
                if _object.name == category:
                    object_list.append(_object)

            for prediction in prediction_list:
                (
                    centerx,
                    centery,
                    minx,
                    miny,
                    maxx,
                    maxy,
                    confidence,
                    supressed,
                ) = prediction
                if supressed == 0.0:
                    if len(object_list) > 0:
                        index_best, score_best = ibsi._accuracy_match(
                            prediction, object_list
                        )
                        _object_best = object_list[index_best]
                        color = color_dict[_object_best]
                        if score_best >= alpha:
                            annotation = 'DETECT [TRUE POS %.2f]' % score_best
                        else:
                            annotation = 'DETECT [FALSE POS %.2f]' % score_best
                        cv2.line(
                            original,
                            (int(minx), int(miny)),
                            (_object_best.xmin, _object_best.ymin),
                            color,
                            1,
                        )
                        cv2.line(
                            original,
                            (int(minx), int(maxy)),
                            (_object_best.xmin, _object_best.ymax),
                            color,
                            1,
                        )
                        cv2.line(
                            original,
                            (int(maxx), int(miny)),
                            (_object_best.xmax, _object_best.ymin),
                            color,
                            1,
                        )
                        cv2.line(
                            original,
                            (int(maxx), int(maxy)),
                            (_object_best.xmax, _object_best.ymax),
                            color,
                            1,
                        )

                    else:
                        annotation = 'DETECT [FALSE POS]'
                        color = [0, 0, 255]
                    _draw_box(
                        original,
                        annotation,
                        int(minx),
                        int(miny),
                        int(maxx),
                        int(maxy),
                        color,
                        stroke=1,
                        position=False,
                    )

        if display:
            cv2.imshow(ibsi.filename + ' with Bounding Boxes', original)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return original
