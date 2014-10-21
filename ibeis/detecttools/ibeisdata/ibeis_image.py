#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import cv2
import os
import math
import xml.etree.ElementTree as xml

from . import common as com
from .ibeis_object import IBEIS_Object


class IBEIS_Image(object):

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
            ibsi.depth = int(com.get(size, 'depth'))

            ibsi.segmented = com.get(size, 'segmented') == "1"

            ibsi.objects = []
            ibsi.objects_invalid = []
            for obj in com.get(_xml, 'object', text=False, singularize=False):
                temp = IBEIS_Object(obj, ibsi.width, ibsi.height)
                if temp.width > kwargs['object_min_width'] and temp.height > kwargs['object_min_height']:
                    ibsi.objects.append(temp)
                else:
                    ibsi.objects_invalid.append(temp)

            flag = True
            for cat in ibsi.categories():
                if cat in kwargs['mine_exclude_categories']:
                    flag = False

            if kwargs['mine_negatives'] and flag:

                def _overlaps(objects, obj, margin):
                    for _obj in objects:
                        leftA   = obj['xmin']
                        rightA  = obj['xmax']
                        bottomA = obj['ymin']
                        topA    = obj['ymax']
                        widthA = rightA - leftA
                        heightA = topA - bottomA

                        leftB   = _obj.xmin + 0.25 * min(_obj.width, widthA)
                        rightB  = _obj.xmax - 0.25 * min(_obj.width, widthA)
                        bottomB = _obj.ymin + 0.25 * min(_obj.height, heightA)
                        topB    = _obj.ymax - 0.25 * min(_obj.height, heightA)

                        if (leftA < rightB) and (rightA > leftB) and \
                           (topA > bottomB) and (bottomA < topB):
                            return True

                    return False

                negatives = 0
                for i in range(kwargs['mine_max_attempts']):
                    if negatives >= kwargs['mine_max_keep']:
                        break

                    width = com.randInt(kwargs['mine_width_min'], min(ibsi.width - 1, kwargs['mine_width_max']))
                    height = com.randInt(kwargs['mine_height_min'], min(ibsi.height - 1, kwargs['mine_height_max']))
                    x = com.randInt(0, ibsi.width - width - 1)
                    y = com.randInt(0, ibsi.height - height - 1)

                    obj = {
                        'xmax': x + width,
                        'xmin': x,
                        'ymax': y + height,
                        'ymin': y,
                    }

                    if _overlaps(ibsi.objects, obj, kwargs["mine_overlap_margin"]):
                        continue

                    ibsi.objects.append(IBEIS_Object(obj, ibsi.width, ibsi.height, implicit=False))
                    negatives += 1

    def __str__(ibsi):
        return "<IBEIS Image Object | %s | %d objects>" \
            % (ibsi.filename, len(ibsi.objects))

    def __repr__(ibsi):
        return "<IBEIS Image Object | %s>" % (ibsi.filename)

    def __len__(ibsi):
        return len(ibsi.objects)

    def image_path(ibsi):
        return os.path.join(ibsi.absolute_dataset_path, "JPEGImages", ibsi.filename)

    def categories(ibsi, unique=True):
        temp = [ _object.name for _object in ibsi.objects ]
        if unique:
            temp = set(temp)
        return sorted(temp)

    def bounding_boxes(ibsi, parts=False):
        return [ _object.bounding_box(parts) for _object in ibsi.objects ]

    def _accuracy_match(ibsi, prediction, object_list):
        def _distance((x1, y1), (x2, y2)):
            return math.sqrt( (x1 - x2) ** 2 + (y1 - y2) ** 2 )

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

                    a = _distance((centerx, centery), (_object_best.xcenter, _object_best.ycenter))
                    b = _distance((centerx, centery), (_object.xcenter, _object.ycenter))
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

        true_positive  = 0
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

    def show(ibsi, objects=True, parts=True, display=True, prediction_list=None, category=None, alpha=0.5):

        def _draw_box(img, annotation, xmin, ymin, xmax, ymax, color, stroke=2, top=True):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            width, height = cv2.getTextSize(annotation, font, scale, -1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, stroke)
            if top:
                cv2.rectangle(img, (xmin, ymin - height), (xmin + width, ymin), color, -1)
                cv2.putText(img, annotation, (xmin + 5, ymin), font, 0.4, (255, 255, 255))
            else:
                cv2.rectangle(img, (xmin, ymax - height), (xmin + width, ymax), color, -1)
                cv2.putText(img, annotation, (xmin + 5, ymax), font, 0.4, (255, 255, 255))

        original = com.openImage(ibsi.image_path(), color=True)
        color_dict = {}
        for _object in ibsi.objects:
            color = com.randColor()
            color_dict[_object] = color
            _draw_box(original, _object.name.upper(), _object.xmin, _object.ymin, _object.xmax, _object.ymax, color)

            if parts:
                for part in _object.parts:
                    _draw_box(original, part.name.upper(), part.xmin, part.ymin, part.xmax, part.ymax, color)

        for _object in ibsi.objects_invalid:
            color = [0, 0, 0]
            color_dict[_object] = color
            _draw_box(original, _object.name.upper(), _object.xmin, _object.ymin, _object.xmax, _object.ymax, color)

            if parts:
                for part in _object.parts:
                    _draw_box(original, part.name.upper(), part.xmin, part.ymin, part.xmax, part.ymax, color)

        if prediction_list is not None:
            assert category is not None
            object_list = []
            for _object in ibsi.objects + ibsi.objects_invalid:
                if _object.name == category:
                    object_list.append(_object)

            for prediction in prediction_list:
                centerx, centery, minx, miny, maxx, maxy, confidence, supressed = prediction
                if supressed == 0.0:
                    if len(object_list) > 0:
                        index_best, score_best = ibsi._accuracy_match(prediction, object_list)
                        _object_best = object_list[index_best]
                        color = color_dict[_object_best]
                        if score_best >= alpha:
                            annotation = 'DETECT [TRUE POS %.2f]' % score_best
                        else:
                            annotation = 'DETECT [FALSE POS %.2f]' % score_best
                        cv2.line(original, (int(minx), int(miny)), (_object_best.xmin, _object_best.ymin), color, 1)
                        cv2.line(original, (int(minx), int(maxy)), (_object_best.xmin, _object_best.ymax), color, 1)
                        cv2.line(original, (int(maxx), int(miny)), (_object_best.xmax, _object_best.ymin), color, 1)
                        cv2.line(original, (int(maxx), int(maxy)), (_object_best.xmax, _object_best.ymax), color, 1)

                    else:
                        annotation = 'DETECT [FALSE POS]'
                        color = [0, 0, 255]
                    _draw_box(original, annotation, int(minx), int(miny), int(maxx), int(maxy), color, stroke=1, top=False)

        if display:
            cv2.imshow(ibsi.filename + " with Bounding Boxes", original)
            cont = raw_input()
            cv2.destroyAllWindows()
            return cont == ""
        else:
            return original
