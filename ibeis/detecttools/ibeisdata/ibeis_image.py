#!/usr/bin/env python

import cv2
import os
import xml.etree.ElementTree as xml

import common as com
from ibeis_object import IBEIS_Object


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

            ibsi.objects = [ IBEIS_Object(obj, ibsi.width, ibsi.height) for obj in com.get(_xml, 'object', text=False, singularize=False) ]

            for _object in ibsi.objects:
                if _object.width <= kwargs['object_min_width'] or \
                   _object.height <= kwargs['object_min_height']:
                    # Remove objects that are too small.
                    ibsi.objects.remove(_object)

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
            %(ibsi.filename, len(ibsi.objects))

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

    def show(ibsi, objects=True, parts=True, display=True):

        def _draw_box(img, annotation, xmin, ymin, xmax, ymax, color):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5

            width, height = cv2.getTextSize(annotation, font, scale, -1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(img, (xmin, ymin - height), (xmin + width, ymin), color, -1)
            cv2.putText(img, annotation, (xmin + 5, ymin), font, 0.4, (255, 255, 255))

        original = com.openImage(ibsi.image_path(), color=True)

        for _object in ibsi.objects:
            color = com.randColor()
            _draw_box(original, _object.name.upper(), _object.xmin, _object.ymin, _object.xmax, _object.ymax, color)

            if parts:
                for part in _object.parts:
                    _draw_box(original, part.name.upper(), part.xmin, part.ymin, part.xmax, part.ymax, color)

        if display:
            cv2.imshow(ibsi.filename + " with Bounding Boxes", original)
            cont = raw_input()
            cv2.destroyAllWindows()
            return cont == ""
        else:
            return original
