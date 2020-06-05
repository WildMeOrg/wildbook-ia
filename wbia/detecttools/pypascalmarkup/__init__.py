#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
from six.moves import map
from datetime import date
import cv2

# import numpy as np
import os


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def _kwargs(kwargs, key, value):
    if key not in kwargs.keys():
        kwargs[key] = value


class PascalVOC_Markup_Annotation(object):  # NOQA
    def __init__(an, fullpath, folder, filename, **kwargs):
        _kwargs(kwargs, 'database_name', 'Image Database')
        _kwargs(kwargs, 'database_year', str(date.today().year))
        _kwargs(
            kwargs, 'source', 'Unknown'
        )  # images source (e.g., flickr, reservation, etc.)
        _kwargs(kwargs, 'segmented', '0')  # image has an associated semgnetation
        _kwargs(kwargs, 'color', True)  # open the image with color
        _kwargs(kwargs, 'alpha', False)  # open the image with alpha

        an.folder = folder
        an.filename = filename

        an.database_name = kwargs['database_name']
        an.database_year = kwargs['database_year']
        an.source = kwargs['source']

        # filepath = os.path.join(folder, filename)
        if not os.path.exists(fullpath):
            raise IOError('Image file not found')

        if not kwargs['color']:
            mode = 0  # Greyscale by default
        elif not kwargs['alpha']:
            mode = 1  # Color without alpha channel
        else:
            mode = -1  # Color with alpha channel

        temp = cv2.imread(fullpath, mode)
        an.height, an.width, an.channels = list(map(str, temp.shape))

        an.segmented = kwargs['segmented']
        an.objects = []

    def add_object(an, name, bounding_box, **kwargs):
        an.objects.append(PascalVOC_Markup_Object(name, bounding_box, **kwargs))

    def add_part(an, object_index, name, bounding_box):
        if object_index <= -len(an.objects) or len(an.objects) <= object_index:
            raise IndexError('Object index is invalid, adding part')
        an.objects[object_index].add_part(name, bounding_box)

    def xml(an):
        return an._export()

    def yml(an):
        return an._export(yml=True)

    def _export(an, yml=False):
        if yml:
            template = open(os.path.join(__location__, 'template_annotation.yml'), 'r')
        else:
            template = open(os.path.join(__location__, 'template_annotation.xml'), 'r')
        template = ''.join(template.readlines())

        template = template.replace('_^_FOLDER_^_', an.folder)
        template = template.replace('_^_FILENAME_^_', an.filename)
        template = template.replace('_^_DARABASE_NAME_^_', an.database_name)
        template = template.replace('_^_DATABASE_YEAR_^_', an.database_year)
        template = template.replace('_^_SOURCE_^_', an.source)
        template = template.replace('_^_WIDTH_^_', an.width)
        template = template.replace('_^_HEIGHT_^_', an.height)
        template = template.replace('_^_CHANNELS_^_', an.channels)
        template = template.replace('_^_SEGMENTED_^_', an.segmented)

        if yml:
            objects = [ob.yml() for ob in an.objects]
        else:
            objects = [ob.xml() for ob in an.objects]
        template = template.replace('_^_OBJECT_MULTIPLE_^_', ''.join(objects))

        return template


class PascalVOC_Markup_Object(object):  # NOQA

    # take bounding box coordinates in the same order as PASCAL-VOC
    def __init__(ob, name, bbox_XxYy, **kwargs):
        (xmax, xmin, ymax, ymin) = bbox_XxYy
        _kwargs(kwargs, 'pose', 'Unspecified')  # Left, Right, Front, Rear
        _kwargs(kwargs, 'interest', 'Unspecified')  # None
        _kwargs(
            kwargs, 'truncated', '0'
        )  # boolean flag, if there exists a partial object in the image
        _kwargs(
            kwargs, 'difficult', '0'
        )  # boolean flag, if difficult case from previous years

        ob.name = name
        ob.pose = kwargs['pose']
        ob.interest = kwargs['interest']
        ob.truncated = kwargs['truncated']
        ob.difficult = kwargs['difficult']

        ob.bounding_box = (xmin, ymin, xmax, ymax)
        ob.xmin = str(xmin)
        ob.ymin = str(ymin)
        ob.xmax = str(xmax)
        ob.ymax = str(ymax)

        ob.parts = []

    def add_part(ob, name, bounding_box):
        ob.parts.append(PascalVOC_Markup_Part(name, bounding_box))

    def xml(ob):
        return ob._export()

    def yml(ob):
        return ob._export(yml=True)

    def _export(ob, yml=False):
        if yml:
            template = open(os.path.join(__location__, 'template_object.yml'), 'r')
        else:
            template = open(os.path.join(__location__, 'template_object.xml'), 'r')
        template = ''.join(template.readlines())

        template = template.replace('_^_NAME_^_', ob.name)
        template = template.replace('_^_POSE_^_', ob.pose)
        template = template.replace('_^_INTEREST_^_', ob.interest)
        template = template.replace('_^_TRUNCATED_^_', ob.truncated)
        template = template.replace('_^_DIFFICULT_^_', ob.difficult)
        template = template.replace('_^_XMIN_^_', ob.xmin)
        template = template.replace('_^_YMIN_^_', ob.ymin)
        template = template.replace('_^_XMAX_^_', ob.xmax)
        template = template.replace('_^_YMAX_^_', ob.ymax)

        if yml:
            parts = [pt.yml() for pt in ob.parts]
        else:
            parts = [pt.xml() for pt in ob.parts]
        template = template.replace('_^_PART_MULTIPLE_OPTIONAL_^_', ''.join(parts))

        return template


class PascalVOC_Markup_Part(object):  # NOQA
    def __init__(pt, name, bbox):
        (xmin, ymin, xmax, ymax) = bbox
        pt.name = name

        pt.bounding_box = (xmin, ymin, xmax, ymax)
        pt.xmin = str(xmin)
        pt.ymin = str(ymin)
        pt.xmax = str(xmax)
        pt.ymax = str(ymax)

    def xml(pt):
        return pt._export()

    def yml(pt):
        return pt._export(yml=True)

    def _export(pt, yml=False):
        if yml:
            template = open(os.path.join(__location__, 'template_part.yml'), 'r')
        else:
            template = open(os.path.join(__location__, 'template_part.xml'), 'r')
        template = ''.join(template.readlines())

        template = template.replace('_^_NAME_^_', pt.name)
        template = template.replace('_^_XMIN_^_', pt.xmin)
        template = template.replace('_^_YMIN_^_', pt.ymin)
        template = template.replace('_^_XMAX_^_', pt.xmax)
        template = template.replace('_^_YMAX_^_', pt.ymax)

        return template


if __name__ == '__main__':

    information = {'database_name': 'IBEIS', 'source': 'olpajeta'}

    annotation = PascalVOC_Markup_Annotation('.', 'test.png', **information)
    annotation.add_object('person', (100, 100, 400, 400))
    annotation.add_object('dog', (10, 200, 100, 800))

    annotation.add_part(0, 'head', (10, 200, 20, 210))
    annotation.add_part(0, 'foot', (1000, 2000, 200, 2100))

    print(annotation.xml())
    print(annotation.yml())
