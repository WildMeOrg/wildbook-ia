#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from wbia.detecttools.directory import Directory
from wbia.detecttools.wbiadata import common as com
from wbia.detecttools.wbiadata.wbia_image import IBEIS_Image
from tqdm import tqdm


class IBEIS_Data(object):  # NOQA
    def __init__(ibsd, dataset_path, **kwargs):
        com._kwargs(kwargs, 'object_min_width', 1)
        com._kwargs(kwargs, 'object_min_height', 1)
        com._kwargs(kwargs, 'mine_negatives', False)
        com._kwargs(kwargs, 'mine_width_min', 50)
        com._kwargs(kwargs, 'mine_width_max', 400)
        com._kwargs(kwargs, 'mine_height_min', 50)
        com._kwargs(kwargs, 'mine_height_max', 400)
        com._kwargs(kwargs, 'mine_max_attempts', 100)
        com._kwargs(kwargs, 'mine_max_keep', 10)
        com._kwargs(kwargs, 'mine_overlap_margin', 0.25)
        com._kwargs(kwargs, 'mine_exclude_categories', [])
        com._kwargs(kwargs, 'mine_patches', False)
        com._kwargs(kwargs, 'mine_patch_width', 32)
        com._kwargs(kwargs, 'mine_patch_height', 32)
        com._kwargs(kwargs, 'mine_patch_stride_suggested', 16)
        com._kwargs(kwargs, 'mine_patch_overlap_margin', 0.5)

        ibsd.dataset_path = dataset_path
        ibsd.absolute_dataset_path = os.path.realpath(dataset_path)

        direct = Directory(
            os.path.join(ibsd.dataset_path, 'Annotations'),
            include_file_extensions=['xml'],
        )
        ibsd.images = []
        files = direct.files()
        print('Loading Database')
        for i, filename in tqdm(list(enumerate(files))):
            ibsd.images.append(
                IBEIS_Image(filename, ibsd.absolute_dataset_path, **kwargs)
            )
        print('    ...Loaded')

        ibsd.categories_images = []
        ibsd.categories_rois = []

        for image in ibsd.images:
            temp = image.categories(unique=False, patches=True)
            ibsd.categories_rois += temp
            ibsd.categories_images += set(temp)

            if len(image.objects) == 0:
                ibsd.categories_images += ['BACKGROUND']

        ibsd.distribution_images = com.histogram(ibsd.categories_images)
        ibsd.distribution_rois = com.histogram(ibsd.categories_rois)
        ibsd.rois = sum(ibsd.distribution_rois.values())

        ibsd.categories = sorted(set(ibsd.categories_images))

    def __str__(ibsd):
        return '<IBEIS Data Object | %s | %d images | %d categories | %d rois>' % (
            ibsd.absolute_dataset_path,
            len(ibsd.images),
            len(ibsd.categories),
            ibsd.rois,
        )

    def __repr__(ibsd):
        return '<IBEIS Data Object | %s>' % (ibsd.absolute_dataset_path)

    def __len__(ibsd):
        return len(ibsd.images)

    def __getitem__(ibsd, key):
        if isinstance(key, str):
            for image in ibsd.images:
                if key in image.filename:
                    return image
            return None
        else:
            return ibsd.images[key]

    def print_distribution(ibsd):
        def _print_line(category, spacing, images, rois):
            try:
                images = int(images)
                images = '% 5d' % (images,)
            except Exception:
                pass
            try:
                rois = int(rois)
                rois = '% 5d' % (rois,)
            except Exception:
                pass
            print('%s%s\t%s' % (category + ' ' * (spacing - len(category)), images, rois))

        key_list = list(ibsd.distribution_rois.keys())
        key_list += ['TOTAL', 'CATEGORY']
        _max = max([len(category) for category in key_list]) + 3

        _print_line('CATEGORY', _max, 'IMGs', 'ROIs')
        if 'BACKGROUND' in ibsd.distribution_images:
            _print_line('BACKGROUND', _max, ibsd.distribution_images['BACKGROUND'], '')

        for category in sorted(ibsd.distribution_rois):
            _print_line(
                category,
                _max,
                ibsd.distribution_images[category],
                ibsd.distribution_rois[category],
            )

        _print_line('TOTAL', _max, len(ibsd.images), ibsd.rois)

    def parse_dataset(ibsd, category, _type):
        filename = _type + '.txt'

        if category is not None:
            filename = category + '_' + filename

        filepath = os.path.join(ibsd.dataset_path, 'ImageSets', 'Main', filename)
        _dict = {}
        try:
            _file = open(filepath)
            for line in _file:
                line = line.strip().split(' ')
                _dict[line[0]] = int(line[-1])
        except IOError as e:
            print('<%r> %s' % (e, filepath))

        return _dict

    def dataset(
        ibsd,
        positive_category,
        neg_exclude_categories=[],
        max_rois_pos=None,
        max_rois_neg=None,
    ):
        def _parse_dataset_file(category, _type):
            filepath = os.path.join(
                ibsd.dataset_path, 'ImageSets', 'Main', category + '_' + _type + '.txt'
            )
            _dict = {}
            try:
                _file = open(filepath)
                for line in _file:
                    line = line.strip().split(' ')
                    _dict[line[0]] = int(line[-1])
            except IOError as e:
                print('<%r> %s' % (e, filepath))

            return _dict

        positives = []
        negatives = []
        validation = []
        test = []

        train_values = ibsd.parse_dataset(positive_category, 'train')
        train_values = ibsd.parse_dataset(positive_category, 'trainval')
        # val_values   = _parse_dataset_file(positive_category, "val")
        test_values = ibsd.parse_dataset(positive_category, 'test')

        pos_rois = 0
        neg_rois = 0
        for image in ibsd.images:
            _train = train_values.get(image.filename, 0)
            # _val   = val_values.get(image.filename,   0)
            _test = test_values.get(image.filename, 0)

            temp = image.categories(unique=False)
            flag = False

            if _train != 0:
                for val in temp:
                    if val == positive_category:
                        flag = True
                        pos_rois += 1
                    elif val not in neg_exclude_categories:
                        neg_rois += 1

                if flag:
                    positives.append(image)
                elif val not in neg_exclude_categories:
                    negatives.append(image)

            # if _val != 0:
            #     validation.append(image)

            if _test != 0:
                test.append(image)

        # Setup auto normalize variables for equal positives and negatives
        if max_rois_pos == 'auto' or max_rois_pos == -1:
            max_rois_pos = neg_rois

        if max_rois_neg == 'auto' or max_rois_neg == -1:
            max_rois_neg = pos_rois

        # Remove positives to target, not gauranteed to give target, but 'close'.
        if max_rois_pos is not None and len(positives) > 0:
            pos_density = float(pos_rois) / len(positives)
            target_num = int(max_rois_pos / pos_density)
            print('Normalizing Positives, Target: %d' % target_num)

            # Remove images to match target
            while len(positives) > target_num:
                positives.pop(com.randInt(0, len(positives) - 1))

            # Recalculate rois left
            pos_rois = 0
            for image in positives:
                temp = image.categories(unique=False)
                for val in temp:
                    if val in positive_category:
                        pos_rois += 1

        # Remove negatives to target, not gauranteed to give target, but 'close'.
        if max_rois_neg is not None and len(negatives) > 0:
            neg_density = float(neg_rois) / len(negatives)
            target_num = int(max_rois_neg / neg_density)
            print('Normalizing Negatives, Target: %d ' % target_num)

            # Remove images to match target
            while len(negatives) > target_num:
                negatives.pop(com.randInt(0, len(negatives) - 1))

            # Recalculate rois left
            neg_rois = 0
            for image in negatives:
                temp = image.categories(unique=False)
                for val in temp:
                    if val not in positive_category:
                        neg_rois += 1

        print('%s\t%s\t%s\t%s\t%s' % ('       ', 'Pos', 'Neg', 'Val', 'Test'))
        print(
            '%s\t%s\t%s\t%s\t%s'
            % ('Images:', len(positives), len(negatives), len(validation), len(test))
        )
        print('%s\t%s\t%s\t%s\t%s' % ('ROIs:  ', pos_rois, neg_rois, '', ''))

        return (positives, pos_rois), (negatives, neg_rois), validation, test


if __name__ == '__main__':

    information = {
        'mine_negatives': True,
        'mine_max_keep': 1,
        'mine_exclude_categories': ['zebra_grevys', 'zebra_plains'],
    }

    dataset = IBEIS_Data('test/', **information)
    print(dataset)

    # Access specific information about the dataset
    print('Categories:', dataset.categories)
    print('Number of images:', len(dataset))

    print('')
    dataset.print_distribution()
    print('')

    # Access specific image from dataset using filename or index
    print(dataset['2014_000002'])
    print(dataset['_000002'])  # partial also works (takes first match)
    cont = True
    while cont:
        # Show the detection regions by drawing them on the source image
        print('Enter something to continue, empty to get new image')
        cont = dataset[com.randInt(0, len(dataset) - 1)].show()

    # Get all images using a specific positive set
    (pos, pos_rois), (neg, neg_rois), val, test = dataset.dataset('zebra_grevys')

    print(pos, pos_rois)
    print(neg, neg_rois)
    print(val)
    print(test)

    # Get a specific number of images (-1 for auto normalize to what the other gives)
    # (pos, pos_rois), (neg, neg_rois), val, test = dataset.dataset('zebra_grevys', max_rois_neg=-1)

    print('\nPositives:')
    for _pos in pos:
        print(_pos.image_path())
        print(_pos.bounding_boxes(parts=True))

    print('\nNegatives:')
    for _neg in neg:
        print(_neg.image_path())
        print(_neg.bounding_boxes(parts=True))

    print('\nValidation:')
    for _val in val:
        print(_val.image_path())
        print(_val.bounding_boxes(parts=True))

    print('\nTest:')
    for _test in test:
        print(_test.image_path())
        print(_test.bounding_boxes(parts=True))
