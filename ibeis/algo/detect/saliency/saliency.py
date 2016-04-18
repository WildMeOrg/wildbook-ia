#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import isfile, join, exists
from jpcnn.core import JPCNN_Network, JPCNN_Data
from model import Saliency_Model
from utils import resample
from os import listdir
import utool as ut
import numpy as np
import cv2
from config import (
    PROPOSALS,
    CHIP_SIZE,
    FILTER_EMPTY_IMAGES,
    MAXIMUM_GT_PER_IMAGE,
    CACHE_DATA,
    LEARNING_RATE,
    BATCH_SIZE,
    MAXIMUM_EPOCHS,
    OUTPUT_PATH,
)
from transform import (
    transform_network_output_to_image_bbox_conf,
)
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[saliency]')


def load_saliency(source_path='saliency_detector',
                  cache_data_filename='data.npy',
                  cache_labels_filename='labels.npy',
                  cache=CACHE_DATA):

    cache_data_filepath = join('extracted', cache_data_filename)
    cache_labels_filepath = join('extracted', cache_labels_filename)

    if exists(cache_data_filepath) and exists(cache_labels_filepath) and cache:
        data_list = np.load(cache_data_filepath)
        label_list = np.load(cache_labels_filepath)
        return data_list, label_list

    label_filepath = join('extracted', 'labels', source_path, 'labels.csv')
    label_dict = {}
    frequency_dict = {}
    with open(label_filepath) as labels:
        label_list = labels.read().split()
        for label in label_list:
            label_list = label.strip().split(',')
            filename = label_list[0]
            label_list_ = label_list[-1]
            label_list_ = label_list_.strip().split(';')
            try:
                label_list_ = [
                    map(float, label.strip().split(':'))
                    for label in label_list_
                ]
                label_dict[filename] = label_list_
            except ValueError:
                label_dict[filename] = []
            num_gt = len(label_dict[filename])
            if num_gt not in frequency_dict:
                frequency_dict[num_gt] = 0
            frequency_dict[num_gt] += 1

    print('\nGround-Truth Distributions:')
    for key in sorted(frequency_dict.keys()):
        value = frequency_dict[key]
        if MAXIMUM_GT_PER_IMAGE is not None:
            if key > MAXIMUM_GT_PER_IMAGE:
                key = '(%d)' % (key, )
        print('\t{0: >4}: {1: <4}'.format(key, value))

    background_path = join('extracted', 'raw', source_path)
    filename_list = [
        f for f in listdir(background_path)
        if isfile(join(background_path, f))
    ]

    assert len(label_dict.keys()) == len(filename_list)

    data_list = []
    label_list = []
    print('Loading images...')
    filename_list = filename_list
    for index, filename in enumerate(filename_list):
        if index % 1000 == 0:
            print(index)
        label = label_dict[filename]
        if MAXIMUM_GT_PER_IMAGE is not None:
            if len(label) > MAXIMUM_GT_PER_IMAGE:
                continue
        if FILTER_EMPTY_IMAGES and len(label) == 0:
            continue
        filepath = join(background_path, filename)
        data = cv2.imread(filepath)
        data_list.append(data)
        label_list.append(label)

    data_list = np.array(data_list, dtype=np.uint8)
    label_list = np.array(label_list)

    np.save(cache_data_filepath, data_list)
    np.save(cache_labels_filepath, label_list)

    return data_list, label_list


def load_test_data():
    background_path = join('test')
    filename_list = [
        f for f in listdir(background_path)
        if isfile(join(background_path, f))
    ]

    data_list = []
    original_list = []
    print('Loading images...')
    filename_list = filename_list
    for index, filename in enumerate(filename_list):
        filepath = join(background_path, filename)
        original = cv2.imread(filepath)
        original_list.append(original)
        data = resample(original, CHIP_SIZE[0], CHIP_SIZE[1])
        data_list.append(data)

    data_list = np.array(data_list, dtype=np.uint8)

    return data_list, original_list


def train_saliency(output_path):
    print('[saliency] Loading the Saliency training data')
    data_list, label_list = load_saliency()

    print('[saliency] Loading the data into a JPCNN_Data')
    data = JPCNN_Data()
    data.set_data_list(data_list)
    data.set_label_list(label_list)

    print('[saliency] Create the JPCNN_Model used for training')
    model = Saliency_Model()

    print('[saliency] Create the JPCNN_network and start training')
    net = JPCNN_Network(model, data)
    net.train(
        output_path,
        train_learning_rate=LEARNING_RATE,
        train_batch_size=BATCH_SIZE,
        train_max_epochs=MAXIMUM_EPOCHS,
    )


def test_saliency(output_path):
    test_path = join(output_path, 'output')
    ut.ensuredir(test_path)

    print('[saliency] Loading the Saliency testing data')
    data_list, original_list = load_test_data()

    print('[saliency] Loading the data into a JPCNN_Data')
    data = JPCNN_Data()
    data.set_data_list(data_list)

    print('[saliency] Create the JPCNN_Model used for testing')
    model = Saliency_Model(join(output_path, 'model.npy'))

    print('[saliency] Create the JPCNN_network and start testing')
    net = JPCNN_Network(model, data)
    test_results = net.test(output_path)

    prediction_list = test_results['probability_list']

    values = transform_network_output_to_image_bbox_conf(prediction_list)
    (prediction_bbox_list, prediction_alpha_list), batch_size = values

    for batch in range(batch_size):
        prediction_bbox_ = prediction_bbox_list[batch]
        prediction_alpha_ = prediction_alpha_list[batch]
        original = original_list[batch]
        original = resample(original, height=800)

        height, width = original.shape[:2]

        # Show predictions
        for index in range(PROPOSALS):
            (xc, yc, xr, yr) = prediction_bbox_[index]
            alpha = prediction_alpha_[index]

            xc *= width
            xr *= width
            yc *= height
            yr *= height

            xtl_ = int(round(xc - xr))
            ytl_ = int(round(yc - yr))
            xbr_ = int(round(xc + xr))
            ybr_ = int(round(yc + yr))

            xtl_ = min(width,  max(0, xtl_))
            ytl_ = min(height, max(0, ytl_))
            xbr_ = min(width,  max(0, xbr_))
            ybr_ = min(height, max(0, ybr_))

            if alpha >= 0.90:
                color = (0, 0, 255)
            elif alpha >= 0.50:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            cv2.rectangle(original, (xtl_, ytl_), (xbr_, ybr_), color, 1)

        output_filepath = join(test_path, 'output_%d.png' % (batch, ))
        cv2.imwrite(output_filepath, original)


if __name__ == '__main__':
    # Train network on Saliency training data
    train_saliency(OUTPUT_PATH)

    # Test trained Saliency model on Saliency test data
    test_saliency(OUTPUT_PATH)
