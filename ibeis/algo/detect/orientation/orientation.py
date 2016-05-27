#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import isfile, join, exists
from ibeis.algo.detect.orientation.model import Orientation_Model
from os import listdir
import utool as ut
import numpy as np
import cv2
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[orientation]')


def load_orientation(source_path='orientation',
                     cache_data_filename='data.npy',
                     cache_labels_filename='labels.npy',
                     cache=True):

    cache_data_filepath = join('extracted', cache_data_filename)
    cache_labels_filepath = join('extracted', cache_labels_filename)

    if exists(cache_data_filepath) and exists(cache_labels_filepath) and cache:
        data_list = np.load(cache_data_filepath)
        label_list = np.load(cache_labels_filepath)
        return data_list, label_list

    label_filepath = join('extracted', 'labels', source_path, 'labels.csv')
    label_dict = {}
    with open(label_filepath) as labels:
        label_list = labels.read().split()
        for label in label_list:
            label_list = label.strip().split(',')
            filename = label_list[0]
            orientation = float(label_list[1])
            label_dict[filename] = orientation

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
        filepath = join(background_path, filename)
        data = cv2.imread(filepath)
        data_list.append(data)
        label_list.append(label)

    data_list = np.array(data_list, dtype=np.uint8)
    label_list = np.array(label_list, dtype=np.float32)

    np.save(cache_data_filepath, data_list)
    np.save(cache_labels_filepath, label_list)

    return data_list, label_list


def train_orientation(output_path):
    from jpcnn.core import JPCNN_Network, JPCNN_Data
    print('[orientation] Loading the Orientation training data')
    data_list, label_list = load_orientation()

    print('[orientation] Loading the data into a JPCNN_Data')
    data = JPCNN_Data()
    data.set_data_list(data_list)
    data.set_label_list(label_list)

    print('[orientation] Create the JPCNN_Model used for training')
    model = Orientation_Model()

    print('[orientation] Create the JPCNN_network and start training')
    net = JPCNN_Network(model, data)
    net.train(
        output_path,
        train_learning_rate=0.01,
        train_batch_size=128,
        train_max_epochs=100,
    )


if __name__ == '__main__':
    OUTPUT_PATH = '.'
    # Train network on Orientation training data
    train_orientation(OUTPUT_PATH)
