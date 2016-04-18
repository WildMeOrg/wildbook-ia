#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from jpcnn.core import JPCNN_Network, JPCNN_Data
from jpcnn.core.model import JPCNN_Default_Model
from jpcnn.tpl import _theano, _lasagne  # NOQA
from jpcnn.tpl._theano import T
from os.path import isfile, join, abspath, exists  # NOQA
from os import listdir
import utool as ut
import cv2  # NOQA
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.linear_assignment_ import linear_assignment
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[saliency]')


PROPOSALS = 64

CHIP_SIZE = (227, 227)
# CHIP_SIZE = (128, 128)
# CHIP_SIZE = (138, 138)

APPLY_RESIDUAL = True
APPLY_NORMALIZATION = True

ALPHA = 0.3
C = 0.01
EPSILON = 0.2

RESIDUALS = None
RESIDUALS_ = None


def F_loc(x, l, g):
    pairwise = cdist(l, g, metric='euclidean')
    pairwise = pairwise ** 2
    pairwise = pairwise * x
    result = 0.5 * np.sum(pairwise)
    return result


def F_conf(x, c):
    value1 = np.log(c)
    value1 = x * value1[:, None]
    value1 = np.sum(value1)
    value2 = np.log(1.0 - c)
    value_ = np.sum(x, axis=1)
    value2 = (1.0 - value_) * value2
    value2 = np.sum(value2)
    result = -1.0 * value1 + -1.0 * value2
    return result


def F(x, c, l, g, alpha, verbose=False, **kwargs):
    f_conf = F_conf(x, c)
    f_loc = F_loc(x, l, g)
    f_final = f_conf + alpha * f_loc

    if verbose:
        print('f_conf: %r, f_loc: %r (%r) [%r]' % (f_conf, f_loc, f_loc * alpha, alpha, ))
        print('f_final: %r' % (f_final, ))

    return f_final


def _assignment_vector(net_output, index):
        x = np.zeros((net_output, 1), dtype=np.uint8)
        x[index, 0] = 1
        return x


def assignment_hungarian(cand_bbox_list, cand_prob_list, bbox_list, **kwargs):
    net_output = cand_bbox_list.shape[0]
    num, _ = bbox_list.shape
    cost_matrix = np.zeros((net_output, num))
    index_list = np.array([ (i, j) for i in range(net_output) for j in range(num) ])
    cost_list = np.array([
        F(
            _assignment_vector(net_output, i),
            cand_prob_list,
            cand_bbox_list,
            bbox_list[j][None, :],
            **kwargs
        )
        for (i, j) in index_list
    ])
    cost_matrix[index_list[:, 0], index_list[:, 1]] = cost_list
    assert not np.isinf(np.max(cost_matrix))
    x = linear_assignment(cost_matrix)

    x = x[:, ::-1]
    indices = np.argsort(x[:, 0], axis=0)
    x = x[indices]
    return x


def assignment_partitioning(cand_bbox_list, cand_prob_list, bbox_list, **kwargs):
    net_output = cand_bbox_list.shape[0]

    bbox_list_ = transform_image_to_residual_space(bbox_list)
    distance_list = cdist(RESIDUALS_, bbox_list_, metric='euclidean')
    selection_list = np.argmin(distance_list, axis=1)

    indices = range(len(bbox_list_))
    assignments = []
    for i in indices:
        selection = np.where(selection_list == i)[0]
        best_energy = np.inf
        best_selection = np.nan
        energy_list = []
        for j in selection:
            energy = F(
                _assignment_vector(net_output, j),
                cand_prob_list,
                cand_bbox_list,
                bbox_list_[i][None, :],
                **kwargs
            )
            energy_list.append(energy)
            if energy <= best_energy:
                best_energy = energy
                best_selection = j
        if np.isnan(best_selection):
            print(i, indices)
            print(distance_list)
            print(selection_list)
            print(selection)
            print(cand_bbox_list)
            print(cand_prob_list)
            print(energy_list)
            # ut.embed()
        assignments.append(best_selection)

    x = np.array(zip(assignments, indices))

    x = x[:, ::-1]
    indices = np.argsort(x[:, 0], axis=0)
    x = x[indices]
    return x


def assignment_solution(cand_bbox_list, cand_prob_list, bbox_list, **kwargs):
    if APPLY_RESIDUAL:
        x = assignment_partitioning(cand_bbox_list, cand_prob_list, bbox_list, **kwargs)
    else:
        x = assignment_hungarian(cand_bbox_list, cand_prob_list, bbox_list, **kwargs)
    assert not np.isnan(np.min(x))
    x = x.astype(np.int_)
    return x


def transform_spaces(vector, transformer):
    theano_ = False
    try:
        cast_ = len(vector.shape) == 2
    except TypeError:
        cast_ = False
        theano_ = True
    if cast_:
        vector = np.array([vector])
    half = 2
    vector_1 = vector[:, :, :half]
    vector_2 = vector[:, :, half:]
    vector_2 = transformer(vector_2)
    if theano_:
        vector = T.concatenate((vector_1, vector_2), axis=2)
    else:
        vector = np.concatenate((vector_1, vector_2), axis=2)
    if cast_:
        vector = vector[0]
    return vector


def transform_image_to_residual_space(vector):
    def _transformer(list):
        return C / (EPSILON + list)
    return transform_spaces(vector, _transformer)


def transform_residual_to_image_space(vector):
    def _transformer(list):
        return (C / list) - EPSILON
    return transform_spaces(vector, _transformer)


def network_output_to_bbox_conf(prediction_list):
    batch_size = prediction_list.shape[0]
    marker = 4 * PROPOSALS
    prediction_bbox_list = prediction_list[:, :marker]
    prediction_alpha_list = prediction_list[:, marker:]
    prediction_bbox_list = prediction_bbox_list.reshape((batch_size, PROPOSALS, -1))
    return (prediction_bbox_list, prediction_alpha_list), batch_size


def network_output_to_image_space(prediction_list):
    values, batch_size = network_output_to_bbox_conf(prediction_list)
    prediction_bbox_list, prediction_alpha_list = values
    prediction_bbox_list = transform_residual_to_image_space(prediction_bbox_list)
    return (prediction_bbox_list, prediction_alpha_list), batch_size


class ResidualCenteringLayer(_lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return -1.0 * (input - RESIDUALS)


class ResidualTransformationLayer(_lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        shape_ = input.shape
        input = input.reshape((shape_[0], PROPOSALS, 4))
        input = transform_image_to_residual_space(input)
        input = input.reshape(shape_)
        return input


class Saliency_Model(JPCNN_Default_Model):
    def __init__(model, *args, **kwargs):
        super(Saliency_Model, model).__init__(*args, **kwargs)
        model.attached_data_whiten_mean = 118.380948
        model.attached_data_whiten_std = 61.896913

    def augmentation(model, X_list, y_list=None):
        for index, y in enumerate(y_list):
            X = X_list[index].copy()
            # Adjust the exposure
            min_ = np.min(X)
            max_ = np.max(X)
            margin = np.min([min_, 255 - max_, 64])
            exposure = np.random.randint(-margin, margin) if margin > 0 else 0
            X += exposure
            # Horizontal flip
            if np.random.uniform() <= 0.5:
                X = cv2.flip(X, 1)
                y_ = [ (1.0 - x_, y_, w_, h_) for (x_, y_, w_, h_) in y ]
            else:
                y_ = y
            # Reshape
            X_ = X.reshape(X_list[index].shape)
            # Save
            X_list[index] = X_
            y_list[index] = y_
        return X_list, y_list

    def _compute_accuracy(model, X_list, y_list, prediction_list, min_conf=0.90, **kwargs):
        values, batch_size = network_output_to_image_space(prediction_list)
        prediction_bbox_list, prediction_alpha_list = values

        alpha_list = []
        min_alpha = 1.0
        max_alpha = 0.0
        score_list = []
        min_score = 1.0
        max_score = 0.0

        assignments_dict = {}
        found_dict = {}
        for batch in range(batch_size):
            prediction_bbox_ = prediction_bbox_list[batch]
            prediction_alpha_ = prediction_alpha_list[batch]
            y_list_ = np.array(y_list[batch])

            num_gt = len(y_list_)
            if num_gt > 0:
                x = assignment_solution(
                    prediction_bbox_,
                    prediction_alpha_,
                    y_list_,
                    alpha=ALPHA,
                    # verbose=batch == 0,
                )
                indices = x[:, 0]
                assignments = x[:, 1]

                for picked_x in assignments:
                    if picked_x not in assignments_dict:
                        assignments_dict[picked_x] = 0
                    assignments_dict[picked_x] += 1

                matched_bbox_list = prediction_bbox_[assignments]
                matched_alpha_list = prediction_alpha_[assignments]
                matched_y_list = y_list_[indices]

                zipped = zip(matched_bbox_list, matched_alpha_list, matched_y_list)
                num_found = 0
                for matched_bbox, matched_alpha, matched_y in zipped:
                    difference = matched_bbox - matched_y
                    score = np.sum(difference * difference)
                    alpha = matched_alpha

                    score_list.append(score)
                    min_score = min(score, min_score)
                    max_score = max(score, max_score)

                    alpha_list.append(alpha)
                    min_alpha = min(alpha, min_alpha)
                    max_alpha = max(alpha, max_alpha)

                    if matched_alpha >= min_conf:
                        num_found += 1

                if num_found not in found_dict:
                    found_dict[num_found] = 0
                found_dict[num_found] += 1

        avg_alpha = np.inf if len(alpha_list) == 0 else sum(alpha_list) / len(alpha_list)
        avg_score = np.inf if len(score_list) == 0 else sum(score_list) / len(score_list)

        print('-' * 80)
        print('alpha min: %0.08f max: %0.08f avg: %0.08f' % (min_alpha, max_alpha, avg_alpha, ))
        print('score min: %0.08f max: %0.08f avg: %0.08f' % (min_score, max_score, avg_score, ))

        print('TOP Picked X Assignments:')
        item_list = list(assignments_dict.iteritems())
        item_list_rev = sorted(item_list, key=lambda tup: tup[1], reverse=True)
        zipped = zip(item_list, item_list_rev)
        for index, ((picked_x, value), (picked_x_rev, value_rev)) in enumerate(zipped):
            if index > 8:
                continue
            print('\t{0: >2}: {1: <2}\t{2: >2}: {3: <2}'.format(picked_x, value, picked_x_rev, value_rev))

        print('TOP Found Assignments:')
        item_list = list(found_dict.iteritems())
        item_list_rev = sorted(item_list, key=lambda tup: tup[1], reverse=True)
        zipped = zip(item_list, item_list_rev)
        for index, ((picked_x, value), (picked_x_rev, value_rev)) in enumerate(zipped):
            if index > 8:
                continue
            print('\t{0: >2}: {1: <2}\t{2: >2}: {3: <2}'.format(picked_x, value, picked_x_rev, value_rev))

        return avg_score

    def _fix_ground_truth(model, yb, prediction):
        new_yb = np.zeros(prediction.shape, dtype=prediction.dtype)

        marker = 4 * PROPOSALS
        shape_ = (marker, )
        values, batch_size = network_output_to_bbox_conf(prediction)
        prediction_bbox, prediction_alpha = values

        for batch in range(batch_size):
            prediction_bbox_ = prediction_bbox[batch]
            prediction_alpha_ = prediction_alpha[batch]
            temp = prediction_bbox_.copy()

            num_gt = len(yb[batch])
            if num_gt > 0:
                # Get yb_
                yb_ = np.array(yb[batch])

                x = assignment_solution(
                    prediction_bbox_,
                    prediction_alpha_,
                    yb_,
                    alpha=ALPHA,
                )
                indices = x[:, 0]
                assignments = x[:, 1]

                # Prepare to set default values for bboxes
                if APPLY_RESIDUAL:
                    yb_ = transform_image_to_residual_space(yb_)
                    output = yb_[indices]
                    residuals_ = np.take(RESIDUALS_, assignments, axis=0)
                    temp[assignments] = -1.0 * (output - residuals_)
                else:
                    temp[assignments] = yb_[indices]

                # Normalize prediction
                if APPLY_NORMALIZATION:
                    maximum = 1.0
                    minimum = 0.0
                    temp[:, 0][np.where(temp[:, 0] > maximum)] = maximum
                    temp[:, 1][np.where(temp[:, 1] > maximum)] = maximum
                    temp[:, 0][np.where(temp[:, 0] < minimum)] = minimum
                    temp[:, 1][np.where(temp[:, 1] < minimum)] = minimum
                    temp[:, 2][np.where(temp[:, 2] < minimum)] = minimum
                    temp[:, 3][np.where(temp[:, 3] < minimum)] = minimum

                # Set default values for alpha
                new_yb[batch][assignments + marker] = 1.0

            # if num_gt == 0, then set predicted bboxes and no alphas
            temp = temp.reshape(shape_)
            new_yb[batch, :marker] = temp

        return new_yb

    def _loss_function(model, prediction, target):
        marker = 4 * PROPOSALS

        prediction_bbox = prediction[:, :marker]
        prediction_alpha = prediction[:, marker:]

        target_bbox = target[:, :marker]
        target_alpha = target[:, marker:]

        loss_bbox = _lasagne.objectives.squared_error(prediction_bbox, target_bbox)
        loss_alpha = _lasagne.objectives.binary_crossentropy(prediction_alpha, target_alpha)

        # loss_bbox *= 0.5
        # loss_bbox *= ALPHA

        return T.concatenate((loss_bbox, loss_alpha), axis=1)

    def get_loss_function(model):
        return model._loss_function

    def architecture(model, batch_size, in_width, in_height, in_channels,
                     out_classes):
        """
        """
        out_classes = PROPOSALS

        _PretrinedNet = _lasagne.PretrainedNetwork('caffenet_full')
        l_in = _lasagne.layers.InputLayer(
            shape=(None, in_channels, in_width, in_height)
            # shape=(None, 3, 227, 227)
        )

        l_conv0 = _lasagne.Conv2DLayer(
            l_in,
            num_filters=96,
            filter_size=(11, 11),
            stride=(4, 4),
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_PretrinedNet.get_pretrained_layer(0),
            b=_PretrinedNet.get_pretrained_layer(1),
        )

        l_pool0 = _lasagne.MaxPool2DLayer(
            l_conv0,
            pool_size=(3, 3),
            stride=(2, 2),
        )

        l_lrn0 = _lasagne.layers.LocalResponseNormalization2DLayer(
            l_pool0,
            alpha=0.0001,
            beta=0.75,
            n=5
        )

        l_conv1 = _lasagne.Conv2DCCLayerGroup(
            l_lrn0,
            num_filters=256,
            filter_size=(5, 5),
            stride=(1, 1),
            group=2,
            pad=2,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_PretrinedNet.get_pretrained_layer(2),
            b=_PretrinedNet.get_pretrained_layer(3),
        )

        l_pool1 = _lasagne.MaxPool2DLayer(
            l_conv1,
            pool_size=(3, 3),
            stride=(2, 2),
        )

        l_lrn1 = _lasagne.layers.LocalResponseNormalization2DLayer(
            l_pool1,
            alpha=0.0001,
            beta=0.75,
            n=5
        )

        l_conv2 = _lasagne.Conv2DLayer(
            l_lrn1,
            num_filters=384,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_PretrinedNet.get_pretrained_layer(4),
            b=_PretrinedNet.get_pretrained_layer(5),
        )

        l_conv3 = _lasagne.Conv2DCCLayerGroup(
            l_conv2,
            num_filters=384,
            filter_size=(3, 3),
            stride=(1, 1),
            group=2,
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_PretrinedNet.get_pretrained_layer(6),
            b=_PretrinedNet.get_pretrained_layer(7),
        )

        l_conv4 = _lasagne.Conv2DCCLayerGroup(
            l_conv3,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            group=2,
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_PretrinedNet.get_pretrained_layer(8),
            b=_PretrinedNet.get_pretrained_layer(9),
        )

        l_pool4 = _lasagne.MaxPool2DLayer(
            l_conv4,
            pool_size=(3, 3),
            stride=(2, 2),
        )

        l_hidden1 = _lasagne.layers.DenseLayer(
            l_pool4,
            num_units=1024,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_lasagne.init.Orthogonal(),
        )

        # l_hidden1_maxout = _lasagne.layers.FeaturePoolLayer(
        #     l_hidden1,
        #     pool_size=2,
        # )

        # l_hidden1_dropout = _lasagne.layers.DropoutLayer(
        #     l_hidden1_maxout,
        #     p=0.5
        # )

        l_hidden2 = _lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=1024,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_lasagne.init.Orthogonal(),
        )

        # l_hidden2_maxout = _lasagne.layers.FeaturePoolLayer(
        #     l_hidden2,
        #     pool_size=2,
        # )

        # l_hidden2_dropout = _lasagne.layers.DropoutLayer(
        #     l_hidden2_maxout,
        #     p=0.5
        # )

        l_out_bbox_raw = _lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=out_classes * 4,
            nonlinearity=_lasagne.nonlinearities.linear,
            W=_lasagne.init.Orthogonal(),
        )

        if APPLY_RESIDUAL and RESIDUALS is not None:
            l_out_bbox_transformed = ResidualTransformationLayer(
                l_out_bbox_raw,
            )

            l_out_bbox_centered = ResidualCenteringLayer(
                l_out_bbox_transformed,
            )

            l_out_bbox = l_out_bbox_centered
        else:
            l_out_bbox = l_out_bbox_raw

        l_out_alpha = _lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=out_classes,
            nonlinearity=_lasagne.nonlinearities.sigmoid,
            W=_lasagne.init.Orthogonal(),
        )

        l_out = _lasagne.layers.ConcatLayer(
            [l_out_bbox, l_out_alpha],
            axis=1,
        )

        return l_out


def resample(image, width=None, height=None):
    if width is None and height is None:
        return None
    if width is not None and height is None:
        height = int((float(width) / len(image[0])) * len(image))
    if height is not None and width is None:
        width = int((float(height) / len(image)) * len(image[0]))

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


def load_saliency(source_path='saliency_detector', cache_data='data.npy',
                  cache_labels='labels.npy', recompute_cache=False):

    if exists(cache_data) and exists(cache_labels) and not recompute_cache:
        data_list = np.load(cache_data)
        label_list = np.load(cache_labels)

        # data_list = data_list[:10]
        # label_list = label_list[:10]

        return data_list, label_list

    label_path = join('extracted', 'labels', source_path, 'labels.csv')
    background_path = join('extracted', 'raw', source_path)

    label_dict = {}
    frequency_dict = {}
    with open(label_path) as labels:
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

    skip_counter = 0
    for filename, label_list in label_dict.iteritems():
        bbox_list = np.array(label_list)
        if len(bbox_list) > 0:
            bbox_list_invert = bbox_list.copy()
            bbox_list_invert[:, 0] = 1.0 - bbox_list_invert[:, 0]

            bbox_list_ = transform_image_to_residual_space(bbox_list)
            bbox_list_invert_ = transform_image_to_residual_space(bbox_list_invert)

            distance_list = cdist(RESIDUALS_, bbox_list_, metric='euclidean')
            selection_list = np.argmin(distance_list, axis=1)

            distance_list_invert = cdist(RESIDUALS_, bbox_list_invert_, metric='euclidean')
            selection_list_invert = np.argmin(distance_list_invert, axis=1)

            indices = range(len(bbox_list_))
            skip = False
            for i in indices:
                selection = np.where(selection_list == i)[0]
                selection_invert = np.where(selection_list_invert == i)[0]
                if len(selection) == 0 or len(selection_invert) == 0:
                    skip = True
                    break
            if skip:
                print('\tSkipped: %r' % (filename, ))
                skip_counter += 1
                label_dict[filename] = []
    print('Skipped: %d' % (skip_counter, ))

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
        if len(label) == 0:
            continue
        filepath = join(background_path, filename)
        data = cv2.imread(filepath)
        data_list.append(data)
        label_list.append(label)

    data_list = np.array(data_list, dtype=np.uint8)
    label_list = np.array(label_list)

    np.save(cache_data, data_list)
    np.save(cache_labels, label_list)

    print('\nGround-Truth Distributions:')
    for key in sorted(frequency_dict.keys()):
        value = frequency_dict[key]
        print('\t{0: >2}: {1: <2}'.format(key, value))

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
        data = cv2.resize(original, CHIP_SIZE)
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
        train_learning_rate=0.1,
        train_batch_size=64,
        train_max_epochs=100,
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

    values, batch_size = network_output_to_image_space(prediction_list)
    prediction_bbox_list, prediction_alpha_list = values

    for batch in range(batch_size):
        prediction_bbox_ = prediction_bbox_list[batch]
        prediction_alpha_ = prediction_alpha_list[batch]
        original = original_list[batch]
        original = resample(original, height=500)

        height, width = original.shape[:2]

        # Show predictions
        counter = 0
        for index in range(PROPOSALS):
            (xc, yc, xr, yr) = prediction_bbox_[index]
            alpha = prediction_alpha_[index]
            xtl_ = int((xc - xr) * width)
            ytl_ = int((yc - yr) * height)
            xbr_ = int((xc + xr) * width)
            ybr_ = int((yc + yr) * height)

            xtl_ = min(width,  max(0, xtl_))
            ytl_ = min(height, max(0, ytl_))
            xbr_ = min(width,  max(0, xbr_))
            ybr_ = min(height, max(0, ybr_))

            if alpha >= 0.90:
                counter += 1
                cv2.rectangle(original, (xtl_, ytl_), (xbr_, ybr_), (0, 255, 0), 2)
            else:
                cv2.rectangle(original, (xtl_, ytl_), (xbr_, ybr_), (0, 0, 255), 2)

        output_filepath = join(test_path, 'output_%d.png' % (batch, ))
        cv2.imwrite(output_filepath, original)


if __name__ == '__main__':
    # Save the clusters
    center_filepath = join('extracted', 'residuals.transformed.npy')
    RESIDUALS = np.load(center_filepath)
    assert RESIDUALS.shape[0] == PROPOSALS

    # Reshape centroids
    RESIDUALS = RESIDUALS.astype(np.float32)
    RESIDUALS_ = RESIDUALS.copy()
    RESIDUALS = RESIDUALS.reshape((1, -1))

    # The output path to store all results and models
    output_path = '.'

    # Train network on Saliency training data
    train_saliency(output_path)

    # Test trained Saliency model on Saliency test data
    test_saliency(output_path)
