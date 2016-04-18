#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from jpcnn.core.model import JPCNN_Default_Model
from jpcnn.tpl._theano import T
from jpcnn.tpl import _lasagne
from utils import resample
from os.path import join
import numpy as np
import cv2
from assignment import assignment_solution
from config import (
    PROPOSALS,
    ALPHA,
    APPLY_PRIOR_CENTERING,
    APPLY_PRIOR_TRANSFORMATION,
    APPLY_DATA_AUGMENTATION,
    APPLY_NORMALIZATION,
    PRIORS,
    PRIORS_,
    MIRROR_LOSS_GRADIENT_WITH_ASSIGNMENT_ERROR,
)
from transform import (
    transform_image_space_to_prior_space,
    transform_network_output_to_image_bbox_conf,
    transform_network_output_to_prior_bbox_conf,
)


def clipped_linear(x):
    x = x + 0.5
    x = x.clip(0.0, 1.0)
    return x


class ResidualCenteringLayer(_lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return -1.0 * (input - PRIORS)


class ResidualTransformationLayer(_lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        shape_ = input.shape
        input = input.reshape((shape_[0], PROPOSALS, 4))
        input = transform_image_space_to_prior_space(input)
        input = input.reshape(shape_)
        return input


class Saliency_Model(JPCNN_Default_Model):
    def __init__(model, *args, **kwargs):
        super(Saliency_Model, model).__init__(*args, **kwargs)
        model.attached_data_whiten_mean = 118.380948
        model.attached_data_whiten_std = 61.896913

    def augmentation(model, X_list, y_list=None):
        if APPLY_DATA_AUGMENTATION:
            for index, y in enumerate(y_list):
                X = X_list[index].copy()
                # Adjust the exposure
                min_ = np.min(X)
                max_ = np.max(X)
                margin = np.min([min_, 255 - max_, 64])
                if margin > 0:
                    exposure = np.random.randint(-margin, margin)
                else:
                    exposure = 0
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

    def _assignment(model, yb_, prediction_bbox_, prediction_alpha_):
        if APPLY_PRIOR_TRANSFORMATION:
            yb_ = transform_image_space_to_prior_space(yb_)
        x = assignment_solution(
            prediction_bbox_,
            prediction_alpha_,
            yb_,
            alpha=ALPHA,
        )
        return x

    def _compute_accuracy(model, X_list, y_list, prediction_list, min_conf=0.50,
                          **kwargs):
        values = transform_network_output_to_image_bbox_conf(prediction_list)
        (prediction_bbox_list, prediction_alpha_list), batch_size = values

        alpha_list = []
        min_alpha = 1.0
        max_alpha = 0.0
        score_list = []
        min_score = 1.0
        max_score = 0.0

        output_path = 'confusion'

        assignments_dict = {}
        found_dict = {}
        for batch in range(batch_size):
            prediction_bbox_ = prediction_bbox_list[batch]
            prediction_alpha_ = prediction_alpha_list[batch]

            epoch = kwargs.get('epoch', None)
            status = kwargs.get('status', None)
            if batch < 5 and epoch is not None and status is not None:
                original = X_list[batch]
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

                output_filepath = join(output_path, 'output_%s_%s_%s.png' % (status, batch, epoch, ))
                cv2.imwrite(output_filepath, original)

            num_gt = len(y_list[batch])
            if num_gt > 0:
                y_list_ = np.array(y_list[batch])
                x = model._assignment(
                    y_list_,
                    prediction_bbox_,
                    prediction_alpha_
                )

                if x is not None:
                    indices = x[:, 0]
                    assignments = x[:, 1]

                    for picked_x in assignments:
                        if picked_x not in assignments_dict:
                            assignments_dict[picked_x] = 0
                        assignments_dict[picked_x] += 1

                    matched_bbox_list = prediction_bbox_[assignments]
                    matched_alpha_list = prediction_alpha_[assignments]
                    matched_y_list = y_list_[indices]

                    zipped = zip(
                        matched_bbox_list,
                        matched_alpha_list,
                        matched_y_list
                    )
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

        if len(alpha_list) == 0:
            avg_alpha = np.inf
        else:
            avg_alpha = sum(alpha_list) / len(alpha_list)

        if len(score_list) == 0:
            avg_score = np.inf
        else:
            avg_score = sum(score_list) / len(score_list)

        print('-' * 80)
        args = (min_alpha, max_alpha, avg_alpha, )
        print('alpha min: %0.08f max: %0.08f avg: %0.08f' % args)
        args = (min_score, max_score, avg_score, )
        print('score min: %0.08f max: %0.08f avg: %0.08f' % args)

        print('TOP Picked X Assignments:')
        item_list = list(assignments_dict.iteritems())
        item_list_rev = sorted(item_list, key=lambda tup: tup[1], reverse=True)
        zipped = zip(item_list, item_list_rev)
        for index, values in enumerate(zipped):
            ((picked_x, value), (picked_x_rev, value_rev)) = values
            if index > 8:
                continue
            fmt_str = '\t{0: >4}: {1: <4}\t{2: >4}: {3: <4}'
            print(fmt_str.format(picked_x, value, picked_x_rev, value_rev))

        print('TOP Found Assignments:')
        item_list = list(found_dict.iteritems())
        item_list_rev = sorted(item_list, key=lambda tup: tup[1], reverse=True)
        zipped = zip(item_list, item_list_rev)
        for index, values in enumerate(zipped):
            ((picked_x, value), (picked_x_rev, value_rev)) = values
            if index > 8:
                continue
            fmt_str = '\t{0: >4}: {1: <4}\t{2: >4}: {3: <4}'
            print(fmt_str.format(picked_x, value, picked_x_rev, value_rev))

        return avg_score

    def _fix_ground_truth(model, yb, prediction):
        new_yb = np.zeros(prediction.shape, dtype=prediction.dtype)

        marker = PROPOSALS * 4
        values = transform_network_output_to_prior_bbox_conf(prediction)
        (prediction_bbox, prediction_alpha), batch_size = values

        for batch in range(batch_size):
            prediction_bbox_ = prediction_bbox[batch]
            prediction_alpha_ = prediction_alpha[batch]
            temp_bbox = prediction_bbox_.copy()
            temp_alpha = prediction_alpha_.copy()

            num_gt = len(yb[batch])
            if num_gt > 0:
                yb_ = np.array(yb[batch])
                x = model._assignment(yb_, prediction_bbox_, prediction_alpha_)

                if x is not None:
                    indices = x[:, 0]
                    assignments = x[:, 1]

                    # Prepare to set default values for bboxes
                    output = yb_[indices]

                    if APPLY_PRIOR_CENTERING:
                        priors_ = np.take(PRIORS_, assignments, axis=0)
                        output = -1.0 * (output - priors_)

                    # Set default values for bbox
                    temp_bbox[assignments] = output

                    # Normalize bbox predictions
                    if APPLY_NORMALIZATION:
                        max_ = 1.0
                        min_ = 0.0
                        temp_bbox[:, 0][np.where(temp_bbox[:, 0] > max_)] = max_
                        temp_bbox[:, 1][np.where(temp_bbox[:, 1] > max_)] = max_
                        temp_bbox[:, 0][np.where(temp_bbox[:, 0] < min_)] = min_
                        temp_bbox[:, 1][np.where(temp_bbox[:, 1] < min_)] = min_
                        temp_bbox[:, 2][np.where(temp_bbox[:, 2] < min_)] = min_
                        temp_bbox[:, 3][np.where(temp_bbox[:, 3] < min_)] = min_

                    # # Set default values for alpha
                    # temp_alpha.fill(0.0)
                    # temp_alpha[assignments] = 1.0

            new_yb[batch, :marker] = np.reshape(temp_bbox, (-1, ))
            new_yb[batch, marker:] = np.reshape(temp_alpha, (-1, ))

        return new_yb

    def _loss_function(model, prediction, target):
        marker = PROPOSALS * 4

        prediction_bbox = prediction[:, :marker]
        prediction_alpha = prediction[:, marker:]

        target_bbox = target[:, :marker]
        target_alpha = target[:, marker:]

        loss_bbox = _lasagne.objectives.squared_error(
            prediction_bbox,
            target_bbox
        )
        loss_alpha = _lasagne.objectives.binary_crossentropy(
            prediction_alpha,
            target_alpha
        )

        if MIRROR_LOSS_GRADIENT_WITH_ASSIGNMENT_ERROR:
            loss_bbox *= 0.5
            # loss_bbox *= ALPHA

        return T.concatenate((loss_bbox, loss_alpha), axis=1)

    def get_loss_function(model):
        return model._loss_function

    def architecture(model, batch_size, in_width, in_height, in_channels,
                     out_classes):
        """
        """
        out_classes = PROPOSALS

        # _PretrinedNet = _lasagne.PretrainedNetwork('caffenet_full')
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
            W=_lasagne.init.Orthogonal(),
            # W=_PretrinedNet.get_pretrained_layer(0),
            # b=_PretrinedNet.get_pretrained_layer(1),
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
            W=_lasagne.init.Orthogonal(),
            # W=_PretrinedNet.get_pretrained_layer(2),
            # b=_PretrinedNet.get_pretrained_layer(3),
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
            W=_lasagne.init.Orthogonal(),
            # W=_PretrinedNet.get_pretrained_layer(4),
            # b=_PretrinedNet.get_pretrained_layer(5),
        )

        l_conv3 = _lasagne.Conv2DCCLayerGroup(
            l_conv2,
            num_filters=384,
            filter_size=(3, 3),
            stride=(1, 1),
            group=2,
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_lasagne.init.Orthogonal(),
            # W=_PretrinedNet.get_pretrained_layer(6),
            # b=_PretrinedNet.get_pretrained_layer(7),
        )

        l_conv4 = _lasagne.Conv2DCCLayerGroup(
            l_conv3,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            group=2,
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_lasagne.init.Orthogonal(),
            # W=_PretrinedNet.get_pretrained_layer(8),
            # b=_PretrinedNet.get_pretrained_layer(9),
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

        l_hidden2 = _lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=1024,
            nonlinearity=_lasagne.nonlinearities.rectify,
            W=_lasagne.init.Orthogonal(),
        )

        l_out_bbox_raw = _lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=out_classes * 4,
            # nonlinearity=_lasagne.nonlinearities.linear,
            nonlinearity=clipped_linear,
            W=_lasagne.init.Orthogonal(),
        )

        if APPLY_PRIOR_TRANSFORMATION:
            l_out_bbox_transformed = ResidualTransformationLayer(
                l_out_bbox_raw,
            )
        else:
            l_out_bbox_transformed = l_out_bbox_raw

        if APPLY_PRIOR_CENTERING:
            l_out_bbox_centered = ResidualCenteringLayer(
                l_out_bbox_transformed,
            )
            l_out_bbox = l_out_bbox_centered
        else:
            l_out_bbox = l_out_bbox_transformed

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
