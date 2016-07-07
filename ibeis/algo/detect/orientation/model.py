#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
try:
    from jpcnn.core.model import JPCNN_Default_Model
    from jpcnn.tpl._theano import T
    from jpcnn.tpl import _lasagne
except:
    JPCNN_Default_Model = object
    pass


def clipped_linear(x):
    x = x.clip(0.0, 1.0)
    return x


def modded_linear(x):
    x = x % 1.0
    return x


class Orientation_Model(JPCNN_Default_Model):
    def __init__(model, *args, **kwargs):
        super(Orientation_Model, model).__init__(*args, **kwargs)

    def augmentation(model, X_list, y_list=None, train=False):
        X_list_ = []
        y_list_ = []
        for index, y in enumerate(y_list):
            X = X_list[index].copy()
            y = y_list[index]

            X = X.astype(np.float32)
            X = cv2.resize(X, (182, 182))

            if train:
                # Adjust the exposure
                min_ = np.min(X)
                max_ = np.max(X)
                margin = np.min([min_, 255 - max_, 64])
                if margin > 0:
                    exposure = np.random.uniform(-margin, margin)
                else:
                    exposure = 0.0
                X += exposure
                # Horizontal flip
                hori_flip = np.random.uniform() <= 0.5
                if hori_flip:
                    X = cv2.flip(X, 1)
                    y = (0.0 - y) % 1.0
                # Vertical flip
                vert_flip = np.random.uniform() <= 0.5
                if vert_flip:
                    X = cv2.flip(X, 0)
                    y = (0.5 + y) % 1.0
                # Rotate
                angle = np.random.randint(0, 359)
                h, w = X.shape[0:2]
                center = (h // 2, w // 2)
                A = cv2.getRotationMatrix2D(center, angle, 1.0)
                X = cv2.warpAffine(X, A, X.shape[:2], flags=cv2.INTER_LINEAR)
                y += angle / 360.0

            # Reshape
            # X_ = X.reshape(X_list[index].shape)
            X = np.around(X)
            X = X.astype(np.uint8)
            X_ = X[27:-27, 27:-27, :]
            y_ = y % 1.0
            # Save
            X_list_.append(X_)
            y_list_.append(y_)

            # if train:
            #     cv2.imshow('%s' % (y_, ), X_)
            #     cv2.waitKey(0)

        X_list_ = np.array(X_list_, dtype=X_list.dtype)
        y_list_ = np.array(y_list_, dtype=y_list.dtype)
        y_list_ = y_list_.reshape((-1, 1))
        return X_list_, y_list_

    def _fix_ground_truth(model, yb):
        yb = yb.reshape((-1, 1))
        return yb

    def _compute_accuracy(model, X_list, y_list, prediction_list, margin=15.0,
                          **kwargs):
        margin = margin / 360.0
        correct = 0.0
        total = len(y_list)
        zipped = zip(y_list, prediction_list)
        for index, (y, prediction) in enumerate(zipped):
            inner_dist = abs(prediction - y)
            outer_dist = 1.0 - inner_dist
            dist = min(inner_dist, outer_dist)
            if dist <= margin:
                correct += 1.0
        return correct / total

    def _loss_function(model, prediction, target):
        inner_loss = T.abs_(target - prediction)
        outer_loss = 1.0 - inner_loss
        combined_loss = T.stack([inner_loss, outer_loss], axis=1)
        return T.min(combined_loss, axis=1)

    def get_loss_function(model):
        return model._loss_function
        # return _lasagne.objectives.squared_error

    def architecture(model, batch_size, in_width, in_height, in_channels,
                     out_classes):
        """
        """

        _PretrainedNet = _lasagne.PretrainedNetwork('vggnet_full')

        l_in = _lasagne.layers.InputLayer(
            # shape=(None, in_channels, in_width, in_height)
            shape=(None, 3, 128, 128)
        )

        l_conv0 = _lasagne.Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(0),
            b=_PretrainedNet.get_pretrained_layer(1),
        )

        l_conv1 = _lasagne.Conv2DLayer(
            l_conv0,
            num_filters=32,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(2),
            b=_PretrainedNet.get_pretrained_layer(3),
        )

        l_pool1 = _lasagne.MaxPool2DLayer(
            l_conv1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv2 = _lasagne.Conv2DLayer(
            l_pool1,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(4),
            b=_PretrainedNet.get_pretrained_layer(5),
        )

        l_conv3 = _lasagne.Conv2DLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(6),
            b=_PretrainedNet.get_pretrained_layer(7),
        )

        l_pool3 = _lasagne.MaxPool2DLayer(
            l_conv3,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv4 = _lasagne.Conv2DLayer(
            l_pool3,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(8),
            b=_PretrainedNet.get_pretrained_layer(9),
        )

        l_conv5 = _lasagne.Conv2DLayer(
            l_conv4,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(10),
            b=_PretrainedNet.get_pretrained_layer(11),
        )

        l_conv6 = _lasagne.Conv2DLayer(
            l_conv5,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(12),
            b=_PretrainedNet.get_pretrained_layer(13),
        )

        l_conv7 = _lasagne.Conv2DLayer(
            l_conv6,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(14),
            b=_PretrainedNet.get_pretrained_layer(15),
        )

        l_pool7 = _lasagne.MaxPool2DLayer(
            l_conv7,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv8 = _lasagne.Conv2DLayer(
            l_pool7,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(16),
            b=_PretrainedNet.get_pretrained_layer(17),
        )

        l_conv9 = _lasagne.Conv2DLayer(
            l_conv8,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(18),
            b=_PretrainedNet.get_pretrained_layer(19),
        )

        l_conv10 = _lasagne.Conv2DLayer(
            l_conv9,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(20),
            b=_PretrainedNet.get_pretrained_layer(21),
        )

        l_conv11 = _lasagne.Conv2DLayer(
            l_conv10,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(22),
            b=_PretrainedNet.get_pretrained_layer(23),
        )

        l_pool12 = _lasagne.MaxPool2DLayer(
            l_conv11,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_hidden1 = _lasagne.layers.DenseLayer(
            l_pool12,
            num_units=256,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_hidden2 = _lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=256,
            nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_out = _lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=1,
            # nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.sigmoid,
            # nonlinearity=clipped_linear,
            nonlinearity=modded_linear,
            # W=_lasagne.init.Orthogonal(1.0),
        )

        return l_out
