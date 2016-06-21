#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import cv2
try:
    from jpcnn.core.model import JPCNN_Auto_Model
    from jpcnn.tpl import _lasagne, _theano
    from jpcnn.tpl._theano import T
except:
    JPCNN_Auto_Model = object
    pass

label_list = [
    'ignore',
    'zebra_plains:left',
    'zebra_plains:frontleft',
    'zebra_plains:front',
    'zebra_plains:frontright',
    'zebra_plains:right',
    'zebra_plains:backright',
    'zebra_plains:back',
    'zebra_plains:backleft',
    'zebra_grevys:left',
    'zebra_grevys:frontleft',
    'zebra_grevys:front',
    'zebra_grevys:frontright',
    'zebra_grevys:right',
    'zebra_grevys:backright',
    'zebra_grevys:back',
    'zebra_grevys:backleft',
]

label_mapping_dict = {
    'left'       : 'right',
    'frontleft'  : 'frontright',
    'front'      : 'front',
    'frontright' : 'frontleft',
    'right'      : 'left',
    'backright'  : 'backleft',
    'back'       : 'back',
    'backleft'   : 'backright',
}


def augmentation_parallel(values):
    X, y = values
    return augmentation_wrapper([X], [y])


def augmentation_wrapper(X_list, y_list):
    import random
    for index, y in enumerate(y_list):
        X = np.copy(X_list[index])
        # Adjust the exposure
        X_Lab = cv2.cvtColor(X, cv2.COLOR_BGR2LAB)
        X_L = X_Lab[:, :, 0].astype(dtype=np.float32)
        # margin = np.min([np.min(X_L), 255.0 - np.max(X_L), 64.0])
        margin = 128.0
        exposure = random.uniform(-margin, margin)
        X_L += exposure
        X_L = np.around(X_L)
        X_L[X_L < 0.0] = 0.0
        X_L[X_L > 255.0] = 255.0
        X_Lab[:, :, 0] = X_L.astype(dtype=X_Lab.dtype)
        X = cv2.cvtColor(X_Lab, cv2.COLOR_LAB2BGR)
        # Rotate and Scale
        h, w, c = X.shape
        degree = random.randint(-30, 30)
        scale = random.uniform(0.80, 1.25)
        padding = np.sqrt((w) ** 2 / 4 - 2 * (w) ** 2 / 16)
        padding /= scale
        padding = int(np.ceil(padding))
        for channel in range(c):
            X_ = X[:, :, channel]
            X_ = np.pad(X_, padding, 'reflect', reflect_type='even')
            h_, w_ = X_.shape
            # Calulate Affine transform
            center = (w_ // 2, h_ // 2)
            A = cv2.getRotationMatrix2D(center, degree, scale)
            X_ = cv2.warpAffine(X_, A, (w_, h_), flags=cv2.INTER_LANCZOS4, borderValue=0)
            X_ = X_[padding: -1 * padding, padding: -1 * padding]
            X[:, :, channel] = X_
        # Horizontal flip
        if random.uniform(0.0, 1.0) <= 0.5:
            X = cv2.flip(X, 1)
            if ':' in y:
                species, viewpoint = y.split(':')
                viewpoint = label_mapping_dict[viewpoint]
                y = '%s:%s' % (species, viewpoint)
        # Blur
        if random.uniform(0.0, 1.0) <= 0.1:
            if random.uniform(0.0, 1.0) <= 0.5:
                X = cv2.blur(X, (3, 3))
            else:
                X = cv2.blur(X, (5, 5))
        # Reshape
        X = X.reshape(X_list[index].shape)
        # Show image
        # canvas = np.hstack((X_list[index], X))
        # cv2.imshow('', canvas)
        # cv2.waitKey(0)
        # Save
        X_list[index] = X
        y_list[index] = y
    return X_list, y_list


class Labeler_Model(JPCNN_Auto_Model):
    def __init__(model, *args, **kwargs):
        super(Labeler_Model, model).__init__(*args, **kwargs)

    def augmentation(model, X_list, y_list=None, train=True, parallel=True):
        if not parallel:
            return augmentation_wrapper(X_list, y_list)
        # Run in paralell
        arg_iter = list(zip(X_list, y_list))
        result_list = ut.util_parallel.generate(augmentation_parallel, arg_iter,
                                                ordered=True, verbose=False,
                                                quiet=True)
        result_list = list(result_list)
        X = [ result[0][0] for result in result_list ]
        y = [ result[1] for result in result_list ]
        X = np.array(X)
        y = np.hstack(y)
        return X, y

    def _compute_accuracy(model, X_list, y_list, prediction_list, **kwargs):
        correct = 0.0
        total = len(y_list)
        zipped = zip(y_list, prediction_list)
        for index, (y, prediction) in enumerate(zipped):
            print(y, prediction)
            if y == prediction:
                correct += 1.0
        return correct / total

    def label_order_mapping(model, label_list):
        return { key: index for index, key in enumerate(label_list) }

    def _loss_function(model, prediction, target):
        loss = _theano.T.nnet.categorical_crossentropy(prediction, target)
        pred = T.argmax(prediction)
        targ = T.argmax(target)
        indices = T.and_(T.neq(pred, targ), T.eq(targ, 1.0))
        loss_ = loss * 5.0
        loss = T.where(indices, loss_, loss)
        return loss

    def get_loss_function(model):
        return model._loss_function

    def architecture(model, batch_size, in_width, in_height, in_channels,
                     out_classes):
        """
        """

        # _PretrainedNet = _lasagne.PretrainedNetwork('vggnet_full')
        _PretrainedNet = _lasagne.PretrainedNetwork('overfeat_full')

        l_in = _lasagne.layers.InputLayer(
            # shape=(None, in_channels, in_width, in_height)
            shape=(None, 3, 128, 128)
        )

        l_conv0 = _lasagne.Conv2DLayer(
            l_in,
            num_filters=64,
            filter_size=(11, 11),
            stride=(2, 2),
            pad=0 if _lasagne.USING_GPU else 9,
            nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(0),
            # b=_PretrainedNet.get_pretrained_layer(1),
        )

        l_batchnorm0 = _lasagne.layers.BatchNormLayer(
            l_conv0,
        )

        l_nonlinear0 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm0,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_conv1 = _lasagne.Conv2DLayer(
            l_nonlinear0,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1, 1),
            # pad=2,
            nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(2),
            # b=_PretrainedNet.get_pretrained_layer(3),
        )

        l_batchnorm1 = _lasagne.layers.BatchNormLayer(
            l_conv1,
        )

        l_nonlinear1 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm1,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_pool1 = _lasagne.MaxPool2DLayer(
            l_nonlinear1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_dropout1 = _lasagne.layers.DropoutLayer(
            l_pool1,
            p=0.1,
        )

        l_conv2 = _lasagne.Conv2DLayer(
            l_dropout1,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(4),
            # b=_PretrainedNet.get_pretrained_layer(5),
        )

        l_batchnorm2 = _lasagne.layers.BatchNormLayer(
            l_conv2,
        )

        l_nonlinear2 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm2,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_conv3 = _lasagne.Conv2DLayer(
            l_nonlinear2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
            W=_PretrainedNet.get_pretrained_layer(6),
            # b=_PretrainedNet.get_pretrained_layer(7),
        )

        l_batchnorm3 = _lasagne.layers.BatchNormLayer(
            l_conv3,
        )

        l_nonlinear3 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm3,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_pool3 = _lasagne.MaxPool2DLayer(
            l_nonlinear3,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_dropout3 = _lasagne.layers.DropoutLayer(
            l_pool3,
            p=0.2,
        )

        l_conv6 = _lasagne.Conv2DLayer(
            l_dropout3,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm6 = _lasagne.layers.BatchNormLayer(
            l_conv6,
        )

        l_nonlinear6 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm6,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_conv7 = _lasagne.Conv2DLayer(
            l_nonlinear6,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm7 = _lasagne.layers.BatchNormLayer(
            l_conv7,
        )

        l_nonlinear7 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm7,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_pool7 = _lasagne.MaxPool2DLayer(
            l_nonlinear7,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_dropout7 = _lasagne.layers.DropoutLayer(
            l_pool7,
            p=0.3,
        )

        l_conv12 = _lasagne.Conv2DLayer(
            l_dropout7,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm12 = _lasagne.layers.BatchNormLayer(
            l_conv12,
        )

        l_nonlinear12 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm12,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_conv13 = _lasagne.Conv2DLayer(
            l_nonlinear12,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm13 = _lasagne.layers.BatchNormLayer(
            l_conv13,
        )

        l_nonlinear13 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm13,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_conv14 = _lasagne.Conv2DLayer(
            l_nonlinear13,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad=1,
            nonlinearity=_lasagne.nonlinearities.linear,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm14 = _lasagne.layers.BatchNormLayer(
            l_conv14,
        )

        l_nonlinear14 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm14,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        # l_conv15 = _lasagne.Conv2DLayer(
        #     l_nonlinear14,
        #     num_filters=256,
        #     filter_size=(3, 3),
        #     stride=(1, 1),
        #     # pad=1,
        #     nonlinearity=_lasagne.nonlinearities.linear,
        #     W=_lasagne.init.Orthogonal('relu'),
        # )

        # l_batchnorm15 = _lasagne.layers.BatchNormLayer(
        #     l_conv15,
        # )

        # l_nonlinear15 = _lasagne.layers.NonlinearityLayer(
        #     l_batchnorm15,
        #     nonlinearity=_lasagne.nonlinearities.rectify,
        # )

        l_pool15 = _lasagne.MaxPool2DLayer(
            l_nonlinear14,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_dropout15 = _lasagne.layers.DropoutLayer(
            l_pool15,
            p=0.4,
        )

        l_hidden1 = _lasagne.layers.DenseLayer(
            l_dropout15,
            num_units=768,
            nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm12 = _lasagne.layers.BatchNormLayer(
            l_hidden1,
        )

        l_nonlinear12 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm12,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_maxout1 = _lasagne.layers.FeaturePoolLayer(
            l_nonlinear12,
            pool_size=2,
        )

        l_dropout = _lasagne.layers.DropoutLayer(
            l_maxout1,
            p=0.5,
        )

        l_hidden2 = _lasagne.layers.DenseLayer(
            l_dropout,
            num_units=768,
            nonlinearity=_lasagne.nonlinearities.linear,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            # W=_lasagne.init.Orthogonal('relu'),
        )

        l_batchnorm13 = _lasagne.layers.BatchNormLayer(
            l_hidden2,
        )

        l_nonlinear13 = _lasagne.layers.NonlinearityLayer(
            l_batchnorm13,
            # nonlinearity=_lasagne.nonlinearities.rectify,
            nonlinearity=_lasagne.nonlinearities.LeakyRectify(leakiness=0.1),
        )

        l_maxout2 = _lasagne.layers.FeaturePoolLayer(
            l_nonlinear13,
            pool_size=2,
        )

        l_dropout2 = _lasagne.layers.DropoutLayer(
            l_maxout2,
            p=0.5,
        )

        l_out = _lasagne.layers.DenseLayer(
            l_dropout2,
            num_units=out_classes,
            nonlinearity=_lasagne.nonlinearities.softmax,
            # W=_lasagne.init.Orthogonal(1.0),
        )

        return l_out
