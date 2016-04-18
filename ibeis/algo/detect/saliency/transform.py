#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from jpcnn.tpl._theano import T
import numpy as np
from config import PROPOSALS, C, EPSILON, APPLY_PRIOR_TRANSFORMATION


def _transform_space(vector, transformer):
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


def transform_image_space_to_prior_space(vector):
    def _transformer(list):
        return C / (EPSILON + list)
    return _transform_space(vector, _transformer)


def transform_prior_space_to_image_space(vector):
    def _transformer(list):
        return (C / list) - EPSILON
    return _transform_space(vector, _transformer)


def transform_network_output_to_prior_bbox_conf(prediction_list):
    batch_size = prediction_list.shape[0]
    marker = PROPOSALS * 4
    prediction_bbox_list = prediction_list[:, :marker]
    prediction_alpha_list = prediction_list[:, marker:]
    new_shape = (batch_size, PROPOSALS, -1)
    prediction_bbox_list = prediction_bbox_list.reshape(new_shape)
    return (prediction_bbox_list, prediction_alpha_list), batch_size


def transform_network_output_to_image_bbox_conf(prediction_list):
    values = transform_network_output_to_prior_bbox_conf(prediction_list)
    (prediction_bbox_list, prediction_alpha_list), batch_size = values
    if APPLY_PRIOR_TRANSFORMATION:
        prediction_bbox_list = transform_prior_space_to_image_space(
            prediction_bbox_list
        )
    return (prediction_bbox_list, prediction_alpha_list), batch_size
