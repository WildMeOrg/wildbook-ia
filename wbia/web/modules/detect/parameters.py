# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-order
"""
Input arguments (Parameters) for User resources RESTful API
-----------------------------------------------------------
"""
import logging

from flask_marshmallow import base_fields

from flask_restx_patched import Parameters

log = logging.getLogger(__name__)


class RunDetectionParameters(Parameters):
    """
    New user creation (sign up) parameters.
    """

    guids = base_fields.List(base_fields.UUID, required=True)

    model = base_fields.String(description='Pre-trained model name', required=True)
    threshold = base_fields.Integer(
        description='Detection score threshold, between 0 and 100',
        required=False,
        default=50,
    )

    nms_mode = base_fields.String(
        description='(Non-Maximum Suppression) NMS mode',
        required=False,
        default='enabled',
    )
    nms_threshold = base_fields.Integer(
        description='NMS threshold, between 0 and 100', required=False, default=40
    )

    job_lane = base_fields.String(description='Job lane', required=False, default='fast')
    job_guid = base_fields.UUID(description='Job GUID', required=False, default=None)

    callback_url = base_fields.String(description='Callback URL', required=False)
    callback_method = base_fields.String(
        description='Callback HTTP method', required=False, default='POST'
    )
    callback_detailed = base_fields.String(
        description='Return a detailed callback with results ',
        required=False,
        default=False,
    )
