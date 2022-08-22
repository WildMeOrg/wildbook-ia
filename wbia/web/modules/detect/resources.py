# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""
RESTful API Detect resources
--------------------------
"""

import logging

from flask import current_app

from flask_restx_patched import Resource
from wbia.web.extensions.api import Namespace

from . import parameters

log = logging.getLogger(__name__)
api = Namespace('detect', description='Detect')


@api.route('/')
class Detection(Resource):
    """
    Manipulations with detect.
    """

    # @api.login_required(oauth_scopes=['detect:read'])
    @api.parameters(parameters.RunDetectionParameters())
    def get(self, args):
        """
        Run detection.
        """
        # Start detection job
        ibs = current_app.ibs

        print(args)

        # CONVERT INPUT PARAMETERS INTO NEEDED FOR LEGACY DETECTION CALL

        jobid = ibs.start_detect_image_lightnet()
        return jobid
