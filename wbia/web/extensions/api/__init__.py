# -*- coding: utf-8 -*-
"""
API extension
=============
"""

import logging

from flask import Blueprint, current_app  # NOQA

from wbia import __version__ as version

from .api import Api  # NOQA
from .http_exceptions import abort  # NOQA
from .namespace import Namespace  # NOQA

log = logging.getLogger(__name__)


AUTHORIZATIONS = {
    'oauth2_password': {
        'type': 'oauth2',
        'flow': 'password',
        'scopes': {},
        'tokenUrl': '/api/v1/auth/tokens',
    },
}


api_v2_blueprint = Blueprint('api', __name__, url_prefix='/api/v2')
api_v2 = Api(  # pylint: disable=invalid-name
    api_v2_blueprint,
    version='Version: {}'.format(version),
    title='Wild Me Sage',
    contact='info@wildme.org',
    # license='Apache License 2.0',
    # license_url='https://www.apache.org/licenses/LICENSE-2.0',
)


def init_app(app, **kwargs):
    # pylint: disable=unused-argument
    """
    API extension initialization point.
    """
    # Prevent config variable modification with runtime changes

    api_v2.authorizations = AUTHORIZATIONS
    app.register_blueprint(api_v2_blueprint)
