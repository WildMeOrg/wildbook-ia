# -*- coding: utf-8 -*-
"""
Users module
============
"""

from wbia.web.extensions.api import api_v2


def init_app(app, **kwargs):
    # pylint: disable=unused-argument,unused-variable
    """
    Init users module.
    """
    api_v2.add_oauth_scope('users:read', 'Provide access to user details')
    api_v2.add_oauth_scope('users:write', 'Provide write access to user details')

    # Touch underlying modules
    from . import models, resources  # NOQA

    api_v2.add_namespace(resources.api)
