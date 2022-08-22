# -*- coding: utf-8 -*-
"""
Auth module
===========
"""
from wbia.web.extensions.api import api_v2


def init_app(app, **kwargs):
    # pylint: disable=unused-argument
    """
    Init auth module.
    """
    # Register OAuth scopes
    api_v2.add_oauth_scope('auth:read', 'Provide access to auth details')
    api_v2.add_oauth_scope('auth:write', 'Provide write access to auth details')

    # Touch underlying modules
    from . import models, resources, views  # pylint: disable=unused-import  # NOQA

    # Mount authentication routes
    api_v2.add_namespace(resources.api)
