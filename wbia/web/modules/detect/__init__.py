# -*- coding: utf-8 -*-
"""
Detect module
============
"""

from wbia.web.extensions.api import api_v2


def init_app(app, **kwargs):
    # pylint: disable=unused-argument,unused-variable
    """
    Init detect module.
    """
    api_v2.add_oauth_scope('detect:read', 'Provide access to detect details')
    api_v2.add_oauth_scope('detect:write', 'Provide write access to detect details')

    # Touch underlying modules
    from . import resources  # NOQA

    api_v2.add_namespace(resources.api)
