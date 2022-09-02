# -*- coding: utf-8 -*-
# flake8: noqa
import logging

import utool

(print, rrr, profile) = utool.inject2(__name__, '[web]')
logger = logging.getLogger('wbia')

from wbia.control import controller_inject
from wbia.web import (
    apis,
    apis_detect,
    apis_engine,
    apis_json,
    apis_query,
    apis_scout,
    apis_sync,
    app,
    appfuncs,
    routes,
    routes_ajax,
    routes_csv,
    routes_demo,
    routes_experiments,
    routes_submit,
)

if controller_inject.MICROSOFT_API_ENABLED:
    from wbia.web import apis_microsoft
# if controller_inject.SCOUT_API_ENABLED:
#     from wbia.web import apis_scout
