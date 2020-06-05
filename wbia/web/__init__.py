# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool

(print, rrr, profile) = utool.inject2(__name__, '[web]')

from wbia.web import apis_detect
from wbia.web import apis_engine
from wbia.web import apis_json
from wbia.web import apis_sync
from wbia.web import apis_query
from wbia.web import apis
from wbia.web import app
from wbia.web import appfuncs
from wbia.web import routes_ajax
from wbia.web import routes_demo
from wbia.web import routes_csv
from wbia.web import routes_experiments
from wbia.web import routes_submit
from wbia.web import routes


from wbia.control import controller_inject

if controller_inject.MICROSOFT_API_ENABLED:
    from wbia.web import apis_microsoft
