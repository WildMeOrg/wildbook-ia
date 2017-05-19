# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
(print, rrr, profile) = utool.inject2(__name__, '[web]')

from ibeis.web import apis_detect
from ibeis.web import apis_engine
from ibeis.web import apis_json
from ibeis.web import apis_sync
from ibeis.web import apis_query
from ibeis.web import apis
from ibeis.web import app
from ibeis.web import appfuncs
from ibeis.web import routes_ajax
from ibeis.web import routes_csv
from ibeis.web import routes_experiments
from ibeis.web import routes_submit
from ibeis.web import routes
