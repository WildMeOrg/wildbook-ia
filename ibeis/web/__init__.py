# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
(print, rrr, profile) = utool.inject2(__name__, '[web]')

from ibeis.web import app
from ibeis.web import appfuncs
