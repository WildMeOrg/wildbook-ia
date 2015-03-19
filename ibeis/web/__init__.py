# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[web]', DEBUG=False)

from ibeis.web import app
from ibeis.web import appfuncs
