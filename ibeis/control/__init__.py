# -*- coding: utf-8 -*-
### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function

import utool as ut
ut.noinject(__name__, '[ibeis.control.__init__]', DEBUG=False)


from ibeis.control import DB_SCHEMA
from ibeis.control import IBEISControl
from ibeis.control import _sql_helpers
from ibeis.control import accessor_decors
import utool
print, rrr, profile = utool.inject2(__name__)


def reload_subs(verbose=True):
    """ Reloads ibeis.control and submodules """
    rrr(verbose=verbose)
    getattr(DB_SCHEMA, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(IBEISControl, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(_sql_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(accessor_decors, 'rrr', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)
rrrr = reload_subs

IMPORT_TUPLES = [
    ('DB_SCHEMA', None, False),
    ('IBEISControl', None, False),
    ('_sql_helpers', None, False),
    ('accessor_decors', None, False),
]
"""
Regen Command:
    cd /home/joncrall/code/ibeis/ibeis/control
    makeinit.py -x DBCACHE_SCHEMA_CURRENT DB_SCHEMA_CURRENT _grave_template manual_ibeiscontrol_funcs template_definitions templates _autogen_ibeiscontrol_funcs
"""
