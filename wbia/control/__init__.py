# -*- coding: utf-8 -*-
### __init__.py ###
# flake8: noqa

import logging

import utool as ut

ut.noinject(__name__, '[wbia.control.__init__]', DEBUG=False)


import utool

from wbia.control import (
    DB_SCHEMA,
    IBEISControl,
    _sql_helpers,
    accessor_decors,
    controller_inject,
    docker_control,
)

print, rrr, profile = utool.inject2(__name__)
logger = logging.getLogger('wbia')


def reload_subs(verbose=True):
    """Reloads wbia.control and submodules"""
    rrr(verbose=verbose)
    getattr(DB_SCHEMA, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(IBEISControl, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(_sql_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(accessor_decors, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(controller_inject, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(docker_control, 'rrr', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)


rrrr = reload_subs

IMPORT_TUPLES = [
    ('DB_SCHEMA', None, False),
    ('IBEISControl', None, False),
    ('_sql_helpers', None, False),
    ('accessor_decors', None, False),
    ('controller_inject', None, False),
    ('docker_control', None, False),
]
"""
Regen Command:
    cd /home/joncrall/code/wbia/wbia/control
    makeinit.py -x DBCACHE_SCHEMA_CURRENT DB_SCHEMA_CURRENT _grave_template manual_wbiacontrol_funcs template_definitions templates _autogen_wbiacontrol_funcs
"""
