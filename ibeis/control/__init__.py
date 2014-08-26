from __future__ import absolute_import, division, print_function
from . import DB_SCHEMA
from . import IBEISControl
from . import SQLDatabaseControl
from . import _sql_helpers
from . import accessor_decors
import utool
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[control]')


def reload_subs():
    """ Reloads control and submodules """
    rrr()
    getattr(DB_SCHEMA, 'rrr', lambda: None)()
    getattr(IBEISControl, 'rrr', lambda: None)()
    getattr(SQLDatabaseControl, 'rrr', lambda: None)()
    getattr(_sql_helpers, 'rrr', lambda: None)()
    getattr(accessor_decors, 'rrr', lambda: None)()
    rrr()
rrrr = reload_subs
