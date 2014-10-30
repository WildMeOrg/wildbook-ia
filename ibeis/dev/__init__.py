# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool


print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[devel]', DEBUG=False)


__LOADED__ = False

def import_subs():
    global __LOADED__
    from . import dbinfo
    from . import main_commands
    from . import main_helpers
    from . import experiment_configs
    from . import experiment_harness
    __LOADED__ = True

import sys
if '--loadall' in sys.argv:
    from ibeis.dev import main_commands
    from ibeis.dev import main_helpers
    from ibeis.dev import experiment_configs
    from ibeis.dev import experiment_harness
    from ibeis.dev import dbinfo

def reload_subs():
    rrr()
    if not __LOADED__:
        import_subs()
    import_subs()
    dbinfo.rrr()
    main_commands.rrr()
    main_helpers.rrr()
    experiment_configs.rrr()
    experiment_harness.rrr()
