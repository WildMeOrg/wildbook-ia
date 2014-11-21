# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool


print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[devel]', DEBUG=False)


__LOADED__ = False

def import_subs():
    global __LOADED__
    #from . import dbinfo
    #from . import main_commands
    #from . import main_helpers
    #from . import experiment_configs
    #from . import experiment_harness
    from ibeis.dev import dbinfo
    from ibeis.dev import duct_tape
    from ibeis.dev import experiment_configs
    from ibeis.dev import experiment_harness
    from ibeis.dev import experiment_helpers
    from ibeis.dev import experiment_printres
    from ibeis.dev import main_commands
    from ibeis.dev import main_helpers
    from ibeis.dev import results_all
    from ibeis.dev import results_analyzer
    from ibeis.dev import results_organizer
    from ibeis.dev import sysres
    __LOADED__ = True

import sys

def reload_subs(verbose=True):
    """ Reloads ibeis.dev and submodules """
    rrr(verbose=verbose)
    if not __LOADED__:
        import_subs()
    getattr(dbinfo, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(duct_tape, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(experiment_configs, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(experiment_harness, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(experiment_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(experiment_printres, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(main_commands, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(main_helpers, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(results_all, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(results_analyzer, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(results_organizer, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(sysres, 'rrr', lambda verbose: None)(verbose=verbose)
    rrr(verbose=verbose)
rrrr = reload_subs


"""
Regen Command:
    Kinda have to work with the output of these. This module is hard to
    autogenerate correctly.

    cd /home/joncrall/code/ibeis/ibeis/dev

    makeinit.py -x web viz tests gui all_imports
    makeinit.py -x constants params main_module dev control ibsfuncs dbio tests all_imports
"""
