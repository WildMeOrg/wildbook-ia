"""
IBEIS: main package init

TODO: LAZY IMPORTS?
    http://code.activestate.com/recipes/473888-lazy-module-imports/
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
import sys

utool.noinject(__name__, '[ibeis.__init__]', DEBUG=False)


# If we dont initialize plottool before <something>
# then it causes a crash in windows. Its so freaking weird.
# something is not guitool, ibeis.viz
# has to be before control, can be after constants, params, and main_module
#import plottool


from ibeis import constants
from ibeis import constants as const
from ibeis import params
from ibeis import main_module
from ibeis import dev
from ibeis.dev import sysres
#main_module._preload()

from ibeis import control
from ibeis import ibsfuncs
from ibeis import dbio

#from ibeis import model
#from ibeis import viz
#from ibeis import web

from ibeis.dev import sysres
from ibeis.main_module import main, _preload, main_loop, test_main, opendb
from ibeis.control.IBEISControl import IBEISController
from ibeis.dev.sysres import get_workdir, set_workdir, ensure_pz_mtest, ensure_nauts

__version__ = '0.1.0.dev1'

def import_subs():
    # Weird / Fancy loading.
    # I want to make this simpler
    from ibeis import model
    from ibeis import viz
    from ibeis import web
    from ibeis import gui

# Utool generated init makeinit.py
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[ibeis]')

def reload_subs(verbose=True):
    """ Reloads ibeis and submodules """
    import_subs()
    rrr(verbose=verbose)
    getattr(constants, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(ibsfuncs, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(main_module, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(params, 'rrr', lambda verbose: None)(verbose=verbose)
    getattr(dev, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(dbio, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(model, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(control, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(viz, 'reload_subs', lambda: None)()

    getattr(gui, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(model, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(viz, 'reload_subs', lambda verbose: None)(verbose=verbose)
    getattr(web, 'reload_subs', lambda verbose: None)(verbose=verbose)

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
