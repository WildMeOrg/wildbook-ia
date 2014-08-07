# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
from . import constants
from . import main_module
from .dev import params
from .dev import sysres
from .main_module import main, _preload, main_loop, test_main, opendb
from .control.IBEISControl import IBEISController
from .dev.sysres import get_workdir, set_workdir

__LOADED__ = False
__version__ = '0.1.0.dev1'

def import_subs():
    # Weird / Fancy loading.
    # I want to make this simpler
    global __LOADED__
    from . import dev
    from . import ibsfuncs
    from . import viz
    from . import control
    from . import model
    __LOADED__ = True

def ensure_subs():
    if not __LOADED__:
        import_subs

# Utool generated init makeinit.py
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[ibeis]')

def reload_subs():
    """ Reloads ibeis and submodules """
    if not __LOADED__:
        import_subs()
    rrr()
    getattr(constants, 'rrr', lambda: None)()
    getattr(ibsfuncs, 'rrr', lambda: None)()
    getattr(main_module, 'rrr', lambda: None)()
    #getattr(control, 'reload_subs', lambda: None)()
    getattr(dev, 'reload_subs', lambda: None)()
    getattr(export, 'reload_subs', lambda: None)()
    #getattr(gui, 'reload_subs', lambda: None)()
    #getattr(ingest, 'reload_subs', lambda: None)()
    getattr(model, 'reload_subs', lambda: None)()
    #getattr(tests, 'reload_subs', lambda: None)()
    getattr(viz, 'reload_subs', lambda: None)()
    rrr()
rrrr = reload_subs
