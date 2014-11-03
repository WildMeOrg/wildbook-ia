"""
IBEIS: main package init

"""
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
import sys


#if '--loadall' in sys.argv:
#    from ibeis import ibsfuncs
#    from ibeis import constants
#    from ibeis import main_module
#    from ibeis import params
#    from ibeis import control
#    from ibeis import dev
#    from ibeis import io
#    from ibeis import model

try:
    # relative from ibeis imports
    #from ibeis import constants
    import ibeis.constants as constants
    import ibeis.params as params
    import ibeis.main_module as main_module
    import ibeis.control as control
    import ibeis.ibsfuncs as ibsfunc
    import ibeis.dev as dev
    import ibeis.io as io
    import ibeis.model as model
    #from ibeis import params
    #from ibeis import main_module
    #from ibeis import control
    #from ibeis import ibsfuncs
    #from ibeis import dev
    #from ibeis import io
    #from ibeis import preproc
except ImportError as ex:
    utool.printex(ex, 'ERROR in ibeis\' __init__', iswarning=False, tb=True)
    raise
    #try:
    #    pass
    #    #from . import constants
    #    #from . import main_module
    #    #from . import params
    #    #from . import control
    #    #from . import dev
    #    #from . import io
    #    #from . import dev
    #    #from . import ibsfuncs
    #    #from . import model
    #    #from . import web
    #except ImportError as ex:
    #    pass
    #else:
    #    utool.printex(ex, 'ERROR in ibeis\' __init__', tb=True)
    #    raise


from .dev import sysres
from .main_module import main, _preload, main_loop, test_main, opendb
from .control.IBEISControl import IBEISController
from .dev.sysres import get_workdir, set_workdir, ensure_pz_mtest

__LOADED__ = False
__version__ = '0.1.0.dev1'

def import_subs():
    # Weird / Fancy loading.
    # I want to make this simpler
    global __LOADED__
    from . import model
    from . import viz
    from . import web
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
    #getattr(params, 'rrr', lambda: None)()
    #getattr(control, 'reload_subs', lambda: None)()
    getattr(dev, 'reload_subs', lambda: None)()
    getattr(io, 'reload_subs', lambda: None)()
    #getattr(gui, 'reload_subs', lambda: None)()
    #getattr(ingest, 'reload_subs', lambda: None)()
    getattr(model, 'reload_subs', lambda: None)()
    #getattr(tests, 'reload_subs', lambda: None)()
    getattr(viz, 'reload_subs', lambda: None)()
    rrr()
rrrr = reload_subs
