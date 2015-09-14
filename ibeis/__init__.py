"""
IBEIS: main package init

TODO: LAZY IMPORTS?
    http://code.activestate.com/recipes/473888-lazy-module-imports/
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import sys

#__version__ = '0.1.0.dev1'
#__version__ = '1.4.3'

utool.noinject(__name__, '[ibeis.__init__]', DEBUG=False)


if ut.is_developer():
    standard_visualization_functions = [
        'show_image',
        'show_chip',
        'show_chipmatch',
        'show_chipmatches',
        'show_vocabulary',
        #'show_vocabulary',
    ]


# If we dont initialize plottool before <something>
# then it causes a crash in windows. Its so freaking weird.
# something is not guitool, ibeis.viz
# has to be before control, can be after constants, params, and main_module
#import plottool


from ibeis import constants
from ibeis import constants as const
from ibeis import params
from ibeis import main_module
from ibeis import species
from ibeis import other
from ibeis.init import sysres
#main_module._preload()

from ibeis import control
from ibeis import ibsfuncs
from ibeis import dbio
#from ibeis import web

from ibeis.init import sysres
from ibeis.main_module import (main, _preload, _init_numpy, main_loop,
                               test_main, opendb, opendb_in_background)
from ibeis.control.IBEISControl import IBEISController
from ibeis.init.sysres import get_workdir, set_workdir, ensure_pz_mtest, ensure_nauts
from ibeis.init import main_helpers

from ibeis import model


from ibeis import experiments

def import_subs():
    # Weird / Fancy loading.
    # I want to make this simpler
    from ibeis import model
    from ibeis import viz
    from ibeis import web
    from ibeis import gui
    from ibeis import templates


def run_experiment(e='print', db='PZ_MTEST', a=['unctrl'], t=['default'], **kwargs):
    """
    Convience function
    """
    import functools
    def find_expt_func(e):
        import ibeis.dev
        for tup in ibeis.dev.REGISTERED_DOCTEST_EXPERIMENTS:
            modname, funcname = tup[:2]
            aliases = tup[2] if len(tup) == 3 else []
            if e == funcname or e in aliases:
                module = ut.import_modname(modname)
                func = module.__dict__[funcname]
                return func

    # Equivalent command line version of this func
    command_parts = ['ibeis',
                     '-e', e,
                     '--db', db,
                     '-a', ' '.join(a),
                     '-t', ' '.join(t),
                    ]
    if 'f' in kwargs:
        command_parts.extend(['-f', ' '.join(kwargs['f'])])
    if 'test_cfgx_slice' in kwargs:
        # very hacky, much more than checking for f
        slice_ = kwargs['test_cfgx_slice']
        slicestr = ':'.join(map(str, ut.replace_nones([getattr(slice_, attr, '') for attr in ['start', 'stop', 'step']], '')))
        command_parts.extend(['--test_cfgx_slice', slicestr])

    command_parts.extend(['--show'])

    command_line_str = ' '.join(command_parts)
    print('Equivalent Command Line:')
    print(command_line_str)

    func = find_expt_func(e)
    assert func is not None, 'unknown experiment e=%r' % (e,)

    # most experiments need a test_result
    ibs, test_result = main_helpers.testdata_expts(db, a=a, t=t)

    draw_func = functools.partial(func, ibs, test_result, **kwargs)
    test_result.draw_func = draw_func
    return test_result

    #ibeis.dev.run_registered_precmd(e)

def testdata_expts(*args, **kwargs):
    ibs, test_result = main_helpers.testdata_expts(*args, **kwargs)
    return test_result

#import_subs()
#from ibeis import gui
#from ibeis import model
#from ibeis import templates
#from ibeis import viz
#from ibeis import web


#class _VizProxy(object):
#    def __init__(self):
#        pass

#    def getattr(self, key):
#        import ibeis.viz as viz
#        return getattr(viz, key)

#    def setattr(self, key, val):
#        import ibeis.viz as viz
#        return getattr(viz, key, val)


#viz = _VizProxy
#import apipkg
#apipkg.initpkg(__name__, {
#    'viz': {
#        'clone': "ibeis.viz",
#    }
#}
#)

from ibeis.init.main_helpers import testdata_qres

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
    getattr(other, 'reload_subs', lambda verbose: None)(verbose=verbose)
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


from ibeis.control.DB_SCHEMA_CURRENT import VERSION_CURRENT
__version__ = VERSION_CURRENT
__version__ = '1.4.4'

if __version__ != VERSION_CURRENT:
    raise AssertionError('need to update version in __init__ file so setup.py can work nicely')

"""
Regen Command:
    Kinda have to work with the output of these. This module is hard to
    autogenerate correctly.

    cd /home/joncrall/code/ibeis/ibeis/other
    makeinit.py -x web viz tests gui all_imports
    makeinit.py -x constants params main_module other control ibsfuncs dbio tests all_imports
"""
