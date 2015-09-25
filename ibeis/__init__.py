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


def run_experiment(e='print', db='PZ_MTEST', a=['unctrl'], t=['default'],
                   qaid_override=None, lazy=False,
                   **kwargs):
    """
    Convience function

    CommandLine:
        ibeis -e print
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

    def build_commandline(e=e, **kwargs):
        # Equivalent command line version of this func
        command_parts = ['ibeis',
                         '-e', e,
                         '--db', db,
                         '-a', ' '.join(a),
                         '-t', ' '.join(t),
                        ]
        if qaid_override is not None:
            command_parts.extend(['--qaid=', ','.join(map(str, qaid_override))])

        # hack parse out important args that were on command line
        if 'f' in kwargs:
            command_parts.extend(['-f', ' '.join(kwargs['f'])])
        if 'test_cfgx_slice' in kwargs:
            # very hacky, much more than checking for f
            slice_ = kwargs['test_cfgx_slice']
            slice_attrs = [getattr(slice_, attr, '')
                           for attr in ['start', 'stop', 'step']]
            slice_attrs = ut.replace_nones(slice_attrs, '')
            slicestr = ':'.join(map(str, slice_attrs))
            command_parts.extend(['--test_cfgx_slice', slicestr])

        command_parts.extend(['--show'])

        command_line_str = ' '.join(command_parts)
        # Warning, not always equivalent
        print('Equivalent Command Line:')
        print(command_line_str)
        return command_line_str
    command_line_str = build_commandline()


    def draw_cases(test_result, **kwargs):
        e_ = 'draw_cases'
        func = find_expt_func(e_)
        ibs = test_result.ibs
        build_commandline(e=e_)
        lazy_func = functools.partial(func, ibs, test_result, show_in_notebook=True, **kwargs)
        return lazy_func

    def draw_taghist(test_result, **kwargs):
        e_ = 'taghist'
        func = find_expt_func(e_)
        ibs = test_result.ibs
        build_commandline(e=e_)
        lazy_func = functools.partial(func, ibs, test_result, **kwargs)
        return lazy_func

    def execute_test():
        func = find_expt_func(e)
        assert func is not None, 'unknown experiment e=%r' % (e,)

        argspec = ut.get_func_argspec(func)
        if len(argspec.args) >= 2 and argspec.args[0] == 'ibs' and argspec.args[1] == 'test_result':
            # most experiments need a test_result
            expts_kw = dict(defaultdb=db, a=a, t=t, qaid_override=qaid_override)
            testdata_expts_func = functools.partial(main_helpers.testdata_expts, **expts_kw)

            ibs, test_result = testdata_expts_func()
            # Build the requested drawing funciton
            draw_func = functools.partial(func, ibs, test_result, **kwargs)
            test_result.draw_func = draw_func
            ut.inject_func_as_method(test_result, draw_cases)
            ut.inject_func_as_method(test_result, draw_taghist)
            #test_result.draw_cases = draw_cases
            return test_result
        else:
            raise AssertionError('Unknown type of function for experiment')

    if lazy:
        return execute_test
    else:
        test_result = execute_test()
        return test_result


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
