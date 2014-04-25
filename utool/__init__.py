"""
UTool
   Your friendly neighborhood utility tools

TODO: INSERT APACHE LICENCE
"""
# disable syntax checking
# flake8: noqa

# We hope to support python3
from __future__ import absolute_import, division, print_function
import time
#starttime = time.time()

__version__ = '(.878 + .478i)'
#if not get_flag('--strict', help='raises all exceptions'):

__NEW_WAY__ = True
__DEVELOPING__ = True

if __NEW_WAY__:
    UTOOLS_LIST = [
        ('util_alg',       ['cartesian', 'almost_eq',]),
        ('util_arg',       ['get_arg', 'get_flag', 'argv_flag_dec', 'QUIET',
                            'VERBOSE']),
        ('util_cache',     ['global_cache_read', 'global_cache_write']),
        ('util_cplat',     ['cmd', 'view_directory',]),
        ('util_csv',       None),
        ('util_dbg',       ['execstr_dict', 'save_testdata', 'load_testdata',
                            'get_caller_name', 'import_testdata', 'embed',
                            'quitflag', 'inIPython', 'printvar2', 'all_rrr']),
        ('util_dev',       ['printable_mystats', 'mystats', 'myprint',
                            'get_object_size']),
        ('util_decor',     ['ignores_exc_tb', 'indent_decor', 'indent_func',
                            'lru_cache', 'accepts_numpy',
                            'accepts_scalar_input_vector_output',
                            'accepts_scalar_input']),
        ('util_distances', ['nearest_point',]),
        ('util_dict',      None),
        ('util_hash',      ['hashstr_arr', 'hashstr',]),
        ('util_inject',    ['inject', 'inject_all', 'inject_print_functions']),
        ('util_iter',      ['iflatten', 'ichunks', 'interleave',]),
        ('util_list',      ['alloc_lists', 'list_index', 'npfind', 'index_of',
                            'flatten']),
        ('util_num',       None),
        ('util_path',      ['checkpath', 'ensuredir', 'assertpath', 'truepath',
                            'list_images']),
        ('util_print',     ['horiz_print', 'printshape', 'Indenter']),
        ('util_progress',  ['progress_func']),
        ('util_str',       ['byte_str2', 'horiz_string', 'theta_str']),
        ('util_sysreq',    None),
        ('util_regex',     ['regex_search']),
        ('util_time',      ['tic', 'toc', 'Timer']),
        ('util_type',      None),
        ('DynStruct',      ['DynStruct']),
        ('Preferences',    ['Pref']),
        ]

    import sys
    __self_module__ = sys.modules[__name__]

    IMPORTS      = [name for name, fromlist in UTOOLS_LIST]
    FROM_IMPORTS = [(name, fromlist) for name, fromlist in UTOOLS_LIST
                    if fromlist is not None and len(fromlist) > 0]

    # Actually imports
    for name in IMPORTS:
        tmp = __import__(name, globals(), locals(), fromlist=[], level=-1)

    for name, fromlist in FROM_IMPORTS:
        tmp = __import__(name, globals(), locals(), fromlist=fromlist, level=-1)
        for member in fromlist:
            setattr(__self_module__, member, getattr(tmp, member))


    # Effectively import * statements
    if __DEVELOPING__ and True:
        FROM_IMPORTS = []
        for name in IMPORTS:
            tmp = sys.modules[__name__ + '.' + name]
            varset = set(vars())
            fromlist_ = [member for member in dir(tmp) if not member.startswith('__') and not member in varset]
            for member in fromlist_:
                setattr(__self_module__, member, getattr(tmp, member))
            FROM_IMPORTS.append((name, fromlist_))

    # Print what this module should look like

    __PRINT_IMPORTS__ = False
    if __PRINT_IMPORTS__:
        import_str = '\n'.join(['import name=%r' % (name,) for name in IMPORTS])
        from_str   = '\n'.join(['from %s import (%s,)' % (name, ', '.join(fromlist))
                            for (name, fromlist) in FROM_IMPORTS])

        print(import_str)
        print(from_str)
else:
#mod_list = vars()
#print('\n'.join(mod_list.keys()))
    # Import each utility module
    from . import util_alg
    from . import util_arg
    from . import util_cache
    from . import util_cplat
    from . import util_csv
    from . import util_dbg
    from . import util_dev
    from . import util_decor
    from . import util_distances
    from . import util_dict
    from . import util_inject
    from . import util_iter
    from . import util_list
    from . import util_num
    from . import util_path
    from . import util_print
    from . import util_progress
    from . import util_str
    from . import util_sysreq
    from . import util_regex
    from . import util_time
    from . import util_type
    #Import common functions from each utilty module
    from .util_alg import (cartesian, almost_eq,)
    from .util_arg import (get_arg, get_flag, argv_flag_dec, QUIET, VERBOSE)
    from .util_cache import (global_cache_read, global_cache_write)
    from .util_cplat import (cmd, view_directory,)
    from .util_dbg import (execstr_dict, save_testdata, load_testdata,
                           get_caller_name, import_testdata, embed, quitflag,
                           inIPython, printvar2, all_rrr)
    from .util_decor import (ignores_exc_tb, indent_decor, indent_func,
                             lru_cache, accepts_numpy,
                             accepts_scalar_input_vector_output,
                             accepts_scalar_input)
    from .util_dev import (printable_mystats, mystats, myprint,
                           get_object_size,)
    from .util_distances import (nearest_point,)
    from .util_hash import (hashstr_arr, hashstr,)
    from .util_inject import (inject, inject_all, inject_print_functions)
    from .util_iter import (iflatten, ichunks, interleave,)
    from .util_list import (alloc_lists, list_index, npfind, index_of, flatten,
                            list_eq, listfind)
    from .util_path import (checkpath, ensuredir, assertpath, truepath,
                            list_images)
    from .util_print import (horiz_print, printshape, Indenter, NpPrintOpts,
                             printVERBOSE, printNOTQUIET)
    from .util_progress import (progress_func)
    from .util_str import (byte_str2, horiz_string, theta_str)
    from .util_regex import (regex_search)
    from .util_time import (tic, toc, Timer)
    from .DynStruct import DynStruct
    from .Preferences import Pref
    from . import Preferences


    # "Good" functions that are commonly used should be explicitly listed because
    # we are trying to develop quickly, we are being a little hacky
    if __DEVELOPING__:
        from .util_alg import *
        from .util_arg import *
        from .util_cache import *
        from .util_csv import *
        from .util_cplat import *
        from .util_dbg import *
        from .util_decor import *
        from .util_dev import *
        from .util_dict import *
        from .util_distances import *
        from .util_inject import *
        from .util_iter import *
        from .util_list import *
        from .util_num import *
        from .util_path import *
        from .util_print import *
        from .util_progress import *
        from .util_str import *
        from .util_sysreq import *
        from .util_regex import *
        from .util_time import *
        from .util_type import *

# Aliases
getflag = util_arg.get_flag

print, print_, printDBG, rrr, profile = inject(__name__, '[util]')

def reload_subs():
    rrr()
    util_alg.rrr()
    util_arg.rrr()
    util_cache.rrr()
    util_cplat.rrr()
    util_dbg.rrr()
    util_decor.rrr()
    util_dev.rrr()
    util_dict.rrr()
    util_distances.rrr()
    util_inject.rrr()
    util_iter.rrr()
    util_list.rrr()
    util_num.rrr()
    util_path.rrr()
    util_print.rrr()
    util_progress.rrr()
    util_str.rrr()
    util_regex.rrr()
    util_sysreq.rrr()
    util_time.rrr()
    util_type.rrr()


def rrrr():
    """Reloads self and submodules """
    reload_subs()

#print('imported utool in %r seconds' % (time.time() - starttime,))
