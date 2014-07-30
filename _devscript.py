from __future__ import absolute_import, division, print_function
import utool
from utool._internal.meta_util_six import get_funcname

# A list of registered development test functions
DEVCMD_FUNCTIONS = []


def devcmd(*args):
    """ Decorator which registers a function as a developer command """
    global DEVCMD_FUNCTIONS
    if len(args) == 1 and not isinstance(args[0], str):
        func = args[0]
        DEVCMD_FUNCTIONS.append(((get_funcname(func),), func))
        return utool.indent_func(func)
        #return func
    else:
        def closure_devcmd(func):
            func_aliases = [get_funcname(func)]
            func_aliases.extend(args)
            DEVCMD_FUNCTIONS.append((tuple(func_aliases), func))
            return utool.indent_func(func)
            #return func
        return closure_devcmd
