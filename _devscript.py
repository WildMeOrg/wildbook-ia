from __future__ import absolute_import, division, print_function
import utool

# A list of registered development test functions
DEVCMD_FUNCTIONS = []


def devcmd(*args):
    """ Decorator which registers a function as a developer command """
    global DEVCMD_FUNCTIONS
    if len(args) == 1 and not isinstance(args[0], str):
        func = args[0]
        DEVCMD_FUNCTIONS.append(((func.func_name,), func))
        return utool.indent_func(func)
        #return func
    else:
        def closure_devcmd(func):
            func_aliases = [func.func_name]
            func_aliases.extend(args)
            DEVCMD_FUNCTIONS.append((tuple(func_aliases), func))
            return utool.indent_func(func)
            #return func
        return closure_devcmd
