from __future__ import division, print_function
import sys
from .util_type import try_cast
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[arg]')


def get_arg(arg, type_=None, default=None):
    arg_after = default
    if type_ is bool:
        arg_after = False if default is None else default
    try:
        argx = sys.argv.index(arg)
        if argx < len(sys.argv):
            if type_ is bool:
                arg_after = True
            else:
                arg_after = try_cast(sys.argv[argx + 1], type_)
    except Exception:
        pass
    return arg_after


def get_flag(arg, default=False):
    'Checks if the commandline has a flag or a corresponding noflag'
    if arg.find('--') != 0:
        raise Exception(arg)
    #if arg.find('--no') == 0:
        #arg = arg.replace('--no', '--')
    noarg = arg.replace('--', '--no')
    if arg in sys.argv:
        return True
    elif noarg in sys.argv:
        return False
    else:
        return default
    return default


def argv_flag(name, default):
    if name.find('--') == 0:
        name = name[2:]
    if '--' + name in sys.argv and default is False:
        return True
    if '--no' + name in sys.argv and default is True:
        return False
    return default
