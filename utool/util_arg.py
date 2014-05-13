from __future__ import absolute_import, division, print_function
import sys
# Python
import argparse
from .util_type import try_cast
from .util_inject import inject
from .util_print import Indenter
print, print_, printDBG, rrr, profile = inject(__name__, '[arg]')

QUIET = '--quiet' in sys.argv
VERBOSE = '--verbose' in sys.argv


def get_arg(arg, type_=None, default=None, **kwargs):
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


def get_flag(arg, default=False, help_='', **kwargs):
    'Checks if the commandline has a flag or a corresponding noflag'
    if isinstance(arg, (tuple, list)):
        arg_list = arg
    else:
        arg_list = [arg]
    for arg in arg_list:
        if not (arg.find('--') == 0 or (arg.find('-') == 0 and len(arg) == 2)):
            raise AssertionError(arg)
        #if arg.find('--no') == 0:
            #arg = arg.replace('--no', '--')
        noarg = arg.replace('--', '--no')
        if arg in sys.argv:
            return True
        elif noarg in sys.argv:
            return False
    return default


def argv_flag(name, default, **kwargs):
    if name.find('--') == 0:
        name = name[2:]
    if '--' + name in sys.argv and default is False:
        return True
    if '--no' + name in sys.argv and default is True:
        return False
    return default


# ---- OnTheFly argparse ^^^^
# ---- Documented argparse VVVV


def switch_sanataize(switch):
    if isinstance(switch, str):
        dest = switch.strip('-').replace('-', '_')
    else:
        if isinstance(switch, tuple):
            switch = switch
        elif isinstance(switch, list):
            switch = tuple(switch)
        dest = switch[0].strip('-').replace('-', '_')
    return dest, switch


class ArgumentParser2(object):
    'Wrapper around argparse.ArgumentParser with convinence functions'
    def __init__(self, parser):
        self.parser = parser
        self._add_arg = parser.add_argument

    def add_arg(self, switch, *args, **kwargs):
        #print('[argparse2] add_arg(%r) ' % (switch,))
        if isinstance(switch, tuple):
            args = tuple(list(switch) + list(args))
            return self._add_arg(*args, **kwargs)
        else:
            return self._add_arg(switch, *args, **kwargs)

    def add_meta(self, switch, type, default=None, help='', **kwargs):
        #print('[argparse2] add_meta()')
        dest, switch = switch_sanataize(switch)
        self.add_arg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)

    def add_flag(self, switch, default=False, **kwargs):
        #print('[argparse2] add_flag()')
        action = 'store_false' if default else 'store_true'
        dest, switch = switch_sanataize(switch)
        self.add_arg(switch, dest=dest, action=action, default=default, **kwargs)

    def add_int(self, switch, *args, **kwargs):
        self.add_meta(switch, int,  *args, **kwargs)

    def add_intlist(self, switch, *args, **kwargs):
        self.add_meta(switch, int,  *args, nargs='*', **kwargs)

    add_ints = add_intlist

    def add_strlist(self, switch, *args, **kwargs):
        self.add_meta(switch, str,  *args, nargs='*', **kwargs)

    add_strs = add_strlist

    def add_float(self, switch, *args, **kwargs):
        self.add_meta(switch, float, *args, **kwargs)

    def add_str(self, switch, *args, **kwargs):
        self.add_meta(switch, str, *args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        return ArgumentParser2(self.parser.add_argument_group(*args, **kwargs))


def make_argparse2(description, *args, **kwargs):
    formatter_classes = [
        argparse.RawDescriptionHelpFormatter,
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter]
    return ArgumentParser2(
        argparse.ArgumentParser(prog='Program',
                                description=description,
                                prefix_chars='+-',
                                formatter_class=formatter_classes[2], *args,
                                **kwargs))


# Decorators which control program flow based on sys.argv
# the decorated function does not execute without its corresponding
# flag


def argv_flag_dec(func):
    return __argv_flag_dec(func, default=False)


def argv_flag_dec_true(func):
    return __argv_flag_dec(func, default=True)


def __argv_flag_dec(func, default=False, quiet=False):
    flag = func.func_name
    if flag.find('no') == 0:
        flag = flag[2:]
    flag = '--' + flag.replace('_', '-')

    def GaurdWrapper(*args, **kwargs):
        # FIXME: the --print-all is a hack
        if get_flag(flag, default) or get_flag('--print-all'):
            indent_lbl = flag.replace('--', '').replace('print-', '')
            print('')
            with Indenter('[%s]' % indent_lbl):
                return func(*args, **kwargs)
            print('')
        else:
            if not quiet:
                print('\n~~~ %s ~~~\n' % flag)
    GaurdWrapper.func_name = func.func_name
    return GaurdWrapper
