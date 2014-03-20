from __future__ import division, print_function
# Python
import argparse


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

    def add_strlist(self, switch, *args, **kwargs):
        self.add_meta(switch, str,  *args, nargs='*', **kwargs)

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
