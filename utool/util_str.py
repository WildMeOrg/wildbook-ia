from __future__ import absolute_import, division, print_function
import sys
import textwrap
from itertools import imap
from os.path import split
import numpy as np
from .util_inject import inject
from .util_time import get_unix_timedelta
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


np.tau = (2 * np.pi)  # tauday.com


def theta_str(theta, taustr=('tau' if '--myway' in sys.argv else '2pi')):
    """ Format theta so it is interpretable in base 10 """
    #coeff = (((tau - theta) % tau) / tau)
    coeff = (theta / np.tau)
    return ('%.2f * ' % coeff) + taustr


def tupstr(tuple_):
    """ maps each item in tuple to a string and doesnt include parens """
    return ', '.join(map(str, tuple_))

# --- Strings ----


def remove_chars(instr, illegals_chars):
    outstr = instr
    for ill_char in iter(illegals_chars):
        outstr = outstr.replace(ill_char, '')
    return outstr


def unindent(string):
    return textwrap.dedent(string)


def indent(string, indent='    '):
    return indent + string.replace('\n', '\n' + indent)


def indentjoin(strlist, indent='\n    '):
    return indent + indent.join(map(str, strlist))


def truncate_str(str_, maxlen=110):
    if len(str_) < maxlen:
        return str_
    else:
        truncmsg = ' ~~~TRUNCATED~~~ '
        maxlen_ = maxlen - len(truncmsg)
        lowerb  = int(maxlen_ * .8)
        upperb  = maxlen_ - lowerb
        return str_[:lowerb] + truncmsg + str_[-upperb:]


def pack_into(instr, textwidth=160, breakchars=' ', break_words=True, newline_prefix=''):
    textwidth_ = textwidth
    line_list = ['']
    word_list = instr.split(breakchars)
    for word in word_list:
        if len(line_list[-1]) + len(word) > textwidth_:
            line_list.append('')
            textwidth_ = textwidth - len(newline_prefix)
        while break_words and len(word) > textwidth_:
            line_list[-1] += word[:textwidth_]
            line_list.append('')
            word = word[textwidth_:]
        line_list[-1] += word + ' '
    return ('\n' + newline_prefix).join(line_list)


def newlined_list(list_, joinstr=', ', textwidth=160):
    """ Converts a list to a string but inserts a new line after textwidth chars """
    newlines = ['']
    for word in list_:
        if len(newlines[-1]) + len(word) > textwidth:
            newlines.append('')
        newlines[-1] += word + joinstr
    return '\n'.join(newlines)


def joins(string, list_, with_head=True, with_tail=False, tostrip='\n'):
    head = string if with_head else ''
    tail = string if with_tail else ''
    to_return = head + string.join(map(str, list_)) + tail
    to_return = to_return.strip(tostrip)
    return to_return


def indent_list(indent, list_):
    return imap(lambda item: indent + str(item), list_)


def filesize_str(fpath):
    _, fname = split(fpath)
    mb_str = file_megabytes_str(fpath)
    return 'filesize(%r)=%s' % (fname, mb_str)


def byte_str2(nBytes):
    if nBytes < 2.0 ** 10:
        return byte_str(nBytes, 'KB')
    if nBytes < 2.0 ** 20:
        return byte_str(nBytes, 'KB')
    if nBytes < 2.0 ** 30:
        return byte_str(nBytes, 'MB')
    else:
        return byte_str(nBytes, 'GB')


def byte_str(nBytes, unit='bytes'):
    if unit.lower().startswith('b'):
        nUnit = nBytes
    elif unit.lower().startswith('k'):
        nUnit =  nBytes / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        nUnit =  nBytes / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        nUnit = nBytes / (2.0 ** 30)
    else:
        raise NotImplementedError('unknown nBytes=%r unit=%r' % (nBytes, unit))
    return '%.2f %s' % (nUnit, unit)


def file_megabytes_str(fpath):
    from . import util_path
    return ('%.2f MB' % util_path.file_megabytes(fpath))


# <Alias repr funcs>
GLOBAL_TYPE_ALIASES = []


def extend_global_aliases(type_aliases):
    global GLOBAL_TYPE_ALIASES
    GLOBAL_TYPE_ALIASES.extend(type_aliases)


def var_aliased_repr(var, type_aliases):
    global GLOBAL_TYPE_ALIASES
    # Replace aliased values
    for alias_type, alias_name in (type_aliases + GLOBAL_TYPE_ALIASES):
        if isinstance(var, alias_type):
            return alias_name + '<' + str(id(var)) + '>'
    return repr(var)


def list_aliased_repr(args, type_aliases=[]):
    return [var_aliased_repr(item, type_aliases)
            for item in args]


def dict_aliased_repr(dict_, type_aliases=[]):
    return ['%s : %s' % (key, var_aliased_repr(val, type_aliases))
            for (key, val) in dict_.iteritems()]

# </Alias repr funcs>


def func_str(func, args=[], kwargs={}, type_aliases=[]):
    """ string representation of function definition """
    repr_list = list_aliased_repr(args, type_aliases) + dict_aliased_repr(kwargs)
    argskwargs_str = newlined_list(repr_list, ', ', textwidth=80)
    func_str = '%s(%s)' % (func.func_name, argskwargs_str)
    return func_str


def dict_itemstr_list(dict_, strvals=False):
    if strvals:
        itemstr_iter = ('%s : %s,' % (key, val) for (key, val) in dict_.iteritems())
    else:
        itemstr_iter = ('%r : %r,' % (key, val) for (key, val) in dict_.iteritems())
    return list(itemstr_iter)


def dict_str(dict_, strvals=False):
    itemstr_list = dict_itemstr_list(dict_, strvals)
    return '{%s\n}' % indentjoin(itemstr_list)


def horiz_string(*args):
    '''
    prints a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.
    str_list = ['A = ', str(np.array(((1,2),(3,4)))), ' * ', str(np.array(((1,2),(3,4))))]
    '''
    if len(args) == 1 and not isinstance(args[0], str):
        str_list = args[0]
    else:
        str_list = args
    all_lines = []
    hpos = 0
    for sx in xrange(len(str_list)):
        str_ = str(str_list[sx])
        lines = str_.split('\n')
        line_diff = len(lines) - len(all_lines)
        # Vertical padding
        if line_diff > 0:
            all_lines += [' ' * hpos] * line_diff
        # Add strings
        for lx, line in enumerate(lines):
            all_lines[lx] += line
            hpos = max(hpos, len(all_lines[lx]))
        # Horizontal padding
        for lx in xrange(len(all_lines)):
            hpos_diff = hpos - len(all_lines[lx])
            if hpos_diff > 0:
                all_lines[lx] += ' ' * hpos_diff
    ret = '\n'.join(all_lines)
    return ret


def listinfo_str(list_):
    info_list = enumerate([(type(item), item) for item in list_])
    info_str  = indentjoin(map(repr, info_list, '\n  '))
    return info_str


def str2(obj):
    if isinstance(obj, dict):
        return str(obj).replace(', ', '\n')[1:-1]
    if isinstance(obj, type):
        return str(obj).replace('<type \'', '').replace('\'>', '')
    else:
        return str(obj)


def get_unix_timedelta_str(unixtime_diff):
    timedelta = get_unix_timedelta(unixtime_diff)
    sign = '+' if unixtime_diff >= 0 else '-'
    timedelta_str = sign + str(timedelta)
    return timedelta_str


class NpPrintOpts(object):
    def __init__(self, **kwargs):
        self.orig_opts = np.get_printoptions()
        self.new_opts = kwargs
    def __enter__(self):
        np.set_printoptions(**self.new_opts)
    def __exit__(self, type, value, trace):
        np.set_printoptions(**self.orig_opts)


def full_numpy_repr(arr):
    with NpPrintOpts(threshold=np.uint64(-1)):
        arr_repr = repr(arr)
    return arr_repr


def str_between(str_, startstr, endstr):
    startpos = str_.find(startstr) + len(startstr)
    endpos = str_.find(endstr) - 1
    return str_[startpos:endpos]
