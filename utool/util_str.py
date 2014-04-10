from __future__ import division, print_function
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


def truncate_str(str, maxlen=110):
    if len(str) < maxlen:
        return str
    else:
        truncmsg = ' ~~~TRUNCATED~~~ '
        maxlen_ = maxlen - len(truncmsg)
        lowerb  = int(maxlen_ * .8)
        upperb  = maxlen_ - lowerb
        return str[:lowerb] + truncmsg + str[-upperb:]


def pack_into(instr, textwidth=160, breakchars=' ', break_words=True):
    newlines = ['']
    word_list = instr.split(breakchars)
    for word in word_list:
        if len(newlines[-1]) + len(word) > textwidth:
            newlines.append('')
        while break_words and len(word) > textwidth:
            newlines[-1] += word[:textwidth]
            newlines.append('')
            word = word[textwidth:]
        newlines[-1] += word + ' '
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


def dict_str(dict_):
    itemstr_iter = ('%s : %r,' % (key, val) for (key, val) in dict_.iteritems())
    return '{%s\n}' % indentjoin(itemstr_iter)


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
