from __future__ import division, print_function
from itertools import imap
from .util_inject import inject
from . import util_path
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


# --- Strings ----
def remove_chars(instr, illegals_chars):
    outstr = instr
    for ill_char in iter(illegals_chars):
        outstr = outstr.replace(ill_char, '')
    return outstr


def indent(string, indent='    '):
    return indent + string.replace('\n', '\n' + indent)


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


def print_filesize(fpath):
    print(filesize_str(fpath))


def filesize_str(fpath):
    _, fname = util_path.split(fpath)
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
    return ('%.2f MB' % util_path.file_megabytes(fpath))
