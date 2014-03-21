from __future__ import division, print_function
from itertools import imap
from .util_inject import inject
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
