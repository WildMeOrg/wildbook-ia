from __future__ import absolute_import, division, print_function
import re
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


RE_FLAGS = re.MULTILINE | re.DOTALL
RE_KWARGS = {'flags': RE_FLAGS}


def get_match_text(match):
    if match is not None:
        start, stop = match.start(), match.end()
        return match.string[start:stop]
    else:
        return None


def regex_search(regex, text):
    if text is None:
        return None
    match = re.search(regex, text, **RE_KWARGS)
    return get_match_text(match)


def regex_split(regex, text):
    return re.split(regex, text, **RE_KWARGS)


def named_field(key, regex):
    if key is None:
        return regex
    return r'(?P<%s>%s)' % (key, regex)


def named_field_regex(keypat_tups):
    named_fields = [named_field(key, pat) for key, pat in keypat_tups]
    regex = ''.join(named_fields)
    return regex


def regex_parse(regex, text):
    match = re.match(regex, text, **RE_KWARGS)
    if match is not None:
        parse_dict = match.groupdict()
        return parse_dict
    return None
