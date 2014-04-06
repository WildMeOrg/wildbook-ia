from __future__ import division, print_function
from itertools import product as iprod
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[dict]')


def all_dict_combinations(varied_dict):
    viter = varied_dict.iteritems()
    tups_list = [[(key, val) for val in val_list] for (key, val_list) in viter]
    dict_list = [{key: val for (key, val) in tups} for tups in iprod(*tups_list)]
    return dict_list


def dict_union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def dict_union(*args):
    return dict([item for dict_ in iter(args) for item in dict_.iteritems()])


def items_sorted_by_value(dict_):
    sorted_items = sorted(dict_.iteritems(), key=lambda (k, v): v[1])
    return sorted_items


def keys_sorted_by_value(dict_):
    sorted_keys = sorted(dict_, key=lambda key: dict_[key])
    return sorted_keys
