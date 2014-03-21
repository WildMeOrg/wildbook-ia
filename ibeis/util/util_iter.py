from __future__ import division, print_function
from .util_type import is_str
import numpy as np
from itertools import chain, cycle
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[iter]')


def ensure_iterable(obj):
    if np.iterable(obj):
        return obj
    else:
        return [obj]


def iflatten(list_):
    flat_iter = chain.from_iterable(list_)  # very fast flatten
    return flat_iter


def ichunks(list_, size):
    'Yield successive n-sized chunks from list_.'
    for ix in xrange(0, len(list_), size):
        yield list_[ix: ix + size]


def interleave(args):
    arg_iters = map(iter, args)
    cycle_iter = cycle(arg_iters)
    for iter_ in cycle_iter:
        yield iter_.next()


def class_iter_input(func):
    '''
    class_iter_input is a decorator which expects to be used on class methods.
    It lets the user pass either a vector or a scalar to a function, as long as
    the function treats everything like a vector. Input and output is sanatized
    to the user expected format on return.
    '''
    def iter_wrapper(self, input_, *args, **kwargs):
        is_scalar = not np.iterable(input_) or is_str(input_)
        result = func(self, (input_,), *args, **kwargs) if is_scalar else \
            func(self, input_, *args, **kwargs)
        if is_scalar:
            result = result[0]
        return result
    iter_wrapper.func_name = func.func_name
    return iter_wrapper
