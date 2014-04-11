from __future__ import absolute_import, division, print_function
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[decor]')

#
#-----------------
# IBEIS DECORATORS
#-----------------


# DECORATORS::ADDER


def adder(func):
    @utool.common_wrapper(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper


# DECORATORS::SETTER

def setter(func):
    @utool.common_wrapper(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper


# DECORATORS::GETTER

def getter(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    @utool.accepts_scalar_input
    @utool.common_wrapper(func)
    def getter_wrapper1(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper1


def getter_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a homogenous list of values """
    @utool.accepts_scalar_input_vector_output
    @utool.common_wrapper(func)
    def getter_wrapper3(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper3


def getter_numpy(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    getter_func = getter(func)
    @utool.accepts_numpy
    def getter_wrapperNP(*args, **kwargs):
        return getter_func(*args, **kwargs)
    return getter_wrapperNP


def getter_numpy_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    getter_func = getter_vector_output(func)
    @utool.accepts_numpy
    def getter_wrapperNP2(*args, **kwargs):
        return getter_func(*args, **kwargs)
    return getter_wrapperNP2


def getter_general(func):
    """ Getter decorator for functions which has no gaurentees """
    @utool.common_wrapper(func)
    def getter_wrapper2(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper2


# DECORATORS::DELETER

def deleter(func):
    @utool.common_wrapper(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper
