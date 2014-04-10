from __future__ import division, print_function
import utool
from functools import wraps
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[decor]')

#
#-----------------
# IBEIS DECORATORS
#-----------------


"""
Common wrappers
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
"""
# DECORATORS::ADDER


def adder(func):
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper


# DECORATORS::SETTER

def setter(func):
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper


# DECORATORS::GETTER

def getter(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    @utool.accepts_scalar_input
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
    def getter_wrapper1(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper1


def getter_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a homogenous list of values """
    @utool.accepts_scalar_input_vector_output
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
    def getter_wrapper3(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper3


def getter_numpy(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    @utool.accepts_numpy
    @getter
    def getter_wrapperNP(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapperNP


def getter_numpy_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    @utool.accepts_numpy
    @getter_vector_output
    def getter_wrapperNP2(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapperNP2


def getter_general(func):
    """ Getter decorator for functions which has no gaurentees """
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
    def getter_wrapper2(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_wrapper2


# DECORATORS::DELETER

def deleter(func):
    @utool.indent_func
    @utool.ignores_exc_tb
    @wraps(func)
    def adder_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return adder_wrapper
