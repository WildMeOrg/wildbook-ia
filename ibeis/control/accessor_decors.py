from __future__ import absolute_import, division, print_function
import utool
from functools import wraps
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[decor]')

#
#-----------------
# IBEIS DECORATORS
#-----------------


# DECORATORS::OTHERS

def default_decorator(func):
    return utool.indent_func(profile(func))


# DECORATORS::ADDER


def adder(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @wraps(func)
    def wrp_adder(*args, **kwargs):
        return func(*args, **kwargs)
    return wrp_adder


# DECORATORS::DELETER

def deleter(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @wraps(func)
    def wrp_adder(*args, **kwargs):
        return func(*args, **kwargs)
    return wrp_adder


# DECORATORS::SETTER

def setter_general(func):
    func = default_decorator(func)
    return func


def setter(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input2(argx_list=range(0, 2))
    @wraps(func)
    def wrp_setter(*args, **kwargs):
        #print('set: func_name=%r, args=%r, kwargs=%r' % (func.func_name, args, kwargs))
        return func(*args, **kwargs)
    return wrp_setter


# DECORATORS::GETTER

def getter(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @wraps(func)
    def wrp_getter(*args, **kwargs):
        return func(*args, **kwargs)
    return wrp_getter


def getter_vector_output(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a homogenous list of values
    """
    func = default_decorator(func)
    @utool.accepts_scalar_input_vector_output
    @wraps(func)
    def getter_vector_wrp(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_vector_wrp


def getter_numpy(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    getter_func = getter(func)
    @utool.accepts_numpy
    @wraps(func)
    def getter_numpy_wrp(*args, **kwargs):
        return getter_func(*args, **kwargs)
    return getter_numpy_wrp


def getter_numpy_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    getter_func = getter_vector_output(func)
    @utool.accepts_numpy
    @wraps(func)
    def getter_numpy_vector_wrp(*args, **kwargs):
        return getter_func(*args, **kwargs)
    return getter_numpy_vector_wrp


def getter_general(func):
    """ Getter decorator for functions which has no gaurentees """
    return default_decorator(func)
