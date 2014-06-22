from __future__ import absolute_import, division, print_function
import utool
from functools import wraps
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[decor]')

#
#-----------------
# IBEIS DECORATORS
#-----------------


# DECORATORS::OTHERS

def default_decorator(input_):
    if utool.is_funclike(input_):
        func_ = input_
        return utool.indent_func(profile(func_))
    else:
        def closure_default(func):
            return utool.indent_func(input_)(profile(func))
        return closure_default


# DECORATORS::ADDER


def adder(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @wraps(func)
    def wrp_adder(*args, **kwargs):
        if not utool.QUIET and utool.VERBOSE:
            print('[ADD]: ' + func.func_name)
        return func(*args, **kwargs)
    return wrp_adder


# DECORATORS::DELETER

def deleter(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @wraps(func)
    def wrp_adder(*args, **kwargs):
        if not utool.QUIET and utool.VERBOSE:
            print('[DELETE]: ' + func.func_name)
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
        if not utool.QUIET and utool.VERBOSE:
            print('[SET]: ' + func.func_name)
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
    @utool.on_exception_report_input
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

getter_1toM = getter_vector_output
getter_1to1 = getter
getter_1to1 = getter


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


def ider(func):
    """ This function takes returns ids subject to conditions """
    return default_decorator(func)
