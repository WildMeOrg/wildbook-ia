from __future__ import absolute_import, division, print_function
import sys
# Science
import numpy as np
import types
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[type]')


# Very odd that I have to put in dtypes in two different ways.
VALID_INT_TYPES = (types.IntType,
                   types.LongType,
                   np.typeDict['int64'],
                   np.typeDict['int32'],
                   np.typeDict['uint8'],
                   np.dtype('int32'),
                   np.dtype('uint8'),
                   np.dtype('int64'),)

VALID_FLOAT_TYPES = (types.FloatType,
                     np.typeDict['float64'],
                     np.typeDict['float32'],
                     np.typeDict['float16'],
                     np.dtype('float64'),
                     np.dtype('float32'),
                     np.dtype('float16'),)


def try_cast(var, type_, default=None):
    if type_ is None:
        return var
    try:
        return type_(var)
    except Exception:
        return default


def assert_int(var, lbl='var'):
    try:
        assert is_int(var), 'type(%s)=%r is not int' % (lbl, get_type(var))
    except AssertionError:
        print('[tools] %s = %r' % (lbl, var))
        print('[tools] VALID_INT_TYPES: %r' % VALID_INT_TYPES)
        raise

if sys.platform == 'win32':
    # Well this is a weird system specific error
    # https://github.com/numpy/numpy/issues/3667
    def get_type(var):
        'Gets types accounting for numpy'
        return var.dtype if isinstance(var, np.ndarray) else type(var)
else:
    def get_type(var):
        'Gets types accounting for numpy'
        return var.dtype.type if isinstance(var, np.ndarray) else type(var)


def is_type(var, valid_types):
    'Checks for types accounting for numpy'
    #printDBG('checking type var=%r' % (var,))
    #var_type = type(var)
    #printDBG('type is type(var)=%r' % (var_type,))
    #printDBG('must be in valid_types=%r' % (valid_types,))
    #ret = var_type in valid_types
    #printDBG('result is %r ' % ret)
    return get_type(var) in valid_types


def is_int(var):
    return is_type(var, VALID_INT_TYPES)


def is_float(var):
    return is_type(var, VALID_FLOAT_TYPES)


def is_str(var):
    return isinstance(var, str)
    #return is_type(var, VALID_STRING_TYPES)


def is_bool(var):
    return isinstance(var, bool) or isinstance(var, np.bool_)
    #return is_type(var, VALID_BOOLEAN_TYPES)


def is_dict(var):
    return isinstance(var, dict)
    #return is_type(var, VALID_BOOLEAN_TYPES)


def is_list(var):
    return isinstance(var, list)
    #return is_type(var, VALID_LIST_TYPES)
