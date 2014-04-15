#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import functools
from pkg_resources import parse_version

ASSERT_FUNCS = []


def version_check(target=None):
    def wrapper1(func):
        # Decorator which adds funcs to ASSERT_FUNCS
        global ASSERT_FUNCS
        @functools.wraps(func)
        def wrapper2(*args, **kwargs):
            name = func.func_name
            current_version = func(*args, **kwargs)
            print('%s: %r >= (target=%r)?' % (name, current_version, target))
            if target is None:
                assert current_version is not None
            else:
                assert parse_version(current_version) >= parse_version(target)
            return current_version, target
        ASSERT_FUNCS.append(wrapper2)
        return wrapper2
    return wrapper1


@version_check('1.1.7')
def pillow_version():
    from PIL import Image
    return Image.VERSION


@version_check('1.3.1')
def matplotlib_version():
    import matplotlib as mpl
    return mpl.__version__


@version_check('0.13.2')
def scipy_version():
    import scipy
    return scipy.__version__


@version_check('1.8.0')
def numpy_version():
    import numpy
    return numpy.__version__


@version_check('4.10.1')
def PyQt4_version():
    from PyQt4 import QtCore
    return QtCore.PYQT_VERSION_STR


def assert_modules():
    for func in ASSERT_FUNCS:
        try:
            func()
            print(func.func_name + ' passed')
        except AssertionError as ex:
            print(func.func_name + ' FAILED!!!')
            print(ex)

if __name__ == '__main__':
    assert_modules()
