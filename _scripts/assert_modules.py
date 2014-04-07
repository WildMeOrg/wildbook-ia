#!/usr/bin/env python
from __future__ import division, print_function

ASSERT_FUNCS = []


def checkme(func):
    # Decorator which adds funcs to ASSERT_FUNCS
    global ASSERT_FUNCS
    ASSERT_FUNCS.append(func)
    return func


@checkme
def assert_pillow():
    from PIL import Image
    assert Image.VERSION == '1.1.7'


@checkme
def assert_matplotlib():
    import matplotlib as mpl
    assert mpl.__version__ == '1.3.1'


@checkme
def assert_scipy():
    import scipy
    assert scipy.__version__ == '0.13.2'


@checkme
def assert_numpy():
    import numpy
    assert numpy.__version__ == '1.8.0'


@checkme
def assert_PyQt4():
    import PyQt4
    assert PyQt4 is not None


def assert_modules():
    for func in ASSERT_FUNCS:
        try:
            func()
            print(func.func_name + ' passed')
        except AssertionError as ex:
            print(func.func_name + ' FAILED')
            print(ex)

if __name__ == '__main__':
    assert_modules()
