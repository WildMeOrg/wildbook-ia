#!/usr/bin/env python2.7
"""
Very useful script to ensure you have all the modules you need
"""
from __future__ import absolute_import, division, print_function
import sys
import functools
import utool as ut
from pkg_resources import parse_version
import utool
from utool._internal.meta_util_six import get_funcname

ASSERT_FUNCS = []


def checkinfo(target=None):
    """
    checkinfo functions return info_dict
    """
    def wrapper1(func):
        # Decorator which adds funcs to ASSERT_FUNCS
        global ASSERT_FUNCS
        @functools.wraps(func)
        def wrapper2(*args, **kwargs):
            funcname = get_funcname(func)
            infodict = func(*args, **kwargs)
            current_version = infodict['__version__']
            msg = utool.dict_str(infodict) + '\n' + '%s: %r >= (target=%r)?' % (funcname, current_version, target)
            statustext = ut.msgblock(infodict['__name__'], msg)
            if target is None:
                assert current_version is not None
            else:
                assert parse_version(current_version.replace('.dev1', '')) >= parse_version(target)
            return current_version, target, infodict, statustext
        ASSERT_FUNCS.append(wrapper2)
        return wrapper2
    return wrapper1


def module_stdinfo_dict(module, versionattr='__version__', **kwargs):
    infodict = {
        '__version__': getattr(module, versionattr, None),
        '__name__': getattr(module, '__name__'),
        '__file__': getattr(module, '__file__'),
    }
    infodict.update(kwargs)
    return infodict


@checkinfo('1.0.0')
def pyhesaff_version():
    import pyhesaff
    return module_stdinfo_dict(pyhesaff)


@checkinfo('1.0.0')
def pyrf_version():
    import pyrf
    return module_stdinfo_dict(pyrf)


@checkinfo('1.0.0')
def utool_version():
    import utool
    return module_stdinfo_dict(utool)


#@checkinfo('1.1.7')
@checkinfo('2.4.0')
def pillow_version():
    from PIL import Image
    import PIL
    pil_path = PIL.__path__
    if len(PIL.__path__) > 1:
        print('WARNING!!! THERE ARE MULTIPLE PILS! %r ' % PIL.__path__)
    return module_stdinfo_dict(
        Image, versionattr='PILLOW_VERSION', image_version=Image.VERSION, pil_path=pil_path)


@checkinfo('1.3.1')
def matplotlib_version():
    import matplotlib as mpl
    return module_stdinfo_dict(mpl)


@checkinfo('2.4.8')
def opencv_version():
    import cv2
    return module_stdinfo_dict(cv2, libdep=utool.get_dynlib_dependencies(cv2.__file__))


@checkinfo('0.13.2')
def scipy_version():
    import scipy
    return module_stdinfo_dict(scipy)


@checkinfo('1.8.0')
def numpy_version():
    import numpy
    return module_stdinfo_dict(numpy)


@checkinfo('4.9.1')  # 4.10.1 on windows
def PyQt4_version():
    from PyQt4 import QtCore
    return module_stdinfo_dict(QtCore, 'PYQT_VERSION_STR')


def assert_modules():
    """
    checkinfo functions return info_dict

    Example:
        >>> # DOCTEST_ENABLE
        >>> from ibeis.tests.assert_modules import *   # NOQA
        >>> assert_modules()
    """

    MACHINE_NAME = utool.get_computer_name()
    print('\n\n\n============================')
    print('Begining assert modules main')
    print('* MACHINE_NAME = %r' % MACHINE_NAME)

    line_list = []
    for func in ASSERT_FUNCS:
        try:
            current_version, target, infodict, statustext = func()
            line_list.append(statustext)
            line_list.append(get_funcname(func) + ' passed')
            line_list.append('')
        except AssertionError as ex:
            line_list.append(get_funcname(func) + ' FAILED!!!')
            line_list.append(ut.formatex(ex))
    output_text = '\n'.join(line_list)
    print(output_text)


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.tests.assert_modules; utool.doctest_funcs(ibeis.tests.assert_modules, allexamples=True)"
        python -m ibeis.tests.assert_modules
        python -m ibeis.tests.assert_modules --allexamples
        python ~/code/ibeis/ibeis/tests/assert_modules.py
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    print('in assert module main')
    if len(sys.argv) == 1:
        assert_modules()
    else:
        ut.doctest_funcs()
