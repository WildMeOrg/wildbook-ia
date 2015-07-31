#!/usr/bin/env python2.7
"""
Very useful script to ensure you have all the modules you need

CommandLine:
    python -m ibeis.tests.assert_modules
    python -m ibeis.tests.assert_modules --test-assert_modules --nolibdep
    python -m ibeis.tests.assert_modules --test-assert_modules


MacFix:
    # Remove the copy of pyrf
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/
    sudo rm -rf pyrf-1.0.0.dev1-py2.7.egg/

Updater For Linux:
    sudo pip install matplotlib --upgrade
    sudo pip install Pillow --upgrade
    sudo pip install scipy --upgrade
    sudo pip install numpy --upgrade

    ./super_setup.py --bootstrap --upgrade
"""
from __future__ import absolute_import, division, print_function
import sys
import functools
import utool as ut
from pkg_resources import parse_version
from utool._internal.meta_util_six import get_funcname

ASSERT_FUNCS = []


def get_site_package_directories():
    import site
    import sys
    import six
    sitepackages = site.getsitepackages()
    if sys.platform.startswith('darwin'):
        if six.PY2:
            macports_site = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages'
        else:
            macports_site = '/opt/local/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages'
            assert six.PY2, 'fix this for python 3'
        sitepackages = [macports_site] + sitepackages
    return sitepackages


def check_alternate_installs():
    '/Library/Python/2.7/site-packages'

    pass


def checkinfo(target=None, pipname=None):
    """
    checkinfo functions return info_dict
    """
    def wrapper1(func):
        """
        Returns;
            tuple : passed, current_version, target, infodict, statustext, suggested_fix
        """

        # Decorator which adds funcs to ASSERT_FUNCS
        global ASSERT_FUNCS
        @functools.wraps(func)
        def checkinfo_wrapper(*args, **kwargs):
            suggested_fix = ''
            funcname = get_funcname(func)
            packagename = funcname.replace('_version', '')
            pipname_ = pipname if pipname is not None else packagename
            try:
                infodict = func(*args, **kwargs)
            except ImportError as ex:
                infodict = module_stdinfo_dict(None, name=pipname_)
                suggested_fix = 'pip install ' + pipname_
                if not sys.platform.startswith('win32'):
                    suggested_fix = 'sudo ' + suggested_fix
                return False, 'None', target, infodict, ut.formatex(ex), suggested_fix
            except Exception as ex:
                infodict = module_stdinfo_dict(None, name=pipname_)
                return False, 'None', target, infodict, ut.formatex(ex), 'Some unknown error in ' + packagename
            current_version = infodict['__version__']
            msg = ut.dict_str(infodict, strvals=True)
            msg += '\n' + '%s: %r >= (target=%r)?' % (funcname, current_version, target)
            statustext = ut.msgblock(infodict['__name__'], msg)
            passed = target is None or (current_version is not None and parse_version(current_version.replace('.dev1', '')) >= parse_version(target))

            if not passed:
                suggested_fix = 'pip install ' + infodict['__name__'] + ' --upgrade'
                if not sys.platform.startswith('win32'):
                    suggested_fix = 'sudo ' + suggested_fix
            return passed, current_version, target, infodict, statustext, suggested_fix
        ASSERT_FUNCS.append(checkinfo_wrapper)
        return checkinfo_wrapper
    return wrapper1


def module_stdinfo_dict(module, versionattr='__version__', version=None, libdep=None, name=None, **kwargs):
    #if module is None:
    #    module = object
    infodict = {
        '__version__': version if module is None or version is not None else getattr(module, versionattr, None),
        '__name__': name if module is None else getattr(module, '__name__', name),
        '__file__': 'None' if module is None else getattr(module, '__file__', None),
    }
    if libdep is not None:
        infodict['libdep'] = libdep
    if not ut.QUIET:
        infodict.update(kwargs)
    return infodict


@checkinfo('6.0.8')
def pip_version():
    import pip
    return module_stdinfo_dict(pip)


@checkinfo(None)
def pyflann_version():
    import pyflann
    if ut.get_argflag('--nolibdep'):
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(pyflann.flannlib._name)
    return module_stdinfo_dict(pyflann, libdep=libdep)


@checkinfo('1.1.1')
def pyhesaff_version():
    import pyhesaff
    if ut.get_argflag('--nolibdep'):
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(pyhesaff.__LIB_FPATH__)
    return module_stdinfo_dict(pyhesaff, libdep=libdep)


@checkinfo('1.0.0')
def pyrf_version():
    import pyrf
    if ut.get_argflag('--nolibdep'):
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(pyrf.RF_CLIB._name)
    return module_stdinfo_dict(pyrf, libdep=libdep)


@checkinfo('1.1.1')
def utool_version():
    import utool
    return module_stdinfo_dict(utool)


@checkinfo('1.0.1')
def vtool_version():
    import vtool
    libdep = None
    if ut.get_argflag('--nolibdep'):
        libdep = None
    else:
        try:
            libdep = ut.get_dynlib_dependencies(vtool.sver_c_wrapper.lib_fname)
        except Exception:
            pass
    return module_stdinfo_dict(vtool, libdep=libdep)


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
    #print(cv2.getBuildInformation())
    if ut.get_argflag('--nolibdep'):
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(cv2.__file__)
    return module_stdinfo_dict(cv2, libdep=libdep)


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


@checkinfo('0.15.1')
def pandas_version():
    import pandas
    version = pandas.version.version
    return module_stdinfo_dict(pandas, version=version)


@checkinfo('0.6.1')
def statsmodels_version():
    import statsmodels
    version = statsmodels.version.version
    return module_stdinfo_dict(statsmodels, version=version)


@checkinfo('0.10.1')
def flask_version():
    import flask
    return module_stdinfo_dict(flask)


@checkinfo('2.0.1')
def flask_cors_version():
    import flask.ext.cors
    return module_stdinfo_dict(flask.ext.cors)


@checkinfo('4.0.2')
def tornado_version():
    import tornado
    return module_stdinfo_dict(tornado, 'version')


def check_modules_exists():
    # Modules in this list don't really need to be inspected
    # just make sure they are there
    modname_list = [
        'simplejson',
        'flask',
        'parse',
        'tornado',
        'pandas',
        'statsmodels',
    ]
    failed_list = []
    for modname in modname_list:
        try:
            globals_ = {}
            locals_ = {}
            exec('import ' + modname, globals_, locals_)
        except ImportError:
            failed_list.append(modname)
        except Exception as ex:
            ut.printex(ex, 'Some othere error happened when importing %r ' % (modname,), iswarning=True)
            failed_list.append(modname)
    if len(failed_list) > 0:
        print('The following modules are not installed')
        print('\n'.join(failed_list))
    return ''


def assert_modules():
    """
    checkinfo functions return info_dict

    CommandLine:
        python -m ibeis.tests.assert_modules --test-assert_modules

    Example:
        >>> # DOCTEST_ENABLE
        >>> from ibeis.tests.assert_modules import *   # NOQA
        >>> detailed_msg = assert_modules()
        >>> print(detailed_msg)
    """

    MACHINE_NAME = ut.get_computer_name()

    machine_info_lines = []

    machine_info_lines.append('sys.version = %r ' % (sys.version))
    machine_info_lines.append('PATH = ' + ut.list_str(ut.get_path_dirs()))
    machine_info_lines.append('\n\n\n============================')
    machine_info_lines.append('Begining assert modules main')
    machine_info_lines.append('* MACHINE_NAME = %r' % MACHINE_NAME)
    machine_info_text = '\n'.join(machine_info_lines)
    print(machine_info_text)

    line_list = []
    failed_list = []
    fix_list = []
    for func in ASSERT_FUNCS:
        passed, current_version, target, infodict, statustext, suggested_fix = func()
        line_list.append(statustext)
        try:
            assert passed, infodict['__name__'] + ' did not pass'
        except AssertionError as ex:
            failed_list.append(get_funcname(func) + ' FAILED!!!')
            fix_list.append(suggested_fix)
            #line_list.append(get_funcname(func) + ' FAILED!!!')
            line_list.append(ut.formatex(ex))
        else:
            line_list.append(get_funcname(func) + ' passed\n')
            line_list.append('')
    output_text = '\n'.join(line_list)
    failed_text = '\n'.join(failed_list)
    print(output_text)
    print(failed_text)
    check_exist_text = check_modules_exists()
    print(check_exist_text)
    fix_text = ''
    if len(fix_list) > 0:
        fix_text += ('suggested fixes:\n')
        fix_text += ('\n'.join(fix_list) + '\n')
        print(fix_text)

    detailed_msg = '\n'.join([
        machine_info_text,
        output_text,
        failed_text,
        check_exist_text,
        fix_text,
    ])

    return detailed_msg


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.tests.assert_modules

        python -c "import utool, ibeis.tests.assert_modules; utool.doctest_funcs(ibeis.tests.assert_modules, allexamples=True)"
        python -m ibeis.tests.assert_modules --allexamples
        python ~/code/ibeis/ibeis/tests/assert_modules.py

        python -m ibeis.tests.assert_modules
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    print('in assert module main')
    if len(sys.argv) == 1:
        assert_modules()
    else:
        ut.doctest_funcs()
