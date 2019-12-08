#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


#LIB_DEP = ut.get_argflag('--nolibdep')
LIB_DEP = not ut.get_argflag('--libdep')


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


def version_ge_target(version, target=None):
    if target is None:
        passed = True
    elif version is None:
        passed = False
    else:
        _version = version.replace('.dev1', '')
        passed = parse_version(_version) >= parse_version(target)
    return passed


def checkinfo(target=None, pipname=None):
    """
    checkinfo functions return info_dict containing __version__
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
            # Build status text
            msg = ut.repr2(infodict, strvals=True)
            msg += '\n' + '%s: %r >= (target=%r)?' % (funcname, current_version, target)
            statustext = ut.msgblock(infodict['__name__'], msg)
            # Check if passed
            passed = version_ge_target(current_version, target)
            # Suggest possible fix
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


def reg_std_version_check(version, modname):
    def checkstdmod_version():
        mod = ut.import_modname(modname)
        return module_stdinfo_dict(mod)
    ut.set_funcname(checkstdmod_version, '%s_version' % (modname,))
    func = checkinfo(version)(checkstdmod_version)
    return func


reg_std_version_check('1.5.3', 'pynmea2')
#reg_std_version_check(None, 'wget')
# reg_std_version_check(None, 'pygco')
reg_std_version_check('6.0.8', 'pip')
reg_std_version_check('1.1.1', 'utool')
reg_std_version_check('0.12.3', 'skimage')
reg_std_version_check('1.1.6', 'cachetools')


# @checkinfo('1.1.1')
# def utool_version():
#     import utool
#     return module_stdinfo_dict(utool)

# @checkinfo('1.5.3')
# def pynmea2_version():
#     # for web
#     import pynmea2
#     return module_stdinfo_dict(pynmea2)

# @checkinfo('6.0.8')
# def pip_version():
#     import pip
#     return module_stdinfo_dict(pip)


@checkinfo(None)
def pyflann_version():
    from vtool_ibeis._pyflann_backend import pyflann as pyflann
    if LIB_DEP:
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(pyflann.flannlib._name)
    return module_stdinfo_dict(pyflann, libdep=libdep)


@checkinfo('1.1.1')
def pyhesaff_version():
    import pyhesaff
    if LIB_DEP:
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(pyhesaff.__LIB_FPATH__)
    return module_stdinfo_dict(pyhesaff, libdep=libdep)


@checkinfo('1.0.0')
def pyrf_version():
    import pyrf
    if LIB_DEP:
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(pyrf.RF_CLIB._name)
    return module_stdinfo_dict(pyrf, libdep=libdep)


@checkinfo('1.0.1')
def vtool_version():
    import vtool_ibeis
    libdep = None
    if LIB_DEP:
        libdep = None
    else:
        try:
            libdep = ut.get_dynlib_dependencies(vtool_ibeis.sver_c_wrapper.lib_fname)
        except Exception:
            pass
    return module_stdinfo_dict(vtool_ibeis, libdep=libdep)


#@checkinfo('1.1.7')
#@checkinfo('2.4.0')
@checkinfo('3.1.0')
def pillow_version():
    from PIL import Image
    import PIL
    pil_path = PIL.__path__
    if len(PIL.__path__) > 1:
        print('WARNING!!! THERE ARE MULTIPLE PILS! %r ' % PIL.__path__)
    return module_stdinfo_dict(
        Image, versionattr='PILLOW_VERSION', image_version=Image.VERSION, pil_path=pil_path)


#@checkinfo('1.3.1')
@checkinfo('1.5.1')
def matplotlib_version():
    import matplotlib as mpl
    return module_stdinfo_dict(mpl)


@checkinfo('2.4.8')
def opencv_version():
    import cv2
    #print(cv2.getBuildInformation())
    if LIB_DEP:
        libdep = None
    else:
        libdep = ut.get_dynlib_dependencies(cv2.__file__)
    return module_stdinfo_dict(cv2, libdep=libdep)


@checkinfo('0.13.2')
def scipy_version():
    import scipy
    return module_stdinfo_dict(scipy)


@checkinfo('0.4.9')
def scipy_linalg_version():
    import scipy.linalg
    return module_stdinfo_dict(scipy.linalg)


@checkinfo('1.8.0')
def numpy_version():
    import numpy
    return module_stdinfo_dict(numpy)


#@checkinfo()
def theano_version():
    import theano
    return module_stdinfo_dict(theano)


#@checkinfo()
def lasagne_version():
    import lasagne
    return module_stdinfo_dict(lasagne)


@checkinfo('4.9.1')  # 4.10.1 on windows
def PyQt4_version():
    # FIXME, pyqt5 is also ok
    from PyQt4 import QtCore
    return module_stdinfo_dict(QtCore, 'PYQT_VERSION_STR')


@checkinfo('0.15.1')
def pandas_version():
    import pandas
    return module_stdinfo_dict(pandas)


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
    #import flask.ext.cors as flask_cors
    import flask_cors
    return module_stdinfo_dict(flask_cors)


@checkinfo('4.0.2')
def tornado_version():
    import tornado
    return module_stdinfo_dict(tornado, 'version')


@checkinfo()
def pygraphviz_version():
    import pygraphviz
    return module_stdinfo_dict(pygraphviz)


@checkinfo(None)
def networkx_version():
    # for web
    import networkx
    return module_stdinfo_dict(networkx)


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
    checkinfo_func

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
    machine_info_lines.append('PATH = ' + ut.repr2(ut.get_path_dirs()))
    machine_info_lines.append('\n\n\n============================')
    machine_info_lines.append('Begining assert modules main')
    machine_info_lines.append('* MACHINE_NAME = %r' % MACHINE_NAME)
    machine_info_text = '\n'.join(machine_info_lines)
    print(machine_info_text)

    statustext_list = []
    failed_list = []
    fix_list = []

    SHOW_STATUS = not ut.get_argflag(('--nostatus', '--nostat'))

    for checkinfo_wrapper in ASSERT_FUNCS:
        passed, current_version, target, infodict, statustext, suggested_fix = checkinfo_wrapper()
        funcname = get_funcname(checkinfo_wrapper)
        if SHOW_STATUS:
            statustext_list.append(statustext)
        if passed:
            statustext_list.append(funcname + ' ' + str(infodict['__version__']) + ' passed')
            #statustext_list.append('')
        else:
            failed_list.append(funcname + ' FAILED!!!')
            fix_list.append(suggested_fix)
            statustext_list.append(funcname + ' FAILED!!!')
        if SHOW_STATUS:
            statustext_list.append('')

    output_text = '\n'.join(statustext_list)

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
    #if len(sys.argv) == 1:
    if not any(argv.startswith('--test-') or argv.startswith('--exec') for argv in sys.argv):
        assert_modules()
    else:
        ut.doctest_funcs()
