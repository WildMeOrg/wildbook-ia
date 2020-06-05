# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
CommandLine:
    python %CODE_DIR%/wbia/_installers/wbia_pyinstaller_data_helper.py --test-get_data_list

"""
from __future__ import absolute_import, division, print_function
import os
import sys
from os.path import join, exists, realpath, abspath, dirname, relpath  # NOQA
import utool as ut


##################################
# System Variables
##################################
PLATFORM = sys.platform
APPLE = PLATFORM.startswith('darwin')
WIN32 = PLATFORM.startswith('win32')
LINUX = PLATFORM.startswith('linux2')

LIB_EXT = {'win32': '.dll', 'darwin': '.dylib', 'linux2': '.so'}[PLATFORM]

##################################
# Asserts
##################################
ibsbuild = ''
root_dir = os.getcwd()


def get_site_package_directories():
    import site
    import sys
    import six

    sitepackages = site.getsitepackages()
    if sys.platform.startswith('darwin'):
        if six.PY2:
            macports_site = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages'
        else:
            # version_str = '.'.join(sys.version.split('.')[0:2])
            macports_site = '/opt/local/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages'
            assert six.PY2, 'fix this for python 3'
        sitepackages = [macports_site] + sitepackages
    return sitepackages


def join_SITE_PACKAGES(*args):
    from os.path import join
    import utool as ut

    fname = join(*args)
    sitepackages = get_site_package_directories()
    path, tried_list = ut.search_in_dirs(
        fname, sitepackages, return_tried=True, strict=True
    )
    return path


def add_data(a, dst, src):
    if dst == '':
        raise ValueError('dst path cannot be the empty string')
    if src == '':
        raise ValueError('src path cannot be the empty string')
    src_ = ut.platform_path(src)
    if not os.path.exists(dirname(dst)) and dirname(dst) != '':
        os.makedirs(dirname(dst))
    _pretty_path = lambda str_: str_.replace('\\', '/')
    # Default datatype is DATA
    dtype = 'DATA'
    # Infer datatype from extension
    # extension = splitext(dst)[1].lower()
    # if extension == LIB_EXT.lower():
    if LIB_EXT[1:] in dst.split('.'):
        dtype = 'BINARY'
    print(
        ut.codeblock(
            """
    [installer] a.add_data(
    [installer]    dst=%r,
    [installer]    src=%r,
    [installer]    dtype=%s)"""
        )
        % (_pretty_path(dst), _pretty_path(src_), dtype)
    )
    assert exists(src_), 'src_=%r does not exist'
    a.datas.append((dst, src_, dtype))


def get_path_extensions():
    """
    Explicitly add modules in case they are not in the Python PATH
    """
    module_repos = [
        'utool',
        'vtool_ibeis',
        'guitool_ibeis',
        'guitool_ibeis.__PYQT__',
        'plottool_ibeis',
        'pyrf',
        'flann/src/python',
        #'pygist',
        'wbia',
        'ibeis_cnn',
        'pydarknet',
        'hesaff',
        'detecttools',
    ]
    pathex = ['.'] + [join('..', repo) for repo in module_repos]
    if APPLE:
        # We need to explicitly add the MacPorts and system Python site-packages folders on Mac
        pathex.append(
            '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/'
        )
        pathex.append(
            '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/'
        )
    return pathex


# IF MPL FAILS:
# MPL has a problem where the __init__.py is not created in the library.  touch __init__.py in the module's path should fix the issue

##################################
# Hesaff + PyRF + FLANN Library
##################################


def get_hidden_imports():
    hiddenimports = [
        'guitool_ibeis.__PYQT__',
        'sklearn.utils.sparsetools._graph_validation',
        'sklearn.utils.sparsetools._graph_tools',
        'scipy.special._ufuncs_cxx',
        'sklearn.utils.lgamma',
        'sklearn.utils.weight_vector',
        'sklearn.neighbors.typedefs',
        'mpl_toolkits.axes_grid1',
        'flask',
        #'flask.ext.cors'  # seems not to work?
        'pandas',
        'pandas.hashtable',
        'statsmodels',
        'theano',
    ]
    return hiddenimports


def get_data_list():
    r"""
    CommandLine:
        python ~/code/wbia/_installers/wbia_pyinstaller_data_helper.py --test-get_data_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_pyinstaller_data_helper import *  # NOQA
        >>> result = get_data_list()
        >>> DATATUP_LIST, BINARYTUP_LIST, iconfile = result
        >>> print('DATATUP_LIST = ' + ut.repr2(DATATUP_LIST))
        >>> print('BINARYTUP_LIST = ' + ut.repr2(BINARYTUP_LIST))
        >>> print(len(DATATUP_LIST))
        >>> print(len(BINARYTUP_LIST))
        >>> print(iconfile)

    """
    # Build data before running analysis for quick debugging
    DATATUP_LIST = []
    BINARYTUP_LIST = []

    # import pyhesaff
    # pyhesaff.HESAFF_CLIB.__LIB_FPATH__
    # import pyrf
    # pyrf.RF_CLIB.__LIB_FPATH__
    # Hesaff
    libhesaff_fname = 'libhesaff' + LIB_EXT
    libhesaff_src = realpath(
        join(root_dir, '..', 'hesaff', 'pyhesaff', libhesaff_fname)
    )
    libhesaff_dst = join(ibsbuild, 'pyhesaff', 'lib', libhesaff_fname)
    DATATUP_LIST.append((libhesaff_dst, libhesaff_src))

    # PyRF
    libpyrf_fname = 'libpyrf' + LIB_EXT
    libpyrf_src = realpath(join(root_dir, '..', 'pyrf', 'pyrf', libpyrf_fname))
    libpyrf_dst = join(ibsbuild, 'pyrf', 'lib', libpyrf_fname)
    DATATUP_LIST.append((libpyrf_dst, libpyrf_src))

    libpyrf_fname = 'libpydarknet' + LIB_EXT
    libpyrf_src = realpath(
        join(root_dir, '..', 'pydarknet', 'pydarknet', libpyrf_fname)
    )
    libpyrf_dst = join(ibsbuild, 'pydarknet', 'lib', libpyrf_fname)
    DATATUP_LIST.append((libpyrf_dst, libpyrf_src))

    # FLANN
    libflann_fname = 'libflann' + LIB_EXT
    # try:
    #    #from vtool_ibeis._pyflann_backend import pyflann as pyflann
    #    #pyflann.__file__
    #    #join(dirname(dirname(pyflann.__file__)), 'build')
    # except ImportError as ex:
    #    print('PYFLANN IS NOT IMPORTABLE')
    #    raise
    # if WIN32 or LINUX:
    # FLANN
    # libflann_src = join_SITE_PACKAGES('pyflann', 'lib', libflann_fname)
    # libflann_dst = join(ibsbuild, libflann_fname)
    # elif APPLE:
    #    # libflann_src = '/pyflann/lib/libflann.dylib'
    #    # libflann_dst = join(ibsbuild, libflann_fname)
    #    libflann_src = join_SITE_PACKAGES('pyflann', 'lib', libflann_fname)
    #    libflann_dst = join(ibsbuild, libflann_fname)
    # This path is when pyflann was built using setup.py develop
    libflann_src = realpath(
        join(root_dir, '..', 'flann', 'build', 'lib', libflann_fname)
    )
    libflann_dst = join(ibsbuild, 'pyflann', 'lib', libflann_fname)
    DATATUP_LIST.append((libflann_dst, libflann_src))

    # VTool
    vtool_libs = ['libsver']
    for libname in vtool_libs:
        lib_fname = libname + LIB_EXT
        vtlib_src = realpath(
            join(root_dir, '..', 'vtool_ibeis', 'vtool_ibeis', lib_fname)
        )
        vtlib_dst = join(ibsbuild, 'vtool_ibeis', lib_fname)
        DATATUP_LIST.append((vtlib_dst, vtlib_src))

    linux_lib_dpaths = ['/usr/lib/x86_64-linux-gnu', '/usr/lib', '/usr/local/lib']

    # OpenMP
    if APPLE:
        # BSDDB, Fix for the modules that PyInstaller needs and (for some reason)
        # are not being added by PyInstaller
        libbsddb_src = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload/_bsddb.so'
        libbsddb_dst = join(ibsbuild, '_bsddb.so')
        DATATUP_LIST.append((libbsddb_dst, libbsddb_src))
        # libgomp_src = '/opt/local/lib/libgomp.dylib'
        # libgomp_src = '/opt/local/lib/gcc48/libgomp.dylib'
        libgomp_src = '/opt/local/lib/libgcc/libgomp.1.dylib'
        BINARYTUP_LIST.append(('libgomp.1.dylib', libgomp_src, 'BINARY'))

        # very hack
        # libiomp_src = '/Users/bluemellophone/code/libomp_oss/exports/mac_32e/lib.thin/libiomp5.dylib'
        # BINARYTUP_LIST.append(('libiomp5.dylib', libiomp_src, 'BINARY'))

    if LINUX:
        libgomp_src = ut.search_in_dirs('libgomp.so.1', linux_lib_dpaths)
        ut.assertpath(libgomp_src)
        BINARYTUP_LIST.append(('libgomp.so.1', libgomp_src, 'BINARY'))

    # MinGW
    if WIN32:
        mingw_root = r'C:\MinGW\bin'
        mingw_dlls = [
            'libgcc_s_dw2-1.dll',
            'libstdc++-6.dll',
            'libgomp-1.dll',
            'pthreadGC2.dll',
        ]
        for lib_fname in mingw_dlls:
            lib_src = join(mingw_root, lib_fname)
            lib_dst = join(ibsbuild, lib_fname)
            DATATUP_LIST.append((lib_dst, lib_src))

    # We need to add these 4 opencv libraries because pyinstaller does not find them.
    # OPENCV_EXT = {'win32': '248.dll',
    #              'darwin': '.2.4.dylib',
    #              'linux2': '.so.2.4'}[PLATFORM]

    target_cv_version = '3.1.0'

    OPENCV_EXT = {
        'win32': target_cv_version.replace('.', '') + '.dll',
        'darwin': '.' + target_cv_version + '.dylib',
        'linux2': '.so.' + target_cv_version,
    }[PLATFORM]

    missing_cv_name_list = [
        'libopencv_videostab',
        'libopencv_superres',
        'libopencv_stitching',
        #'libopencv_gpu',
        'libopencv_core',
        'libopencv_highgui',
        'libopencv_imgproc',
    ]
    # Hack to find the appropriate opencv libs
    for name in missing_cv_name_list:
        fname = name + OPENCV_EXT
        src = ''
        dst = ''
        if APPLE:
            src = join('/opt/local/lib', fname)
        elif LINUX:
            # src = join('/usr/lib', fname)
            src, tried = ut.search_in_dirs(
                fname, linux_lib_dpaths, strict=True, return_tried=True
            )
        elif WIN32:
            if ut.get_computer_name() == 'Ooo':
                src = join(r'C:/Program Files (x86)/OpenCV/x86/mingw/bin', fname)
            else:
                src = join(root_dir, '../opencv/build/bin', fname)
        dst = join(ibsbuild, fname)
        # ut.assertpath(src)
        DATATUP_LIST.append((dst, src))

    ##################################
    # QT Gui dependencies
    ##################################
    if APPLE:
        walk_path = '/opt/local/Library/Frameworks/QtGui.framework/Versions/4/Resources/qt_menu.nib'
        for root, dirs, files in os.walk(walk_path):
            for lib_fname in files:
                toc_src = join(walk_path, lib_fname)
                toc_dst = join('qt_menu.nib', lib_fname)
                DATATUP_LIST.append((toc_dst, toc_src))

    ##################################
    # Documentation, Icons, and Web Assets
    ##################################
    # Documentation
    # userguide_dst = join('.', '_docs', 'IBEISUserGuide.pdf')
    # userguide_src = join(root_dir, '_docs', 'IBEISUserGuide.pdf')
    # DATATUP_LIST.append((userguide_dst, userguide_src))

    # Icon File
    ICON_EXT = {'darwin': '.icns', 'win32': '.ico', 'linux2': '.ico'}[PLATFORM]
    iconfile = join('_installers', 'ibsicon' + ICON_EXT)
    icon_src = join(root_dir, iconfile)
    icon_dst = join(ibsbuild, iconfile)
    DATATUP_LIST.append((icon_dst, icon_src))

    print('[installer] Checking Data (preweb)')
    try:
        for (dst, src) in DATATUP_LIST:
            assert ut.checkpath(src, verbose=True), 'checkpath for src=%r failed' % (
                src,
            )
    except Exception as ex:
        ut.printex(ex, 'Checking data failed DATATUP_LIST=' + ut.repr2(DATATUP_LIST))
        raise

    # Web Assets
    INSTALL_WEB = True and not ut.get_argflag('--noweb')
    if INSTALL_WEB:
        web_root = join('wbia', 'web/')
        # walk_path = join(web_root, 'static')
        # static_data = []
        # for root, dirs, files in os.walk(walk_path):
        #    root2 = root.replace(web_root, '')
        #    for icon_fname in files:
        #        if '.DS_Store' not in icon_fname:
        #            toc_src = join(abspath(root), icon_fname)
        #            toc_dst = join(root2, icon_fname)
        #            static_data.append((toc_dst, toc_src))
        # ut.get_list_column(static_data, 1) == ut.glob(walk_path, '*', recursive=True, with_dirs=False, exclude_dirs=['.DS_Store'])
        static_src_list = ut.glob(
            join(web_root, 'static'),
            '*',
            recursive=True,
            with_dirs=False,
            exclude_dirs=['.DS_Store'],
        )
        static_dst_list = [
            relpath(src, join(root_dir, 'wbia')) for src in static_src_list
        ]
        static_data = zip(static_dst_list, static_src_list)
        DATATUP_LIST.extend(static_data)

        # walk_path = join(web_root, 'templates')
        # template_data = []
        # for root, dirs, files in os.walk(walk_path):
        #    root2 = root.replace(web_root, '')
        #    for icon_fname in files:
        #        if '.DS_Store' not in icon_fname:
        #            toc_src = join(abspath(root), icon_fname)
        #            toc_dst = join(root2, icon_fname)
        #            template_data.append((toc_dst, toc_src))
        template_src_list = ut.glob(
            join(web_root, 'templates'),
            '*',
            recursive=True,
            with_dirs=False,
            exclude_dirs=['.DS_Store'],
        )
        template_dst_list = [
            relpath(src, join(root_dir, 'wbia')) for src in template_src_list
        ]
        template_data = zip(template_dst_list, template_src_list)
        DATATUP_LIST.extend(template_data)

    print('[installer] Checking Data (postweb)')
    try:
        for (dst, src) in DATATUP_LIST:
            assert ut.checkpath(src, verbose=False), 'checkpath for src=%r failed' % (
                src,
            )
    except Exception as ex:
        ut.printex(ex, 'Checking data failed DATATUP_LIST=' + ut.repr2(DATATUP_LIST))
        raise

    return DATATUP_LIST, BINARYTUP_LIST, iconfile


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/wbia/_installers/wbia_pyinstaller_data_helper.py --test-get_data_list
        python ~/code/wbia/_installers/wbia_pyinstaller_data_helper.py
        python ~/code/wbia/_installers/wbia_pyinstaller_data_helper.py --allexamples
        python ~/code/wbia/_installers/wbia_pyinstaller_data_helper.py --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
