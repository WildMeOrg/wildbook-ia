# -*- mode: python -*-
import os
import sys
from os.path import join, exists


def join_SITE_PACKAGES(*args):
    import site
    from os.path import join, exists
    tried_list = []
    for dir_ in site.getsitepackages():
        path = join(dir_, *args)
        tried_list.append(path)
        if exists(path):
            return path
    msg = ('Cannot find: join_SITE_PACKAGES(*%r)\n'  % (args,))
    msg += 'Tried: \n    ' + '\n    '.join(tried_list)
    print(msg)
    raise Exception(msg)


def add_data(a, dst, src):
    import textwrap
    from os.path import dirname, normpath, splitext
    global LIB_EXT

    def fixwin32_shortname(path1):
        import ctypes
        try:
            #import win32file
            #buf = ctypes.create_unicode_buffer(buflen)
            path1 = unicode(path1)
            buflen = 260  # max size
            buf = ctypes.create_unicode_buffer(buflen)
            ctypes.windll.kernel32.GetLongPathNameW(path1, buf, buflen)
            #win32file.GetLongPathName(path1, )
            path2 = buf.value
        except Exception as ex:
            path2 = path1
            print(ex)
        return path2

    def platform_path(path):
        path1 = normpath(path)
        if sys.platform == 'win32':
            path2 = fixwin32_shortname(path1)
        else:
            path2 = path1
        return path2

    src = platform_path(src)
    dst = dst
    if not os.path.exists(dirname(dst)) and dirname(dst) != "":
        os.makedirs(dirname(dst))
    pretty_path = lambda str_: str_.replace('\\', '/')
    # Default datatype is DATA
    dtype = 'DATA'
    # Infer datatype from extension
    extension = splitext(dst)[1].lower()
    if extension == LIB_EXT.lower():
        dtype = 'BINARY'
    print(textwrap.dedent('''
    [setup] a.add_data(
    [setup]    dst=%r,
    [setup]    src=%r,
    [setup]    dtype=%s)''').strip('\n') %
          (pretty_path(dst), pretty_path(src), dtype))
    a.datas.append((dst, src, dtype))


##################################
# System Variables
##################################
PLATFORM = sys.platform
APPLE = PLATFORM == 'darwin'
WIN32 = PLATFORM == 'win32'
LINUX = PLATFORM == 'linux2'

LIB_EXT = {'win32': '.dll',
           'darwin': '.dylib',
           'linux2': '.so'}[PLATFORM]

##################################
# Asserts
# This needs to be relative to build directory. Leave as is.
# run from root
##################################
ibsbuild = ''
root_dir = os.getcwd()
try:
    assert exists(join(root_dir, 'installers.py'))
    assert exists('../ibeis')
    assert exists('../ibeis/ibeis')
    assert exists(root_dir)
except AssertionError:
    raise Exception('installers.py must be run from ibeis root')

##################################
# Explicitly add modules in case they are not in the Python PATH
##################################
modules = ['utool', 'vtool', 'guitool', 'plottool', 'pyrf', 'pygist', 'ibeis', 'hesaff', 'detecttools']

apple = []
if APPLE:
    # We need to explicitly add the MacPorts and system Python site-packages folders on Mac
    apple.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/')
    apple.append('/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/')

a = Analysis(
    ['main.py'],
    pathex=['.'] + [ join('..', module) for module in modules ] + apple,
    hiddenimports=[
        'sklearn.utils.sparsetools._graph_validation',
        'sklearn.utils.sparsetools._graph_tools',
        'scipy.special._ufuncs_cxx',
        'sklearn.utils.lgamma',
        'sklearn.utils.weight_vector',
        'sklearn.neighbors.typedefs',
        'mpl_toolkits.axes_grid1'
    ],
    hookspath=None,
    runtime_hooks=None
)
# IF MPL FAILS:
# MPL has a problem where the __init__.py is not created in the library.  touch __init__.py in the module's path should fix the issue

##################################
# Specify Data in TOC (table of contents) format (SRC, DEST, TYPE)
##################################
src = join(root_dir, '_installers/ibsicon.ico')
dst = join(ibsbuild, '_installers/ibsicon.ico')
add_data(a, dst, src)

src = join(root_dir, '_installers/resources_MainSkel.qrc')
dst = join(ibsbuild, '_installers/resources_MainSkel.qrc')
add_data(a, dst, src)

##################################
# Hesaff + FLANN + PyRF Libraries
##################################
libflann_fname = 'libflann' + LIB_EXT
if WIN32:
    libflann_src = join_SITE_PACKAGES('pyflann', 'lib', libflann_fname)
    libflann_dst = join(ibsbuild, libflann_fname)
    add_data(a, libflann_dst, libflann_src)

if APPLE:
    # FLANN
    try:
        libflann_src = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pyflann/lib/libflann.dylib'
        libflann_dst = join(ibsbuild, libflann_fname)
        add_data(a, libflann_dst, libflann_src)
    except Exception as ex:
        print(repr(ex))

    # BSDDB, Fix for the modules that PyInstaller needs and (for some reason) are not being added by PyInstaller
    try:
        libbsddb_src = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload/_bsddb.so'
        libbsddb_dst = join(ibsbuild, '_bsddb.so')
        add_data(a, libbsddb_dst, libbsddb_src)
    except Exception as ex:
        print(repr(ex))

    # Hesaff
    libhesaff_fname = 'libhesaff' + LIB_EXT
    libhesaff_src = join('..', 'hesaff', 'build', libhesaff_fname)
    libhesaff_dst = join(ibsbuild, 'pyhesaff', 'lib', libhesaff_fname)
    add_data(a, libhesaff_dst, libhesaff_src)

    # PyRF
    libpyrf_fname = 'libpyrf' + LIB_EXT
    libpyrf_src = join('..', 'pyrf', 'build', libpyrf_fname)
    libpyrf_dst = join(ibsbuild, 'pyrf', 'lib', libpyrf_fname)
    add_data(a, libpyrf_dst, libpyrf_src)

    # We need to add these 4 opencv libraries because pyinstaller does not find them.
    missing_cv_name_list = [
        'libopencv_videostab.2.4',
        'libopencv_superres.2.4',
        'libopencv_stitching.2.4',
    ]
    for name in missing_cv_name_list:
        fname = name + LIB_EXT
        src = join('/opt/local/lib', fname)
        dst = join(ibsbuild, fname)
        add_data(a, dst, src)


##################################
# QT Gui dependencies
##################################
walk_path = '/opt/local/Library/Frameworks/QtGui.framework/Versions/4/Resources/qt_menu.nib'
for root, dirs, files in os.walk(walk_path):
    for lib_fname in files:
        toc_src = join(walk_path, lib_fname)
        toc_dst = join('qt_menu.nib', lib_fname)
        add_data(a, toc_dst, toc_src)

##################################
# Documentation and Icon
##################################
# Documentation
userguide_dst = join('.', '_docs', 'IBEISUserGuide.pdf')
userguide_src = join(root_dir, '_docs', 'IBEISUserGuide.pdf')
add_data(a, userguide_dst, userguide_src)

# Icon File
ICON_EXT = {'darwin': 'icns',
            'win32':  'ico',
            'linux2': 'ico'}[PLATFORM]
iconfile = join(root_dir, '_installers', 'ibsicon.' + ICON_EXT)

##################################
# Build executable
##################################
# Executable name
exe_name = {'win32':  'build/IBEISApp.exe',
            'darwin': 'build/pyi.darwin/IBEISApp/IBEISApp',
            'linux2': 'build/IBEISApp.ln'}[PLATFORM]

pyz = PYZ(a.pure)   # NOQA

exe_kwargs = {
    'console': True,
    'debug': False,
    'name': exe_name,
    'exclude_binaries': True,
    'append_pkg': False,
}

collect_kwargs = {
    'strip': None,
    'upx': True,
    'name': join('dist', 'ibeis')
}

# Windows only EXE options
if WIN32:
    exe_kwargs['icon'] = iconfile
    #exe_kwargs['version'] = 1.5


if APPLE:
    exe_kwargs['console'] = False

# Pyinstaller will gather .pyos
opt_flags = [('O', '', 'OPTION')]
exe = EXE(pyz, a.scripts + opt_flags, **exe_kwargs)   # NOQA

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, **collect_kwargs)  # NOQA

bundle_name = 'IBEIS'
if APPLE:
    bundle_name += '.app'

app = BUNDLE(coll, name=join('dist', bundle_name))  # NOQA
