# -*- mode: python -*-
import os
import sys
from os.path import join, exists, realpath, abspath  # NOQA
import utool as ut  # NOQA
# import utool

# Pyinstaller Variables (enumerated for readability, not needed)
#Analysis = Analysis  # NOQA


def add_data(a, dst, src):
    global LIB_EXT
    from os.path import dirname, exists
    import utool as ut  # NOQA
    if dst == '':
        raise ValueError('dst path cannot be the empty string')
    if src == '':
        raise ValueError('src path cannot be the empty string')
    src_ = ut.platform_path(src)
    if not os.path.exists(dirname(dst)) and dirname(dst) != "":
        os.makedirs(dirname(dst))
    _pretty_path = lambda str_: str_.replace('\\', '/')
    # Default datatype is DATA
    dtype = 'DATA'
    # Infer datatype from extension
    #extension = splitext(dst)[1].lower()
    #if extension == LIB_EXT.lower():
    if LIB_EXT[1:] in dst.split('.'):
        dtype = 'BINARY'
    print(ut.codeblock('''
    [installer] a.add_data(
    [installer]    dst=%r,
    [installer]    src=%r,
    [installer]    dtype=%s)''') %
          (_pretty_path(dst), _pretty_path(src_), dtype))
    assert exists(src_), 'src_=%r does not exist'
    a.datas.append((dst, src_, dtype))


##################################
# System Variables
##################################
PLATFORM = sys.platform
APPLE = PLATFORM.startswith('darwin')
WIN32 = PLATFORM.startswith('win32')
LINUX = PLATFORM.startswith('linux2')

LIB_EXT = {'win32': '.dll',
           'darwin': '.dylib',
           'linux2': '.so'}[PLATFORM]

##################################
# Asserts
##################################
ibsbuild = ''
root_dir = os.getcwd()
try:
    assert exists(join(root_dir, 'installers.py'))
    assert exists('../ibeis')
    assert exists('_installers')
    assert exists('../ibeis/ibeis')
    assert exists(root_dir)
except AssertionError:
    raise Exception('installers.py must be run from ibeis root')


sys.path.append('_installers')

import ibeis_pyinstaller_data_helper

# Build data before running analysis for quick debugging
pathex = ibeis_pyinstaller_data_helper.get_path_extensions()
DATATUP_LIST, BINARYTUP_LIST, iconfile = ibeis_pyinstaller_data_helper.get_data_list()
hiddenimports = ibeis_pyinstaller_data_helper.get_hidden_imports()


##################################
# Build executable
##################################
# Executable name
exe_name = {'win32':  'build/IBEISApp.exe',
            'darwin': 'build/pyi.darwin/IBEISApp/IBEISApp',
            'linux2': 'build/IBEISApp'}[PLATFORM]

#import sys
#print('exiting')
#sys.exit(1)

print('[installer] Running Analysis')
a = Analysis(  # NOQA
    #['main.py'],
    ['ibeis/__main__.py'],
    pathex=pathex,
    hiddenimports=hiddenimports,
    hookspath=None,
    runtime_hooks=[
        '_installers/rthook_pyqt4.py'
    ]
)

print('[installer] Adding %d Datatups' % (len(DATATUP_LIST,)))
for (dst, src) in DATATUP_LIST:
    add_data(a, dst, src)

print('[installer] Adding %d Binaries' % (len(BINARYTUP_LIST),))
for binarytup in BINARYTUP_LIST:
    a.binaries.append(binarytup)

print('[installer] PYZ Step')
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
    exe_kwargs['icon'] = join(root_dir, iconfile)
    #exe_kwargs['version'] = 1.5
if APPLE:
    # Console must be False for osx
    exe_kwargs['console'] = False

# Pyinstaller will gather .pyos
print('[installer] EXE Step')
opt_flags = [('O', '', 'OPTION')]
exe = EXE(pyz, a.scripts + opt_flags, **exe_kwargs)   # NOQA

print('[installer] COLLECT Step')
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, **collect_kwargs)  # NOQA

bundle_name = 'IBEIS'
if APPLE:
    bundle_name += '.app'

print('[installer] BUNDLE Step')
app = BUNDLE(coll, name=join('dist', bundle_name))  # NOQA
