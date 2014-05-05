#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
import sys
from os.path import join, splitext, exists, split
from utool import util_path
#import stat

VERBOSE = '--verbose' in sys.argv


def chmod_tests():
    cwd = get_project_repo_dir()
    st = os.stat(__file__)
    print(st)
    test_dirs = [
        ('injest', join(cwd, 'ibeis', 'injest')),
        ('test_', join(cwd, 'vtool', 'tests')),
        ('test_', join(cwd, 'ibeis', 'tests')),
        ('test_', join(cwd, 'utool', 'tests')),
    ]
    for prefix, test_dir in test_dirs:
        for pyscript_fpath in util_path.glob(test_dir, prefix + '*.py'):
            print('[setup] chmod fpath=%r' % pyscript_fpath)
            os.chmod(pyscript_fpath, st.st_mode)


def get_project_repo_dir():
    dpath, fname = split(__file__)
    cwd = util_path.truepath(dpath)
    assert exists('setup.py'), 'must be run in ibeis directory'
    assert exists('../ibeis/setup.py'), 'must be run in ibeis directory'
    assert exists('../ibeis/ibeis/control'), 'must be run in ibeis directory'
    assert exists('_setup'), 'must be run in ibeis directory'
    assert fname == 'setup.py', 'this file is not setup.py'
    return cwd


CYTHON_FILES = [
    'vtool/chip.py',
    'vtool/image.py',
    'vtool/exif.py',
    'vtool/histogram.py',
    'vtool/ellipse.py',
    'vtool/keypoint.py',
    'vtool/linalg.py',
    'vtool/math.py',
    'vtool/patch.py',
    'vtool/segmentation.py',
    'vtool/spatial_verification.py',

    'ibeis/model/hots/QueryRequest.py',
    'ibeis/model/hots/QueryResult.py',
    'ibeis/model/hots/voting_rules2.py',
    'ibeis/model/hots/nn_filters.py',
    'ibeis/model/hots/matching_functions.py'
]

def build_cython():
    from utool.util_dev import compile_cython
    for fpath in CYTHON_FILES:
        utool.util_dev.compile_cython(fpath)


def build_pyo():
    from utool import util_cplat
    PROJECT_DIRS = [
        '.'
        'guitool',
        'plotool',
        'vtool',
        'utool'
        'ibeis',
        'ibeis/control',
        'ibeis/dev',
        'ibeis/gui',
        'ibeis/injest',
        'ibeis/model',
        'ibeis/hots',
        'ibeis/preproc',
        'ibeis/viz',
        'ibeis/viz/interact',
    ]
    for projdir in PROJECT_DIRS:
        util_cplat.shell('python -O -m compileall ' + projdir + '/*.py')
    #util_cplat.shell('python -O -m compileall ibeis/*.py')
    #util_cplat.shell('python -O -m compileall utool/*.py')
    #util_cplat.shell('python -O -m compileall vtool/*.py')
    #util_cplat.shell('python -O -m compileall plottool/*.py')
    #util_cplat.shell('python -O -m compileall guitool/*.py')


def compile_ui():
    'Compiles the qt designer *.ui files into python code'
    print('[setup] compile_ui()')
    pyuic4_cmd = {'win32':  'C:\Python27\Lib\site-packages\PyQt4\pyuic4',
                  'linux2': 'pyuic4',
                  'darwin': 'pyuic4'}[sys.platform]
    cwd = get_project_repo_dir()
    widget_dir = join(cwd, 'ibeis')
    print('[setup] Compiling qt designer files in %r' % widget_dir)
    for widget_ui in util_path.glob(widget_dir, '*.ui'):
        widget_py = splitext(widget_ui)[0] + '.py'
        cmd = ' '.join([pyuic4_cmd, '-x', widget_ui, '-o', widget_py])
        print('[setup] compile_ui()>' + cmd)
        os.system(cmd)


def clean():
    """ Cleans up temporary and compiled files in the IBEIS directory """
    print('[setup] clean()')
    cwd = get_project_repo_dir()
    print('[setup] Current working directory: %r' % cwd)
    # Remove python compiled files
    pattern_list = ['*.dump.txt', '*.sqlite3', '*.pyc', '*.pyo', '*.prof',
                    '*.prof.txt', '*.lprof', '\'']
    util_path.remove_files_in_dir(cwd, pattern_list, recursive=True, verbose=VERBOSE)
    # Remove cython compiled files
    for fpath in CYTHON_FILES:
        fname, ext = splitext(fpath)
        util_path.remove_file(fname + '.so')
        #util_path.remove_file(fname + '.dll')
        util_path.remove_file(fname + '.c')
    # Remove logs
    util_path.remove_files_in_dir(join(cwd, 'logs'), verbose=VERBOSE)
    # Remove misc
    util_path.delete(join(cwd, "'"))  # idk where this file comes from


if __name__ == '__main__':
    import utool
    utool.inject_colored_exceptions()
    print('[setup] Entering IBEIS setup')
    for arg in iter(sys.argv[1:]):

        # Build PyQt UI files
        if arg in ['clean']:
            clean()
            sys.exit(0)

        if arg in ['clean']:
            clean()
            sys.exit(0)

        # Build PyQt UI files
        if arg in ['buildui', 'ui', 'compile_ui']:
            compile_ui()
            sys.exit(0)

        # Build optimized files
        if arg in ['o', 'pyo']:
            build_pyo()

        if arg in ['c', 'cython']:
            build_cython()

        if arg in ['chmod']:
            chmod_tests()
