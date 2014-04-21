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
    test_dirs = [join(cwd, 'ibeis', 'tests'),
                 join(cwd, 'vtool', 'tests')]
    for test_dir in test_dirs:
        for pyscript_fpath in util_path.glob(test_dir, 'test_*.py'):
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
    # Remove logs
    util_path.remove_files_in_dir(join(cwd, 'logs'), verbose=VERBOSE)
    # Remove misc
    util_path.delete(join(cwd, "'"))  # idk where this file comes from


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    for arg in iter(sys.argv[1:]):
        # Build PyQt UI files
        if arg in ['clean']:
            clean()
            sys.exit(0)
        if arg in ['chmod']:
            chmod_tests()
        if arg in ['buildui', 'ui', 'compile_ui']:
            compile_ui()
            sys.exit(0)
