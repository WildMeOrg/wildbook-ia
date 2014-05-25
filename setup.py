#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup


CHMOD_PATTERNS = [
    'run_tests.sh',
    'test_*.py',
    'injest_*.py',
]

CYTHON_FILES = [
    'ibeis/model/hots/QueryRequest.py',
    'ibeis/model/hots/QueryResult.py',
    'ibeis/model/hots/voting_rules2.py',
    'ibeis/model/hots/nn_filters.py',
    'ibeis/model/hots/matching_functions.py'
]


PROJECT_DIRS = [
    '.',
    'guitool',
    'plotool',
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

CLUTTER_PATTERNS = [
    '*.pyc',
    '*.pyo',
    '*.dump.txt',
    '*.sqlite3',
    '*.prof',
    '*.prof.txt'
    '*.lprof',
    '\''
]

CLUTTER_DIRS = ['logs']

INSTALL_REQUIRES = [
    'utool >= 1.0.0.dev1',
    'vtool >= 1.0.0.dev1',
    'pyhesaff >= 1.0.0.dev1',
    'pyrf >= 1.0.0.dev1',
    'guitool >= 1.0.0.dev1',
    'plottool >= 1.0.0.dev1',
    'matplotlib >= 1.3.1',
    'scipy >= 0.13.2',
    'numpy >= 1.8.0',
    'Pillow >= 2.4.0',
    'functools32 >= 3.2.3-1',
    'psutil',
    #'PyQt4 >= 4.9.1', # cannot include because pyqt4 is not in pip
]

if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    setuptools_setup(
        setup_fpath=__file__,
        name='ibeis',
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_dirs=CLUTTER_DIRS,
        clutter_patterns=CLUTTER_PATTERNS,
        install_requires=INSTALL_REQUIRES
        #cython_files=CYTHON_FILES,
    )
