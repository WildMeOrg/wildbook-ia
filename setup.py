#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool import util_setup

PROJECT_NAME = 'IBEIS'

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


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    util_setup.presetup(
        __file__,
        project_name=PROJECT_NAME,
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_dirs=CLUTTER_DIRS,
        clutter_patterns=CLUTTER_PATTERNS,
        cython_files=CYTHON_FILES,
    )
