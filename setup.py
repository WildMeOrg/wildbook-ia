#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import six
from utool.util_setup import setuptools_setup
from setuptools import setup


CHMOD_PATTERNS = [
    'run_tests.sh',
    'test_*.py',
    'ingest_*.py',
]

CYTHON_FILES = [
    'ibeis/model/hots/hots_query_result.py.py',
    'ibeis/model/hots/hots_query_request.py.py',
    'ibeis/model/hots/voting_rules2.py',
    'ibeis/model/hots/nn_filters.py',
    'ibeis/model/hots/matching_functions.py',
]


PROJECT_DIRS = [
    '.',
    'guitool',
    'plotool',
    'ibeis',
    'ibeis/control',
    'ibeis/dev',
    'ibeis/gui',
    'ibeis/ingest',
    'ibeis/model',
    'ibeis/hots',
    'ibeis/preproc',
    'ibeis/viz',
    'ibeis/viz/interact',
]

CLUTTER_PATTERNS = [
    'failed.txt',
    '*.pyc',
    '*.pyo',
    '*.dump.txt',
    '*.sqlite3',
    '*.prof',
    '*.prof.txt',
    '*.lprof',
    '\'',
    '*.ln.pkg',
    '*.egg-info',
    'test_times.txt',
    'logs/',
    '__pycache__/',
    'dist/'
]

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
    'psutil',
    'requests >= 0.8.2',
    'setproctitle >= 1.1.8',
    'objgraph',
    'gevent',
    #'PyQt 4/5 >= 4.9.1', # cannot include because pyqt4 is not in pip
]

if six.PY2:
    INSTALL_REQUIRES.append('requests >= 0.8.2')


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    kwargs = setuptools_setup(
        setup_fpath=__file__,
        name='ibeis',
        packages=['ibeis', 'ibeis.dev', 'ibeis.gui', 'ibeis.model',
                  'ibeis.tests', 'ibeis.model.detect', 'ibeis.model.preproc',
                  'ibeis.model.hots'],
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_patterns=CLUTTER_PATTERNS,
        install_requires=INSTALL_REQUIRES
        #cython_files=CYTHON_FILES,
    )
    setup(**kwargs)
