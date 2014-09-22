#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import six
from utool import util_setup
from setuptools import setup


CHMOD_PATTERNS = [
    'run_tests.sh',
    'test_*.py',
    'ingest_*.py',
]

PROJECT_DIRS = ['.', 'guitool', 'plotool', 'ibeis', 'ibeis/control',
                'ibeis/dev', 'ibeis/gui', 'ibeis/ingest', 'ibeis/model',
                'ibeis/hots', 'ibeis/preproc', 'ibeis/viz',
                'ibeis/viz/interact', ]

CLUTTER_PATTERNS = [
    '\'',
    '*.pyc',
    '*.pyo',
    '*.dump.txt',
    '*.sqlite3',
    '*.prof',
    '*.prof.txt',
    '*.lprof',
    '*.ln.pkg',
    '*.egg-info',
    'failed.txt',
    '_test_times.txt',
    '_*_cyth.o',
    '_*_cyth_bench.py',
    'run_cyth_benchmarks.sh',
    '_*_cyth.c',
    '_*_cyth.pyd',
    '_*_cyth.pxd',
    '_*_cyth.html',
    '_*_cyth.pyx',
    '_*_cyth.so',
    '_*_cyth.dylib'
]

CLUTTER_DIRS = [
    'logs/',
    'dist/',
    '__pycache__/',
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
    'scikit-learn >= 0.15.2',
    #'objgraph',
    #'pycallgraph',
    #'gevent',
    #'PyQt 4/5 >= 4.9.1', # cannot include because pyqt4 is not in pip
]

if six.PY2:
    INSTALL_REQUIRES.append('requests >= 0.8.2')


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    kwargs = util_setup.setuptools_setup(
        setup_fpath=__file__,
        name='ibeis',
        author='Jon Crall',
        author_email='erotemic@gmail.com',
        packages=util_setup.find_packages(),
        version=util_setup.parse_package_for_version('ibeis'),
        license=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        ext_modules=util_setup.find_ext_modules(),
        cmdclass=util_setup.get_cmdclass(),
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_patterns=CLUTTER_PATTERNS,
        clutter_dirs=CLUTTER_DIRS,
        install_requires=INSTALL_REQUIRES
        #cython_files=CYTHON_FILES,
    )
    setup(**kwargs)
