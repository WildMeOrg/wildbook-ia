#!/usr/bin/env python
from util_cplat_packages import upgrade, ensure_packages, ensure_python_packages, APPLE, UBUNTU

NON_PYTHON_PREREQ = [
    'git',
    'cmake',
    'gcc',  # need a C compiler for numpy
    'g++',
    'gfortran',  # need a fortran compiler for numpy (avoid mixing g77 and gfortran!)
    'ffmpeg',
    #'zmq',
]

if APPLE:
    NON_PYTHON_PREREQ.extend([
        'opencv',
        'libpng',
        'zlib',
        'freetype',
    ])

if UBUNTU:
    NON_PYTHON_PREREQ.extend([
        #'libeigen3-dev',
        'libatlas-base-dev',  # ATLAS for numpy no UBUNTU
        #'libatlas3gf-sse2',  # ATLAS SSE2 for numpy no UBUNTU
        'libfreetype6-dev',  # for matplotlib
        'libpng-dev',
    ])

PYTHON_PREREQ = [
    #'distribute',
    #'setuptools',
    'pip',
    'Pygments',
    'six',
    #'openpyxl',
    'dateutils',
    'pyreadline',
    'pyparsing',
    'sip',
    'pyqt4',
    'Pillow',
    #'pyzmq',
    'numpy',
    'scipy',
    'ipython',
    #'pandas',
    'matplotlib'
    'tornado'
    'matplotlib'
]

#upgrade()
ensure_packages(NON_PYTHON_PREREQ)
ensure_python_packages(PYTHON_PREREQ)
