#!/usr/bin/env python
from util_cplat_packages import upgrade, ensure_packages, ensure_python_packages, APPLE

NON_PYTHON_PREREQ = [
    'git',
    'cmake',
    'g++',
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

PYTHON_PREREQ = [
    'distribute',
    'setuptools',
    'pip',
    'Pygments',
    'six',
    #'openpyxl',
    'dateutil',
    'readline',
    'sip',
    'pyqt4',
    'Pillow',
    #'pyzmq',
    'numpy',
    'scipy',
    'ipython',
    #'pandas',
    'matplotlib'
]

#upgrade()
ensure_packages(NON_PYTHON_PREREQ)
ensure_python_packages(PYTHON_PREREQ)
