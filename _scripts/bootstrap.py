from util_cplat_packages import upgrade, ensure_packages

NON_PYTHON_PREREQ = [
    'git',
    'cmake',
    'g++',
    'ffmpeg',
    'opencv',
    'libpng',
    'zlib',
    'freetype',
    'freetype',
]

PYTHON_PREREQ = [
    'setuptools',
    'pip',
    'Pygments',
    'six',
    'openpyxl',
    'dateutil',
    'readline',
    'sip',
    'pyqt4',
    'Pillow',
    'zmq',
    'pyzmq',
    'numpy',
    'scipy',
    'ipython',
    'pandas',
    'matplotlib'
]

upgrade()
ensure_packages(NON_PYTHON_PREREQ)
