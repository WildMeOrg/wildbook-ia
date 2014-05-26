#!/usr/bin/env python
import sys
import os
from util_cplat_packages import make_prereq_script, APPLE, UBUNTU, print_sysinfo

#git clone https://github.com/Erotemic/ibeis.git

DRYRUN = '--dry' in sys.argv or '--dryrun' in sys.argv

PREREQ_PKG_LIST = [
    'git',
    'cmake',
    'gcc',  # need a C compiler for numpy
    'g++',
    'gfortran',  # need a fortran compiler for numpy (avoid mixing g77 and gfortran!)
    'ffmpeg',
    #'zmq',
]

if APPLE:
    PREREQ_PKG_LIST.extend([
        'opencv',
        'libpng',
        'zlib',
        'freetype',
    ])

if UBUNTU:
    PREREQ_PKG_LIST.extend([
        'libfftw3-dev',
        #'libeigen3-dev',
        'libatlas-base-dev',  # ATLAS for numpy no UBUNTU
        #'libatlas3gf-sse2',  # ATLAS SSE2 for numpy no UBUNTU
        'libfreetype6-dev',  # for matplotlib
        'libpng12-dev',
        'python-dev',
    ])

PREREQ_PYPKG_LIST = [
    'pip',
    #'distribute',
    'setuptools',
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
    'tornado',
    'matplotlib',
]

# Need to do a distribute upgrade before matplotlib on Ubuntu?
# not sure if that will work yet

print_sysinfo()
#upgrade()
output = make_prereq_script(PREREQ_PKG_LIST, PREREQ_PYPKG_LIST)
if output == '':
    print('System has all prerequisites!')
elif not DRYRUN:
    filename = '__install_prereqs__.sh'
    with open(filename, 'w') as file_:
        file_.write(output)
    os.system('chmod +x ' + filename)
    print('# wrote: %r' % os.path.realpath(filename))
    #sudo python super_setup.py --build --develop
