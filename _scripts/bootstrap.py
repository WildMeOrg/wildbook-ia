#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
git config --global push.default current
export CODE_DIR=~/code
mkdir $CODE_DIR
cd $CODE_DIR
git clone https://github.com/Erotemic/ibeis.git
cd ibeis
./_scripts/bootstrap.py
./_scripts/__install_prereqs__.sh
./super_setup.py --build --develop
./super_setup.py --build --develop
"""

"""
pip list --outdated

python -c
sudo apt-get remove python-pip
sudo apt-get remove python-setuptools
sudo apt-get remove python-numpy
sudo apt-get remove python-scipy
sudo pip install numpy --upgrade
sudo pip install scipy --upgrade


# sudo pip install distribute --upgrade
sudo pip remove PIL
sudo pip remove pillow
sudo pip install pillow



pip uninstall utool -y
pip uninstall vtool -y
pip uninstall pyhesaff -y
pip uninstall detectools -y
pip uninstall pyrf -y
pip uninstall guitool -y
pip uninstall plottool -y

cd ~/code/utool
python setup.py install
cd ~/code/hesaff
python setup.py install
cd ~/code/vtool
python setup.py install
cd ~/code/hesaff
python setup.py install
cd ~/code/detecttools/
python setup.py install
cd ~/code/pyrf
python setup.py install
cd ~/code/guitool
python setup.py install
cd ~/code/plottool
python setup.py install

CommandLine:
    # upgrade pip packages
    ./_scripts/bootstrap.py --no-syspkg --upgrade


"""
import sys
import os
from os.path import dirname, realpath, join


DRYRUN = '--dry' in sys.argv or '--dryrun' in sys.argv
OPTIONAL = '--optional' in sys.argv
UPGRADE = '--upgrade' in sys.argv or '-U' in sys.argv
WITH_SYSPKG = '--no-syspkg' not in sys.argv


def import_module_from_fpath(module_fpath):
    """ imports module from a file path """
    import platform
    from os.path import basename, splitext
    python_version = platform.python_version()
    modname = splitext(basename(module_fpath))[0]
    if python_version.startswith('2'):
        import imp
        module = imp.load_source(modname, module_fpath)
    elif python_version.startswith('3'):
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader(modname, module_fpath)
        module = loader.load_module()
    else:
        raise AssertionError('invalid python version')
    return module

try:
    import util_cplat_packages
    #from util_cplat_packages import make_prereq_script, APPLE, FEDORA_FAMILY, DEBIAN_FAMILY, print_sysinfo
except ImportError as ex:
    module_fpath = os.path.abspath(join(dirname(__file__), 'util_cplat_packages.py'))
    util_cplat_packages = import_module_from_fpath(module_fpath)


def bootstrap_sysreq(dry=DRYRUN, justpip=False, with_optional=OPTIONAL):
    PREREQ_PKG_LIST = [
        'git',
        'gcc',  # need a C compiler for numpy
        'g++',
        # g++ may need to be removed for apple
        'gfortran',  # need a fortran compiler for numpy (avoid mixing g77 and gfortran!)
        # gfortran may need to be removed for apple
        'cmake',
        'ffmpeg',  # need -dev / -devel versions of all these as well / libav
        'libpng',
        'libjpg',
        'libtiff',  # 'libtiff4-dev', libtiff5-dev
        'littlecms',  # libcms?
        'openjpeg',
        'zlib-dev',
        'freetype',
        'fftw3',
        'atlas',
        'libgeos-dev',  # for shapely
        'python-qt4',
        #'jasper',  # hyrule cannot handle this
        #'zmq',
        #libgeos-dev
    ]

    if util_cplat_packages.APPLE:
        PREREQ_PKG_LIST.extend([
            'opencv',
            'libpng',
            'zlib',
            'freetype',
        ])

    if util_cplat_packages.DEBIAN_FAMILY:
        PREREQ_PKG_LIST.extend([
            'libfftw3-dev',
            #'libeigen3-dev',
            'libatlas-base-dev',  # ATLAS for numpy no UBUNTU
            #'libatlas3gf-sse2',  # ATLAS SSE2 for numpy no UBUNTU
            'libfreetype6-dev',  # for matplotlib
            #'libpng12-dev',
            #'libjpeg-dev',
            #'zlib1g-dev',
            'python-dev',
            'libopencv-dev',  # Do we need these?
            'python-opencv',  # Do we really need these~these?
        ])

    if util_cplat_packages.FEDORA_FAMILY:
        PREREQ_PKG_LIST.extend([
            'python-dev',
        ])
        pass

    PREREQ_PYPKG_LIST = [
        'pip',
        'setuptools',
        'Pygments',
        'Cython',
        # 'requests==2.5.1',
        'requests',
        'colorama',
        'psutil',
        'functools32',
        'six',
        'dateutils',
        'pyreadline',
        'pyparsing',
        #'sip',
        #'PyQt4',
        'Pillow',
        'numpy',
        'scipy',
        'ipython',
        'tornado',
        'flask',
        'flask-cors',
        'matplotlib',
        'scikit-learn',
        'parse',
        'simplejson',
        # 'pyinstaller',
        'statsmodels',
        'lockfile',  # Need to do upgrade on this
        'lru-dict',
        'shapely',
    ]

    OPTIONAL_PYPKG_LIST = [
        #'pandas',
        'Sphinx',
        'astor',
        'autopep8',
        'flake8',
        'guppy',
        'functools32',
        'argparse',
        'h5py',
        'memory-profiler',
        'objgraph',
        'openpyxl',
        'pyfiglet',
        'pyflakes',
        'pyreadline',
        'pyzmq',
        'scikit-image',
        'sphinxcontrib-napoleon',
        'virtualenv',
    ]

    if with_optional:
        PREREQ_PYPKG_LIST += OPTIONAL_PYPKG_LIST

    #http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/
    '''
    # Install mpl_toolkits.basemap
    sudo apt-get install libgeos-dev -y
    cd ~/tmp
    wget http://downloads.sourceforge.net/project/matplotlib/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz
    gunzip -c basemap-1.0.7.tar.gz | tar xvf -
    cd basemap-1.0.7
    sudo checkinstall sudo python setup.py install
    python -c "from mpl_toolkits.basemap import Basemap"
    '''

    # Need to do a distribute upgrade before matplotlib on Ubuntu?
    # not sure if that will work yet

    util_cplat_packages.print_sysinfo()
    #upgrade()
    with_sysfix = True
    with_syspkg = WITH_SYSPKG
    with_pypkg  = True
    output = util_cplat_packages.make_prereq_script(
        PREREQ_PKG_LIST, PREREQ_PYPKG_LIST, with_sysfix,
        with_syspkg, with_pypkg, upgrade=UPGRADE)
    if output == '':
        print('System has all prerequisites!')
    elif not DRYRUN:
        script_dir = realpath(dirname(__file__))
        fpath = join(script_dir, '__install_prereqs__.sh')
        with open(fpath, 'w') as file_:
            file_.write(output)
        os.system('chmod +x ' + fpath)
        print('# wrote: %r' % fpath)
        #sudo python super_setup.py --build --develop


def install_vtk():
    pass
    #sudo apt-get install python-vtk
    #import utool as ut
    #linux_url = 'http://www.vtk.org/files/release/6.1/vtkpython-6.1.0-Linux-64bit.tar.gz'
    #zipped_url = linux_url
    #vtk_fpath = ut.grab_zipped_url(linux_url)
    #vtk_dir = '/home/joncrall/.config/utool/vtkpython-6.1.0-Linux-64bit'

if __name__ == '__main__':
    bootstrap_sysreq()
