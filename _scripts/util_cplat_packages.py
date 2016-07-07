# HELPS GENERATE CROSS PLATFORM INSTALL SCRIPTS
# -*- coding: utf-8 -*-
"""
TODO:
    needs a big cleanup.
    Remove global variables, use functions instead.
    Make an easy, versionable, and overrideable set of aliases.

CommandLine:
    ib
    python ./_scripts/util_cplat_packages.py
    python ./_scripts/util_cplat_packages.py --os darwin
    python ./_scripts/util_cplat_packages.py --os linux --distro Ubuntu --distro_version 15.04

    python ./_scripts/bootstrap.py --os darwin
    python ./_scripts/bootstrap.py --os centos
    python ./_scripts/bootstrap.py --os win32
    python ./_scripts/bootstrap.py --os ubuntu --distro_version 15.04

"""
import sys
import platform
import textwrap

# Behavior variables

APTDCON = '--aptdcon' in sys.argv
UPGRADE_PIP     = '--upgrade' in sys.argv
CHECK_INSTALLED = '--exhaustive' in sys.argv
CRASH_ON_FAIL   = True


def parse_args():
    arg_dict = {}
    key = None
    for arg in sys.argv:
        if arg.startswith('--'):
            key = arg.replace('--', '').lower().replace('-', '_')
            arg_dict[key] = []
        elif key is not None:
            arg_dict[key].append(arg)
    return arg_dict

ARG_DICT = parse_args()


# GET SYSTEM CONFIGURATION (OR DESIRED CONFIGURATION)

if 'os' in ARG_DICT:
    __OS__ = ''.join(ARG_DICT['os'])
    distro         = ''.join(ARG_DICT.get('distro', []))
    distro_version = ''.join(ARG_DICT.get('distro_version', []))
    WIN32         = False
    MACPORTS, APPLE = False, False
    LINUX, DEBIAN_FAMILY, FEDORA_FAMILY, ARCH = False, False, False, False
    if __OS__.lower() in ['apple', 'darwin', 'mac']:
        MACPORTS = True
        APPLE    = True
    elif __OS__.lower() in ['debian', 'ubuntu', 'linuxmint']:
        if distro == '':
            distro = 'Ubuntu'
        LINUX         = True
        DEBIAN_FAMILY = True
    elif __OS__.lower() in ['centos']:
        if distro == '':
            distro = 'Centos'
        LINUX    = True
        FEDORA_FAMILY   = True
    elif __OS__.lower() in ['arch']:
        LINUX    = True
        ARCH     = True
    elif __OS__.lower() in ['win', 'win32']:
        WIN32    = True
else:
    __OS__ = sys.platform
    # References: https://docs.python.org/2/library/platform.html#platform.linux_distribution

    APPLE = __OS__.startswith('darwin')
    WIN32 = __OS__.startswith('win32')
    LINUX = __OS__.startswith('linux')

    # FIXME: platform.dist is depricated
    if LINUX:
        KNOWN_LINUX_DISTS = ('SuSE', 'debian', 'fedora', 'redhat', 'centos',
                             'mandrake', 'mandriva', 'rocks', 'slackware',
                             'yellowdog', 'gentoo', 'UnitedLinux', 'turbolinux',
                             'Ubuntu')
        (distro, distro_version, distro_tag) = platform.dist()
        #(distro, distro_version, distro_tag) = platform.linux_distribution()
    elif WIN32:
        (distro, distro_version, distro_tag) = platform.dist()
        #(release, version, csd, ptype) = platform.win32_ver()
    elif APPLE:
        (distro, distro_version, distro_tag) = platform.dist()
        #release, versioninfo, machine = platform.mac_ver()
    #platform.dist()

    DEBIAN_FAMILY = (distro.lower() in ['ubuntu', 'debian', 'linuxmint'])
    FEDORA_FAMILY = (distro.lower() in ['centos', 'fedora', 'redhat', 'yellowdog', 'turbolinux'])
    ARCH = (distro == 'arch')
    if FEDORA_FAMILY:
        WIN32 = False
        LINUX = True
        APPLE = False
    MACPORTS = APPLE  # We force macports right now


def version_ge(version1, version2):
    """
    >>> from util_cplat_packages import *
    >>> version1 = distro_version
    >>> version2 = '15.03'
    """
    import distutils.version
    flag = distutils.version.LooseVersion(version1) >= distutils.version.LooseVersion(version2)
    return flag


# PRINT WHAT WE ARE WORKING WITH
def print_sysinfo():
    """
    >>> from util_cplat_packages import *
    """
    print('# sysinfo: (%s, %s, %s) ' % (__OS__, distro, distro_version,))


# TARGET PYTHON PLATFORM
TARGET_PY_VERSION = '2.7.6'
MACPORTS_PY_SUFFIX = TARGET_PY_VERSION.replace('.', '')[0:2]
MACPORTS_PY_PREFIX = 'py' + MACPORTS_PY_SUFFIX + '-'


def get_pip_installed():
    try:
        import pip
        pypkg_list = [item.key for item in pip.get_installed_distributions()]
        return pypkg_list
    except ImportError:
        #out, err, ret = shell('pip list')
        #if ret == 0:
        #    pypkg_list = [_.split(' ')[0] for  _ in out.split('\n')]
        return []


def _std_pkgwrap(pkg):
    if DEBIAN_FAMILY:
        return 'lib' + pkg + '-dev'
    if FEDORA_FAMILY:
        return pkg + '-devel'

# Special cases
# Different package managers name different packages different things. This is
# an attempt to locally stanadardize them

APPLE_PYPKG_MAP = {
    'dateutils'    : 'dateutil',
    'pyreadline'   : 'readline',
    'pyparsing'    : 'parsing',
}

# Need to use pip for these
APPLE_NONPORTS_PYPKGS = [
]


MACPORTS_PKGMAP = {
    'gcc'                  : None,
    'g++'                  : None,
    'gfortran'             : None,
    'libjpg'               : 'jpeg',
    'libjpeg'              : 'jpeg',
    'libtiff'              : 'tiff',
    'fftw3'                : 'fftw-3',
    'python-pyqt4'         : 'py27-pyqt4',
    'littlecms'            : 'lcms',
    'libhdf5-dev'          : 'hdf5',
    'libeigen2-dev'        : None,
    'libeigen3-dev'        : 'eigen3',
    'graphviz-dev'         : None,
    'libgraphviz-dev'      : None,
    'zlib-dev'             : 'zlib',
    'libgeos-dev'          : 'geos',
    'atlas'                : None,
    'py27-ndg-httpsclient' : None,
    'py27-qt4'             : 'py27-pyqt4',
    'py27-tk'              : 'py27-tkinter',
    'pkg-config'           : 'pkgconfig',
    'libffi-dev'           : 'libffi',
    'libssl-dev'           : 'openssl',
    'py27-pyopenssl'       : 'py27-openssl',
    'py27-pyasn1'          : 'py27-asn1',
}


MACPORTS_PYPKG_IGNORE_LIST = [
    'flask-cas',
    'flask-cors',
    'parse',
    'lru-dict',
    'pyfiglet',
]


APT_GET_PKGMAP = {
    'python-pyqt4' : 'python-qt4',
    'zlib-dev'     : 'zlib1g-dev',
    'libjpg'       : 'libjpeg-dev',
    'libjpeg'      : 'libjpeg-dev',
    'libpng'       : 'libpng12-dev',
    'libtiff'      : 'libtiff5-dev',
    'openjpeg'     : 'libopenjpeg-dev',
    'freetype'     : 'libfreetype6-dev',
    'atlas'        : 'libatlas-base-dev',
    'fftw3'        : 'libfftw3-dev',
    'openssl'      : 'libopenssl-devel',
    'ffmpeg'       : 'libav-tools',
    'littlecms'    : 'liblcms1-dev',
}

YUM_PKGMAP = {
    'g++'        : 'gcc-c++',
    'gfortran'   : 'gcc-gfortran',
    'ffmpeg'     : 'ffmpeg-devel',
    'libpng'     : 'libpng-devel',
    'zlib-dev'   : 'zlib-devel',
    'libjpg'     : 'libjpeg-devel',
    'freetype'   : 'freetype-devel',
    'fftw3'      : 'fftw3-devel',
    'atlas'      : 'atlas-devel',
    'python-dev' : 'python-devel',
    'libgeos-dev' : 'geos-devel',
}

PACMAN_PKGMAP = {
    'g++'       : 'gcc',
    'gfortran'  : 'gcc-fortran',
    'libjpg'    : 'libjpeg-turbo',
    'littlecms' : 'lcms',
    'freetype'  : 'freetype2',
    'fftw3'     : 'fftw',
    'atlas'     : '$AUR atlas-lapack'  # atlas isn't in the main repositories, $AUR will be interpreted to tell the user to install the package from AUR
}


# Additional hacks to the package maps
if distro == 'Ubuntu' and version_ge(distro_version, '15.04'):
    APT_GET_PKGMAP['littlecms'] = 'liblcms2-dev'


def _fix_yum_repos():
    # This puts a config file to tell yum about the ffmpeg repos
    # The $ signs are escaped here. They wont be in the file that is
    # written
    make_dag_repo_command = r'''
sudo sh -c 'cat > /etc/yum.repos.d/dag.repo << EOL
[dag]
name=Dag RPM Repository for Red Hat Enterprise Linux
baseurl=http://apt.sw.be/redhat/el\$releasever/en/\$basearch/dag
gpgcheck=1
enabled=1
EOL'
    '''.strip()
    fixyum_cmds = []
    fixyum_cmds.append(
        '''
        su -c 'rpm -Uvh http://download.fedoraproject.org/pub/epel/6/i386/epel-release-6-8.noarch.rpm'
        '''.strip())
    fixyum_cmds.append(make_dag_repo_command)
    fixyum_cmds.append('cat /etc/yum.repos.d/dag.repo')
    fixyum_cmds.append('sudo yum groupinstall -y \'development tools\'')
    fixyum_cmds.append('sudo yum groupinstall -y \'development tools\'')
    # Import pbulic key for epel repo's
    fixyum_cmds.append('sudo rpm --import http://dl.fedoraproject.org/pub/epel/RPM-GPG-KEY-EPEL-6')
    fixyum_cmds.append('sudo yum install -y zlib-dev openssl-devel sqlite-devel bzip2-devel')
    fixyum_cmds.append(textwrap.dedent(
        '''
        sudo yum install bash-completion
        '''
    ))
    # Get ability to download python
    # Prereqs for python 2.7
    fixyum_cmds.append(textwrap.dedent(
        '''
        sudo yum install -y zlib-dev
        sudo yum install -y openssl-devel
        sudo yum install -y openssl
        sudo yum install -y sqlite-devel
        sudo yum install -y bzip2-devel
        sudo yum upgrade -y wget
        sudo yum install xz-libs -y
        sudo yum install qt -y
        sudo yum install qt-devel -y
        sudo yum install readline-devel -y
        sudo yum install ncurses-devel ncurses -y
        sudo yum install tk-devel -y
        # sudo apt-get install libncurses5-dev libncursesw5-dev

        sudo yum install qt -y
        sudo yum install qt-devel -y
        '''
    ))
    # Download and unzip python
    fixyum_cmds.append(textwrap.dedent(
        '''
        wget https://www.python.org/ftp/python/2.7.6/Python-2.7.6.tgz
        gunzip Python-2.7.6.tgz
        tar -xvf Python-2.7.6.tar
        '''))
    # Configure Python
    fixyum_cmds.append(textwrap.dedent(
        '''
        cd Python-2.7.6
        ./configure --prefix=/usr/local --enable-unicode=ucs4 --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib"
        make
        sudo make altinstall
        cd ~
        '''))
    #sudo yum install make automake gcc gcc-c++ kernel-devel git-core -y

    #wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
    #python2.7 ez_setup.py
    #easy_install-2.7 pip

    # Get Pip
    fixyum_cmds.append(textwrap.dedent(
        '''
        cd ~/tmp
        wget https://bootstrap.pypa.io/get-pip.py
        sudo /usr/local/bin/python2.7 get-pip.py
        sudo /usr/local/bin/pip install pip --upgrade

        ln -s /usr/local/bin/pip2.7 /usr/local/bin/pip27
        ln -s /usr/local/bin/python2.7 /usr/local/bin/python27
        sudo ln -s /usr/local/bin/pip2.7 /usr/bin/pip2.7
        sudo ln -s /usr/local/bin/python2.7 /usr/bin/python2.7
        sudo ln -s /usr/local/bin/pip2.7 /usr/bin/pip27
        sudo ln -s /usr/local/bin/python2.7 /usr/bin/python27

        sudo -H pip2.7 install virtualenv
        virtualenv-2.7 ~/ibeis27
        source ibeis27/bin/activate
        python --version
        '''))

    # Get SIP
    fixyum_cmds.append(textwrap.dedent(
        '''
        cd ~/tmp
        wget http://sourceforge.net/projects/pyqt/files/sip/sip-4.16/sip-4.16.tar.gz
        gunzip sip-4.16.tar.gz && tar -xvf sip-4.16.tar
        cd sip-4.16
        python27 configure.py
        make
        sudo make install
        '''))
    # Get PyQt4
    fixyum_cmds.append(textwrap.dedent(
        '''
        cd ~/tmp
        wget http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11/PyQt-x11-gpl-4.11.tar.gz
        gunzip PyQt-x11-gpl-4.11.tar.gz && tar -xvf PyQt-x11-gpl-4.11.tar
        cd PyQt-x11-gpl-4.11
        sudo yum upgrade qt -y
        python27 configure-ng.py
        make
        sudo make install
        '''))

    return fixyum_cmds
    #with open('/etc/yum.repos.d/dag.repo', 'w') as file_:
    #     file_.write(dag_repo)


PIP_PYPKG_SET = get_pip_installed()
"""
elif FEDORA_FAMILY:
    return __check
"""
#print('\n'.join(sorted(list(PIP_PYPKG_SET))))

PIP_PYPKG_MAP = {
    'dateutils': 'python-dateutil',
    'pyreadline': 'readline',
    'pyqt4': 'PyQt4',
}

NOPIP_PYPKGS = set([
    'pip',
    #'setuptools',
    'pyqt4',
    'sip',
    #'scipy'
])


# Convience

def shell(*args, **kwargs):
    """ A really roundabout way to issue a system call """
    import subprocess
    sys.stdout.flush()
    # Parse the keyword arguments
    verbose = kwargs.get('verbose', False)
    detatch = kwargs.get('detatch', False)
    shell   = kwargs.get('shell', True)
    # Print what you are about to do
    if verbose:
        print('[cplat] RUNNING: %r' % (args,))
    # Open a subprocess with a pipe
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=shell)
    if detatch:
        if verbose:
            print('[cplat] PROCESS DETATCHING')
        return None, None, 1
    if verbose and not detatch:
        if verbose:
            print('[cplat] RUNNING WITH VERBOSE OUTPUT')
        logged_out = []
        def run_process():
            while True:
                # returns None while subprocess is running
                retcode = proc.poll()
                line = proc.stdout.readline()
                yield line
                if retcode is not None:
                    raise StopIteration('process finished')
        for line in run_process():
            sys.stdout.write(line)
            sys.stdout.flush()
            logged_out.append(line)
        out = '\n'.join(logged_out)
        (out_, err) = proc.communicate()
        print(err)
    else:
        # Surpress output
        if verbose:
            print('[cplat] RUNNING WITH SUPRESSED OUTPUT')
        (out, err) = proc.communicate()
    # Make sure process if finished
    ret = proc.wait()
    if verbose:
        print('[cplat] PROCESS FINISHED')
    return out, err, ret


# MACPORTS COMMANDS

def __install_command_macports(pkg):
    pkg = pkg.replace('python-', MACPORTS_PY_PREFIX)
    pkg = MACPORTS_PKGMAP.get(pkg, pkg)
    if pkg is None:
        return ('')
    extra = ''
    if pkg == 'python':
        pkg += MACPORTS_PY_SUFFIX
        pyselect_inst = __install_command_macports('python_select')
        pyselect_set1 = 'sudo python_select python' + MACPORTS_PY_SUFFIX
        pyselect_set2 = 'sudo select python python' + MACPORTS_PY_SUFFIX + ' @' +  TARGET_PY_VERSION
        extra = ' && ' + pyselect_inst + ' && ' + pyselect_set1 + ' && ' + pyselect_set2
    elif pkg == 'opencv':
        pkg += ' +python27'
    elif pkg == 'py27-matplotlib':
        pkg += ' +qt +tkinter'
    command = ('sudo port install %s' % pkg) + extra
    #command = ('port search %s' % pkg) + extra
    return command


def __update_macports():
    return 'sudo port selfupdate && sudo port upgrade outdated'


def __check_installed_macports(pkg):
    # First try which
    out, err, ret = shell('which ' + pkg)
    if ret == 0:
        return True
    # Then use dpkg to check if we have it
    out, err, ret = shell('port search ' + pkg)
    if ret == 0:
        return True
    else:
        return False


# APT_GET COMMANDS


def __check_installed_apt_get(pkg):
    # First try which
    out, err, ret = shell('which ' + pkg)
    if ret == 0:
        return True
    # Then use dpkg to check if we have it
    pkg = fix_pkgname_apt_get(pkg)
    out, err, ret = shell('dpkg -s ' + pkg)
    if ret == 0:
        return True
    else:
        return False


def fix_pkgname_apt_get(pkg):
    """ Returns the correct package name for apt_get if given a known alias """
    return APT_GET_PKGMAP.get(pkg, pkg)


def __install_command_apt_get(pkg):
    """
    Returns the apt_get install command for a package (accepts known aliases)
    """
    pkg = fix_pkgname_apt_get(pkg)
    if APTDCON:
        return 'yes | sudo aptdcon --install %s' % pkg
    return 'sudo apt-get install -y %s' % pkg


def __uninstall_command_apt_get(pkg):
    """
    Returns the apt_get uninstall command for a package (accepts known aliases)
    """
    pkg = fix_pkgname_apt_get(pkg)
    return 'sudo apt-get remove -y %s' % pkg


def __update_apt_get():
    return 'sudo apt-get update && sudo apt-get upgrade -y'


# FEDORA_FAMILY YUM COMMANDS

def __check_installed_yum(pkg):
    # First try which
    out, err, ret = shell('which ' + pkg)
    if ret == 0:
        return True
    # Then use yum to check if we have it
    pkg = fix_pkgname_yum(pkg)
    out, err, ret = shell('yum list installed ' + pkg)
    if ret == 0:
        return True
    else:
        return False


def fix_pkgname_yum(pkg):
    """ Returns the correct package name for apt_get if given a known alias """
    return YUM_PKGMAP.get(pkg, pkg)


def __install_command_yum(pkg):
    pkg = fix_pkgname_yum(pkg)
    return 'sudo yum install -y %s' % pkg


def __update_yum():
    return 'sudo yum -y update'

# ARCH PACMAN COMMANDS


def __check_installed_pacman(pkg):
    out, err, ret = shell('which ' + pkg)
    if ret == 0:
        return True
    # Otherwise use pacman to check for it
    pkg = fix_pkgname_pacman(pkg)
    out, err, ret = shell('pacman -Qi ' + pkg)
    if ret == 0:
        return True
    else:
        return False


def fix_pkgname_pacman(pkg):
    return PACMAN_PKGMAP.get(pkg, pkg)


def __install_command_pacman(pkg):
    pkg = fix_pkgname_pacman(pkg)
    if '$AUR' in pkg:
        return '#Install %s from the AUR' % pkg.replace('$AUR ', '')
    return 'sudo pacman -S --needed %s' % pkg


def __update_pacman():
    return 'sudo pacman -Sy'
# PIP COMMANDS


def get_pypkg_aliases(pkg):
    alias1 = PIP_PYPKG_MAP.get(pkg, pkg)
    alias2 = pkg.lower()
    return list(set([pkg, alias1, alias2]))


def check_python_installed(pkg, target_version=None):
    if not CHECK_INSTALLED:
        return False
    # Get aliases for this pypkg
    pypkg_aliases = get_pypkg_aliases(pkg)
    # First check to see if its in our installed set
    if any([alias in PIP_PYPKG_SET for alias in pypkg_aliases]):
        return True
    # Then check to see if we can import it
    for alias in pypkg_aliases:
        try:
            module = __import__(alias, globals(), locals(), fromlist=[], level=0)
            if target_version is not None:
                try:
                    assert module.__version__ == target_version
                except Exception:
                    continue
            return True
        except ImportError:
            continue
    return False


def __install_command_pip(pkg, upgrade=None):
    import platform
    python_version = platform.python_version()
    PYTHON3 = python_version.startswith('3')
    if PYTHON3:
        pipcmd = 'pip3'
    else:
        if FEDORA_FAMILY:
            pipcmd = 'pip27'
        if ARCH:
            pipcmd = 'pip2'  # Otherwise it will default to using Python 3
        else:
            pipcmd = 'pip'
    fmtstr_install_pip = pipcmd + ' install %s'
    # TODO: test if in virtualenv instead of using nosudo commandline
    WITH_SUDO = not WIN32 and '--nosudo' not in sys.argv
    if WITH_SUDO:
        fmtstr_install_pip = 'sudo ' + fmtstr_install_pip
    # First check if we already have this package
    if upgrade is None:
        upgrade = UPGRADE_PIP
    if check_python_installed(pkg):
        return ''
    # See if this package should be installed through
    # the os package manager
    if MACPORTS and pkg not in MACPORTS_PYPKG_IGNORE_LIST:
        # Apple should prefer macports
        pkg = APPLE_PYPKG_MAP.get(pkg, pkg)
        command = __install_command_macports('python-' + pkg)
    elif DEBIAN_FAMILY and pkg in NOPIP_PYPKGS:
        command = __install_command_apt_get('python-' + pkg)
        if pkg == 'pip':
            # PIP IS VERY SPECIAL. HANDLE VERY EXPLICITLY
            # Installing pip is very weird on Ubuntu, apt-get installs pip 1.0,
            # but then pip can upgrade itself to 1.5.3, but then we have to
            # remove the apt-get-pip as well as the apt-get-setuptools which is
            # ninja installed. setuptools and pip go hand-in-hand. so ensure
            # that as well
            command = [
                __install_command_apt_get('python-pip'),
                fmtstr_install_pip % 'pip',
                fmtstr_install_pip % 'setuptools',
                fmtstr_install_pip % 'pip --upgrade',
                fmtstr_install_pip % 'setuptools --upgrade',
                __uninstall_command_apt_get('python-pip'),
                __uninstall_command_apt_get('python-setuptools'),
                fmtstr_install_pip % 'pip --upgrade',
                fmtstr_install_pip % 'setuptools --upgrade',
            ]
        else:
            command = __install_command_apt_get('python-' + pkg)
    elif FEDORA_FAMILY and pkg in NOPIP_PYPKGS:
        return ''
    else:
        # IF not then try and install through pip
        command = fmtstr_install_pip % pkg
        # I dont know why it gets a weird version
        if pkg in ['setuptools', 'numpy']:
            upgrade = True
        if upgrade:
            #command = [command, command + ' --upgrade']
            command = [command + ' --upgrade']
    return command


# GENERAL COMMANDS

def apply_preinstall_fixes():
    if FEDORA_FAMILY:
        prefixes = []
        prefixes.append(update_and_upgrade())
        return [cmd(command, lbl='_fix_yum_repos: ' + str(count))
                for count, command in enumerate(_fix_yum_repos())]
    else:
        return []


def apply_postinstall_fixes():
    if MACPORTS:
        return [
            'sudo port select --set python python27\n',
            'sudo port select --set ipython ipython27\n',
            'sudo port select --set cython cython27\n',
            'sudo port select --set pip pip27\n',
            'echo "NEED TO INSTALL CLANG2: http://stackoverflow.com/questions/20321988/error-enabling-openmp-ld-library-not-found-for-lgomp-and-clang-errors/21789869#21789869"\n'
        ]
    else:
        return []


def update_and_upgrade():
    if DEBIAN_FAMILY:
        return cmd(__update_apt_get())
    if FEDORA_FAMILY:
        return cmd(__update_yum())
    if MACPORTS:
        return cmd(__update_macports())
    if ARCH:
        return cmd(__update_pacman())


def check_installed(pkg):
    if not CHECK_INSTALLED:
        return False
    if DEBIAN_FAMILY:
        return __check_installed_apt_get(pkg)
    elif FEDORA_FAMILY:
        return __check_installed_yum(pkg)
    elif ARCH:
        return __check_installed_pacman(pkg)
    else:
        raise NotImplemented('fixme')


def ensure_package(pkg):
    #if isinstance(pkg, (list, tuple)):
    #    return install_packages(pkg)
    if check_installed(pkg):
        return ''
    if DEBIAN_FAMILY:
        command = __install_command_apt_get(pkg)
    elif FEDORA_FAMILY:
        command = __install_command_yum(pkg)
    elif ARCH:
        command = __install_command_pacman(pkg)
    elif MACPORTS:
        command = __install_command_macports(pkg)
    elif WIN32:
        command = ''
        #raise Exception('Win32: not a chance.')
    else:
        raise NotImplementedError('%r is not yet supported' % ((__OS__, distro, distro_version,),))
    return cmd(command)


def ensure_python_package(pkg, upgrade=None):
    command = __install_command_pip(pkg, upgrade=upgrade)
    return cmd(command)


# CONVINENCE COMMANDS


def cmd(command, lbl=None):
    if command == '':
        # Base Case
        return ''
    elif isinstance(command, (list, tuple)):
        # Recursive Case
        return ''.join([cmd(_) for _ in command])
    else:
        # Base Case
        print(command)
        if lbl is None:
            lbl = command.split('\n')[0]
        delim1 = 'echo "************"'
        delim2 = 'echo "command = %r"' % lbl
        write_list = [delim1]
        write_list += [delim2]
        if CRASH_ON_FAIL:
            # Augments the bash script to exit on the failure of a command
            fail_extra = '|| { echo "FAILED ON COMMAND: %r" ; exit 1; }' % lbl
            write_list += [command + fail_extra]
        else:
            write_list += [command]
        write_list += [delim1]
        output = '\n'.join(write_list) + '\n'
        #INSTALL_PREREQ_FILE.write(output)
        return output
        #os.system(command)


def make_prereq_script(pkg_list, pypkg_list, with_sysfix=True, with_syspkg=True,
                       with_pypkg=True, with_config=True, upgrade=None):
    output_list = []
    if with_sysfix:
        output_list.extend(apply_preinstall_fixes())
    if with_syspkg:
        output_list.extend([ensure_package(pkg) for pkg in pkg_list])
    if with_pypkg:
        output_list.extend([ensure_python_package(pypkg, upgrade=upgrade) for pypkg in pypkg_list])
    if with_config:
        output_list.extend(apply_postinstall_fixes())
    output = ''.join(output_list)
    return output


if __name__ == '__main__':
    print_sysinfo()
