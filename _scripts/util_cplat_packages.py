# HELPS GENERATE CROSS PLATFORM INSTALL SCRIPTS
import sys
import platform

# Behavior variables

UPGRADE_PIP     = '--upgrade' in sys.argv
CHECK_INSTALLED = True
CRASH_ON_FAIL   = True


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


# Special cases

APPLE_PYPKG_MAP = {
    'dateutils': 'dateutil',
    'pyreadline': 'readline',
}

PIP_PYPKG_SET = get_pip_installed()

#print('\n'.join(sorted(list(PIP_PYPKG_SET))))

PIP_PYPKG_MAP = {
    'dateutils': 'python-dateutil',
    'pyreadline': 'readline',
    'pyqt4': 'PyQt4',
}

UBUNTU_NOPIP_PYPKGS = set([
    #'pip',
    #'setuptools',
    'pyqt4',
    'sip',
    'scipy'
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
__OS__ = ''.join(ARG_DICT.get('os', sys.platform))

distro_tup = platform.dist()
DISTRO = ''.join(ARG_DICT.get('distro', [distro_tup[0]]))
DISTRO_VERSION = ''.join(ARG_DICT.get('distro_verions', [distro_tup[1]]))
DISTRO_TAG = ''.join(ARG_DICT.get('distro_tag', [distro_tup[2]]))
del distro_tup

APPLE = __OS__.startswith('darwin')
WIN32 = __OS__.startswith('win32')
LINUX = __OS__.startswith('linux')

UBUNTU = (DISTRO == 'Ubuntu')
CENTOS = (DISTRO == 'Centos')
MACPORTS = APPLE  # We force macports right now


# PRINT WHAT WE ARE WORKING WITH
def print_sysinfo():
    print('# sysinfo: (%s, %s, %s) ' % (__OS__, DISTRO, DISTRO_VERSION,))


# TARGET PYTHON PLATFORM
TARGET_PY_VERSION = '2.7.6'
MACPORTS_PY_SUFFIX = TARGET_PY_VERSION.replace('.', '')[0:2]
MACPORTS_PY_PREFIX = 'py' + MACPORTS_PY_SUFFIX + '-'


# MACPORTS COMMANDS

def __install_command_macports(pkg):
    pkg = pkg.replace('python-', MACPORTS_PY_PREFIX)
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
    return command


def __update_macports():
    return 'sudo port selfupdate && sudo port upgrade outdated'


# APT_GET COMMANDS


def __check_installed_apt_get(pkg):
    # First try which
    out, err, ret = shell('which ' + pkg)
    if ret == 0:
        return True
    # Then use dpkg to check if we have it
    out, err, ret = shell('dpkg -s ' + pkg)
    if ret == 0:
        return True
    else:
        return False


def __install_command_apt_get(pkg):
    if pkg == 'python-pyqt4':
        pkg = 'python-qt4'
    return 'sudo apt-get install -y %s' % pkg


def __update_apt_get():
    return 'sudo apt-get update && sudo apt-get upgrade -y'


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


def __install_command_pip(pkg):
    # First check if we already have this package
    if check_python_installed(pkg):
        return ''
    # See if this package should be installed through
    # the os package manager
    if UBUNTU and pkg in UBUNTU_NOPIP_PYPKGS:
        command = __install_command_apt_get('python-' + pkg)
    elif APPLE and pkg in APPLE_PYPKG_MAP:
        pkg = APPLE_PYPKG_MAP[pkg]
        command = __install_command_macports('python-' + pkg)
    elif pkg == 'pip':
        # Pip cannot install pip if it doesn't exist
        try:
            import pip  # noqa
        except ImportError:
            command = 'easy_install pip'
    else:
        # IF not then try and install through pip
        if WIN32:
            command = 'pip install %s' % pkg
        else:
            command = 'sudo pip install %s' % pkg
        if UPGRADE_PIP:
            return command + ' && ' + command + ' --upgrade'
    return command


# GENERAL COMMANDS


def upgrade():
    if LINUX and UBUNTU:
        return cmd(__update_apt_get())
    if APPLE and MACPORTS:
        return cmd(__update_macports())


def check_installed(pkg):
    if not CHECK_INSTALLED:
        return False
    if UBUNTU:
        return __check_installed_apt_get(pkg)
    else:
        raise NotImplemented('fixme')


def ensure_package(pkg):
    #if isinstance(pkg, (list, tuple)):
    #    return install_packages(pkg)
    if check_installed(pkg):
        return ''
    if UBUNTU:
        command = __install_command_apt_get(pkg)
    elif MACPORTS:
        command = __install_command_macports(pkg)
    elif WIN32:
        raise Exception('not a chance.')
    if command == '':
        return ''
    return cmd(command)


def ensure_python_package(pkg):
    command = __install_command_pip(pkg)
    if command == '':
        return ''
    return cmd(command)


# CONVINENCE COMMANDS


def cmd(command):
    print(command)
    delim1 = 'echo "************"'
    delim2 = 'echo "command = %r"' % command
    write_list = [delim1]
    write_list += [delim2]
    if CRASH_ON_FAIL:
        fail_extra = '|| { echo "FAILED ON COMMAND: %r" ; exit 1; }' % command
        write_list += [command + fail_extra]
    else:
        write_list += [command]
    write_list += [delim1]
    output = '\n'.join(write_list) + '\n'
    #INSTALL_PREREQ_FILE.write(output)
    return output
    #os.system(command)


def make_prereq_script(pkg_list, pypkg_list):
    output_list = []
    output_list.extend([ensure_package(pkg) for pkg in pkg_list])
    output_list.extend([ensure_python_package(pypkg) for pypkg in pypkg_list])
    output = ''.join(output_list)
    return output
