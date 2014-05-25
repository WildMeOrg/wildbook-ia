# HELPS GENERATE CROSS PLATFORM INSTALL SCRIPTS
import sys
import atexit
import os
import platform


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
print('Working on: %r %r %r ' % (__OS__, DISTRO, DISTRO_VERSION,))


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


def __install_command_apt_get(pkg):
    if pkg == 'python-pyqt4':
        pkg = 'python-qt4'
    return 'sudo apt-get install -y %s' % pkg


# PIP COMMANDS

UPGRADE_PIP = '--upgrade' in sys.argv


def __install_command_pip(pkg):
    if WIN32:
        command = 'pip install %s' % pkg
    else:
        command = 'sudo pip install %s' % pkg
    if UPGRADE_PIP:
        return command + ' && ' + command + ' --upgrade'
    return command


def __update_apt_get():
    return 'sudo apt-get update && sudo apt-get upgrade -y'


def upgrade():
    if LINUX and UBUNTU:
        return cmd(__update_apt_get())
    if APPLE and MACPORTS:
        return cmd(__update_macports())


def ensure_package(pkg):
    #if isinstance(pkg, (list, tuple)):
    #    return install_packages(pkg)
    if LINUX and UBUNTU:
        command = __install_command_apt_get(pkg)
    if APPLE and MACPORTS:
        command = __install_command_macports(pkg)
    if WIN32:
        raise Exception('hahahaha, not a chance.')
    cmd(command)


APPLE_PYPKG_MAP = {
    'dateutils': 'dateutil',
    'pyreadline': 'readline',
}


def ensure_python_package(pkg):
    if LINUX and UBUNTU:
        if pkg in ['pip', 'setuptools', 'pyqt4', 'sip', 'scipy']:
            return cmd(__install_command_apt_get('python-' + pkg))
    if APPLE:
        if pkg in APPLE_PYPKG_MAP:
            pkg = APPLE_PYPKG_MAP[pkg]
            return cmd(__install_command_macports('python-' + pkg))
    return cmd(__install_command_pip(pkg))


def ensure_python_packages(pkg_list):
    output_list = []
    for pkg_ in pkg_list:
        output = ensure_python_package(pkg_)
        output_list.append(output)
    return output_list


def ensure_packages(pkg_list):
    output_list = []
    for pkg_ in pkg_list:
        output = ensure_package(pkg_)
        output_list.append(output)
    return output_list


INSTALL_PREREQ_FILE = None


def __ensure_output_file():
    global INSTALL_PREREQ_FILE
    if INSTALL_PREREQ_FILE is None:
        filename = 'install_prereqs.sh'
        file_ = open(filename, 'w')
        def close_file():
            print('# wrote: %r' % os.path.realpath(filename))
            file_.close()
            os.system('chmod +x ' + filename)
            #os.system('cat ' + filename)
        INSTALL_PREREQ_FILE = file_
        atexit.register(close_file)


CRASH_ON_FAIL = True


def cmd(command):
    __ensure_output_file()
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
    INSTALL_PREREQ_FILE.write('\n'.join(write_list) + '\n')
    #os.system(command)
