# HELPS GENERATE CROSS PLATFORM INSTALL SCRIPTS
import sys
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
MACPORTS = APPLE  # We force macports right now

print('Working on: %r %r %r ' % (__OS__, DISTRO, DISTRO_VERSION,))

TARGET_PY_VERSION = '2.7.6'
MACPORTS_PY_SUFFIX = TARGET_PY_VERSION.replace('.', '')[0:2]
MACPORTS_PY_PREFIX = 'py' + MACPORTS_PY_SUFFIX + '-'


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


def __install_command_apt_get(pkg):
    return 'sudo apt-get install -y %s' % pkg


def __install_command_pip(pkg):
    if WIN32:
        command = 'pip install %s' % pkg
    else:
        command = 'sudo pip install %s' % pkg
    return command + ' && ' + command + ' --upgrade'


def __update_macports():
    return 'sudo port selfupdate && sudo port upgrade outdated'


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


def ensure_python_package(pkg):
    if LINUX and UBUNTU:
        if pkg in ['pip', 'setuptools', 'pyqt4', 'sip']:
            return cmd(__install_command_apt_get('python-' + pkg))
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
        import atexit
        import os
        filename = 'install_prereqs.sh'
        file_ = open(filename, 'w')
        def close_file():
            print('# wrote: %r' % os.path.realpath(filename))
            file_.close()
            os.system('chmod +x ' + filename)
        INSTALL_PREREQ_FILE = file_
        atexit.register(close_file)


def cmd(command):
    __ensure_output_file()
    delim = 'echo "************"'
    INSTALL_PREREQ_FILE.write(delim + '\n')
    INSTALL_PREREQ_FILE.write(command + '\n')
    INSTALL_PREREQ_FILE.write(delim + '\n')
    print(command)
    #os.system(command)
