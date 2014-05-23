import sys
import os
import platform

OPERATING_SYSTEM = sys.platform
APPLE = OPERATING_SYSTEM.startswith('darwin')
WIN32 = OPERATING_SYSTEM.startswith('win32')
LINUX = OPERATING_SYSTEM.startswith('linux')
(DISTRO, DISTRO_VERSION, DISTRO_TAG) = platform.dist()

UBUNTU = (DISTRO == 'Ubuntu')
MACPORTS = APPLE  # We force macports right now

print('Working on: %r %r %r ' % OPERATING_SYSTEM, DISTRO, DISTRO_VERSION)

TARGET_PY_VERSION = '2.7'
MACPORTS_PY_VERSION = TARGET_PY_VERSION.replace('.', '')
MACPORTS_PY_PREFIX = 'py' + MACPORTS_PY_VERSION + '-'


def __fix_pkg_macports(pkg):
    pkg = pkg.replace('python-', MACPORTS_PY_PREFIX)
    if pkg == 'python':
        pkg += MACPORTS_PY_VERSION
    if pkg == 'opencv':
        pkg += ' +python27'
    if pkg == 'py27-matplotlib':
        pkg += ' +qt +tkinter'
    return pkg


def __install_macports_python():
    install_package('python')
    install_package('python_select')
    cmd('sudo port install python_select')
    cmd('sudo python_select python27')
    cmd('sudo select python python27 @2.7.6')


def __update_macports():
    cmd('sudo port selfupdate')
    cmd('sudo port upgrade outdated')


def __update_apt_get():
    cmd('sudo apt-get update')
    cmd('sudo apt-get upgrade')


def ensure_packages(pkg_list):
    output_list = []
    for pkg_ in pkg_list:
        output = ensure_package(pkg_)
        output_list.append(output)
    return output_list


def ensure_package(pkg):
    #if isinstance(pkg, (list, tuple)):
    #    return install_packages(pkg)
    if LINUX and UBUNTU:
        command = 'sudo apt-get install -y %s' % pkg
    if APPLE and MACPORTS:
        pkg = __fix_pkg_macports(pkg)
        command = 'sudo port install %' % pkg
    if WIN32:
        raise Exception('hahahaha, not a chance.')
    cmd(command)


def upgrade():
    if LINUX and UBUNTU:
        __update_apt_get()
    if APPLE and MACPORTS:
        __update_macports()


def cmd(command):
    print(command)
    #os.system(command)
