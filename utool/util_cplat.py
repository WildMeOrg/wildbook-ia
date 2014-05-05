'''
cross platform utilities
'''
from __future__ import absolute_import, division, print_function
import os
import sys
import platform
import subprocess
import shlex
from os.path import exists, normpath
from os.path import expanduser
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[cplat]')

COMPUTER_NAME = platform.node()

WIN32         = sys.platform.startswith('win32')
LINUX         = sys.platform.startswith('linux')
DARWIN        = sys.platform.startswith('darwin')


def get_computer_name():
    return COMPUTER_NAME


def getroot():
    root = {
        'WIN32': 'C:\\',
        'LINUX': '/',
        'DARWIN': '/',
    }[sys.platform]
    return root


def startfile(fpath):
    print('[cplat] startfile(%r)' % fpath)
    if not exists(fpath):
        raise Exception('Cannot start nonexistant file: %r' % fpath)
    if LINUX:
        out, err, ret = cmd(['xdg-open', fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    elif DARWIN:
        out, err, ret = cmd(['open', fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    else:
        os.startfile(fpath)
    pass


def view_directory(dname=None):
    'view directory'
    print('[cplat] view_directory(%r) ' % dname)
    dname = os.getcwd() if dname is None else dname
    open_prog = {'win32': 'explorer.exe',
                 'linux2': 'nautilus',
                 'darwin': 'open'}[sys.platform]
    os.system(open_prog + ' ' + normpath(dname))


def get_resource_dir():
    if WIN32:
        return expanduser('~/AppData/Roaming')
    if LINUX:
        return expanduser('~/.config')
    if DARWIN:
        return expanduser('~/Library/Application Support')


def shell(*args, **kwargs):
    """
    Dangerous. Take out of production code
    """
    kwargs['shell'] = True
    return cmd(*args, **kwargs)


def cmd(*args, **kwargs):
    sys.stdout.flush()
    verbose = kwargs.get('verbose', True)
    detatch = kwargs.get('detatch', False)
    shell = kwargs.get('shell', False)
    sudo = kwargs.get('sudo', False)
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    if isinstance(args, str):
        if os.name == 'posix':
            args = shlex.split(args)
        else:
            args = [args]
    if sudo is True and not WIN32:
        args = ['sudo'] + args
    print('[cplat] Running: %r' % (args,))
    PIPE = subprocess.PIPE
    proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE, shell=shell)
    if detatch:
        return None, None, 1
    if verbose and not detatch:
        logged_list = []
        append = logged_list.append
        write = sys.stdout.write
        flush = sys.stdout.flush
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            write(line)
            flush()
            append(line)
        out = '\n'.join(logged_list)
        (out_, err) = proc.communicate()
        print(err)
    else:
        # Surpress output
        (out, err) = proc.communicate()
    # Make sure process if finished
    ret = proc.wait()
    return out, err, ret


def get_flops():
    from sys import stderr, stdout
    from re import compile

    filename = "linpack.out"
    fpnum = r'\d+\.\d+E[+-]\d\d'
    fpnum_1 = fpnum + r' +'
    pattern = compile(r'^ *' + fpnum_1 + fpnum_1 + fpnum_1 + r'(' + fpnum + r') +' + fpnum_1 + fpnum + r' *\n$')
    speeds = [0.0, 1.0e75, 0.0]

    file = open(filename)
    count = 0
    while file :
        line = file.readline()
        if not line :
            break
        if pattern.match(line) :
            count = count + 1
            x = float(pattern.sub(r'\1', line))
            if x < 1.0 :
                print(count)
            speeds[0] = speeds[0] + x
            speeds[1] = min(speeds[1], x)
            speeds[2] = max(speeds[2], x)
    file.close()
    if count != 0 :
        speeds[0] = speeds[0] / count

    stdout.write("%6.1f MFlops (%d from %.1f to %.1f)\n" % (speeds[0], count, speeds[1], speeds[2]))
