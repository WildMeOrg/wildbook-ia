from __future__ import division, print_function
import utool
import sys
from ibeis.dev import params


def vdq():
    'view directory and quit'
    dbdir = params.args.dbdir
    utool.util_cplat.view_directory(dbdir)
    sys.exit(1)


def vd(ibs):
    utool.util_cplat.view_directory(ibs.dbdir)


def preload_commands():
    if utool.get_flag('--vdq'):
        vdq()


def postload_commands(ibs):
    if utool.get_flag('--vd'):
        vd(ibs)
