from __future__ import absolute_import, division, print_function
import utool
import sys
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[main_cmds]')


def vdq():
    'view directory and quit'
    dbdir = params.args.dbdir
    utool.util_cplat.view_directory(dbdir)
    sys.exit(1)


def vd(ibs):
    utool.util_cplat.view_directory(ibs.dbdir)


def preload_commands():
    if utool.get_flag('--set-workdir'):
        set_workdir(params.args.dbdir)
    if utool.get_flag('--vdq'):
        vdq()


def postload_commands(ibs):
    if utool.get_flag('--vd'):
        vd(ibs)


def set_workdir(dbdir):
    print('seting default work directory to: %r' % dbdir)
    utool.global_cache_write('[main_cmd] cached_dbdir', dbdir)
