from __future__ import absolute_import, division, print_function
import utool
import sys
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[main_cmds]')


def vdq():
    """view directory and quit"""
    dbdir = params.args.dbdir
    utool.util_cplat.view_directory(dbdir)
    sys.exit(1)


def vd(ibs):
    utool.util_cplat.view_directory(ibs.dbdir)


def parse_cfgstr_list(cfgstr_list):
    """
    Parses a list of items in the format
    ['var1:val1', 'var2:val2', 'var3:val3']
    the '=' character can be used instead of the ':' character if desired
    """
    cfgdict = {}
    for item in cfgstr_list:

        varval_tup = item.replace('=', ':').split(':')
        assert len(varval_tup) == 2, '[!] Invalid cfgitem=%r' % (item,)
        var, val = varval_tup
        cfgdict[var] = val
    return cfgdict


def preload_commands():
    #print('[main_cmd] preload_commands')
    if params.args.dump_global_cache:
        utool.global_cache_dump()
    if params.args.workdir is not None:
        params.set_workdir(params.args.workdir)
    if params.args.set_default_dbdir:
        set_default_dbdir(params.args.dbdir)
    if utool.get_flag('--vdq'):
        print('got arg --vdq')
        vdq()


def postload_commands(ibs):
    #print('[main_cmd] postload_commands')
    if params.args.dump_argv:
        print(utool.dict_str(vars(params.args)))
    if params.args.view_database_directory:
        print('got arg --vd')
        vd(ibs)
    if params.args.update_cfg is not None:
        cfgdict = parse_cfgstr_list(params.args.update_cfg)
        ibs.update_cfg(**cfgdict)
    if params.args.postload_exit:
        print('[main_cmd] postload exit')
        sys.exit(1)


def set_default_dbdir(dbdir):
    print('seting default database directory to: %r' % dbdir)
    utool.global_cache_write('cached_dbdir', dbdir)
