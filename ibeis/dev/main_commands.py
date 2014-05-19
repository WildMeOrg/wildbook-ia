from __future__ import absolute_import, division, print_function
import utool
import sys
from ibeis.dev import params
from ibeis.dev import ibsfuncs
from ibeis.dev import sysres
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[main_cmds]')


def vdq(dbdir):
    """view directory and quit"""
    utool.util_cplat.view_directory(dbdir + '/_ibsdb')
    sys.exit(1)


def vdd(ibs):
    " view data dir "
    utool.util_cplat.view_directory(ibs.dbdir)


def vwd():
    """ view work dir """
    utool.util_cplat.view_directory(sysres.get_workdir())


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


def preload_convert(dbdir):
    """ Convert the database before loading (A bit hacky) """
    from ibeis.injest.injest_my_hotspotter_dbs import my_convert_hsdb_to_ibeis
    my_convert_hsdb_to_ibeis(dbdir, force=True)


def preload_commands(dbdir, defaultdb):
    """ Preload commands work with command line arguments and global caches """
    #print('[main_cmd] preload_commands')

    def get_dbdir_hack(dbdir, defaultdb):
        # HACKY: Copied code from _init_ibeis to get the
        # dbdir before creating an IBEISControl
        # Use command line dbdir unless user specifies it
        if dbdir is None:
            dbdir = sysres.get_args_dbdir(defaultdb, False)
        return dbdir
    if params.args.dump_argv:
        print(utool.dict_str(vars(params.args)))
    if params.args.dump_global_cache:
        utool.global_cache_dump()  # debug command, dumps to stdout
    if params.args.workdir is not None:
        sysres.set_workdir(params.args.workdir)
    if utool.get_flag('--vwd'):
        vwd()
    if params.args.convert:
        preload_convert(get_dbdir_hack(dbdir, defaultdb))
    if utool.get_flag('--vdq'):
        print('got arg --vdq')
        vdq(get_dbdir_hack(dbdir, defaultdb))
    if params.args.preload_exit:
        print('[main_cmd] preload exit')
        sys.exit(1)


def postload_commands(ibs, back):
    """ Postload commands deal with a specific ibeis database """
    print('[main_cmd] postload_commands')
    if params.args.view_database_directory:
        print('got arg --vdd')
        vdd(ibs)
    if params.args.set_default_dbdir:
        sysres.set_default_dbdir(ibs.get_dbdir())
    if params.args.update_cfg is not None:
        cfgdict = parse_cfgstr_list(params.args.update_cfg)
        ibs.update_cfg(**cfgdict)
    if params.args.select_rid is not None:
        try:
            ibsfuncs.assert_valid_rids(ibs, (params.args.select_rid,))
        except AssertionError:
            print('Valid RIDs are: %r' % (ibs.get_valid_rids(),))
            raise
        back.select_rid(params.args.select_rid)
    if params.args.select_gid is not None:
        back.select_gid(params.args.select_gid)
    if params.args.select_nid is not None:
        back.select_nid(params.args.select_nid)
    if params.args.postload_exit:
        print('[main_cmd] postload exit')
        sys.exit(1)
