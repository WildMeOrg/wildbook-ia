"""
TODO: Rename to ibeis/init/commands.py
"""
from __future__ import absolute_import, division, print_function
import utool
import utool as ut  # NOQA
import sys
from ibeis import constants
from ibeis import params
from ibeis import ibsfuncs
from ibeis.dev import sysres
from os.path import join
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[commands]')


def vdq(dbdir):
    """view directory and quit"""
    _ibsdb = constants.PATH_NAMES._ibsdb
    utool.util_cplat.view_directory(join(dbdir, _ibsdb))
    sys.exit(1)


def vdd(ibs):
    " view data dir "
    utool.util_cplat.view_directory(ibs.dbdir)


def vwd():
    """ view work dir """
    utool.util_cplat.view_directory(sysres.get_workdir())


def preload_convert_hsdb(dbdir):
    """ Convert the database before loading (A bit hacky) """
    from ibeis.dbio import ingest_hsdb
    ingest_hsdb.convert_hsdb_to_ibeis(dbdir, force_delete=params.args.force_delete)


def preload_commands(dbdir, **kwargs):
    """ Preload commands work with command line arguments and global caches """
    #print('[main_cmd] preload_commands')
    if params.args.dump_argv:
        print(utool.dict_str(vars(params.args), sorted_=False))
    if params.args.dump_global_cache:
        utool.global_cache_dump()  # debug command, dumps to stdout
    if params.args.set_workdir is not None:
        sysres.set_workdir(params.args.set_workdir)
    if params.args.get_workdir:
        print(' Current work dir = %s' % sysres.get_workdir())
    if params.args.logdir is not None:
        sysres.set_logdir(params.args.logdir)
    if utool.get_argflag('--vwd'):
        vwd()
    if utool.get_argflag('--vdq'):
        print('got arg --vdq')
        vdq(dbdir)
    if kwargs.get('delete_ibsdir', False):
        ibsfuncs.delete_ibeis_database(dbdir)
    if params.args.convert:
        preload_convert_hsdb(dbdir)
    if params.args.preload_exit:
        print('[main_cmd] preload exit')
        sys.exit(1)


def postload_commands(ibs, back):
    """ Postload commands deal with a specific ibeis database """
    if utool.NOT_QUIET:
        print('[main_cmd] postload_commands')
    if params.args.view_database_directory:
        print('got arg --vdd')
        vdd(ibs)
    if params.args.set_default_dbdir:
        sysres.set_default_dbdir(ibs.get_dbdir())
    if params.args.update_query_cfg is not None:
        # Set query parameters from command line using the --cfg flag
        cfgdict = ut.parse_cfgstr_list(params.args.update_query_cfg)
        print('Custom cfgdict specified')
        print(ut.dict_str(cfgdict))
        ibs.update_query_cfg(**cfgdict)
        #print(ibs.cfg.query_cfg.get_cfgstr())
    if params.args.edit_notes:
        utool.editfile(ibs.get_dbnotes_fpath(ensure=True))
    if params.args.delete_cache:
        ibs.delete_cache()
    if params.args.delete_cache_complete:
        ibs.delete_cache(delete_chips=True, delete_encounters=True)
    if params.args.delete_query_cache:
        ibs.delete_qres_cache()
    if params.args.set_notes is not None:
        ibs.set_dbnotes(params.args.set_notes)
    if params.args.set_aids_as_hard is not None:
        aid_list = params.args.set_aids_as_hard
        ibs.set_annot_is_hard(aid_list, [True] * len(aid_list))
    if params.args.set_all_species is not None:
        ibs._overwrite_all_annot_species_to(params.args.set_all_species)
    if params.args.dump_schema:
        ibs.db.print_schema()
    if params.args.select_aid is not None:
        try:
            ibsfuncs.assert_valid_aids(ibs, (params.args.select_aid,))
        except AssertionError:
            print('Valid RIDs are: %r' % (ibs.get_valid_aids(),))
            raise
        back.select_aid(params.args.select_aid)
    if params.args.query_aid is not None:
        from ibeis.constants import VS_EXEMPLARS_KEY
        back.query(params.args.query_aid, query_mode=VS_EXEMPLARS_KEY)
    if params.args.select_gid is not None:
        back.select_gid(params.args.select_gid)
    if params.args.select_nid is not None:
        back.select_nid(params.args.select_nid)
    if params.args.postload_exit:
        print('[main_cmd] postload exit')
        sys.exit(1)
