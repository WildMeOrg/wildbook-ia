# -*- coding: utf-8 -*-
"""
TODO: Rename to wbia/init/commands.py

TODO; remove params module
"""
import logging
import utool as ut
import sys
from wbia import constants as const
from wbia import params
from wbia.other import ibsfuncs
from wbia.init import sysres
from os.path import join

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger('wbia')


def vdq(dbdir):
    """view directory and quit"""
    _ibsdb = const.PATH_NAMES._ibsdb
    ut.util_cplat.view_directory(join(dbdir, _ibsdb))
    sys.exit(0)


def vdd(ibs):
    """view data dir"""
    ut.util_cplat.view_directory(ibs.dbdir)


def vwd():
    """ view work dir """
    ut.util_cplat.view_directory(sysres.get_workdir())


# def preload_convert_hsdb(dbdir):
#     """ Convert the database before loading (A bit hacky) """
#     from wbia.dbio import ingest_hsdb
#     ingest_hsdb.convert_hsdb_to_wbia(dbdir, force_delete=params.args.force_delete)


def preload_commands(dbdir, **kwargs):
    """ Preload commands work with command line arguments and global caches """
    # logger.info('[main_cmd] preload_commands')
    params.parse_args()
    if params.args.dump_argv:
        logger.info(ut.repr2(vars(params.args), sorted_=False))
    if params.args.dump_global_cache:
        ut.global_cache_dump()  # debug command, dumps to stdout
    if params.args.set_workdir is not None:
        sysres.set_workdir(params.args.set_workdir)
    if params.args.get_workdir:
        logger.info(' Current work dir = %s' % sysres.get_workdir())
    # if params.args.logdir is not None:
    #     sysres.set_logdir(params.args.logdir)
    if params.args.get_logdir:
        logger.info(' Current local  log dir = %s' % (sysres.get_logdir_local(),))
        logger.info(' Current global log dir = %s' % (sysres.get_logdir_global(),))
    if params.args.view_logdir:
        ut.view_directory(sysres.get_logdir_local())
        ut.view_directory(sysres.get_logdir_global())
    if params.args.view_logdir_local:
        ut.view_directory(sysres.get_logdir_local())
    if params.args.view_logdir_global:
        ut.view_directory(sysres.get_logdir_local())
    if ut.get_argflag('--vwd'):
        vwd()
    if ut.get_argflag('--vdq'):
        logger.info('got arg --vdq')
        vdq(dbdir)
    if kwargs.get('delete_ibsdir', False):
        ibsfuncs.delete_wbia_database(dbdir)
    if params.args.preload_exit:
        logger.info('[main_cmd] preload exit')
        sys.exit(0)


def postload_commands(ibs, back):
    """
    Postload commands deal with a specific wbia database

    wbia --db PZ_MTEST --occur "*All Images" --query 1
    wbia --db PZ_MTEST --occur "*All Images" --query-intra

    """
    params.parse_args()
    if ut.NOT_QUIET:
        logger.info('\n[main_cmd] postload_commands')
    if params.args.view_database_directory:
        logger.info('got arg --vdd')
        vdd(ibs)
    if params.args.set_default_dbdir:
        sysres.set_default_dbdir(ibs.get_dbdir())
    if params.args.update_query_cfg is not None:
        # Set query parameters from command line using the --cfg flag
        cfgdict = ut.parse_cfgstr_list(params.args.update_query_cfg)
        logger.info('Custom cfgdict specified')
        logger.info(ut.repr2(cfgdict))
        ibs.update_query_cfg(**cfgdict)
    if params.args.edit_notes:
        ut.editfile(ibs.get_dbnotes_fpath(ensure=True))
    if params.args.delete_cache:
        ibs.delete_cache()
    if params.args.delete_cache_complete:
        ibs.delete_cache(delete_imagesets=True)
    if params.args.delete_query_cache:
        ibs.delete_qres_cache()
    if params.args.set_all_species is not None:
        ibs._overwrite_all_annot_species_to(params.args.set_all_species)
    if params.args.dump_schema:
        ibs.db.print_schema()

    if ut.get_argflag('--ipynb'):
        back.launch_ipy_notebook()

    select_imgsetid = ut.get_argval(
        ('--select-imgsetid', '--imgsetid', '--occur', '--gsid'), None
    )
    if select_imgsetid is not None:
        logger.info('\n+ --- CMD SELECT IMGSETID=%r ---' % (select_imgsetid,))
        # Whoa: this doesnt work. weird.
        # back.select_imgsetid(select_imgsetid)
        # This might be the root of gui problems
        # back.front._change_imageset(select_imgsetid)
        back.front.select_imageset_tab(select_imgsetid)
        logger.info('L ___ CMD SELECT IMGSETID=%r ___\n' % (select_imgsetid,))
    # Send commands to GUIBack
    if params.args.select_aid is not None:
        if back is not None:
            try:
                ibsfuncs.assert_valid_aids(ibs, (params.args.select_aid,))
            except AssertionError:
                logger.info('Valid RIDs are: %r' % (ibs.get_valid_aids(),))
                raise
            back.select_aid(params.args.select_aid)
    if params.args.select_gid is not None:
        back.select_gid(params.args.select_gid)
    if params.args.select_nid is not None:
        back.select_nid(params.args.select_nid)

    select_name = ut.get_argval('--select-name')
    if select_name is not None:
        import wbia.gui.guiheaders as gh

        back.ibswgt.select_table_indicies_from_text(
            gh.NAMES_TREE, select_name, allow_table_change=True
        )

    if ut.get_argflag(('--intra-occur-query', '--query-intra-occur', '--query-intra')):
        back.special_query_funcs['intra_occurrence'](cfgdict={'use_k_padding': False})

    qaid_list = ut.get_argval(('--query-aid', '--query'), type_=list, default=None)

    if qaid_list is not None:
        # qaid_list = params.args.query_aid
        # fix stride case
        if len(qaid_list) == 1 and isinstance(qaid_list[0], tuple):
            qaid_list = list(qaid_list[0])
        daids_mode = ut.get_argval(
            '--daids-mode', type_=str, default=const.VS_EXEMPLARS_KEY
        )
        back.compute_queries(qaid_list=qaid_list, daids_mode=daids_mode, ranks_top=10)

    if ut.get_argflag('--inc-query'):
        back.incremental_query()

    if ut.get_argflag(('--dbinfo', '--display_dbinfo')):
        back.display_dbinfo()
        pass

    aidcmd = ut.get_argval('--aidcmd', default=None)
    aid = ut.get_argval('--aid', type_=int, default=1)
    if aidcmd:
        # aidcmd = 'Interact image'
        metadata = ibs.get_annot_lazy_dict(aid)
        annot_context_options = metadata['annot_context_options']
        aidcmd_dict = dict(annot_context_options)
        logger.info('aidcmd_dict = %s' % (ut.repr3(aidcmd_dict),))
        command = aidcmd_dict[aidcmd]
        command()
        # import utool
        # utool.embed()
        # back.start_web_server_parallel()

    if ut.get_argflag('--start-web'):
        back.start_web_server_parallel()

    if ut.get_argflag('--name-tab'):
        from wbia.gui.guiheaders import NAMES_TREE

        back.front.set_table_tab(NAMES_TREE)
        view = back.front.views[NAMES_TREE]
        model = view.model()
        view._set_sort(model.col_name_list.index('nAids'), col_sort_reverse=True)

    if ut.get_argflag('--graph'):
        back.make_qt_graph_interface()

    if params.args.postload_exit:
        logger.info('[main_cmd] postload exit')
        sys.exit(0)
