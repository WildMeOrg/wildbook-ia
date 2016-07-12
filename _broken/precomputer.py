# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
from ibeis.expt import experiment_helpers


def precfg_dbs(db_list):
    r"""
    Helper to precompute information Runs precfg on multiple databases

    Args:
        db_list (list):

    CommandLine:
        python -m ibeis.expt.precomputer --exec-precfg_dbs
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist testdb1 PZ_MTEST
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist testdb1 PZ_MTEST --preload -t custom
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=PZ_MTEST,NNP_MasterGIRM_core,PZ_Master0,NNP_Master3,GZ_ALL,PZ_FlankHack --preload --delete-nn-cache

        #python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=PZ_Master0 -t candidacy1 --preload-chip --controlled --species=primary
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy --preload

        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-chip --species=primary --controlled
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-chip --species=primary --allgt
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-feat
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-featweight
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy -t candidacy --preload
        python -m ibeis.expt.precomputer --exec-precfg_dbs --dblist=candidacy --delete-nn-cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.expt.precomputer import *  # NOQA
        >>> db_list = ut.get_argval('--dblist', type_=list, default=['testdb1'])
        >>> result = precfg_dbs(db_list)
        >>> print(result)
    """
    import ibeis.init.main_helpers
    import ibeis
    if db_list == ['candidacy']:
        from ibeis.expt import experiment_configs
        db_list = experiment_configs.get_candidacy_dbnames()  # HACK
    print('db_list = %s' % (ut.list_str(db_list),))
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=[])
    for db in db_list:
        ibs = ibeis.opendb(db=db)
        ibs, qaids, daids = ibeis.init.main_helpers.testdata_expanded_aids(verbose=False, ibs=ibs)
        precfg(ibs, qaids, daids, test_cfg_name_list)


def precfg(ibs, acfg_name_list, test_cfg_name_list):
    r"""
    Helper to precompute information

    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (list):  query annotation ids
        daids (list):  database annotation ids
        test_cfg_name_list (list):

    CommandLine:
        python -m ibeis.expt.precomputer --exec-precfg -t custom --expt-preload

        python -m ibeis.expt.precomputer --exec-precfg -t candidacy -a default:qaids=allgt --preload
        python -m ibeis.expt.precomputer --exec-precfg -t candidacy_invariance -a default:qaids=allgt --preload

        python -m ibeis.expt.precomputer --exec-precfg --delete-nn-cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.expt.precomputer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> default_acfgstrs = ['default:qaids=allgt']
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstrs)
        >>> test_cfg_name_list = ut.get_argval('-t', type_=list, default=['custom'])
        >>> result = precfg(ibs, acfg_name_list, test_cfg_name_list)
        >>> print(result)
    """
    # Generate list of database annotation configurations
    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, acfg_name_list)
    # Generate list of query pipeline param configs
    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(test_cfg_name_list, ibs=ibs)
    #cfgx2_lbl = experiment_helpers.get_varied_cfg_lbls(cfgdict_list)

    expanded_aids_iter = ut.ProgressIter(expanded_aids_list, lbl='annot config', freq=1, autoadjust=False)
    nAcfg = len(acfg_list)

    for acfgx, (qaids, daids) in enumerate(expanded_aids_iter):
        if len(qaids) == 0:
            print('[harness] WARNING No query annotations specified')
            continue
        if len(daids) == 0:
            print('[harness] WARNING No database annotations specified')
            continue
        ut.colorprint('\n---Annot config', 'turquoise')

        nCfg     = len(pipecfg_list)   # number of configurations (cols)
        dbname = ibs.get_dbname()

        cfgiter = ut.ProgressIter(pipecfg_list, lbl='query config', freq=1, autoadjust=False, parent_index=acfgx, parent_nTotal=nAcfg)

        flag = False
        if ut.get_argflag('--delete-nn-cache'):
            ibs.delete_neighbor_cache()
            flag = True

        for cfgx, query_cfg in enumerate(cfgiter):
            print('')
            ut.colorprint(query_cfg.get_cfgstr(), 'turquoise')
            verbose = True
            with ut.Indenter('[%s cfg %d/%d]' % (dbname, (acfgx * nCfg) + cfgx * + 1, nCfg * nAcfg)):

                qreq_ = ibs.new_query_request(qaids, daids, verbose=True, query_cfg=query_cfg)
                if ut.get_argflag('--preload'):
                    qreq_.lazy_preload(verbose=verbose)
                    flag = True
                if ut.get_argflag('--preload-chip'):
                    qreq_.ensure_chips(verbose=verbose, num_retries=1)
                    flag = True
                if ut.get_argflag('--preload-feat'):
                    qreq_.ensure_features(verbose=verbose)
                    flag = True
                if ut.get_argflag('--preload-featweight'):
                    qreq_.ensure_featweights(verbose=verbose)
                    flag = True
                if ut.get_argflag('--preindex'):
                    flag = True
                    if qreq_.qparams.pipeline_root in ['vsone', 'vsmany']:
                        qreq_.load_indexer(verbose=verbose)
            assert flag is True, 'no flag specified'
        assert flag is True, 'no flag specified'


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.expt.precomputer
        python -m ibeis.expt.precomputer --allexamples
        python -m ibeis.expt.precomputer --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
