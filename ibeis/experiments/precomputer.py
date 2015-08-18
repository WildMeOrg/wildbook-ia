def precfg_dbs(db_list):
    r"""
    Helper to precompute information Runs precfg on multiple databases

    Args:
        db_list (list):

    CommandLine:
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist testdb1 PZ_MTEST
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist testdb1 PZ_MTEST --preload -t custom
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=PZ_MTEST,NNP_MasterGIRM_core,PZ_Master0,NNP_Master3,GZ_ALL,PZ_FlankHack --preload --delete-nn-cache

        #python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=PZ_Master0 -t candidacy1 --preload-chip --controlled --species=primary
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy --preload

        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-chip --species=primary --controlled
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-chip --species=primary --allgt
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-feat
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-featweight
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy -t candidacy --preload
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy --delete-nn-cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> db_list = ut.get_argval('--dblist', type_=list, default=['testdb1'])
        >>> result = precfg_dbs(db_list)
        >>> print(result)
    """
    import ibeis.init.main_helpers
    import ibeis
    if db_list == ['candidacy']:
        db_list = experiment_configs.get_candidacy_dbnames()  # HACK
    print('db_list = %s' % (ut.list_str(db_list),))
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=[])
    for db in db_list:
        ibs = ibeis.opendb(db=db)
        ibs, qaids, daids = ibeis.init.main_helpers.testdata_ibeis(verbose=False, ibs=ibs)
        precfg(ibs, qaids, daids, test_cfg_name_list)


def precfg(ibs, qaids, daids, test_cfg_name_list):
    r"""
    Helper to precompute information

    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (list):  query annotation ids
        daids (list):  database annotation ids
        test_cfg_name_list (list):

    Returns:
        ?:

    CommandLine:
        python -m ibeis.experiments.experiment_harness --exec-precfg -t custom --expt-preload
        # Repeatidly causes freezes (IF CHIPS PARALLEL IS ON)
        ./reset_dbs.py
        python -m ibeis.experiments.experiment_harness --exec-precfg -t custom --preload
        python -m ibeis.experiments.experiment_harness --exec-precfg -t custom --preload-chips --controlled
        python -m ibeis.experiments.experiment_harness --exec-precfg -t custom --preload --species=zebra_plains --serial
        python -m ibeis.experiments.experiment_harness --exec-precfg -t custom --preload --expt-indexer --species=zebra_plains --serial
        python -m ibeis.experiments.experiment_harness --exec-precfg -t custom --preload --expt-indexer --species=zebra_plains --serial

        python -m ibeis.experiments.experiment_harness --exec-precfg --delete-nn-cache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> ibs, qaids, daids = main_helpers.testdata_ibeis(verbose=False)
        >>> test_cfg_name_list = ut.get_argval('-t', type_=list, default=[])
        >>> result = precfg(ibs, qaids, daids, test_cfg_name_list)
        >>> print(result)
    """
    cfg_list, cfgx2_lbl, cfgdict_list = experiment_helpers.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)

    lbl = '[harn] ENSURE_CFG ' + str(test_cfg_name_list)
    nCfg     = len(cfg_list)   # number of configurations (cols)
    dbname = ibs.get_dbname()

    cfgiter = ut.ProgressIter(cfg_list, nTotal=nCfg, lbl=lbl,
                              freq=1, autoadjust=False)

    flag = False
    if ut.get_argflag('--delete-nn-cache'):
        ibs.delete_neighbor_cache()
        flag = True

    for cfgx, query_cfg in enumerate(cfgiter):
        print('')
        ut.colorprint(query_cfg.get_cfgstr(), 'turquoise')
        verbose = True
        with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
            qreq_ = ibs.new_query_request(qaids, daids, verbose=True, query_cfg=query_cfg)
            if ut.get_argflag('--preload'):
                qreq_.lazy_preload(verbose=verbose)
                flag = True
            if ut.get_argflag('--preload-chip'):
                qreq_.ensure_chips(verbose=verbose, extra_tries=1)
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
