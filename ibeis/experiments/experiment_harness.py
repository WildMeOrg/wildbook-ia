"""
Runs many queries and keeps track of some results
"""
from __future__ import absolute_import, division, print_function
import sys
import textwrap
import numpy as np
#import six
import utool as ut
from functools import partial
from ibeis.experiments import experiment_helpers
#from ibeis.experiments import experiment_configs
#from ibeis.experiments import experiment_printres
#from ibeis.experiments import experiment_drawing
from ibeis.experiments import experiment_storage
#from ibeis.experiments import annotation_configs
#from ibeis.experiments import cfghelpers
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[expt_harn]')

NOMEMORY = ut.get_argflag('--nomemory')
TESTRES_VERBOSITY = 2 - (2 * ut.QUIET)
NOCACHE_TESTRES =  ut.get_argflag(('--nocache-testres', '--nocache-big'), False)
USE_BIG_TEST_CACHE = not ut.get_argflag(('--no-use-testcache', '--nocache-test')) and ut.USE_CACHE and not NOCACHE_TESTRES
TEST_INFO = True

DRY_RUN =  ut.get_argflag(('--dryrun', '--dry'))  # dont actually query. Just print labels and stuff


def run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list, use_cache=None, qaid_override=None):
    """
    Loops over annot configs.

    Try and use this function as a starting point to clean up this module.
    The code is getting too untenable.

    CommandLine:
        python -m ibeis.experiments.experiment_harness --exec-run_test_configurations2

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> default_acfgstrs = ['controlled:qsize=20,dpername=1,dsize=10', 'controlled:qsize=20,dpername=10,dsize=100']
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstrs)
        >>> test_cfg_name_list = ut.get_argval('-t', type_=list, default=['custom', 'custom:fg_on=False'])
        >>> use_cache = False
        >>> test_result_list = run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list, use_cache)
    """
    testnameid = ibs.get_dbname() + ' ' + str(test_cfg_name_list) + str(acfg_name_list)
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list) + str(acfg_name_list)

    # Generate list of database annotation configurations
    if len(acfg_name_list) == 0:
        raise ValueError('must give acfg name list')

    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, acfg_name_list, qaid_override=qaid_override)
    # Generate list of query pipeline param configs
    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(test_cfg_name_list, ibs=ibs)

    cfgx2_lbl = experiment_helpers.get_varied_pipecfg_lbls(cfgdict_list)

    if ut.NOT_QUIET:
        ut.colorprint(textwrap.dedent("""

        [harn]================
        [harn] experiment_harness.test_configurations2()""").strip(), 'white')
        msg = '[harn] Running %s using %s and %s' % (
            ut.quantity_str('test', len(acfg_list) * len(pipecfg_list)),
            ut.quantity_str('pipeline config', len(pipecfg_list)),
            ut.quantity_str('annot config', len(acfg_list)),
        )
        ut.colorprint(msg, 'white')

    test_result_list = []

    nAcfg = len(acfg_list)

    expanded_aids_iter = ut.ProgressIter(expanded_aids_list,
                                         lbl='annot config',
                                         freq=1, autoadjust=False,
                                         enabled=ut.NOT_QUIET)

    #if ut.get_argflag(('--pcfginfo', '--pipecfginfo')):
    #    ut.colorprint('Requested PcfgInfo for tests... ', 'red')
    #    for pcfgx, pipecfg in enumerate(pipecfg_list):
    #        print('+--- %d / %d ===' % (pcfgx, (len(pipecfg_list))))
    #        print(pipecfg.get_cfgstr())
    #        print('L___')
    #    ut.colorprint('Finished Reporting PcfgInfo. Exiting', 'red')
    #    sys.exit(1)

    #if ut.get_argflag(('--acfginfo', '--aidcfginfo')):
    #    # Print info about annots for the test
    #    ut.colorprint('Requested AcfgInfo for tests... ', 'red')
    #    annotation_configs.print_acfg_list(acfg_list, expanded_aids_list, ibs)
    #    ut.colorprint('Finished Reporting AcfgInfo. Exiting', 'red')
    #    sys.exit(1)

    for acfgx, (qaids, daids) in enumerate(expanded_aids_iter):
        if len(qaids) == 0:
            raise AssertionError('[harness] No query annotations specified')
        if len(daids) == 0:
            raise AssertionError('[harness] No database annotations specified')
        acfg = acfg_list[acfgx]
        if ut.NOT_QUIET:
            ut.colorprint('\n---Annot config testnameid=%r' % (testnameid,), 'turquoise')
        subindexer_partial = partial(ut.ProgressIter, parent_index=acfgx,
                                     parent_nTotal=nAcfg, enabled=ut.NOT_QUIET)
        test_result = run_test_configurations(ibs, qaids, daids, pipecfg_list,
                                              cfgx2_lbl, cfgdict_list, lbl,
                                              testnameid, use_cache=use_cache,
                                              subindexer_partial=subindexer_partial)
        if DRY_RUN:
            continue
        test_result.acfg = acfg
        test_result.test_cfg_name_list = test_cfg_name_list
        test_result_list.append(test_result)
    if DRY_RUN:
        print('DRYRUN: Cannot continue past run_test_configurations2')
        sys.exit(0)

    return test_result_list


def get_big_test_cache_info(ibs, cfgx2_qreq_):
    if ut.is_developer():
        import ibeis
        from os.path import dirname, join
        repodir = dirname(ut.get_module_dir(ibeis))
        bt_cachedir = join(repodir, 'BIG_TEST_CACHE')
    else:
        bt_cachedir = './BIG_TEST_CACHE'
    ut.ensuredir(bt_cachedir)
    bt_cachestr = ut.hashstr_arr27([qreq_.get_cfgstr(with_query=True) for qreq_ in cfgx2_qreq_], ibs.get_dbname() + '_cfgs')
    bt_cachename = 'BIGTESTCACHE'
    return bt_cachedir, bt_cachename, bt_cachestr


@profile
def run_test_configurations(ibs, qaids, daids, pipecfg_list, cfgx2_lbl,
                            cfgdict_list, lbl, testnameid,
                            use_cache=None,
                            subindexer_partial=ut.ProgressIter):
    """

    CommandLine:
        python -m ibeis.experiments.experiment_harness --exec-run_test_configurations2

    """
    cfgslice = None
    if cfgslice is not None:
        pipecfg_list = pipecfg_list[cfgslice]

    dbname = ibs.get_dbname()

    if ut.NOT_QUIET:
        print('Constructing query requests')
    cfgx2_qreq_ = [ibs.new_query_request(qaids, daids, verbose=False, query_cfg=pipe_cfg)
                   for pipe_cfg in ut.ProgressIter(pipecfg_list, lbl='Building qreq_', enabled=False)]

    if use_cache is None:
        use_cache = USE_BIG_TEST_CACHE

    if use_cache:
        get_big_test_cache_info(ibs, cfgx2_qreq_)
        try:
            bt_cachedir, bt_cachename, bt_cachestr = get_big_test_cache_info(ibs, cfgx2_qreq_)
            test_result = ut.load_cache(bt_cachedir, bt_cachename, bt_cachestr)
            test_result.cfgdict_list = cfgdict_list
            test_result.cfgx2_lbl = cfgx2_lbl  # hack override
        except IOError:
            pass
        else:
            if ut.NOT_QUIET:
                ut.colorprint('Experiment Harness Cache Hit... Returning', 'turquoise')
            return test_result

    cfgx2_cfgresinfo = []
    #nPipeCfg = len(pipecfg_list)
    cfgiter = subindexer_partial(range(len(cfgx2_qreq_)), lbl='query config', freq=1, adjust=False, separate=True)
    # Run each pipeline configuration
    prev_feat_cfgstr = None
    for cfgx in cfgiter:
        qreq_ = cfgx2_qreq_[cfgx]

        ut.colorprint('testnameid=%r' % (testnameid,), 'green')
        ut.colorprint('annot_cfgstr = %s' % (qreq_.get_cfgstr(with_query=True, with_pipe=False),), 'yellow')
        ut.colorprint('pipe_cfgstr= %s' % (qreq_.get_cfgstr(with_data=False),), 'turquoise')
        ut.colorprint('pipe_hashstr = %s' % (qreq_.get_pipe_hashstr(),), 'teal')
        if DRY_RUN:
            continue

        indent_prefix = '[%s cfg %d/%d]' % (
            dbname,
            (cfgiter.parent_index * cfgiter.nTotal) + cfgx ,  # cfgiter.count (doesnt work when quiet)
            cfgiter.nTotal * cfgiter.parent_nTotal
        )

        #with ut.Indenter('[%s cfg %d/%d]' % (dbname, (acfgx * nCfg) + cfgx * + 1, nCfg * nAcfg)):
        with ut.Indenter(indent_prefix):
            # Run the test / read cache
            _need_compute = True
            if use_cache:
                # smaller cache for individual configuration runs
                st_cfgstr = qreq_.get_cfgstr(with_query=True)
                st_cachedir = ut.unixjoin(bt_cachedir, 'small_tests')
                st_cachename = 'smalltest'
                ut.ensuredir(st_cachedir)
                try:
                    cfgres_info = ut.load_cache(st_cachedir, st_cachename, st_cfgstr)
                except IOError:
                    _need_compute = True
                else:
                    _need_compute = False
            if _need_compute:
                if prev_feat_cfgstr is not None and prev_feat_cfgstr != qreq_.qparams.feat_cfgstr:
                    # Clear features to preserve memory
                    ibs.clear_table_cache()
                    #qreq_.ibs.print_cachestats_str()
                cfgres_info = get_query_result_info(qreq_)
                prev_feat_cfgstr = qreq_.qparams.feat_cfgstr  # record previous feature configuration

                if use_cache:
                    ut.save_cache(st_cachedir, st_cachename, st_cfgstr, cfgres_info)
        if not NOMEMORY:
            # Store the results
            cfgx2_cfgresinfo.append(cfgres_info)
        else:
            cfgx2_qreq_[cfgx] = None
    if ut.NOT_QUIET:
        ut.colorprint('[harn] Completed running test configurations', 'white')
    if DRY_RUN:
        print('ran tests dryrun mode.')
        return
    if NOMEMORY:
        print('ran tests in memory savings mode. Cannot Print. exiting')
        return
    # Store all pipeline config results in a test result object
    test_result = experiment_storage.TestResult(pipecfg_list, cfgx2_lbl, cfgx2_cfgresinfo, cfgx2_qreq_)
    test_result.testnameid = testnameid
    test_result.lbl = lbl
    test_result.cfgdict_list = cfgdict_list
    test_result.aidcfg = None
    if use_cache:
        ut.save_cache(bt_cachedir, bt_cachename, bt_cachestr, test_result)
    return test_result


@profile
def get_qres_name_result_info(ibs, qres):
    """
    these are results per query we care about
     * gt (best correct match) and gf (best incorrect match) rank, their score
       and the difference

    """
    qaid = qres.get_qaid()
    qnid = ibs.get_annot_name_rowids(qaid)
    nscoretup = qres.get_nscoretup(ibs)
    (sorted_nids, sorted_nscores, sorted_aids, sorted_scores)  = nscoretup
    #sorted_score_diff = -np.diff(sorted_nscores.tolist())
    sorted_nids = np.array(sorted_nids)
    is_positive  = sorted_nids == qnid
    is_negative = np.logical_and(~is_positive, sorted_nids > 0)
    gt_rank = None if not np.any(is_positive) else np.where(is_positive)[0][0]
    gf_rank = None if not np.any(is_negative) else np.nonzero(is_negative)[0][0]

    if gt_rank is None or gf_rank is None:
        NEW = True
        if NEW:
            # Should a random groundtruth result be chosen if it exists here?
            gt_aids = qres.get_groundtruth_aids(ibs)
            if len(gt_aids) > 0:
                gt_aid = gt_aids[0]
            else:
                gt_aid = None
        else:
            gt_aid = None
        gf_aid = None
        gt_raw_score = None
        gf_raw_score = None
        scorediff = scorefactor = None
        #scorelogfactor = scoreexpdiff = None
    else:

        gt_aid = sorted_aids[gt_rank][0]
        gf_aid = sorted_aids[gf_rank][0]
        gt_raw_score = sorted_nscores[gt_rank]
        gf_raw_score = sorted_nscores[gf_rank]
        # different comparison methods
        scorediff      = gt_raw_score - gf_raw_score
        scorefactor    = gt_raw_score / gf_raw_score
        #scorelogfactor = np.log(gt_raw_score) / np.log(gf_raw_score)
        #scoreexpdiff   = np.exp(gt_raw_score) - np.log(gf_raw_score)

        # TEST SCORE COMPARISON METHODS
        #truescore  = np.random.rand(4)
        #falsescore = np.random.rand(4)
        #score_diff      = truescore - falsescore
        #scorefactor    = truescore / falsescore
        #scorelogfactor = np.log(truescore) / np.log(falsescore)
        #scoreexpdiff   = np.exp(truescore) - np.exp(falsescore)
        #for x in [score_diff, scorefactor, scorelogfactor, scoreexpdiff]:
        #    print(x.argsort())

    qresinfo_dict = dict(
        bestranks=gt_rank,
        next_bestranks=gf_rank,
        # TODO remove prev dup entries
        gt_rank=gt_rank,
        gf_rank=gf_rank,
        gt_aid=gt_aid,
        gf_aid=gf_aid,
        gt_raw_score=gt_raw_score,
        gf_raw_score=gf_raw_score,
        scorediff=scorediff,
        scorefactor=scorefactor,
        #scorelogfactor=scorelogfactor,
        #scoreexpdiff=scoreexpdiff
    )
    return qresinfo_dict


@profile
def get_query_result_info(qreq_):
    """
    Helper function.

    Runs queries of a specific configuration returns the best rank of each query

    Args:
        qaids (list) : query annotation ids
        daids (list) : database annotation ids

    Returns:
        qx2_bestranks

    CommandLine:
        python -m ibeis.experiments.experiment_harness --test-get_query_result_info
        python -m ibeis.experiments.experiment_harness --test-get_query_result_info:0
        python -m ibeis.experiments.experiment_harness --test-get_query_result_info:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaids = ibs.get_valid_aids()[0:3]
        >>> daids = ibs.get_valid_aids()[0:5]
        >>> qreq_ = ibs.new_query_request(qaids, daids, verbose=True, cfgdict={}, query_cfg=ibs.cfg.query_cfg)
        >>> cfgres_info = get_query_result_info(qreq_)
        >>> print(ut.dict_str(cfgres_info))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> #cfgdict = dict(codename='vsone')
        >>> # ibs.cfg.query_cfg.codename = 'vsone'
        >>> qaids = ibs.get_valid_aids()[0:3]
        >>> daids = ibs.get_valid_aids()[0:5]
        >>> qreq_ = ibs.new_query_request(qaids, daids, verbose=True, cfgdict={}, query_cfg=ibs.cfg.query_cfg)
        >>> cfgres_info = get_query_result_info(qreq_)
        >>> print(ut.dict_str(cfgres_info))

    Ignore:
        for qaid, qres in six.iteritems(qaid2_qres):
            break
        for qaid, qres in six.iteritems(qaid2_qres):
            qres.ishow_top(ibs)
    """
    # Execute or load query
    #ibs.get_annot_name_rowids(qaids)
    #ibs.get_annot_name_rowids(daids)

    #qreq_ = ibs.new_query_request(qaids, daids, verbose=True)
    #qx2_qres = ibs.query_chips(qreq_=qreq_)
    #assert [x.qaid for x in qx2_qres] == qaids, 'request missmatch'
    #qaid2_qres, qreq_ = ibs._query_chips4(qaids, daids, return_request=True)
    #qx2_qres = ut.dict_take(qaid2_qres, qaids)
    # Get the groundtruth that could have been matched in this experiment
    qx2_qres = qreq_.ibs.query_chips(qreq_=qreq_)
    qaids = qreq_.get_external_qaids()
    daids = qreq_.get_external_daids()
    ibs = qreq_.ibs
    qx2_gtaids = ibs.get_annot_groundtruth(qaids, daid_list=daids)
    # Get the groundtruth ranks and accuracy measures
    qx2_qresinfo = [get_qres_name_result_info(ibs, qres) for qres in qx2_qres]

    cfgres_info = ut.dict_stack(qx2_qresinfo, 'qx2_')
    keys = qx2_qresinfo[0].keys()
    for key in keys:
        'qx2_' + key
        ut.get_list_column(qx2_qresinfo, key)

    qx2_avepercision = np.array(
        [qres.get_average_percision(ibs=ibs, gt_aids=gt_aids) for
         (qres, gt_aids) in zip(qx2_qres, qx2_gtaids)])
    cfgres_info['qx2_avepercision'] = qx2_avepercision
    # Compute mAP score  # TODO: use mAP score
    # (Actually map score doesn't make much sense if using name scoring
    #mAP = qx2_avepercision[~np.isnan(qx2_avepercision)].mean()  # NOQA
    cfgres_info['qx2_bestranks'] = ut.replace_nones(cfgres_info['qx2_bestranks'] , -1)
    return cfgres_info


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.experiments.experiment_harness; utool.doctest_funcs(ibeis.experiments.experiment_harness, allexamples=True)"
        python -c "import utool, ibeis.experiments.experiment_harness; utool.doctest_funcs(ibeis.experiments.experiment_harness)"
        python -m ibeis.experiments.experiment_harness
        python -m ibeis.experiments.experiment_harness --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
