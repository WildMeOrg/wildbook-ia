"""
Runs many queries and keeps track of some results
"""
from __future__ import absolute_import, division, print_function
import sys
import textwrap
import numpy as np
#import six
import utool
import utool as ut
from ibeis.experiments import experiment_helpers
#from ibeis.experiments import experiment_configs
#from ibeis.experiments import experiment_printres
#from ibeis.experiments import experiment_drawing
from ibeis.experiments import experiment_storage
from ibeis.experiments import annotation_configs
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]')

BATCH_MODE = '--nobatch' not in sys.argv
NOMEMORY   = '--nomemory' in sys.argv
TESTRES_VERBOSITY = 2 - (2 * utool.QUIET)
NOCACHE_TESTRES =  utool.get_argflag('--nocache-testres', False)
TEST_INFO = True


def testdata_expts(defaultdb='testdb1',
                   default_acfgstr_name_list=['candidacy:qsize=20,dper_name=1,dsize=10',
                                              'candidacy:qsize=20,dper_name=10,dsize=100'],
                   default_test_cfg_name_list=['custom', 'custom:fg_on=False']):
    """
    Command line interface to quickly get testdata for test_results
    """
    import ibeis
    #from ibeis.experiments import experiment_helpers
    ibs = ibeis.opendb(defaultdb=defaultdb)
    acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstr_name_list)
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=default_test_cfg_name_list)
    test_result_list = run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list)
    test_result = experiment_storage.combine_test_results(ibs, test_result_list)
    return ibs, test_result
    #return ibs, test_result_list


def run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list):
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
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> default_acfgstrs = ['candidacy:qsize=20,dper_name=1,dsize=10', 'candidacy:qsize=20,dper_name=10,dsize=100']
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstrs)
        >>> test_cfg_name_list = ut.get_argval('-t', type_=list, default=['custom', 'custom:fg_on=False'])
        >>> test_result_list = run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list)
    """
    # Generate list of database annotation configurations
    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, acfg_name_list)
    # Generate list of query pipeline param configs
    cfg_list, cfgx2_lbl, cfgdict_list = experiment_helpers.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)

    if not utool.QUIET:
        ut.colorprint(textwrap.dedent("""

        [harn]================
        [harn] experiment_harness.test_configurations2()""").strip(), 'white')
        ut.colorprint('[harn] Testing %d parameter configurations' % len(cfg_list), 'white')
        ut.colorprint('[harn]    with %d annotations configurations' % len(acfg_list), 'white')
        ut.colorprint('[harn]    running %d queries' % (len(acfg_list) * len(cfg_list)), 'white')

    test_result_list = []

    testnameid = ibs.get_dbname() + ' ' + str(test_cfg_name_list) + str(acfg_name_list)
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list) + str(acfg_name_list)

    nAcfg = len(acfg_list)

    expanded_aids_iter = ut.ProgressIter(expanded_aids_list, lbl='annot config', freq=1, autoadjust=False)

    if ut.get_argflag('--acfginfo'):
        # Print info about annots for the test
        ut.colorprint('Requested AcfgInfo for tests... ', 'red')
        nonvaried_compressed_dict, varied_compressed_dict_list = annotation_configs.compress_acfg_list_for_printing(acfg_list)
        print('non-varied aidcfg = ' + ut.dict_str(nonvaried_compressed_dict))
        for acfgx, (qaids, daids) in enumerate(expanded_aids_list):
            ut.colorprint('+-----------', 'white')
            print('acfg = ' + ut.dict_str(varied_compressed_dict_list[acfgx]))
            #_ = ibs.get_annotconfig_stats(qaids, daids)
            ut.colorprint('L___________', 'white')
        ut.colorprint('Finished Reporting AcfgInfo. Exiting', 'red')
        sys.exit(1)

    for acfgx, (qaids, daids) in enumerate(expanded_aids_iter):
        if len(qaids) == 0:
            raise AssertionError('[harness] No query annotations specified')
        if len(daids) == 0:
            raise AssertionError('[harness] No query annotations specified')
        acfg = acfg_list[acfgx]
        ut.colorprint('\n---Annot config', 'turquoise')
        test_result = run_test_configurations(ibs, qaids, daids, cfg_list, cfgx2_lbl, cfgdict_list, lbl, testnameid, acfgx, nAcfg)
        test_result.acfg = acfg
        test_result.test_cfg_name_list = test_cfg_name_list
        test_result_list.append(test_result)
    return test_result_list


@profile
def run_test_configurations(ibs, qaids, daids, cfg_list, cfgx2_lbl, cfgdict_list, lbl, testnameid, acfgx, nAcfg):
    #orig_query_cfg = ibs.cfg.query_cfg  # Remember original query config

    # Parse cfgstr list
    #cfgstr_list = [query_cfg.get_cfgstr() for query_cfg in cfg_list]
    #for cfgstr in cfgstr_list:
    #    cfgstr_terms = cfgstr.split(')_')
    #    cfgstr_terms = [x + ')_' for x in cfgstr_terms[1:-1]] + [cfgstr_terms[-1]]  # NOQA

    #print('cfgx2_lbl = ' + ut.list_str(cfgx2_lbl))
    #print('cfgstr_list = ' + ut.list_str(cfgstr_list))

    nCfg     = len(cfg_list)   # number of configurations (cols)
    dbname = ibs.get_dbname()

    DRY_RUN =  ut.get_argflag('--dryrun')  # dont actually query. Just print labels and stuff

    print('Constructing query requests')
    cfgx2_qreq_ = [ibs.new_query_request(qaids, daids, verbose=False, query_cfg=query_cfg) for query_cfg in cfg_list]

    #USE_BIG_TEST_CACHE = ut.get_argflag('--use-testcache')
    USE_BIG_TEST_CACHE = not ut.get_argflag(('--no-use-testcache', '--nocache-test'))
    if USE_BIG_TEST_CACHE:
        bigtest_cachestr = ut.hashstr_arr27([qreq_.get_cfgstr(with_query=True) for qreq_ in cfgx2_qreq_], ibs.get_dbname() + '_cfgs')
        bigtest_cachedir = './BIG_TEST_CACHE'
        bigtest_cachename = 'BIGTESTCACHE'
        ut.ensuredir(bigtest_cachedir)
        try:
            test_result = ut.load_cache(bigtest_cachedir, bigtest_cachename, bigtest_cachestr)
            test_result.cfgdict_list = cfgdict_list
        except IOError:
            pass
        else:
            ut.colorprint('Experiment Harness Cache Hit... Returning', 'turquoise')
            return test_result

    #qreq_ = ibs.new_query_request(qaids, d aids, verbose=True, query_cfg=ibs.cfg.query_cfg)

    cfgx2_cfgresinfo = []
    #with utool.Timer('experiment_harness'):
    cfgiter = ut.ProgressIter(cfg_list, lbl='query config', freq=1, autoadjust=False, parent_index=acfgx, parent_nTotal=nAcfg)
    # Run each test configuration
    # Query Config / Col Loop
    #for cfgx, query_cfg in enumerate(cfg_list):
    for cfgx, query_cfg in enumerate(cfgiter):
        #print('+--- REQUESTING TEST CONFIG ---')
        #print(query_cfg.get_cfgstr())
        ut.colorprint(query_cfg.get_cfgstr(), 'turquoise')
        qreq_ = cfgx2_qreq_[cfgx]

        with utool.Indenter('[%s cfg %d/%d]' % (dbname, (acfgx * nCfg) + cfgx * + 1, nCfg * nAcfg)):
            if DRY_RUN:
                continue
            # Set data to the current config
            #ibs.set_query_cfg(query_cfg)  # TODO: make this not even matter
            # Run the test / read cache
            if USE_BIG_TEST_CACHE:
                # smaller cache for individual configuration runs
                smalltest_cfgstr = qreq_.get_cfgstr(with_query=True)
                smalltest_cachedir = ut.unixjoin(bigtest_cachedir, 'small_tests')
                smalltest_cachename = 'smalltest'
                ut.ensuredir(smalltest_cachedir)
                try:
                    cfgres_info = ut.load_cache(smalltest_cachedir, smalltest_cachename, smalltest_cfgstr)
                except IOError:
                    cfgres_info = get_query_result_info(qreq_)
                    ut.save_cache(smalltest_cachedir, smalltest_cachename, smalltest_cfgstr, cfgres_info)
            else:
                cfgres_info = get_query_result_info(qreq_)
            #qx2_bestranks, qx2_next_bestranks, qx2_scorediff, qx2_avepercision = cfgres_info
        if not NOMEMORY:
            # Store the results
            cfgx2_cfgresinfo.append(cfgres_info)
            #cfgx2_qreq_.append(qreq_)
        else:
            cfgx2_qreq_[cfgx] = None
        print('\n +------ \n')

    if not utool.QUIET:
        ut.colorprint('[harn] Completed running test configurations', 'white')
        #print(msg)
    if DRY_RUN:
        print('ran tests dryrun mode. Cannot Print. exiting')
        return
    if NOMEMORY:
        print('ran tests in memory savings mode. Cannot Print. exiting')
        return
    # Reset query config so nothing unexpected happens
    # TODO: should probably just use a cfgdict to build a list of QueryRequest
    # objects. That would avoid the entire problem
    #ibs.set_query_cfg(orig_query_cfg)
    test_result = experiment_storage.TestResult(cfg_list, cfgx2_lbl, lbl, testnameid, cfgx2_cfgresinfo, cfgx2_qreq_, qaids)
    test_result.cfgdict_list = cfgdict_list
    test_result.aidcfg = None
    if USE_BIG_TEST_CACHE:
        ut.save_cache(bigtest_cachedir, bigtest_cachename, bigtest_cachestr, test_result)
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
        scorediff = scorefactor = scorelogfactor = scoreexpdiff = None
    else:

        gt_aid = sorted_aids[gt_rank][0]
        gf_aid = sorted_aids[gf_rank][0]
        gt_raw_score = sorted_nscores[gt_rank]
        gf_raw_score = sorted_nscores[gf_rank]
        # different comparison methods
        scorediff      = gt_raw_score - gf_raw_score
        scorefactor    = gt_raw_score / gf_raw_score
        scorelogfactor = np.log(gt_raw_score) / np.log(gf_raw_score)
        scoreexpdiff   = np.exp(gt_raw_score) - np.log(gf_raw_score)
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
        scorelogfactor=scorelogfactor,
        scoreexpdiff=scoreexpdiff
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
