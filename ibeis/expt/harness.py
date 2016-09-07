# -*- coding: utf-8 -*-
"""
Runs many queries and keeps track of some results
"""
from __future__ import absolute_import, division, print_function
import sys
import textwrap
import numpy as np
import utool as ut
from functools import partial
from ibeis.expt import experiment_helpers
from os.path import dirname, join
from ibeis.expt import test_result
print, rrr, profile = ut.inject2(__name__, '[expt_harn]')

NOMEMORY = ut.get_argflag('--nomemory')
TESTRES_VERBOSITY = 2 - (2 * ut.QUIET)
NOCACHE_TESTRES =  ut.get_argflag(('--nocache-testres', '--nocache-big'), False)
USE_BIG_TEST_CACHE = (not ut.get_argflag(('--no-use-testcache',
                                          '--nocache-test')) and ut.USE_CACHE and
                      not NOCACHE_TESTRES)
USE_BIG_TEST_CACHE = False
TEST_INFO = True

# dont actually query. Just print labels and stuff
DRY_RUN =  ut.get_argflag(('--dryrun', '--dry'))


#def pre_test_expand(ibs, acfg_name_list, test_cfg_name_list,
#                    use_cache=None, qaid_override=None,
#                    daid_override=None, initial_aids=None):
#    """ FIXME: incorporate into run_test_configs2 """
#    # Generate list of database annotation configurations
#    if len(acfg_name_list) == 0:
#        raise ValueError('must give acfg name list')

#    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
#        ibs, acfg_name_list, qaid_override=qaid_override,
#        daid_override=daid_override, initial_aids=initial_aids,
#        use_cache=use_cache)

#    # Generate list of query pipeline param configs
#    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(
#        test_cfg_name_list, ibs=ibs)
#    return expanded_aids_list, pipecfg_list


def run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list,
                             use_cache=None, qaid_override=None,
                             daid_override=None, initial_aids=None):
    """
    Loops over annot configs.

    Try and use this function as a starting point to clean up this module.
    The code is getting too untenable.

    CommandLine:
        python -m ibeis.expt.harness --exec-run_test_configurations2

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.expt.harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> default_acfgstrs = ['controlled:qsize=20,dpername=1,dsize=10', 'controlled:qsize=20,dpername=10,dsize=100']
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstrs)
        >>> test_cfg_name_list = ut.get_argval(('-t', '-p')), type_=list, default=['custom', 'custom:fg_on=False'])
        >>> use_cache = False
        >>> testres_list = run_test_configurations2(ibs, acfg_name_list, test_cfg_name_list, use_cache)
    """
    print('[harn] run_test_configurations2')
    # Generate list of database annotation configurations
    if len(acfg_name_list) == 0:
        raise ValueError('must give acfg name list')

    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
        ibs, acfg_name_list, qaid_override=qaid_override,
        daid_override=daid_override, initial_aids=initial_aids,
        use_cache=use_cache)

    # Generate list of query pipeline param configs
    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(
        test_cfg_name_list, ibs=ibs)

    cfgx2_lbl = experiment_helpers.get_varied_pipecfg_lbls(cfgdict_list)
    # NOTE: Can specify --pcfginfo or --acfginfo

    if ut.NOT_QUIET:
        ut.colorprint(textwrap.dedent("""

        [harn]================
        [harn] harness.test_configurations2()""").strip(), 'white')
        msg = '[harn] Running %s using %s and %s' % (
            ut.quantstr('test', len(acfg_list) * len(pipecfg_list)),
            ut.quantstr('pipeline config', len(pipecfg_list)),
            ut.quantstr('annot config', len(acfg_list)),
        )
        ut.colorprint(msg, 'white')

    testres_list = []

    nAcfg = len(acfg_list)

    testnameid = ibs.get_dbname() + ' ' + str(test_cfg_name_list) + str(acfg_name_list)
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list) + str(acfg_name_list)
    expanded_aids_iter = ut.ProgressIter(expanded_aids_list,
                                         lbl='annot config',
                                         freq=1, autoadjust=False,
                                         enabled=ut.NOT_QUIET)

    for acfgx, (qaids, daids) in enumerate(expanded_aids_iter):
        assert len(qaids) != 0, (
            '[harness] No query annotations specified')
        assert len(daids) != 0, (
            '[harness] No database annotations specified')
        acfg = acfg_list[acfgx]
        if ut.NOT_QUIET:
            ut.colorprint('\n---Annot config testnameid=%r' % (
                testnameid,), 'turquoise')
        subindexer_partial = partial(ut.ProgressIter, parent_index=acfgx,
                                     parent_nTotal=nAcfg, enabled=ut.NOT_QUIET)
        testres = make_single_testres(ibs, qaids, daids, pipecfg_list,
                                      cfgx2_lbl, cfgdict_list, lbl, testnameid,
                                      use_cache=use_cache,
                                      subindexer_partial=subindexer_partial)
        if DRY_RUN:
            continue
        testres.acfg = acfg
        testres.test_cfg_name_list = test_cfg_name_list
        testres_list.append(testres)
    if DRY_RUN:
        print('DRYRUN: Cannot continue past run_test_configurations2')
        sys.exit(0)

    return testres_list


def get_big_test_cache_info(ibs, cfgx2_qreq_):
    """
    Args:
        ibs (ibeis.IBEISController):
        cfgx2_qreq_ (dict):
    """
    if ut.is_developer():
        import ibeis
        repodir = dirname(ut.get_module_dir(ibeis))
        bt_cachedir = join(repodir, 'BIG_TEST_CACHE2')
    else:
        bt_cachedir = join(ibs.get_cachedir(), 'BIG_TEST_CACHE2')
        #bt_cachedir = './localdata/BIG_TEST_CACHE2'
    ut.ensuredir(bt_cachedir)
    bt_cachestr = ut.hashstr_arr27([
        qreq_.get_cfgstr(with_input=True)
        for qreq_ in cfgx2_qreq_],
        ibs.get_dbname() + '_cfgs')
    bt_cachename = 'BIGTESTCACHE2'
    return bt_cachedir, bt_cachename, bt_cachestr


@profile
def make_single_testres(ibs, qaids, daids, pipecfg_list, cfgx2_lbl,
                        cfgdict_list, lbl, testnameid, use_cache=None,
                        subindexer_partial=ut.ProgressIter):
    """
    CommandLine:
        python -m ibeis.expt.harness --exec-run_test_configurations2
    """
    cfgslice = None
    if cfgslice is not None:
        pipecfg_list = pipecfg_list[cfgslice]

    dbname = ibs.get_dbname()

    if ut.NOT_QUIET:
        print('[harn] Make single testres')

    cfgx2_qreq_ = [
        ibs.new_query_request(qaids, daids, verbose=False, query_cfg=pipe_cfg)
        for pipe_cfg in ut.ProgressIter(pipecfg_list, lbl='Building qreq_',
                                        enabled=False)
    ]

    if use_cache is None:
        use_cache = USE_BIG_TEST_CACHE

    if use_cache:
        get_big_test_cache_info(ibs, cfgx2_qreq_)
        try:
            cachetup = get_big_test_cache_info(ibs, cfgx2_qreq_)
            testres = ut.load_cache(*cachetup)
            testres.cfgdict_list = cfgdict_list
            testres.cfgx2_lbl = cfgx2_lbl  # hack override
        except IOError:
            pass
        else:
            if ut.NOT_QUIET:
                ut.colorprint('[harn] single testres cache hit... returning', 'turquoise')
            return testres

    if ibs.table_cache:
        # HACK
        prev_feat_cfgstr = None

    cfgx2_cfgresinfo = []
    #nPipeCfg = len(pipecfg_list)
    cfgiter = subindexer_partial(range(len(cfgx2_qreq_)),
                                 lbl='query config',
                                 freq=1, adjust=False,
                                 separate=True)
    # Run each pipeline configuration
    for cfgx in cfgiter:
        qreq_ = cfgx2_qreq_[cfgx]

        ut.colorprint('testnameid=%r' % (
            testnameid,), 'green')
        ut.colorprint('annot_cfgstr = %s' % (
            qreq_.get_cfgstr(with_input=True, with_pipe=False),), 'yellow')
        ut.colorprint('pipe_cfgstr= %s' % (
            qreq_.get_cfgstr(with_data=False),), 'turquoise')
        ut.colorprint('pipe_hashstr = %s' % (
            qreq_.get_pipe_hashid(),), 'teal')
        if DRY_RUN:
            continue

        indent_prefix = '[%s cfg %d/%d]' % (
            dbname,
            # cfgiter.count (doesnt work when quiet)
            (cfgiter.parent_index * cfgiter.nTotal) + cfgx ,
            cfgiter.nTotal * cfgiter.parent_nTotal
        )

        with ut.Indenter(indent_prefix):
            # Run the test / read cache
            _need_compute = True
            if use_cache:
                # smaller cache for individual configuration runs
                st_cfgstr = qreq_.get_cfgstr(with_input=True)
                bt_cachedir = cachetup[0]
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
                assert not ibs.table_cache
                if ibs.table_cache:
                    if (len(prev_feat_cfgstr is not None and
                            prev_feat_cfgstr != qreq_.qparams.feat_cfgstr)):
                        # Clear features to preserve memory
                        ibs.clear_table_cache()
                        #qreq_.ibs.print_cachestats_str()
                cfgres_info = get_query_result_info(qreq_)
                # record previous feature configuration
                if ibs.table_cache:
                    prev_feat_cfgstr = qreq_.qparams.feat_cfgstr
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
    testres = test_result.TestResult(pipecfg_list, cfgx2_lbl, cfgx2_cfgresinfo, cfgx2_qreq_)
    testres.testnameid = testnameid
    testres.lbl = lbl
    testres.cfgdict_list = cfgdict_list
    testres.aidcfg = None
    if use_cache:
        try:
            ut.save_cache(*tuple(list(cachetup) + [testres]))
        except Exception as ex:
            ut.printex(ex, 'error saving testres cache', iswarning=True)
            if ut.SUPER_STRICT:
                raise
    return testres


@profile
def get_qres_name_result_info(ibs, cm, qreq_):
    """
    these are results per query we care about
     * gt (best correct match) and gf (best incorrect match) rank, their score
       and the difference

    """
    #from ibeis.algo.hots import chip_match
    qnid = cm.qnid
    nscoretup = cm.get_ranked_nids_and_aids()
    sorted_nids, sorted_nscores, sorted_aids, sorted_scores = nscoretup

    is_positive = sorted_nids == qnid
    is_negative = np.logical_and(~is_positive, sorted_nids > 0)
    gt_rank = None if not np.any(is_positive) else np.where(is_positive)[0][0]
    gf_rank = None if not np.any(is_negative) else np.nonzero(is_negative)[0][0]

    if gt_rank is None or gf_rank is None:
        #if isinstance(qres, chip_match.ChipMatch):
        gt_aids = ibs.get_annot_groundtruth(cm.qaid, daid_list=qreq_.daids)
        #else:
        #    gt_aids = cm.get_groundtruth_daids()
        cm.get_groundtruth_daids()
        gt_aid = gt_aids[0] if len(gt_aids) > 0 else None
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


def build_testres_from_cm_list(qreq_, cm_list):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.expt.harness import *  # NOQA
    """
    cfgx2_qreq_ = [qreq_]
    cfgx2_lbl = ['custom']
    pipecfg_list = [qreq_.qparams]
    #testres.testnameid = testnameid
    cfgres_info = _build_qresinfo(qreq_.ibs, qreq_, cm_list)
    cfgx2_cfgresinfo = [cfgres_info]
    testres = test_result.TestResult(pipecfg_list, cfgx2_lbl, cfgx2_cfgresinfo,
                                     cfgx2_qreq_)
    return testres


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
        python -m ibeis.expt.harness --test-get_query_result_info
        python -m ibeis.expt.harness --test-get_query_result_info:0
        python -m ibeis.expt.harness --test-get_query_result_info:1
        python -m ibeis.expt.harness --test-get_query_result_info:0 --db lynx -a default:qsame_imageset=True,been_adjusted=True,excluderef=True -t default:K=1
        python -m ibeis.expt.harness --test-get_query_result_info:0 --db lynx -a default:qsame_imageset=True,been_adjusted=True,excluderef=True -t default:K=1 --cmd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.expt.harness import *  # NOQA
        >>> import ibeis
        >>> qreq_ = ibeis.main_helpers.testdata_qreq_(a=['default:qindex=0:3,dindex=0:5'])
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> #qaids = ibs.get_valid_aids()[0:3]
        >>> #daids = ibs.get_valid_aids()[0:5]
        >>> #qreq_ = ibs.new_query_request(qaids, daids, verbose=True, cfgdict={})
        >>> cfgres_info = get_query_result_info(qreq_)
        >>> print(ut.dict_str(cfgres_info))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.expt.harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> #cfgdict = dict(codename='vsone')
        >>> # ibs.cfg.query_cfg.codename = 'vsone'
        >>> qaids = ibs.get_valid_aids()[0:3]
        >>> daids = ibs.get_valid_aids()[0:5]
        >>> qreq_ = ibs.new_query_request(qaids, daids, verbose=True, cfgdict={})
        >>> cfgres_info = get_query_result_info(qreq_)
        >>> print(ut.dict_str(cfgres_info))

    Ignore:

        ibeis -e rank_cdf --db humpbacks -a default:has_any=hasnotch,mingt=2 -t default:proot=BC_DTW --show --nocache-big

        ibeis -e rank_cdf --db humpbacks -a default:is_known=True,mingt=2 -t default:pipeline_root=BC_DTW
        --show --debug-depc
        ibeis -e rank_cdf --db humpbacks -a default:is_known=True -t default:pipeline_root=BC_DTW --qaid=1,9,15,16,18 --daid-override=1,9,15,16,18,21,22 --show --debug-depc
        --clear-all-depcache
    """
    try:
        ibs = qreq_.ibs
    except AttributeError:
        ibs = qreq_.depc.controller
    cm_list = qreq_.execute()
    return _build_qresinfo(ibs, qreq_, cm_list)


def _build_qresinfo(ibs, qreq_, cm_list):
    qx2_cm = cm_list
    qaids = qreq_.qaids
    #qaids2 = [cm.qaid for cm in cm_list]
    qnids = ibs.get_annot_name_rowids(qaids)

    unique_dnids = np.unique(ibs.get_annot_name_rowids(qreq_.daids))
    unique_qnids, groupxs = ut.group_indices(qnids)
    cm_group_list = ut.apply_grouping(cm_list, groupxs)
    qnid2_aggnamescores = {}

    qnx2_nameres_info = []

    #import utool
    #utool.embed()

    # Ranked list aggregation-ish
    nameres_info_list = []
    for qnid, cm_group in zip(unique_qnids, cm_group_list):
        nid2_name_score_group = [
            dict([(nid, cm.name_score_list[nidx]) for nid, nidx in cm.nid2_nidx.items()])
            for cm in cm_group
        ]
        aligned_name_scores = np.array([
            ut.dict_take(nid2_name_score, unique_dnids.tolist(), -np.inf)
            for nid2_name_score in nid2_name_score_group
        ]).T
        name_score_list = np.nanmax(aligned_name_scores, axis=1)
        qnid2_aggnamescores[qnid] = name_score_list
        # sort
        sortx = name_score_list.argsort()[::-1]
        sorted_namescores = name_score_list[sortx]
        sorted_dnids = unique_dnids[sortx]

        ## infer agg name results
        is_positive = sorted_dnids == qnid
        is_negative = np.logical_and(~is_positive, sorted_dnids > 0)
        gt_name_rank = None if not np.any(is_positive) else np.where(is_positive)[0][0]
        gf_name_rank = None if not np.any(is_negative) else np.nonzero(is_negative)[0][0]
        gt_nid = sorted_dnids[gt_name_rank]
        gf_nid = sorted_dnids[gf_name_rank]
        gt_name_score = sorted_namescores[gt_name_rank]
        gf_name_score = sorted_namescores[gf_name_rank]
        qnx2_nameres_info = {}
        qnx2_nameres_info['qnid'] = qnid
        qnx2_nameres_info['gt_nid'] = gt_nid
        qnx2_nameres_info['gf_nid'] = gf_nid
        qnx2_nameres_info['gt_name_rank'] = gt_name_rank
        qnx2_nameres_info['gf_name_rank'] = gf_name_rank
        qnx2_nameres_info['gt_name_score'] = gt_name_score
        qnx2_nameres_info['gf_name_score'] = gf_name_score

        nameres_info_list.append(qnx2_nameres_info)
        nameres_info = ut.dict_stack(nameres_info_list, 'qnx2_')

    qaids = qreq_.qaids
    daids = qreq_.daids
    qx2_gtaids = ibs.get_annot_groundtruth(qaids, daid_list=daids)
    # Get the groundtruth ranks and accuracy measures
    qx2_qresinfo = [get_qres_name_result_info(ibs, cm, qreq_) for cm in qx2_cm]

    cfgres_info = ut.dict_stack(qx2_qresinfo, 'qx2_')
    #for key in qx2_qresinfo[0].keys():
    #    'qx2_' + key
    #    ut.get_list_column(qx2_qresinfo, key)

    if False:
        qx2_avepercision = np.array(
            [cm.get_average_percision(ibs=ibs, gt_aids=gt_aids) for
             (cm, gt_aids) in zip(qx2_cm, qx2_gtaids)])
        cfgres_info['qx2_avepercision'] = qx2_avepercision
    # Compute mAP score  # TODO: use mAP score
    # (Actually map score doesn't make much sense if using name scoring
    #mAP = qx2_avepercision[~np.isnan(qx2_avepercision)].mean()  # NOQA
    cfgres_info['qx2_bestranks'] = ut.replace_nones(cfgres_info['qx2_bestranks'] , -1)
    cfgres_info.update(nameres_info)
    return cfgres_info


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.expt.harness
        python -m ibeis.expt.harness --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
