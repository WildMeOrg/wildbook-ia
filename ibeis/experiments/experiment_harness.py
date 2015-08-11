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
from ibeis.experiments import experiment_helpers as expt_helpers
from ibeis.experiments import experiment_printres
from ibeis.experiments import experiment_drawing
from ibeis.experiments import experiment_storage
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]')

BATCH_MODE = '--nobatch' not in sys.argv
NOMEMORY   = '--nomemory' in sys.argv
TESTRES_VERBOSITY = 2 - (2 * utool.QUIET)
NOCACHE_TESTRES =  utool.get_argflag('--nocache-testres', False)
TEST_INFO = True


@profile
def get_qres_name_result_info(ibs, qres):
    """
    these are results per query we care about
     * gt and gf rank, their score and the difference

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
def get_query_result_info(qx2_qres, qreq_):
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
        >>> qx2_qres = ibs.query_chips(qreq_=qreq_)
        >>> cfgres_info = get_query_result_info(qx2_qres, qreq_)
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
        >>> qx2_qres = ibs.query_chips(qreq_=qreq_)
        >>> cfgres_info = get_query_result_info(qx2_qres, qreq_)
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


#-----------
#@utool.indent_func('[harn]')
@profile
def test_configurations(ibs, qaids, daids, test_cfg_name_list):
    r"""
    Test harness driver function

    CommandLine:
        python -m ibeis.experiments.experiment_harness --test-test_configurations

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaids = ibs.get_valid_aids()[::40]
        >>> daids = ibs.get_valid_aids()
        >>> #test_cfg_name_list = ['best', 'smk2']
        >>> test_cfg_name_list = ['custom']
        >>> test_configurations(ibs, qaids, daids, test_cfg_name_list)
    """
    if len(qaids) == 0:
        print('[harness] No query annotations specified')
        return None
    if len(daids) == 0:
        print('[harness] No query annotations specified')
        return None
    test_result = run_test_configurations(ibs, qaids, daids, test_cfg_name_list)
    if test_result is None:
        return
    else:
        experiment_printres.print_results(ibs, test_result)
        experiment_drawing.draw_results(ibs, test_result)


def get_candidacy_dbnames():
    return [
        'PZ_MTEST',
        'NNP_MasterGIRM_core',
        'PZ_Master0',
        'NNP_Master3',
        'GZ_ALL',
        'PZ_FlankHack',
        #'JAG_Kelly',
        #'JAG_Kieryn',
        #'LF_Bajo_bonito',
        #'LF_OPTIMIZADAS_NI_V_E',
        #'LF_WEST_POINT_OPTIMIZADAS',
        #'GZ_Master0',
        #'GIR_Tanya',
    ]


def precfg_dbs(db_list):
    r"""
    Runs precfg on multiple databases

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
        python -m ibeis.experiments.experiment_harness --exec-precfg_dbs --dblist=candidacy -t candidacy --preload-feeatweight
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
        db_list = get_candidacy_dbnames()  # HACK
    print('db_list = %s' % (ut.list_str(db_list),))
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=[])
    for db in db_list:
        ibs = ibeis.opendb(db=db)
        ibs, qaids, daids = ibeis.init.main_helpers.testdata_ibeis(verbose=False, ibs=ibs)
        precfg(ibs, qaids, daids, test_cfg_name_list)


def precfg(ibs, qaids, daids, test_cfg_name_list):
    r"""
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
        >>> import ibeis.init.main_helpers
        >>> ibs, qaids, daids = ibeis.init.main_helpers.testdata_ibeis(verbose=False)
        >>> test_cfg_name_list = ut.get_argval('-t', type_=list, default=[])
        >>> result = precfg(ibs, qaids, daids, test_cfg_name_list)
        >>> print(result)
    """
    cfg_list, cfgx2_lbl = expt_helpers.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)

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
            if ut.get_argflag('--preload-feeatweight'):
                qreq_.ensure_featweights(verbose=verbose)
                flag = True
            if ut.get_argflag('--preindex'):
                flag = True
                if qreq_.qparams.pipeline_root in ['vsone', 'vsmany']:
                    qreq_.load_indexer(verbose=verbose)
        assert flag is True, 'no flag specified'
    assert flag is True, 'no flag specified'


@profile
def run_test_configurations(ibs, qaids, daids, test_cfg_name_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (list):  query annotation ids
        daids (list):  database annotation ids
        test_cfg_name_list (list):

    Returns:
        ?:

    CommandLine:
        python -m ibeis.experiments.experiment_harness --test-run_test_configurations
        python -m ibeis.experiments.experiment_harness --test-run_test_configurations --serial

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> species = ibeis.const.Species.ZEB_PLAIN
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> qaids = ibs.get_valid_aids(species=species)
        >>> daids = ibs.get_valid_aids(species=species)
        >>> test_cfg_name_list = ['custom', 'custom:fg_on=False']
        >>> result = run_test_configurations(ibs, qaids, daids, test_cfg_name_list)
        >>> print(result)
    """
    # Grab list of algorithm configurations to test
    # Test Each configuration
    cfg_list, cfgx2_lbl = expt_helpers.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)

    if not utool.QUIET:
        ut.colorprint(textwrap.dedent("""

        [harn]================
        [harn] experiment_harness.test_configurations()""").strip(), 'white')

    #orig_query_cfg = ibs.cfg.query_cfg  # Remember original query config

    # Parse cfgstr list
    #cfgstr_list = [query_cfg.get_cfgstr() for query_cfg in cfg_list]
    #for cfgstr in cfgstr_list:
    #    cfgstr_terms = cfgstr.split(')_')
    #    cfgstr_terms = [x + ')_' for x in cfgstr_terms[1:-1]] + [cfgstr_terms[-1]]  # NOQA

    #ut.embed()
    #print('cfgx2_lbl = ' + ut.list_str(cfgx2_lbl))
    #print('cfgstr_list = ' + ut.list_str(cfgstr_list))

    cfgx2_lbl = np.array(cfgx2_lbl)
    if not utool.QUIET:
        ut.colorprint('[harn] Testing %d parameter configurations' % len(cfg_list), 'white')
        ut.colorprint('[harn]    with %d query annotations' % len(qaids), 'white')

    testnameid = ibs.get_dbname() + ' ' + str(test_cfg_name_list)
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list)
    nCfg     = len(cfg_list)   # number of configurations (cols)
    dbname = ibs.get_dbname()

    DRY_RUN =  ut.get_argflag('--dryrun')  # dont actually query. Just print labels and stuff

    print('Constructing query requests')
    cfgx2_qreq_ = [ibs.new_query_request(qaids, daids, verbose=True, query_cfg=query_cfg) for query_cfg in cfg_list]

    qreq_ = ibs.new_query_request(qaids, daids, verbose=True, query_cfg=ibs.cfg.query_cfg)

    cfgx2_cfgresinfo = []
    #with utool.Timer('experiment_harness'):
    cfgiter = ut.ProgressIter(enumerate(cfg_list), nTotal=nCfg, lbl=lbl,
                              freq=1, autoadjust=False)
    # Run each test configuration
    # Query Config / Col Loop
    #for cfgx, query_cfg in enumerate(cfg_list):
    for cfgx, query_cfg in cfgiter:
        #print('+--- REQUESTING TEST CONFIG ---')
        #print(query_cfg.get_cfgstr())
        ut.colorprint(query_cfg.get_cfgstr(), 'turquoise')
        qreq_ = cfgx2_qreq_[cfgx]

        with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
            if DRY_RUN:
                continue
            # Set data to the current config
            #ibs.set_query_cfg(query_cfg)  # TODO: make this not even matter
            # Run the test / read cache
            qx2_qres = ibs.query_chips(qreq_=qreq_)
            cfgres_info = get_query_result_info(qx2_qres, qreq_)
            del qx2_qres
            #qx2_bestranks, qx2_next_bestranks, qx2_scorediff, qx2_avepercision = cfgres_info
        if not NOMEMORY:
            # Store the results
            cfgx2_cfgresinfo.append(cfgres_info)
            cfgx2_qreq_.append(qreq_)
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
    test_result = experiment_storage.TestResult(cfg_list, cfgx2_lbl, lbl, testnameid, cfgx2_cfgresinfo, cfgx2_qreq_, daids, qaids)
    return test_result

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
