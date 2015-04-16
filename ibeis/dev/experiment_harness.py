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
from ibeis.dev import experiment_helpers as eh
from ibeis.dev import experiment_printres

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
def get_config_result_info(ibs, qaids, daids):
    """
    Helper function.

    Runs queries of a specific configuration returns the best rank of each query

    Args:
        qaids (list) : query annotation ids
        daids (list) : database annotation ids

    Returns:
        qx2_bestranks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.dev.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaids = ibs.get_valid_aids()[0:3]
        >>> daids = ibs.get_valid_aids()[0:5]
        >>> qresinfotup, qreq_ = get_config_result_info(ibs, qaids, daids)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.dev.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> cfgdict = dict(codename='vsone')
        >>> # ibs.cfg.query_cfg.codename = 'vsone'
        >>> qaids = ibs.get_valid_aids()[0:3]
        >>> daids = ibs.get_valid_aids()[0:5]
        >>> qresinfotup, qreq_ = get_config_result_info(ibs, qaids, daids)

    Ignore:
        for qaid, qres in six.iteritems(qaid2_qres):
            break
        for qaid, qres in six.iteritems(qaid2_qres):
            qres.ishow_top(ibs)
    """
    # Execute or load query
    ibs.get_annot_name_rowids(qaids)
    ibs.get_annot_name_rowids(daids)
    qaid2_qres, qreq_ = ibs._query_chips4(qaids, daids, return_request=True)
    qx2_qres = ut.dict_take(qaid2_qres, qaids)
    # Get the groundtruth that could have been matched in this experiment
    qx2_gtaids = ibs.get_annot_groundtruth(qaids, daid_list=daids)
    # Get the groundtruth ranks and accuracy measures
    qx2_qresinfo = [get_qres_name_result_info(ibs, qres) for qres in qx2_qres]

    cfgres_info = ut.dict_stack(qx2_qresinfo, 'qx2_')
    keys = qx2_qresinfo[0].keys()
    for key in keys:
        'qx2_' + key
        ut.get_list_column(qx2_qresinfo, key)

    qx2_avepercision = np.array(
        [qaid2_qres[qaid].get_average_percision(ibs=ibs, gt_aids=gt_aids) for
         (qaid, gt_aids) in zip(qaids, qx2_gtaids)])
    cfgres_info['qx2_avepercision'] = qx2_avepercision
    # Compute mAP score  # TODO: use mAP score
    # (Actually map score doesn't make much sense if using name scoring
    #mAP = qx2_avepercision[~np.isnan(qx2_avepercision)].mean()  # NOQA
    cfgres_info['qx2_bestranks'] = ut.replace_nones(cfgres_info['qx2_bestranks'] , -1)
    return cfgres_info, qreq_


#-----------
#@utool.indent_func('[harn]')
@profile
def test_configurations(ibs, qaids, daids, test_cfg_name_list):
    r"""
    Test harness driver function

    CommandLine:
        python -m ibeis.dev.experiment_harness --test-test_configurations

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.dev.experiment_harness import *  # NOQA
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
        experiment_printres.draw_results(ibs, test_result)


class TestResult(object):
    def __init__(test_result, cfg_list, cfgx2_lbl, lbl, testnameid, cfgx2_cfgresinfo, cfgx2_qreq_, daids, qaids):
        test_result.qaids = qaids
        test_result.daids = daids
        test_result.cfg_list         = cfg_list
        test_result.cfgx2_lbl        = cfgx2_lbl
        test_result.lbl              = lbl
        test_result.testnameid       = testnameid
        test_result.cfgx2_cfgresinfo = cfgx2_cfgresinfo
        test_result.cfgx2_qreq_      = cfgx2_qreq_

    @ut.memoize
    def get_rank_mat(test_result):
        # Ranks of Best Results
        cfgx2_bestranks = ut.get_list_column(test_result.cfgx2_cfgresinfo, 'qx2_bestranks')
        rank_mat = np.vstack(cfgx2_bestranks).T  # concatenate each query rank across configs
        # Set invalid ranks to the worse possible rank
        worst_possible_rank = max(9001, len(test_result.daids) + 1)
        rank_mat[rank_mat == -1] =  worst_possible_rank
        return rank_mat

    @ut.memoize
    def get_new_hard_qx_list(test_result):
        """ Mark any query as hard if it didnt get everything correct """
        rank_mat = test_result.get_rank_mat()
        is_new_hard_list = rank_mat.max(axis=1) > 0
        new_hard_qx_list = np.where(is_new_hard_list)[0]
        return new_hard_qx_list


@profile
def run_test_configurations(ibs, qaids, daids, test_cfg_name_list):
    # Grab list of algorithm configurations to test
    testnameid = ibs.get_dbname() + ' ' + str(test_cfg_name_list)
    cfg_list, cfgx2_lbl = eh.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list)
    # Test Each configuration
    if not utool.QUIET:
        ut.colorprint(textwrap.dedent("""

        [harn]================
        [harn] experiment_harness.test_configurations()""").strip(), 'white')

    orig_query_cfg = ibs.cfg.query_cfg  # Remember original query config
    cfgx2_lbl = np.array(cfgx2_lbl)
    if not utool.QUIET:
        ut.colorprint('[harn] Testing %d different parameters' % len(cfg_list), 'white')
        ut.colorprint('[harn]         %d query annotations' % len(qaids), 'white')

    nCfg     = len(cfg_list)   # number of configurations (cols)
    dbname = ibs.get_dbname()
    cfgx2_cfgresinfo = []
    cfgx2_qreq_      = []
    with utool.Timer('experiment_harness'):
        cfgiter = ut.ProgressIter(enumerate(cfg_list), nTotal=nCfg, lbl=lbl,
                                  freq=1, time_thresh=4)
        # Run each test configuration
        # Query Config / Col Loop
        #for cfgx, query_cfg in enumerate(cfg_list):
        for cfgx, query_cfg in cfgiter:
            print('+--- REQUESTING CONFIG ---')
            print(query_cfg.get_cfgstr())
            print('L____')
            # Set data to the current config
            ibs.set_query_cfg(query_cfg)
            # Run the test / read cache
            with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
                cfgres_info, qreq_ = get_config_result_info(ibs, qaids, daids)
                #qx2_bestranks, qx2_next_bestranks, qx2_scorediff, qx2_avepercision = cfgres_info
            if not NOMEMORY:
                # Store the results
                cfgx2_cfgresinfo.append(cfgres_info)
                cfgx2_qreq_.append(qreq_)
    if not utool.QUIET:
        ut.colorprint('[harn] Completed running test configurations', 'white')
        #print(msg)
    if NOMEMORY:
        print('ran tests in memory savings mode. Cannot Print. exiting')
        return
    # Reset query config so nothing unexpected happens
    # TODO: should probably just use a cfgdict to build a list of QueryRequest
    # objects. That would avoid the entire problem
    ibs.set_query_cfg(orig_query_cfg)
    test_result = TestResult(cfg_list, cfgx2_lbl, lbl, testnameid, cfgx2_cfgresinfo, cfgx2_qreq_, daids, qaids)
    return test_result

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.dev.experiment_harness; utool.doctest_funcs(ibeis.dev.experiment_harness, allexamples=True)"
        python -c "import utool, ibeis.dev.experiment_harness; utool.doctest_funcs(ibeis.dev.experiment_harness)"
        python -m ibeis.dev.experiment_harness
        python -m ibeis.dev.experiment_harness --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
