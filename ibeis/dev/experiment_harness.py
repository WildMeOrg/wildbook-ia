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
from ibeis import params
from ibeis.dev import experiment_helpers as eh
from ibeis.dev import experiment_printres

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]')

BATCH_MODE = '--nobatch' not in sys.argv
NOMEMORY   = '--nomemory' in sys.argv
TESTRES_VERBOSITY = 2 - (2 * utool.QUIET)
NOCACHE_TESTRES =  utool.get_argflag('--nocache-testres', False)
TEST_INFO = True


#@profile
def get_config_result_info(ibs, qaids, daids):
    """
    Helper function.

    Runs queries of a specific configuration returns the best rank of each query

    Args:
        ibs : IBEIS Controller
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
    qaid2_qres, qreq_ = ibs._query_chips4(qaids, daids, return_request=True)
    qx2_qres = ut.dict_take(qaid2_qres, qaids)
    # Get the groundtruth that could have been matched in this experiment
    qx2_gtaids = ibs.get_annot_groundtruth(qaids, daid_list=daids)
    # Get the groundtruth ranks
    #
    # <TECHNICALLY NONNECESSARY, BUT MAKES LIFE EASIER>
    # Pick a good max rank that isn't None
    #worst_possible_value = max(9001, len(daids) + len(qaids) + 2)
    #ut.embed()
    #qx2_gtranks = [qaid2_qres[qaid].get_gt_ranks(ibs=ibs, gt_aids=gt_aids, fillvalue=worst_possible_value)
    #               for qaid, gt_aids in zip(qaids, qx2_gtaids)]
    #qx2_sorted_gtaids = [np.array(ut.sortedby(gt_aids, gt_ranks))
    #                     for gt_aids, gt_ranks in zip(qx2_gtaids, qx2_gtranks)]
    #qx2_sorted_gtranks = [qaid2_qres[qaid].get_gt_ranks(ibs=ibs, gt_aids=gt_aids, fillvalue=worst_possible_value)
    #                      for qaid, gt_aids in zip(qaids, qx2_sorted_gtaids)]
    #qx2_gtaids = qx2_sorted_gtaids
    #qx2_gtranks = qx2_sorted_gtranks
    # </TECHNICALLY NONNECESSARY, BUT MAKES LIFE EASIER>
    #
    #with ut.EmbedOnException():
    # qres = qaid2_qres[qaid]

    # Compute accuracy measures

    def get_qres_name_result_info(qres):
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
    #ibs.get_aids_and_scores()

    qx2_qresinfo = [get_qres_name_result_info(qres) for qres in qx2_qres]

    cfgres_info = ut.dict_stack(qx2_qresinfo, 'qx2_')
    keys = qx2_qresinfo[0].keys()
    for key in keys:
        'qx2_' + key
        ut.get_list_column(qx2_qresinfo, key)

    #qx2_bestranks      = ut.get_list_column(qx2_qresinfotup, 0)
    #qx2_next_bestranks = ut.get_list_column(qx2_qresinfotup, 1)
    #qx2_gt_raw_score   = ut.get_list_column(qx2_qresinfotup, 4)
    #qx2_gf_raw_score   = ut.get_list_column(qx2_qresinfotup, 5)
    #qx2_scorediff      = ut.get_list_column(qx2_qresinfotup, 6)
    #qx2_scorefactor    = ut.get_list_column(qx2_qresinfotup, 7)
    #qx2_scorelogfactor = ut.get_list_column(qx2_qresinfotup, 8)
    #qx2_scoreexpdiff   = ut.get_list_column(qx2_qresinfotup, 9)

    #qx2_gtranks = [qaid2_qres[qaid].get_aid_ranks(gt_aids)
    #               for qaid, gt_aids in zip(qaids, qx2_gtaids)]

    #qx2_gfranks = [list(set(range(len(gtranks) + 1)) - set(gtranks))
    #               for gtranks in qx2_gtranks]

    #qx2_best_gt_aid = [-1 ]

    ##qx2_bestranks = [[qaid2_qres[qaid].get_best_gt_rank(ibs=ibs, gt_aids=gt_aids)]
    ##                 for qaid, gt_aids in zip(qaids, qx2_gtaids)]
    #qx2_bestranks = [-1 if len(gtranks) == 0 else min(gtranks) for gtranks in qx2_gtranks]

    #qx2_bestranks_gf = [-1 if len(gtranks) == 0 else min(gtranks) for gtranks in qx2_gfranks]

    #qx2_bestscores = []
    #qx2_bestscores_gf = []

    qx2_avepercision = np.array(
        [qaid2_qres[qaid].get_average_percision(ibs=ibs, gt_aids=gt_aids) for
         (qaid, gt_aids) in zip(qaids, qx2_gtaids)])
    cfgres_info['qx2_avepercision'] = qx2_avepercision

    # Compute mAP score  # TODO: use mAP score
    # (Actually map score doesn't make much sense if using name scoring
    mAP = qx2_avepercision[~np.isnan(qx2_avepercision)].mean()  # NOQA

    #qx2_bestranks = ut.replace_nones(qx2_bestranks, -1)

    #cfgres_info = qx2_bestranks, qx2_next_bestranks, qx2_scorediff, qx2_avepercision
    #cfgres_info = {
    #    'qx2_bestranks'      : qx2_bestranks,
    #    'qx2_next_bestranks' : qx2_next_bestranks,
    #    'qx2_avepercision'   : qx2_avepercision,
    #    'qx2_gt_raw_score'   : qx2_gt_raw_score,
    #    'qx2_gf_raw_score'   : qx2_gf_raw_score,
    #    'qx2_scorediff'      : qx2_scorediff,
    #    'qx2_scorefactor'    : qx2_scorefactor,
    #    'qx2_scorelogfactor' : qx2_scorelogfactor,
    #    'qx2_scoreexpdiff'   : qx2_scoreexpdiff,
    #}
    cfgres_info['qx2_bestranks'] = ut.replace_nones(cfgres_info['qx2_bestranks'] , -1)
    return cfgres_info, qreq_


#-----------
#@utool.indent_func('[harn]')
#@profile
def test_configurations(ibs, qaid_list, daid_list, test_cfg_name_list):
    r"""
    Test harness driver function

    Args:
        ibs (IBEISController):
        qaid_list (int): query annotation id
        daid_list (int): data annotation id
        test_cfg_name_list (list):


    CommandLine:
        python -m ibeis.dev.experiment_harness --test-test_configurations

    Example:
        >>> from ibeis.dev.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[::40]
        >>> daid_list = ibs.get_valid_aids()
        >>> #test_cfg_name_list = ['best', 'smk2']
        >>> test_cfg_name_list = ['custom']
        >>> test_configurations(ibs, qaid_list, daid_list, test_cfg_name_list)
    """
    # Test Each configuration
    if not utool.QUIET:
        print(textwrap.dedent("""
        [harn]================
        [harn] experiment_harness.test_configurations()""").strip())

    qaids = qaid_list
    daids = daid_list

    orig_query_cfg = ibs.cfg.query_cfg  # Remember original query config
    #if daids is None:
    #    daids = ibs.get_recognition_database_aids()

    # Grab list of algorithm configurations to test
    #cfg_list = eh.get_cfg_list(test_cfg_name_list, ibs=ibs)
    cfg_list, cfgx2_lbl = eh.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)
    cfgx2_lbl = np.array(cfgx2_lbl)
    if not utool.QUIET:
        print('[harn] Testing %d different parameters' % len(cfg_list))
        print('[harn]         %d query annotations' % len(qaids))

    # Preallocate test result aggregation structures
    sel_cols = params.args.sel_cols  # FIXME
    sel_rows = params.args.sel_rows  # FIXME
    sel_cols = [] if sel_cols is None else sel_cols
    sel_rows = [] if sel_rows is None else sel_rows

    nCfg     = len(cfg_list)   # number of configurations (cols)
    #nQuery   = len(qaids)  # number of queries (rows)

    #bestranks_list = []
    #cfgx2_aveprecs = []
    #ibs._init_query_requestor()

    dbname = ibs.get_dbname()
    testnameid = dbname + ' ' + str(test_cfg_name_list)
    #msg = textwrap.dedent('''
    #---------------------
    #[harn] TEST_CFG %d/%d: ''' + testnameid + '''
    #---------------------''')
    lbl = '[harn] TEST_CFG ' + str(test_cfg_name_list)
    #mark_prog = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '+')
    # Run each test configuration
    # Query Config / Col Loop
    #nTotalQueries  = nQuery * nCfg  # number of quieries to run in total
    cfgx2_cfgresinfo = []
    cfgx2_qreq_ = []
    with utool.Timer('experiment_harness'):
        cfgiter = ut.ProgressIter(enumerate(cfg_list), nTotal=nCfg, lbl=lbl,
                                  freq=1, time_thresh=4)
        #for cfgx, query_cfg in enumerate(cfg_list):
        for cfgx, query_cfg in cfgiter:
            #if not utool.QUIET:
            #    mark_prog(cfgx + 1, nCfg)
            #    print(query_cfg.get_cfgstr())
            #cfglbl = cfgx2_lbl[cfgx]
            print('+----')
            print(query_cfg.get_cfgstr())
            print('L____')
            ibs.set_query_cfg(query_cfg)
            # Set data to the current config
            #nPrevQueries = nQuery * cfgx  # number of pervious queries
            # Run the test / read cache
            with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
                cfgres_info, qreq_ = get_config_result_info(ibs, qaids, daids)
                #qx2_bestranks, qx2_next_bestranks, qx2_scorediff, qx2_avepercision = cfgres_info
            if not NOMEMORY:
                cfgx2_cfgresinfo.append(cfgres_info)
                cfgx2_qreq_.append(qreq_)
                #bestranks_list.append(qx2_bestranks)
                #cfgx2_aveprecs.append(qx2_avepercision)
            # Store the results
    if not utool.QUIET:
        print('[harn] Finished testing parameters')
    if NOMEMORY:
        print('ran tests in memory savings mode. Cannot Print. exiting')
        return
    experiment_printres.print_results(ibs, qaids, daids, cfg_list,
                                      cfgx2_cfgresinfo, testnameid, sel_rows,
                                      sel_cols, cfgx2_lbl, cfgx2_qreq_)
    # Reset query config so nothing unexpected happens
    # TODO: should probably just use a cfgdict to build a list of QueryRequest
    # objects. That would avoid the entire problem
    ibs.set_query_cfg(orig_query_cfg)


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
