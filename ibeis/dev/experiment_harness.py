"""
Runs many queries and keeps track of some results
"""
from __future__ import absolute_import, division, print_function
import sys
import textwrap
import numpy as np
import six
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
def get_qx2_bestrank(ibs, qaids, daids):
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
        >>> qx2_bestranks, qx2_avepercision = get_qx2_bestrank(ibs, qaids, daids)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.dev.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> cfgdict = dict(codename='vsone')
        >>> # ibs.cfg.query_cfg.codename = 'vsone'
        >>> qaids = ibs.get_valid_aids()[0:3]
        >>> daids = ibs.get_valid_aids()[0:5]
        >>> qx2_bestranks, qx2_avepercision = get_qx2_bestrank(ibs, qaids, daids)
    """
    # Execute or load query
    qaid2_qres = ibs._query_chips4(qaids, daids)
    INTERACT_HARNESS = ut.get_argflag('--interact-harness')
    #ut.embed()
    if INTERACT_HARNESS:
        for qaid, qres in six.iteritems(qaid2_qres):
            break
        for qaid, qres in six.iteritems(qaid2_qres):
            qres.ishow_top(ibs)
        pass
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
    # Compute measures
    #with ut.EmbedOnException():
    # qres = qaid2_qres[qaid]
    qx2_bestranks = [[qaid2_qres[qaid].get_best_gt_rank(ibs=ibs, gt_aids=gt_aids)]
                     for qaid, gt_aids in zip(qaids, qx2_gtaids)]
    qx2_avepercision = [qaid2_qres[qaid].get_average_percision(ibs=ibs, gt_aids=gt_aids) for
                        (qaid, gt_aids) in zip(qaids, qx2_gtaids)]
    # Compute mAP score  # TODO: use mAP score
    qx2_avepercision = np.array(qx2_avepercision)
    mAP = qx2_avepercision[~np.isnan(qx2_avepercision)].mean()  # NOQA
    return qx2_bestranks, qx2_avepercision


#-----------
#@utool.indent_func('[harn]')
#@profile
def test_configurations(ibs, qaid_list, daid_list, test_cfg_name_list):
    """
    Test harness driver function

    Args:
        ibs (IBEISController):
        qaid_list (int): query annotation id
        daid_list (int): data annotation id
        test_cfg_name_list (list):

    Example:
        >>> from ibeis.dev.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibs.opendb('PZ_MTEST')
        >>> qaid_list = [0]
        >>> test_cfg_name_list = ['best', 'smk2']
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

    bestranks_list = []
    cfgx2_aveprecs = []
    #ibs._init_query_requestor()

    dbname = ibs.get_dbname()
    testnameid = dbname + ' ' + str(test_cfg_name_list)
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST_CFG %d/%d: ''' + testnameid + '''
    ---------------------''')
    mark_prog = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '+')
    # Run each test configuration
    # Query Config / Col Loop
    #nTotalQueries  = nQuery * nCfg  # number of quieries to run in total
    with utool.Timer('experiment_harness'):
        for cfgx, query_cfg in enumerate(cfg_list):
            if not utool.QUIET:
                mark_prog(cfgx + 1, nCfg)
                print(query_cfg.get_cfgstr())
            #cfglbl = cfgx2_lbl[cfgx]
            ibs.set_query_cfg(query_cfg)
            # Set data to the current config
            #nPrevQueries = nQuery * cfgx  # number of pervious queries
            # Run the test / read cache
            with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
                qx2_bestranks, qx2_avepercision = get_qx2_bestrank(ibs, qaids, daids)
            if not NOMEMORY:
                bestranks_list.append(qx2_bestranks)
                cfgx2_aveprecs.append(qx2_avepercision)
            # Store the results
    if not utool.QUIET:
        print('[harn] Finished testing parameters')
    if NOMEMORY:
        print('ran tests in memory savings mode. Cannot Print. exiting')
        return
    experiment_printres.print_results(ibs, qaids, daids, cfg_list,
                                      bestranks_list, cfgx2_aveprecs, testnameid, sel_rows,
                                      sel_cols, cfgx2_lbl)
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
