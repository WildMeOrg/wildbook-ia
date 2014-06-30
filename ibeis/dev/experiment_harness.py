from __future__ import absolute_import, division, print_function
# Python
import sys
import textwrap
# Scientific
import numpy as np
# Tools
import utool
# IBEIS
from ibeis.model.hots import match_chips3 as mc3
from ibeis.model.hots import matching_functions as mf
from ibeis.dev import params
from ibeis.dev import experiment_helpers as eh
from ibeis.dev import experiment_printres

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]', DEBUG=False)

BATCH_MODE = '--nobatch' not in sys.argv
NOMEMORY   = '--nomemory' in sys.argv
QUIET      = '--quiet' in sys.argv
TESTRES_VERBOSITY = 2 - (2 * QUIET)
NOCACHE_TESTRES =  utool.get_flag('--nocache-testres', False)
TEST_INFO = True
STRICT = utool.STRICT


def _get_qx2_besrank_batch(ibs, qreq):
    print('[harn] querying in batch mode')
    # Query Chip / Row Loop
    qaid2_qres = mc3.process_query_request(ibs, qreq, safe=False)
    qx2_bestranks = [[qaid2_qres[qaid].get_best_gt_rank(ibs)] for qaid in qreq.qaids]
    return qx2_bestranks


def _get_qx2_besrank_iterative(ibs, qreq, nTotalQueries, nPrevQueries, cfglbl=''):
    # TODO: INCORPORATE MINIBATCH SIZE TO MATCH_CHIPS3 AND DEPRICATE THIS
    print('[harn] querying one query at a time')
    # Make progress message
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST %d/%d ''' + cfglbl + '''
    ---------------------''')
    qx2_bestranks = []
    qaids = qreq.qaids  # Query one ANNOTATION at a time
    mark_prog = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '.')
    # Query Chip / Row Loop
    for qx, qaid in enumerate(qaids):
        mark_prog(qx + nPrevQueries, nTotalQueries)
        try:
            qreq.qaids = [qaid]  # hacky
            qaid2_qres = mc3.process_query_request(ibs, qreq, safe=False)
        except mf.QueryException as ex:
            utool.printex(ex, 'Harness caught Query Exception')
            qx2_bestranks.append([-1])
            if not STRICT:
                continue
            raise
        try:
            assert len(qaid2_qres) == 1, ''
        except AssertionError as ex:
            utool.printex(ex, key_list=['qaid2_qres'])
            raise
        # record the best rank from this groundtruth
        best_rank = qaid2_qres[qaid].get_best_gt_rank(ibs)
        qx2_bestranks.append([best_rank])
    qreq.qaids = qaids  # fix previous hack
    return qx2_bestranks


@profile
def get_qx2_bestrank(ibs, qaids, nTotalQueries, nPrevQueries, cfglbl):
    """
    Runs queries of a specific configuration returns the best rank of each query

    qaids - query annotation ids
    """
    daids = ibs.get_recognition_database_aids()
    # High level cache load
    #qx2_bestranks = eh.load_cached_test_results(ibs, qaids, daids, #NOCACHE_TESTRES, #TESTRES_VERBOSITY)
    #if qx2_bestranks is not None: #return qx2_bestranks
    qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                  qaids=qaids,
                                  daids=daids,
                                  query_cfg=ibs.cfg.query_cfg)
    qreq = mc3.pre_exec_checks(ibs, qreq)  # Preform invx checks first so we can be unsafe
    if BATCH_MODE:
        qx2_bestranks = _get_qx2_besrank_batch(ibs, qreq)
    else:
        qx2_bestranks = _get_qx2_besrank_iterative(ibs, qreq, nTotalQueries, nPrevQueries, cfglbl)
    qx2_bestranks = np.array(qx2_bestranks)
    # High level cache save
    #eh.cache_test_results(qx2_bestranks, ibs, qaids, daids)
    return qx2_bestranks


#-----------
#@utool.indent_func('[harn]')
@profile
def test_configurations(ibs, qaid_list, test_cfg_name_list, fnum=1):
    # Test Each configuration
    if not QUIET:
        print(textwrap.dedent("""
        [harn]================
        [harn] experiment_harness.test_configurations()""").strip())

    qaids = qaid_list

    # Grab list of algorithm configurations to test
    #cfg_list = eh.get_cfg_list(test_cfg_name_list, ibs=ibs)
    cfg_list, cfgx2_lbl = eh.get_cfg_list_and_lbls(test_cfg_name_list, ibs=ibs)
    cfgx2_lbl = np.array(cfgx2_lbl)
    if not QUIET:
        print('[harn] Testing %d different parameters' % len(cfg_list))
        print('[harn]         %d different chips' % len(qaids))

    # Preallocate test result aggregation structures
    sel_cols = params.args.sel_cols  # FIXME
    sel_rows = params.args.sel_rows  # FIXME
    sel_cols = [] if sel_cols is None else sel_cols
    sel_rows = [] if sel_rows is None else sel_rows

    nCfg     = len(cfg_list)   # number of configurations (cols)
    nQuery   = len(qaids)  # number of queries (rows)

    mat_list = []
    ibs._init_query_requestor()

    dbname = ibs.get_dbname()
    testnameid = dbname + ' ' + str(test_cfg_name_list)
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST_CFG %d/%d: ''' + testnameid + '''
    ---------------------''')
    mark_prog = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '+')
    # Run each test configuration
    # Query Config / Col Loop
    daids = ibs.get_recognition_database_aids()
    nTotalQueries  = nQuery * nCfg  # number of quieries to run in total
    for cfgx, query_cfg in enumerate(cfg_list):
        if not QUIET:
            mark_prog(cfgx + 1, nCfg)
            print(query_cfg.get_cfgstr())
        cfglbl = cfgx2_lbl[cfgx]
        ibs.set_query_cfg(query_cfg)
        # Set data to the current config
        nPrevQueries = nQuery * cfgx  # number of pervious queries
        # Run the test / read cache
        with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
            qx2_bestranks = get_qx2_bestrank(ibs, qaids, nTotalQueries, nPrevQueries, cfglbl)
        if not NOMEMORY:
            mat_list.append(qx2_bestranks)
        # Store the results
    if not QUIET:
        print('[harn] Finished testing parameters')
    if NOMEMORY:
        print('ran tests in memory savings mode. exiting')
        return
    experiment_printres.print_results(ibs, qaids, daids, cfg_list,
                                          mat_list, testnameid, sel_rows,
                                          sel_cols, cfgx2_lbl=cfgx2_lbl)
