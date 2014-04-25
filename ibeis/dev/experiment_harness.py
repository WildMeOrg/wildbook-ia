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
from ibeis.dev import report_experiment_results

print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]', DEBUG=False)

BATCH_MODE = '--batch' in sys.argv
NOMEMORY   = '--nomemory' in sys.argv
QUIET      = '--quiet' in sys.argv
TESTRES_VERBOSITY = 2 - (2 * QUIET)
NOCACHE_TESTRES =  utool.get_flag('--nocache-testres', False)
TEST_INFO = True


def _get_qx2_besrank_batch(ibs, qreq):
    print('[harn] querying in batch mode')
    #mc3.pre_cache_checks(ibs, qreq)
    # Query Chip / Row Loop
    qrids = qreq.qrids
    qrid2_res = mc3.process_query_request(ibs, qreq, safe=False)
    qx2_bestranks = [[qrid2_res[qrid].get_best_gt_rank(ibs)] for qrid in qrids]
    return qx2_bestranks


def _get_qx2_besrank_iterative(ibs, qreq, cfgx, nCfg, nTotalQueries, nPrevQueries):
    print('[harn] querying one query at a time')
    # Make progress message
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST %d/%d
    ---------------------''')
    mark_progress = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '.')
    # Query Chip / Row Loop
    qx2_bestranks = []
    qrids = qreq.qrids
    for qx, qrid in enumerate(qrids):
        count = qx + nPrevQueries + 1
        mark_progress(count, nTotalQueries)
        if TEST_INFO:
            print('qrid=%r. quid=%r' % (qrid, qreq.get_uid()))
        try:
            qreq.qrids = [qrid]  # hacky
            qrid2_res = mc3.process_query_request(ibs, qreq, safe=False)
        except mf.QueryException as ex:
            utool.printex(ex, 'Harness caught Query Exception')
            if params.args.strict:
                raise
            else:
                qx2_bestranks.append([-1])
                continue
        try:
            assert len(qrid2_res) == 1, ''
        except AssertionError as ex:
            utool.printex(ex, key_list=['qrid2_res'])
            raise
        # record metadata
        qx2_bestranks.append([qrid2_res[qrid].get_best_gt_rank(ibs)])
        if qrid % 4 == 0:
            sys.stdout.flush()
    qreq.qrids = qrids  # fix previous hack
    return qx2_bestranks


@profile
def get_qx2_bestrank(ibs, qrids, nTotalQueries, nPrevQueries):
    """
    Runs queries of a specific configuration returns the best rank of each query

    qrids - query roi ids
    """
    drids = ibs.get_recognition_database_rids()
    # High level cache load
    #qx2_bestranks = eh.load_cached_test_results(ibs, qrids, drids, #NOCACHE_TESTRES, #TESTRES_VERBOSITY)
    #if qx2_bestranks is not None: #return qx2_bestranks
    qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                  qrids=qrids,
                                  drids=drids,
                                  query_cfg=ibs.cfg.query_cfg)
    qreq = mc3.pre_exec_checks(ibs, qreq)  # Preform invx checks first so we can be unsafe
    if BATCH_MODE:
        qx2_bestranks = _get_qx2_besrank_batch(ibs, qreq)
    else:
        qx2_bestranks = _get_qx2_besrank_iterative(ibs, qreq, nTotalQueries, nPrevQueries)
    qx2_bestranks = np.array(qx2_bestranks)
    # High level cache save
    #eh.cache_test_results(qx2_bestranks, ibs, qrids, drids)
    return qx2_bestranks


#-----------
@utool.indent_func('[harn]')
@profile
def test_configurations(ibs, qrids, test_cfg_name_list, fnum=1):
    # Test Each configuration
    if not QUIET:
        print(textwrap.dedent("""
        [harn]================
        [harn] experiment_harness.test_configurations()""").strip())

    # Grab list of algorithm configurations to test
    cfg_list = eh.get_cfg_list(test_cfg_name_list, ibs=ibs)
    if not QUIET:
        print('[harn] Testing %d different parameters' % len(cfg_list))
        print('[harn]         %d different chips' % len(qrids))

    # Preallocate test result aggregation structures
    sel_cols = params.args.sel_cols  # FIXME
    sel_rows = params.args.sel_rows  # FIXME
    sel_cols = [] if sel_cols is None else sel_cols
    sel_rows = [] if sel_rows is None else sel_rows

    nCfg     = len(cfg_list)   # number of configurations (cols)
    nQuery   = len(qrids)  # number of queries (rows)

    mat_list = []
    ibs._init_query_requestor()

    dbname = ibs.get_dbname()
    testnameid = dbname + ' ' + str(test_cfg_name_list)
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST_CFG %d/%d: ''' + testnameid + '''
    ---------------------''')
    mark_progress = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '+')

    # Run each test configuration
    # Query Config / Col Loop
    drids = ibs.get_recognition_database_rids()
    nTotalQueries  = nQuery * nCfg  # number of quieries to run in total
    for cfgx, query_cfg in enumerate(cfg_list):
        if not QUIET:
            mark_progress(cfgx + 1, nCfg)
            print(query_cfg.get_uid())
        ibs.set_query_cfg(query_cfg)
        # Set data to the current config
        nPrevQueries = nQuery * cfgx  # number of pervious queries
        # Run the test / read cache
        with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
            qx2_bestranks = get_qx2_bestrank(ibs, qrids, nTotalQueries, nPrevQueries)
        if not NOMEMORY:
            mat_list.append(qx2_bestranks)
        # Store the results

    if not QUIET:
        print('[harn] Finished testing parameters')
    if NOMEMORY:
        print('ran tests in memory savings mode. exiting')
        return
    report_experiment_results.print_results(ibs, qrids, drids, cfg_list, mat_list, testnameid, sel_rows, sel_cols)
