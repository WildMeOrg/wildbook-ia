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

QUIET      = '--quiet' in sys.argv
BATCH_MODE = '--batch' in sys.argv
NOMEMORY   = '--nomemory' in sys.argv
NOCACHE_TESTRES =  utool.get_flag('--nocache-testres', False)
TEST_INFO = True


@profile
def get_qx2_bestrank(ibs, qreq, qrids, cfgx=0, nCfg=1):
    """
    Runs queries of a specific configuration returns the best rank of each query

    qrids - query roi ids

    """
    nQuery = len(qrids)
    drids = ibs.get_recognition_database_rids()
    TESTRES_VERBOSITY = 2 - (2 * QUIET)
    qx2_bestranks = eh.load_cached_test_results(ibs, qreq, qrids, drids, NOCACHE_TESTRES, TESTRES_VERBOSITY)
    if qx2_bestranks is not None:
        return qx2_bestranks

    qx2_bestranks = []

    # Perform queries
    if BATCH_MODE:
        print('[harn] querying in batch mode')
        #mc3.pre_cache_checks(ibs, qreq)
        qreq = mc3.prep_query_request(qreq=ibs.qreq,
                                      qrids=qrids,
                                      drids=drids,
                                      query_cfg=ibs.cfg.query_cfg)
        mc3.pre_exec_checks(ibs, qreq)
        qx2_bestranks = [None for qrid in qrids]
        # Query Chip / Row Loop
        qrid2_res = mc3.process_query_request(ibs, qreq, safe=False)
        qrid2_bestranks = {}
        for qrid, qres in qrid2_res.iteritems():
            gt_ranks = qres.get_gt_ranks(ibs=ibs)
            _rank = -1 if len(gt_ranks) == 0 else min(gt_ranks)
            qrid2_bestranks[qrid] = _rank
        try:
            for qx, qrid in enumerate(qrids):
                qx2_bestranks[qx] = [qrid2_bestranks[qrid]]
        except Exception as ex:
            utool.printex(ex, 'BATCH ERROR', key_list=[
                'qcid2_bestranks', 'qrid2_res', 'qrid', 'qx'])
            raise
    else:
        print('[harn] querying one query at a time')
        # Make progress message
        msg = textwrap.dedent('''
        ---------------------
        [harn] TEST %d/%d
        ---------------------''')
        mark_progress = utool.simple_progres_func(TESTRES_VERBOSITY, msg, '.')
        total = nQuery * nCfg
        nPrevQ = nQuery * cfgx
        #mc3.pre_cache_checks(ibs, qreq)
        mc3.pre_exec_checks(ibs, qreq)
        # Query Chip / Row Loop
        for qx, qrid in enumerate(qrids):
            count = qx + nPrevQ + 1
            mark_progress(count, total)
            if TEST_INFO:
                print('qrid=%r. quid=%r' % (qrid, qreq.get_uid()))
            try:
                qreq.qrids = [qrid]
                qrid2_res = mc3.process_query_request(ibs, qreq, safe=False)
            except mf.QueryException as ex:
                utool.printex(ex, 'Harness caught Query Exception')
                if params.args.strict:
                    raise
                else:
                    qx2_bestranks += [[-1]]
                    continue
            try:
                assert len(qrid2_res) == 1, ''
            except AssertionError as ex:
                utool.printex(ex, key_list=['qrid2_res'])
                raise
            qres = qrid2_res[qrid]
            gt_ranks = qres.get_gt_ranks(ibs=ibs)
            _rank = -1 if len(gt_ranks) == 0 else min(gt_ranks)
            # record metadata
            qx2_bestranks.append([_rank])
            if qrid % 4 == 0:
                sys.stdout.flush()
        print('')
    qx2_bestranks = np.array(qx2_bestranks)
    # High level caching
    eh.cache_test_results(qx2_bestranks, ibs, qreq, qrids, drids)
    return qx2_bestranks


#-----------
@utool.indent_func('[harn]')
@profile
def test_configurations(ibs, qrid_list, test_cfg_name_list, fnum=1):
    # Test Each configuration
    if not QUIET:
        print(textwrap.dedent("""
        [harn]================
        [harn] experiment_harness.test_configurations()""").strip())

    # Grab list of algorithm configurations to test
    cfg_list = eh.get_cfg_list(ibs, test_cfg_name_list)
    if not QUIET:
        print('[harn] Testing %d different parameters' % len(cfg_list))
        print('[harn]         %d different chips' % len(qrid_list))

    # Preallocate test result aggregation structures
    sel_cols = params.args.sel_cols  # FIXME
    sel_rows = params.args.sel_rows  # FIXME
    sel_cols = [] if sel_cols is None else sel_cols
    sel_rows = [] if sel_rows is None else sel_rows
    nCfg     = len(cfg_list)
    mat_list = []
    ibs._init_query_requestor()
    qreq = ibs.qreq

    test_cfg_verbosity = 2

    dbname = ibs.get_dbname()
    testnameid = dbname + ' ' + str(test_cfg_name_list)
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST_CFG %d/%d: ''' + testnameid + '''
    ---------------------''')
    mark_progress = utool.simple_progres_func(test_cfg_verbosity, msg, '+')

    # Run each test configuration
    # Query Config / Col Loop
    drids = ibs.get_recognition_database_rids()
    for cfgx, query_cfg in enumerate(cfg_list):
        if not QUIET:
            mark_progress(cfgx + 1, nCfg)
        # Set data to the current config
        qreq = mc3.prep_query_request(qreq=qreq, qrids=qrid_list, drids=drids, query_cfg=query_cfg)
        # Run the test / read cache
        with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
            qx2_bestranks = get_qx2_bestrank(ibs, qreq, qrid_list, cfgx, nCfg)
        if not NOMEMORY:
            mat_list.append(qx2_bestranks)
        # Store the results

    if not QUIET:
        print('[harn] Finished testing parameters')
    if NOMEMORY:
        print('ran tests in memory savings mode. exiting')
        return

    report_experiment_results.print_results(ibs, qrid_list, drids, cfg_list, mat_list, testnameid, sel_rows, sel_cols)
