from __future__ import division, print_function
import utool
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[harn]', DEBUG=False)
# Python
import sys
import itertools
import textwrap
from itertools import chain
#from os.path import join
from itertools import imap
# Scientific
import numpy as np
# Hotspotter
import experiment_configs
from ibeis.model import Config
from ibeis.dev import ibsfuncs
from ibeis.model.hots import match_chips3 as mc3
from ibeis.model.hots import matching_functions as mf
from ibeis.dev import params
from plottool import draw_func2 as df2
#from hscom import fileio as io
#from hscom import util
#from hscom import latex_formater
#from hscom import csvtool
#from match_chips3 import *
#import draw_func2 as df2
# What are good ways we can divide up FLANN indexes instead of having one
# monolithic index? Divide up in terms of properties of the database chips

# Can also reduce the chips being indexed

# What happens when we take all other possible ground truth matches out
# of the database index?


def get_valid_testcfg_names():
    testcfg_keys = vars(experiment_configs).keys()
    testcfg_locals = [key for key in testcfg_keys if key.find('_') != 0]
    valid_cfg_names = utool.indent('\n'.join(testcfg_locals), '  * ')
    return valid_cfg_names


def get_vary_dicts(test_cfg_name_list):
    vary_dicts = []
    for cfg_name in test_cfg_name_list:
        test_cfg = experiment_configs.__dict__[cfg_name]
        vary_dicts.append(test_cfg)
    if len(vary_dicts) == 0:
        valid_cfg_names = get_valid_testcfg_names()
        raise Exception('Choose a valid testcfg:\n' + valid_cfg_names)
    return vary_dicts


__QUIET__ = '--quiet' in sys.argv

#---------
# Helpers
#---------------
# Display Test Results


def argv_flag_dec(func):
    return __argv_flag_dec(func, default=False)


def argv_flag_dec_true(func):
    return __argv_flag_dec(func, default=True)


def __argv_flag_dec(func, default=False):
    flag = func.func_name
    if flag.find('no') == 0:
        flag = flag[2:]
    flag = '--' + flag.replace('_', '-')

    def GaurdWrapper(*args, **kwargs):
        if utool.get_flag(flag, default):
            indent_lbl = flag.replace('--', '').replace('print-', '')
            with utool.Indenter('[%s]' % indent_lbl):
                return func(*args, **kwargs)
        else:
            if not __QUIET__:
                print('\n~~~ %s ~~~\n' % flag)
    GaurdWrapper.func_name = func.func_name
    return GaurdWrapper


def rankscore_str(thresh, nLess, total):
    #helper to print rank scores of configs
    percent = 100 * nLess / total
    fmtsf = '%' + str(utool.num2_sigfig(total)) + 'd'
    fmtstr = '#ranks < %d = ' + fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + ')'
    rankscore_str = fmtstr % (thresh, nLess, total, percent, (total - nLess))
    return rankscore_str


def wrap_uid(uid):
    import re
    # REGEX to locate _XXXX(
    cfg_regex = r'_[A-Z][A-Z]*\('
    uidmarker_list = re.findall(cfg_regex, uid)
    uidconfig_list = re.split(cfg_regex, uid)
    args = [uidconfig_list, uidmarker_list]
    interleave_iter = utool.interleave(args)
    new_uid_list = []
    total_len = 0
    prefix_str = ''
    # If unbalanced there is a prefix before a marker
    if len(uidmarker_list) < len(uidconfig_list):
        frag = interleave_iter.next()
        new_uid_list += [frag]
        total_len = len(frag)
        prefix_str = ' ' * len(frag)
    # Iterate through markers and config strings
    while True:
        try:
            marker_str = interleave_iter.next()
            config_str = interleave_iter.next()
            frag = marker_str + config_str
        except StopIteration:
            break
        total_len += len(frag)
        new_uid_list += [frag]
        # Go to newline if past 80 chars
        if total_len > 80:
            total_len = 0
            new_uid_list += ['\n' + prefix_str]
    wrapped_uid = ''.join(new_uid_list)
    return wrapped_uid


def format_uid_list(uid_list):
    indented_list = utool.indent_list('    ', uid_list)
    wrapped_list = imap(wrap_uid, indented_list)
    return utool.joins('\n', wrapped_list)


#---------------
# Big Test Cache
#-----------

def load_cached_test_results(ibs, qreq, qrids, drids, nocache_testres, test_results_verbosity):
    pass
    #test_uid = qreq.get_query_uid(ibs, qrids)
    #cache_dir = join(ibs.dirs.cache_dir, 'experiment_harness_results')
    #io_kwargs = {'dpath': cache_dir,
                 #'fname': 'test_results',
                 #'uid': test_uid,
                 #'ext': '.cPkl'}

    #if test_results_verbosity == 2:
        #print('[harn] test_uid = %r' % test_uid)

    ## High level caching
    #if not params.args.nocache_query and (not nocache_testres):
        #qx2_bestranks = io.smart_load(**io_kwargs)
        #if qx2_bestranks is None:
            #print('[harn] qx2_bestranks cache returned None!')
        #elif len(qx2_bestranks) != len(qrids):
            #print('[harn] Re-Caching qx2_bestranks')
        #else:
            #return qx2_bestranks


def cache_test_results(qx2_bestranks, ibs, qreq, qrids, drids):
    pass
    #test_uid = qreq.get_query_uid(ibs, qrids)
    #cache_dir = join(ibs.dirs.cache_dir, 'experiment_harness_results')
    #utool.ensuredir(cache_dir)
    #io_kwargs = {'dpath': cache_dir,
                 #'fname': 'test_results',
                 #'uid': test_uid,
                 #'ext': '.cPkl'}
    #io.smart_save(qx2_bestranks, **io_kwargs)


#---------------
# Display Test Results
#-----------
# Run configuration for each query
@profile
def get_test_results2(ibs, qrids, qreq, cfgx=0, nCfg=1, nocache_testres=False,
                      test_results_verbosity=2):
    TEST_INFO = True
    nQuery = len(qrids)
    drids = ibs.get_recognition_database_rids()
    qx2_bestranks = load_cached_test_results(ibs, qreq, qrids, drids,
                                             nocache_testres, test_results_verbosity)
    if qx2_bestranks is not None:
        return qx2_bestranks

    qx2_bestranks = []

    # Perform queries
    BATCH_MODE = '--batch' in sys.argv
    #BATCH_MODE = '--batch' in sys.argv
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
            print('[harn] ERROR')
            print(ex)
            print('qrid2_bestranks=%r' % qrid2_bestranks)
            print('qrid2_res=%r' % qrid2_res)
            print('qrid=%r' % qrid)
            print('qx=%r' % qx)
            raise
    else:
        print('[harn] querying one query at a time')
        # Make progress message
        msg = textwrap.dedent('''
        ---------------------
        [harn] TEST %d/%d
        ---------------------''')
        mark_progress = utool.simple_progres_func(test_results_verbosity, msg, '.')
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
                qreq._qrids = [qrid]
                qrid2_res = mc3.process_query_request(ibs, qreq, safe=False)
            except mf.QueryException as ex:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Harness caught Query Exception: ')
                print(ex)
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                if params.args.strict:
                    raise
                else:
                    qx2_bestranks += [[-1]]
                    continue

            assert len(qrid2_res) == 1
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
    cache_test_results(qx2_bestranks, ibs, qreq, qrids, drids)
    return qx2_bestranks


def get_varied_params_list(test_cfg_name_list):
    vary_dicts = get_vary_dicts(test_cfg_name_list)
    get_all_dict_comb = utool.all_dict_combinations
    dict_comb_list = [get_all_dict_comb(_dict) for _dict in vary_dicts]
    varied_params_list = [comb for dict_comb in dict_comb_list for comb in dict_comb]
    #map(lambda x: print('\n' + str(x)), varied_params_list)
    return varied_params_list


def get_cfg_list(ibs, test_cfg_name_list):
    print('[harn] building cfg_list: %s' % test_cfg_name_list)
    if 'custom' == test_cfg_name_list:
        print('   * custom cfg_list')
        cfg_list = [ibs.prefs.query_cfg]
        return cfg_list
    varied_params_list = get_varied_params_list(test_cfg_name_list)
    # Add unique configs to the list
    cfg_list = []
    cfg_set = set([])
    for _dict in varied_params_list:
        cfg = Config.QueryConfig(**_dict)
        if not cfg in cfg_set:
            cfg_list.append(cfg)
            cfg_set.add(cfg)
    if not __QUIET__:
        print('[harn] reduced equivilent cfgs %d / %d cfgs' % (len(cfg_list),
                                                               len(varied_params_list)))

    return cfg_list


#-----------
#@utool.indent_decor('[harn]')
@profile
def test_configurations(ibs, qrid_list, test_cfg_name_list, fnum=1):
    if __QUIET__:
        mc3.print_off()
        from hsapi import HotSpotterAPI as api
        api.print_off()

    # Test Each configuration
    if not __QUIET__:
        print(textwrap.dedent("""
        [harn]================
        [harn] experiment_harness.test_configurations()""").strip())

    # Grab list of algorithm configurations to test
    cfg_list = get_cfg_list(ibs, test_cfg_name_list)
    if not __QUIET__:
        print('[harn] Testing %d different parameters' % len(cfg_list))
        print('[harn]         %d different chips' % len(qrid_list))

    # Preallocate test result aggregation structures
    sel_cols = params.args.sel_cols  # FIXME
    sel_rows = params.args.sel_rows  # FIXME
    sel_cols = [] if sel_cols is None else sel_cols
    sel_rows = [] if sel_rows is None else sel_rows
    nCfg     = len(cfg_list)
    nQuery   = len(qrid_list)
    #rc2_res = np.empty((nQuery, nCfg), dtype=list)  # row/col -> result
    mat_list = []
    #_class = QueryResult.QueryResult
    ibs._init_query_requestor()
    qreq = ibs.qreq

    # TODO Add to argparse2
    nocache_testres =  utool.get_flag('--nocache-testres', False)

    test_results_verbosity = 2 - (2 * __QUIET__)
    test_cfg_verbosity = 2

    dbname = ibs.get_dbname()
    testnameid = dbname + ' ' + str(test_cfg_name_list)
    msg = textwrap.dedent('''
    ---------------------
    [harn] TEST_CFG %d/%d: ''' + testnameid + '''
    ---------------------''')
    mark_progress = utool.simple_progres_func(test_cfg_verbosity, msg, '+')

    nomemory = utool.get_flag('--nomemory')

    # Run each test configuration
    # Query Config / Col Loop
    drids = ibs.get_recognition_database_rids()
    for cfgx, query_cfg in enumerate(cfg_list):
        if not __QUIET__:
            mark_progress(cfgx + 1, nCfg)
        # Set data to the current config
        qreq = mc3.prep_query_request(qreq=qreq, qrids=qrid_list, drids=drids, query_cfg=query_cfg)
        # Run the test / read cache
        #with utool.Indenter('[%s cfg %d/%d]' % (dbname, cfgx + 1, nCfg)):
        qx2_bestranks = get_test_results2(ibs, qrid_list, qreq, cfgx,
                                          nCfg, nocache_testres,
                                          test_results_verbosity)
        if not nomemory:
            mat_list.append(qx2_bestranks)
        # Store the results

    if not __QUIET__:
        print('[harn] Finished testing parameters')
    if nomemory:
        print('ran tests in memory savings mode. exiting')
        return
    #--------------------
    # Print Best Results
    rank_mat = np.hstack(mat_list)  # concatenate each query rank across configs
    # Label the rank matrix:
    _colxs = np.arange(nCfg)
    lbld_mat = utool.debug_vstack([_colxs, rank_mat])

    _rowxs = np.arange(nQuery + 1).reshape(nQuery + 1, 1) - 1
    lbld_mat = np.hstack([_rowxs, lbld_mat])
    #------------
    # Build row labels
    qx2_lbl = []
    for qx in xrange(nQuery):
        qrid = qrid_list[qx]
        label = 'qx=%d) q%s ' % (qx, ibsfuncs.ridstr(qrid, notes=True))
        qx2_lbl.append(label)
    qx2_lbl = np.array(qx2_lbl)
    #------------
    # Build col labels
    cfgx2_lbl = []
    for cfgx in xrange(nCfg):
        test_uid  = cfg_list[cfgx].get_uid()
        test_uid  = cfg_list[cfgx].get_uid()
        cfg_label = 'cfgx=(%3d) %s' % (cfgx, test_uid)
        cfgx2_lbl.append(cfg_label)
    cfgx2_lbl = np.array(cfgx2_lbl)
    #------------
    indent = utool.indent

    @argv_flag_dec
    def print_rowlbl():
        print('=====================')
        print('[harn] Row/Query Labels: %s' % testnameid)
        print('=====================')
        print('[harn] queries:\n%s' % '\n'.join(qx2_lbl))
        print('--- /Row/Query Labels ---')
    print_rowlbl()

    #------------

    @argv_flag_dec
    def print_collbl():
        print('')
        print('=====================')
        print('[harn] Col/Config Labels: %s' % testnameid)
        print('=====================')
        print('[harn] configs:\n%s' % '\n'.join(cfgx2_lbl))
        print('--- /Col/Config Labels ---')
    print_collbl()

    #------------
    # Build Colscore
    qx2_min_rank = []
    qx2_argmin_rank = []
    new_hard_qx_list = []
    new_qrid_list = []
    new_hardtup_list = []
    for qx in xrange(nQuery):
        ranks = rank_mat[qx]
        min_rank = ranks.min()
        bestCFG_X = np.where(ranks == min_rank)[0]
        qx2_min_rank.append(min_rank)
        qx2_argmin_rank.append(bestCFG_X)
        # Mark examples as hard
        if ranks.max() > 0:
            new_hard_qx_list += [qx]
    for qx in new_hard_qx_list:
        # New list is in rid format instead of cx format
        # because you should be copying and pasting it
        notes = ' ranks = ' + str(rank_mat[qx])
        qrid = qrid_list[qx]
        new_hardtup_list += [(qrid, notes)]
        new_qrid_list += [qrid]

    @argv_flag_dec
    def print_rowscore():
        print('')
        print('=======================')
        print('[harn] Scores per Query: %s' % testnameid)
        print('=======================')
        for qx in xrange(nQuery):
            bestCFG_X = qx2_argmin_rank[qx]
            min_rank = qx2_min_rank[qx]
            minimizing_cfg_str = indent('\n'.join(cfgx2_lbl[bestCFG_X]), '    ')
            #minimizing_cfg_str = str(bestCFG_X)

            print('-------')
            print(qx2_lbl[qx])
            print(' best_rank = %d ' % min_rank)
            if len(cfgx2_lbl) != 1:
                print(' minimizing_cfg_x\'s = %s ' % minimizing_cfg_str)

    print_rowscore()

    #------------

    @argv_flag_dec
    def print_hardcase():
        print('===')
        print('--- hard new_hardtup_list (w.r.t these configs): %s' % testnameid)
        print('\n'.join(map(repr, new_hardtup_list)))
        print('There are %d hard cases ' % len(new_hardtup_list))
        print(sorted([x[0] for x in new_hardtup_list]))
        print('--- /Print Hardcase ---')
    print_hardcase()

    @argv_flag_dec
    def echo_hardcase():
        print('====')
        print('--- hardcase commandline: %s' % testnameid)
        hardrids_str = ' '.join(map(str, ['    ', '--qrid'] + new_qrid_list))
        print(hardrids_str)
        print('--- /Echo Hardcase ---')
    echo_hardcase()

    #------------
    # Build Colscore
    X_list = [1, 5]
    # Build a dictionary mapping X (as in #ranks < X) to a list of cfg scores
    nLessX_dict = {int(X): np.zeros(nCfg) for X in iter(X_list)}
    for cfgx in xrange(nCfg):
        ranks = rank_mat[:, cfgx]
        for X in iter(X_list):
            #nLessX_ = sum(np.bitwise_and(ranks < X, ranks >= 0))
            nLessX_ = sum(np.logical_and(ranks < X, ranks >= 0))
            nLessX_dict[int(X)][cfgx] = nLessX_

    @argv_flag_dec
    def print_colscore():
        print('')
        print('==================')
        print('[harn] Scores per Config: %s' % testnameid)
        print('==================')
        for cfgx in xrange(nCfg):
            print('[score] %s' % (cfgx2_lbl[cfgx]))
            for X in iter(X_list):
                nLessX_ = nLessX_dict[int(X)][cfgx]
                print('        ' + rankscore_str(X, nLessX_, nQuery))
        print('--- /Scores per Config ---')
    print_colscore()

    #------------

    @argv_flag_dec
    def print_latexsum():
        print('')
        print('==========================')
        print('[harn] LaTeX: %s' % testnameid)
        print('==========================')
        # Create configuration latex table
        criteria_lbls = ['#ranks < %d' % X for X in X_list]
        db_name = ibs.get_dbname(True)
        cfg_score_title = db_name + ' rank scores'
        cfgscores = np.array([nLessX_dict[int(X)] for X in X_list]).T

        replace_rowlbl = [(' *cfgx *', ' ')]
        tabular_kwargs = dict(title=cfg_score_title, out_of=nQuery,
                              bold_best=True, replace_rowlbl=replace_rowlbl,
                              flip=True)
        tabular_str = utool.util_latex.make_score_tabular(cfgx2_lbl,
                                                          criteria_lbls,
                                                          cfgscores,
                                                          **tabular_kwargs)
        #latex_formater.render(tabular_str)
        print(tabular_str)
        print('--- /LaTeX ---')
    print_latexsum()

    #------------
    best_rankscore_summary = []
    to_intersect_list = []
    # print each configs scores less than X=thresh
    for X, cfgx2_nLessX in nLessX_dict.iteritems():
        max_LessX = cfgx2_nLessX.max()
        bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
        best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestCFG_X)
        best_rankscore += rankscore_str(X, max_LessX, nQuery)
        best_rankscore_summary += [best_rankscore]
        to_intersect_list += [cfgx2_lbl[bestCFG_X]]

    intersected = to_intersect_list[0] if len(to_intersect_list) > 0 else []
    for ix in xrange(1, len(to_intersect_list)):
        intersected = np.intersect1d(intersected, to_intersect_list[ix])

    @argv_flag_dec
    def print_bestcfg():
        print('')
        print('==========================')
        print('[harn] Best Configurations: %s' % testnameid)
        print('==========================')
        # print each configs scores less than X=thresh
        for X, cfgx2_nLessX in nLessX_dict.iteritems():
            max_LessX = cfgx2_nLessX.max()
            bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
            best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestCFG_X)
            best_rankscore += rankscore_str(X, max_LessX, nQuery)
            uid_list = cfgx2_lbl[bestCFG_X]

            #best_rankcfg = ''.join(map(wrap_uid, uid_list))
            best_rankcfg = format_uid_list(uid_list)
            #indent('\n'.join(uid_list), '    ')
            print(best_rankscore)
            print(best_rankcfg)

        print('[cfg*]  %d cfg(s) are the best of %d total cfgs' % (len(intersected), nCfg))
        print(format_uid_list(intersected))

        print('--- /Best Configurations ---')
    print_bestcfg()

    #------------

    @argv_flag_dec
    def print_rankmat():
        print('')
        print('-------------')
        print('RankMat: %s' % testnameid)
        print(' nRows=%r, nCols=%r' % lbld_mat.shape)
        print(' labled rank matrix: rows=queries, cols=cfgs:')
        #np.set_printoptions(threshold=5000, linewidth=5000, precision=5)
        with utool.NpPrintOpts(threshold=5000, linewidth=5000, precision=5):
            print(lbld_mat)
        print('[harn]-------------')
    print_rankmat()

    row2_rid = np.array(qrid_list)
    # Find rows which scored differently over the various configs
    diff_rows = np.where([not np.all(row == row[0]) for row in rank_mat])[0]
    diff_rids = row2_rid[diff_rows]
    diff_rank = rank_mat[diff_rows]
    diff_mat = np.vstack((diff_rids, diff_rank.T)).T
    col_labels = list(chain(['qrid'], imap(lambda x: 'cfg%d_rank' % x, xrange(nCfg))))
    col_types  = list(chain([int], [int] * nCfg))
    header = 'rankmat2'
    diff_matstr = utool.numpy_to_csv(diff_mat, col_labels, header, col_types)
    @argv_flag_dec
    def print_diffmat():
        print('')
        print('-------------')
        print('RankMat2: %s' % testnameid)
        print(diff_matstr)
        print('[harn]-------------')
    print_diffmat()

    #------------
    sumstrs = []
    sumstrs.append('')
    sumstrs.append('||===========================')
    sumstrs.append('|| [cfg*] SUMMARY: %s' % testnameid)
    sumstrs.append('||---------------------------')
    sumstrs.append(utool.joins('\n|| ', best_rankscore_summary))
    sumstrs.append('||===========================')
    print('\n' + '\n'.join(sumstrs) + '\n')
    #print('--- /SUMMARY ---')

    # Draw results
    if not __QUIET__:
        print('remember to inspect with --sel-rows (-r) and --sel-cols (-c) ')
    if len(sel_rows) > 0 and len(sel_cols) == 0:
        sel_cols = range(len(cfg_list))
    if len(sel_cols) > 0 and len(sel_rows) == 0:
        sel_rows = range(len(qrid_list))
    if utool.get_arg('--view-all'):
        sel_rows = range(len(qrid_list))
        sel_cols = range(len(cfg_list))
    sel_cols = list(sel_cols)
    sel_rows = list(sel_rows)
    total = len(sel_cols) * len(sel_rows)
    rciter = itertools.product(sel_rows, sel_cols)

    prev_cfg = None

    skip_to = utool.get_arg('--skip-to', default=None)

    dev_mode = utool.get_arg('--devmode', default=False)
    skip_list = []
    if dev_mode:
        ibs.prefs.display_cfg.N = 3
        df2.FONTS.axtitle = df2.FONTS.smaller
        df2.FONTS.xlabel = df2.FONTS.smaller
        df2.FONTS.figtitle = df2.FONTS.smaller
        df2.SAFE_POS['top']    = .8
        df2.SAFE_POS['bottom'] = .01

    for count, (r, c) in enumerate(rciter):
        if skip_to is not None:
            if count < skip_to:
                continue
        if count in skip_list:
            continue
        # Get row and column index
        qrid       = qrid_list[r]
        query_cfg = cfg_list[c]
        print('\n\n___________________________________')
        print('      --- VIEW %d / %d ---        '
              % (count + 1, total))
        print('--------------------------------------')
        print('viewing (r, c) = (%r, %r)' % (r, c))
        # Load / Execute the query
        qreq = mc3.prep_query_request(qreq=qreq, qrids=[qrid], drids=drids, query_cfg=query_cfg)
        qrid2_res = mc3.process_query_request(ibs, qreq, safe=True)
        qres = qrid2_res[qrid]
        # Print Query UID
        print(qres.uid)
        # Draw Result
        qres.show_top(ibs, fnum=fnum)
        if prev_cfg != query_cfg:
            # This is way too aggro. Needs to be a bit lazier
            #ibs.refresh_features()
            print('change')
        prev_cfg = query_cfg
        fnum = count
        title_uid = qres.uid
        title_uid = title_uid.replace('_FEAT', '\n_FEAT')
        qres.show_analysis(ibs, fnum=fnum, aug='\n' + title_uid, annote=1,
                           show_name=False, show_gname=False, time_appart=False)
        df2.adjust_subplots_safe()
        if utool.get_flag('--save-figures'):
            from hsviz import allres_viz
            allres_viz.dump(ibs, 'analysis', quality=True, overwrite=False)
    if not __QUIET__:
        print('[harn] EXIT EXPERIMENT HARNESS')
