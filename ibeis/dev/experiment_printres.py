from __future__ import absolute_import, division, print_function
import itertools
from itertools import chain, imap
import utool
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
import numpy as np
from ibeis.dev import ibsfuncs
from ibeis.dev import experiment_helpers as eh
from ibeis.model.hots import match_chips3 as mc3
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_report]', DEBUG=False)


QUIET = utool.QUIET
SKIP_TO = utool.get_arg('--skip-to', default=None)
SAVE_FIGURES = utool.get_flag(('--save-figures', '--sf'))


def get_diffmat_str(rank_mat, qrids, nCfg):
    # Find rows which scored differently over the various configs
    row2_rid = np.array(qrids)
    diff_rows = np.where([not np.all(row == row[0]) for row in rank_mat])[0]
    diff_rids = row2_rid[diff_rows]
    diff_rank = rank_mat[diff_rows]
    diff_mat = np.vstack((diff_rids, diff_rank.T)).T
    col_labels = list(chain(['qrid'], imap(lambda x: 'cfg%d_rank' % x, xrange(nCfg))))
    col_types  = list(chain([int], [int] * nCfg))
    header = 'diffmat'
    diff_matstr = utool.numpy_to_csv(diff_mat, col_labels, header, col_types)
    return diff_matstr


def print_results(ibs, qrids, drids, cfg_list, mat_list, testnameid,
                  sel_rows, sel_cols, cfgx2_lbl=None):
    nCfg = len(cfg_list)
    nQuery = len(qrids)
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
        qrid = qrids[qx]
        label = 'qx=%d) q%s ' % (qx, ibsfuncs.ridstr(qrid, ibs=ibs, notes=True))
        qx2_lbl.append(label)
    qx2_lbl = np.array(qx2_lbl)
    #------------
    # Build col labels
    if cfgx2_lbl is None:
        cfgx2_lbl = []
        for cfgx in xrange(nCfg):
            test_cfgstr  = cfg_list[cfgx].get_cfgstr()
            cfg_label = 'cfgx=(%3d) %s' % (cfgx, test_cfgstr)
            cfgx2_lbl.append(cfg_label)
        cfgx2_lbl = np.array(cfgx2_lbl)
    #------------
    indent = utool.indent

    @utool.argv_flag_dec
    def print_rowlbl():
        print('=====================')
        print('[harn] Row/Query Labels: %s' % testnameid)
        print('=====================')
        print('[harn] queries:\n%s' % '\n'.join(qx2_lbl))
        print('--- /Row/Query Labels ---')
    print_rowlbl()

    #------------

    @utool.argv_flag_dec
    def print_collbl():
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
    new_qrids = []
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
        qrid = qrids[qx]
        name = ibs.get_roi_names(qrid)
        new_hardtup_list += [(qrid, name + " - " + notes)]
        new_qrids += [qrid]

    @utool.argv_flag_dec
    def print_rowscore():
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

    @utool.argv_flag_dec
    def print_hardcase():
        print('===')
        print('--- hard new_hardtup_list (w.r.t these configs): %s' % testnameid)
        print('\n'.join(map(repr, new_hardtup_list)))
        print('There are %d hard cases ' % len(new_hardtup_list))
        rid_list = [rid_notes[0] for rid_notes in new_hardtup_list]
        name_list = ibs.get_roi_names(rid_list)
        name_set = set(name_list)
        print(sorted(rid_list))
        print('Names: %r' % (name_set,))
        print('--- /Print Hardcase ---')
    print_hardcase()

    @utool.argv_flag_dec
    def echo_hardcase():
        print('====')
        print('--- hardcase commandline: %s' % testnameid)
        hardrids_str = ' '.join(map(str, ['    ', '--qrid'] + new_qrids))
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

    @utool.argv_flag_dec
    def print_colscore():
        print('==================')
        print('[harn] Scores per Config: %s' % testnameid)
        print('==================')
        for cfgx in xrange(nCfg):
            print('[score] %s' % (cfgx2_lbl[cfgx]))
            for X in iter(X_list):
                nLessX_ = nLessX_dict[int(X)][cfgx]
                print('        ' + eh.rankscore_str(X, nLessX_, nQuery))
        print('--- /Scores per Config ---')
    print_colscore()

    #------------

    @utool.argv_flag_dec
    def print_latexsum():
        print('==========================')
        print('[harn] LaTeX: %s' % testnameid)
        print('==========================')
        # Create configuration latex table
        criteria_lbls = ['#ranks < %d' % X for X in X_list]
        dbname = ibs.get_dbname()
        cfg_score_title = dbname + ' rank scores'
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
        best_rankscore += eh.rankscore_str(X, max_LessX, nQuery)
        best_rankscore_summary += [best_rankscore]
        to_intersect_list += [cfgx2_lbl[bestCFG_X]]

    intersected = to_intersect_list[0] if len(to_intersect_list) > 0 else []
    for ix in xrange(1, len(to_intersect_list)):
        intersected = np.intersect1d(intersected, to_intersect_list[ix])

    @utool.argv_flag_dec
    def print_bestcfg():
        print('==========================')
        print('[harn] Best Configurations: %s' % testnameid)
        print('==========================')
        # print each configs scores less than X=thresh
        for X, cfgx2_nLessX in nLessX_dict.iteritems():
            max_LessX = cfgx2_nLessX.max()
            bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
            best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestCFG_X)
            best_rankscore += eh.rankscore_str(X, max_LessX, nQuery)
            cfgstr_list = cfgx2_lbl[bestCFG_X]

            best_rankcfg = eh.format_cfgstr_list(cfgstr_list)
            #indent('\n'.join(cfgstr_list), '    ')
            print(best_rankscore)
            print(best_rankcfg)
        print('[cfg*]  %d cfg(s) are the best of %d total cfgs' % (len(intersected), nCfg))
        print(eh.format_cfgstr_list(intersected))

        print('--- /Best Configurations ---')
    print_bestcfg()

    #------------

    @utool.argv_flag_dec
    def print_rankmat():
        print('-------------')
        print('RankMat: %s' % testnameid)
        print(' nRows=%r, nCols=%r' % lbld_mat.shape)
        print(' labled rank matrix: rows=queries, cols=cfgs:')
        #np.set_printoptions(threshold=5000, linewidth=5000, precision=5)
        with utool.NpPrintOpts(threshold=5000, linewidth=5000, precision=5):
            print(lbld_mat)
        print('[harn]-------------')
    print_rankmat()

    @utool.argv_flag_dec
    def print_diffmat():
        print('-------------')
        print('Diffmat: %s' % testnameid)
        diff_matstr = get_diffmat_str(rank_mat, qrids, nCfg)
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
    if not QUIET:
        print('remember to inspect with --sel-rows (-r) and --sel-cols (-c) ')
    if len(sel_rows) > 0 and len(sel_cols) == 0:
        sel_cols = range(len(cfg_list))
    if len(sel_cols) > 0 and len(sel_rows) == 0:
        sel_rows = range(len(qrids))
    if utool.get_arg('--view-all'):
        sel_rows = range(len(qrids))
        sel_cols = range(len(cfg_list))
    sel_cols = list(sel_cols)
    sel_rows = list(sel_rows)
    total = len(sel_cols) * len(sel_rows)
    rciter = itertools.product(sel_rows, sel_cols)

    skip_list = []

    def load_qres(ibs, qrid, drids, query_cfg):
        # Load / Execute the query w/ correct config
        ibs.set_query_cfg(query_cfg)
        qreq = mc3.prep_query_request(qreq=ibs.qreq, qrids=[qrid], drids=drids, query_cfg=query_cfg)
        qrid2_qres = mc3.process_query_request(ibs, qreq, safe=True,
                                               use_bigcache=True,
                                               use_cache=True)
        qres = qrid2_qres[qrid]
        return qres

    for count, (r, c) in enumerate(rciter):
        if SKIP_TO is not None:
            if count < SKIP_TO:
                continue
        if count in skip_list:
            continue
        # Get row and column index
        qrid      = qrids[r]
        query_cfg = cfg_list[c]
        query_lbl = cfgx2_lbl[c]
        print(utool.unindent('''
        __________________________________
        --- VIEW %d / %d --- (r=%r, c=%r)
        ----------------------------------
        ''')  % (count + 1, total, r, c))
        qres = load_qres(ibs, qrid, drids, query_cfg)
        # Draw Result
        show_kwargs = {
            'N': 3,
            'ori': True,
            'ell_alpha': .9,
        }
        qres.show(ibs, 'analysis', figtitle=query_lbl, **show_kwargs)
        df2.adjust_subplots_safe()
        if SAVE_FIGURES:
            ph.dump_figure(ibs.get_ibsdir(),
                           subdir='figures_analysis',
                           quality=False,
                           overwrite=True,
                           verbose=1)
    if not QUIET:
        print('[harn] EXIT EXPERIMENT HARNESS')
