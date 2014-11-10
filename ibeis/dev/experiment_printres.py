"""
displays results from experiment_harness
"""
from __future__ import absolute_import, division, print_function
import itertools
import numpy as np
import six
import utool
import utool as ut  # NOQA
from ibeis import ibsfuncs
from ibeis.dev import experiment_helpers as eh
from ibeis.model.hots import match_chips4 as mc4
from itertools import chain
from os.path import join, dirname, split, basename, splitext
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
from six.moves import map, range
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[expt_report]')


SKIP_TO = utool.get_argval('--skip-to', default=None)
#SAVE_FIGURES = utool.get_argflag(('--save-figures', '--sf'))
SAVE_FIGURES = not utool.get_argflag(('--nosave-figures', '--nosf'))


def get_diffmat_str(rank_mat, qaids, nCfg):
    # Find rows which scored differently over the various configs
    row2_aid = np.array(qaids)
    diff_rows = np.where([not np.all(row == row[0]) for row in rank_mat])[0]
    diff_aids = row2_aid[diff_rows]
    diff_rank = rank_mat[diff_rows]
    diff_mat = np.vstack((diff_aids, diff_rank.T)).T
    col_lbls = list(chain(['qaid'], map(lambda x: 'cfg%d_rank' % x, range(nCfg))))
    col_types  = list(chain([int], [int] * nCfg))
    header = 'diffmat'
    diff_matstr = utool.numpy_to_csv(diff_mat, col_lbls, header, col_types)
    return diff_matstr


def print_results(ibs, qaids, daids, cfg_list, bestranks_list, cfgx2_aveprecs,
                  testnameid, sel_rows, sel_cols, cfgx2_lbl):
    """
    Prints results from an experiment harness run.
    Rows store different qaids (query annotation ids)
    Cols store different configurations (algorithm parameters)
    """
    print(' --- PRINT RESULTS ---')
    nCfg = len(cfg_list)
    nQuery = len(qaids)
    #--------------------
    # Print Best Results
    rank_mat = np.hstack(bestranks_list)  # concatenate each query rank across configs
    # Label the rank matrix:
    _colxs = np.arange(nCfg)
    lbld_mat = utool.debug_vstack([_colxs, rank_mat])

    _rowxs = np.arange(nQuery + 1).reshape(nQuery + 1, 1) - 1
    lbld_mat = np.hstack([_rowxs, lbld_mat])
    #------------
    # Build row lbls
    qx2_lbl = []
    for qx in range(nQuery):
        qaid = qaids[qx]
        lbl = 'qx=%d) q%s ' % (qx, ibsfuncs.aidstr(qaid, ibs=ibs, notes=True))
        qx2_lbl.append(lbl)
    qx2_lbl = np.array(qx2_lbl)
    #------------
    # Build col lbls (this info is passed in)
    #if cfgx2_lbl is None:
    #    cfgx2_lbl = []
    #    for cfgx in range(nCfg):
    #        test_cfgstr  = cfg_list[cfgx].get_cfgstr()
    #        cfg_lbl = 'cfgx=(%3d) %s' % (cfgx, test_cfgstr)
    #        cfgx2_lbl.append(cfg_lbl)
    #    cfgx2_lbl = np.array(cfgx2_lbl)
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
        enum_cfgx2_lbl = ['%2d) %s' % (count, cfglbl)
                            for count, cfglbl in enumerate(cfgx2_lbl)]
        print('[harn] cfglbl:\n%s' % '\n'.join(enum_cfgx2_lbl))
        print('--- /Col/Config Labels ---')
    print_collbl()

    #------------

    @utool.argv_flag_dec_true
    def print_cfgstr():
        print('=====================')
        print('[harn] Config Strings: %s' % testnameid)
        print('=====================')
        cfgstr_list = [query_cfg.get_cfgstr() for query_cfg in cfg_list]
        enum_cfgstr_list = ['%2d) %s' % (count, cfgstr)
                            for count, cfgstr in enumerate(cfgstr_list)]
        print('\n[harn] cfgstr:\n%s' % '\n'.join(enum_cfgstr_list))
        print('--- /Config Strings ---')
    print_cfgstr()

    #------------
    # Build Colscore and hard cases
    qx2_min_rank = []
    qx2_argmin_rank = []
    new_hard_qx_list = []
    new_qaids = []
    new_hardtup_list = []

    worst_possible_rank = max(9001, len(daids) + len(qaids) + 1)

    for qx in range(nQuery):
        ranks = rank_mat[qx]
        ranks[ranks == -1] = worst_possible_rank
        valid_ranks = ranks[ranks < 0]
        min_rank = ranks.min() if len(valid_ranks) > 0 else -3
        bestCFG_X = np.where(ranks == min_rank)[0]
        qx2_min_rank.append(min_rank)
        # Find the best rank over all configurations
        qx2_argmin_rank.append(bestCFG_X)
        # Mark examples as hard
        worst_rank = ranks.max()
        if worst_rank > 0 or worst_rank < 0:
            new_hard_qx_list += [qx]
    for qx in new_hard_qx_list:
        # New list is in aid format instead of cx format
        # because you should be copying and pasting it
        notes = ' ranks = ' + str(rank_mat[qx])
        qaid = qaids[qx]
        name = ibs.get_annot_names(qaid)
        new_hardtup_list += [(qaid, name + " - " + notes)]
        new_qaids += [qaid]

    @utool.argv_flag_dec
    def print_rowscore():
        print('=======================')
        print('[harn] Scores per Query: %s' % testnameid)
        print('=======================')
        for qx in range(nQuery):
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

    @utool.argv_flag_dec
    def print_row_ave_precision():
        print('=======================')
        print('[harn] Scores per Query: %s' % testnameid)
        print('=======================')
        for qx in range(nQuery):
            aveprecs = ', '.join(['%.2f' % (aveprecs[qx],) for aveprecs in cfgx2_aveprecs])
            print('-------')
            print(qx2_lbl[qx])
            print(' aveprecs = %s ' % aveprecs)
    print_row_ave_precision()

    #------------

    @utool.argv_flag_dec
    def print_hardcase():
        print('===')
        print('--- hard new_hardtup_list (w.r.t these configs): %s' % testnameid)
        print('\n'.join(map(repr, new_hardtup_list)))
        print('There are %d hard cases ' % len(new_hardtup_list))
        aid_list = [aid_notes[0] for aid_notes in new_hardtup_list]
        name_list = ibs.get_annot_names(aid_list)
        name_set = set(name_list)
        print(sorted(aid_list))
        print('Names: %r' % (name_set,))
        print('--- /Print Hardcase ---')
    print_hardcase()

    @utool.argv_flag_dec_true
    def echo_hardcase():
        print('====')
        print('--- hardcase commandline: %s' % testnameid)
        print('--index ' + (' '.join(map(str, new_hard_qx_list))))
        #print('--take new_hard_qx_list')
        #hardaids_str = ' '.join(map(str, ['    ', '--qaid'] + new_qaids))
        hardaids_str = ' '.join(map(str, ['    ', '--set-aids-as-hard'] + new_qaids))
        print(hardaids_str)
        print('--- /Echo Hardcase ---')
    echo_hardcase()

    #------------

    @utool.argv_flag_dec_true
    def print_colmap():
        print('==================')
        print('[harn] mAP per Config: %s (sorted by mAP)' % testnameid)
        print('==================')
        cfgx2_mAP = np.array([aveprec_list.mean() for aveprec_list in cfgx2_aveprecs])
        sortx = cfgx2_mAP.argsort()
        for cfgx in sortx:
            print('[mAP] cfgx=%r) mAP=%.3f -- %s' % (cfgx, cfgx2_mAP[cfgx], cfgx2_lbl[cfgx]))
        #print('--- /Scores per Config ---')
    print_colmap()
    #------------

    #------------
    # Build Colscore
    X_list = [1, 5]
    # Build a dictionary mapping X (as in #ranks < X) to a list of cfg scores
    nLessX_dict = {int(X): np.zeros(nCfg) for X in X_list}
    for cfgx in range(nCfg):
        ranks = rank_mat[:, cfgx]
        for X in X_list:
            #nLessX_ = sum(np.bitwise_and(ranks < X, ranks >= 0))
            # Ranks less than 0 are invalid
            nLessX_ = sum(np.logical_and(ranks < X, ranks >= 0))
            nLessX_dict[int(X)][cfgx] = nLessX_

    @utool.argv_flag_dec_true
    def print_colscore():
        print('==================')
        print('[harn] Scores per Config: %s' % testnameid)
        print('==================')
        #for cfgx in range(nCfg):
        #    print('[score] %s' % (cfgx2_lbl[cfgx]))
        #    for X in X_list:
        #        nLessX_ = nLessX_dict[int(X)][cfgx]
        #        print('        ' + eh.rankscore_str(X, nLessX_, nQuery))

        print('\n[harn] ... sorted scores')
        for X in X_list:
            print('\n[harn] Sorted #ranks < %r scores' % (X))
            sortx = np.array(nLessX_dict[int(X)]).argsort()
            for cfgx in sortx:
                nLessX_ = nLessX_dict[int(X)][cfgx]
                rankstr = eh.rankscore_str(X, nLessX_, nQuery, withlbl=False)
                print('[score] %s --- %s' % (rankstr, cfgx2_lbl[cfgx]))
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
    for X, cfgx2_nLessX in six.iteritems(nLessX_dict):
        max_LessX = cfgx2_nLessX.max()
        bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
        best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestCFG_X)
        best_rankscore += eh.rankscore_str(X, max_LessX, nQuery)
        best_rankscore_summary += [best_rankscore]
        to_intersect_list += [cfgx2_lbl[bestCFG_X]]

    intersected = to_intersect_list[0] if len(to_intersect_list) > 0 else []
    for ix in range(1, len(to_intersect_list)):
        intersected = np.intersect1d(intersected, to_intersect_list[ix])

    @utool.argv_flag_dec_true
    def print_bestcfg():
        print('==========================')
        print('[harn] Best Configurations: %s' % testnameid)
        print('==========================')
        # print each configs scores less than X=thresh
        for X, cfgx2_nLessX in six.iteritems(nLessX_dict):
            max_LessX = cfgx2_nLessX.max()
            bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
            best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestCFG_X)
            best_rankscore += eh.rankscore_str(X, max_LessX, nQuery)
            cfglbl_list = cfgx2_lbl[bestCFG_X]

            best_rankcfg = eh.format_cfgstr_list(cfglbl_list)
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
        diff_matstr = get_diffmat_str(rank_mat, qaids, nCfg)
        print(diff_matstr)
        print('[harn]-------------')
    print_diffmat()

    #------------
    # Print summary
    print(' --- SUMMARY ---')
    sumstrs = []
    sumstrs.append('')
    sumstrs.append('||===========================')
    sumstrs.append('|| [cfg*] SUMMARY: %s' % testnameid)
    sumstrs.append('||---------------------------')
    sumstrs.append(utool.joins('\n|| ', best_rankscore_summary))
    sumstrs.append('||===========================')
    print('\n' + '\n'.join(sumstrs) + '\n')

    # Draw result figures
    draw_results(ibs, qaids, daids, sel_rows, sel_cols, cfg_list, cfgx2_lbl, new_hard_qx_list)


def draw_results(ibs, qaids, daids, sel_rows, sel_cols, cfg_list, cfgx2_lbl, new_hard_qx_list):
    """
    Draws results from an experiment harness run.
    Rows store different qaids (query annotation ids)
    Cols store different configurations (algorithm parameters)
    """
    print(' --- DRAW RESULTS ---')
    if utool.NOT_QUIET:
        print('remember to inspect with --sel-rows (-r) and --sel-cols (-c) ')
    if len(sel_rows) > 0 and len(sel_cols) == 0:
        sel_cols = list(range(len(cfg_list)))
    if len(sel_cols) > 0 and len(sel_rows) == 0:
        sel_rows = list(range(len(qaids)))
    if utool.get_argflag(('--view-all', '--va')):
        sel_rows = list(range(len(qaids)))
        sel_cols = list(range(len(cfg_list)))
    if utool.get_argflag(('--view-hard', '--vh')):
        sel_rows = new_hard_qx_list
        sel_cols = list(range(len(cfg_list)))
    if utool.get_argflag(('--view-easy', '--vz')):
        sel_rows = np.setdiff1d(np.arange(len(qaids)), new_hard_qx_list)
        sel_cols = list(range(len(cfg_list)))

    # It is very inefficient to turn off caching when view_all is true
    if not mc4.USE_CACHE:
        print('WARNING: view_all specified with USE_CACHE == False')
        print('WARNING: we will try to turn cache on when reloading results')
        #mc4.USE_CACHE = True

    sel_cols = list(sel_cols)
    sel_rows = list(sel_rows)
    total = len(sel_cols) * len(sel_rows)
    rciter = list(itertools.product(sel_rows, sel_cols))

    skip_list = []
    cp_src_list = []
    cp_dst_list = []

    def append_copy_task(fpath_orig):
        """ helper which copies a summary figure to root dir """
        fname_orig, ext = splitext(basename(fpath_orig))
        outdir = dirname(fpath_orig)
        fdir_clean, cfgdir = split(outdir)
        #aug = cfgdir[0:min(len(cfgdir), 10)]
        aug = cfgdir
        fname_fmt = '{aug}_{fname_orig}{ext}'
        fmt_dict = {'aug': aug, 'fname_orig': fname_orig, 'ext': ext}
        fname_clean = utool.long_fname_format(fname_fmt, fmt_dict, ['fname_orig'], max_len=128)
        fdst_clean = join(fdir_clean, fname_clean)
        cp_src_list.append(fpath_orig)
        cp_dst_list.append(fdst_clean)

    def load_qres(ibs, qaid, daids, query_cfg):
        # Load / Execute the query w/ correct config
        ibs.set_query_cfg(query_cfg)
        # Force program to use cache here
        qres = ibs._query_chips([qaid], daids,
                                use_cache=True,
                                use_bigcache=False)[qaid]
        return qres

    #DELETE              = False
    USE_FIGCACHE = False
    DUMP_EXTRA   = utool.get_argflag('--dump-extra')
    DUMP_QANNOT         = DUMP_EXTRA
    DUMP_QANNOT_DUMP_GT = DUMP_EXTRA
    DUMP_TOP_CONTEXT    = DUMP_EXTRA

    figdir = join(ibs.get_fig_dir(), 'query_analysis')
    utool.ensuredir(ibs.get_fig_dir())
    utool.ensuredir(figdir)

    #utool.view_directory(figdir, verbose=True)

    VIEW_FIG_DIR = utool.get_argflag(('--view-fig-dir', '--vf'))
    if VIEW_FIG_DIR:
        utool.view_directory(figdir, verbose=True)

    #if DELETE:
    #    utool.delete(figdir)

    # Save DEFAULT=True
    def _show_chip(aid, prefix, rank=None, in_image=False, seen=set([]), **dumpkw):
        print('[PRINT_RESULTS] show_chip(aid=%r)' % (aid,))
        from ibeis import viz
        if aid in seen:
            return
        viz.show_chip(ibs, aid, in_image=in_image)
        if rank is not None:
            prefix += 'rank%d_' % rank
        df2.set_figtitle(prefix + ibs.annotstr(aid))
        seen.add(aid)
        if utool.VERBOSE:
            print('[expt] dumping fig to %s' % figdir)
        fpath_clean = ph.dump_figure(figdir, **dumpkw)
        return fpath_clean

    chunksize = 10
    # <FOR RCITER_CHUNK>
    for rciter_chunk in ut.ichunks(enumerate(rciter), chunksize):
        # First load a chunk of query results
        qres_list = []
        # <FOR RCITER>
        for count, rctup in rciter_chunk:
            if (count in skip_list) or (SKIP_TO and count < SKIP_TO):
                qres_list.append(None)
                continue
            else:
                # Get row and column index
                (r, c) = rctup
                qaid      = qaids[r]
                query_cfg = cfg_list[c]
                qres = load_qres(ibs, qaid, daids, query_cfg)
                qres_list.append(qres)
        # Iterate over chunks a second time, but
        # with loaded query results
        for item, qres in zip(rciter_chunk, qres_list):
            count, rctup = item
            if (count in skip_list) or (SKIP_TO and count < SKIP_TO):
                continue
            (r, c) = rctup
            # Get row and column index
            qaid      = qaids[r]
            query_cfg = cfg_list[c]
            query_lbl = cfgx2_lbl[c]
            print(utool.unindent('''
            __________________________________
            --- VIEW %d / %d --- (r=%r, c=%r)
            ----------------------------------
            ''')  % (count + 1, total, r, c))
            #qres = load_qres(ibs, qaid, daids, query_cfg)
            qres_cfg = qres.get_fname(ext='')
            subdir = qres_cfg
            # Draw Result
            dumpkw = {
                'subdir'    : subdir,
                'quality'   : utool.get_argflag('--quality'),
                'overwrite' : True,
                'verbose'   : 0,
            }
            show_kwargs = {
                'N': 3,
                'ori': True,
                'ell_alpha': .9,
            }

            if not SAVE_FIGURES:
                continue

            if USE_FIGCACHE and utool.checkpath(join(figdir, subdir)):
                continue

            print('[harn] showing analysis')

            # Show Figure
            # try to shorten query labels a bit
            query_lbl = query_lbl.replace(' ', '').replace('\'', '')
            qres.show(ibs, 'analysis', figtitle=query_lbl, **show_kwargs)

            # Adjust subplots
            df2.adjust_subplots_safe()
            fpath_orig = ph.dump_figure(figdir, **dumpkw)
            append_copy_task(fpath_orig)

            print('[harn] showing other plots')

            if DUMP_QANNOT:
                _show_chip(qres.qaid, 'QUERY_', **dumpkw)
                _show_chip(qres.qaid, 'QUERY_CXT_', in_image=True, **dumpkw)

            if DUMP_QANNOT_DUMP_GT:
                gtaids = ibs.get_annot_groundtruth(qres.qaid)
                for aid in gtaids:
                    rank = qres.get_aid_ranks(aid)
                    _show_chip(aid, 'GT_CXT_', rank=rank, in_image=True, **dumpkw)

            if DUMP_TOP_CONTEXT:
                topids = qres.get_top_aids(num=3)
                for aid in topids:
                    rank = qres.get_aid_ranks(aid)
                    _show_chip(aid, 'TOP_CXT_', rank=rank, in_image=True, **dumpkw)

            if utool.get_argflag('--show'):
                print('[PRINT_RESULTS] df2.present()')
                df2.present()
    # </FOR RCITER>

    # Copy summary images to query_analysis folder
    print('[PRINT_RESULTS] copying summaries')
    for src, dst in zip(cp_src_list, cp_dst_list):
        utool.copy(src, dst)

    if utool.NOT_QUIET:
        print('[PRINT_RESULTS] EXIT EXPERIMENT HARNESS')
