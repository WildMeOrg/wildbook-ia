# -*- coding: utf-8 -*-
"""
displays results from harness

TODO: save a testres variable so reloading and regenration becomes easier.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import six
import utool as ut
#from ibeis.other import ibsfuncs
#from ibeis.expt import experiment_drawing
from six.moves import map, range, input  # NOQA
import vtool as vt
print, rrr, profile = ut.inject2(__name__, '[expt_printres]')


def get_diffranks(rank_mat, qaids):
    """ Find rows which scored differently over the various configs
    FIXME: duplicated
    """
    isdiff_flags = [not np.all(row == row[0]) for row in rank_mat]
    diff_aids    = ut.compress(qaids, isdiff_flags)
    diff_rank    = rank_mat.compress(isdiff_flags, axis=0)
    diff_qxs     = np.where(isdiff_flags)[0]
    return diff_aids, diff_rank, diff_qxs


def get_diffmat_str(rank_mat, qaids, nConfig):
    from itertools import chain
    diff_aids, diff_rank, diff_qxs = get_diffranks(rank_mat, qaids)
    # Find columns that ore strictly better than other columns
    #def find_strictly_better_columns(diff_rank):
    #    colmat = diff_rank.T
    #    pairwise_betterness_ranks = np.array([np.sum(col <= colmat, axis=1) / len(col) for col in colmat], dtype=np.float).T
    diff_mat = np.vstack((diff_aids, diff_rank.T)).T
    col_lbls = list(chain(['qaid'], map(lambda x: 'cfg%d_rank' % x, range(nConfig))))
    col_type  = list(chain([int], [int] * nConfig))
    header = 'diffmat'
    diff_matstr = ut.numpy_to_csv(diff_mat, col_lbls, header, col_type)
    return diff_matstr


def print_latexsum(ibs, testres, verbose=True):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        testres (?):

    CommandLine:
        python -m ibeis.expt.experiment_printres --exec-print_latexsum
        python -m ibeis.scripts.gen_cand_expts --exec-gen_script

        python -m ibeis --tf print_latexsum -t candidacy --db PZ_Master0 -a controlled --rank-lt-list=1,5,10,100
        python -m ibeis --tf print_latexsum -t candidacy --db PZ_MTEST -a controlled --rank-lt-list=1,5,10,100

    Example:
        >>> # SCRIPT
        >>> from ibeis.expt.experiment_printres import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts()
        >>> tabular_str2 = print_latexsum(ibs, testres)
    """
    print('==========================')
    print('[harn] LaTeX: %s' % testres.testnameid)
    print('==========================')
    # Create configuration latex table
    X_LIST = testres.get_X_LIST()
    criteria_lbls = [r'#ranks $\leq$ %d' % X for X in X_LIST]
    dbname = ibs.get_dbname()
    cfg_score_title = dbname + ' rank scores'
    nLessX_dict = testres.get_nLessX_dict()
    cfgscores = np.array([nLessX_dict[int(X)] for X in X_LIST]).T

    # For mat row labels
    row_lbls = testres.get_short_cfglbls()
    # Order cdf list by rank0
    row_lbls = ut.sortedby(row_lbls, cfgscores.T[0], reverse=True)
    cfgscores = np.array(ut.sortedby(cfgscores.tolist(), cfgscores.T[0], reverse=True))

    cmdaug = testres.get_title_aug()
    #if testres.common_acfg is not None:
    #    cfgname = testres.common_acfg['common']['_cfgname']
    #    cmdaug += '_' + cfgname
    #if hasattr(testres, 'common_cfgdict'):
    #    cmdaug += '_' + (testres.common_cfgdict['_cfgname'])
    #    cfg_score_title += ' ' + cmdaug
    #else:
    #    #ut.embed()
    #    assert False, 'need common_cfgdict'

    tabular_kwargs = dict(
        title=cfg_score_title,
        out_of=testres.nQuery,
        bold_best=True,
        flip=False,
        SHORTEN_ROW_LBLS=False
    )
    col_lbls = criteria_lbls
    tabular_str = ut.util_latex.make_score_tabular(
        row_lbls, col_lbls, cfgscores, **tabular_kwargs)
    #latex_formater.render(tabular_str)
    cmdname = ut.latex_sanitize_command_name('Expmt' + ibs.get_dbname() + '_' + cmdaug + 'Table')
    tabular_str2 = ut.latex_newcommand(cmdname, tabular_str)
    print(tabular_str2)
    return tabular_str2


@profile
def print_results(ibs, testres):
    """
    Prints results from an experiment harness run.
    Rows store different qaids (query annotation ids)
    Cols store different configurations (algorithm parameters)

    Args:
        ibs (IBEISController):  ibeis controller object
        testres (test_result.TestResult):

    CommandLine:
        python dev.py -e print --db PZ_MTEST -a default:dpername=1,qpername=[1,2]  -t default:fg_on=False

        python dev.py -e print -t best --db seals2 --allgt --vz
        python dev.py -e print --db PZ_MTEST --allgt -t custom --print-confusion-stats
        python dev.py -e print --db PZ_MTEST --allgt --noqcache --index 0:10:2 -t custom:rrvsone_on=True --print-confusion-stats
        python dev.py -e print --db PZ_MTEST --allgt --noqcache --qaid4 -t custom:rrvsone_on=True --print-confusion-stats
        python -m ibeis --tf print_results -t default --db PZ_MTEST -a ctrl
        python -m ibeis --tf print_results -t default --db PZ_MTEST -a ctrl
        python -m ibeis --tf print_results --db PZ_MTEST -a default -t default:lnbnn_on=True default:lnbnn_on=False,bar_l2_on=True default:lnbnn_on=False,normonly_on=True

    CommandLine:
        python -m ibeis.expt.experiment_printres --test-print_results
        utprof.py -m ibeis.expt.experiment_printres --test-print_results

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.expt.experiment_printres import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts(
        >>>     'PZ_MTEST', a='default:dpername=1,qpername=[1,2]', t='default:fg_on=False')
        >>> result = print_results(ibs, testres)
        >>> print(result)
    """

    (cfg_list, cfgx2_cfgresinfo, testnameid, cfgx2_lbl, cfgx2_qreq_) = ut.dict_take(
        testres.__dict__, ['cfg_list', 'cfgx2_cfgresinfo', 'testnameid', 'cfgx2_lbl', 'cfgx2_qreq_'])

    # cfgx2_cfgresinfo is a list of dicts of lists
    # Parse result info out of the lists
    cfgx2_nextbestranks  = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_next_bestranks')
    cfgx2_gt_rawscores   = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_gt_raw_score')
    cfgx2_gf_rawscores   = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_gf_raw_score')
    #cfgx2_aveprecs       = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_avepercision')

    cfgx2_scorediffs     = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_scorediff')
    #cfgx2_gt_raw_score   = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_gt_raw_score')

    column_lbls = [ut.remove_chars(ut.remove_vowels(lbl), [' ', ','])
                   for lbl in cfgx2_lbl]

    scorediffs_mat     = np.array(ut.replace_nones(cfgx2_scorediffs, np.nan))

    print(' --- PRINT RESULTS ---')
    print(' use --rank-lt-list=1,5 to specify X_LIST')
    if True:
        # Num of ranks less than to score
        X_LIST = testres.get_X_LIST()
        #X_LIST = [1, 5]

        #nConfig = len(cfg_list)
        #nQuery = len(testres.qaids)
        cfgx2_nQuery = list(map(len, testres.cfgx2_qaids))
        #cfgx2_qx2_ranks = testres.get_infoprop_list('qx2_bestranks')
        #--------------------

        # A positive scorediff indicates the groundtruth was better than the
        # groundfalse scores
        istrue_list  = [scorediff > 0 for scorediff in scorediffs_mat]
        isfalse_list = [~istrue for istrue in istrue_list]

        #------------
        # Build Colscore
        nLessX_dict = testres.get_nLessX_dict()

        #------------
        best_rankscore_summary = []
        #to_intersect_list = []
        # print each configs scores less than X=thresh
        for X, cfgx2_nLessX in six.iteritems(nLessX_dict):
            max_nLessX = cfgx2_nLessX.max()
            bestX_cfgx_list = np.where(cfgx2_nLessX == max_nLessX)[0]
            best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestX_cfgx_list)
            # FIXME
            best_rankscore += rankscore_str(X, max_nLessX, cfgx2_nQuery[bestX_cfgx_list[0]])
            best_rankscore_summary += [best_rankscore]
            #to_intersect_list.append(ut.take(cfgx2_lbl, max_nLessX))

        #intersected = to_intersect_list[0] if len(to_intersect_list) > 0 else []
        #for ix in range(1, len(to_intersect_list)):
        #    intersected = np.intersect1d(intersected, to_intersect_list[ix])

    #if False:
    #    #gt_raw_score_mat = np.vstack(cfgx2_gt_raw_score).T

    #    #rank_mat = testres.get_rank_mat()

    #    #------------
    #    # Build row lbls
    #    if False:
    #        qx2_lbl = np.array([
    #            'qx=%d) q%s ' % (qx, ibsfuncs.aidstr(testres.qaids[qx], ibs=ibs, notes=True))
    #            for qx in range(nQuery)])

    #    #------------
    #    # Build Colscore and hard cases
    #    if False:
    #        qx2_min_rank = []
    #        qx2_argmin_rank = []
    #        new_hard_qaids = []
    #        new_hardtup_list = []

    #        for qx in range(nQuery):
    #            ranks = rank_mat[qx]
    #            valid_ranks = ranks[ranks >= 0]
    #            min_rank = ranks.min() if len(valid_ranks) > 0 else -3
    #            bestCFG_X = np.where(ranks == min_rank)[0]
    #            qx2_min_rank.append(min_rank)
    #            # Find the best rank over all configurations
    #            qx2_argmin_rank.append(bestCFG_X)

    #@ut.memoize
    #def get_new_hard_qx_list(testres):
    #    """ Mark any query as hard if it didnt get everything correct """
    #    rank_mat = testres.get_rank_mat()
    #    is_new_hard_list = rank_mat.max(axis=1) > 0
    #    new_hard_qx_list = np.where(is_new_hard_list)[0]
    #    return new_hard_qx_list

    #        new_hard_qx_list = testres.get_new_hard_qx_list()

    #        for qx in new_hard_qx_list:
    #            # New list is in aid format instead of cx format
    #            # because you should be copying and pasting it
    #            notes = ' ranks = ' + str(rank_mat[qx])
    #            qaid = testres.qaids[qx]
    #            name = ibs.get_annot_names(qaid)
    #            new_hardtup_list += [(qaid, name + " - " + notes)]
    #            new_hard_qaids += [qaid]

    @ut.argv_flag_dec
    def intersect_hack():
        failed = testres.rank_mat > 0
        colx2_failed = [np.nonzero(failed_col)[0] for failed_col in failed.T]
        #failed_col2_only = np.setdiff1d(colx2_failed[1], colx2_failed[0])
        #failed_col2_only_aids = ut.take(testres.qaids, failed_col2_only)
        failed_col1_only = np.setdiff1d(colx2_failed[0], colx2_failed[1])
        failed_col1_only_aids = ut.take(testres.qaids, failed_col1_only)
        gt_aids1 = ibs.get_annot_groundtruth(failed_col1_only_aids, daid_list=testres.cfgx2_qreq_[0].daids)
        gt_aids2 = ibs.get_annot_groundtruth(failed_col1_only_aids, daid_list=testres.cfgx2_qreq_[1].daids)

        qaids_expt = failed_col1_only_aids
        gt_avl_aids1 = ut.flatten(gt_aids1)
        gt_avl_aids2 = list(set(ut.flatten(gt_aids2)).difference(gt_avl_aids1))

        ibs.print_annotconfig_stats(qaids_expt, gt_avl_aids1)
        ibs.print_annotconfig_stats(qaids_expt, gt_avl_aids2)
        #jsontext = ut.to_json({
        #    'qaids': list(qaids_expt),
        #    'dinclude_aids1': list(gt_aids_expt1),
        #    'dinclude_aids2': list(gt_aids_expt2),
        #})
        #annotation_configs.varysize_pzm
        #from ibeis.expt import annotation_configs

        acfg = testres.acfg_list[0]
        import copy
        acfg1 = copy.deepcopy(acfg)
        acfg2 = copy.deepcopy(acfg)
        acfg1['qcfg']['min_pername'] = None
        acfg2['qcfg']['min_pername'] = None
        acfg1['dcfg']['min_pername'] = None
        acfg2['dcfg']['min_gt_per_name'] = None

        acfg1['qcfg']['default_aids'] = qaids_expt
        acfg1['dcfg']['gt_avl_aids'] = gt_avl_aids1
        acfg2['qcfg']['default_aids'] = qaids_expt
        acfg2['dcfg']['gt_avl_aids'] = gt_avl_aids2

        from ibeis.init import filter_annots
        from ibeis.expt import experiment_helpers

        annots1 = filter_annots.expand_acfgs(ibs, acfg1, verbose=True)
        annots2 = filter_annots.expand_acfgs(ibs, acfg2, verbose=True)

        acfg_name_list = dict(  # NOQA
            acfg_list=[acfg1, acfg2],
            expanded_aids_list=[annots1, annots2],
        )
        test_cfg_name_list = ['candidacy_k']
        cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(test_cfg_name_list, ibs=ibs)

        t1, t2 = testres_list  # NOQA
    #ut.embed()
    #intersect_hack()

    #@ut.argv_flag_dec
    #def print_rowlbl():
    #    print('=====================')
    #    print('[harn] Row/Query Labels: %s' % testnameid)
    #    print('=====================')
    #    print('[harn] queries:\n%s' % '\n'.join(qx2_lbl))
    #print_rowlbl()
    #------------

    @ut.argv_flag_dec
    def print_collbl():
        print('=====================')
        print('[harn] Col/Config Labels: %s' % testnameid)
        print('=====================')
        enum_cfgx2_lbl = ['%2d) %s' % (count, cfglbl)
                            for count, cfglbl in enumerate(cfgx2_lbl)]
        print('[harn] cfglbl:\n%s' % '\n'.join(enum_cfgx2_lbl))
    print_collbl()

    #------------

    @ut.argv_flag_dec
    def print_cfgstr():
        print('=====================')
        print('[harn] Config Strings: %s' % testnameid)
        print('=====================')
        cfgstr_list = [query_cfg.get_cfgstr() for query_cfg in cfg_list]
        enum_cfgstr_list = ['%2d) %s' % (count, cfgstr)
                            for count, cfgstr in enumerate(cfgstr_list)]
        print('\n[harn] cfgstr:\n%s' % '\n'.join(enum_cfgstr_list))
    print_cfgstr()

    #------------

    #@ut.argv_flag_dec
    #def print_rowscore():
    #    print('=======================')
    #    print('[harn] Scores per Query: %s' % testnameid)
    #    print('=======================')
    #    for qx in range(nQuery):
    #        bestCFG_X = qx2_argmin_rank[qx]
    #        min_rank = qx2_min_rank[qx]
    #        minimizing_cfg_str = ut.indentjoin(cfgx2_lbl[bestCFG_X], '\n  * ')
    #        #minimizing_cfg_str = str(bestCFG_X)

    #        print('-------')
    #        print(qx2_lbl[qx])
    #        print(' best_rank = %d ' % min_rank)
    #        if len(cfgx2_lbl) != 1:
    #            print(' minimizing_cfg_x\'s = %s ' % minimizing_cfg_str)
    #print_rowscore()

    #------------

    #@ut.argv_flag_dec
    #def print_row_ave_precision():
    #    print('=======================')
    #    print('[harn] Scores per Query: %s' % testnameid)
    #    print('=======================')
    #    for qx in range(nQuery):
    #        aveprecs = ', '.join(['%.2f' % (aveprecs[qx],) for aveprecs in cfgx2_aveprecs])
    #        print('-------')
    #        print(qx2_lbl[qx])
    #        print(' aveprecs = %s ' % aveprecs)
    #print_row_ave_precision()

    ##------------

    #@ut.argv_flag_dec
    #def print_hardcase():
    #    print('--- hard new_hardtup_list (w.r.t these configs): %s' % testnameid)
    #    print('\n'.join(map(repr, new_hardtup_list)))
    #    print('There are %d hard cases ' % len(new_hardtup_list))
    #    aid_list = [aid_notes[0] for aid_notes in new_hardtup_list]
    #    name_list = ibs.get_annot_names(aid_list)
    #    name_set = set(name_list)
    #    print(sorted(aid_list))
    #    print('Names: %r' % (name_set,))
    #print_hardcase()
    #default=not ut.get_argflag('--allhard'))

    #------------

    #@ut.argv_flag_dec
    #def echo_hardcase():
    #    print('--- hardcase commandline: %s' % testnameid)
    #    # Show index for current query where hardids reside
    #    #print('--index ' + (' '.join(map(str, new_hard_qx_list))))
    #    #print('--take new_hard_qx_list')
    #    #hardaids_str = ' '.join(map(str, ['    ', '--qaid'] + new_hard_qaids))
    #    hardaids_str = ' '.join(map(str, ['    ', '--set-aids-as-hard'] + new_hard_qaids))
    #    print(hardaids_str)
    ##echo_hardcase(default=not ut.get_argflag('--allhard'))
    #echo_hardcase()

    #@ut.argv_flag_dec
    #def print_bestcfg():
    #    print('==========================')
    #    print('[harn] Best Configurations: %s' % testnameid)
    #    print('==========================')
    #    # print each configs scores less than X=thresh
    #    for X, cfgx2_nLessX in six.iteritems(nLessX_dict):
    #        max_LessX = cfgx2_nLessX.max()
    #        bestCFG_X = np.where(cfgx2_nLessX == max_LessX)[0]
    #        best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestCFG_X)
    #        best_rankscore += rankscore_str(X, max_LessX, nQuery)
    #        cfglbl_list = cfgx2_lbl[bestCFG_X]

    #        best_rankcfg = format_cfgstr_list(cfglbl_list)
    #        #indent('\n'.join(cfgstr_list), '    ')
    #        print(best_rankscore)
    #        print(best_rankcfg)
    #    print('[cfg*]  %d cfg(s) are the best of %d total cfgs' % (len(intersected), nConfig))
    #    print(format_cfgstr_list(intersected))
    #print_bestcfg()

    #------------

    #@ut.argv_flag_dec
    #def print_gtscore():
    #    # Prints best ranks
    #    print('gtscore_mat: %s' % testnameid)
    #    print(' nRows=%r, nCols=%r' % (nQuery, nConfig))
    #    header = (' labled rank matrix: rows=queries, cols=cfgs:')
    #    print('\n'.join(cfgx2_lbl))
    #    column_list = gt_raw_score_mat.T
    #    print(ut.make_csv_table(column_list, row_lbls=testres.qaids,
    #                            column_lbls=column_lbls, header=header,
    #                            transpose=False,
    #                            use_lbl_width=len(cfgx2_lbl) < 5))
    #print_gtscore()

    #------------

    #@ut.argv_flag_dec
    #def print_best_rankmat():
    #    # Prints best ranks
    #    print('-------------')
    #    print('RankMat: %s' % testnameid)
    #    print(' nRows=%r, nCols=%r' % (nQuery, nConfig))
    #    header = (' labled rank matrix: rows=queries, cols=cfgs:')
    #    print('\n'.join(cfgx2_lbl))
    #    column_list = rank_mat.T
    #    print(ut.make_csv_table(column_list, row_lbls=testres.qaids,
    #                            column_lbls=column_lbls, header=header,
    #                            transpose=False,
    #                            use_lbl_width=len(cfgx2_lbl) < 5))
    #print_best_rankmat()

    #@ut.argv_flag_dec
    #def print_diffmat():
    #    # score differences over configs
    #    print('-------------')
    #    print('Diffmat: %s' % testnameid)
    #    diff_matstr = get_diffmat_str(rank_mat, testres.qaids, nConfig)
    #    print(diff_matstr)
    #print_diffmat()

    #@ut.argv_flag_dec
    #def print_rankhist_time():
    #    print('A rank histogram is a dictionary. '
    #          'The keys denote the range of the ranks that the values fall in')
    #    # TODO: rectify this code with other hist code

    #    config_gt_aids = ut.get_list_column(testres.cfgx2_cfgresinfo, 'qx2_gt_aid')
    #    config_rand_bin_qxs = testres.get_rank_histogram_qx_binxs()

    #    _iter = enumerate(zip(rank_mat.T, agg_hist_dict, config_gt_aids, config_rand_bin_qxs))
    #    for cfgx, (ranks, agg_hist_dict, qx2_gt_aid, config_binxs) in _iter:
    #        #full_cfgstr = testres.cfgx2_qreq_[cfgx].get_full_cfgstr()
    #        #ut.print_dict(ut.dict_hist(ranks), 'rank histogram', sorted_=True)
    #        # find the qxs that belong to each bin
    #        aid_list1 = testres.qaids
    #        aid_list2 = qx2_gt_aid
    #        ibs.assert_valid_aids(aid_list1)
    #        ibs.assert_valid_aids(aid_list2)
    #        timedelta_list = ibs.get_annot_pair_timdelta(aid_list1, aid_list2)
    #        #timedelta_str_list = [ut.get_posix_timedelta_str2(delta)
    #        #                      for delta in timedelta_list]

    #        bin_edges = testres.get_rank_histogram_bin_edges()
    #        timedelta_groups = ut.dict_take(ut.group_items(timedelta_list, config_binxs), np.arange(len(bin_edges)), [])

    #        timedelta_stats = [ut.get_stats(deltas, use_nan=True, datacast=ut.get_posix_timedelta_str2) for deltas in timedelta_groups]
    #        print('Time statistics for each rank range:')
    #        print(ut.dict_str(dict(zip(bin_edges, timedelta_stats)), sorted_=True))
    #print_rankhist_time()

    #@ut.argv_flag_dec
    #def print_rankhist():
    #    print('A rank histogram is a dictionary. '
    #          'The keys denote the range of the ranks that the values fall in')
    #    # TODO: rectify this code with other hist code

    #    config_gt_aids = ut.get_list_column(testres.cfgx2_cfgresinfo, 'qx2_gt_aid')
    #    config_rand_bin_qxs = testres.get_rank_histogram_qx_binxs()

    #    _iter = enumerate(zip(rank_mat.T, agg_hist_dict, config_gt_aids, config_rand_bin_qxs))
    #    for cfgx, (ranks, agg_hist_dict, qx2_gt_aid, config_binxs) in _iter:
    #        print('Frequency of rank ranges:')
    #        ut.print_dict(agg_hist_dict, 'agg rank histogram', sorted_=True)
    #print_rankhist()

    #------------
    # Print summary
    #print(' --- SUMMARY ---')

    #------------

    #@ut.argv_flag_dec
    #def print_colmap():
    #    print('==================')
    #    print('[harn] mAP per Config: %s (sorted by mAP)' % testnameid)
    #    print('==================')
    #    cfgx2_mAP = np.array([aveprec_list.mean() for aveprec_list in cfgx2_aveprecs])
    #    sortx = cfgx2_mAP.argsort()
    #    for cfgx in sortx:
    #        print('[mAP] cfgx=%r) mAP=%.3f -- %s' % (cfgx, cfgx2_mAP[cfgx], cfgx2_lbl[cfgx]))
    #    #print('L___ Scores per Config ___')
    #print_colmap()
    #------------

    @ut.argv_flag_dec_true
    def print_colscore():
        print('==================')
        print('[harn] Scores per Config: %s' % testnameid)
        print('==================')
        #for cfgx in range(nConfig):
        #    print('[score] %s' % (cfgx2_lbl[cfgx]))
        #    for X in X_LIST:
        #        nLessX_ = nLessX_dict[int(X)][cfgx]
        #        print('        ' + rankscore_str(X, nLessX_, nQuery))
        print('\n[harn] ... sorted scores')
        for X in X_LIST:
            print('\n[harn] Sorted #ranks < %r scores' % (X))
            sortx = np.array(nLessX_dict[int(X)]).argsort()
            #frac_list = (nLessX_dict[int(X)] / cfgx2_nQuery)[:, None]
            #print('cfgx2_nQuery = %r' % (cfgx2_nQuery,))
            #print('frac_list = %r' % (frac_list,))
            #print('Pairwise Difference: ' + str(ut.safe_pdist(frac_list, metric=ut.absdiff)))
            for cfgx in sortx:
                nLessX_ = nLessX_dict[int(X)][cfgx]
                rankstr = rankscore_str(X, nLessX_, cfgx2_nQuery[cfgx], withlbl=False)
                print('[score] %s --- %s' % (rankstr, cfgx2_lbl[cfgx]))
    print_colscore()

    #------------

    ut.argv_flag_dec(print_latexsum)(ibs, testres)

    @ut.argv_flag_dec
    def print_next_rankmat():
        # Prints nextbest ranks
        print('-------------')
        print('NextRankMat: %s' % testnameid)
        header = (' top false rank matrix: rows=queries, cols=cfgs:')
        print('\n'.join(cfgx2_lbl))
        column_list = cfgx2_nextbestranks
        print(ut.make_csv_table(column_list, row_lbls=testres.qaids,
                                column_lbls=column_lbls, header=header,
                                transpose=False,
                                use_lbl_width=len(cfgx2_lbl) < 5))
    print_next_rankmat()

    #------------

    @ut.argv_flag_dec
    def print_scorediff_mat():
        # Prints nextbest ranks
        print('-------------')
        print('ScoreDiffMat: %s' % testnameid)
        header = (' score difference between top true and top false: rows=queries, cols=cfgs:')
        print('\n'.join(cfgx2_lbl))
        column_list = cfgx2_scorediffs
        column_type = [float] * len(column_list)
        print(ut.make_csv_table(column_list, row_lbls=testres.qaids,
                                column_lbls=column_lbls,
                                column_type=column_type,
                                header=header,
                                transpose=False,
                                use_lbl_width=len(cfgx2_lbl) < 5))
    print_scorediff_mat(alias_flags=['--sdm'])

    #------------
    def jagged_stats_info(arr_, lbl, col_lbls):
        arr = ut.recursive_replace(arr_, np.inf, np.nan)
        # Treat infinite as nan
        stat_dict = ut.get_jagged_stats(arr, use_nan=True, use_sum=True)
        sel_stat_dict, sel_indices = ut.find_interesting_stats(stat_dict, col_lbls)
        sel_col_lbls = ut.take(col_lbls, sel_indices)
        statstr_kw   = dict(precision=3, newlines=True, lbl=lbl, align=True)
        stat_str     = ut.get_stats_str(stat_dict=stat_dict, **statstr_kw)
        sel_stat_str = ut.get_stats_str(stat_dict=sel_stat_dict, **statstr_kw)
        sel_stat_str = 'sel_col_lbls = %s' % (ut.list_str(sel_col_lbls),) + '\n' + sel_stat_str
        return stat_str, sel_stat_str

    @ut.argv_flag_dec
    def print_confusion_stats():
        """
        CommandLine:
            python dev.py --allgt --print-scorediff-mat-stats --print-confusion-stats -t rrvsone_grid
        """
        # Prints nextbest ranks
        print('-------------')
        print('ScoreDiffMatStats: %s' % testnameid)
        print('column_lbls = %r' % (column_lbls,))

        #cfgx2_gt_rawscores  = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_gt_raw_score')
        #cfgx2_gf_rawscores  = ut.get_list_column(cfgx2_cfgresinfo, 'qx2_gf_raw_score')

        gt_rawscores_mat = ut.replace_nones(cfgx2_gt_rawscores, np.nan)
        gf_rawscores_mat = ut.replace_nones(cfgx2_gf_rawscores, np.nan)

        tp_rawscores = vt.zipcompress(gt_rawscores_mat, istrue_list)
        fp_rawscores = vt.zipcompress(gt_rawscores_mat, isfalse_list)
        tn_rawscores = vt.zipcompress(gf_rawscores_mat, istrue_list)
        fn_rawscores = vt.zipcompress(gf_rawscores_mat, isfalse_list)

        tp_rawscores_str, tp_rawscore_statstr = jagged_stats_info(tp_rawscores, 'tp_rawscores', cfgx2_lbl)
        fp_rawscores_str, fp_rawscore_statstr = jagged_stats_info(fp_rawscores, 'fp_rawscores', cfgx2_lbl)
        tn_rawscores_str, tn_rawscore_statstr = jagged_stats_info(tn_rawscores, 'tn_rawscores', cfgx2_lbl)
        fn_rawscores_str, fn_rawscore_statstr = jagged_stats_info(fn_rawscores, 'fn_rawscores', cfgx2_lbl)

        #print(tp_rawscores_str)
        #print(fp_rawscores_str)
        #print(tn_rawscores_str)
        #print(fn_rawscores_str)

        print(tp_rawscore_statstr)
        print(fp_rawscore_statstr)
        print(tn_rawscore_statstr)
        print(fn_rawscore_statstr)

    print_confusion_stats(alias_flags=['--cs'])

    ut.argv_flag_dec_true(testres.print_percent_identification_success)()

    sumstrs = []
    sumstrs.append('')
    sumstrs.append('||===========================')
    sumstrs.append('|| [cfg*] SUMMARY: %s' % testnameid)
    sumstrs.append('||---------------------------')
    sumstrs.append(ut.joins('\n|| ', best_rankscore_summary))
    sumstrs.append('||===========================')
    summary_str = '\n' + '\n'.join(sumstrs) + '\n'
    #print(summary_str)
    ut.colorprint(summary_str, 'blue')

    print('To enable all printouts add --print-all to the commandline')


def rankscore_str(thresh, nLess, total, withlbl=True):
    #helper to print rank scores of configs
    percent = 100 * nLess / total
    fmtsf = '%' + str(ut.num2_sigfig(total)) + 'd'
    if withlbl:
        fmtstr = ':#ranks < %d = ' + fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + '/' + str(total) + ')'
        rankscore_str = fmtstr % (thresh, nLess, total, percent, (total - nLess))
    else:
        fmtstr = fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + '/' + str(total) + ')'
        rankscore_str = fmtstr % (nLess, total, percent, (total - nLess))
    return rankscore_str


#def wrap_cfgstr(cfgstr):
#    # REGEX to locate _XXXX(
#    import re
#    cfg_regex = r'_[A-Z][A-Z]*\('
#    cfgstrmarker_list = re.findall(cfg_regex, cfgstr)
#    cfgstrconfig_list = re.split(cfg_regex, cfgstr)
#    args = [cfgstrconfig_list, cfgstrmarker_list]
#    interleave_iter = ut.interleave(args)
#    new_cfgstr_list = []
#    total_len = 0
#    prefix_str = ''
#    # If unbalanced there is a prefix before a marker
#    if len(cfgstrmarker_list) < len(cfgstrconfig_list):
#        frag = interleave_iter.next()
#        new_cfgstr_list += [frag]
#        total_len = len(frag)
#        prefix_str = ' ' * len(frag)
#    # Iterate through markers and config strings
#    while True:
#        try:
#            marker_str = interleave_iter.next()
#            config_str = interleave_iter.next()
#            frag = marker_str + config_str
#        except StopIteration:
#            break
#        total_len += len(frag)
#        new_cfgstr_list += [frag]
#        # Go to newline if past 80 chars
#        if total_len > 80:
#            total_len = 0
#            new_cfgstr_list += ['\n' + prefix_str]
#    wrapped_cfgstr = ''.join(new_cfgstr_list)
#    return wrapped_cfgstr


#def format_cfgstr_list(cfgstr_list):
#    indented_list = ut.indent_list('    ', cfgstr_list)
#    wrapped_list = list(map(wrap_cfgstr, indented_list))
#    return ut.joins('\n', wrapped_list)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.expt.experiment_printres
        python -m ibeis.expt.experiment_printres --allexamples
        python -m ibeis.expt.experiment_printres --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
