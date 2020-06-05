# -*- coding: utf-8 -*-
"""
displays results from harness

TODO: save a testres variable so reloading and regenration becomes easier.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import six
import utool as ut
from six.moves import map, range

print, rrr, profile = ut.inject2(__name__)


def get_diffranks(rank_mat, qaids):
    """ Find rows which scored differently over the various configs
    FIXME: duplicated
    """
    isdiff_flags = [not np.all(row == row[0]) for row in rank_mat]
    diff_aids = ut.compress(qaids, isdiff_flags)
    diff_rank = rank_mat.compress(isdiff_flags, axis=0)
    diff_qxs = np.where(isdiff_flags)[0]
    return diff_aids, diff_rank, diff_qxs


def get_diffmat_str(rank_mat, qaids, nConfig):
    from itertools import chain

    diff_aids, diff_rank, diff_qxs = get_diffranks(rank_mat, qaids)
    # Find columns that ore strictly better than other columns
    # def find_strictly_better_columns(diff_rank):
    #    colmat = diff_rank.T
    #    pairwise_betterness_ranks = np.array([np.sum(col <= colmat, axis=1) / len(col) for col in colmat], dtype=np.float).T
    diff_mat = np.vstack((diff_aids, diff_rank.T)).T
    col_lbls = list(chain(['qaid'], map(lambda x: 'cfg%d_rank' % x, range(nConfig))))
    col_type = list(chain([int], [int] * nConfig))
    header = 'diffmat'
    diff_matstr = ut.numpy_to_csv(diff_mat, col_lbls, header, col_type)
    return diff_matstr


def print_latexsum(ibs, testres, verbose=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        testres (?):

    CommandLine:
        python -m wbia.expt.experiment_printres --exec-print_latexsum
        python -m wbia.scripts.gen_cand_expts --exec-gen_script

        python -m wbia --tf print_latexsum -t candidacy --db PZ_Master0 -a controlled --rank-lt-list=1,5,10,100
        python -m wbia --tf print_latexsum -t candidacy --db PZ_MTEST -a controlled --rank-lt-list=1,5,10,100

    Example:
        >>> # SCRIPT
        >>> from wbia.expt.experiment_printres import *  # NOQA
        >>> from wbia.init import main_helpers
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
    # if testres.common_acfg is not None:
    #    cfgname = testres.common_acfg['common']['_cfgname']
    #    cmdaug += '_' + cfgname
    # if hasattr(testres, 'common_cfgdict'):
    #    cmdaug += '_' + (testres.common_cfgdict['_cfgname'])
    #    cfg_score_title += ' ' + cmdaug

    tabular_kwargs = dict(
        title=cfg_score_title,
        out_of=testres.nQuery,
        bold_best=True,
        flip=False,
        SHORTEN_ROW_LBLS=False,
    )
    col_lbls = criteria_lbls
    tabular_str = ut.util_latex.make_score_tabular(
        row_lbls, col_lbls, cfgscores, **tabular_kwargs
    )
    # latex_formater.render(tabular_str)
    cmdname = ut.latex_sanitize_command_name(
        'Expmt' + ibs.get_dbname() + '_' + cmdaug + 'Table'
    )
    tabular_str2 = ut.latex_newcommand(cmdname, tabular_str)
    print(tabular_str2)
    return tabular_str2


@profile
def print_results(ibs, testres, **kwargs):
    """
    Prints results from an experiment harness run.
    Rows store different qaids (query annotation ids)
    Cols store different configurations (algorithm parameters)

    TODO: join acfgs

    Args:
        ibs (IBEISController):  wbia controller object
        testres (test_result.TestResult):

    CommandLine:
        python dev.py -e print --db PZ_MTEST \
            -a default:dpername=1,qpername=[1,2]  -t default:fg_on=False
        python dev.py -e print -t best --db seals2 --allgt --vz
        python dev.py -e print --db PZ_MTEST --allgt -t custom \
            --print-confusion-stats
        python dev.py -e print --db PZ_MTEST --allgt --noqcache \
            --index 0:10:2 -t custom:rrvsone_on=True --print-confusion-stats
        python dev.py -e print --db PZ_MTEST --allgt --noqcache --qaid4 \
            -t custom:rrvsone_on=True --print-confusion-stats
        python -m wbia print_results -t default --db PZ_MTEST -a ctrl
        python -m wbia print_results -t default --db PZ_MTEST -a ctrl
        python -m wbia print_results --db PZ_MTEST -a default
            -t default:lnbnn_on=True default:lnbnn_on=False,bar_l2_on=True \
            default:lnbnn_on=False,normonly_on=True

    CommandLine:
        python -m wbia.expt.experiment_printres --test-print_results
        utprof.py -m wbia.expt.experiment_printres --test-print_results

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_printres import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts(
        >>>     'pz_mtest', a='default:dpername=1,qpername=[1,2]',
        >>>     t='default:fg_on=false')
        >>> result = print_results(ibs, testres)
        >>> print(result)
    """

    tup = ut.dict_take(
        testres.__dict__,
        ['cfg_list', 'cfgx2_cmsinfo', 'testnameid', 'cfgx2_lbl', 'cfgx2_qreq_'],
    )
    (cfg_list, cfgx2_cmsinfo, testnameid, cfgx2_lbl, cfgx2_qreq_) = tup

    # join_acfgs = kwargs.get('join_acfgs', False)

    print(' --- PRINT RESULTS ---')
    # print(' use --rank-lt-list=1,5 to specify X_LIST')
    if True:
        # Num of ranks less than to score
        X_LIST = testres.get_X_LIST()
        # X_LIST = [1, 5]

        # nConfig = len(cfg_list)
        # nQuery = len(testres.qaids)
        cfgx2_nQuery = list(map(len, testres.cfgx2_qaids))
        # cfgx2_qx2_ranks = testres.get_infoprop_list('qx2_gt_rank')
        # --------------------

        # A positive scorediff indicates the groundtruth was better than the
        # groundfalse scores
        # istrue_list  = [scorediff > 0 for scorediff in scorediffs_mat]
        # isfalse_list = [~istrue for istrue in istrue_list]

        # ------------
        # Build Colscore
        nLessX_dict = testres.get_nLessX_dict()

        # cfgx2_hist, edges = testres.get_rank_histograms(bins=X_LIST + [np.inf],
        #                                                join_acfgs=join_acfgs)
        # cfgx2_cumsum = cfgx2_hist.cumsum(axis=1)

        # ------------
        best_rankscore_summary = []
        # to_intersect_list = []
        # print each configs scores less than X=thresh
        for X, cfgx2_nLessX in six.iteritems(nLessX_dict):
            max_nLessX = cfgx2_nLessX.max()
            bestX_cfgx_list = np.where(cfgx2_nLessX == max_nLessX)[0]
            best_rankscore = '[cfg*] %d cfg(s) scored ' % len(bestX_cfgx_list)
            # FIXME
            best_rankscore += rankscore_str(
                X, max_nLessX, cfgx2_nQuery[bestX_cfgx_list[0]]
            )
            best_rankscore_summary += [best_rankscore]

    @ut.argv_flag_dec
    def intersect_hack():
        failed = testres.rank_mat > 0
        colx2_failed = [np.nonzero(failed_col)[0] for failed_col in failed.T]
        # failed_col2_only = np.setdiff1d(colx2_failed[1], colx2_failed[0])
        # failed_col2_only_aids = ut.take(testres.qaids, failed_col2_only)
        failed_col1_only = np.setdiff1d(colx2_failed[0], colx2_failed[1])
        failed_col1_only_aids = ut.take(testres.qaids, failed_col1_only)
        gt_aids1 = ibs.get_annot_groundtruth(
            failed_col1_only_aids, daid_list=testres.cfgx2_qreq_[0].daids
        )
        gt_aids2 = ibs.get_annot_groundtruth(
            failed_col1_only_aids, daid_list=testres.cfgx2_qreq_[1].daids
        )

        qaids_expt = failed_col1_only_aids
        gt_avl_aids1 = ut.flatten(gt_aids1)
        gt_avl_aids2 = list(set(ut.flatten(gt_aids2)).difference(gt_avl_aids1))

        ibs.print_annotconfig_stats(qaids_expt, gt_avl_aids1)
        ibs.print_annotconfig_stats(qaids_expt, gt_avl_aids2)
        # jsontext = ut.to_json({
        #    'qaids': list(qaids_expt),
        #    'dinclude_aids1': list(gt_aids_expt1),
        #    'dinclude_aids2': list(gt_aids_expt2),
        # })
        # annotation_configs.varysize_pzm
        # from wbia.expt import annotation_configs

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

        from wbia.init import filter_annots
        from wbia.expt import experiment_helpers

        annots1 = filter_annots.expand_acfgs(ibs, acfg1, verbose=True)
        annots2 = filter_annots.expand_acfgs(ibs, acfg2, verbose=True)

        acfg_name_list = dict(  # NOQA
            acfg_list=[acfg1, acfg2], expanded_aids_list=[annots1, annots2],
        )
        test_cfg_name_list = ['candidacy_k']
        cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(
            test_cfg_name_list, ibs=ibs
        )

        t1, t2 = testres_list  # NOQA

    # intersect_hack()

    # @ut.argv_flag_dec
    # def print_rowlbl():
    #    print('=====================')
    #    print('[harn] Row/Query Labels: %s' % testnameid)
    #    print('=====================')
    #    print('[harn] queries:\n%s' % '\n'.join(qx2_lbl))
    # print_rowlbl()
    # ------------

    @ut.argv_flag_dec
    def print_collbl():
        print('=====================')
        print('[harn] Col/Config Labels: %s' % testnameid)
        print('=====================')
        enum_cfgx2_lbl = [
            '%2d) %s' % (count, cfglbl) for count, cfglbl in enumerate(cfgx2_lbl)
        ]
        print('[harn] cfglbl:\n%s' % '\n'.join(enum_cfgx2_lbl))

    print_collbl()

    # ------------

    @ut.argv_flag_dec
    def print_cfgstr():
        print('=====================')
        print('[harn] Config Strings: %s' % testnameid)
        print('=====================')
        cfgstr_list = [query_cfg.get_cfgstr() for query_cfg in cfg_list]
        enum_cfgstr_list = [
            '%2d) %s' % (count, cfgstr) for count, cfgstr in enumerate(cfgstr_list)
        ]
        print('\n[harn] cfgstr:\n%s' % '\n'.join(enum_cfgstr_list))

    print_cfgstr(**kwargs)

    @ut.argv_flag_dec()
    def print_colscore():
        print('==================')
        print('[harn] Scores per Config: %s' % testnameid)
        print('==================')
        # for cfgx in range(nConfig):
        #    print('[score] %s' % (cfgx2_lbl[cfgx]))
        #    for X in X_LIST:
        #        nLessX_ = nLessX_dict[int(X)][cfgx]
        #        print('        ' + rankscore_str(X, nLessX_, nQuery))
        print('\n[harn] ... sorted scores')
        for X in X_LIST:
            print('\n[harn] Sorted #ranks < %r scores' % (X))
            sortx = np.array(nLessX_dict[int(X)]).argsort()
            # frac_list = (nLessX_dict[int(X)] / cfgx2_nQuery)[:, None]
            # print('cfgx2_nQuery = %r' % (cfgx2_nQuery,))
            # print('frac_list = %r' % (frac_list,))
            # print('Pairwise Difference: ' + str(ut.safe_pdist(frac_list, metric=ut.absdiff)))
            for cfgx in sortx:
                nLessX_ = nLessX_dict[int(X)][cfgx]
                rankstr = rankscore_str(X, nLessX_, cfgx2_nQuery[cfgx], withlbl=False)
                print('[score] %s --- %s' % (rankstr, cfgx2_lbl[cfgx]))

    print_colscore(**kwargs)

    ut.argv_flag_dec(testres.print_percent_identification_success)(**kwargs)

    sumstrs = []
    sumstrs.append('++===========================')
    sumstrs.append('|| [cfg*] TestName: %s' % testnameid)
    sumstrs.append('||---------------------------')
    sumstrs.append(ut.joins('\n|| ', best_rankscore_summary))
    sumstrs.append('LL===========================')
    summary_str = '\n'.join(sumstrs)
    # print(summary_str)
    ut.colorprint(summary_str, 'blue')

    print('To enable all printouts add --print-all to the commandline')


def rankscore_str(thresh, nLess, total, withlbl=True):
    # helper to print rank scores of configs
    percent = 100 * nLess / total
    fmtsf = '%' + str(ut.num2_sigfig(total)) + 'd'
    if withlbl:
        fmtstr = (
            ':#ranks < %d = '
            + fmtsf
            + '/%d = (%.1f%%) (err='
            + fmtsf
            + '/'
            + str(total)
            + ')'
        )
        rankscore_str = fmtstr % (thresh, nLess, total, percent, (total - nLess))
    else:
        fmtstr = fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + '/' + str(total) + ')'
        rankscore_str = fmtstr % (nLess, total, percent, (total - nLess))
    return rankscore_str


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.experiment_printres
        python -m wbia.expt.experiment_printres --allexamples
        python -m wbia.expt.experiment_printres --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
