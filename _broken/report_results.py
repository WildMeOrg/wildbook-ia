# flake8: noqa
#!/usr/env python
from __future__ import absolute_import, division, print_function
# Matplotlib
import matplotlib
matplotlib.use('Qt4Agg')
# Python
import os
import sys
import textwrap
import warnings
from os.path import join, exists
# Scientific imports
import numpy as np
import scipy
# Tool
from plottool import draw_func2 as df2
import utool
from vtool import keypoint as ktool
# IBEIS imports
from ibies.dev import params
from ibeis import viz
from ibeis.model.hots import QueryResult
from ibeis.model.hots import match_chips3 as mc3
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[rr2]', DEBUG=False)
#import datetime
#import subprocess

REPORT_MATRIX  = True
REPORT_MATRIX_VIZ = True

# ========================================================
# Report result initialization
# ========================================================


class AllResults(DynStruct):
    'Data container for all compiled results'
    def __init__(self, ibs, qrid2_qres):
        super(DynStruct, self).__init__()
        self.ibs                 = ibs
        self.qrid2_qres          = qrid2_qres
        self.rankres_str         = None
        self.title_suffix        = None
        self.scalar_mAP_str      = '# mAP score = NA\n'
        self.scalar_summary      = None
        self.problem_false_pairs = None
        self.problem_true_pairs  = None
        self.greater1_rids       = None
        self.greater5_rids       = None
        self.matrix_str          = None

    def __str__(allres):
        #print = tores.append
        ibs = allres.ibs
        toret = ('+======================\n')
        scalar_summary = str(allres.scalar_summary).strip()
        toret += ('| All Results: %s \n' % ibs.get_db_name())
        toret += ('| title_suffix=%s\n' % str(allres.title_suffix))
        toret += ('| scalar_summary=\n%s\n' % utool.indent(scalar_summary, '|   '))
        toret += ('| ' + str(allres.scalar_mAP_str))
        toret += ('|---\n')
        toret += ('| greater5_%s \n' % (ibs.cidstr(allres.greater5_rids),))
        toret += ('|---\n')
        toret += ('| greater1_%s \n' % (ibs.cidstr(allres.greater1_rids),))
        toret += ('|---\n')
        toret += ('+======================.\n')
        #toret+=('| problem_false_pairs=\n%r' % allres.problem_false_pairs)
        #toret+=('| problem_true_pairs=\n%r' % allres.problem_true_pairs)
        return toret


def get_false_match_distances(allres):
    false_distances = get_orgres_match_distances(allres, 'false')
    return false_distances


def get_true_match_distances(allres):
    true_distances = get_orgres_match_distances(allres, 'true')
    return true_distances


def init_score_matrix(allres):
    print('[rr2] init score matrix')
    ibs = allres.ibs
    qrid2_qres = allres.qrid2_qres
    qrid_list = allres.qrid_list
    nx_list = np.unique(ibs.tables.cx2_nx[qrid_list])
    #nx_list = ibs.get_valid_nxs(unknown=False)
    cxs_list = ibs.nx2_rids(nx_list, aslist=True)
    # Sort names by number of chips
    nx_size = map(len, cxs_list)
    # Build sorted chip list
    nx_cxs_tuples = zip(nx_size, cxs_list)
    # Sort by name
    cx_sorted = [x for (y, x) in sorted(nx_cxs_tuples)]
    # Subsort by chip
    cx_sorted = map(sorted, cx_sorted)
    cx_sorted = utool.flatten(cx_sorted)
    row_label_rid = []
    row_scores = []
    qcx_set = set(qrid_list)
    # Build each row in the score matrix
    for qrid in iter(cx_sorted):
        if not qrid in qcx_set:
            continue
        try:
            qres = qrid2_qres[qrid]
        except IndexError:
            print('qrid = %r' % qrid)
            print('len(qrid2_qres) = %r' % len(qrid2_qres))
            raise
        if qres is None:
            continue
        # Append a label to score matrix
        row_label_rid.append(qrid)
        # Append a column to score matrix
        row_scores.append(qres.cx2_score[cx_sorted])
    col_label_rid = cx_sorted
    # convert to numpy matrix array
    score_matrix = np.array(row_scores, dtype=np.float64)
    # Fill diagonal with -1's
    np.fill_diagonal(score_matrix, -np.ones(len(row_label_rid)))
    # Add score matrix to allres
    allres.score_matrix = score_matrix
    allres.col_label_rid = col_label_rid
    allres.row_label_rid = row_label_rid


def get_title_suffix(ibs):
    title_suffix = ibs.get_cache_rowid()
    return title_suffix


# ========================================================
# Build textfile result strings
# ========================================================


def build_matrix_str(allres):
    ibs = allres.ibs
    cx2_gx = ibs.tables.cx2_gx
    gx2_gname = ibs.tables.gx2_gname

    def cx2_gname(rid):
        return [os.path.splitext(gname)[0] for gname in gx2_gname[cx2_gx]]
    col_label_gname = cx2_gname(allres.col_label_rid)
    row_label_gname = cx2_gname(allres.row_label_rid)
    timestamp =  utool.get_timestamp(format_='comment') + '\n'
    header = '\n'.join(
        ['# Result score matrix',
         '# Generated on: ' + timestamp,
         '# Format: rows separated by newlines, cols separated by commas',
         '# num_queries  / rows = ' + repr(len(row_label_gname)),
         '# num_indexed  / cols = ' + repr(len(col_label_gname)),
         '# row_labels = ' + repr(row_label_gname),
         '# col_labels = ' + repr(col_label_gname)])
    row_strings = []
    for row in allres.score_matrix:
        row_str = map(lambda x: '%5.2f' % x, row)
        row_strings.append(', '.join(row_str))
    body = '\n'.join(row_strings)
    matrix_str = '\n'.join([header, body])
    allres.matrix_str = matrix_str


def build_rankres_str(allres):
    'Builds csv files showing the rids/scores/ranks of the query results'
    ibs = allres.ibs
    #qrid2_qres = allres.qrid2_qres
    cx2_cid = ibs.tables.cx2_cid
    #cx2_nx = ibs.tables.cx2_nx
    test_samp = allres.qrid_list
    train_samp = ibs.train_sample_rid
    indx_samp = ibs.indexed_sample_rid
    # Get organized data for csv file
    (qcx2_top_true_rank,
     qcx2_top_true_score,
     qcx2_top_true_rid)  = allres.top_true_qcx_arrays

    (qcx2_bot_true_rank,
     qcx2_bot_true_score,
     qcx2_bot_true_rid)  = allres.bot_true_qcx_arrays

    (qcx2_top_false_rank,
     qcx2_top_false_score,
     qcx2_top_false_rid) = allres.top_false_qcx_arrays
    # Number of groundtruth per query
    qcx2_numgt = np.zeros(len(cx2_cid)) - 2
    for qrid in test_samp:
        qcx2_numgt[qrid] = len(ibs.get_other_indexed_rids(qrid))
    # Easy to digest results
    num_chips = len(test_samp)
    num_nonquery = len(np.setdiff1d(indx_samp, test_samp))
    # Find the test samples WITH ground truth
    test_samp_with_gt = np.array(test_samp)[qcx2_numgt[test_samp] > 0]
    if len(test_samp_with_gt) == 0:
        warnings.warn('[rr2] there were no queries with ground truth')
    #train_nxs_set = set(cx2_nx[train_samp])
    flag_cxs_fn = ibs.flag_cxs_with_name_in_sample

    def ranks_less_than_(thresh, intrain=None):
        #Find the number of ranks scoring more than thresh
        # Get statistics with respect to the training set
        if len(test_samp_with_gt) == 0:
            test_cxs_ = np.array([])
        elif intrain is None:  # report all
            test_cxs_ =  test_samp_with_gt
        else:  # report either or
            in_train_flag = flag_cxs_fn(test_samp_with_gt, train_samp)
            if intrain is False:
                in_train_flag = True - in_train_flag
            test_cxs_ =  test_samp_with_gt[in_train_flag]
        # number of test samples with ground truth
        num_with_gt = len(test_cxs_)
        if num_with_gt == 0:
            return [], ('NoGT', 'NoGT', -1, 'NoGT')
        # find tests with ranks greater and less than thresh
        testcx2_ttr = qcx2_top_true_rank[test_cxs_]
        greater_rids = test_cxs_[np.where(testcx2_ttr >= thresh)[0]]
        num_greater = len(greater_rids)
        num_less    = num_with_gt - num_greater
        num_greater = num_with_gt - num_less
        frac_less   = 100.0 * num_less / num_with_gt
        fmt_tup     = (num_less, num_with_gt, frac_less, num_greater)
        return greater_rids, fmt_tup

    greater5_rids, fmt5_tup = ranks_less_than_(5)
    greater1_rids, fmt1_tup = ranks_less_than_(1)
    #
    gt5_intrain_rids, fmt5_in_tup = ranks_less_than_(5, intrain=True)
    gt1_intrain_rids, fmt1_in_tup = ranks_less_than_(1, intrain=True)
    #
    gt5_outtrain_rids, fmt5_out_tup = ranks_less_than_(5, intrain=False)
    gt1_outtrain_rids, fmt1_out_tup = ranks_less_than_(1, intrain=False)
    #
    allres.greater1_rids = greater1_rids
    allres.greater5_rids = greater5_rids
    #print('greater5_rids = %r ' % (allres.greater5_rids,))
    #print('greater1_rids = %r ' % (allres.greater1_rids,))
    # CSV Metadata
    header = '# Experiment allres.title_suffix = ' + allres.title_suffix + '\n'
    header +=  utool.get_timestamp(format_='comment') + '\n'
    # Scalar summary
    scalar_summary  = '# Num Query Chips: %d \n' % num_chips
    scalar_summary += '# Num Query Chips with at least one match: %d \n' % len(test_samp_with_gt)
    scalar_summary += '# Num NonQuery Chips: %d \n' % num_nonquery
    scalar_summary += '# Ranks <= 5: %r/%r = %.1f%% (missed %r)\n' % (fmt5_tup)
    scalar_summary += '# Ranks <= 1: %r/%r = %.1f%% (missed %r)\n\n' % (fmt1_tup)

    scalar_summary += '# InTrain Ranks <= 5: %r/%r = %.1f%% (missed %r)\n' % (fmt5_in_tup)
    scalar_summary += '# InTrain Ranks <= 1: %r/%r = %.1f%% (missed %r)\n\n' % (fmt1_in_tup)

    scalar_summary += '# OutTrain Ranks <= 5: %r/%r = %.1f%% (missed %r)\n' % (fmt5_out_tup)
    scalar_summary += '# OutTrain Ranks <= 1: %r/%r = %.1f%% (missed %r)\n\n' % (fmt1_out_tup)
    header += scalar_summary
    # Experiment parameters
    #header += '# Full Parameters: \n' + utool.indent(params.param_string(), '#') + '\n\n'
    # More Metadata
    header += textwrap.dedent('''
    # Rank Result Metadata:
    #   QCX  = Query chip-index
    # QGNAME = Query images name
    # NUMGT  = Num ground truth matches
    #    TT  = top true
    #    BT  = bottom true
    #    TF  = top false''').strip()
    # Build the CSV table
    test_sample_gx = ibs.tables.cx2_gx[test_samp]
    test_sample_gname = ibs.tables.gx2_gname[test_sample_gx]
    test_sample_gname = [g.replace('.jpg', '') for g in test_sample_gname]
    column_labels = ['QCX', 'NUM GT',
                     'TT CX', 'BT CX', 'TF CX',
                     'TT SCORE', 'BT SCORE', 'TF SCORE',
                     'TT RANK', 'BT RANK', 'TF RANK',
                     'QGNAME', ]
    column_list = [
        test_samp, qcx2_numgt[test_samp],
        qcx2_top_true_rid[test_samp], qcx2_bot_true_rid[test_samp],
        qcx2_top_false_rid[test_samp], qcx2_top_true_score[test_samp],
        qcx2_bot_true_score[test_samp], qcx2_top_false_score[test_samp],
        qcx2_top_true_rank[test_samp], qcx2_bot_true_rank[test_samp],
        qcx2_top_false_rank[test_samp], test_sample_gname, ]
    column_type = [int, int, int, int, int,
                   float, float, float, int, int, int, str, ]
    rankres_str = utool.util_csv.make_csv_table(column_labels, column_list, header, column_type)
    # Put some more data at the end
    problem_true_pairs = zip(allres.problem_true.qrids, allres.problem_true.rids)
    problem_false_pairs = zip(allres.problem_false.qrids, allres.problem_false.rids)
    problem_str = '\n'.join( [
        '#Problem Cases: ',
        '# problem_true_pairs = ' + repr(problem_true_pairs),
        '# problem_false_pairs = ' + repr(problem_false_pairs)])
    rankres_str += '\n' + problem_str
    # Attach results to allres structure
    allres.rankres_str = rankres_str
    allres.scalar_summary = scalar_summary
    allres.problem_false_pairs = problem_false_pairs
    allres.problem_true_pairs = problem_true_pairs
    allres.problem_false_pairs = problem_false_pairs
    allres.problem_true_pairs = problem_true_pairs


# ===========================
# Helper Functions
# ===========================


def __dump_text_report(allres, report_type):
    if not 'report_type' in vars():
        report_type = 'rankres_str'
    print('[rr2] Dumping textfile: ' + report_type)
    report_str = allres.__dict__[report_type]
    # Get directories
    result_dir    = allres.ibs.dirs.result_dir
    timestamp_dir = join(result_dir, 'timestamped_results')
    utool.ensurepath(timestamp_dir)
    utool.ensurepath(result_dir)
    # Write to timestamp and result dir
    timestamp = utool.get_timestamp()
    csv_timestamp_fname = report_type + allres.title_suffix + timestamp + '.csv'
    csv_timestamp_fpath = join(timestamp_dir, csv_timestamp_fname)
    csv_fname  = report_type + allres.title_suffix + '.csv'
    csv_fpath = join(result_dir, csv_fname)
    utool.write_to(csv_fpath, report_str)
    utool.write_to(csv_timestamp_fpath, report_str)

# ===========================
# Driver functions
# ===========================


TMP = False
SCORE_PDF  = TMP
RANK_HIST  = TMP
PARI_ANALY = TMP
STEM       = TMP
TOP5       = TMP
#if TMP:
ALLQUERIES = False
ANALYSIS = True


def dump_all(allres,
             matrix=REPORT_MATRIX,  #
             matrix_viz=REPORT_MATRIX_VIZ,  #
             score_pdf=SCORE_PDF,
             rank_hist=RANK_HIST,
             ttbttf=False,
             problems=False,
             gtmatches=False,
             oxford=False,
             no_viz=False,
             rankres=True,
             stem=STEM,
             missed_top5=TOP5,
             analysis=ANALYSIS,
             pair_analysis=PARI_ANALY,
             allqueries=ALLQUERIES):
    print('\n======================')
    print('[rr2] DUMP ALL')
    print('======================')
    viz.BROWSE = False
    viz.DUMP = True
    # Text Reports
    if rankres:
        dump_rankres_str_results(allres)
    if matrix:
        dump_matrix_str_results(allres)
    if oxford:
        dump_oxsty_mAP_results(allres)
    if no_viz:
        print('\n --- (NO VIZ) END DUMP ALL ---\n')
        return
    # Viz Reports
    if stem:
        dump_rank_stems(allres)
    if matrix_viz:
        dump_score_matrixes(allres)
    if rank_hist:
        dump_rank_hists(allres)
    if score_pdf:
        dump_score_pdfs(allres)
    #
    #if ttbttf:
        #dump_ttbttf_matches(allres)
    if problems:
        dump_problem_matches(allres)
    if gtmatches:
        dump_gt_matches(allres)
    if missed_top5:
        dump_missed_top5(allres)
    if analysis:
        dump_analysis(allres)
    if pair_analysis:
        dump_feature_pair_analysis(allres)
    if allqueries:
        dump_all_queries(allres)
    print('\n --- END DUMP ALL ---\n')


def dump_oxsty_mAP_results(allres):
    #print('\n---DUMPING OXSTYLE RESULTS---')
    __dump_text_report(allres, 'oxsty_map_csv')


def dump_rankres_str_results(allres):
    #print('\n---DUMPING RANKRES RESULTS---')
    __dump_text_report(allres, 'rankres_str')


def dump_matrix_str_results(allres):
    #print('\n---DUMPING MATRIX STRING RESULTS---')
    __dump_text_report(allres, 'matrix_str')


def dump_problem_matches(allres):
    #print('\n---DUMPING PROBLEM MATCHES---')
    dump_orgres_matches(allres, 'problem_false')
    dump_orgres_matches(allres, 'problem_true')


def dump_score_matrixes(allres):
    #print('\n---DUMPING SCORE MATRIX---')
    try:
        allres_viz.plot_score_matrix(allres)
    except Exception as ex:
        print('[dump_score_matixes] IMPLEMENTME: %r ' % ex)
    pass


def dump_rank_stems(allres):
    #print('\n---DUMPING RANK STEMS---')
    viz.plot_rank_stem(allres, 'true')


def dump_rank_hists(allres):
    #print('\n---DUMPING RANK HISTS---')
    viz.plot_rank_histogram(allres, 'true')


def dump_score_pdfs(allres):
    #print('\n---DUMPING SCORE PDF ---')
    viz.plot_score_pdf(allres, 'true',      colorx=0.0, variation_truncate=True)
    viz.plot_score_pdf(allres, 'false',     colorx=0.2)
    viz.plot_score_pdf(allres, 'top_true',  colorx=0.4, variation_truncate=True)
    viz.plot_score_pdf(allres, 'bot_true',  colorx=0.6)
    viz.plot_score_pdf(allres, 'top_false', colorx=0.9)


def dump_gt_matches(allres):
    #print('\n---DUMPING GT MATCHES ---')
    'Displays the matches to ground truth for all queries'
    qrid2_qres = allres.qrid2_qres
    for qrid in xrange(0, len(qrid2_qres)):
        viz.show_chip(allres, qrid, 'gt_matches')


def dump_missed_top5(allres):
    #print('\n---DUMPING MISSED TOP 5---')
    'Displays the top5 matches for all queries'
    greater5_rids = allres.greater5_rids
    #qrid = greater5_rids[0]
    for qrid in greater5_rids:
        viz.show_chip(allres, qrid, 'top5', 'missed_top5')
        viz.show_chip(allres, qrid, 'gt_matches', 'missed_top5')


def dump_analysis(allres):
    print('[rr2] dump analysis')
    greater1_rids = allres.greater1_rids
    #qrid = greater5_rids[0]
    for qrid in greater1_rids:
        viz.show_chip(allres, qrid, 'analysis', 'analysis')
        viz.show_chip(allres, qrid, 'analysis', 'analysis', annotations=False, title_aug=' noanote')


def dump_all_queries2(ibs):
    test_rids = ibs.test_sample_rid
    title_suffix = get_title_suffix(ibs)
    print('[rr2] dumping all %r queries' % len(test_rids))
    for qrid in test_rids:
        qres = QueryResult.QueryResult(qrid)
        qres.load(ibs)
        # SUPER HACK (I don't know the figurename a priori, I have to contstruct
        # it to not duplciate dumping a figure)
        title_aug = ' noanote'
        fpath = ibs.dirs.result_dir
        subdir = 'allqueries'
        N = 5
        topN_rids = qres.topN_rids(N)
        topscore = qres.cx2_score[topN_rids][0]

        dump_dir = join(fpath, subdir + title_suffix)

        fpath     = join(dump_dir, ('topscore=%r -- qcid=%r' % (topscore, qres.qcid)))
        fpath_aug = join(dump_dir, ('topscore=%r -- qcid=%r' % (topscore, qres.qcid))) + title_aug

        fpath_clean = df2.sanatize_img_fpath(fpath)
        fpath_aug_clean = df2.sanatize_img_fpath(fpath_aug)
        print('----')
        print(fpath_clean)
        print(fpath_clean)
        if not exists(fpath_aug_clean):
            viz.plot_cx2(ibs, qres, 'analysis', subdir=subdir, annotations=False, title_aug=title_aug)
        if not exists(fpath_clean):
            viz.plot_cx2(ibs, qres, 'analysis', subdir=subdir)
        print('----')


def dump_all_queries(allres):
    test_rids = allres.qrid_list
    print('[rr2] dumping all %r queries' % len(test_rids))
    for qrid in test_rids:
        viz.show_chip(allres, qrid, 'analysis', subdir='allqueries',
                      annotations=False, title_aug=' noanote')
        viz.show_chip(allres, qrid, 'analysis', subdir='allqueries')


def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    ibs = allres.ibs
    qrid2_qres = allres.qrid2_qres
    # loop over each query / result of interest
    for qrid, rid, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(ibs.tables.gx2_gname[ibs.tables.cx2_gx[qrid]])
        result_gname, _ = os.path.splitext(ibs.tables.gx2_gname[ibs.tables.cx2_gx[rid]])
        qres = qrid2_qres[qrid]
        df2.figure(fnum=1, plotnum=121)
        df2.show_matches_annote_res(qres, ibs, rid, fnum=1, plotnum=121)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % (score, rank, query_gname, result_gname)
        df2.set_figtitle(big_title)
        viz.__dump_or_browse(allres, orgres_type + '_matches' + allres.title_suffix)


def dump_feature_pair_analysis(allres):
    print('[rr2] Doing: feature pair analysis')
    # TODO: Measure score consistency over a spatial area.
    # Measures entropy of matching vs nonmatching descriptors
    # Measures scale of m vs nm desc
    ibs = allres.ibs
    qrid2_qres = allres.qrid2_qres

    def _hist_prob_x(desc, bw_factor):
        # Choose number of bins based on the bandwidth
        bin_range = (0, 256)  # assuming input is uint8
        bins = bin_range[1] // bw_factor
        bw_factor = bin_range[1] / bins
        # Compute the probabilty mass function, each w.r.t a single descriptor
        hist_params = dict(bins=bins, range=bin_range, density=True)
        hist_func = np.histogram
        desc_pmf = [hist_func(d, **hist_params)[0] for d in desc]
        # Compute the probability that you saw what you saw
        # TODO: could use linear interpolation for a bit more robustness here
        bin_vals = [np.array(np.floor(d / bw_factor), dtype=np.uint8) for d in desc]
        hist_prob_x = [pmf[vals] for pmf, vals in zip(desc_pmf, bin_vals)]
        return hist_prob_x

    def _gkde_prob_x(desc, bw_factor):
        # Estimate the probabilty density function, each w.r.t a single descriptor
        gkde_func = scipy.stats.gaussian_kde
        desc_pdf = [gkde_func(d, bw_factor) for d in desc]
        gkde_prob_x = [pdf(d) for pdf, d in zip(desc_pdf, desc)]
        return gkde_prob_x

    def descriptor_entropy(desc, bw_factor=4):
        'computes the shannon entropy of each descriptor in desc'
        # Compute shannon entropy = -sum(p(x)*log(p(x)))
        prob_x = _hist_prob_x(desc, bw_factor)
        entropy = [-(px * np.log2(px)).sum() for px in prob_x]
        return entropy

    # Load features if we need to
    if ibs.feats.cx2_desc.size == 0:
        print(' * forcing load of descriptors')
        ibs.load_features()
    cx2_desc = ibs.feats.cx2_desc
    cx2_kpts = ibs.feats.cx2_kpts

    def measure_feat_pairs(allres, orgtype='top_true'):
        print('Measure ' + orgtype + ' pairs')
        orgres = allres.__dict__[orgtype]
        entropy_list = []
        scale_list = []
        score_list = []
        lbl = 'Measuring ' + orgtype + ' pair '
        fmt_str = utool.make_progress_fmt_str(len(orgres), lbl)
        rank_skips = []
        gt_skips = []
        for ix, (qrid, rid, score, rank) in enumerate(orgres.iter()):
            utool.print_(fmt_str % (ix + 1,))
            # Skip low ranks
            if rank > 5:
                rank_skips.append(qrid)
                continue
            other_rids = ibs.get_other_indexed_rids(qrid)
            # Skip no groundtruth
            if len(other_rids) == 0:
                gt_skips.append(qrid)
                continue
            qres = qrid2_qres[qrid]
            # Get matching feature indexes
            fm = qres.cx2_fm[rid]
            # Get their scores
            fs = qres.cx2_fs[rid]
            # Get matching descriptors
            printDBG('\nfm.shape=%r' % (fm.shape,))
            desc1 = cx2_desc[qrid][fm[:, 0]]
            desc2 = cx2_desc[rid][fm[:, 1]]
            # Get matching keypoints
            kpts1 = cx2_kpts[qrid][fm[:, 0]]
            kpts2 = cx2_kpts[rid][fm[:, 1]]
            # Get their scale
            scale1_m = ktool.get_scales(kpts1)
            scale2_m = ktool.get_scales(kpts2)
            # Get their entropy
            entropy1 = descriptor_entropy(desc1, bw_factor=1)
            entropy2 = descriptor_entropy(desc2, bw_factor=1)
            # Append to results
            entropy_tup = np.array(zip(entropy1, entropy2))
            scale_tup   = np.array(zip(scale1_m, scale2_m))
            entropy_tup = entropy_tup.reshape(len(entropy_tup), 2)
            scale_tup   = scale_tup.reshape(len(scale_tup), 2)
            entropy_list.append(entropy_tup)
            scale_list.append(scale_tup)
            score_list.append(fs)
        print('Skipped %d total.' % (len(rank_skips) + len(gt_skips),))
        print('Skipped %d for rank > 5, %d for no gt' % (len(rank_skips), len(gt_skips),))
        print(np.unique(map(len, entropy_list)))

        def evstack(tup):
            return np.vstack(tup) if len(tup) > 0 else np.empty((0, 2))

        def ehstack(tup):
            return np.hstack(tup) if len(tup) > 0 else np.empty((0, 2))

        entropy_pairs = evstack(entropy_list)
        scale_pairs   = evstack(scale_list)
        scores        = ehstack(score_list)
        print('\n * Measured %d pairs' % len(entropy_pairs))
        return entropy_pairs, scale_pairs, scores

    tt_entropy, tt_scale, tt_scores = measure_feat_pairs(allres, 'top_true')
    tf_entropy, tf_scale, tf_scores = measure_feat_pairs(allres, 'top_false')

    # Measure ratios
    def measure_ratio(arr):
        return arr[:, 0] / arr[:, 1] if len(arr) > 0 else np.array([])
    tt_entropy_ratio = measure_ratio(tt_entropy)
    tf_entropy_ratio = measure_ratio(tf_entropy)
    tt_scale_ratio   = measure_ratio(tt_scale)
    tf_scale_ratio   = measure_ratio(tf_scale)

    title_suffix = allres.title_suffix

    # Entropy vs Score
    df2.figure(fnum=1, docla=True)
    df2.figure(fnum=1, plotnum=(2, 2, 1))
    df2.plot2(tt_entropy[:, 0], tt_scores, 'gx', 'entropy1', 'score', 'Top True')
    df2.figure(fnum=1, plotnum=(2, 2, 2))
    df2.plot2(tf_entropy[:, 0], tf_scores, 'rx', 'entropy1', 'score', 'Top False')
    df2.figure(fnum=1, plotnum=(2, 2, 3))
    df2.plot2(tt_entropy[:, 1], tt_scores, 'gx', 'entropy2', 'score', 'Top True')
    df2.figure(fnum=1, plotnum=(2, 2, 4))
    df2.plot2(tf_entropy[:, 1], tf_scores, 'rx', 'entropy2', 'score', 'Top False')
    df2.set_figtitle('Entropy vs Score -- ' + title_suffix)
    viz.__dump_or_browse(allres, 'pair_analysis')

    # Scale vs Score
    df2.figure(fnum=2, plotnum=(2, 2, 1), docla=True)
    df2.plot2(tt_scale[:, 0], tt_scores, 'gx', 'scale1', 'score', 'Top True')
    df2.figure(fnum=2, plotnum=(2, 2, 2))
    df2.plot2(tf_scale[:, 0], tf_scores, 'rx', 'scale1', 'score', 'Top False')
    df2.figure(fnum=2, plotnum=(2, 2, 3))
    df2.plot2(tt_scale[:, 1], tt_scores, 'gx', 'scale2', 'score', 'Top True')
    df2.figure(fnum=2, plotnum=(2, 2, 4))
    df2.plot2(tf_scale[:, 1], tf_scores, 'rx', 'scale2', 'score', 'Top False')
    df2.set_figtitle('Scale vs Score -- ' + title_suffix)
    viz.__dump_or_browse(allres, 'pair_analysis')

    # Entropy Ratio vs Score
    df2.figure(fnum=3, plotnum=(1, 2, 1), docla=True)
    df2.plot2(tt_entropy_ratio, tt_scores, 'gx', 'entropy-ratio', 'score', 'Top True')
    df2.figure(fnum=3, plotnum=(1, 2, 2))
    df2.plot2(tf_entropy_ratio, tf_scores, 'rx', 'entropy-ratio', 'score', 'Top False')
    df2.set_figtitle('Entropy Ratio vs Score -- ' + title_suffix)
    viz.__dump_or_browse(allres, 'pair_analysis')

    # Scale Ratio vs Score
    df2.figure(fnum=4, plotnum=(1, 2, 1), docla=True)
    df2.plot2(tt_scale_ratio, tt_scores, 'gx', 'scale-ratio', 'score', 'Top True')
    df2.figure(fnum=4, plotnum=(1, 2, 2))
    df2.plot2(tf_scale_ratio, tf_scores, 'rx', 'scale-ratio', 'score', 'Top False')
    df2.set_figtitle('Entropy Ratio vs Score -- ' + title_suffix)
    viz.__dump_or_browse(allres, 'pair_analysis')

    #df2.rrr(); viz.rrr(); clf(); df2.show_chip(ibs, 14, allres=allres)
    #viz.show_chip(allres, 14, 'top5')
    #viz.show_chip(allres, 14, 'gt_matches')
    #df2.show_chip(ibs, 1, allres=allres)


def possible_problems():
    # Perhaps overlapping keypoints are causing more harm than good.
    # Maybe there is a way of grouping them or averaging them into a
    # better descriptor.
    pass


#===============================
# MAIN SCRIPT
#===============================


def report_all(ibs, qrid2_qres, qrid_list, **kwargs):
    allres = init_allres(ibs, qrid2_qres, qrid_list=qrid_list, **kwargs)
    #if not 'kwargs' in vars():
        #kwargs = dict(rankres=True, stem=False, matrix=False, pdf=False,
                      #hist=False, oxford=False, ttbttf=False, problems=False,
                      #gtmatches=False)
    try:
        dump_all(allres, **kwargs)
    except Exception as ex:
        import traceback
        print('\n\n-----------------')
        print('report_all(ibs, qrid2_qres, **kwargs=%r' % (kwargs))
        print('Caught Error in rr2.dump_all')
        print(repr(ex))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print('Caught Error in rr2.dump_all')
        print('-----------------\n')
        raise
        return allres, ex
    return allres


def read_until(file, target):
    curent_line = file.readline()
    while not curent_line is None:
        if curent_line.find(target) > -1:
            return curent_line
        curent_line = file.readline()


def _get_orgres2_distances(allres, orgres_list=None):
    if orgres_list is None:
        #orgres_list = ['true', 'false', 'top_true', 'bot_true', 'top_false']
        orgres_list = ['true', 'false']
    #print(allres)
    dist_fn = lambda orgres: get_orgres_match_distances(allres, orgres)
    orgres2_distance = {}
    for orgres in orgres_list:
        try:
            orgres2_distance[orgres] = dist_fn(orgres)
        except Exception as ex:
            print(ex)
            print('failed dist orgres=%r' % orgres)
    return orgres2_distance


@profile
def get_orgres_match_distances(allres, orgtype_='false'):
    import algos
    qrids = allres[orgtype_].qrids
    rids  = allres[orgtype_].rids
    match_list = zip(qrids, rids)
    printDBG('[rr2] getting orgtype_=%r distances between sifts' % orgtype_)
    adesc1, adesc2 = get_matching_descriptors(allres, match_list)
    printDBG('[rr2]  * adesc1.shape = %r' % (adesc1.shape,))
    printDBG('[rr2]  * adesc2.shape = %r' % (adesc2.shape,))
    #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
    #dist_list = ['L1', 'L2', 'hist_isect']
    dist_list = ['L2', 'hist_isect']
    hist1 = np.array(adesc1, dtype=np.float64)
    hist2 = np.array(adesc2, dtype=np.float64)
    distances = algos.compute_distances(hist1, hist2, dist_list)
    return distances


def get_matching_descriptors(allres, match_list):
    ibs = allres.ibs
    qrid2_qres = allres.qrid2_qres
    # FIXME: More intelligent feature loading
    if len(ibs.feats.cx2_desc) == 0:
        ibs.refresh_features()
    cx2_desc = ibs.feats.cx2_desc
    desc1_list = []
    desc2_list = []
    desc1_append = desc1_list.append
    desc2_append = desc2_list.append
    for qrid, rid in match_list:
        fx2_desc1 = cx2_desc[qrid]
        fx2_desc2 = cx2_desc[rid]
        qres = qrid2_qres[qrid]

        fm = qres.cx2_fm[rid]
        #fs = qres.cx2_fs[rid]
        if len(fm) == 0:
            continue
        fx1_list = fm.T[0]
        fx2_list = fm.T[1]
        desc1 = fx2_desc1[fx1_list]
        desc2 = fx2_desc2[fx2_list]
        desc1_append(desc1)
        desc2_append(desc2)
    aggdesc1 = np.vstack(desc1_list)
    aggdesc2 = np.vstack(desc2_list)
    return aggdesc1, aggdesc2


def load_qcx2_res(ibs, qrid_list, nocache=False):
    'Prefrosm / loads all queries'
    qreq = mc3.quickly_ensure_qreq(ibs, qrids=qrid_list)
    # Build query big cache rowid
    query_rowid = qreq.get_rowid()
    hs_rowid    = ibs.get_db_name()
    qcxs_rowid  = utool.hashstr_arr(qrid_list, lbl='_qcxs')
    qres_rowid  = hs_rowid + query_rowid + qcxs_rowid
    cache_dir = join(ibs.dirs.cache_dir, 'query_results_bigcache')
    print('[rr2] load_qcx2_res(): %r' % qres_rowid)
    io_kwargs = dict(dpath=cache_dir, fname='query_results', rowid=qres_rowid, ext='.cPkl')
    # Return cache if available
    if not params.args.nocache_query and (not nocache):
        qrid2_qres = io.smart_load(**io_kwargs)
        if qrid2_qres is not None:
            print('[rr2]  *  cache hit')
            return qrid2_qres
        print('[rr2]  *  cache miss')
    else:
        print('[rr2]  *  cache off')
    # Individually load / compute queries
    if isinstance(qrid_list, list):
        qcx_set = set(qrid_list)
    else:
        qcx_set = set(qrid_list.tolist())
    qcx_max = max(qrid_list) + 1
    qrid2_qres = [ibs.query(qrid) if qrid in qcx_set else None for qrid in xrange(qcx_max)]
    # Save to the cache
    print('[rr2] Saving query_results to bigcache: %r' % qres_rowid)
    utool.ensuredir(cache_dir)
    io.smart_save(qrid2_qres, **io_kwargs)
    return qrid2_qres


def get_allres(ibs, qrid_list):
    'Performs / Loads all queries and build allres structure'
    print('[rr2] get_allres()')
    #valid_rids = ibs.get_valid_rids()
    qrid2_qres = load_qcx2_res(ibs, qrid_list)
    allres = init_allres(ibs, qrid2_qres, qrid_list=qrid_list)
    return allres
