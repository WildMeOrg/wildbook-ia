#!/usr/env python
from __future__ import absolute_import, division, print_function
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[rr2]', DEBUG=False)
# Matplotlib
import matplotlib
matplotlib.use('Qt4Agg')
# Python
import os
import sys
import textwrap
import warnings
from itertools import izip
from os.path import join, exists
# Scientific imports
import numpy as np
# Hotspotter imports
from ibies.dev import params
from hscom.Printable import DynStruct
from plottool import draw_func2 as df2
from ibeis import viz
from hscom import csvtool
from hsapi import spatial_verification2 as sv2
#import datetime
#import subprocess

REPORT_MATRIX  = True
REPORT_MATRIX_VIZ = True

# ========================================================
# Report result initialization
# ========================================================


class AllResults(DynStruct):
    'Data container for all compiled results'
    def __init__(self, hs, qcx2_res, qcx_list):
        super(DynStruct, self).__init__()
        self.hs       = hs
        self.qcx2_res = qcx2_res
        self.qcx_list = hs.test_sample_cx if qcx_list is None else qcx_list
        self.rankres_str         = None
        self.title_suffix        = None
        self.scalar_mAP_str      = '# mAP score = NA\n'
        self.scalar_summary      = None
        self.problem_false_pairs = None
        self.problem_true_pairs  = None
        self.greater1_cxs = None
        self.greater5_cxs = None
        self.matrix_str          = None

    def get_orgres2_distances(allres, *args, **kwargs):
        return _get_orgres2_distances(allres, *args, **kwargs)

    def __str__(allres):
        #print = tores.append
        hs = allres.hs
        toret = ('+======================\n')
        scalar_summary = str(allres.scalar_summary).strip()
        toret += ('| All Results: %s \n' % hs.get_db_name())
        toret += ('| title_suffix=%s\n' % str(allres.title_suffix))
        toret += ('| scalar_summary=\n%s\n' % utool.indent(scalar_summary, '|   '))
        toret += ('| ' + str(allres.scalar_mAP_str))
        toret += ('|---\n')
        toret += ('| greater5_%s \n' % (hs.cidstr(allres.greater5_cxs),))
        toret += ('|---\n')
        toret += ('| greater1_%s \n' % (hs.cidstr(allres.greater1_cxs),))
        toret += ('|---\n')
        toret += ('+======================.\n')
        #toret+=('| problem_false_pairs=\n%r' % allres.problem_false_pairs)
        #toret+=('| problem_true_pairs=\n%r' % allres.problem_true_pairs)
        return toret


class OrganizedResult(DynStruct):
    '''
    Maintains an organized list of query chip indexes, their top matching
    result, the score, and the rank. What chips are populated depends on the
    type of organization
    '''
    def __init__(self):
        super(DynStruct, self).__init__()
        self.qcxs   = []
        self.cxs    = []
        self.scores = []
        self.ranks  = []

    def append(self, qcx, cx, rank, score):
        self.qcxs.append(qcx)
        self.cxs.append(cx)
        self.scores.append(score)
        self.ranks.append(rank)

    def __len__(self):
        num_qcxs   = len(self.qcxs)
        num_cxs    = len(self.cxs)
        num_scores = len(self.scores)
        num_ranks  = len(self.ranks)
        assert num_qcxs == num_cxs
        assert num_cxs == num_scores
        assert num_scores == num_ranks
        return num_qcxs

    def iter(self):
        'useful for plotting'
        result_iter = izip(self.qcxs, self.cxs, self.scores, self.ranks)
        for qcx, cx, score, rank in result_iter:
            yield qcx, cx, score, rank

    def qcx_arrays(self, hs):
        'useful for reportres_str'
        cx2_cid     = hs.tables.cx2_cid
        qcx2_rank   = np.zeros(len(cx2_cid)) - 2
        qcx2_score  = np.zeros(len(cx2_cid)) - 2
        qcx2_cx     = np.arange(len(cx2_cid)) * -1
        #---
        for (qcx, cx, score, rank) in self.iter():
            qcx2_rank[qcx] = rank
            qcx2_score[qcx] = score
            qcx2_cx[qcx] = cx
        return qcx2_rank, qcx2_score, qcx2_cx

    def printme3(self):
        for qcx, cx, score, rank in self.iter():
            print('%4d %4d %6.1f %4d' % (qcx, cx, score, rank))


def get_false_match_distances(allres):
    false_distances = get_orgres_match_distances(allres, 'false')
    return false_distances


def get_true_match_distances(allres):
    true_distances = get_orgres_match_distances(allres, 'true')
    return true_distances


def res2_true_and_false(hs, res):
    '''
    Organizes results into true positive and false positive sets
    a set is a query, its best match, and a score
    '''
    #if not 'res' in vars():
        #res = qcx2_res[qcx]
    indx_samp = hs.indexed_sample_cx
    qcx = res.qcx
    cx2_score = res.cx2_score
    unfilt_top_cx = np.argsort(cx2_score)[::-1]
    # Get top chip indexes and scores
    top_cx    = np.array(utool.intersect_ordered(unfilt_top_cx, indx_samp))
    top_score = cx2_score[top_cx]
    # Get the true and false ground truth ranks
    qnx         = hs.tables.cx2_nx[qcx]
    if qnx <= 1:
        qnx = -1  # disallow uniden animals from being marked as true
    top_nx      = hs.tables.cx2_nx[top_cx]
    true_ranks  = np.where(np.logical_and(top_nx == qnx, top_cx != qcx))[0]
    false_ranks = np.where(np.logical_and(top_nx != qnx, top_cx != qcx))[0]
    # Construct the true positive tuple
    true_scores  = top_score[true_ranks]
    true_cxs     = top_cx[true_ranks]
    true_tup     = (true_cxs, true_scores, true_ranks)
    # Construct the false positive tuple
    false_scores = top_score[false_ranks]
    false_cxs    = top_cx[false_ranks]
    false_tup    = (false_cxs, false_scores, false_ranks)
    # Return tuples
    return true_tup, false_tup


def init_organized_results(allres):
    print('[rr2] init_organized_results()')
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    allres.true          = OrganizedResult()
    allres.false         = OrganizedResult()
    allres.top_true      = OrganizedResult()
    allres.top_false     = OrganizedResult()
    allres.bot_true      = OrganizedResult()
    allres.problem_true  = OrganizedResult()
    allres.problem_false = OrganizedResult()
    # -----------------
    # Query result loop

    def _organize_result(res):
        # Use ground truth to sort into true/false
        true_tup, false_tup = res2_true_and_false(hs, res)
        last_rank     = -1
        skipped_ranks = set([])
        # Record: all_true, missed_true, top_true, bot_true
        topx = 0
        for cx, score, rank in zip(*true_tup):
            allres.true.append(qcx, cx, rank, score)
            if rank - last_rank > 1:
                skipped_ranks.add(rank - 1)
                allres.problem_true.append(qcx, cx, rank, score)
            if topx == 0:
                allres.top_true.append(qcx, cx, rank, score)
            last_rank = rank
            topx += 1
        if topx > 1:
            allres.bot_true.append(qcx, cx, rank, score)
        # Record the all_false, false_positive, top_false
        topx = 0
        for cx, score, rank in zip(*false_tup):
            allres.false.append(qcx, cx, rank, score)
            if rank in skipped_ranks:
                allres.problem_false.append(qcx, cx, rank, score)
            if topx == 0:
                allres.top_false.append(qcx, cx, rank, score)
            topx += 1

    for qcx in allres.qcx_list:
        res = qcx2_res[qcx]
        if res is not None:
            _organize_result(res)
    #print('[rr2] len(allres.true)          = %r' % len(allres.true))
    #print('[rr2] len(allres.false)         = %r' % len(allres.false))
    #print('[rr2] len(allres.top_true)      = %r' % len(allres.top_true))
    #print('[rr2] len(allres.top_false)     = %r' % len(allres.top_false))
    #print('[rr2] len(allres.bot_true)      = %r' % len(allres.bot_true))
    #print('[rr2] len(allres.problem_true)  = %r' % len(allres.problem_true))
    #print('[rr2] len(allres.problem_false) = %r' % len(allres.problem_false))
    # qcx arrays for ttbttf
    allres.top_true_qcx_arrays  = allres.top_true.qcx_arrays(hs)
    allres.bot_true_qcx_arrays  = allres.bot_true.qcx_arrays(hs)
    allres.top_false_qcx_arrays = allres.top_false.qcx_arrays(hs)


def init_score_matrix(allres):
    print('[rr2] init score matrix')
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    qcx_list = allres.qcx_list
    nx_list = np.unique(hs.tables.cx2_nx[qcx_list])
    #nx_list = hs.get_valid_nxs(unknown=False)
    cxs_list = hs.nx2_cxs(nx_list, aslist=True)
    # Sort names by number of chips
    nx_size = map(len, cxs_list)
    # Build sorted chip list
    nx_cxs_tuples = zip(nx_size, cxs_list)
    # Sort by name
    cx_sorted = [x for (y, x) in sorted(nx_cxs_tuples)]
    # Subsort by chip
    cx_sorted = map(sorted, cx_sorted)
    cx_sorted = utool.flatten(cx_sorted)
    row_label_cx = []
    row_scores = []
    qcx_set = set(qcx_list)
    # Build each row in the score matrix
    for qcx in iter(cx_sorted):
        if not qcx in qcx_set:
            continue
        try:
            res = qcx2_res[qcx]
        except IndexError:
            print('qcx = %r' % qcx)
            print('len(qcx2_res) = %r' % len(qcx2_res))
            raise
        if res is None:
            continue
        # Append a label to score matrix
        row_label_cx.append(qcx)
        # Append a column to score matrix
        row_scores.append(res.cx2_score[cx_sorted])
    col_label_cx = cx_sorted
    # convert to numpy matrix array
    score_matrix = np.array(row_scores, dtype=np.float64)
    # Fill diagonal with -1's
    np.fill_diagonal(score_matrix, -np.ones(len(row_label_cx)))
    # Add score matrix to allres
    allres.score_matrix = score_matrix
    allres.col_label_cx = col_label_cx
    allres.row_label_cx = row_label_cx


def get_title_suffix(hs):
    title_suffix = hs.get_cache_uid()
    return title_suffix


def init_allres(hs, qcx2_res,
                qcx_list=None,
                matrix=(REPORT_MATRIX or REPORT_MATRIX_VIZ),
                oxford=False,
                **kwargs):
    'Organizes results into a visualizable data structure'
    # Make AllResults data containter
    allres = AllResults(hs, qcx2_res, qcx_list)
    allres.title_suffix = get_title_suffix(hs)
    #utool.ensurepath(allres.summary_dir)
    print('[rr2] init_allres()')
    #---
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    #cx2_cid  = hs.tables.cx2_cid
    # Initialize
    if matrix:
        init_score_matrix(allres)
    init_organized_results(allres)
    # Build
    build_rankres_str(allres)
    if matrix:
        build_matrix_str(allres)
    if oxford is True:
        import oxsty_results
        oxsty_map_csv, scalar_mAP_str = oxsty_results.oxsty_mAP_results(allres)
        allres.scalar_mAP_str = scalar_mAP_str
        allres.oxsty_map_csv = oxsty_map_csv
    #print(allres)
    return allres


# ========================================================
# Build textfile result strings
# ========================================================


def build_matrix_str(allres):
    hs = allres.hs
    cx2_gx = hs.tables.cx2_gx
    gx2_gname = hs.tables.gx2_gname

    def cx2_gname(cx):
        return [os.path.splitext(gname)[0] for gname in gx2_gname[cx2_gx]]
    col_label_gname = cx2_gname(allres.col_label_cx)
    row_label_gname = cx2_gname(allres.row_label_cx)
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
    'Builds csv files showing the cxs/scores/ranks of the query results'
    hs = allres.hs
    #qcx2_res = allres.qcx2_res
    cx2_cid = hs.tables.cx2_cid
    #cx2_nx = hs.tables.cx2_nx
    test_samp = allres.qcx_list
    train_samp = hs.train_sample_cx
    indx_samp = hs.indexed_sample_cx
    # Get organized data for csv file
    (qcx2_top_true_rank,
     qcx2_top_true_score,
     qcx2_top_true_cx)  = allres.top_true_qcx_arrays

    (qcx2_bot_true_rank,
     qcx2_bot_true_score,
     qcx2_bot_true_cx)  = allres.bot_true_qcx_arrays

    (qcx2_top_false_rank,
     qcx2_top_false_score,
     qcx2_top_false_cx) = allres.top_false_qcx_arrays
    # Number of groundtruth per query
    qcx2_numgt = np.zeros(len(cx2_cid)) - 2
    for qcx in test_samp:
        qcx2_numgt[qcx] = len(hs.get_other_indexed_cxs(qcx))
    # Easy to digest results
    num_chips = len(test_samp)
    num_nonquery = len(np.setdiff1d(indx_samp, test_samp))
    # Find the test samples WITH ground truth
    test_samp_with_gt = np.array(test_samp)[qcx2_numgt[test_samp] > 0]
    if len(test_samp_with_gt) == 0:
        warnings.warn('[rr2] there were no queries with ground truth')
    #train_nxs_set = set(cx2_nx[train_samp])
    flag_cxs_fn = hs.flag_cxs_with_name_in_sample

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
        greater_cxs = test_cxs_[np.where(testcx2_ttr >= thresh)[0]]
        num_greater = len(greater_cxs)
        num_less    = num_with_gt - num_greater
        num_greater = num_with_gt - num_less
        frac_less   = 100.0 * num_less / num_with_gt
        fmt_tup     = (num_less, num_with_gt, frac_less, num_greater)
        return greater_cxs, fmt_tup

    greater5_cxs, fmt5_tup = ranks_less_than_(5)
    greater1_cxs, fmt1_tup = ranks_less_than_(1)
    #
    gt5_intrain_cxs, fmt5_in_tup = ranks_less_than_(5, intrain=True)
    gt1_intrain_cxs, fmt1_in_tup = ranks_less_than_(1, intrain=True)
    #
    gt5_outtrain_cxs, fmt5_out_tup = ranks_less_than_(5, intrain=False)
    gt1_outtrain_cxs, fmt1_out_tup = ranks_less_than_(1, intrain=False)
    #
    allres.greater1_cxs = greater1_cxs
    allres.greater5_cxs = greater5_cxs
    #print('greater5_cxs = %r ' % (allres.greater5_cxs,))
    #print('greater1_cxs = %r ' % (allres.greater1_cxs,))
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
    test_sample_gx = hs.tables.cx2_gx[test_samp]
    test_sample_gname = hs.tables.gx2_gname[test_sample_gx]
    test_sample_gname = [g.replace('.jpg', '') for g in test_sample_gname]
    column_labels = ['QCX', 'NUM GT',
                     'TT CX', 'BT CX', 'TF CX',
                     'TT SCORE', 'BT SCORE', 'TF SCORE',
                     'TT RANK', 'BT RANK', 'TF RANK',
                     'QGNAME', ]
    column_list = [
        test_samp, qcx2_numgt[test_samp],
        qcx2_top_true_cx[test_samp], qcx2_bot_true_cx[test_samp],
        qcx2_top_false_cx[test_samp], qcx2_top_true_score[test_samp],
        qcx2_bot_true_score[test_samp], qcx2_top_false_score[test_samp],
        qcx2_top_true_rank[test_samp], qcx2_bot_true_rank[test_samp],
        qcx2_top_false_rank[test_samp], test_sample_gname, ]
    column_type = [int, int, int, int, int,
                   float, float, float, int, int, int, str, ]
    rankres_str = csvtool.make_csv_table(column_labels, column_list, header, column_type)
    # Put some more data at the end
    problem_true_pairs = zip(allres.problem_true.qcxs, allres.problem_true.cxs)
    problem_false_pairs = zip(allres.problem_false.qcxs, allres.problem_false.cxs)
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
    result_dir    = allres.hs.dirs.result_dir
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
    qcx2_res = allres.qcx2_res
    for qcx in xrange(0, len(qcx2_res)):
        viz.show_chip(allres, qcx, 'gt_matches')


def dump_missed_top5(allres):
    #print('\n---DUMPING MISSED TOP 5---')
    'Displays the top5 matches for all queries'
    greater5_cxs = allres.greater5_cxs
    #qcx = greater5_cxs[0]
    for qcx in greater5_cxs:
        viz.show_chip(allres, qcx, 'top5', 'missed_top5')
        viz.show_chip(allres, qcx, 'gt_matches', 'missed_top5')


def dump_analysis(allres):
    print('[rr2] dump analysis')
    greater1_cxs = allres.greater1_cxs
    #qcx = greater5_cxs[0]
    for qcx in greater1_cxs:
        viz.show_chip(allres, qcx, 'analysis', 'analysis')
        viz.show_chip(allres, qcx, 'analysis', 'analysis', annotations=False, title_aug=' noanote')


def dump_all_queries2(hs):
    import QueryResult as qr
    test_cxs = hs.test_sample_cx
    title_suffix = get_title_suffix(hs)
    print('[rr2] dumping all %r queries' % len(test_cxs))
    for qcx in test_cxs:
        res = qr.QueryResult(qcx)
        res.load(hs)
        # SUPER HACK (I don't know the figurename a priori, I have to contstruct
        # it to not duplciate dumping a figure)
        title_aug = ' noanote'
        fpath = hs.dirs.result_dir
        subdir = 'allqueries'
        N = 5
        topN_cxs = res.topN_cxs(N)
        topscore = res.cx2_score[topN_cxs][0]

        dump_dir = join(fpath, subdir + title_suffix)

        fpath     = join(dump_dir, ('topscore=%r -- qcid=%r' % (topscore, res.qcid)))
        fpath_aug = join(dump_dir, ('topscore=%r -- qcid=%r' % (topscore, res.qcid))) + title_aug

        fpath_clean = df2.sanatize_img_fpath(fpath)
        fpath_aug_clean = df2.sanatize_img_fpath(fpath_aug)
        print('----')
        print(fpath_clean)
        print(fpath_clean)
        if not exists(fpath_aug_clean):
            viz.plot_cx2(hs, res, 'analysis', subdir=subdir, annotations=False, title_aug=title_aug)
        if not exists(fpath_clean):
            viz.plot_cx2(hs, res, 'analysis', subdir=subdir)
        print('----')


def dump_all_queries(allres):
    test_cxs = allres.qcx_list
    print('[rr2] dumping all %r queries' % len(test_cxs))
    for qcx in test_cxs:
        viz.show_chip(allres, qcx, 'analysis', subdir='allqueries',
                      annotations=False, title_aug=' noanote')
        viz.show_chip(allres, qcx, 'analysis', subdir='allqueries')


def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    # loop over each query / result of interest
    for qcx, cx, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[cx]])
        res = qcx2_res[qcx]
        df2.figure(fnum=1, plotnum=121)
        df2.show_matches_annote_res(res, hs, cx, fnum=1, plotnum=121)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % (score, rank, query_gname, result_gname)
        df2.set_figtitle(big_title)
        viz.__dump_or_browse(allres, orgres_type + '_matches' + allres.title_suffix)


def dump_feature_pair_analysis(allres):
    print('[rr2] Doing: feature pair analysis')
    # TODO: Measure score consistency over a spatial area.
    # Measures entropy of matching vs nonmatching descriptors
    # Measures scale of m vs nm desc
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    import scipy

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
    if hs.feats.cx2_desc.size == 0:
        print(' * forcing load of descriptors')
        hs.load_features()
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts

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
        for ix, (qcx, cx, score, rank) in enumerate(orgres.iter()):
            utool.print_(fmt_str % (ix + 1,))
            # Skip low ranks
            if rank > 5:
                rank_skips.append(qcx)
                continue
            other_cxs = hs.get_other_indexed_cxs(qcx)
            # Skip no groundtruth
            if len(other_cxs) == 0:
                gt_skips.append(qcx)
                continue
            res = qcx2_res[qcx]
            # Get matching feature indexes
            fm = res.cx2_fm[cx]
            # Get their scores
            fs = res.cx2_fs[cx]
            # Get matching descriptors
            printDBG('\nfm.shape=%r' % (fm.shape,))
            desc1 = cx2_desc[qcx][fm[:, 0]]
            desc2 = cx2_desc[cx][fm[:, 1]]
            # Get matching keypoints
            kpts1 = cx2_kpts[qcx][fm[:, 0]]
            kpts2 = cx2_kpts[cx][fm[:, 1]]
            # Get their scale
            scale1_m = sv2.keypoint_scale(kpts1)
            scale2_m = sv2.keypoint_scale(kpts2)
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

    #df2.rrr(); viz.rrr(); clf(); df2.show_chip(hs, 14, allres=allres)
    #viz.show_chip(allres, 14, 'top5')
    #viz.show_chip(allres, 14, 'gt_matches')
    #df2.show_chip(hs, 1, allres=allres)


def possible_problems():
    # Perhaps overlapping keypoints are causing more harm than good.
    # Maybe there is a way of grouping them or averaging them into a
    # better descriptor.
    pass


#===============================
# MAIN SCRIPT
#===============================


def report_all(hs, qcx2_res, qcx_list, **kwargs):
    allres = init_allres(hs, qcx2_res, qcx_list=qcx_list, **kwargs)
    #if not 'kwargs' in vars():
        #kwargs = dict(rankres=True, stem=False, matrix=False, pdf=False,
                      #hist=False, oxford=False, ttbttf=False, problems=False,
                      #gtmatches=False)
    try:
        dump_all(allres, **kwargs)
    except Exception as ex:
        import traceback
        print('\n\n-----------------')
        print('report_all(hs, qcx2_res, **kwargs=%r' % (kwargs))
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
    qcxs = allres[orgtype_].qcxs
    cxs  = allres[orgtype_].cxs
    match_list = zip(qcxs, cxs)
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
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    # FIXME: More intelligent feature loading
    if len(hs.feats.cx2_desc) == 0:
        hs.refresh_features()
    cx2_desc = hs.feats.cx2_desc
    desc1_list = []
    desc2_list = []
    desc1_append = desc1_list.append
    desc2_append = desc2_list.append
    for qcx, cx in match_list:
        fx2_desc1 = cx2_desc[qcx]
        fx2_desc2 = cx2_desc[cx]
        res = qcx2_res[qcx]

        fm = res.cx2_fm[cx]
        #fs = res.cx2_fs[cx]
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


def load_qcx2_res(hs, qcx_list, nocache=False):
    'Prefrosm / loads all queries'
    import match_chips3 as mc3
    qreq = mc3.quickly_ensure_qreq(hs, qcxs=qcx_list)
    # Build query big cache uid
    query_uid = qreq.get_uid()
    hs_uid    = hs.get_db_name()
    qcxs_uid  = utool.hashstr_arr(qcx_list, lbl='_qcxs')
    qres_uid  = hs_uid + query_uid + qcxs_uid
    cache_dir = join(hs.dirs.cache_dir, 'query_results_bigcache')
    print('[rr2] load_qcx2_res(): %r' % qres_uid)
    io_kwargs = dict(dpath=cache_dir, fname='query_results', uid=qres_uid, ext='.cPkl')
    # Return cache if available
    if not params.args.nocache_query and (not nocache):
        qcx2_res = io.smart_load(**io_kwargs)
        if qcx2_res is not None:
            print('[rr2]  *  cache hit')
            return qcx2_res
        print('[rr2]  *  cache miss')
    else:
        print('[rr2]  *  cache off')
    # Individually load / compute queries
    if isinstance(qcx_list, list):
        qcx_set = set(qcx_list)
    else:
        qcx_set = set(qcx_list.tolist())
    qcx_max = max(qcx_list) + 1
    qcx2_res = [hs.query(qcx) if qcx in qcx_set else None for qcx in xrange(qcx_max)]
    # Save to the cache
    print('[rr2] Saving query_results to bigcache: %r' % qres_uid)
    utool.ensuredir(cache_dir)
    io.smart_save(qcx2_res, **io_kwargs)
    return qcx2_res


def get_allres(hs, qcx_list):
    'Performs / Loads all queries and build allres structure'
    print('[rr2] get_allres()')
    #valid_cxs = hs.get_valid_cxs()
    qcx2_res = load_qcx2_res(hs, qcx_list)
    allres = init_allres(hs, qcx2_res, qcx_list=qcx_list)
    return allres
