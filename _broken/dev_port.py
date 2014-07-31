# flake8: noqa
from __future__ import absolute_import, division, print_function
import numpy as np
from plottool import draw_func2 as df2
from vtool import keypoint as ktool
from utool import util_latex
from ibeis.model import Config
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[devport]', DEBUG=False)


def plot_keypoint_scales(hs, fnum=1):
    print('[dev] plot_keypoint_scales()')
    cx2_kpts = hs.feats.cx2_kpts
    if len(cx2_kpts) == 0:
        hs.refresh_features()
        cx2_kpts = hs.feats.cx2_kpts
    cx2_nFeats = map(len, cx2_kpts)
    kpts = np.vstack(cx2_kpts)
    print('[dev] --- LaTeX --- ')
    _printopts = np.get_printoptions()
    np.set_printoptions(precision=3)
    print(util_latex.latex_scalar(r'\# keypoints, ', len(kpts)))
    print(util_latex.latex_mystats(r'\# keypoints per image', cx2_nFeats))
    scales = ktool.get_scales(kpts)
    scales = np.array(sorted(scales))
    print(util_latex.latex_mystats(r'keypoint scale', scales))
    np.set_printoptions(**_printopts)
    print('[dev] ---/LaTeX --- ')
    #
    df2.figure(fnum=fnum, docla=True, title='sorted scales')
    df2.plot(scales)
    df2.adjust_subplots_safe()
    #ax = df2.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #
    fnum += 1
    df2.figure(fnum=fnum, docla=True, title='hist scales')
    df2.show_histogram(scales, bins=20)
    df2.adjust_subplots_safe()
    #ax = df2.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    return fnum


def investigate_vsone_groundtruth(hs, qcx_list, fnum=1):
    print('--------------------------------------')
    print('[dev] investigate_vsone_groundtruth')
    query_cfg = Config.get_vsone_cfg(sv_on=True, ratio_thresh=1.5)
    for qcx in qcx_list:
        res = hs.query_groundtruth(hs, qcx, query_cfg)
        #print(query_cfg)
        #print(res)
        #res.show_query(hs, fnum=fnum)
        fnum += 1
        res.show_topN(hs, fnum=fnum, query_cfg=query_cfg)
        fnum += 1
    return fnum


def plot_feature_distances(allres, orgres_list=None, fnum=1):
    print('[dev] plot_feature_distances()')
    orgres2_distance = allres.get_orgres2_distances(orgres_list=orgres_list)
    db_name = allres.hs.get_db_name()
    allres_viz.show_descriptors_match_distances(orgres2_distance,
                                                db_name=db_name, fnum=fnum)
    fnum += 1
    return fnum


YSCALE = utool.get_arg('--yscale', default='symlog')  # 'symlog'
XSCALE = 'linear'


def plot_seperability(hs, qcx_list, fnum=1):
    print('[dev] plot_seperability(fnum=%r)' % fnum)
    qcx2_res = get_qcx2_res(hs, qcx_list)
    qcx2_separability = get_seperatbility(hs, qcx2_res)
    sep_score_list = qcx2_separability.values()
    df2.figure(fnum=fnum, doclf=True, docla=True)
    print('[dev] seperability stats: ' + utool.stats_str(sep_score_list))
    sorted_sepscores = sorted(sep_score_list)
    df2.plot(sorted_sepscores, color=df2.DEEP_PINK, label='seperation score',
             yscale=YSCALE)
    df2.set_xlabel('true chipmatch index (%d)' % len(sep_score_list))
    df2.set_logyscale_from_data(sorted_sepscores)
    df2.dark_background()
    rowid = qcx2_res.itervalues().next().rowid
    df2.set_figtitle('seperability\n' + rowid)
    df2.legend()
    fnum += 1
    return fnum


def plot_scores2(hs, qcx_list, fnum=1):
    print('[dev] plot_scores(fnum=%r)' % fnum)
    qcx2_res = get_qcx2_res(hs, qcx_list)
    all_score_list = []
    gtscore_ys = []
    gtscore_xs = []
    gtscore_ranks = []
    EXCLUDE_ZEROS = False
    N = 1
    # Append all scores to a giant list
    for res in qcx2_res.itervalues():
        cx2_score = res.cx2_score
        # Get gt scores first
        #gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        gt_cxs = np.array(res.topN_cxs(hs, N=N, only_gt=True))
        gt_ys = cx2_score[gt_cxs]
        if EXCLUDE_ZEROS:
            nonzero_cxs = np.where(cx2_score != 0)[0]
            gt_cxs = gt_cxs[gt_ys != 0]
            gt_ranks = res.get_gt_ranks(gt_cxs)
            gt_cxs = np.array(utool.list_index(nonzero_cxs, gt_cxs))
            gt_ys  = gt_ys[gt_ys != 0]
            score_list = cx2_score[nonzero_cxs].tolist()
        else:
            score_list = cx2_score.tolist()
            gt_ranks = res.get_gt_ranks(gt_cxs)
        gtscore_ys.extend(gt_ys)
        gtscore_xs.extend(gt_cxs + len(all_score_list))
        gtscore_ranks.extend(gt_ranks)
        # Append all scores
        all_score_list.extend(score_list)
    all_score_list = np.array(all_score_list)
    gtscore_ranks = np.array(gtscore_ranks)
    gtscore_ys = np.array(gtscore_ys)

    # Sort all chipmatch scores
    allx_sorted = all_score_list.argsort()  # mapping from sortedallx to allx
    allscores_sorted = all_score_list[allx_sorted]
    # Change the groundtruth positions to correspond to sorted cmatch scores
    # Find position of gtscore_xs in allx_sorted
    gtscore_sortxs = utool.list_index(allx_sorted, gtscore_xs)
    gtscore_sortxs = np.array(gtscore_sortxs)
    # Draw and info
    rank_bounds = [
        (0, 1),
        (1, 5),
        (5, None)
    ]
    rank_colors = [
        df2.TRUE_GREEN,
        df2.UNKNOWN_PURP,
        df2.FALSE_RED
    ]
    print('[dev] matching chipscore stats: ' + utool.stats_str(all_score_list))
    df2.figure(fnum=fnum, doclf=True, docla=True)
    # Finds the knee
    df2.plot(allscores_sorted, color=df2.ORANGE, label='all scores')

    # get positions which are within rank bounds
    for count, ((low, high), rankX_color) in reversed(list(enumerate(zip(rank_bounds, rank_colors)))):
        rankX_flag_low = gtscore_ranks >= low
        if high is not None:
            rankX_flag_high = gtscore_ranks < high
            rankX_flag = np.logical_and(rankX_flag_low, rankX_flag_high)
        else:
            rankX_flag = rankX_flag_low
        rankX_allgtx = np.where(rankX_flag)[0]
        rankX_gtxs = gtscore_sortxs[rankX_allgtx]
        rankX_gtys = gtscore_ys[rankX_allgtx]
        rankX_label = '%d <= gt rank' % low
        if high is not None:
            rankX_label += ' < %d' % high
        if len(rankX_gtxs) > 0:
            df2.plot(rankX_gtxs, rankX_gtys, 'o', color=rankX_color, label=rankX_label)

    rowid = qcx2_res.itervalues().next().rowid

    df2.set_logyscale_from_data(allscores_sorted)

    df2.set_xlabel('chipmatch index')
    df2.dark_background()
    df2.set_figtitle('matching scores\n' + rowid)
    df2.legend(loc='upper left')
    #xmin = 0
    #xmax = utool.order_of_magnitude_ceil(len(allscores_sorted))
    #print('len_ = %r' % (len(allscores_sorted),))
    #print('xmin = %r' % (xmin,))
    #print('xmax = %r' % (xmax,))
    #df2.gca().set_xlim(xmin, xmax)
    df2.update()

    #utool.embed()
    # Second Plot
    #data = sorted(zip(gtscore_sortxs, gtscore_ys, gtscore_ranks))
    #gtxs = [x for (x, y, z) in data]
    #gtys = [y for (x, y, z) in data]
    #gtrs = [z for (x, y, z) in data]
    #nongtxs = np.setdiff1d(np.arange(gtxs[0], gtxs[-1]), gtxs)

    #min_ = min(gtxs)
    #max_ = len(allscores_sorted)
    #len_ = max_ - min_
    #normsum = 0
    #ratsum = 0
    #gtxs = np.array(gtxs)
    #nongtxs = np.array(nongtxs)
    #for ix in xrange(min_, max_):
        #nongtxs_ = nongtxs[nongtxs >= ix]
        #gtxs_ = gtxs[gtxs >= ix]
        #numer = allscores_sorted[gtxs_].sum()
        #denom = allscores_sorted[nongtxs_].sum()
        #ratio = (1 + numer) / (denom + 1)
        #total_support = (len(gtxs_) + len(nongtxs_))
        #normrat = ratio / total_support
        #ratsum += ratio
        #normsum += normrat
        #print(total_support)
    #print(ratsum / len_)
    #print(normsum / len_)
    #print(ratsum / allscores_sorted[min_:max_].sum())
    #print(normsum / allscores_sorted[min_:max_].sum())

    #index_gap = np.diff(gtxs)
    #score_gap = np.diff(gtys)
    #badness = (index_gap - 1) * score_gap
    #np.arange(len(gtxs))

    fnum += 1
    return fnum


def plot_scores(hs, qcx_list, fnum=1):
    print('[dev] plot_scores(fnum=%r)' % fnum)
    topN_gt    = 1
    topN_ranks = 3
    qcx2_res = get_qcx2_res(hs, qcx_list)
    data_scores = []  # matching scores
    data_qpairs = []  # query info (qcx, cx)
    data_gtranks = []  # query info (gtrank)
    # Append all scores to a giant list
    for res in qcx2_res.itervalues():
        # Get gt scores first
        qcx = res.qcx
        top_cxs = np.array(res.topN_cxs(hs, N=topN_ranks, only_gt=False))
        gt_cxs  = np.array(res.topN_cxs(hs, N=topN_gt, only_gt=True))
        top_scores = res.cx2_score[top_cxs]
        np.intersect1d(top_cxs, gt_cxs)
        # list of ranks if score is ground truth otherwise -1
        isgt_ranks = [tx if cx in set(gt_cxs) else -1 for (tx, cx) in enumerate(top_cxs)]
        qcx_pairs  = [(hs.cx2_cid(qcx), hs.cx2_cid(cx)) for cx in top_cxs]
        # Append all scores
        data_scores.extend(top_scores.tolist())
        data_qpairs.extend(qcx_pairs)
        data_gtranks.extend(isgt_ranks)
    data_scores = np.array(data_scores)
    data_qpairs = np.array(data_qpairs)
    data_gtranks = np.array(data_gtranks)

    data_sortx = data_scores.argsort()
    sorted_scores = data_scores[data_sortx]

    # Draw and info
    rank_colorbounds = [
        ((-1, 0), df2.GRAY),
        ((0, 1), df2.TRUE_GREEN),
        ((1, 5), df2.UNKNOWN_PURP),
        ((5, None), df2.FALSE_RED),
    ]
    print('[dev] matching chipscore stats: ' + utool.stats_str(data_scores))
    df2.figure(fnum=fnum, doclf=True, docla=True)
    # Finds the knee
    df2.plot(sorted_scores, color=df2.ORANGE, label='all scores')

    # Plot results with ranks within (low, high) bounds
    colorbounds_iter = reversed(list(enumerate(rank_colorbounds)))
    for count, ((low, high), rankX_color) in colorbounds_iter:
        datarank_flag = utool.inbounds(data_gtranks, low, high)
        rankX_xs = np.where(datarank_flag[data_sortx])[0]
        rankX_ys = sorted_scores[rankX_xs]
        if high is None:
            rankX_label = '%d <= gt rank' % low
        else:
            rankX_label = '%d <= gt rank < %d' % (low, high)
        if len(rankX_ys) > 0:
            df2.plot(rankX_xs, rankX_ys, 'o', color=rankX_color, label=rankX_label, alpha=.5)

    rowid = qcx2_res.itervalues().next().rowid

    df2.set_logyscale_from_data(data_scores)

    df2.set_xlabel('chipmatch index')
    df2.dark_background()
    df2.set_figtitle('matching scores\n' + rowid)
    df2.legend(loc='upper left')
    df2.iup()
    fnum += 1

    score_table = np.vstack((data_scores, data_gtranks, data_qpairs.T)).T
    score_table = score_table[data_sortx[::-1]]

    column_labels = ['score', 'gtrank', 'qcid', 'cid']
    header = 'score_table\nuid=%r' % rowid
    column_type = [float, int, int, int]
    csv_txt = utool.util_csv.numpy_to_csv(score_table,  column_labels, header, column_type)
    print(csv_txt)
    #utool.embed()
    return fnum


def get_seperatbility(hs, qcx2_res):
    qcx2_separability = {qcx: res.compute_seperability(hs) for qcx, res in qcx2_res.iteritems()}
    qcx2_separability = {qcx: sepscore for qcx, sepscore in qcx2_separability.iteritems() if sepscore is not None}
    return qcx2_separability
