#flake8:noqa
from __future__ import absolute_import, division, print_function
from os.path import join
from plottool import draw_func2 as df2
import numpy as np
import os
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[viz_allres]', DEBUG=False)
# Global variables
BROWSE = True
DUMP = False
FIGNUM = 1


def plot_rank_stem(allres, orgres_type='true'):
    print('[viz] plotting rank stem')
    # Visualize rankings with the stem plot
    ibs = allres.ibs
    title = orgres_type + 'rankings stem plot\n' + allres.title_suffix
    orgres = allres.__dict__[orgres_type]
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    x_data = orgres.qcxs
    y_data = orgres.ranks
    df2.draw_stems(x_data, y_data)
    slice_num = int(np.ceil(np.log10(len(orgres.qcxs))))
    df2.set_xticks(ibs.test_sample_cx[::slice_num])
    df2.set_xlabel('query chip indeX (qcx)')
    df2.set_ylabel('groundtruth chip ranks')
    #df2.set_yticks(list(seen_ranks))
    __dump_or_browse(allres.ibs, 'rankviz')


def plot_rank_histogram(allres, orgres_type):
    print('[viz] plotting %r rank histogram' % orgres_type)
    ranks = allres.__dict__[orgres_type].ranks
    label = 'P(rank | ' + orgres_type + ' match)'
    title = orgres_type + ' match rankings histogram\n' + allres.title_suffix
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    df2.draw_histpdf(ranks, label=label)  # FIXME
    df2.set_xlabel('ground truth ranks')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.ibs, 'rankviz')


def plot_score_pdf(allres, orgres_type, colorx=0.0, variation_truncate=False):
    print('[viz] plotting ' + orgres_type + ' score pdf')
    title  = orgres_type + ' match score frequencies\n' + allres.title_suffix
    scores = allres.__dict__[orgres_type].scores
    print('[viz] len(scores) = %r ' % (len(scores),))
    label  = 'P(score | %r)' % orgres_type
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    df2.draw_pdf(scores, label=label, colorx=colorx)
    if variation_truncate:
        df2.variation_trunctate(scores)
    #df2.variation_trunctate(false.scores)
    df2.set_xlabel('score')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.ibs, 'scoreviz')


def plot_score_matrix(allres):
    print('[viz] plotting score matrix')
    score_matrix = allres.score_matrix
    title = 'Score Matrix\n' + allres.title_suffix
    # Find inliers
    #inliers = util.find_std_inliers(score_matrix)
    #max_inlier = score_matrix[inliers].max()
    # Trunate above 255
    score_img = np.copy(score_matrix)
    #score_img[score_img < 0] = 0
    #score_img[score_img > 255] = 255
    #dim = 0
    #score_img = util.norm_zero_one(score_img, dim=dim)
    # Make colors
    scores = score_img.flatten()
    colors = df2.scores_to_color(scores, logscale=True)
    cmap = df2.scores_to_cmap(scores, colors)
    df2.figure(fnum=FIGNUM, doclf=True, title=title)
    # Show score matrix
    df2.imshow(score_img, fnum=FIGNUM, cmap=cmap)
    # Colorbar
    df2.colorbar(scores, colors)
    df2.set_xlabel('database')
    df2.set_ylabel('queries')
    #__dump_or_browse(allres.ibs, 'scoreviz')


# Dump logic
def __browse():
    print('[viz] Browsing Image')
    df2.show()


def save_if_requested(ibs, subdir):
    if not ibs.args.save_figures:
        return
    #print('[viz] Dumping Image')
    fpath = ibs.dirs.result_dir
    if not subdir is None:
        subdir = utool.sanatize_fname2(subdir)
        fpath = join(fpath, subdir)
        utool.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()


def __dump_or_browse(ibs, subdir=None):
    #fig = df2.plt.gcf()
    #fig.tight_layout()
    if BROWSE:
        __browse()
    if DUMP:
        dump(ibs, subdir)


def plot_tt_bt_tf_matches(ibs, allres, qcx):
    #print('Visualizing result: ')
    #res.printme()
    res = allres.qcx2_res[qcx]
    ranks = (allres.top_true_qcx_arrays[0][qcx],
             allres.bot_true_qcx_arrays[0][qcx],
             allres.top_false_qcx_arrays[0][qcx])
    #scores = (allres.top_true_qcx_arrays[1][qcx],
             #allres.bot_true_qcx_arrays[1][qcx],
             #allres.top_false_qcx_arrays[1][qcx])
    cxs = (allres.top_true_qcx_arrays[2][qcx],
           allres.bot_true_qcx_arrays[2][qcx],
           allres.top_false_qcx_arrays[2][qcx])
    titles = ('best True rank=' + str(ranks[0]) + ' ',
              'worst True rank=' + str(ranks[1]) + ' ',
              'best False rank=' + str(ranks[2]) + ' ')
    df2.figure(fnum=1, pnum=231)
    res.plot_matches(res, ibs, cxs[0], False, fnum=1, pnum=131, title_aug=titles[0])
    res.plot_matches(res, ibs, cxs[1], False, fnum=1, pnum=132, title_aug=titles[1])
    res.plot_matches(res, ibs, cxs[2], False, fnum=1, pnum=133, title_aug=titles[2])
    fig_title = 'fig q' + ibs.cidstr(qcx) + ' TT BT TF -- ' + allres.title_suffix
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)


def dump_gt_matches(allres):
    ibs = allres.ibs
    qcx2_res = allres.qcx2_res
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        res = qcx2_res[qcx]
        res.show_gt_matches(ibs, fnum=FIGNUM)
        __dump_or_browse(allres.ibs, 'gt_matches' + allres.title_suffix)


def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    ibs = allres.ibs
    qcx2_res = allres.qcx2_res
    # loop over each query / result of interest
    for qcx, cx, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(ibs.tables.gx2_gname[ibs.tables.cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(ibs.tables.gx2_gname[ibs.tables.cx2_gx[cx]])
        res = qcx2_res[qcx]
        df2.figure(fnum=FIGNUM, pnum=121)
        df2.show_matches3(res, ibs, cx, SV=False, fnum=FIGNUM, pnum=121)
        df2.show_matches3(res, ibs, cx, SV=True,  fnum=FIGNUM, pnum=122)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % (score, rank, query_gname,
                                                      result_gname)
        df2.set_figtitle(big_title)
        __dump_or_browse(allres.ibs, orgres_type + '_matches' + allres.title_suffix)


@profile
def show_descriptors_match_distances(orgres2_distance, fnum=1, db_name='', **kwargs):
    disttype_list = orgres2_distance.itervalues().next().keys()
    orgtype_list = orgres2_distance.keys()
    (nRow, nCol) = len(orgtype_list), len(disttype_list)
    nColors = nRow * nCol
    color_list = df2.distinct_colors(nColors)
    df2.figure(fnum=fnum, docla=True, doclf=True)
    pnum_ = lambda px: (nRow, nCol, px + 1)
    plot_type = utool.get_arg('--plot-type', default='plot')

    # Remember min and max val for each distance type (l1, emd...)
    distkey2_min = {distkey: np.uint64(-1) for distkey in disttype_list}
    distkey2_max = {distkey: 0 for distkey in disttype_list}

    def _distplot(dists, color, label, distkey, plot_type=plot_type):
        data = sorted(dists)
        ax = df2.gca()
        min_ = distkey2_min[distkey]
        max_ = distkey2_max[distkey]
        if plot_type == 'plot':
            df2.plot(data, color=color, label=label, yscale='linear')
            #xticks = np.linspace(np.min(data), np.max(data), 3)
            #yticks = np.linspace(0, len(data), 5)
            #ax.set_xticks(xticks)
            #ax.set_yticks(yticks)
            ax.set_ylim(min_, max_)
            ax.set_xlim(0, len(dists))
            ax.set_ylabel('distance')
            ax.set_xlabel('matches indexes (sorted by distance)')
            df2.legend(loc='lower right')
        if plot_type == 'pdf':
            df2.plot_pdf(data, color=color, label=label)
            ax.set_ylabel('pr')
            ax.set_xlabel('distance')
            ax.set_xlim(min_, max_)
            df2.legend(loc='upper left')
        df2.dark_background(ax)
        df2.small_xticks(ax)
        df2.small_yticks(ax)

    px = 0
    for orgkey in orgtype_list:
        for distkey in disttype_list:
            dists = orgres2_distance[orgkey][distkey]
            if len(dists) == 0:
                continue
            min_ = dists.min()
            max_ = dists.max()
            distkey2_min[distkey] = min(distkey2_min[distkey], min_)
            distkey2_max[distkey] = max(distkey2_max[distkey], max_)

    for count, orgkey in enumerate(orgtype_list):
        for distkey in disttype_list:
            printDBG('[allres-viz] plotting: %r' % ((orgkey, distkey),))
            dists = orgres2_distance[orgkey][distkey]
            df2.figure(fnum=fnum, pnum=pnum_(px))
            color = color_list[px]
            title = distkey + ' ' + orgkey
            label = 'P(%s | %s)' % (distkey, orgkey)
            _distplot(dists, color, label, distkey, **kwargs)
            if count == 0:
                ax = df2.gca()
                ax.set_title(distkey)
            px += 1

    subtitle = 'the matching distances between sift descriptors'
    title = '(sift) matching distances'
    if db_name != '':
        title = db_name + ' ' + title
    df2.set_figtitle(title, subtitle)
    df2.adjust_subplots_safe()
