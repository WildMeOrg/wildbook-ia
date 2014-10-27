#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
from ibeis.model.hots.smk import smk_debug
from vtool import patch as ptool
from vtool import image as gtool
import six
import scipy.stats.mstats as spms
from os.path import join
from os.path import basename
import scipy.spatial.distance as spdist
from collections import namedtuple
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[smk_plots]')


Metrics = namedtuple('Metrics', ('wx2_pdist', 'wx2_wdist', 'wx2_nMembers', 'wx2_pdist_stats', 'wx2_wdist_stats',))


def compute_word_metrics(invindex):
    invindex.idx2_wxs = np.array(invindex.idx2_wxs)
    wx2_idxs  = invindex.wx2_idxs
    idx2_dvec = invindex.idx2_dvec
    words     = invindex.words
    wx2_pdist = {}
    wx2_wdist = {}
    wx2_nMembers = {}
    wx2_pdist_stats = {}
    wx2_wdist_stats = {}
    wordidx_iter = ut.progiter(six.iteritems(wx2_idxs), lbl='Word Dists: ', num=len(wx2_idxs), freq=200)

    for _item in wordidx_iter:
        wx, idxs = _item
        dvecs = idx2_dvec.take(idxs, axis=0)
        word = words[wx:wx + 1]
        wx2_pdist[wx] = spdist.pdist(dvecs)  # pairwise dist between words
        wx2_wdist[wx] = euclidean_dist(dvecs, word)  # dist to word center
        wx2_nMembers[wx] = len(idxs)

    for wx, pdist in ut.progiter(six.iteritems(wx2_pdist), lbl='Word pdist Stats: ', num=len(wx2_idxs), freq=2000):
        wx2_pdist_stats[wx] = ut.get_stats(pdist)

    for wx, wdist in ut.progiter(six.iteritems(wx2_wdist), lbl='Word wdist Stats: ', num=len(wx2_idxs), freq=2000):
        wx2_wdist_stats[wx] = ut.get_stats(wdist)

    ut.print_stats(wx2_nMembers.values(), 'word members')
    metrics = Metrics(wx2_pdist, wx2_wdist, wx2_nMembers, wx2_pdist_stats, wx2_wdist_stats)
    return metrics
    #word_pdist = spdist.pdist(invindex.words)


def euclidean_dist(vecs1, vec2):
    return np.sqrt(((vecs1.astype(np.float32) - vec2.astype(np.float32)) ** 2).sum(1))


def dict_where0(dict_):
    keys = np.array(dict_.keys())
    flags = np.array(list(map(len, dict_.values()))) == 0
    indices = np.where(flags)[0]
    return keys[indices]


def get_cached_vocabs():
    import parse
    # Parse some of the training data from fname
    parse_str = '{}nC={num_cent},{}_DPTS(({num_dpts},{dim}){}'
    smkdir = ut.get_app_resource_dir('smk')
    fname_list = ut.glob(smkdir, 'akmeans*')
    fpath_list = [join(smkdir, fname) for fname in fname_list]
    result_list = [parse.parse(parse_str, fpath) for fpath in fpath_list]
    nCent_list = [int(res['num_cent']) for res in result_list]
    nDpts_list = [int(res['num_dpts']) for res in result_list]
    key_list = zip(nCent_list, nDpts_list)
    fpath_sorted = ut.sortedby(fpath_list, key_list, reverse=True)
    return fpath_sorted


def negative_minclamp(arr):
    arr[arr > 0] -= arr[arr > 0].min()
    arr[arr <= 0] = arr[arr > 0].min()
    return arr


def plot_chip_metric(ibs, aid, metric=None, fnum=1, lbl='', colortype='score',
                     darken=.5, **kwargs):
    """
    the word metric is used liberally
    Example:
        >>> from ibeis.model.hots.smk.smk_plots import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> from ibeis.model.hots.smk import smk_plots
        >>> from ibeis.model.hots.smk import smk_repr
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=8000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='PZ_Master0', nWords=64000)
        >>> tup = smk_debug.testdata_raw_internals0(db='PZ_Mothers', nWords=8000)
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = tup
        >>> smk_repr.compute_data_internals_(invindex, qreq_.qparams, delete_rawvecs=False)
        >>> invindex.idx2_wxs = np.array(invindex.idx2_wxs)
        >>> metric = None
        >>> aid = 1
        >>> fnum = 0
        >>> lbl='test'
        >>> colortype='scores'
        >>> kwargs = {'annote': False}
        #>>> df2.rrr()
        >>> smk_plots.plot_chip_metric(ibs, aid, metric, fnum, lbl, colortype, **kwargs)
        >>> df2.present()
    """
    import plottool.draw_func2 as df2
    from ibeis.viz import viz_chip
    df2.figure(fnum=fnum, doclf=True, docla=True)
    if metric is not None:
        if  colortype == 'score':
            colors = df2.scores_to_color(metric)
        elif colortype == 'label':
            colors = df2.label_to_colors(metric)
        else:
            raise ValueError('no known colortype = %r' % (colortype,))
    else:
        colors = 'distinct'
    viz_chip.show_chip(ibs, aid, color=colors, darken=darken, **kwargs)
    if metric is not None:
        cb = df2.colorbar(metric, colors)
        cb.set_label(lbl)
    df2.set_figtitle(lbl)


def viz_annot_with_metrics(ibs, invindex, aid, metrics, metric_keys=['mean', 'std', 'min', 'max']):
    """
    Args:
        ibs (IBEISController):
        invindex (InvertedIndex): object for fast vocab lookup
        aid (int):
        metrics (namedtuple):

    Example:
        >>> from ibeis.model.hots.smk.smk_plots import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> from ibeis.model.hots.smk import smk_repr
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=8000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='PZ_Master0', nWords=64000)
        >>> tup = smk_debug.testdata_raw_internals0(db='PZ_Mothers', nWords=8000)
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = tup
        >>> smk_repr.compute_data_internals_(invindex, qreq_.qparams, delete_rawvecs=False)
        >>> invindex.idx2_wxs = np.array(invindex.idx2_wxs)
        >>> metric_keys=['mean', 'std', 'min', 'max']
        >>> metrics = compute_word_metrics(invindex)
        >>> aid = 1

    """
    #viz_chip.rrr()
    #df2.rrr()
    kpts = ibs.get_annot_kpts(aid)
    if ut.VERBOSE:
        ut.super_print(kpts)
    invindex.idx2_wxs

    _mask = invindex.idx2_daid == aid
    fxs = invindex.idx2_dfx[_mask]
    wxs = invindex.idx2_wxs[_mask].T[0].T

    assert len(fxs) == len(kpts)
    assert len(fxs) == len(wxs)

    idf_list = np.array(list(ut.dict_take_gen(invindex.wx2_idf, wxs)))

    wx2_pdist_stats = metrics.wx2_pdist_stats
    wx2_wdist_stats = metrics.wx2_wdist_stats

    def wx2_pdiststat(wx, key='mean'):
        return wx2_pdist_stats[wx][key] if wx in wx2_pdist_stats and key in wx2_pdist_stats[wx] else -1

    def wx2_wdiststat(wx, key='mean'):
        return wx2_wdist_stats[wx][key] if wx in wx2_wdist_stats and key in wx2_wdist_stats[wx] else -1

    #wx2_idf = invindex.wx2_idf

    # data
    #idf_list =  [wx2_idf[wx] for wx in wxs]

    clamp = negative_minclamp

    avepdist_list =  clamp(np.array([wx2_pdiststat(wx, 'mean') for wx in wxs]))
    avewdist_list =  clamp(np.array([wx2_wdiststat(wx, 'mean') for wx in wxs]))
    #maxwdist_list =  clamp(np.array([wx2_wdiststat(wx, 'max') for wx in wxs]))
    #minwdist_list =  clamp(np.array([wx2_wdiststat(wx, 'min') for wx in wxs]))
    #stdwdist_list =  clamp(np.array([wx2_wdiststat(wx, 'std') for wx in wxs]))

    plot_chip_metric(ibs, aid, metric=None, fnum=1, lbl='', annote=False, darken=None)
    plot_chip_metric(ibs, aid, idf_list, fnum=2, lbl='idf', colortype='score')
    plot_chip_metric(ibs, aid, avepdist_list, fnum=3, lbl='avepdist_list', colortype='score')
    plot_chip_metric(ibs, aid, avewdist_list, fnum=4, lbl='avewdist_list', colortype='score')
    #plot_chip_metric(ibs, aid, wxs, fnum=3, lbl='wx', colortype='label')
    #plot_chip_metric(ibs, aid, stdwdist_list, fnum=5, lbl='stdwdist_list', colortype='score')
    #plot_chip_metric(ibs, aid, maxwdist_list, fnum=6, lbl='maxwdist_list', colortype='score')
    #plot_chip_metric(ibs, aid, minwdist_list, fnum=7, lbl='minwdist_list', colortype='score')


def draw_scatterplot(figdir, datax, datay, xlabel, ylabel, color, fnum=None):
    datac = [color for _ in range(len(datax))]
    assert len(datax) == len(datay), '%r %r' % (len(datax), len(datay))
    df2.figure(fnum=fnum, doclf=True, docla=True)
    df2.plt.scatter(datax, datay,  c=datac, s=20, marker='o', alpha=.9)
    ax = df2.gca()
    title = '%s vs %s.\nnWords=%r. db=%r' % (xlabel, ylabel, len(datax), ibs.get_dbname())
    df2.set_xlabel(xlabel)
    df2.set_ylabel(ylabel)
    ax.set_ylim(min(datay) - 1, max(datay) + 1)
    ax.set_xlim(min(datax) - 1, max(datax) + 1)
    df2.dark_background()
    df2.set_figtitle(title)
    figpath = join(figdir, title)
    df2.save_figure(fnum, figpath)


def make_scatterplots(figdir, invindex, metrics):
    from plottool import draw_func2 as df2
    wx2_pdist_stats = metrics.wx2_pdist_stats
    wx2_wdist_stats = metrics.wx2_pdist_stats
    wx2_nMembers = metrics.wx2_nMembers

    def wx2_avepdist(wx):
        return wx2_pdist_stats[wx]['mean'] if wx in wx2_pdist_stats and 'mean' in wx2_pdist_stats[wx] else -1
    def wx2_avewdist(wx):
        return wx2_wdist_stats[wx]['mean'] if wx in wx2_wdist_stats and 'mean' in wx2_wdist_stats[wx] else -1

    wx2_idf = invindex.wx2_idf

    # data
    wx_list  =  list(wx2_idf.keys())
    idf_list =  [wx2_idf[wx] for wx in wx_list]
    nPoints_list =  [wx2_nMembers[wx] if wx in wx2_nMembers else -1 for wx in wx_list]
    avepdist_list = [wx2_avepdist(wx) for wx in wx_list]
    avewdist_list = [wx2_avewdist(wx) for wx in wx_list]

    df2.reset()
    draw_scatterplot(figdir, idf_list, avepdist_list, 'idf', 'mean(pdist)', df2.WHITE, fnum=1)
    draw_scatterplot(figdir, idf_list, avewdist_list, 'idf', 'mean(wdist)', df2.PINK, fnum=3)
    draw_scatterplot(figdir, nPoints_list, avepdist_list, 'nPointsInWord', 'mean(pdist)', df2.GREEN, fnum=2)
    draw_scatterplot(figdir, avepdist_list, avewdist_list, 'mean(pdist)', 'mean(wdist)', df2.YELLOW, fnum=4)
    draw_scatterplot(figdir, nPoints_list, avewdist_list, 'nPointsInWord', 'mean(wdist)', df2.ORANGE, fnum=5)
    draw_scatterplot(figdir, idf_list, nPoints_list, 'idf', 'nPointsInWord', df2.LIGHT_BLUE, fnum=6)
    #df2.present()


def dump_word_patches(vocabdir, invindex, wx_sample):
    """
    Dumps word member patches to disk
    """
    wx2_dpath = get_word_dpaths(vocabdir, wx_sample)

    # Write each patch from each annotation to disk
    idx2_daid = invindex.idx2_daid
    daids = invindex.daids
    idx2_dfx = invindex.idx2_dfx
    #maws_list = invindex.idx2_wxs[idxs]

    # Loop over all annotations skipping the ones without any words in the sample
    ax2_idxs = [np.where(idx2_daid == aid_)[0] for aid_ in ut.progiter(daids, 'Building Forward Index: ', freq=100)]
    patchdump_iter = ut.progiter(zip(daids, ax2_idxs), freq=1,
                                    lbl='Dumping Selected Patches: ', num=len(daids))
    for aid, idxs in patchdump_iter:
        wxs_list  = invindex.idx2_wxs[idxs]
        if len(set(ut.flatten(wxs_list)).intersection(set(wx_sample))) == 0:
            # skip this annotation
            continue
        fx_list   = idx2_dfx[idxs]
        chip      = ibs.get_annot_chips(aid)
        chip_kpts = ibs.get_annot_kpts(aid)
        nid       = ibs.get_annot_nids(aid)
        patches, subkpts = ptool.get_warped_patches(chip, chip_kpts)
        for fx, wxs, patch in zip(fx_list, wxs_list, patches):
            assert len(wxs) == 1, 'did you multiassign the database? If so implement it here too'
            for k, wx in enumerate(wxs):
                if wx not in wx_sample:
                    continue
                patch_fname = 'patch_nid=%04d_aid=%04d_fx=%04d_k=%d' % (nid, aid, fx, k)
                fpath = join(wx2_dpath[wx], patch_fname)
                #gtool.imwrite(fpath, patch, fallback=True)
                gtool.imwrite_fallback(fpath, patch)


def get_word_dname(wx, metrics):
    stats_ = metrics.wx2_wdist_stats[wx]
    wname_clean = 'wx=%06d' % wx
    stats1 = 'max={max},min={min},mean={mean},'.format(**stats_)
    stats2 = 'std={std},nMaxMin=({nMax},{nMin}),shape={shape}'.format(**stats_)
    fname_fmt = wname_clean + '_{stats1}{stats2}'
    fmt_dict = dict(stats1=stats1, stats2=stats2)
    word_dname = ut.long_fname_format(fname_fmt, fmt_dict, ['stats2', 'stats1'], max_len=250, hashlen=4)
    return word_dname


def get_word_dpaths(vocabdir, wx_sample):
    """
    Gets word folder names and ensure they exist
    """
    ut.ensuredir(vocabdir)
    wx2_dpath = {wx: join(vocabdir, get_word_dname(wx)) for wx in wx_sample}
    progiter = ut.progiter(lbl='Ensuring word_dpath: ', freq=200)
    for dpath in progiter(six.itervalues(wx2_dpath)):
        ut.ensuredir(dpath)
    return wx2_dpath


def make_wordfigures(figdir, wx_sample, wx2_dpath):
    """
    Builds mosaics of patches assigned to words in sample
    ouptuts them to disk
    """
    from plottool import draw_func2 as df2
    import parse

    vocabdir = join(figdir, 'vocab_patches2')
    ut.ensuredir(vocabdir)
    dump_word_patches(invindex, wx_sample)

    # COLLECTING PART --- collects patches in word folders
    #vocabdir

    seldpath = vocabdir + '_selected'
    ut.ensurepath(seldpath)
    # stack for show
    for wx, dpath in ut.progiter(six.iteritems(wx2_dpath), lbl='Dumping Word Images:', num=len(wx2_dpath), freq=1, backspace=False):
        #df2.rrr()
        fpath_list = ut.ls(dpath)
        fname_list = [basename(fpath_) for fpath_ in fpath_list]
        patch_list = [gtool.imread(fpath_) for fpath_ in fpath_list]
        # color each patch by nid
        nid_list = [int(parse.parse('{}_nid={nid}_{}', fname)['nid']) for fname in fname_list]
        nid_set = set(nid_list)
        nid_list = np.array(nid_list)
        if len(nid_list) == len(nid_set):
            # no duplicate names
            newpatch_list = patch_list
        else:
            # duplicate names. do coloring
            sortx = nid_list.argsort()
            patch_list = np.array(patch_list, dtype=object)[sortx]
            fname_list = np.array(fname_list, dtype=object)[sortx]
            nid_list = nid_list[sortx]
            colors = (255 * np.array(df2.distinct_colors(len(nid_set)))).astype(np.int32)
            color_dict = dict(zip(nid_set, colors))
            wpad, hpad = 3, 3
            newshape_list = [tuple((np.array(patch.shape) + (wpad * 2, hpad * 2, 0)).tolist()) for patch in patch_list]
            color_list = [color_dict[nid_] for nid_ in nid_list]
            newpatch_list = [np.zeros(shape) + color[None, None] for shape, color in zip(newshape_list, color_list)]
            for patch, newpatch in zip(patch_list, newpatch_list):
                newpatch[wpad:-wpad, hpad:-hpad, :] = patch
            #img_list = patch_list
            #bigpatch = df2.stack_image_recurse(patch_list)
        #bigpatch = df2.stack_image_list(patch_list, vert=False)
        bigpatch = df2.stack_square_images(newpatch_list)
        bigpatch_fpath = join(seldpath, basename(dpath) + '_patches.png')

        #
        def _dictstr(dict_):
            str_ = ut.dict_str(dict_, newlines=False)
            str_ = str_.replace('\'', '').replace(': ', '=').strip('{},')
            return str_

        figtitle = '\n'.join([
            'wx=%r' % wx,
            'stat(pdist): %s' % _dictstr(metrics.wx2_pdist_stats[wx]),
            'stat(wdist): %s' % _dictstr(metrics.wx2_wdist_stats[wx]),
        ])
        metrics.wx2_nMembers[wx]

        df2.figure(fnum=1, doclf=True, docla=True)
        fig, ax = df2.imshow(bigpatch, figtitle=figtitle)
        #fig.show()
        df2.set_figtitle(figtitle)
        df2.adjust_subplots(top=.878, bottom=0)
        df2.save_figure(1, bigpatch_fpath)
        #gtool.imwrite(bigpatch_fpath, bigpatch)


def select_by_metric(wx2_metric, per_quantile=20):
    # sample a few words around the quantile points
    metric_list = np.array(list(wx2_metric.values()))
    wx_list = np.array(list(wx2_metric.keys()))
    metric_quantiles = spms.mquantiles(metric_list)
    metric_quantiles = np.array(metric_quantiles.tolist() + [metric_list.max(), metric_list.min()])
    wx_interest = []
    for scalar in metric_quantiles:
        dist = (metric_list - scalar) ** 2
        wx_quantile = wx_list[dist.argsort()[0:per_quantile]]
        wx_interest.extend(wx_quantile.tolist())
    overlap = len(wx_interest) - len(set(wx_interest))
    if overlap > 0:
        print('warning: overlap=%r' % overlap)
    return wx_interest


def get_metric(metrics, tupkey, statkey=None):
    wx2_metric = metrics.__dict__[tupkey]
    if statkey is not None:
        wx2_submetric = [stats_[statkey] for wx, stats_ in six.iteritems(wx2_metric) if statkey in stats_]
        return wx2_submetric
    return wx2_metric

#{wx: pdist_stats['max'] for wx, pdist_stats in six.iteritems(wx2_pdist_stats) if 'max' in pdist_stats}
#wx2_wrad = {wx: wdist_stats['max'] for wx, wdist_stats in six.iteritems(wx2_wdist_stats) if 'max' in wdist_stats}


def vizualize_vocabulary(ibs, invindex):
    """
    cleaned up version of dump_word_patches

    Dev:
        >>> from ibeis.model.hots.smk.smk_plots import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> from ibeis.model.hots.smk import smk_repr
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=64000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='GZ_ALL', nWords=8000)
        >>> #tup = smk_debug.testdata_raw_internals0(db='PZ_Master0', nWords=64000)
        >>> tup = smk_debug.testdata_raw_internals0(db='PZ_Mothers', nWords=8000)
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = tup
        >>> smk_repr.compute_data_internals_(invindex, qreq_.qparams, delete_rawvecs=False)
        >>> #aid = qaids[0]
    """
    invindex.idx2_wxs = np.array(invindex.idx2_wxs)

    print('[smk_plots] Vizualizing vocabulary')

    # DUMPING PART --- dumps patches to disk
    figdir = ibs.get_fig_dir()
    ut.ensuredir(figdir)
    if ut.get_argflag('--vf'):
        ut.view_directory(figdir)

    # Compute Word Statistics
    metrics = compute_word_metrics(invindex)
    (wx2_pdist, wx2_wdist, wx2_nMembers, wx2_pdist_stats, wx2_wdist_stats) = metrics

    #wx2_prad = {wx: pdist_stats['max'] for wx, pdist_stats in six.iteritems(wx2_pdist_stats) if 'max' in pdist_stats}
    #wx2_wrad = {wx: wdist_stats['max'] for wx, wdist_stats in six.iteritems(wx2_wdist_stats) if 'max' in wdist_stats}

    wx2_prad = get_metric(metrics, 'wx2_pdist_stats', 'max')
    wx2_wrad = get_metric(metrics, 'wx2_wdist_stats', 'max')

    wx_sample1 = select_by_metric(wx2_nMembers)
    wx_sample2 = select_by_metric(wx2_prad)
    wx_sample3 = select_by_metric(wx2_wrad)

    wx_sample = wx_sample1 + wx_sample2 + wx_sample3
    overlap123 = len(wx_sample) - len(set(wx_sample))
    print('overlap123 = %r' % overlap123)
    wx_sample  = set(wx_sample)
    print('len(wx_sample) = %r' % len(wx_sample))

    make_scatterplots()

    make_wordfigures()


def view_vocabs():
    """
    looks in vocab cachedir and prints info / vizualizes the vocabs
    """
    from vtool import clustering2 as clustertool
    import numpy as np

    fpath_sorted = get_cached_vocabs()

    num_pca_dims = 2  # 3
    whiten       = False
    kwd = dict(num_pca_dims=num_pca_dims,
               whiten=whiten,)

    def view_vocab(fpath):
        # QUANTIZED AND FLOATING POINT STATS
        centroids = ut.load_cPkl(fpath)
        print('viewing vocat fpath=%r' % (fpath,))
        smk_debug.vector_stats(centroids, 'centroids')
        #centroids_float = centroids.astype(np.float64) / 255.0
        centroids_float = centroids.astype(np.float64) / 512.0
        smk_debug.vector_stats(centroids_float, 'centroids_float')

        fig = clustertool.plot_centroids(centroids, centroids, labels='centroids',
                                         fnum=1, prefix='centroid vecs\n', **kwd)
        fig.show()

    for count, fpath in enumerate(fpath_sorted):
        if count > 0:
            break
        view_vocab(fpath)


if __name__ == '__main__':
    #from ibeis.model.hots.smk.smk_plots import *  # NOQA
    from plottool import draw_func2 as df2
    kwargs = {
        #'db': 'GZ_ALL',
        #'db': 'PZ_MTEST',
        'db': 'testdb1',
        'nWords': 8000,
        'delete_rawvecs': False,
    }
    (ibs, annots_df, daids, qaids, invindex, qreq_) = smk_debug.testdata_internals_full(**kwargs)
    metrics = compute_word_metrics(invindex)
    kwargs = {}
    valid_aids =  ibs.get_valid_aids()
    aid = 3
    for aid in ut.InteractiveIter(valid_aids):
        print('[smk_plot] visualizing annotation aid=%r' % (aid,))
        viz_annot_with_metrics(ibs, invindex, aid, metrics)
        execstr = df2.present()
    #exec(execstr)
