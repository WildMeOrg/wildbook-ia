# flake8:noqa
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[QRes]', DEBUG=False)
# Python
from itertools import izip
# Scientific
import numpy as np






def get_cid2_score(qres):
    return qres.cid2_score


def get_cid2_fm(qres):
    return qres.cid2_fm


def get_cid2_fs(qres):
    return qres.cid2_fs


def get_cid2_fk(qres):
    return qres.cid2_fk


def get_fmatch_iter(qres):
    fmfsfk_enum = enumerate(izip(qres.cid2_fm, qres.cid2_fs, qres.cid2_fk))
    fmatch_iter = ((cid, fx_tup, score, rank)
                    for cid, (fm, fs, fk) in fmfsfk_enum
                    for (fx_tup, score, rank) in izip(fm, fs, fk))
    return fmatch_iter


def topN_cids(qres, ibs, N=None, only_gt=False, only_nongt=False):
    score_list = np.array(qres.cid2_score.values())
    cid_list   = np.array(qres.cid2_score.keys())
    #if ibs.cfg.display_cfg.name_scoring:
        #cid2_chipscore = np.array(cid2_score)
        #cid2_score = vr2.enforce_one_name(ibs, cid2_score,
                                            #cid2_chipscore=cid2_chipscore)
    top_cids = cid_list[score_list.argsort()[::-1]]
    #top_cids = np.intersect1d(top_cids, ibs.get_indexed_sample())
    if only_gt:
        gt_cids = set(ibs.get_chip_groundtruth(qres.qcid))
        top_cids = [cid for cid in iter(top_cids) if cid in gt_cids]
    if only_nongt:
        gt_cids = set(ibs.get_chip_groundtruth(qres.qcid))
        top_cids = [cid for cid in iter(top_cids) if not cid in gt_cids]
    nIndexed = len(top_cids)
    if N is None:
        N = 5
        #N = ibs.prefs.display_cfg.N
    #if N == 'all':
        #N = nIndexed
    #print('[qr] cid2_score = %r' % (cid2_score,))
    #print('[qr] returning top_cids = %r' % (top_cids,))
    nTop = min(N, nIndexed)
    #print('[qr] returning nTop = %r' % (nTop,))
    topN_cids = top_cids[0:nTop]
    return topN_cids


def compute_seperability(qres, ibs):
    top_gt = qres.topN_cids(ibs, N=1, only_gt=True)
    top_nongt = qres.topN_cids(ibs, N=1, only_nongt=True)
    if len(top_gt) == 0:
        return None
    score_true = qres.cid2_score[top_gt[0]]
    score_false = qres.cid2_score[top_nongt[0]]
    seperatiblity = score_true - score_false
    return seperatiblity


def show_query(qres, ibs, **kwargs):
    from ibeis import viz
    print('[qr] show_query')
    qaid = ibs.get_chip_aids(qres.qcid)
    viz.show_chip(ibs, qaid, **kwargs)


def show_analysis(qres, ibs, *args, **kwargs):
    from ibeis import viz
    return viz.res_show_analysis(qres, ibs, *args, **kwargs)


def show_gt_matches(qres, ibs, *args, **kwargs):
    from ibeis import viz
    figtitle = ('q%s -- GroundTruth' % (ibs.cidstr(qres.qcid)))
    gt_cids = ibs.get_other_indexed_cids(qres.qcid)
    return viz._show_chip_matches(ibs, qres, gt_cids=gt_cids, figtitle=figtitle,
                                  all_kpts=True, *args, **kwargs)


def show_chipres(qres, ibs, cid, **kwargs):
    from ibeis import viz
    return viz.res_show_chipres(qres, ibs, cid, **kwargs)


def interact_chipres(qres, ibs, cid, **kwargs):
    from ibeis.viz import interact
    return interact.interact_chipres(ibs, qres, cid, **kwargs)


def interact_top_chipres(qres, ibs, tx, **kwargs):
    from ibeis.viz import interact
    cid = qres.topN_cids(ibs, tx + 1)[tx]
    return interact.interact_chipres(ibs, qres, cid, **kwargs)


def show_nearest_descriptors(qres, ibs, qfx, dodraw=True):
    from ibeis import viz
    qcid = qres.qcid
    viz.show_nearest_descriptors(ibs, qcid, qfx, fnum=None)
    if dodraw:
        viz.draw()
