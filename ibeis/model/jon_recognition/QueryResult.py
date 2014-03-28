from __future__ import division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[QRes]', DEBUG=False)
# Python
from itertools import izip
from os.path import exists, split, join
from zipfile import error as BadZipFile  # Screwy naming convention.
import os
# Scientific
import numpy as np
# IBEIS
from ibeis.dev import params
# HotSpotter
import voting_rules2 as vr2


FM_DTYPE  = np.uint32   # Feature Match datatype
FS_DTYPE  = np.float32  # Feature Score datatype
FK_DTYPE  = np.int16    # Feature Position datatype

HASH_LEN = 16

#=========================
# Query Result Class
#=========================


def remove_corrupted_queries(ibs, res, dryrun=True):
    # This res must be corrupted!
    uid = res.uid
    hash_id = utool.hashstr(uid, HASH_LEN)
    qres_dir  = ibs.dirs.qres_dir
    testres_dir = join(ibs.dirs.cache_dir, 'experiment_harness_results')
    utool.remove_files_in_dir(testres_dir, dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + uid + '*', dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + hash_id + '*', dryrun=dryrun)


def query_result_fpath(ibs, qcx, uid):
    qres_dir  = ibs.dirs.qres_dir
    qcid  = ibs.tables.cid2_cid[qcx]
    fname = 'res_%s_qcid=%d.npz' % (uid, qcid)
    if len(fname) > 64:
        hash_id = utool.hashstr(uid, HASH_LEN)
        fname = 'res_%s_qcid=%d.npz' % (hash_id, qcid)
    fpath = join(qres_dir, fname)
    return fpath


def query_result_exists(ibs, qcx, uid):
    fpath = query_result_fpath(ibs, qcx, uid)
    return exists(fpath)


__OBJECT_BASE__ = object if not utool.get_flag('--debug') else utool.DynStrucct


class QueryResult(__OBJECT_BASE__):
    #__slots__ = ['qcx', 'uid', 'nn_time',
                 #'weight_time', 'filt_time', 'build_time', 'verify_time',
                 #'cid2_fm', 'cid2_fs', 'cid2_fk', 'cid2_score']
    def __init__(res, qcx, uid):
        super(QueryResult, res).__init__()
        res.qcx = qcx
        res.uid = uid
        # Assigned features matches
        res.cid2_fm = np.array([], dtype=FM_DTYPE)
        # TODO: Merge FS and FK
        res.cid2_fs = np.array([], dtype=FS_DTYPE)
        res.cid2_fk = np.array([], dtype=FK_DTYPE)
        res.cid2_score = np.array([])
        res.filt2_meta = {}  # messy

    def has_cache(res, ibs):
        return query_result_exists(ibs, res.qcx)

    def get_fpath(res, ibs):
        return query_result_fpath(ibs, res.qcx, res.uid)

    @profile
    def save(res, ibs):
        fpath = res.get_fpath(ibs)
        print('[qr] cache save: %r' % (fpath if params.args.verbose_cache
                                       else split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            np.savez(file_, **res.__dict__.copy())

    @profile
    def load(res, ibs):
        'Loads the result from the given database'
        fpath = res.get_fpath(ibs)
        qcx_good = res.qcx
        try:
            with open(fpath, 'rb') as file_:
                npz = np.load(file_)
                for _key in npz.files:
                    res.__dict__[_key] = npz[_key]
                npz.close()
            print('[qr] res.load() fpath=%r' % (split(fpath)[1],))
            # These are nonarray items even if they are not lists
            # tolist seems to convert them back to their original
            # python representation
            res.qcx = res.qcx.tolist()
            try:
                res.filt2_meta = res.filt2_meta.tolist()
            except AttributeError:
                print('[qr] loading old result format')
                res.filt2_meta = {}
            res.uid = res.uid.tolist()
            return True
        except IOError as ex:
            #print('[qr] encountered IOError: %r' % ex)
            if not exists(fpath):
                print('[qr] query result cache miss')
                #print(fpath)
                #print('[qr] QueryResult(qcx=%d) does not exist' % res.qcx)
                raise
            else:
                msg = ['[qr] QueryResult(qcx=%d) is corrupted' % (res.qcx)]
                msg += ['\n%r' % (ex,)]
                print(''.join(msg))
                raise Exception(msg)
        except BadZipFile as ex:
            print('[qr] Caught other BadZipFile: %r' % ex)
            msg = ['[qr] Attribute Error: QueryResult(qcx=%d) is corrupted' % (res.qcx)]
            msg += ['\n%r' % (ex,)]
            print(''.join(msg))
            if exists(fpath):
                print('[qr] Removing corrupted file: %r' % fpath)
                os.remove(fpath)
                raise IOError(msg)
            else:
                raise Exception(msg)
        except Exception as ex:
            print('Caught other Exception: %r' % ex)
            raise
        res.qcx = qcx_good

    def cache_bytes(res, ibs):
        fpath = res.get_fpath(ibs)
        return utool.file_bytes(fpath)

    def get_gt_ranks(res, gt_cxs=None, ibs=None):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_cxs is None and ibs is None:
            raise Exception('[qr] error')
        if gt_cxs is None:
            gt_cxs = ibs.get_other_indexed_cxs(res.qcx)
        return res.get_cx_ranks(gt_cxs)

    def get_cx_ranks(res, cid_list):
        'get ranks of chip indexes in cid_list'
        cid2_score = res.get_cx2_score()
        top_cxs  = cid2_score.argsort()[::-1]
        foundpos = [np.where(top_cxs == cid)[0] for cid in cid_list]
        ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        assert all([len(r) == 1 for r in ranks_])
        rank_list = [r[0] for r in ranks_]
        return rank_list

    def get_cx2_score(res):
        return res.cid2_score

    def get_cx2_fm(res):
        return res.cid2_fm

    def get_cx2_fs(res):
        return res.cid2_fs

    def get_cx2_fk(res):
        return res.cid2_fk

    def get_fmatch_iter(res):
        fmfsfk_enum = enumerate(izip(res.cid2_fm, res.cid2_fs, res.cid2_fk))
        fmatch_iter = ((cid, fx_tup, score, rank)
                       for cid, (fm, fs, fk) in fmfsfk_enum
                       for (fx_tup, score, rank) in izip(fm, fs, fk))
        return fmatch_iter

    def topN_cxs(res, ibs, N=None, only_gt=False, only_nongt=False):
        cid2_score = np.array(res.get_cx2_score())
        if ibs.prefs.display_cfg.name_scoring:
            cid2_chipscore = np.array(cid2_score)
            cid2_score = vr2.enforce_one_name(ibs, cid2_score,
                                              cid2_chipscore=cid2_chipscore)
        top_cxs = cid2_score.argsort()[::-1]
        dcxs_ = set(ibs.get_indexed_sample()) - set([res.qcx])
        top_cxs = [cid for cid in iter(top_cxs) if cid in dcxs_]
        #top_cxs = np.intersect1d(top_cxs, ibs.get_indexed_sample())
        if only_gt:
            gt_cxs = set(ibs.get_other_indexed_cxs(res.qcx))
            top_cxs = [cid for cid in iter(top_cxs) if cid in gt_cxs]
        if only_nongt:
            gt_cxs = set(ibs.get_other_indexed_cxs(res.qcx))
            top_cxs = [cid for cid in iter(top_cxs) if not cid in gt_cxs]
        nIndexed = len(top_cxs)
        if N is None:
            N = ibs.prefs.display_cfg.N
        if N == 'all':
            N = nIndexed
        #print('[qr] cid2_score = %r' % (cid2_score,))
        #print('[qr] returning top_cxs = %r' % (top_cxs,))
        nTop = min(N, nIndexed)
        #print('[qr] returning nTop = %r' % (nTop,))
        topN_cxs = top_cxs[0:nTop]
        return topN_cxs

    def compute_seperability(res, ibs):
        top_gt = res.topN_cxs(ibs, N=1, only_gt=True)
        top_nongt = res.topN_cxs(ibs, N=1, only_nongt=True)
        if len(top_gt) == 0:
            return None
        score_true = res.cid2_score[top_gt[0]]
        score_false = res.cid2_score[top_nongt[0]]
        seperatiblity = score_true - score_false
        return seperatiblity

    def show_query(res, ibs, **kwargs):
        from ibeis.view import viz
        print('[qr] show_query')
        viz.show_chip(ibs, res=res, **kwargs)

    def show_analysis(res, ibs, *args, **kwargs):
        from ibeis.view import viz
        return viz.res_show_analysis(res, ibs, *args, **kwargs)

    def show_top(res, ibs, *args, **kwargs):
        from ibeis.view import viz
        return viz.show_top(res, ibs, *args, **kwargs)

    def show_gt_matches(res, ibs, *args, **kwargs):
        from ibeis.view import viz
        figtitle = ('q%s -- GroundTruth' % (ibs.cidstr(res.qcx)))
        gt_cxs = ibs.get_other_indexed_cxs(res.qcx)
        return viz._show_chip_matches(ibs, res, gt_cxs=gt_cxs, figtitle=figtitle,
                                      all_kpts=True, *args, **kwargs)

    def show_chipres(res, ibs, cid, **kwargs):
        from ibeis.view import viz
        return viz.res_show_chipres(res, ibs, cid, **kwargs)

    def interact_chipres(res, ibs, cid, **kwargs):
        from ibeis.view import interact
        return interact.interact_chipres(ibs, res, cid, **kwargs)

    def interact_top_chipres(res, ibs, tx, **kwargs):
        from ibeis.view import interact
        cid = res.topN_cxs(ibs, tx + 1)[tx]
        return interact.interact_chipres(ibs, res, cid, **kwargs)

    def show_nearest_descriptors(res, ibs, qfx, dodraw=True):
        from ibeis.view import viz
        qcx = res.qcx
        viz.show_nearest_descriptors(ibs, qcx, qfx, fnum=None)
        if dodraw:
            viz.draw()

    def get_match_index(res, ibs, cid, qfx, strict=True):
        qcx = res.qcx
        fm = res.cid2_fm[cid]
        mx_list = np.where(fm[:, 0] == qfx)[0]
        if len(mx_list) != 1:
            if strict:
                raise IndexError('qfx=%r not found in query %s' %
                                 (qfx, ibs.vs_str(qcx, cid)))
            else:
                return None
        else:
            mx = mx_list[0]
            return mx
