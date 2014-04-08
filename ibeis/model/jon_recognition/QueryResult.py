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
# HotSpotter
#import voting_rules2 as vr2


FM_DTYPE  = np.uint32   # Feature Match datatype
FS_DTYPE  = np.float32  # Feature Score datatype
FK_DTYPE  = np.int16    # Feature Position datatype


#=========================
# Query Result Class
#=========================


def remove_corrupted_queries(ibs, qres, dryrun=True):
    # This qres must be corrupted!
    uid = qres.uid
    hash_id = utool.hashstr(uid)
    qres_dir  = ibs.qresdir
    testres_dir = join(ibs.cachedir, 'experiment_harness_results')
    utool.remove_files_in_dir(testres_dir, dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + uid + '*', dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + hash_id + '*', dryrun=dryrun)


def query_result_fpath(ibs, qcid, uid):
    qres_dir  = ibs.qresdir
    fname = 'res_%s_qcid=%d.npz' % (uid, qcid)
    if len(fname) > 64:
        hash_id = utool.hashstr(uid)
        fname = 'res_%s_qcid=%d.npz' % (hash_id, qcid)
    fpath = join(qres_dir, fname)
    return fpath


def query_result_exists(ibs, qcid, uid):
    fpath = query_result_fpath(ibs, qcid, uid)
    return exists(fpath)


__OBJECT_BASE__ = utool.util_dev.get_object_base()


class QueryResult(__OBJECT_BASE__):
    #__slots__ = ['qcid', 'uid', 'nn_time',
                 #'weight_time', 'filt_time', 'build_time', 'verify_time',
                 #'cid2_fm', 'cid2_fs', 'cid2_fk', 'cid2_score']
    def __init__(qres, qcid, uid):
        # TODO: Merge FS and FK
        super(QueryResult, qres).__init__()
        qres.qcid = qcid
        qres.uid = uid
        # Assigned features matches
        qres.cid2_fm = np.array([], dtype=FM_DTYPE)
        qres.cid2_fs = np.array([], dtype=FS_DTYPE)
        qres.cid2_fk = np.array([], dtype=FK_DTYPE)
        qres.cid2_score = np.array([])
        qres.filt2_meta = {}  # messy

    def has_cache(qres, ibs):
        return query_result_exists(ibs, qres.qcid)

    def get_fpath(qres, ibs):
        return query_result_fpath(ibs, qres.qcid, qres.uid)

    @profile
    def save(qres, ibs):
        fpath = qres.get_fpath(ibs)
        if utool.VERBOSE:
            print('[qr] cache save: %r' % (split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            np.savez(file_, **qres.__dict__.copy())

    @profile
    def load(qres, ibs):
        'Loads the result from the given database'
        fpath = qres.get_fpath(ibs)
        qcid_good = qres.qcid
        try:
            with open(fpath, 'rb') as file_:
                npz = np.load(file_)
                for _key in npz.files:
                    qres.__dict__[_key] = npz[_key]
                npz.close()
            print('[qr] qres.load() fpath=%r' % (split(fpath)[1],))
            # These are nonarray items even if they are not lists
            # tolist seems to convert them back to their original
            # python representation
            qres.qcid = qres.qcid.tolist()
            try:
                qres.filt2_meta = qres.filt2_meta.tolist()
            except AttributeError:
                print('[qr] loading old result format')
                qres.filt2_meta = {}
            qres.uid = qres.uid.tolist()
            return True
        except IOError as ex:
            #print('[qr] encountered IOError: %r' % ex)
            if not exists(fpath):
                print('[qr] query result cache miss')
                #print(fpath)
                #print('[qr] QueryResult(qcid=%d) does not exist' % qres.qcid)
                raise
            else:
                msg = ['[qr] QueryResult(qcid=%d) is corrupted' % (qres.qcid)]
                msg += ['\n%r' % (ex,)]
                print(''.join(msg))
                raise Exception(msg)
        except BadZipFile as ex:
            print('[qr] Caught other BadZipFile: %r' % ex)
            msg = ['[qr] Attribute Error: QueryResult(qcid=%d) is corrupted' % (qres.qcid)]
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
        qres.qcid = qcid_good

    def cache_bytes(qres, ibs):
        fpath = qres.get_fpath(ibs)
        return utool.file_bytes(fpath)

    def get_gt_ranks(qres, gt_cids=None, ibs=None):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_cids is None and ibs is None:
            raise Exception('[qr] error')
        if gt_cids is None:
            gt_cids = ibs.get_other_indexed_cids(qres.qcid)
        return qres.get_cid_ranks(gt_cids)

    def get_cid_ranks(qres, cid_list):
        'get ranks of chip indexes in cid_list'
        score_list = np.array(qres.cid2_score.values())
        cid_list   = np.array(qres.cid2_score.keys())
        top_cids = cid_list[score_list.argsort()[::-1]]
        foundpos = [np.where(top_cids == cid)[0] for cid in cid_list]
        ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        assert all([len(r) == 1 for r in ranks_])
        rank_list = [r[0] for r in ranks_]
        return rank_list

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
        from ibeis.view import viz
        print('[qr] show_query')
        qrid = ibs.get_chip_rids(qres.qcid)
        viz.show_chip(ibs, qrid, **kwargs)

    def show_analysis(qres, ibs, *args, **kwargs):
        from ibeis.view import viz
        return viz.res_show_analysis(qres, ibs, *args, **kwargs)

    def show_top(qres, ibs, *args, **kwargs):
        from ibeis.view import viz
        return viz.show_top(qres, ibs, *args, **kwargs)

    def show_gt_matches(qres, ibs, *args, **kwargs):
        from ibeis.view import viz
        figtitle = ('q%s -- GroundTruth' % (ibs.cidstr(qres.qcid)))
        gt_cids = ibs.get_other_indexed_cids(qres.qcid)
        return viz._show_chip_matches(ibs, qres, gt_cids=gt_cids, figtitle=figtitle,
                                      all_kpts=True, *args, **kwargs)

    def show_chipres(qres, ibs, cid, **kwargs):
        from ibeis.view import viz
        return viz.res_show_chipres(qres, ibs, cid, **kwargs)

    def interact_chipres(qres, ibs, cid, **kwargs):
        from ibeis.view import interact
        return interact.interact_chipres(ibs, qres, cid, **kwargs)

    def interact_top_chipres(qres, ibs, tx, **kwargs):
        from ibeis.view import interact
        cid = qres.topN_cids(ibs, tx + 1)[tx]
        return interact.interact_chipres(ibs, qres, cid, **kwargs)

    def show_nearest_descriptors(qres, ibs, qfx, dodraw=True):
        from ibeis.view import viz
        qcid = qres.qcid
        viz.show_nearest_descriptors(ibs, qcid, qfx, fnum=None)
        if dodraw:
            viz.draw()

    def get_match_index(qres, ibs, cid, qfx, strict=True):
        qcid = qres.qcid
        fm = qres.cid2_fm[cid]
        mx_list = np.where(fm[:, 0] == qfx)[0]
        if len(mx_list) != 1:
            if strict:
                raise IndexError('qfx=%r not found in query %s' %
                                 (qfx, ibs.vs_str(qcid, cid)))
            else:
                return None
        else:
            mx = mx_list[0]
            return mx
