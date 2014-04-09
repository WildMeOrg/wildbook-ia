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


def assert_qres(qres):
    try:
        assert len(qres.cid2_fm) == len(qres.cid2_fs)
        assert len(qres.cid2_fm) == len(qres.cid2_fk)
        assert len(qres.cid2_fm) == len(qres.cid2_score)
    except AssertionError:
        raise AssertionError('[!qr] matching dicts do not agree')
    nFeatMatch_list = get_num_feats_in_matches(qres)
    assert all([num1 == num2 for (num1, num2) in
                izip(nFeatMatch_list, (len(fm) for fm in qres.cid2_fm.itervalues()))])
    assert all([num1 == num2 for (num1, num2) in
                izip(nFeatMatch_list, (len(fs) for fs in qres.cid2_fs.itervalues()))])
    assert all([num1 == num2 for (num1, num2) in
                izip(nFeatMatch_list, (len(fk) for fk in qres.cid2_fk.itervalues()))])


def get_num_chip_matches(qres):
    return len(qres.cid2_fm)


def get_num_feats_in_matches(qres):
    return [len(fm) for fm in qres.cid2_fm.itervalues()]


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
            if not exists(fpath):
                print('[qr] query result cache miss')
                raise
            else:
                msg_list = ['[!qr] QueryResult(qcid=%d) is corrupted' % (qres.qcid),
                            '%r' % (ex,)]
                msg = ('\n'.join(msg_list))
                print(msg)
                raise Exception(msg)
        except BadZipFile as ex:
            print('[!qr] Caught other BadZipFile: %r' % ex)
            msg_list = ['[!qr] QueryResult(qcid=%d) is corrupted' % (qres.qcid),
                        '%r' % (ex,)]
            msg = '\n'.join(msg_list)
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
        """ Size of the cached query result on disk """
        fpath  = qres.get_fpath(ibs)
        nBytes = utool.file_bytes(fpath)
        return nBytes

    def get_fmatch_index(qres, ibs, cid, qfx):
        """ Returns the feature index in cid matching the query's qfx-th feature
            (if it exists)
        """
        fm = qres.cid2_fm[cid]
        mx_list = np.where(fm[:, 0] == qfx)[0]
        if len(mx_list) != 1:
            raise IndexError('[!qr] qfx=%r not found' % (qfx))
        else:
            mx = mx_list[0]
            return mx

    def get_cids_and_scores(qres):
        """ returns a chip index list and associated score list """
        score_list = np.array(qres.cid2_score.values())
        cid_list   = np.array(qres.cid2_score.keys())
        return cid_list, score_list

    def get_top_cids(qres, num=None):
        """ Returns a ranked list of chip indexes """
        cid_list, score_list = qres.get_cids_and_scores()
        # Get chip-ids sorted by scores
        top_cids = cid_list[score_list.argsort()[::-1]]
        num_indexed = len(top_cids)
        if num is None:
            num = num_indexed
        return top_cids[0:min(num, num_indexed)]

    def get_cid_scores(qres, cid_list):
        return [qres.cid2_score[cid] for cid in cid_list]

    def get_cid_ranks(qres, cid_list):
        'get ranks of chip indexes in cid_list'
        top_cids = qres.get_top_cids()
        foundpos = [np.where(top_cids == cid)[0] for cid in cid_list]
        ranks_   = [ranks if len(ranks) > 0 else [-1] for ranks in foundpos]
        assert all([len(ranks) == 1 for ranks in ranks_]), 'len(cid_ranks) != 1'
        rank_list = [ranks[0] for ranks in ranks_]
        return rank_list

    def get_inspect_str(qres):
        assert_qres(qres)
        nFeatMatch_list = get_num_feats_in_matches(qres)
        nFeatMatch_stats = utool.mystats(nFeatMatch_list)

        top_lbl = utool.unindent('''
                                 top5 cids
                                 scores
                                 ranks''').strip()

        top_cids = qres.get_top_cids(num=5)
        top_scores = qres.get_cid_scores(top_cids)
        top_ranks = qres.get_cid_ranks(top_cids)

        top_stack = np.vstack((top_cids, top_scores, top_ranks))
        top_stack = np.array(top_stack, dtype=np.int32)
        top_str = str(top_stack)

        inspect_str = '\n'.join([
            'QueryResult',
            'qcid=%r ' % qres.qcid,
            utool.horiz_string(top_lbl, ' ', top_str),
            'num Feat Matches stats:',
            utool.indent(utool.dict_str(nFeatMatch_stats)),
        ])
        inspect_str = utool.indent(inspect_str, '[qr.INSPECT] ')
        return inspect_str
