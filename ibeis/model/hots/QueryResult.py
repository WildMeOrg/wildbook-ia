from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[QRes]', DEBUG=False)
# Python
import cPickle
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


def query_result_fpath(ibs, qrid, uid):
    qres_dir  = ibs.qresdir
    fname = 'res_%s_qrid=%d.npz' % (uid, qrid)
    if len(fname) > 64:
        hash_id = utool.hashstr(uid)
        fname = 'res_%s_qrid=%d.npz' % (hash_id, qrid)
    fpath = join(qres_dir, fname)
    return fpath


def query_result_exists(ibs, qrid, uid):
    fpath = query_result_fpath(ibs, qrid, uid)
    return exists(fpath)


__OBJECT_BASE__ = utool.util_dev.get_object_base()


def assert_qres(qres):
    try:
        assert len(qres.rid2_fm) == len(qres.rid2_fs)
        assert len(qres.rid2_fm) == len(qres.rid2_fk)
        assert len(qres.rid2_fm) == len(qres.rid2_score)
    except AssertionError:
        raise AssertionError('[!qr] matching dicts do not agree')
    nFeatMatch_list = get_num_feats_in_matches(qres)
    assert all([num1 == num2 for (num1, num2) in
                izip(nFeatMatch_list, (len(fm) for fm in qres.rid2_fm.itervalues()))])
    assert all([num1 == num2 for (num1, num2) in
                izip(nFeatMatch_list, (len(fs) for fs in qres.rid2_fs.itervalues()))])
    assert all([num1 == num2 for (num1, num2) in
                izip(nFeatMatch_list, (len(fk) for fk in qres.rid2_fk.itervalues()))])


def get_num_chip_matches(qres):
    return len(qres.rid2_fm)


def get_num_feats_in_matches(qres):
    return [len(fm) for fm in qres.rid2_fm.itervalues()]


class QueryResult(__OBJECT_BASE__):
    #__slots__ = ['qrid', 'uid', 'nn_time',
                 #'weight_time', 'filt_time', 'build_time', 'verify_time',
                 #'rid2_fm', 'rid2_fs', 'rid2_fk', 'rid2_score']
    def __init__(qres, qrid, uid):
        # TODO: Merge FS and FK
        super(QueryResult, qres).__init__()
        qres.qrid = qrid
        qres.uid = uid
        # Assigned features matches
        qres.rid2_fm = None
        qres.rid2_fs = None
        qres.rid2_fk = None
        qres.rid2_score = None
        qres.filt2_meta = None  # messy

    def has_cache(qres, ibs):
        return query_result_exists(ibs, qres.qrid)

    def get_fpath(qres, ibs):
        return query_result_fpath(ibs, qres.qrid, qres.uid)

    @profile
    def save(qres, ibs):
        fpath = qres.get_fpath(ibs)
        if utool.VERBOSE:
            print('[qr] cache save: %r' % (split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            cPickle.dump(qres.__dict__, file_)

    @profile
    def load(qres, ibs):
        'Loads the result from the given database'
        fpath = qres.get_fpath(ibs)
        qrid_good = qres.qrid
        try:
            print('[qr] qres.load() fpath=%r' % (split(fpath)[1],))
            with open(fpath, 'rb') as file_:
                loaded_dict = cPickle.load(file_)
                qres.__dict__.update(loaded_dict)
            if not isinstance(qres.filt2_meta, dict):
                print('[qr] loading old result format')
                qres.filt2_meta = {}
        except IOError as ex:
            if not exists(fpath):
                print('[qr] query result cache miss')
                raise
            else:
                msg_list = ['[!qr] QueryResult(qrid=%d) is corrupted' % (qres.qrid),
                            '%r' % (ex,)]
                msg = ('\n'.join(msg_list))
                print(msg)
                raise Exception(msg)
        except BadZipFile as ex:
            print('[!qr] Caught other BadZipFile: %r' % ex)
            msg_list = ['[!qr] QueryResult(qrid=%d) is corrupted' % (qres.qrid),
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
        qres.qrid = qrid_good

    def cache_bytes(qres, ibs):
        """ Size of the cached query result on disk """
        fpath  = qres.get_fpath(ibs)
        nBytes = utool.file_bytes(fpath)
        return nBytes

    def get_fmatch_index(qres, ibs, rid, qfx):
        """ Returns the feature index in rid matching the query's qfx-th feature
            (if it exists)
        """
        fm = qres.rid2_fm[rid]
        mx_list = np.where(fm[:, 0] == qfx)[0]
        if len(mx_list) != 1:
            raise IndexError('[!qr] qfx=%r not found' % (qfx))
        else:
            mx = mx_list[0]
            return mx

    def get_rids_and_scores(qres):
        """ returns a chip index list and associated score list """
        score_list = np.array(qres.rid2_score.values())
        rid_list   = np.array(qres.rid2_score.keys())
        return rid_list, score_list

    def get_top_rids(qres, num=None):
        """ Returns a ranked list of chip indexes """
        rid_list, score_list = qres.get_rids_and_scores()
        # Get chip-ids sorted by scores
        top_rids = rid_list[score_list.argsort()[::-1]]
        num_indexed = len(top_rids)
        if num is None:
            num = num_indexed
        return top_rids[0:min(num, num_indexed)]

    def get_rid_scores(qres, rid_list):
        return [qres.rid2_score[rid] for rid in rid_list]

    def get_rid_ranks(qres, rid_list):
        """ get ranks of chip indexes in rid_list """
        top_rids = qres.get_top_rids()
        foundpos = [np.where(top_rids == rid)[0] for rid in rid_list]
        ranks_   = [ranks if len(ranks) > 0 else [-1] for ranks in foundpos]
        assert all([len(ranks) == 1 for ranks in ranks_]), 'len(rid_ranks) != 1'
        rank_list = [ranks[0] for ranks in ranks_]
        return rank_list

    def get_inspect_str(qres):
        assert_qres(qres)
        nFeatMatch_list = get_num_feats_in_matches(qres)
        nFeatMatch_stats = utool.mystats(nFeatMatch_list)

        top_lbl = utool.unindent('''
                                 top rids
                                 scores
                                 ranks''').strip()

        top_rids = qres.get_top_rids(num=5)
        top_scores = qres.get_rid_scores(top_rids)
        top_ranks = qres.get_rid_ranks(top_rids)

        top_stack = np.vstack((top_rids, top_scores, top_ranks))
        top_stack = np.array(top_stack, dtype=np.int32)
        top_str = str(top_stack)

        inspect_str = '\n'.join([
            'QueryResult',
            'qrid=%r ' % qres.qrid,
            utool.horiz_string(top_lbl, ' ', top_str),
            'num Feat Matches stats:',
            utool.indent(utool.dict_str(nFeatMatch_stats)),
        ])
        inspect_str = utool.indent(inspect_str, '[INSPECT] ')
        return inspect_str
