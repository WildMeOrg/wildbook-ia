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

FM_DTYPE  = np.uint32   # Feature Match datatype
FS_DTYPE  = np.float32  # Feature Score datatype
FK_DTYPE  = np.int16    # Feature Position datatype


#=========================
# Query Result Class
#=========================


def qres_get_matching_keypoints(qres, ibs, rid2_list):  # rid2 is a name. 2 != 2 to here
    rid1 = qres.qrid
    kpts1 = ibs.get_roi_kpts(rid1)
    kpts2_list = ibs.get_roi_kpts(rid2_list)
    matching_kpts_list = []
    empty_fm = np.empty((0, 2))
    for rid2, kpts2 in izip(rid2_list, kpts2_list):
        fm = qres.rid2_fm.get(rid2, empty_fm)
        if len(fm) == 0:
            continue
        kpts1_m = kpts1[fm.T[0]]
        kpts2_m = kpts2[fm.T[1]]
        kpts_match = (kpts1_m, kpts2_m)
        matching_kpts_list.append(kpts_match)
    return matching_kpts_list


def remove_corrupted_queries(qreq, qres, dryrun=True):
    # This qres must be corrupted!
    uid = qres.uid
    hash_id = utool.hashstr(uid)
    qres_dir  = qreq.qresdir
    testres_dir = join(qreq.qresdir, '..', 'experiment_harness_results')
    utool.remove_files_in_dir(testres_dir, dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + uid + '*', dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + hash_id + '*', dryrun=dryrun)


def query_result_fpath(qreq, qrid, uid):
    qres_dir  = qreq.qresdir
    fname = 'res_%s_qrid=%d.npz' % (uid, qrid)
    if len(fname) > 64:
        hash_id = utool.hashstr(uid)
        fname = 'res_%s_qrid=%d.npz' % (hash_id, qrid)
    fpath = join(qres_dir, fname)
    return fpath


def query_result_exists(qreq, qrid, uid):
    fpath = query_result_fpath(qreq, qrid, uid)
    return exists(fpath)


__OBJECT_BASE__ = object  # utool.util_dev.get_object_base()


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
    """
    __slots__ = ['qrid', 'uid', 'nn_time',
                 'weight_time', 'filt_time', 'build_time', 'verify_time',
                 'rid2_fm', 'rid2_fs', 'rid2_fk', 'rid2_score']
    """
    def __init__(qres, qrid, uid):
        # THE UID MUST BE SPECIFIED CORRECTLY AT CREATION TIME
        # TODO: Merge FS and FK
        super(QueryResult, qres).__init__()
        qres.qrid = qrid
        qres.uid = uid
        qres.eid = None  # encounter id
        # Assigned features matches
        qres.rid2_fm = None  # feat_match_list
        qres.rid2_fs = None  # feat_score_list
        qres.rid2_fk = None  # feat_rank_list
        qres.rid2_score = None  # roi score
        qres.filt2_meta = None  # messy

    def has_cache(qres, qreq):
        return query_result_exists(qreq, qres.qrid)

    def get_fpath(qres, qreq):
        return query_result_fpath(qreq, qres.qrid, qres.uid)

    @profile
    def save(qres, qreq):
        fpath = qres.get_fpath(qreq)
        if utool.VERBOSE:
            print('[qr] cache save: %r' % (split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            cPickle.dump(qres.__dict__, file_)

    @profile
    def load(qres, qreq):
        'Loads the result from the given database'
        fpath = qres.get_fpath(qreq)
        qrid_good = qres.qrid
        try:
            #print('[qr] qres.load() fpath=%r' % (split(fpath)[1],))
            with open(fpath, 'rb') as file_:
                loaded_dict = cPickle.load(file_)
                qres.__dict__.update(loaded_dict)
            if not isinstance(qres.filt2_meta, dict):
                print('[qr] loading old result format')
                qres.filt2_meta = {}
            print('... qres cache hit: %r' % (split(fpath)[1],))
        except IOError as ex:
            if not exists(fpath):
                print('... qres cache miss: %r' % (split(fpath)[1],))
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

    def cache_bytes(qres, qreq):
        """ Size of the cached query result on disk """
        fpath  = qres.get_fpath(qreq)
        nBytes = utool.file_bytes(fpath)
        return nBytes

    def get_fmatch_index(qres, rid, qfx):
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

    def get_rid_ranks(qres, rid_list):
        """ get ranks of chip indexes in rid_list """
        top_rids = qres.get_top_rids()
        foundpos = [np.where(top_rids == rid)[0] for rid in rid_list]
        ranks_   = [ranks if len(ranks) > 0 else [None] for ranks in foundpos]
        assert all([len(ranks) == 1 for ranks in ranks_]), 'len(rid_ranks) != 1'
        rank_list = [ranks[0] for ranks in ranks_]
        return rank_list

    #def get_rid_ranks(qres, rid_list):
        #'get ranks of chip indexes in rid_list'
        #score_list = np.array(qres.rid2_score.values())
        #rid_list   = np.array(qres.rid2_score.keys())
        #top_rids = rid_list[score_list.argsort()[::-1]]
        #foundpos = [np.where(top_rids == rid)[0] for rid in rid_list]
        #ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        #assert all([len(r) == 1 for r in ranks_])
        #rank_list = [r[0] for r in ranks_]
        #return rank_list

    def get_gt_ranks(qres, gt_rids=None, ibs=None, return_gtrids=False):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_rids is None and ibs is None:
            raise Exception('[qr] must pass in the gt_rids or ibs object')
        if gt_rids is None:
            gt_rids = ibs.get_roi_groundtruth(qres.qrid)
        gt_ranks = qres.get_rid_ranks(gt_rids)
        if return_gtrids:
            return gt_ranks, gt_rids
        else:
            return gt_ranks

    def get_best_gt_rank(qres, ibs):
        """ Returns the best rank over all the groundtruth """
        gt_ranks, gt_rids = qres.get_gt_ranks(ibs=ibs, return_gtrids=True)
        ridrank_tups = list(izip(gt_rids, gt_ranks))
        # Get only the rids that placed in the shortlist
        #valid_gtrids = np.array([ rid for rid, rank in ridrank_tups if rank is not None])
        valid_ranks  = np.array([rank for rid, rank in ridrank_tups if rank is not None])
        # Sort so lowest score is first
        best_rankx = valid_ranks.argsort()
        #best_gtrids  = best_gtrids[best_rankx]
        best_gtranks = valid_ranks[best_rankx]
        if len(best_gtranks) == 0:
            best_rank = -1
        else:
            best_rank = best_gtranks[0]
        return best_rank

    def get_classified_pos(qres):
        top_rids = np.array(qres.get_top_rids())
        pos_rids = top_rids[0:1]
        return pos_rids

    def show_top(qres, ibs, *args, **kwargs):
        from ibeis.viz import viz_qres
        return viz_qres.show_qres_top(ibs, qres, *args, **kwargs)

    def show_analysis(qres, ibs, *args, **kwargs):
        from ibeis.viz import viz_qres
        return viz_qres.show_qres_analysis(ibs, qres, *args, **kwargs)

    def show(qres, ibs, type_, *args, **kwargs):
        if type_ == 'top':
            return qres.show_top(ibs, *args, **kwargs)
        elif type_ == 'analysis':
            return qres.show_analysis(ibs, *args, **kwargs)
        else:
            raise AssertionError('Uknown type=%r' % type_)

    def get_matching_keypoints(qres, ibs, rid2_list):
        return qres_get_matching_keypoints(qres, ibs, rid2_list)
