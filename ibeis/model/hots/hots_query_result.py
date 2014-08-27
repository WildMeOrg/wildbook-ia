from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[QRes]', DEBUG=False)
# Python
import six
from six.moves import zip, cPickle
from os.path import exists, split, join
from zipfile import error as BadZipFile  # Screwy naming convention.
import os
# Scientific
import numpy as np
from ibeis.model.hots import exceptions as hsexcept

FM_DTYPE  = np.uint32   # Feature Match datatype
FS_DTYPE  = np.float32  # Feature Score datatype
FK_DTYPE  = np.int16    # Feature Position datatype


#=========================
# Query Result Class
#=========================


def qres_get_matching_keypoints(qres, ibs, aid2_list):  # aid2 is a name. 2 != 2 to here
    aid1 = qres.qaid
    kpts1 = ibs.get_annot_kpts(aid1)
    kpts2_list = ibs.get_annot_kpts(aid2_list)
    matching_kpts_list = []
    empty_fm = np.empty((0, 2))
    for aid2, kpts2 in zip(aid2_list, kpts2_list):
        fm = qres.aid2_fm.get(aid2, empty_fm)
        if len(fm) == 0:
            continue
        kpts1_m = kpts1[fm.T[0]]
        kpts2_m = kpts2[fm.T[1]]
        kpts_match = (kpts1_m, kpts2_m)
        matching_kpts_list.append(kpts_match)
    return matching_kpts_list


def remove_corrupted_queries(qresdir, qres, dryrun=True):
    # This qres must be corrupted!
    cfgstr = qres.cfgstr
    hash_id = utool.hashstr(cfgstr)
    qres_dir  = qresdir
    testres_dir = join(qresdir, '..', 'experiment_harness_results')
    utool.remove_files_in_dir(testres_dir, dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + cfgstr + '*', dryrun=dryrun)
    utool.remove_files_in_dir(qres_dir, '*' + hash_id + '*', dryrun=dryrun)


def query_result_fpath(qresdir, qaid, qauuid, cfgstr):
    """
    cdef:
        long qaid
        object qauuid
        str cfgstr, qres_dir, fpath, hash_id, fname
    """
    fname_fmt = 'res_%s_qaid=%d_qauuid=%s.npz'
    fname = fname_fmt % (cfgstr, qaid, str(qauuid))
    if len(fname) > 64:
        hash_id = utool.hashstr(cfgstr)
        fname = fname_fmt % (hash_id, qaid, str(qauuid))
    fpath = join(qresdir, fname)
    return fpath


def _qres_dicteq(aid2_xx1, aid2_xx2):
    """ Checks to see if qres dicts are the same """
    try:
        for (aid1, xx1), (aid2, xx2) in zip(aid2_xx1.items(),
                                            aid2_xx2.items()):
            assert aid1 == aid2, 'key mismatch'
            if np.iterable(xx1):
                assert all([np.all(x1 == x2) for (x1, x2) in zip(xx1, xx2)])
            else:
                assert xx1 == xx2
    except AssertionError:
        return False
    return True


__OBJECT_BASE__ = object  # utool.util_dev.get_object_base()


def assert_qres(qres):
    try:
        assert len(qres.aid2_fm) == len(qres.aid2_fs)
        assert len(qres.aid2_fm) == len(qres.aid2_fk)
        assert len(qres.aid2_fm) == len(qres.aid2_score)
    except AssertionError:
        raise AssertionError('[!qr] matching dicts do not agree')
    nFeatMatch_list = get_num_feats_in_matches(qres)
    assert all([num1 == num2 for (num1, num2) in
                zip(nFeatMatch_list, (len(fm) for fm in six.itervalues(qres.aid2_fm)))])
    assert all([num1 == num2 for (num1, num2) in
                zip(nFeatMatch_list, (len(fs) for fs in six.itervalues(qres.aid2_fs)))])
    assert all([num1 == num2 for (num1, num2) in
                zip(nFeatMatch_list, (len(fk) for fk in six.itervalues(qres.aid2_fk)))])


def get_num_chip_matches(qres):
    return len(qres.aid2_fm)


def get_num_feats_in_matches(qres):
    return [len(fm) for fm in six.itervalues(qres.aid2_fm)]


class QueryResult(__OBJECT_BASE__):
    #__slots__ = ['qaid', 'qauuid', 'cfgstr', 'eid',
    #             'aid2_fm', 'aid2_fs', 'aid2_fk', 'aid2_score',
    #             'filt2_meta']
    def __init__(qres, qaid, cfgstr, qauuid=None):
        # THE UID MUST BE SPECIFIED CORRECTLY AT CREATION TIME
        # TODO: Merge FS and FK
        super(QueryResult, qres).__init__()
        qres.qaid = qaid
        qres.qauuid = qauuid  # query annot uuid
        #qres.qauuid = qauuid
        qres.cfgstr = cfgstr
        qres.eid = None  # encounter id
        # Assigned features matches
        qres.aid2_fm = None  # feat_match_list
        qres.aid2_fs = None  # feat_score_list
        qres.aid2_fk = None  # feat_rank_list
        qres.aid2_score = None  # annotation score
        qres.filt2_meta = None  # messy

    @profile
    def load(qres, qresdir, verbose=utool.NOT_QUIET):
        """ Loads the result from the given database """
        fpath = qres.get_fpath(qresdir)
        qaid_good = qres.qaid
        qauuid_good = qres.qauuid
        try:
            #print('[qr] qres.load() fpath=%r' % (split(fpath)[1],))
            with open(fpath, 'rb') as file_:
                loaded_dict = cPickle.load(file_)
                qres.__dict__.update(loaded_dict)
            #if not isinstance(qres.filt2_meta, dict):
            #    print('[qr] loading old result format')
            #    qres.filt2_meta = {}
            if verbose:
                print('... qres cache hit: %r' % (split(fpath)[1],))
        except IOError as ex:
            if not exists(fpath):
                msg = '... qres cache miss: %r' % (split(fpath)[1],)
                if verbose:
                    print(msg)
                raise hsexcept.HotsCacheMissError(msg)
            msg = '[!qr] QueryResult(qaid=%d) is corrupt' % (qres.qaid)
            utool.printex(ex, msg, iswarning=True)
            raise hsexcept.HotsNeedsRecomputeError(msg)
        except BadZipFile as ex:
            msg = '[!qr] QueryResult(qaid=%d) has bad zipfile' % (qres.qaid)
            utool.printex(ex, msg, iswarning=True)
            if exists(fpath):
                print('[qr] Removing corrupted file: %r' % fpath)
                os.remove(fpath)
                raise hsexcept.HotsNeedsRecomputeError(msg)
            else:
                raise Exception(msg)
        except (ValueError, TypeError) as ex:
            utool.printex(ex, iswarning=True)
            exstr = str(ex)
            print(exstr)
            if exstr == 'unsupported pickle protocol: 3':
                raise hsexcept.HotsNeedsRecomputeError(str(ex))
            elif exstr.startswith('("\'numpy.ndarray\' object is not callable",'):
                raise hsexcept.HotsNeedsRecomputeError(str(ex))
            raise
        except Exception as ex:
            utool.printex(ex, 'unknown exception while loading query result')
            raise
        assert qauuid_good == qres.qauuid
        qres.qauuid = qauuid_good
        qres.qaid = qaid_good

    def __eq__(self, other):
        """ For testing """
        return all([
            self.qaid == other.qaid,
            self.cfgstr == other.cfgstr,
            self.eid == other.eid,
            _qres_dicteq(self.aid2_fm, other.aid2_fm),
            _qres_dicteq(self.aid2_fs, self.aid2_fs),
            _qres_dicteq(self.aid2_fk, self.aid2_fk),
            _qres_dicteq(self.aid2_score, other.aid2_score),
            _qres_dicteq(self.filt2_meta, other.filt2_meta),
        ])

    def has_cache(qres, qresdir):
        return exists(qres.get_fpath(qresdir))

    def get_fpath(qres, qresdir):
        return query_result_fpath(qresdir, qres.qaid, qres.qauuid, qres.cfgstr)

    @profile
    def save(qres, qresdir):
        fpath = qres.get_fpath(qresdir)
        if utool.NOT_QUIET:  # and utool.VERBOSE:
            print('[qr] cache save: %r' % (split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            cPickle.dump(qres.__dict__, file_)

    def cache_bytes(qres, qresdir):
        """ Size of the cached query result on disk """
        fpath  = qres.get_fpath(qresdir)
        nBytes = utool.file_bytes(fpath)
        return nBytes

    def get_fmatch_index(qres, aid, qfx):
        """ Returns the feature index in aid matching the query's qfx-th feature
            (if it exists)
        """
        fm = qres.aid2_fm[aid]
        mx_list = np.where(fm[:, 0] == qfx)[0]
        if len(mx_list) != 1:
            raise IndexError('[!qr] qfx=%r not found' % (qfx))
        else:
            mx = mx_list[0]
            return mx

    def get_match_tbldata(qres, ranks_lt=5):
        """ Returns matchinfo in table format (qaids, aids, scores, ranks) """
        aid_arr, score_arr = qres.get_aids_and_scores()
        # Sort the scores in rank order
        sortx = score_arr.argsort()[::-1]
        score_arr = score_arr[sortx]
        aid_arr   = aid_arr[sortx]
        rank_arr  = np.arange(sortx.size)
        # Return only rows where rank < ranks_lt
        isvalid = rank_arr < ranks_lt
        aids    =   aid_arr[isvalid]
        scores  = score_arr[isvalid]
        ranks   =  rank_arr[isvalid]
        qaids   = np.full(aids.shape, qres.qaid, dtype=aids.dtype)
        tbldata = (qaids, aids, scores, ranks)
        # DEBUG
        #column_lbls = ['qaids', 'aids', 'scores', 'ranks']
        #qaid_arr      = np.full(aid_arr.shape, qres.qaid, dtype=aid_arr.dtype)
        #tbldata2      = (qaid_arr, aid_arr, score_arr, rank_arr)
        #print(utool.make_csv_table(tbldata, column_lbls))
        #print(utool.make_csv_table(tbldata2, column_lbls))
        #utool.embed()
        return tbldata

    def get_aids_and_scores(qres):
        """ returns a chip index list and associated score list """
        aid_arr   = np.array(list(qres.aid2_score.keys()), dtype=np.int32)
        score_arr = np.array(list(qres.aid2_score.values()), dtype=np.float64)
        return aid_arr, score_arr

    def get_top_aids(qres, num=None):
        """ Returns a ranked list of chip indexes """
        # TODO: rename num to ranks_lt
        aid_arr, score_arr = qres.get_aids_and_scores()
        # Get chip-ids sorted by scores
        top_aids = aid_arr[score_arr.argsort()[::-1]]
        num_indexed = len(top_aids)
        if num is None:
            num = num_indexed
        return top_aids[0:min(num, num_indexed)]

    def get_aid_scores(qres, aid_arr):
        return [qres.aid2_score.get(aid, None) for aid in aid_arr]

    def get_aid_truth(qres, ibs, aid_list):
        # 0: false, 1: True, 2: unknown
        isgt_list = [ibs.get_match_truth(qres.qaid, aid) for aid in aid_list]
        return isgt_list

    def get_inspect_str(qres, ibs=None):
        assert_qres(qres)
        nFeatMatch_list = get_num_feats_in_matches(qres)
        nFeatMatch_stats = utool.mystats(nFeatMatch_list)

        top_lbls = [' top aids', ' scores', ' ranks']

        top_aids   = qres.get_top_aids(num=5)
        top_scores = qres.get_aid_scores(top_aids)
        top_ranks  = qres.get_aid_ranks(top_aids)
        top_list   = [top_aids, top_scores, top_ranks]

        if ibs is not None:
            top_lbls += [' isgt']
            istrue = qres.get_aid_truth(ibs, top_aids)
            top_list.append(istrue)

        top_stack = np.vstack(top_list)
        top_stack = np.array(top_stack, dtype=np.int32)
        top_str = str(top_stack)

        top_lbl = '\n'.join(top_lbls)
        inspect_list = ['QueryResult',
                        qres.cfgstr,
                        ]
        if ibs is not None:
            gt_ranks  = qres.get_gt_ranks(ibs=ibs)
            gt_scores = qres.get_gt_scores(ibs=ibs)
            inspect_list.append('gt_ranks = %r' % gt_ranks)
            inspect_list.append('gt_scores = %r' % gt_scores)

        inspect_list.extend([
            'qaid=%r ' % qres.qaid,
            utool.hz_str(top_lbl, ' ', top_str),
            'num Feat Matches stats:',
            utool.indent(utool.dict_str(nFeatMatch_stats)),
        ])

        inspect_str = '\n'.join(inspect_list)

        inspect_str = utool.indent(inspect_str, '[INSPECT] ')
        return inspect_str

    def get_aid_ranks(qres, aid_arr):
        """ get ranks of chip indexes in aid_arr """
        top_aids = qres.get_top_aids()
        foundpos = [np.where(top_aids == aid)[0] for aid in aid_arr]
        ranks_   = [ranks if len(ranks) > 0 else [None] for ranks in foundpos]
        assert all([len(ranks) == 1 for ranks in ranks_]), 'len(aid_ranks) != 1'
        rank_list = [ranks[0] for ranks in ranks_]
        return rank_list

    #def get_aid_ranks(qres, aid_arr):
        #'get ranks of chip indexes in aid_arr'
        #score_arr = np.array(qres.aid2_score.values())
        #aid_arr   = np.array(qres.aid2_score.keys())
        #top_aids = aid_arr[score_arr.argsort()[::-1]]
        #foundpos = [np.where(top_aids == aid)[0] for aid in aid_arr]
        #ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        #assert all([len(r) == 1 for r in ranks_])
        #rank_list = [r[0] for r in ranks_]
        #return rank_list

    def get_gt_ranks(qres, gt_aids=None, ibs=None, return_gtaids=False):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_aids is None and ibs is None:
            raise Exception('[qr] must pass in the gt_aids or ibs object')
        if gt_aids is None:
            gt_aids = ibs.get_annot_groundtruth(qres.qaid)
        gt_ranks = qres.get_aid_ranks(gt_aids)
        if return_gtaids:
            return gt_ranks, gt_aids
        else:
            return gt_ranks

    def get_gt_scores(qres, gt_aids=None, ibs=None, return_gtaids=False):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_aids is None and ibs is None:
            raise Exception('[qr] must pass in the gt_aids or ibs object')
        if gt_aids is None:
            gt_aids = ibs.get_annot_groundtruth(qres.qaid)
        gt_scores = qres.get_aid_scores(gt_aids)
        if return_gtaids:
            return gt_scores, gt_aids
        else:
            return gt_scores

    def get_best_gt_rank(qres, ibs):
        """ Returns the best rank over all the groundtruth """
        gt_ranks, gt_aids = qres.get_gt_ranks(ibs=ibs, return_gtaids=True)
        aidrank_tups = list(zip(gt_aids, gt_ranks))
        # Get only the aids that placed in the shortlist
        #valid_gtaids = np.array([ aid for aid, rank in aidrank_tups if rank is not None])
        valid_ranks  = np.array([rank for aid, rank in aidrank_tups if rank is not None])
        # Sort so lowest score is first
        best_rankx = valid_ranks.argsort()
        #best_gtaids  = best_gtaids[best_rankx]
        best_gtranks = valid_ranks[best_rankx]
        if len(best_gtranks) == 0:
            best_rank = -1
        else:
            best_rank = best_gtranks[0]
        return best_rank

    def get_classified_pos(qres):
        top_aids = np.array(qres.get_top_aids())
        pos_aids = top_aids[0:1]
        return pos_aids

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

    def get_matching_keypoints(qres, ibs, aid2_list):
        return qres_get_matching_keypoints(qres, ibs, aid2_list)
