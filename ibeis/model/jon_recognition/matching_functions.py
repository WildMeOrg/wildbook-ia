# TODO: These functions can go a shit-ton faster if they are put into list
# comprehensions
# TODO: Remove ibs control as much as possible or abstract it away
from __future__ import division, print_function
# Python
from itertools import izip
from collections import defaultdict
import sys
# Scientific
import numpy as np
from vtool import keypoint as ktool
# Hotspotter
from ibeis.model.jon_recognition import QueryResult
from ibeis.model.jon_recognition import coverage
from ibeis.model.jon_recognition import nn_filters
from ibeis.model.jon_recognition import spatial_verification2 as sv2
from ibeis.model.jon_recognition import voting_rules2 as vr2
import utool
print, print_,  printDBG, rrr, profile =\
    utool.inject(__name__, '[mf]', DEBUG=False)


QUIET = utool.QUIET or utool.get_flag('--quiet-query')


#=================
# Module Concepts
#=================
"""
PREFIXES:
qcid2_XXX - prefix mapping query chip index to
qfx2_XXX  - prefix mapping query chip feature index to

TUPLES:
 * nns    - a (qfx2_ax, qfx2_dist) tuple
 * nnfilt - a (qfx2_fs, qfx2_valid) tuple

SCALARS
 * ax     - the index into the database of features
 * dist   - the distance to a corresponding feature
 * fs     - a score of a corresponding feature
 * valid  - a valid bit for a corresponding feature

REALIZATIONS:
qcid2_nns - maping from query chip index to nns
{
 * qfx2_ax   - ranked list of query feature indexes to database feature indexes
 * qfx2_dist - ranked list of query feature indexes to database feature indexes
}

* qcid2_norm_weight - mapping from qcid to (qfx2_normweight, qfx2_selnorm)
         = qcid2_nnfilt[qcid]
"""
#=================
# Globals
#=================

MARK_AFTER = 2

#=================
# Helpers
#=================


def progress_func(maxval=0):
    mark_prog, end_prog = utool.progress_func(
        maxval, mark_after=MARK_AFTER, progress_type='fmtstr')
    return mark_prog, end_prog


class QueryException(Exception):
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(ibs, qcid):
    msg = ('QUERY ERROR IN %s: qcid=%r has no descriptors!' +
           'Please delete it.') % (ibs.dbfname, qcid)
    ex = QueryException(msg)
    return ex


#============================
# 1) Nearest Neighbors
#============================


def nearest_neighbors(ibs, qcids, qreq):
    """ Plain Nearest Neighbors """
    # Neareset neighbor configuration
    nn_cfg = qreq.cfg.nn_cfg
    K      = nn_cfg.K
    Knorm  = nn_cfg.Knorm
    checks = nn_cfg.checks
    uid_   = nn_cfg.get_uid()
    if not QUIET:
        print('[mf] Step 1) Assign nearest neighbors: ' + uid_)
    # Grab descriptors
    qfids = ibs.get_chip_fids(qcids)
    qdesc_list = ibs.get_feat_desc(qfids)

    # NNIndex
    flann = qreq.data_index.flann
    # Output
    qcid2_nns = {}
    nNN, nDesc = 0, 0
    mark_prog, end_prog = progress_func(len(qcids))
    for count, qcid in enumerate(qcids):
        mark_prog(count)
        qfx2_desc = qdesc_list[count]
        # Check that we can query this chip
        if len(qfx2_desc) == 0:
            # Raise error if strict
            if '--strict' in sys.argv:
                raise NoDescriptorsException(ibs, qcid)
            else:
                # Assign empty nearest neighbors
                empty_qfx2_ax   = np.empty((0, K + Knorm), dtype=np.int)
                empty_qfx2_dist = np.empty((0, K + Knorm), dtype=np.float)
                qcid2_nns[qcid] = (empty_qfx2_ax, empty_qfx2_dist)
                continue
        # Find Neareset Neighbors
        (qfx2_ax, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm,
                                              checks=checks)
        # Store nearest neighbors
        qcid2_nns[qcid] = (qfx2_ax, qfx2_dist)
        # record number of query and result desc
        nNN += qfx2_ax.size
        nDesc += len(qfx2_desc)
    end_prog()
    if not QUIET:
        print('[mf] * assigned %d desc from %d chips to %r nearest neighbors' % (nDesc, len(qcids), nNN))
    return qcid2_nns


#============================
# 2) Nearest Neighbor weights
#============================


def weight_neighbors(ibs, qcid2_nns, qreq):
    if not QUIET:
        print('[mf] Step 2) Weight neighbors: ' + qreq.cfg.filt_cfg.get_uid())
    if qreq.cfg.filt_cfg.filt_on:
        return _weight_neighbors(ibs, qcid2_nns, qreq)
    else:
        return  {}


def _weight_neighbors(ibs, qcid2_nns, qreq):
    nnfilter_list = qreq.cfg.filt_cfg.get_active_filters()
    filt2_weights = {}
    filt2_meta = {}
    for nnfilter in nnfilter_list:
        nn_filter_fn = nn_filters.NN_FILTER_FUNC_DICT[nnfilter]
        # Apply [nnfilter] weight to each nearest neighbor
        # TODO FIX THIS!
        qcid2_norm_weight, qcid2_selnorms = nn_filter_fn(ibs, qcid2_nns, qreq)
        filt2_weights[nnfilter] = qcid2_norm_weight
        filt2_meta[nnfilter] = qcid2_selnorms
    return filt2_weights, filt2_meta


#==========================
# 3) Neighbor scoring (Voting Profiles)
#==========================


def _apply_filter_scores(qcid, qfx2_nnax, filt2_weights, filt_cfg):
    qfx2_score = np.ones(qfx2_nnax.shape, dtype=QueryResult.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nnax.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, cid2_weights in filt2_weights.iteritems():
        qfx2_weights = cid2_weights[qcid]
        sign, thresh, weight = filt_cfg.get_stw(filt)
        if thresh is not None and thresh != 'None':
            thresh = float(thresh)  # corrects for thresh being strings sometimes
            if isinstance(thresh, (int, float)):
                qfx2_passed = sign * qfx2_weights <= sign * thresh
                qfx2_valid  = np.logical_and(qfx2_valid, qfx2_passed)
        if not weight == 0:
            qfx2_score += weight * qfx2_weights
    return qfx2_score, qfx2_valid


@profile
def filter_neighbors(ibs, qcid2_nns, filt2_weights, qreq):
    qcid2_nnfilt = {}
    # Configs
    filt_cfg = qreq.cfg.filt_cfg
    cant_match_sameimg  = not filt_cfg.can_match_sameimg
    cant_match_samename = not filt_cfg.can_match_samename
    K = qreq.cfg.nn_cfg.K
    if not QUIET:
        print('[mf] Step 3) Filter neighbors: ')
    #+ filt_cfg.get_uid())
    # NNIndex
    # Database feature index to chip index
    # Filter matches based on config and weights
    mark_prog, end_prog = progress_func(len(qcid2_nns))
    for count, qcid in enumerate(qcid2_nns.iterkeys()):
        mark_prog(count)
        (qfx2_ax, _) = qcid2_nns[qcid]
        qfx2_nnax = qfx2_ax[:, 0:K]
        # Get a numeric score score and valid flag for each feature match
        qfx2_score, qfx2_valid = _apply_filter_scores(
            qcid, qfx2_nnax, filt2_weights, filt_cfg)
        qfx2_cid = qreq.data_index.ax2_cid[qfx2_nnax]
        if not QUIET:
            print('[mf] * %d assignments are invalid by thresh' %
                  ((True - qfx2_valid).sum()))
        # Remove Impossible Votes:
        # dont vote for yourself or another chip in the same image
        cant_match_self = not cant_match_sameimg
        if cant_match_self:
            ####DBG
            qfx2_notsamechip = qfx2_cid != qcid
            nChip_all_invalid = ((True - qfx2_notsamechip)).sum()
            nChip_new_invalid = (qfx2_valid * (True - qfx2_notsamechip)).sum()
            if not QUIET:
                print('[mf] * %d assignments are invalid by self' % nChip_all_invalid)
                print('[mf] * %d are newly invalided by self' % nChip_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        if cant_match_sameimg:
            qfx2_gid = ibs.get_chip_gids(qfx2_cid)
            qgid =  ibs.get_chip_gids(qcid)
            qfx2_notsameimg = qfx2_gid != qgid
            ####DBG
            nImg_all_invalid = ((True - qfx2_notsameimg)).sum()
            nImg_new_invalid = (qfx2_valid * (True - qfx2_notsameimg)).sum()
            if not QUIET:
                print('[mf] * %d assignments are invalid by gid' % nImg_all_invalid)
                print('[mf] * %d are newly invalided by gid' % nImg_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsameimg)
        if cant_match_samename:
            qfx2_nid = ibs.get_chip_nids(qfx2_cid)
            qnid = ibs.get_chip_nids(qcid)
            qfx2_notsamename = qfx2_nid != qnid
            ####DBG
            nName_all_invalid = ((True - qfx2_notsamename)).sum()
            nName_new_invalid = (qfx2_valid * (True - qfx2_notsamename)).sum()
            if not QUIET:
                print('[mf] * %d assignments are invalid by nid' % nName_all_invalid)
                print('[mf] * %d are newly invalided by nid' % nName_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamename)
        printDBG('[mf] * Marking %d assignments as invalid' % ((True - qfx2_valid).sum()))
        qcid2_nnfilt[qcid] = (qfx2_score, qfx2_valid)
    end_prog()
    return qcid2_nnfilt


def identity_filter(qcid2_nns, qreq):
    """ testing function returns unfiltered nearest neighbors
    this does check that you are not matching yourself
    """
    qcid2_nnfilt = {}
    K = qreq.cfg.nn_cfg.K
    for count, qcid in enumerate(qcid2_nns.iterkeys()):
        (qfx2_ax, _) = qcid2_nns[qcid]
        qfx2_nnax = qfx2_ax[:, 0:K]
        qfx2_score = np.ones(qfx2_nnax.shape, dtype=QueryResult.FS_DTYPE)
        qfx2_valid = np.ones(qfx2_nnax.shape, dtype=np.bool)
        # Check that you are not matching yourself
        qfx2_cid = qreq.data_index.ax2_cid[qfx2_nnax]
        qfx2_notsamechip = qfx2_cid != qcid
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        qcid2_nnfilt[qcid] = (qfx2_score, qfx2_valid)

    return qcid2_nnfilt


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> cid2
#============================


def _fix_fmfsfk(cid2_fm, cid2_fs, cid2_fk):
    minMatches = 2  # TODO: paramaterize
    # Convert to numpy
    fm_dtype = QueryResult.FM_DTYPE
    fs_dtype = QueryResult.FS_DTYPE
    fk_dtype = QueryResult.FK_DTYPE
    cid2_fm_ = {cid: np.array(fm, fm_dtype)
                for cid, fm in cid2_fm.iteritems()
                if len(fm) > minMatches}
    cid2_fs_ = {cid: np.array(fs, fs_dtype)
                for cid, fs in cid2_fs.iteritems()
                if len(fs) > minMatches}
    cid2_fk_ = {cid: np.array(fk, fk_dtype)
                for cid, fk in cid2_fk.iteritems()
                if len(fk) > minMatches}
    # Ensure shape
    for cid, fm in cid2_fm_.iteritems():
        fm.shape = (fm.size // 2, 2)
    chipmatch = (cid2_fm_, cid2_fs_, cid2_fk_)
    return chipmatch


def new_fmfsfk():
    cid2_fm = defaultdict(list)
    cid2_fs = defaultdict(list)
    cid2_fk = defaultdict(list)
    return cid2_fm, cid2_fs, cid2_fk


@profile
def build_chipmatches(qcid2_nns, qcid2_nnfilt, qreq):
    '''vsmany/vsone counts here. also this is where the filter
    weights and thershold are applied to the matches. Essientally
    nearest neighbors are converted into weighted assignments'''
    # Config
    K = qreq.cfg.nn_cfg.K
    query_type = qreq.cfg.agg_cfg.query_type
    is_vsone = query_type == 'vsone'
    if not QUIET:
        print('[mf] Step 4) Building chipmatches %s' % (query_type,))
    # Return var
    qcid2_chipmatch = {}

    #Vsone
    if is_vsone:
        assert len(qreq.qcids) == 1
        cid2_fm, cid2_fs, cid2_fk = new_fmfsfk()

    # Iterate over chips with nearest neighbors
    mark_prog, end_prog = progress_func(len(qcid2_nns))
    for count, qcid in enumerate(qcid2_nns.iterkeys()):
        mark_prog(count)
        (qfx2_ax, _) = qcid2_nns[qcid]
        (qfx2_fs, qfx2_valid) = qcid2_nnfilt[qcid]
        nQKpts = len(qfx2_ax)
        # Build feature matches
        qfx2_nnax = qfx2_ax[:, 0:K]
        qfx2_cid  = qreq.data_index.ax2_cid[qfx2_nnax]
        qfx2_fx   = qreq.data_index.ax2_fx[qfx2_nnax]
        qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
        # Pack valid feature matches into an interator
        valid_lists = [qfx2[qfx2_valid] for qfx2 in (qfx2_qfx, qfx2_cid, qfx2_fx, qfx2_fs, qfx2_k,)]
        match_iter = izip(*valid_lists)
        # Vsmany - Append query feature matches to database cids
        if not is_vsone:
            cid2_fm, cid2_fs, cid2_fk = new_fmfsfk()
            for qfx, cid, fx, fs, fk in match_iter:
                cid2_fm[cid].append((qfx, fx))  # Note the difference
                cid2_fs[cid].append(fs)
                cid2_fk[cid].append(fk)
            chipmatch = _fix_fmfsfk(cid2_fm, cid2_fs, cid2_fk)
            qcid2_chipmatch[qcid] = chipmatch
            #if not QUIET:
                #nFeats_in_matches = [len(fm) for fm in cid2_fm.itervalues()]
                #print('nFeats_in_matches_stats = ' + utool.dict_str(utool.mystats(nFeats_in_matches)))
        # Vsone - Append database feature matches to query cids
        else:
            for qfx, cid, fx, fs, fk in match_iter:
                cid2_fm[qcid].append((fx, qfx))  # Note the difference
                cid2_fs[qcid].append(fs)
                cid2_fk[qcid].append(fk)
    #Vsone
    if is_vsone:
        chipmatch = _fix_fmfsfk(cid2_fm, cid2_fs, cid2_fk)
        qcid = qreq.qcids[0]
        qcid2_chipmatch[qcid] = chipmatch

    end_prog()
    return qcid2_chipmatch


#============================
# 5) Spatial Verification
#============================


def spatial_verification(ibs, qcid2_chipmatch, qreq, dbginfo=False):
    sv_cfg = qreq.cfg.sv_cfg
    if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
        print('[mf] Step 5) Spatial verification: off')
        return qcid2_chipmatch, {} if dbginfo else qcid2_chipmatch
    else:
        return spatial_verification_(ibs, qcid2_chipmatch, qreq, dbginfo=dbginfo)


def spatial_verification_(ibs, qcid2_chipmatch, qreq, dbginfo=False):
    sv_cfg = qreq.cfg.sv_cfg
    print('[mf] Step 5) Spatial verification: ' + sv_cfg.get_uid())
    prescore_method = sv_cfg.prescore_method
    nShortlist      = sv_cfg.nShortlist
    xy_thresh       = sv_cfg.xy_thresh
    min_scale = sv_cfg.scale_thresh_low
    max_scale = sv_cfg.scale_thresh_high
    use_chip_extent = sv_cfg.use_chip_extent
    min_nInliers    = sv_cfg.min_nInliers
    just_affine     = sv_cfg.just_affine
    qcid2_chipmatchSV = {}
    if dbginfo:
        qcid2_svtups = {}  # dbg info (can remove if there is a speed issue)
    def print_(msg, count=0):
        """ temp print_. Using count in this way is a hack """
        if not QUIET:
            if count % 25 == 0:
                sys.stdout.write(msg)
            count += 1
    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    for qcid in qcid2_chipmatch.iterkeys():
        chipmatch = qcid2_chipmatch[qcid]
        cid2_prescore = score_chipmatch(ibs, qcid, chipmatch, prescore_method, qreq)
        (cid2_fm, cid2_fs, cid2_fk) = chipmatch
        topx2_cid = utool.util_dict.keys_sorted_by_value(cid2_prescore)[::-1]
        nRerank = min(len(topx2_cid), nShortlist)
        # Precompute output container
        if dbginfo:
            cid2_svtup = {}  # dbg info (can remove if there is a speed issue)
        cid2_fm_V, cid2_fs_V, cid2_fk_V = new_fmfsfk()
        # Query Keypoints
        kpts1 = ibs.get_chip_kpts(qcid)
        topx2_kpts = ibs.get_chip_kpts(topx2_cid)
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = _precompute_topx2_dlen_sqrd(ibs, cid2_fm, topx2_cid,
                                                      topx2_kpts, nRerank,
                                                      use_chip_extent)
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(nRerank):
            cid = topx2_cid[topx]
            fm = cid2_fm[cid]
            dlen_sqrd = topx2_dlen_sqrd[topx]
            kpts2 = topx2_kpts[topx]
            fs    = cid2_fs[cid]
            fk    = cid2_fk[cid]
            sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh,
                                            max_scale, min_scale, dlen_sqrd,
                                            min_nInliers, just_affine)
            if sv_tup is None:
                    print_('o')  # sv failure
            else:
                # Return the inliers to the homography
                (H, inliers) = sv_tup
                cid2_svtup[cid] = sv_tup
                cid2_fm_V[cid] = fm[inliers, :]
                cid2_fs_V[cid] = fs[inliers]
                cid2_fk_V[cid] = fk[inliers]
                if not QUIET:
                    #print(inliers)
                    print_('.')  # verified something
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(cid2_fm_V, cid2_fs_V, cid2_fk_V)
        if dbginfo:
            qcid2_svtups[qcid] = cid2_svtup
        qcid2_chipmatchSV[qcid] = chipmatchSV
    print_('\n')
    if not QUIET:
        print('[mf] Finished sv')
    if dbginfo:
        return qcid2_chipmatchSV, qcid2_svtups
    else:
        return qcid2_chipmatchSV


def _precompute_topx2_dlen_sqrd(ibs, cid2_fm, topx2_cid, topx2_kpts,
                                nRerank, use_chip_extent):
    '''helper for spatial verification, computes the squared diagonal length of
    matching chips'''
    if use_chip_extent:
        topx2_chipsize = list(ibs.get_chip_sizes(topx2_cid))
        def chip_dlen_sqrd(tx):
            (chipw, chiph) = topx2_chipsize[tx]
            dlen_sqrd = chipw ** 2 + chiph ** 2
            return dlen_sqrd
        topx2_dlen_sqrd = [chip_dlen_sqrd(tx) for tx in xrange(nRerank)]
    else:
        # Use extent of matching keypoints
        def kpts_dlen_sqrd(tx):
            kpts2 = topx2_kpts[tx]
            cid = topx2_cid[tx]
            fm    = cid2_fm[cid]
            x_m, y_m = ktool.get_xys(kpts2[fm[:, 1]])
            dlensqrd = (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            return dlensqrd
        topx2_dlen_sqrd = [kpts_dlen_sqrd(tx) for tx in xrange(nRerank)]
    return topx2_dlen_sqrd


#============================
# 6) QueryResult Format
#============================


@profile
def chipmatch_to_resdict(ibs, qcid2_chipmatch, filt2_meta, qreq):
    if not QUIET:
        print('[mf] Step 6) Convert chipmatch -> res')
    uid = qreq.get_uid()
    score_method = qreq.cfg.agg_cfg.score_method
    # Create the result structures for each query.
    qcid2_res = {}
    for qcid in qcid2_chipmatch.iterkeys():
        # For each query's chipmatch
        chipmatch = qcid2_chipmatch[qcid]
        # Perform final scoring
        cid2_score = score_chipmatch(ibs, qcid, chipmatch, score_method, qreq)
        # Create a query result structure
        res = QueryResult.QueryResult(qcid, uid)
        res.cid2_score = cid2_score
        (res.cid2_fm, res.cid2_fs, res.cid2_fk) = chipmatch
        res.filt2_meta = {}  # dbgstats
        for filt, qcid2_meta in filt2_meta.iteritems():
            res.filt2_meta[filt] = qcid2_meta[qcid]  # things like k+1th
        qcid2_res[qcid] = res
    # Retain original score method
    return qcid2_res


def load_resdict(ibs, qreq):
    # Load the result structures for each query.
    qcids = qreq.qcids
    uid = qreq.get_uid()
    ##IF DICT_COMPREHENSION
    qcid2_res = {qcid: QueryResult.QueryResult(qcid, uid) for qcid in iter(qcids)}
    [res.load(ibs) for res in qcid2_res.itervalues()]
    ##ELSE
    #qcid2_res = {}
    #for qcid in qcids:
        #res = QueryResult.QueryResult(qcid, uid)
        #res.load(ibs)
        #qcid2_res[qcid] = res
    ##ENDIF
    return qcid2_res


def try_load_resdict(ibs, qreq):
    # Load the result structures for each query.
    qcids = qreq.qcids
    uid = qreq.get_uid()
    qcid2_res = {}
    failed_qcids = []
    for qcid in qcids:
        try:
            res = QueryResult.QueryResult(qcid, uid)
            res.load(ibs)
            qcid2_res[qcid] = res
        except IOError:
            failed_qcids.append(qcid)
    return qcid2_res, failed_qcids


#============================
# Scoring Mechanism
#============================


def score_chipmatch(ibs, qcid, chipmatch, score_method, qreq=None):
    (cid2_fm, cid2_fs, cid2_fk) = chipmatch
    # HACK: Im not even sure if the 'w' suffix is correctly handled anymore
    if score_method.find('w') == len(score_method) - 1:
        score_method = score_method[:-1]
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        cid2_score = vr2.score_chipmatch_csum(chipmatch)
    elif score_method == 'pl':
        cid2_score, nid2_score = vr2.score_chipmatch_PL(ibs, qcid, chipmatch, qreq)
    elif score_method == 'borda':
        cid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qcid, chipmatch, qreq, 'borda')
    elif score_method == 'topk':
        cid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qcid, chipmatch, qreq, 'topk')
    elif score_method.startswith('coverage'):
        # Method num is at the end of coverage
        method = int(score_method.replace('coverage', '0'))
        cid2_score = coverage.score_chipmatch_coverage(ibs, qcid, chipmatch, qreq, method=method)
    else:
        raise Exception('[mf] unknown scoring method:' + score_method)
    return cid2_score
