# TODO: These functions can go a shit-ton faster if they are put into list
# comprehensions
# TODO: Remove ibs control as much as possible or abstract it away
from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
from collections import defaultdict
import sys
# Scientific
import numpy as np
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import spatial_verification as sver
# Hotspotter
from ibeis.model.hots import QueryResult
from ibeis.model.hots import coverage_image
from ibeis.model.hots import nn_filters
from ibeis.model.hots import voting_rules2 as vr2
import utool
print, print_,  printDBG, rrr, profile =\
    utool.inject(__name__, '[mf]', DEBUG=False)


np.tau = 2 * np.pi  # tauday.com
QUIET = utool.QUIET or utool.get_flag('--quiet-query')
VERBOSE = utool.VERBOSE or utool.get_flag('--verbose-query')


#=================
# Module Concepts
#=================
"""
PREFIXES:
qrid2_XXX - prefix mapping query chip index to
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
qrid2_nns - maping from query chip index to nns
{
 * qfx2_ax   - ranked list of query feature indexes to database feature indexes
 * qfx2_dist - ranked list of query feature indexes to database feature indexes
}

* qrid2_norm_weight - mapping from qrid to (qfx2_normweight, qfx2_selnorm)
         = qrid2_nnfilt[qrid]
"""
#=================
# Globals
#=================

MARK_AFTER = 2

#=================
# Helpers
#=================


def progress_func(maxval=0, lbl='Match Progress: '):
    mark_prog, end_prog = utool.progress_func(
        maxval, mark_after=MARK_AFTER, progress_type='fmtstr', lbl=lbl)
    return mark_prog, end_prog


class QueryException(Exception):
    def __init__(self, msg):
        super(QueryException, self).__init__(msg)


def NoDescriptorsException(ibs, qrid):
    msg = ('QUERY ERROR IN %s: qrid=%r has no descriptors!' +
           'Please delete it.') % (ibs.get_dbname(), qrid)
    ex = QueryException(msg)
    return ex


#============================
# 1) Nearest Neighbors
#============================


@profile
def nearest_neighbors(ibs, qrids, qreq):
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
    qdesc_list = ibs.get_roi_desc(qrids)
    # NNIndex
    flann = qreq.data_index.flann
    # Output
    qrid2_nns = {}
    nNN, nDesc = 0, 0
    mark_prog, end_prog = progress_func(len(qrids), lbl='Assign NN: ')
    for count, qrid in enumerate(qrids):
        mark_prog(count)
        qfx2_desc = qdesc_list[count]
        # Check that we can query this chip
        if len(qfx2_desc) == 0:
            # Raise error if strict
            if '--strict' in sys.argv:
                raise NoDescriptorsException(ibs, qrid)
            else:
                # Assign empty nearest neighbors
                empty_qfx2_ax   = np.empty((0, K + Knorm), dtype=np.int)
                empty_qfx2_dist = np.empty((0, K + Knorm), dtype=np.float)
                qrid2_nns[qrid] = (empty_qfx2_ax, empty_qfx2_dist)
                continue
        # Find Neareset Neighbors
        (qfx2_ax, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm,
                                              checks=checks)
        # Store nearest neighbors
        qrid2_nns[qrid] = (qfx2_ax, qfx2_dist)
        # record number of query and result desc
        nNN += qfx2_ax.size
        nDesc += len(qfx2_desc)
    end_prog()
    if not QUIET:
        print('[mf] * assigned %d desc from %d chips to %r nearest neighbors' % (nDesc, len(qrids), nNN))
    return qrid2_nns


#============================
# 2) Nearest Neighbor weights
#============================


def weight_neighbors(ibs, qrid2_nns, qreq):
    if not QUIET:
        print('[mf] Step 2) Weight neighbors: ' + qreq.cfg.filt_cfg.get_uid())
    if qreq.cfg.filt_cfg.filt_on:
        return _weight_neighbors(ibs, qrid2_nns, qreq)
    else:
        return  {}


@profile
def _weight_neighbors(ibs, qrid2_nns, qreq):
    nnfilter_list = qreq.cfg.filt_cfg.get_active_filters()
    filt2_weights = {}
    filt2_meta = {}
    for nnfilter in nnfilter_list:
        nn_filter_fn = nn_filters.NN_FILTER_FUNC_DICT[nnfilter]
        # Apply [nnfilter] weight to each nearest neighbor
        # TODO FIX THIS!
        qrid2_norm_weight, qrid2_selnorms = nn_filter_fn(ibs, qrid2_nns, qreq)
        filt2_weights[nnfilter] = qrid2_norm_weight
        filt2_meta[nnfilter] = qrid2_selnorms
    return filt2_weights, filt2_meta


#==========================
# 3) Neighbor scoring (Voting Profiles)
#==========================


@profile
def _apply_filter_scores(qrid, qfx2_nnax, filt2_weights, filt_cfg):
    qfx2_score = np.ones(qfx2_nnax.shape, dtype=QueryResult.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nnax.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, rid2_weights in filt2_weights.iteritems():
        qfx2_weights = rid2_weights[qrid]
        sign, thresh, weight = filt_cfg.get_stw(filt)  # stw = sign, thresh, weight
        if thresh is not None and thresh != 'None':
            thresh = float(thresh)  # corrects for thresh being strings sometimes
            if isinstance(thresh, (int, float)):
                qfx2_passed = sign * qfx2_weights <= sign * thresh
                qfx2_valid  = np.logical_and(qfx2_valid, qfx2_passed)
        if not weight == 0:
            qfx2_score += weight * qfx2_weights

    return qfx2_score, qfx2_valid


@profile
def filter_neighbors(ibs, qrid2_nns, filt2_weights, qreq):
    qrid2_nnfilt = {}
    # Configs
    filt_cfg = qreq.cfg.filt_cfg
    cant_match_sameimg  = not filt_cfg.can_match_sameimg
    cant_match_samename = not filt_cfg.can_match_samename
    K = qreq.cfg.nn_cfg.K
    if not QUIET:
        print('[mf] Step 3) Filter neighbors: ')
    if filt_cfg.gravity_weighting:
        # We dont have an easy way to access keypoints from nearest neighbors yet
        rid_list = np.unique(qreq.data_index.ax2_rid)  # FIXME: Highly inefficient
        kpts_list = ibs.get_roi_kpts(rid_list)
        ax2_kpts = np.vstack(kpts_list)
        ax2_oris = ktool.get_oris(ax2_kpts)
        assert len(ax2_oris) == len(qreq.data_index.ax2_data)
    # Filter matches based on config and weights
    mark_prog, end_prog = progress_func(len(qrid2_nns), lbl='Filter NN: ')
    for count, qrid in enumerate(qrid2_nns.iterkeys()):
        mark_prog(count)
        (qfx2_ax, _) = qrid2_nns[qrid]
        qfx2_nnax = qfx2_ax[:, 0:K]
        # Get a numeric score score and valid flag for each feature match
        qfx2_score, qfx2_valid = _apply_filter_scores(
            qrid, qfx2_nnax, filt2_weights, filt_cfg)
        qfx2_rid = qreq.data_index.ax2_rid[qfx2_nnax]
        if VERBOSE:
            print('[mf] * %d assignments are invalid by thresh' %
                  ((True - qfx2_valid).sum()))
        if filt_cfg.gravity_weighting:
            qfx2_nnori = ax2_oris[qfx2_nnax]
            qfx2_kpts  = ibs.get_roi_kpts(qrid)  # FIXME: Highly inefficient
            qfx2_oris  = ktool.get_oris(qfx2_kpts)
            # Get the orientation distance
            qfx2_oridist = ltool.rowwise_oridist(qfx2_nnori, qfx2_oris)
            # Normalize into a weight (close orientations are 1, far are 0)
            qfx2_gvweight = (np.tau - qfx2_oridist) / np.tau
            # Apply gravity vector weight to the score
            qfx2_score *= qfx2_gvweight
        # Remove Impossible Votes:
        # dont vote for yourself or another chip in the same image
        cant_match_self = not cant_match_sameimg
        if cant_match_self:
            ####DBG
            qfx2_notsamechip = qfx2_rid != qrid
            if VERBOSE:
                nChip_all_invalid = ((True - qfx2_notsamechip)).sum()
                nChip_new_invalid = (qfx2_valid * (True - qfx2_notsamechip)).sum()
                print('[mf] * %d assignments are invalid by self' % nChip_all_invalid)
                print('[mf] * %d are newly invalided by self' % nChip_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        if cant_match_sameimg:
            qfx2_gid = ibs.get_roi_gids(qfx2_rid)
            qgid     = ibs.get_roi_gids(qrid)
            qfx2_notsameimg = qfx2_gid != qgid
            ####DBG
            if VERBOSE:
                nImg_all_invalid = ((True - qfx2_notsameimg)).sum()
                nImg_new_invalid = (qfx2_valid * (True - qfx2_notsameimg)).sum()
                print('[mf] * %d assignments are invalid by gid' % nImg_all_invalid)
                print('[mf] * %d are newly invalided by gid' % nImg_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsameimg)
        if cant_match_samename:
            qfx2_nid = ibs.get_roi_nids(qfx2_rid)
            qnid = ibs.get_roi_nids(qrid)
            qfx2_notsamename = qfx2_nid != qnid
            ####DBG
            if VERBOSE:
                nName_all_invalid = ((True - qfx2_notsamename)).sum()
                nName_new_invalid = (qfx2_valid * (True - qfx2_notsamename)).sum()
                print('[mf] * %d assignments are invalid by nid' % nName_all_invalid)
                print('[mf] * %d are newly invalided by nid' % nName_new_invalid)
            ####
            qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamename)
        printDBG('[mf] * Marking %d assignments as invalid' % ((True - qfx2_valid).sum()))
        qrid2_nnfilt[qrid] = (qfx2_score, qfx2_valid)
    end_prog()
    return qrid2_nnfilt


@profile
def identity_filter(qrid2_nns, qreq):
    """ testing function returns unfiltered nearest neighbors
    this does check that you are not matching yourself
    """
    qrid2_nnfilt = {}
    K = qreq.cfg.nn_cfg.K
    for count, qrid in enumerate(qrid2_nns.iterkeys()):
        (qfx2_ax, _) = qrid2_nns[qrid]
        qfx2_nnax = qfx2_ax[:, 0:K]
        qfx2_score = np.ones(qfx2_nnax.shape, dtype=QueryResult.FS_DTYPE)
        qfx2_valid = np.ones(qfx2_nnax.shape, dtype=np.bool)
        # Check that you are not matching yourself
        qfx2_rid = qreq.data_index.ax2_rid[qfx2_nnax]
        qfx2_notsamechip = qfx2_rid != qrid
        qfx2_valid = np.logical_and(qfx2_valid, qfx2_notsamechip)
        qrid2_nnfilt[qrid] = (qfx2_score, qfx2_valid)

    return qrid2_nnfilt


#============================
# 4) Conversion from featurematches to chipmatches qfx2 -> rid2
#============================


@profile
def _fix_fmfsfk(rid2_fm, rid2_fs, rid2_fk):
    minMatches = 2  # TODO: paramaterize
    # Convert to numpy
    fm_dtype = QueryResult.FM_DTYPE
    fs_dtype = QueryResult.FS_DTYPE
    fk_dtype = QueryResult.FK_DTYPE
    # FIXME: This is slow
    rid2_fm_ = {rid: np.array(fm, fm_dtype)
                for rid, fm in rid2_fm.iteritems()
                if len(fm) > minMatches}
    rid2_fs_ = {rid: np.array(fs, fs_dtype)
                for rid, fs in rid2_fs.iteritems()
                if len(fs) > minMatches}
    rid2_fk_ = {rid: np.array(fk, fk_dtype)
                for rid, fk in rid2_fk.iteritems()
                if len(fk) > minMatches}
    # Ensure shape
    for rid, fm in rid2_fm_.iteritems():
        fm.shape = (fm.size // 2, 2)
    chipmatch = (rid2_fm_, rid2_fs_, rid2_fk_)
    return chipmatch


def new_fmfsfk():
    rid2_fm = defaultdict(list)
    rid2_fs = defaultdict(list)
    rid2_fk = defaultdict(list)
    return rid2_fm, rid2_fs, rid2_fk


@profile
def build_chipmatches(qrid2_nns, qrid2_nnfilt, qreq):
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
    qrid2_chipmatch = {}

    nFeatMatches = 0
    #Vsone
    if is_vsone:
        assert len(qreq.qrids) == 1
        rid2_fm, rid2_fs, rid2_fk = new_fmfsfk()

    # Iterate over chips with nearest neighbors
    mark_prog, end_prog = progress_func(len(qrid2_nns), 'Build Chipmatch: ')
    for count, qrid in enumerate(qrid2_nns.iterkeys()):
        mark_prog(count)
        (qfx2_ax, _) = qrid2_nns[qrid]
        (qfx2_fs, qfx2_valid) = qrid2_nnfilt[qrid]
        nQKpts = len(qfx2_ax)
        # Build feature matches
        qfx2_nnax = qfx2_ax[:, 0:K]
        qfx2_rid  = qreq.data_index.ax2_rid[qfx2_nnax]
        qfx2_fx   = qreq.data_index.ax2_fx[qfx2_nnax]
        qfx2_qfx = np.tile(np.arange(nQKpts), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQKpts, 1))
        # Pack valid feature matches into an interator
        valid_lists = [qfx2[qfx2_valid] for qfx2 in (qfx2_qfx, qfx2_rid, qfx2_fx, qfx2_fs, qfx2_k,)]
        # TODO: Sorting the valid lists by rid might help the speed of this
        # code. Also, consolidating fm, fs, and fk into one vector will reduce
        # the amount of appends.
        match_iter = izip(*valid_lists)
        # Vsmany - Append query feature matches to database rids
        if not is_vsone:
            rid2_fm, rid2_fs, rid2_fk = new_fmfsfk()
            for qfx, rid, fx, fs, fk in match_iter:
                rid2_fm[rid].append((qfx, fx))  # Note the difference
                rid2_fs[rid].append(fs)
                rid2_fk[rid].append(fk)
                nFeatMatches += 1
            chipmatch = _fix_fmfsfk(rid2_fm, rid2_fs, rid2_fk)
            qrid2_chipmatch[qrid] = chipmatch
            #if not QUIET:
                #nFeats_in_matches = [len(fm) for fm in rid2_fm.itervalues()]
                #print('nFeats_in_matches_stats = ' + utool.dict_str(utool.mystats(nFeats_in_matches)))
        # Vsone - Append database feature matches to query rids
        else:
            for qfx, rid, fx, fs, fk in match_iter:
                rid2_fm[qrid].append((fx, qfx))  # Note the difference
                rid2_fs[qrid].append(fs)
                rid2_fk[qrid].append(fk)
                nFeatMatches += 1
    #Vsone
    if is_vsone:
        chipmatch = _fix_fmfsfk(rid2_fm, rid2_fs, rid2_fk)
        qrid = qreq.qrids[0]
        qrid2_chipmatch[qrid] = chipmatch
    end_prog()
    if not QUIET:
        print('[mf] * made %d feat matches' % nFeatMatches)
    return qrid2_chipmatch


#============================
# 5) Spatial Verification
#============================


def spatial_verification(ibs, qrid2_chipmatch, qreq, dbginfo=False):
    sv_cfg = qreq.cfg.sv_cfg
    if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
        print('[mf] Step 5) Spatial verification: off')
        return qrid2_chipmatch, {} if dbginfo else qrid2_chipmatch
    else:
        return spatial_verification_(ibs, qrid2_chipmatch, qreq, dbginfo=dbginfo)


@profile
def spatial_verification_(ibs, qrid2_chipmatch, qreq, dbginfo=False):
    sv_cfg = qreq.cfg.sv_cfg
    print('[mf] Step 5) Spatial verification: ' + sv_cfg.get_uid())
    prescore_method = sv_cfg.prescore_method
    nShortlist      = sv_cfg.nShortlist
    xy_thresh       = sv_cfg.xy_thresh
    scale_thresh    = sv_cfg.scale_thresh
    ori_thresh      = sv_cfg.ori_thresh
    use_chip_extent = sv_cfg.use_chip_extent
    min_nInliers    = sv_cfg.min_nInliers
    qrid2_chipmatchSV = {}
    nFeatSVTotal = 0
    nFeatMatchSV = 0
    nFeatMatchSVAff = 0
    if dbginfo:
        qrid2_svtups = {}  # dbg info (can remove if there is a speed issue)
    def print_(msg, count=0):
        """ temp print_. Using count in this way is a hack """
        if not QUIET:
            if count % 25 == 0:
                sys.stdout.write(msg)
            count += 1
    # Find a transform from chip2 to chip1 (the old way was 1 to 2)
    for qrid in qrid2_chipmatch.iterkeys():
        chipmatch = qrid2_chipmatch[qrid]
        rid2_prescore = score_chipmatch(ibs, qrid, chipmatch, prescore_method, qreq)
        #print('Prescore: %r' % (rid2_prescore,))
        (rid2_fm, rid2_fs, rid2_fk) = chipmatch
        topx2_rid = utool.util_dict.keys_sorted_by_value(rid2_prescore)[::-1]
        nRerank = min(len(topx2_rid), nShortlist)
        # Precompute output container
        if dbginfo:
            rid2_svtup = {}  # dbg info (can remove if there is a speed issue)
        rid2_fm_V, rid2_fs_V, rid2_fk_V = new_fmfsfk()
        # Query Keypoints
        kpts1 = ibs.get_roi_kpts(qrid)
        topx2_kpts = ibs.get_roi_kpts(topx2_rid)
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = _precompute_topx2_dlen_sqrd(ibs, rid2_fm, topx2_rid,
                                                      topx2_kpts, nRerank,
                                                      use_chip_extent)
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(nRerank):
            rid = topx2_rid[topx]
            fm = rid2_fm[rid]
            dlen_sqrd = topx2_dlen_sqrd[topx]
            kpts2 = topx2_kpts[topx]
            fs    = rid2_fs[rid]
            fk    = rid2_fk[rid]
            sv_tup = sver.spatial_verification(kpts1, kpts2, fm,
                                               xy_thresh, scale_thresh, ori_thresh, dlen_sqrd,
                                               min_nInliers)
            nFeatSVTotal += len(fm)
            if sv_tup is None:
                    print_('o')  # sv failure
            else:
                # Return the inliers to the homography
                homog_inliers, H, aff_inliers, Aff = sv_tup
                if dbginfo:
                    rid2_svtup[rid] = sv_tup
                rid2_fm_V[rid] = fm[homog_inliers, :]
                rid2_fs_V[rid] = fs[homog_inliers]
                rid2_fk_V[rid] = fk[homog_inliers]
                nFeatMatchSV += len(homog_inliers)
                nFeatMatchSVAff += len(aff_inliers)
                if not QUIET:
                    #print(inliers)
                    print_('.')  # verified something
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(rid2_fm_V, rid2_fs_V, rid2_fk_V)
        if dbginfo:
            qrid2_svtups[qrid] = rid2_svtup
        qrid2_chipmatchSV[qrid] = chipmatchSV
    print_('\n')
    if not QUIET:
        print('[mf] * Affine verified %d/%d feat matches' % (nFeatMatchSVAff, nFeatSVTotal))
        print('[mf] * Homog  verified %d/%d feat matches' % (nFeatMatchSV, nFeatSVTotal))
    if dbginfo:
        return qrid2_chipmatchSV, qrid2_svtups
    else:
        return qrid2_chipmatchSV


def _precompute_topx2_dlen_sqrd(ibs, rid2_fm, topx2_rid, topx2_kpts,
                                nRerank, use_chip_extent):
    '''helper for spatial verification, computes the squared diagonal length of
    matching chips'''
    if use_chip_extent:
        topx2_chipsize = list(ibs.get_roi_chipsizes(topx2_rid))
        def chip_dlen_sqrd(tx):
            (chipw, chiph) = topx2_chipsize[tx]
            dlen_sqrd = chipw ** 2 + chiph ** 2
            return dlen_sqrd
        topx2_dlen_sqrd = [chip_dlen_sqrd(tx) for tx in xrange(nRerank)]
    else:
        # Use extent of matching keypoints
        def kpts_dlen_sqrd(tx):
            kpts2 = topx2_kpts[tx]
            rid = topx2_rid[tx]
            fm    = rid2_fm[rid]
            x_m, y_m = ktool.get_xys(kpts2[fm[:, 1]])
            dlensqrd = (x_m.max() - x_m.min()) ** 2 + (y_m.max() - y_m.min()) ** 2
            return dlensqrd
        topx2_dlen_sqrd = [kpts_dlen_sqrd(tx) for tx in xrange(nRerank)]
    return topx2_dlen_sqrd


#============================
# 6) QueryResult Format
#============================


@profile
def chipmatch_to_resdict(ibs, qrid2_chipmatch, filt2_meta, qreq):
    if not QUIET:
        print('[mf] Step 6) Convert chipmatch -> res')
    uid = qreq.get_uid()
    score_method = qreq.cfg.agg_cfg.score_method
    # Create the result structures for each query.
    qrid2_res = {}
    for qrid in qrid2_chipmatch.iterkeys():
        # For each query's chipmatch
        chipmatch = qrid2_chipmatch[qrid]
        # Perform final scoring
        rid2_score = score_chipmatch(ibs, qrid, chipmatch, score_method, qreq)
        # Create a query result structure
        res = QueryResult.QueryResult(qrid, uid)
        res.rid2_score = rid2_score
        (res.rid2_fm, res.rid2_fs, res.rid2_fk) = chipmatch
        res.filt2_meta = {}  # dbgstats
        for filt, qrid2_meta in filt2_meta.iteritems():
            res.filt2_meta[filt] = qrid2_meta[qrid]  # things like k+1th
        qrid2_res[qrid] = res
    # Retain original score method
    return qrid2_res


#@profile
#def load_resdict(qreq):
    #""" Load the result structures for each query. """
    #qrids = qreq.qrids
    #uid = qreq.get_uid()  # this is the correct uid to use
    ###IF DICT_COMPREHENSION
    #qrid2_res = {qrid: QueryResult.QueryResult(qrid, uid) for qrid in iter(qrids)}
    #[res.load(qreq) for res in qrid2_res.itervalues()]
    ##ELSE
    #qrid2_res = {}
    #for qrid in qrids:
        #res = QueryResult.QueryResult(qrid, uid)
        #res.load(qreq)
        #qrid2_res[qrid] = res
    ##ENDIF
    #return qrid2_res


@profile
def try_load_resdict(qreq):
    """ Try and load the result structures for each query.
    returns a list of failed qrids
    """
    qrids = qreq.qrids
    uid = qreq.get_uid()
    qrid2_res = {}
    failed_qrids = []
    for qrid in qrids:
        try:
            res = QueryResult.QueryResult(qrid, uid)
            res.load(qreq)
            qrid2_res[qrid] = res
        except IOError:
            failed_qrids.append(qrid)
    return qrid2_res, failed_qrids


#============================
# Scoring Mechanism
#============================

@profile
def score_chipmatch(ibs, qrid, chipmatch, score_method, qreq=None):
    (rid2_fm, rid2_fs, rid2_fk) = chipmatch
    # HACK: Im not even sure if the 'w' suffix is correctly handled anymore
    if score_method.find('w') == len(score_method) - 1:
        score_method = score_method[:-1]
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        rid2_score = vr2.score_chipmatch_csum(chipmatch)
    elif score_method == 'pl':
        rid2_score, nid2_score = vr2.score_chipmatch_PL(ibs, qrid, chipmatch, qreq)
    elif score_method == 'borda':
        rid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qrid, chipmatch, qreq, 'borda')
    elif score_method == 'topk':
        rid2_score, nid2_score = vr2.score_chipmatch_pos(ibs, qrid, chipmatch, qreq, 'topk')
    elif score_method.startswith('coverage'):
        # Method num is at the end of coverage
        method = int(score_method.replace('coverage', '0'))
        rid2_score = coverage_image.score_chipmatch_coverage(ibs, qrid, chipmatch, qreq, method=method)
    else:
        raise Exception('[mf] unknown scoring method:' + score_method)
    return rid2_score
